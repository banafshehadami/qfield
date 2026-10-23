"""Amortized designed quadrature for the pretrained VSP DDPM sampler.

Uses the pretrained conditional DDPM to draw samples given normalized
VSP observations y, and trains a vector-conditioned QField amortizer.
Evaluation compares i.i.d. samples, reweighting, and QField emissions
using independent reference banks and MMD.
"""

from __future__ import annotations
from projorg import configsdir, logsdir, plotsdir  # noqa: E402

import json
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from projorg import (  # noqa: E402
    checkpointsdir,
    plotsdir,
    setup_environment,
    upload_to_cloud,
)
from tqdm import tqdm  # noqa: E402

from _figstyle import apply_house_style, despine  # noqa: E402

import h5py  # noqa: E402

from qfield.designed_quadrature import (  # noqa: E402
    constrained_weights,
    constrained_weights_batched,
    mmd_sq,
    mmd_sq_batched,
    move_descend,
    sampled_bank_target,
    select_emission,
)
from qfield.designed_quadrature.cond_net import (  # noqa: E402
    ConditionalQuadratureAmortizer,
)
from qfield.kernels import median_heuristic_sigma_sq  # noqa: E402


from vspsampling.download import REPO as VSP_ROOT, DATASET as VSP_DATASET, DDPM as VSP_DDPM
from vspsampling.models.conditional_ddpm import ConditionalDDPM


class VSPDDPMSampler:
    """Pretrained VSP DDPM; observations must already be normalized."""

    def __init__(self, device, n_steps=50):
        self.device = torch.device(device)
        self.n_steps = int(n_steps)

        ckpt = torch.load(
            os.path.join(VSP_ROOT, VSP_DDPM),
            map_location="cpu",
            weights_only=False,
        )
        cfg = ckpt["args"]
        self.model = ConditionalDDPM(
            d_x=41, d_s=20,
            width=cfg.width, depth=cfg.depth, T=cfg.T,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.float().eval()

        self.normalizer = ckpt["y_normalizer"]

    def normalize(self, y):
        y = y.to(device="cpu", dtype=torch.float32)
        return self.normalizer.normalize(y)

    @torch.no_grad()
    def draw(self, y_normalized, n, generator):
        y = y_normalized.to(self.device, dtype=torch.float32)
        n = int(n)
        seed = int(torch.randint(
            0, 2**31 - 1, (1,), generator=generator
        ).item())
        g = torch.Generator(device=self.device).manual_seed(seed)

        samples = []
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            for obs in y:
                # Use the same observation y for all n DDPM samples.
                cond = obs.unsqueeze(0).expand(n, -1)
                samples.append(self.model.sample(
                    cond, n_steps=self.n_steps, generator=g
                ).cpu())  # Move samples to CPU for QField.
        finally:
            torch.set_default_dtype(previous_dtype)

        return torch.stack(samples).to(torch.float64)


CONFIG_FILE = "dq_vsp.json"
DTYPE = torch.float64
torch.set_num_threads(4)

_PALETTE = {"floor": "#3a7ca5", "reweight": "#2a9d8f", "ours": "#ff7f0e",
            "opt": "#7f7f7f", "null": "#b06fa8"}
_LABELS = {"floor": "i.i.d. floor", "reweight": "reweight only",
           "ours": "ours (1 fwd pass)", "opt": "per-instance (unamortized)",
           "null": "shuffled conditioner"}
COND_LIMIT = 1e13


def _median_iqr(vals) -> tuple[float, float, float]:
    t = torch.tensor(list(vals), dtype=DTYPE)
    t = t[torch.isfinite(t)]
    if t.numel() == 0:
        return float("nan"),  float("nan"), float("nan")
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], dtype=DTYPE))
    return float(q[1]), float(q[0]), float(q[2])


class DQVSP:
    """Amortized designed quadrature for VSP DDPM samples."""

    def __init__(self, args):
        self.args = args
        self.d, self.dy = int(args.d), int(args.dy)
        self.m_list = list(args.m_list)
        dev = "cuda" if (int(args.gpu_id) >= 0 and torch.cuda.is_available()
                         and torch.cuda.mem_get_info()[0] / 1e9 > 1.0) else "cpu"
        self.flow_dev = dev
        # DDPM uses float32, but I keep QField in float64 for the MMD calculations.
        # Load DDPM first so changing the default dtype does not affect it.
        self.sampler = VSPDDPMSampler(dev, n_steps=int(args.ddim_steps))
        torch.set_default_dtype(DTYPE)

        # Read only selected rows; the complete VSP dataset is very large.
        import numpy as np
        with h5py.File(os.path.join(VSP_ROOT, VSP_DATASET), "r") as f:
            splits = f["split"][:]
            rng = np.random.default_rng(int(args.base_seed) + 41)
            train_pool = np.flatnonzero(splits == 0)
            eval_pool = np.flatnonzero(splits == 1)

            tr_idx = np.sort(rng.choice(
                train_pool, size=int(args.n_train_obs), replace=False
            ))
            ev_idx = np.sort(rng.choice(
                eval_pool, size=int(args.n_eval_obs), replace=False
            ))

            self.y_tr_raw = torch.as_tensor(
                f["y"][tr_idx], dtype=torch.float32
            )
            self.y_ev_raw = torch.as_tensor(
                f["y"][ev_idx], dtype=torch.float32
            )
            self.xi_true_ev = torch.as_tensor(
                f["xi"][ev_idx], dtype=torch.float32
            )
            self.depths = f["depths"][:]
            self.attrs = dict(f.attrs)

        # Normalize y using the normalizer from the DDPM checkpoint.
        self.y_tr = self.sampler.normalize(self.y_tr_raw)
        self.y_ev = self.sampler.normalize(self.y_ev_raw)

        # Check that the data dimensions match the pretrained DDPM.
        assert self.d == 41 and self.dy == 20
        assert self.y_tr.shape[-1] == self.dy
        assert self.y_ev.shape[-1] == self.dy
        assert self.xi_true_ev.shape[-1] == self.d

        print(
            f"[vsp] DDPM on {dev} | d={self.d} dy={self.dy} "
            f"| train={len(self.y_tr)} eval={len(self.y_ev)}",
            flush=True,
        )

    # ---- bandwidth -------------------------------------------------------
    def _sigma(self) -> float:
        """Median heuristic using DDPM samples from training observations."""
        a = self.args
        g = torch.Generator().manual_seed(int(a.base_seed) + 1234)
        n_obs = min(64, self.y_tr.shape[0])
        per = max(2, int(a.sigma_n_sample) // n_obs)
        draws = self._draw(
            self.y_tr[:n_obs], per, g
        ).reshape(-1, self.d)[: int(a.sigma_n_sample)]
        sig = math.sqrt(median_heuristic_sigma_sq(draws.double().cpu()))
        print(f"[sigma] pooled median heuristic (mmd rule) over "
              f"{draws.shape[0]} DDPM draws: {sig:.4f}", flush=True)
        return float(sig)

    def _build_net(self) -> ConditionalQuadratureAmortizer:
        a = self.args
        return ConditionalQuadratureAmortizer(
            d=self.d, hidden=int(a.hidden), n_blocks=int(a.n_blocks),
            n_heads=int(a.n_heads), cond_hidden=int(a.cond_hidden),
            delta_scale=float(a.delta_scale),
            cond_token_dim=int(a.cond_token_dim),
            cond_hidden_ch=int(a.cond_hidden_ch),
            cond_kind="vector", cond_vec_dim=self.dy,
            cond_n_tokens=int(a.cond_n_tokens),
        ).to(DTYPE)

    def _draw(self, y_std_rows: torch.Tensor, n: int,
              gen: torch.Generator) -> torch.Tensor:
        """DDPM samples ``(n_obs, n, d)`` in float64 on CPU."""
        return self.sampler.draw(y_std_rows, n, gen)

    # ---- training (on-the-fly reference) ---------------------------------
    def train(self):
        a = self.args
        self.sigma = self._sigma()
        inv2s = 1.0 / (2.0 * self.sigma ** 2)
        torch.manual_seed(int(a.seed))
        net = self._build_net()
        n_par = sum(p.numel() for p in net.parameters())
        print(f"[net] vector-conditioned amortizer, params {n_par:,}",
              flush=True)
        opt = torch.optim.Adam(net.parameters(), lr=float(a.lr))
        gen = torch.Generator().manual_seed(int(a.base_seed) + 7)
        m_choices = torch.tensor(self.m_list)
        n_mu = int(a.n_mu_train)
        # Held-out validation family: whole observations, never seen by the
        # trainer, scored on the same objective at a fixed generator every
        # val_every steps. Early stop on validation, patience 8.
        n_vf = max(1, min(64, self.y_tr.shape[0] // 8))
        y_val = self.y_tr[-n_vf:]
        self.y_tr = self.y_tr[:-n_vf]
        val_every, patience = 200, 8
        best_val, n_stale, best_sd = float("inf"), 0, None

        def _val() -> float:
            gv = torch.Generator().manual_seed(int(a.base_seed) + 55)
            tot = 0.0
            with torch.no_grad():
                for j in range(0, n_vf, 8):
                    yb = y_val[j : j + 8]
                    n_val_nodes = 16 if int(a.n_steps) <= 10 else 128
                    n_val_bank = 48 if int(a.n_steps) <= 10 else 1024
                    av = self._draw(yb, n_val_nodes + n_val_bank, gv)
                    zv, bv = av[:, :n_val_nodes], av[:, n_val_nodes:]
                    zv = net(zv, yb.double().cpu())
                    sq = torch.cdist(zv, bv) ** 2
                    muv = torch.exp(-sq * inv2s).mean(-1)
                    wv = constrained_weights_batched(zv, muv, self.sigma,
                                                     a.jitter)
                    gv2 = torch.exp(-(torch.cdist(zv, zv) ** 2) * inv2s)
                    tot += float((torch.einsum("bi,bij,bj->b", wv, gv2, wv)
                                  - 2 * torch.einsum("bi,bi->b", wv, muv)
                                  ).sum())
            return tot / n_vf

        curve, val_curve, ema = [], [], None
        log_every = max(1, int(a.n_steps) // 20)
        t0 = time.time()
        net.train()
        sampling_times, training_times = [], []
        for step in range(int(a.n_steps)):
            M = int(m_choices[torch.randint(len(m_choices), (1,),
                                            generator=gen)])
            obs = torch.randperm(self.y_tr.shape[0], generator=gen)
            obs = obs[: int(a.batch_obs)]
            y_b = self.y_tr[obs]
            # Fresh reference every step: M seeds plus an n_mu estimand bank
            # per observation, both sampled from the DDPM.
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            sampling_start = time.perf_counter()
            allv = self._draw(y_b, M + n_mu, gen)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            sampling_times.append(time.perf_counter() - sampling_start)
            training_start = time.perf_counter()
            z0, bank = allv[:, :M], allv[:, M:]
            # Pass both the DDPM samples and y to QField.
            # For now I keep QField on CPU.
            z = net(z0, y_b.double().cpu())
            # mu_hat over the per-step bank; c_rho drops (net-independent).
            sq = torch.cdist(z, bank) ** 2
            mu = torch.exp(-sq * inv2s).mean(dim=-1)          # (B, M)
            w = constrained_weights_batched(z, mu, self.sigma, a.jitter)
            gram = torch.exp(-(torch.cdist(z, z) ** 2) * inv2s)
            loss = (
                torch.einsum("bi,bij,bj->b", w, gram, w)
                - 2.0 * torch.einsum("bi,bi->b", w, mu)
            ).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            training_times.append(time.perf_counter() - training_start)
            v = float(loss.detach())
            ema = v if ema is None else 0.98 * ema + 0.02 * v
            if step % log_every == 0 or step == int(a.n_steps) - 1:
                curve.append((step, ema))
                print(f"  step {step:>5d}/{a.n_steps}: M={M:>3d} "
                      f"J(mean)={v:.4e}  EMA={ema:.4e}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if step % val_every == 0 or step == int(a.n_steps) - 1:
                net.eval()
                jv = _val()
                net.train()
                val_curve.append((step, jv))
                print(f"    [val] step {step}: J_val={jv:.4e} "
                      f"(best {best_val:.4e})", flush=True)
                if jv < best_val - 1e-9:
                    best_val, n_stale = jv, 0
                    best_sd = {k: t.clone() for k, t in
                               net.state_dict().items()}
                else:
                    n_stale += 1
                    if n_stale >= patience:
                        print(f"[early stop] val stale {n_stale}x at step "
                              f"{step}; best {best_val:.4e} restored",
                              flush=True)
                        break
        print(f"[benchmark] Training loop wall: {time.time()-t0:.3f} s")
        print(f"[benchmark] DDPM sampling: {sum(sampling_times)/len(sampling_times)*1000:.2f} ms/step")
        print(f"[benchmark] QField training: {sum(training_times)/len(training_times)*1000:.2f} ms/step")
        if best_sd is not None:
            net.load_state_dict(best_sd)
        print(f"[train] wall {((time.time() - t0) / 60):.1f} min", flush=True)

        net.eval()
        rows, nulls, ctrl = self._evaluate(net)
        # VSP smoke test: Darcy-specific diagnostics are disabled.
        recon = None
        pcn = None
        results = {
            "sigma": self.sigma, "m_list": self.m_list,
            "n_params": int(n_par), "train_curve": curve,
            "val_curve": val_curve, "best_val": best_val,
            "n_val_family": n_vf,
            "rows": rows, "null_rows": nulls, "untrained_control": ctrl,
            "recon": recon, "pcn_audit": pcn,
            "posterior_sampler": "vsp_ddpm", "bank_L": int(a.bank_L),
            "verdict": {
                "ours_below_floor_every_M": all(
                    r["ours"][0] < r["floor"][0] for r in rows
                ),
                "safeguard_violations": sum(
                    r["n_worse_than_menu_bankA"] for r in rows
                ),
                "n_scored_emissions": sum(r["n_draws"] for r in rows),
            },
        }
        ck = os.path.join(checkpointsdir(a.experiment), "checkpoint.pth")
        torch.save({"results": results, "state_dict": net.state_dict()}, ck)
        with open(os.path.join(checkpointsdir(a.experiment),
                               "results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"Saved results to {ck}", flush=True)

    # ---- evaluation ------------------------------------------------------
    def _eval_target(self, o: int, gen: torch.Generator,
                     L: int | None = None):
        """The o-th eval reference: an L-atom flow bank (returns bank too).

        L defaults to bank_L and is raised by the caller at M >= 256, because
        at L = 8192 the bank's own resolution (1-c)/L is an appreciable
        fraction of the reading at the largest budgets.
        """
        L = int(self.args.bank_L) if L is None else int(L)
        bank = self._draw(self.y_ev[o : o + 1], L, gen)[0]
        return sampled_bank_target(
            bank, self.sigma, name="vsp_ddpm", dtype=DTYPE, replace=True
        ), bank

    def _evaluate(self, net):
        """Two-bank protocol: bank A defines the target -- seeds, weight
        solves, selection -- and a disjoint bank B scores every arm. On bank A
        "ours <= floor" holds by construction (exact estimand); on bank B it is
        falsifiable. Both are recorded.

        Seeds are sampled with replacement from bank A and deduped with
        multiplicity before any solve: duplicates follow the birthday law
        M^2/2L and leave the Gram singular. Dedupe is exact -- the floor keeps
        w_j = count_j / M on unique atoms (the same empirical measure), and the
        constrained QP over duplicated coordinates has the same optimum.
        lambda_min of the raw deduped Gram is recorded beside the jittered
        condition number.
        """
        a = self.args
        rep = int(a.eval_repeats)
        rows = []
        y_cpu = self.y_ev.double().cpu()
        # Shuffle y to check if QField is actually using the conditioning.
        assert self.y_ev.shape[0] >= 2
        perm = torch.roll(torch.arange(self.y_ev.shape[0]), shifts=1)
        y_const = y_cpu.mean(dim=0)
        ctrl = None
        for M in self.m_list:
            # 2x, not 4x: 32768-atom banks exhaust memory; 16384 keeps the
            # L/M cap at 64x/32x for M=256/512 and halves the transients.
            L = int(a.bank_L) if int(M) < 256 else 2 * int(a.bank_L)
            keys = ("floorA", "rwA", "oursA", "rawA", "optA",
                    "floorB", "rwB", "oursB", "rawB", "optB",
                    "nullB", "constB", "condD", "lmin", "yard")
            acc = {k: [] for k in keys}
            n_active = n_draws = n_worse = 0
            t0 = time.time()
            for o in tqdm(range(self.y_ev.shape[0]), desc=f"eval M={M}",
                          leave=False):
                gA = torch.Generator().manual_seed(
                    int(a.base_seed) + 50_000 + 1000 * int(M) + o)
                gB = torch.Generator().manual_seed(
                    int(a.base_seed) + 350_000 + 1000 * int(M) + o)
                # Use bank A for fitting and bank B for evaluation.
                tgtA, bankA = self._eval_target(o, gA, L)
                tgtB, bankB = self._eval_target(o, gB, L)
                inv2s = 1.0 / (2.0 * self.sigma ** 2)

                def muB(z):
                    return torch.exp(
                        -(torch.cdist(z, bankB) ** 2) * inv2s).mean(-1)

                o_yard = None
                if o == 0:
                    # conditioning yardstick: how far apart the two paired
                    # references actually are (for the null's reading)
                    mAB = muB(bankA).mean()
                    o_yard = float(tgtA.c_rho + tgtB.c_rho - 2 * mAB)
                per_draw = {k: [] for k in keys if k not in ("yard",)}
                for k in range(rep):
                    idx = torch.multinomial(
                        torch.ones(L, dtype=DTYPE), int(M),
                        replacement=True, generator=gA)
                    uniq, counts = torch.unique(idx, return_counts=True)
                    zu = bankA[uniq]                       # (Mu, d) deduped
                    w_eq = counts.to(DTYPE) / float(M)     # floor weights
                    muA_u = tgtA.mu_fn(zu)
                    fA = float(mmd_sq(zu, w_eq, muA_u, tgtA.c_rho,
                                      self.sigma))
                    w_c = constrained_weights(zu, muA_u, self.sigma, a.jitter)
                    rA = float(mmd_sq(zu, w_c, muA_u, tgtA.c_rho, self.sigma))
                    cnd = y_cpu[o].unsqueeze(0)
                    with torch.no_grad():
                        z_n = net(zu.unsqueeze(0), cnd)[0]
                        mu_n = tgtA.mu_fn(z_n)
                        w_n = constrained_weights(z_n, mu_n, self.sigma,
                                                  a.jitter)
                        vA = float(mmd_sq(z_n, w_n, mu_n, tgtA.c_rho,
                                          self.sigma))
                        z_s = net(zu.unsqueeze(0),
                                  y_cpu[perm[o]].unsqueeze(0))[0]
                        w_s = constrained_weights(
                            z_s, tgtA.mu_fn(z_s), self.sigma, a.jitter)
                        z_k = net(zu.unsqueeze(0), y_const.unsqueeze(0))[0]
                        w_k = constrained_weights(
                            z_k, tgtA.mu_fn(z_k), self.sigma, a.jitter)
                    sel = select_emission(
                        [(zu, w_eq), (zu, w_c.detach()), (z_n, w_n)],
                        tgtA.mu_fn, self.sigma,
                        names=("seeds", "reweight", "move"),
                        fitting_criterion=True)
                    zS, wS = sel["z"], sel["w"]
                    sA = max(float(mmd_sq(zS, wS, tgtA.mu_fn(zS), tgtA.c_rho,
                                          self.sigma)), 0.0)
                    n_active += int(sel["active"]); n_draws += 1
                    n_worse += int(sA > min(fA, rA, vA) + 1e-12)
                    # ---- held-out scoring on bank B, every arm ----
                    fB = float(mmd_sq(zu, w_eq, muB(zu), tgtB.c_rho,
                                      self.sigma))
                    rB = float(mmd_sq(zu, w_c, muB(zu), tgtB.c_rho,
                                      self.sigma))
                    vB = float(mmd_sq(z_n, w_n, muB(z_n), tgtB.c_rho,
                                      self.sigma))
                    sB = float(mmd_sq(zS, wS, muB(zS), tgtB.c_rho,
                                      self.sigma))
                    nB = float(mmd_sq(z_s, w_s, muB(z_s), tgtB.c_rho,
                                      self.sigma))
                    kB = float(mmd_sq(z_k, w_k, muB(z_k), tgtB.c_rho,
                                      self.sigma))
                    if k < 4:
                        gram = torch.exp(-(torch.cdist(zu, zu) ** 2) * inv2s)
                        ev = torch.linalg.eigvalsh(gram)
                        per_draw["lmin"].append(float(ev.min()))
                        evj = ev + a.jitter
                        per_draw["condD"].append(
                            float(evj.max() / evj.min().clamp_min(1e-300)))
                    if k < 4:
                        _, ovA = move_descend(
                            zu, tgtA.mu_fn, tgtA.c_rho, self.sigma,
                            float(a.opt_lr), int(a.opt_iters), a.jitter)
                        per_draw["optA"].append(float(ovA))
                    for kk, vv in (("floorA", fA), ("rwA", rA), ("oursA", sA),
                                   ("rawA", vA), ("floorB", fB), ("rwB", rB),
                                   ("oursB", sB), ("rawB", vB), ("nullB", nB),
                                   ("constB", kB)):
                        per_draw[kk].append(vv)
                for kk in per_draw:
                    if per_draw[kk]:
                        acc[kk].append(float(np.nanmean(per_draw[kk])))
                if o_yard is not None:
                    acc["yard"].append(o_yard)
            row = {"M": int(M), "L": int(L), "cap_L_over_M": L / int(M)}
            for kk, name in (("floorA", "floor"), ("rwA", "reweight"),
                             ("oursA", "ours"), ("rawA", "ours_move_raw"),
                             ("optA", "opt"), ("floorB", "floor_heldout"),
                             ("rwB", "reweight_heldout"),
                             ("oursB", "ours_heldout"),
                             ("rawB", "raw_heldout"),
                             ("nullB", "null_heldout"),
                             ("constB", "const_cond_heldout")):
                row[name] = _median_iqr(acc[kk])
            row.update({
                "lambda_min_median": float(np.median(acc["lmin"])),
                "gram_singular_frac": float(np.mean(
                    np.asarray(acc["lmin"]) <= 0)),
                "cond_median": float(np.median(acc["condD"])),
                "cond_limited": bool(np.median(acc["condD"]) > COND_LIMIT),
                "pair_yardstick_mmd2": acc["yard"][0] if acc["yard"] else None,
                "activation": n_active / max(1, n_draws),
                "n_draws": n_draws, "n_worse_than_menu_bankA": n_worse,
                "secs": time.time() - t0,
            })
            row["ours_over_floor"] = row["ours"][0] / row["floor"][0]
            row["ours_over_floor_heldout"] = (
                row["ours_heldout"][0] / row["floor_heldout"][0])
            rows.append(row)
            print(f"  [eval] M={M:>3d} L={L}: "
                  f"A: fl={row['floor'][0]:.3e} ours={row['ours'][0]:.3e} "
                  f"({row['ours_over_floor']:.3f}) | "
                  f"B: fl={row['floor_heldout'][0]:.3e} "
                  f"ours={row['ours_heldout'][0]:.3e} "
                  f"({row['ours_over_floor_heldout']:.3f}) "
                  f"null={row['null_heldout'][0]:.3e} "
                  f"const={row['const_cond_heldout'][0]:.3e} | "
                  f"worse(A)={n_worse}/{n_draws} "
                  f"lmin~{row['lambda_min_median']:.1e} "
                  f"cond~{row['cond_median']:.1e} | {row['secs']:.0f}s",
                  flush=True)
            if ctrl is None:
                torch.manual_seed(int(a.seed))
                net0 = self._build_net(); net0.eval()
                genc = torch.Generator().manual_seed(int(a.base_seed) + 99)
                tgt0, bank0 = self._eval_target(0, genc, int(a.bank_L))
                z0c = tgt0.sampler(int(M), genc).unsqueeze(0)
                with torch.no_grad():
                    zc = net0(z0c, y_cpu[0].unsqueeze(0))
                ctrl = {"max_displacement_untrained":
                        float((zc - z0c).abs().max()), "M": int(M)}
                print(f"  [control] untrained net max|dz|="
                      f"{ctrl['max_displacement_untrained']:.2e}", flush=True)
        return rows, [], ctrl

    # ---- restore + render ------------------------------------------------
    def load_checkpoint(self):
        ck = os.path.join(checkpointsdir(self.args.experiment),
                          "checkpoint.pth")
        if not os.path.isfile(ck):
            raise ValueError(f"Checkpoint does not exist: {ck}")
        self.ckpt = torch.load(ck, weights_only=False)
        self.results = self.ckpt["results"]

    def visualize(self):
        out = plotsdir(self.args.experiment)
        res = self.results
        apply_house_style()
        rows = res["rows"]
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        m_vals = [r["M"] for r in rows]
        # The held-out (bank-B) readings are what is plotted: on bank A
        # "ours <= floor" is true by construction. The per-instance arm stays
        # the in-bank descent.
        for arm, key in (("floor", "floor_heldout"),
                         ("reweight", "reweight_heldout"),
                         ("null", "null_heldout"),
                         ("ours", "ours_heldout")):
            med = [r[key][0] for r in rows]
            lo = [r[key][1] for r in rows]
            hi = [r[key][2] for r in rows]
            yerr = [[m - l for m, l in zip(med, lo)],
                    [h - m for m, h in zip(med, hi)]]
            ax.errorbar(m_vals, med, yerr=yerr, color=_PALETTE[arm],
                        marker="o", markersize=4, linewidth=1.5,
                        capsize=2.5, label=_LABELS[arm])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(m_vals)
        ax.set_xticklabels([str(m) for m in m_vals])
        ax.set_xlabel(r"node budget $M$")
        ax.set_ylabel(r"$\mathrm{MMD}^2(Q, \rho_y)$")
        despine(ax)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = os.path.join(out, f"dq_vsp_mmd_vs_M.{ext}")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            print(f"Saved to {p}")
        plt.close(fig)

        # VSP smoke test: reconstruction visualization is not yet implemented.
        print("[vsp] MMD visualization complete.", flush=True)


if __name__ == "__main__":
    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "upload",
                         "recon_M", "n_recon", "eval_repeats",
                         "n_pcn_audit", "ddim_steps"],
        sequence_args_and_types=[("m_list", int)],
    )
    exp = DQVSP(args)
    if args.phase == "train":
        exp.train()
    exp.load_checkpoint()
    exp.visualize()
    if args.upload:
        upload_to_cloud(args)
