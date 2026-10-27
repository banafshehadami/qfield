"""Amortized designed quadrature for the pretrained VSP DDPM sampler.

Uses the pretrained conditional DDPM to draw samples given normalized
VSP observations y, and trains a vector-conditioned QField amortizer.
Evaluation compares i.i.d. samples, reweighting, and QField emissions
using independent reference banks and MMD.

Run with ``--phase benchmark`` for GPU timing per node budget M
(DDPM posterior sampling vs one QField forward+backward pass).
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.utils.benchmark as benchmark  # noqa: E402
from projorg import (  # noqa: E402
    checkpointsdir,
    plotsdir,
    setup_environment,
    upload_to_cloud,
)
from tqdm import tqdm  # noqa: E402

from _figstyle import apply_house_style, despine  # noqa: E402
from qfield.designed_quadrature import (  # noqa: E402
    constrained_weights,
    constrained_weights_batched,
    mmd_sq,
    move_descend,
    sampled_bank_target,
    select_emission,
)
from qfield.designed_quadrature.cond_net import (  # noqa: E402
    ConditionalQuadratureAmortizer,
)
from qfield.kernels import median_heuristic_sigma_sq  # noqa: E402
from vspsampling.download import DATASET as VSP_DATASET  # noqa: E402
from vspsampling.download import DDPM as VSP_DDPM  # noqa: E402
from vspsampling.download import REPO as VSP_ROOT  # noqa: E402
from vspsampling.models.conditional_ddpm import ConditionalDDPM  # noqa: E402

CONFIG_FILE = "dq_vsp.json"
DTYPE = torch.float64
COND_LIMIT = 1e13
BENCH_DIR = Path(__file__).resolve().parents[1] / "plots" / "vsp_benchmark"
torch.set_num_threads(4)

_PALETTE = {"floor": "#3a7ca5", "reweight": "#2a9d8f", "ours": "#ff7f0e",
            "opt": "#7f7f7f", "null": "#b06fa8"}
_LABELS = {"floor": "i.i.d. floor", "reweight": "reweight only",
           "ours": "ours (1 fwd pass)", "opt": "per-instance (unamortized)",
           "null": "shuffled conditioner"}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _median_iqr(vals) -> tuple[float, float, float]:
    t = torch.tensor(list(vals), dtype=DTYPE)
    t = t[torch.isfinite(t)]
    if t.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], dtype=DTYPE))
    return float(q[1]), float(q[0]), float(q[2])


class VSPDDPMSampler:
    """Pretrained VSP DDPM; observations must already be normalized."""

    def __init__(self, device, n_steps=50):
        self.device = torch.device(device)
        self.n_steps = int(n_steps)

        ckpt = torch.load(os.path.join(VSP_ROOT, VSP_DDPM),
                          map_location="cpu", weights_only=False)
        cfg = ckpt["args"]
        self.model = ConditionalDDPM(
            d_x=41, d_s=20, width=cfg.width, depth=cfg.depth, T=cfg.T,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.float().eval()
        self.normalizer = ckpt["y_normalizer"]

    def normalize(self, y):
        return self.normalizer.normalize(y.to(device="cpu",
                                              dtype=torch.float32))

    @torch.no_grad()
    def draw(self, y_normalized, n, generator):
        """``(n_obs, n, d)`` float64 samples, n per observation row."""
        y = y_normalized.to(self.device, dtype=torch.float32)
        n = int(n)
        seed = int(torch.randint(0, 2**31 - 1, (1,),
                                 generator=generator).item())
        g = torch.Generator(device=self.device).manual_seed(seed)

        samples = []
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            for obs in y:
                cond = obs.unsqueeze(0).expand(n, -1)
                samples.append(self.model.sample(
                    cond, n_steps=self.n_steps, generator=g))
        finally:
            torch.set_default_dtype(previous_dtype)
        return torch.stack(samples).to(torch.float64)


class DQVSP:
    """Amortized designed quadrature for VSP DDPM samples."""

    def __init__(self, args):
        self.args = args
        self.d, self.dy = int(args.d), int(args.dy)
        self.m_list = list(args.m_list)
        self.dev = "cuda" if (
            int(args.gpu_id) >= 0 and torch.cuda.is_available()
            and torch.cuda.mem_get_info()[0] / 1e9 > 1.0) else "cpu"
        # DDPM runs in float32, QField in float64 (MMD differences are tiny).
        # Load the DDPM before switching the default dtype.
        self.sampler = VSPDDPMSampler(self.dev, n_steps=int(args.ddim_steps))
        torch.set_default_dtype(DTYPE)

        # Read only the selected rows; the full dataset is large.
        with h5py.File(os.path.join(VSP_ROOT, VSP_DATASET), "r") as f:
            splits = f["split"][:]
            rng = np.random.default_rng(int(args.base_seed) + 41)
            tr_idx = np.sort(rng.choice(np.flatnonzero(splits == 0),
                                        size=int(args.n_train_obs),
                                        replace=False))
            ev_idx = np.sort(rng.choice(np.flatnonzero(splits == 1),
                                        size=int(args.n_eval_obs),
                                        replace=False))
            self.y_tr_raw = torch.as_tensor(f["y"][tr_idx],
                                            dtype=torch.float32)
            self.y_ev_raw = torch.as_tensor(f["y"][ev_idx],
                                            dtype=torch.float32)
            self.xi_true_ev = torch.as_tensor(f["xi"][ev_idx],
                                              dtype=torch.float32)
            self.depths = f["depths"][:]
            self.attrs = dict(f.attrs)

        self.y_tr = self.sampler.normalize(self.y_tr_raw)
        self.y_ev = self.sampler.normalize(self.y_ev_raw)

        assert self.d == 41 and self.dy == 20
        assert self.y_tr.shape[-1] == self.dy
        assert self.y_ev.shape[-1] == self.dy
        assert self.xi_true_ev.shape[-1] == self.d

        print(f"[vsp] DDPM on {self.dev} | d={self.d} dy={self.dy} "
              f"| train={len(self.y_tr)} eval={len(self.y_ev)}", flush=True)

    # ---- bandwidth -------------------------------------------------------
    def _sigma(self) -> float:
        """Median heuristic over DDPM draws at training observations."""
        a = self.args
        g = torch.Generator().manual_seed(int(a.base_seed) + 1234)
        n_obs = min(64, self.y_tr.shape[0])
        per = max(2, int(a.sigma_n_sample) // n_obs)
        draws = self._draw(self.y_tr[:n_obs], per, g).reshape(
            -1, self.d)[: int(a.sigma_n_sample)]
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
        """DDPM samples ``(n_obs, n, d)`` in float64 on the sampler device."""
        return self.sampler.draw(y_std_rows, n, gen)

    # ---- training (on-the-fly reference) ---------------------------------
    def train(self):
        a = self.args
        self.sigma = self._sigma()
        inv2s = 1.0 / (2.0 * self.sigma ** 2)
        torch.manual_seed(int(a.seed))
        net = self._build_net().to(self.dev)
        n_par = sum(p.numel() for p in net.parameters())
        print(f"[net] vector-conditioned amortizer, params {n_par:,}",
              flush=True)
        opt = torch.optim.Adam(net.parameters(), lr=float(a.lr))
        gen = torch.Generator().manual_seed(int(a.base_seed) + 7)
        n_mu = int(a.n_mu_train)

        # Held-out validation family: whole observations, never seen by the
        # trainer, scored on the same objective at a fixed generator every
        # val_every steps. Early stop on validation, patience 8.
        n_vf = max(1, min(64, self.y_tr.shape[0] // 8))
        y_val = self.y_tr[-n_vf:]
        self.y_tr = self.y_tr[:-n_vf]
        val_every, patience = 200, 8
        best_val, n_stale, best_sd = float("inf"), 0, None
        small = int(a.n_steps) <= 10
        n_val_nodes, n_val_bank = (16, 48) if small else (128, 1024)

        def _val() -> float:
            gv = torch.Generator().manual_seed(int(a.base_seed) + 55)
            tot = 0.0
            with torch.no_grad():
                for j in range(0, n_vf, 8):
                    yb = y_val[j : j + 8]
                    av = self._draw(yb, n_val_nodes + n_val_bank, gv)
                    zv, bv = av[:, :n_val_nodes], av[:, n_val_nodes:]
                    zv = net(zv.to(self.dev), yb.to(self.dev, dtype=DTYPE))
                    sq = torch.cdist(zv, bv.to(self.dev)) ** 2
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
        timing_by_m = {int(m): {"sampling": [], "gradient": [],
                                "training": []} for m in self.m_list}
        # M cycles round-robin so every budget is timed equally often; the
        # first 5 passes per M are warm-up and excluded from timing. While
        # warm-up is on, early stopping is off to keep the schedule balanced.
        warmup_per_m = 5 if int(a.n_steps) >= 5 * len(self.m_list) else 0
        n_warmup = warmup_per_m * len(self.m_list)

        for step in range(int(a.n_steps)):
            M = int(self.m_list[step % len(self.m_list)])
            timed = step >= n_warmup
            obs = torch.randperm(self.y_tr.shape[0], generator=gen)
            y_b = self.y_tr[obs[: int(a.batch_obs)]]

            # Fresh reference every step: M seeds plus an n_mu estimand bank
            # per observation, both sampled from the DDPM.
            _sync()
            t_s = time.perf_counter()
            allv = self._draw(y_b, M + n_mu, gen)
            _sync()
            sampling_times.append(time.perf_counter() - t_s)

            t_tr = time.perf_counter()
            z0, bank = allv[:, :M], allv[:, M:]
            z = net(z0.to(self.dev), y_b.to(self.dev, dtype=DTYPE))
            # mu_hat over the per-step bank; c_rho drops (net-independent).
            sq = torch.cdist(z, bank.to(self.dev)) ** 2
            mu = torch.exp(-sq * inv2s).mean(dim=-1)          # (B, M)
            w = constrained_weights_batched(z, mu, self.sigma, a.jitter)
            gram = torch.exp(-(torch.cdist(z, z) ** 2) * inv2s)
            loss = (torch.einsum("bi,bij,bj->b", w, gram, w)
                    - 2.0 * torch.einsum("bi,bi->b", w, mu)).mean()
            opt.zero_grad()

            _sync()
            t_g = time.perf_counter()
            loss.backward()
            _sync()
            gradient_time = time.perf_counter() - t_g

            opt.step()
            _sync()
            training_times.append(time.perf_counter() - t_tr)

            if timed:
                timing_by_m[M]["sampling"].append(sampling_times[-1])
                timing_by_m[M]["gradient"].append(gradient_time)
                timing_by_m[M]["training"].append(training_times[-1])

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
                    best_sd = {k: t.clone()
                               for k, t in net.state_dict().items()}
                else:
                    n_stale += 1
                    if n_stale >= patience and warmup_per_m == 0:
                        print(f"[early stop] val stale {n_stale}x at step "
                              f"{step}; best {best_val:.4e} restored",
                              flush=True)
                        break

        print(f"[timing] loop wall {time.time() - t0:.3f} s | "
              f"DDPM {1e3 * np.mean(sampling_times):.2f} ms/step | "
              f"QField {1e3 * np.mean(training_times):.2f} ms/step | "
              f"warm-up {warmup_per_m}/M", flush=True)
        self._report_timing(timing_by_m)

        if best_sd is not None:
            net.load_state_dict(best_sd)
        print(f"[train] wall {(time.time() - t0) / 60:.1f} min", flush=True)

        net.eval()
        rows, nulls, ctrl = self._evaluate(net)
        results = {
            "sigma": self.sigma, "m_list": self.m_list,
            "n_params": int(n_par), "train_curve": curve,
            "val_curve": val_curve, "best_val": best_val,
            "n_val_family": n_vf,
            "rows": rows, "null_rows": nulls, "untrained_control": ctrl,
            "recon": None, "pcn_audit": None,
            "posterior_sampler": "vsp_ddpm", "bank_L": int(a.bank_L),
            "verdict": {
                "ours_below_floor_every_M": all(
                    r["ours"][0] < r["floor"][0] for r in rows),
                "safeguard_violations": sum(
                    r["n_worse_than_menu_bankA"] for r in rows),
                "n_scored_emissions": sum(r["n_draws"] for r in rows),
            },
        }
        ck_dir = checkpointsdir(a.experiment)
        ck = os.path.join(ck_dir, "checkpoint.pth")
        torch.save({"results": results, "state_dict": net.state_dict()}, ck)
        with open(os.path.join(ck_dir, "results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"Saved results to {ck}", flush=True)

    def _report_timing(self, timing_by_m: dict) -> None:
        """Per-M timing table and plot (DDPM sampling vs QField gradient)."""
        ms = sorted(M for M, r in timing_by_m.items() if r["sampling"])
        if not ms:
            return
        print(f"[timing] {'M':>6} {'N':>6} {'DDPM ms':>10} "
              f"{'grad ms':>10} {'step ms':>10}")
        for M in ms:
            r = timing_by_m[M]
            print(f"[timing] {M:>6} {len(r['sampling']):>6} "
                  f"{1e3 * np.mean(r['sampling']):>10.2f} "
                  f"{1e3 * np.mean(r['gradient']):>10.2f} "
                  f"{1e3 * np.mean(r['training']):>10.2f}")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ms, [1e3 * np.mean(timing_by_m[M]["sampling"]) for M in ms],
                "o-", linewidth=2, label="DDPM posterior sampling")
        ax.plot(ms, [1e3 * np.mean(timing_by_m[M]["gradient"]) for M in ms],
                "s-", linewidth=2, label="QField gradient calculation")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ms)
        ax.set_xticklabels([str(M) for M in ms])
        ax.set_xlabel("Node budget M")
        ax.set_ylabel("Mean time per iteration (ms)")
        ax.set_title("VSP: Posterior Sampling vs QField Gradient")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        BENCH_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(BENCH_DIR / "dq_vsp_gpu_timing_vs_M.png", dpi=200,
                    bbox_inches="tight")
        fig.savefig(BENCH_DIR / "dq_vsp_gpu_timing_vs_M.pdf",
                    bbox_inches="tight")
        plt.close(fig)
        print(f"[timing] plots saved to {BENCH_DIR}", flush=True)

    # ---- evaluation ------------------------------------------------------
    def _eval_target(self, o: int, gen: torch.Generator,
                     L: int | None = None):
        """The o-th eval reference: an L-atom DDPM bank (returns bank too).

        L defaults to bank_L and is raised by the caller at M >= 256, because
        at L = 8192 the bank's own resolution (1-c)/L is an appreciable
        fraction of the reading at the largest budgets.
        """
        L = int(self.args.bank_L) if L is None else int(L)
        bank = self._draw(self.y_ev[o : o + 1], L, gen)[0].cpu()
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

        Conditioning nulls: "null" feeds the neighbouring observation's y,
        "const" feeds the mean y.
        """
        a = self.args
        rep = int(a.eval_repeats)
        rows = []
        y_cpu = self.y_ev.double().cpu()
        assert self.y_ev.shape[0] >= 2
        perm = torch.roll(torch.arange(self.y_ev.shape[0]), shifts=1)
        y_const = y_cpu.mean(dim=0)
        inv2s = 1.0 / (2.0 * self.sigma ** 2)
        ctrl = None

        def emit(zu, y_row):
            z = net(zu.unsqueeze(0).to(self.dev),
                    y_row.unsqueeze(0).to(self.dev))[0].cpu()
            return z

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
                tgtA, bankA = self._eval_target(o, gA, L)
                tgtB, bankB = self._eval_target(o, gB, L)

                def muB(z):
                    return torch.exp(
                        -(torch.cdist(z, bankB) ** 2) * inv2s).mean(-1)

                o_yard = None
                if o == 0:
                    # conditioning yardstick: how far apart the two paired
                    # references actually are (for the null's reading)
                    o_yard = float(tgtA.c_rho + tgtB.c_rho
                                   - 2 * muB(bankA).mean())
                per_draw = {k: [] for k in keys if k != "yard"}
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
                    with torch.no_grad():
                        z_n = emit(zu, y_cpu[o])
                        mu_n = tgtA.mu_fn(z_n)
                        w_n = constrained_weights(z_n, mu_n, self.sigma,
                                                  a.jitter)
                        vA = float(mmd_sq(z_n, w_n, mu_n, tgtA.c_rho,
                                          self.sigma))
                        z_s = emit(zu, y_cpu[perm[o]])
                        w_s = constrained_weights(
                            z_s, tgtA.mu_fn(z_s), self.sigma, a.jitter)
                        z_k = emit(zu, y_const)
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
                    n_active += int(sel["active"])
                    n_draws += 1
                    n_worse += int(sA > min(fA, rA, vA) + 1e-12)

                    # ---- held-out scoring on bank B, every arm ----
                    def scoreB(z, w):
                        return float(mmd_sq(z, w, muB(z), tgtB.c_rho,
                                            self.sigma))

                    fB, rB = scoreB(zu, w_eq), scoreB(zu, w_c)
                    vB, sB = scoreB(z_n, w_n), scoreB(zS, wS)
                    nB, kB = scoreB(z_s, w_s), scoreB(z_k, w_k)

                    if k < 4:
                        gram = torch.exp(-(torch.cdist(zu, zu) ** 2) * inv2s)
                        ev = torch.linalg.eigvalsh(gram)
                        per_draw["lmin"].append(float(ev.min()))
                        evj = ev + a.jitter
                        per_draw["condD"].append(
                            float(evj.max() / evj.min().clamp_min(1e-300)))
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
                net0 = self._build_net()
                net0.eval()
                genc = torch.Generator().manual_seed(int(a.base_seed) + 99)
                tgt0, _ = self._eval_target(0, genc, int(a.bank_L))
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
        apply_house_style()
        rows = self.results["rows"]
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        m_vals = [r["M"] for r in rows]
        # Held-out (bank-B) readings are plotted: on bank A
        # "ours <= floor" is true by construction.
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


# ---- GPU benchmark -------------------------------------------------------

def gpu_benchmark(args) -> None:
    """GPU timing per M: DDPM sampling vs QField forward+backward."""
    assert torch.cuda.is_available(), "CUDA is unavailable"
    torch.manual_seed(int(args.seed))
    exp = DQVSP(args)
    device = torch.device(exp.dev)
    assert device.type == "cuda", "Experiment fell back to CPU"

    net = exp._build_net().to(device)
    net.train()
    assert next(exp.sampler.model.parameters()).device == \
        next(net.parameters()).device

    sigma = exp._sigma()
    inv2s = 1.0 / (2.0 * sigma ** 2)
    y_cpu = exp.y_tr[: int(args.batch_obs)]
    y_gpu = y_cpu.to(device=device, dtype=DTYPE)
    n_mu = int(args.n_mu_train)

    def sync():
        torch.cuda.synchronize(device)

    def loss_fn(z0, bank):
        z = net(z0, y_gpu)
        mu = torch.exp(-(torch.cdist(z, bank) ** 2) * inv2s).mean(dim=-1)
        w = constrained_weights_batched(z, mu, sigma, args.jitter)
        gram = torch.exp(-(torch.cdist(z, z) ** 2) * inv2s)
        return (torch.einsum("bi,bij,bj->b", w, gram, w)
                - 2.0 * torch.einsum("bi,bi->b", w, mu)).mean()

    results = []
    for M in args.m_list:
        M = int(M)
        print(f"\n[bench] M={M}", flush=True)
        gen = torch.Generator().manual_seed(int(args.base_seed) + M)

        # QField inputs are drawn once, outside the timed region.
        allv = exp._draw(y_cpu, M + n_mu, gen)
        z0, bank = allv[:, :M].to(device), allv[:, M:].to(device)
        sync()

        def sampling_step():
            exp._draw(y_cpu, M + n_mu, gen)
            sync()

        def qfield_step():
            net.zero_grad(set_to_none=True)
            loss_fn(z0, bank).backward()
            sync()

        for name, fn in (("DDPM sampling", sampling_step),
                         ("QField forward+backward", qfield_step)):
            m = benchmark.Timer(
                stmt="fn()", globals={"fn": fn}, num_threads=4,
                label=f"M={M}", description=name,
            ).blocked_autorange(min_run_time=1)
            med, iqr = 1e3 * m.median, 1e3 * m.iqr
            print(f"  {name:25s} median={med:.3f} ms IQR={iqr:.3f} ms",
                  flush=True)
            results.append((M, name, med, iqr))

        allv = z0 = bank = None
        torch.cuda.empty_cache()

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out = BENCH_DIR / "dq_vsp_timing.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "operation", "median_ms", "iqr_ms"])
        writer.writerows(results)
    print(f"\n[bench] saved {out}", flush=True)


def parse_args():
    return setup_environment(
        CONFIG_FILE,
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "upload",
                         "recon_M", "n_recon", "eval_repeats",
                         "n_pcn_audit", "ddim_steps"],
        sequence_args_and_types=[("m_list", int)],
    )


if __name__ == "__main__":
    args = parse_args()
    if args.phase == "benchmark":
        gpu_benchmark(args)
    else:
        exp = DQVSP(args)
        if args.phase == "train":
            exp.train()
        exp.load_checkpoint()
        exp.visualize()
        if args.upload:
            upload_to_cloud(args)
