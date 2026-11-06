"""GPU QField training on pretrained VSP DDPM posterior samples.

Training:
    - Read pregenerated posterior samples from HDF5 in batches.
    - Train QField with the original kernel quadrature objective.
    - Validate on held-out observations.
    - Save the trained model before evaluation.

Evaluation:
    - Generate independent reference banks using pretrained VSP DDPM.
    - Compare i.i.d., reweighting, and QField emissions.
    - Report MMD on fitting bank A and independent bank B.


HDF5 storage and file I/O remain on CPU/disk.
QField, DDPM, kernels, gradients and weight solves use CUDA.
"""

from __future__ import annotations

import json
import math
import time
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from projorg import (
    checkpointsdir,
    plotsdir,
    setup_environment,
    upload_to_cloud,
)
from _figstyle import apply_house_style, despine

from qfield.designed_quadrature import (
    constrained_weights,
    constrained_weights_batched,
)

from qfield.designed_quadrature.cond_net import (
    ConditionalQuadratureAmortizer,
)

from qfield.kernels import median_heuristic_sigma_sq

from vspsampling.download import (
    DATASET as VSP_DATASET,
    DDPM as VSP_DDPM,
    REPO as VSP_ROOT,
)
from vspsampling.models.conditional_ddpm import ConditionalDDPM


CONFIG_FILE = "dq_vsp.json"
DTYPE = torch.float64

ROOT = Path(__file__).resolve().parents[1]
BANK_FILE = ROOT / "data" / "vsp_posterior_train_51200.h5"

PALETTE = {
    "floor": "#3a7ca5",
    "reweight": "#2a9d8f",
    "ours": "#ff7f0e",
    "null": "#b06fa8",
}



def median_iqr(vals) -> tuple[float, float, float]:
    t = torch.tensor(list(vals), dtype=DTYPE)
    t = t[torch.isfinite(t)]
    if t.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], dtype=DTYPE))
    return float(q[1]), float(q[0]), float(q[2])
    

class VSPDDPMSampler:
    """Use the pretrained VSP DDPM checkpoint."""

    def __init__(self, device, n_steps=50):
        self.device = torch.device(device)
        self.n_steps = int(n_steps)

        ckpt_path = os.path.join(VSP_ROOT, VSP_DDPM)

        print(f"[DDPM] Loading {ckpt_path}", flush=True)

        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )

        cfg = ckpt["args"]

        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            self.model = ConditionalDDPM(
                d_x=41,
                d_s=20,
                width=cfg.width,
                depth=cfg.depth,
                T=cfg.T,
            )
        finally:
            torch.set_default_dtype(old_dtype)

        self.model.load_state_dict(ckpt["model"])
        self.model = self.model.float().to(self.device).eval()

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.normalizer = ckpt["y_normalizer"]

    def normalize(self, y):
        return self.normalizer.normalize(
            y.to(device="cpu", dtype=torch.float32)
        )

    @torch.no_grad()
    def draw(self, y_normalized, n, generator):
        """Generate (n_obs, n, 41) posterior samples on CUDA."""

        y = y_normalized.to(
            device=self.device,
            dtype=torch.float32,
        )

        seed = int(
            torch.randint(
                0, 2**31 - 1, (1,), generator=generator
            ).item()
        )

        gpu_gen = torch.Generator(
            device=self.device
        ).manual_seed(seed)

        samples = []
        old_dtype = torch.get_default_dtype()

        try:
            torch.set_default_dtype(torch.float32)

            for obs in y:
                cond = obs.unsqueeze(0).expand(int(n), -1)

                result = self.model.sample(
                    cond,
                    n_steps=self.n_steps,
                    generator=gpu_gen,
                )
                samples.append(result)

        finally:
            torch.set_default_dtype(old_dtype)

        return torch.stack(samples).to(dtype=DTYPE)


class DQVSP:
    """GPU QField amortized quadrature for VSP."""

    def __init__(self, args):
        self.args = args
        self.dev = torch.device(f"cuda:{int(args.gpu_id)}")

        self.d = int(args.d)
        self.dy = int(args.dy)
        self.m_list = [int(m) for m in args.m_list]

        if (self.d, self.dy) != (41, 20):
            raise ValueError("VSP requires d=41 and dy=20.")

        self.bank_path = BANK_FILE

        # Load metadata, not the full posterior bank.
        with h5py.File(self.bank_path, "r") as f:
            self.indices = f["indices"][:]
            self.bank_shape = tuple(
                f["posterior_samples"].shape
            )
            self.bank_size = self.bank_shape[1]
            self.n_obs = len(self.indices)

            if int(f.attrs["ddim_steps"]) != int(
                args.ddim_steps
            ):
                raise ValueError("DDIM steps mismatch.")

            if "completed" in f:
                if not bool(np.all(f["completed"][:])):
                    raise ValueError(
                        "Posterior dataset generation is incomplete."
                    )

            self.y_train = torch.from_numpy(
                np.asarray(
                    f["y_normalized"][:],
                    dtype=np.float32,
                )
            ).to(self.dev, dtype=DTYPE)

        # Training rows must come from the DDPM's train split.
        with h5py.File(
            os.path.join(VSP_ROOT, VSP_DATASET), "r"
        ) as f:
            splits = f["split"][:]

            if not np.all(splits[self.indices] == 0):
                raise ValueError(
                    "Posterior bank contains non-training observations."
                )

            sigma_y = float(f.attrs["sigma"])

        # Eval observations are fresh simulator samples, not split-1 rows:
        # those rows early-stopped the DDPM (as in dq_darcy_stuart.py).
        from vspsampling import LayeredPrior, TraveltimeOperator, vsp_grid
        gen = torch.Generator().manual_seed(int(args.base_seed) + 777_000)
        rng = np.random.default_rng(int(args.base_seed) + 777_000)
        _, m_ev = LayeredPrior(vsp_grid().zcoords()).sample(
            int(args.n_eval_obs), gen
        )
        op = TraveltimeOperator()
        with torch.no_grad():
            tau = op(m_ev).numpy()
        op.close()
        y_ev = tau + sigma_y * rng.standard_normal(tau.shape)
        self.y_eval_raw = torch.as_tensor(y_ev, dtype=torch.float32)

        # DDPM is initialized only when evaluation needs it.
        self.sampler = None

        print(
            f"[HDF5] {self.bank_path}\n"
            f"[HDF5] shape={self.bank_shape}\n"
            f"[HDF5] posterior samples remain on disk",
            flush=True,
        )

    def get_sampler(self):
        if self.sampler is None:
            self.sampler = VSPDDPMSampler(
                self.dev,
                n_steps=int(self.args.ddim_steps),
            )
        return self.sampler

    def draw_from_hdf5(self, obs_rows, n, generator):
        """Randomly sample stored posterior particles.

        HDF5 I/O occurs on CPU; returned tensors are on CUDA.
        """
        n = int(n)

        if n > self.bank_size:
            raise ValueError(
                f"Requested {n} samples, "
                f"but bank contains {self.bank_size}."
            )

        rows = (
            obs_rows.tolist()
            if isinstance(obs_rows, torch.Tensor)
            else list(obs_rows)
        )

        batches = []

        with h5py.File(self.bank_path, "r") as f:
            dataset = f["posterior_samples"]

            for row in rows:
                chosen = torch.randperm(
                    self.bank_size,
                    generator=generator,
                )[:n].numpy()

                # Sorted indices are required by h5py.
                order = np.argsort(chosen)

                if n == self.bank_size:
                    block = np.asarray(
                        dataset[int(row), :, :]
                    )[chosen]
                else:
                    block = np.asarray(
                        dataset[int(row), chosen[order], :]
                    )[np.argsort(order)]

                batches.append(
                    torch.from_numpy(
                        np.ascontiguousarray(block)
                    )
                )

        return torch.stack(batches).to(
            device=self.dev,
            dtype=DTYPE,
        )

    def sigma_estimate(self):
        n_obs = min(64, self.n_obs)
        n_samples = int(self.args.sigma_n_sample)

        per = min(
            self.bank_size,
            max(2, math.ceil(n_samples / n_obs)),
        )

        generator = torch.Generator().manual_seed(
            int(self.args.base_seed) + 1234
        )

        draws = self.draw_from_hdf5(
            range(n_obs), per, generator
        ).reshape(-1, self.d)[:n_samples]

        sigma_sq = median_heuristic_sigma_sq(draws)
        sigma = math.sqrt(float(sigma_sq))

        print(f"[sigma] {sigma:.6f}", flush=True)
        return sigma

    def build_net(self):
        a = self.args

        net = ConditionalQuadratureAmortizer(
            d=self.d,
            hidden=int(a.hidden),
            n_blocks=int(a.n_blocks),
            n_heads=int(a.n_heads),
            cond_hidden=int(a.cond_hidden),
            delta_scale=float(a.delta_scale),
            cond_token_dim=int(a.cond_token_dim),
            cond_hidden_ch=int(a.cond_hidden_ch),
            cond_kind="vector",
            cond_vec_dim=self.dy,
            cond_n_tokens=int(a.cond_n_tokens),
        )

        return net.to(device=self.dev, dtype=DTYPE)

    def objective(self, net, z0, bank, y, sigma):
        """Original QField MMD objective."""

        inv2s = 1.0 / (2.0 * sigma**2)

        z = net(z0, y)

        mu = torch.exp(
            -torch.cdist(z, bank).square() * inv2s
        ).mean(dim=-1)

        w = constrained_weights_batched(
            z, mu, sigma, self.args.jitter
        )

        gram = torch.exp(
            -torch.cdist(z, z).square() * inv2s
        )

        loss = (
            torch.einsum("bi,bij,bj->b", w, gram, w)
            - 2.0 * torch.einsum("bi,bi->b", w, mu)
        ).mean()

        return loss

    def train(self):
        a = self.args

        if max(self.m_list) + int(a.n_mu_train) > self.bank_size:
            raise ValueError(
                "Need at least max(M) + n_mu_train samples."
            )

        torch.manual_seed(int(a.seed))
        torch.cuda.manual_seed_all(int(a.seed))

        sigma = self.sigma_estimate()
        net = self.build_net()

        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=float(a.lr),
        )

        n_val = min(64, max(1, self.n_obs // 8))
        n_train = self.n_obs - n_val

        train_rows = torch.arange(n_train)
        val_rows = torch.arange(n_train, self.n_obs)

        generator = torch.Generator().manual_seed(
            int(a.base_seed) + 7
        )

        val_every = 200
        patience = int(getattr(a, "patience", 8))
        best_val = float("inf")
        stale = 0
        best_state = None

        train_curve = []
        val_curve = []
        ema = None

        # Directory for periodic checkpoints and training plots
        ck_dir = Path(checkpointsdir(a.experiment))
        ck_dir.mkdir(parents=True, exist_ok=True)

        plot_dir = Path(plotsdir(a.experiment))
        plot_dir.mkdir(parents=True, exist_ok=True)

        save_every = int(getattr(a, "save_every", 300))
        log_every = max(1, int(a.n_steps) // 20)
        val_nodes, val_bank = 128, 1024

        print(
            f"[split] train={n_train}, validation={n_val}",
            flush=True,
        )

        @torch.no_grad()
        def validate():
            net.eval()

            val_gen = torch.Generator().manual_seed(
                int(a.base_seed) + 55
            )

            total = 0.0

            for start in range(0, n_val, 8):
                rows = val_rows[start:start + 8]

                samples = self.draw_from_hdf5(
                    rows,
                    val_nodes + val_bank,
                    val_gen,
                )

                y = self.y_train[rows.to(self.dev)]

                loss = self.objective(
                    net,
                    samples[:, :val_nodes],
                    samples[:, val_nodes:],
                    y,
                    sigma,
                )

                total += float(loss) * len(rows)

            net.train()
            return total / n_val

        net.train()

        t0 = time.time()
        for step in range(int(a.n_steps)):
            M = self.m_list[
                torch.randint(
                    len(self.m_list), (1,), generator=generator
                ).item()
            ]

            selected = torch.randperm(
                n_train, generator=generator
            )[:int(a.batch_obs)]

            rows = train_rows[selected]

            samples = self.draw_from_hdf5(
                rows,
                M + int(a.n_mu_train),
                generator,
            )

            y = self.y_train[rows.to(self.dev)]

            z0 = samples[:, :M]
            bank = samples[:, M:]

            loss = self.objective(
                net, z0, bank, y, sigma
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Periodic checkpoint every save_every steps
            completed_step = step + 1

            if completed_step % save_every == 0:
                checkpoint_path = ck_dir / (
                    f"checkpoint_step_{completed_step:04d}.pth"
                )

                torch.save(
                    {
                        "step": completed_step,
                        "state_dict": {
                            k: v.detach().cpu().clone()
                            for k, v in net.state_dict().items()
                        },
                        "optimizer_state_dict": optimizer.state_dict(),
                        "sigma": sigma,
                        "train_curve": train_curve,
                        "val_curve": val_curve,
                    },
                    checkpoint_path,
                )

                print(
                    f"[checkpoint] Saved at step {completed_step}: "
                    f"{checkpoint_path}",
                    flush=True,
                )

            value = float(loss.detach())
            ema = (
                value if ema is None
                else 0.98 * ema + 0.02 * value
            )

            if (
                step % log_every == 0
                or step == int(a.n_steps) - 1
            ):
                train_curve.append([step, ema])

                print(
                    f"[train] step={step}/{a.n_steps} "
                    f"M={M} loss={value:.5e} "
                    f"EMA={ema:.5e}",
                    flush=True,
                )

            if (
                step % val_every == 0
                or step == int(a.n_steps) - 1
            ):
                score = validate()
                val_curve.append([step, score])

                print(
                    f"[val] step={step} "
                    f"J={score:.5e} best={best_val:.5e}",
                    flush=True,
                )

                if (
                    math.isfinite(score)
                    and score < best_val - 1e-9
                ):
                    best_val = score
                    stale = 0

                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in net.state_dict().items()
                    }
                else:
                    stale += 1

                    if stale >= patience:
                        print("[early-stop] patience reached")
                        break

        # Save final-step model before restoring best validation weights
        final_step = step + 1
        print(
            f"[train] wall {(time.time() - t0) / 60:.1f} min",
            flush=True,
        )
        torch.save(
            {
                "step": final_step,
                "state_dict": {
                    k: v.detach().cpu().clone()
                    for k, v in net.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "sigma": sigma,
                "train_curve": train_curve,
                "val_curve": val_curve,
            },
            ck_dir / "checkpoint_last.pth",
        )

        print("[checkpoint] Saved final training step.", flush=True)

        # Plot training and validation objectives
        fig, ax = plt.subplots(figsize=(7, 4.5))

        if len(train_curve) > 1:
            steps, values = zip(*train_curve[1:])
            ax.plot(steps, values, label="Training EMA")

        if val_curve:
            steps, values = zip(*val_curve)
            ax.plot(
                steps,
                values,
                marker="o",
                markersize=3,
                label="Validation",
            )

        ax.set_xlabel("Training step")
        ax.set_ylabel("QField objective")
        ax.set_title("QField VSP Training and Validation")
        ax.legend()
        ax.grid(True, alpha=0.25)

        fig.tight_layout()

        for ext in ("png", "pdf"):
            fig.savefig(
                plot_dir / f"dq_vsp_training_validation.{ext}",
                dpi=300,
                bbox_inches="tight",
            )

        plt.close(fig)

        print("[plot] Training/validation curves saved.", flush=True)

        # Restore best-validation model
        if best_state is not None:
            net.load_state_dict(best_state)

        results = {
            "train_wall_min": (time.time() - t0) / 60,
            "sigma": sigma,
            "m_list": self.m_list,
            "best_val": best_val,
            "n_train": n_train,
            "n_val_family": n_val,
            "n_params": sum(
                p.numel() for p in net.parameters()
            ),
            "train_curve": train_curve,
            "val_curve": val_curve,
            "posterior_source": str(self.bank_path),
            "posterior_sampler": "pretrained_vsp_ddpm",
            "rows": [],
        }

        self.save(
            ck_dir,
            net,
            results,
            "trained_before_evaluation.pth",
        )

        print(
            "[checkpoint] Trained QField saved "
            "before evaluation.",
            flush=True,
        )

        # Save training results without expensive DDPM evaluation
        with open(ck_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(
            "[training complete] Best model and training results saved. "
            "Run --phase evaluate separately.",
            flush=True,
        )

        return

    @staticmethod
    def save(directory, net, results, name):
        torch.save(
            {
                "state_dict": {
                    k: v.detach().cpu()
                    for k, v in net.state_dict().items()
                },
                "results": results,
            },
            directory / name,
        )

    @staticmethod
    def kernel_mean(z, bank, sigma, chunk=1024):
        """GPU kernel mean, chunked to limit memory."""

        inv2s = 1.0 / (2.0 * sigma**2)

        result = torch.zeros(
            len(z),
            device=z.device,
            dtype=z.dtype,
        )

        for start in range(0, len(bank), chunk):
            block = bank[start:start + chunk]

            result += torch.exp(
                -torch.cdist(z, block).square() * inv2s
            ).sum(dim=1)

        return result / len(bank)

    @classmethod
    def bank_constant(cls, bank, sigma):
        """Compute empirical kernel constant blockwise."""

        total = torch.zeros(
            (),
            device=bank.device,
            dtype=bank.dtype,
        )

        for start in range(0, len(bank), 512):
            block = bank[start:start + 512]
            total += cls.kernel_mean(
                block, bank, sigma
            ).sum()

        return float(total / len(bank))

    @classmethod
    def score(cls, z, w, bank, sigma, c_bank):
        mu = cls.kernel_mean(z, bank, sigma)

        gram = torch.exp(
            -torch.cdist(z, z).square()
            / (2.0 * sigma**2)
        )

        value = (
            w @ gram @ w
            - 2.0 * (w @ mu)
        )

        return max(float(value) + c_bank, 0.0)

    @torch.no_grad()
    def evaluate(self, net, sigma):
        """Two independent reference banks: A for fit, B for scoring."""

        a = self.args
        sampler = self.get_sampler()

        y_eval = sampler.normalize(
            self.y_eval_raw
        ).to(self.dev, dtype=DTYPE)

        net.eval()
        rows = []

        for M in self.m_list:
            L = (
                int(a.bank_L)
                if M < 256
                else 2 * int(a.bank_L)
            )

            keys = (
                "floor",
                "reweight",
                "raw",
                "ours",
                "null",
            )

            scores = {
                name: [] for name in keys
            }

            scores.update({
                name + "_heldout": []
                for name in keys
            })

            activations = 0
            violations = 0
            n_draws = 0
            for o in tqdm(
                range(len(y_eval)),
                desc=f"eval M={M}",
            ):
                gen_a = torch.Generator().manual_seed(
                    int(a.base_seed)
                    + 50000 + 1000 * M + o
                )

                gen_b = torch.Generator().manual_seed(
                    int(a.base_seed)
                    + 350000 + 1000 * M + o
                )

                bank_a = sampler.draw(
                    y_eval[o:o+1], L, gen_a
                )[0]

                bank_b = sampler.draw(
                    y_eval[o:o+1], L, gen_b
                )[0]

                c_a = self.bank_constant(bank_a, sigma)
                c_b = self.bank_constant(bank_b, sigma)

                per_observation = {name: [] for name in scores}
                for _ in range(int(a.eval_repeats)):
                    # Sample with replacement, preserving multiplicity.
                    idx = torch.randint(
                        L, (M,), generator=gen_a
                    ).to(self.dev)

                    uniq, counts = torch.unique(
                        idx, return_counts=True
                    )

                    z0 = bank_a[uniq]
                    w0 = counts.to(DTYPE) / float(M)

                    mu0 = self.kernel_mean(
                        z0, bank_a, sigma
                    )

                    wr = constrained_weights(
                        z0, mu0, sigma, a.jitter
                    )

                    z = net(
                        z0.unsqueeze(0),
                        y_eval[o:o+1],
                    )[0]

                    mu = self.kernel_mean(
                        z, bank_a, sigma
                    )

                    w = constrained_weights(
                        z, mu, sigma, a.jitter
                    )

                    null_index = (o - 1) % len(y_eval)

                    z_null = net(
                        z0.unsqueeze(0),
                        y_eval[
                            null_index:null_index+1
                        ],
                    )[0]

                    w_null = constrained_weights(
                        z_null,
                        self.kernel_mean(
                            z_null, bank_a, sigma
                        ),
                        sigma,
                        a.jitter,
                    )

                    candidates = [
                        (z0, w0),
                        (z0, wr),
                        (z, w),
                    ]

                    fit_scores = [
                        self.score(
                            zz, ww, bank_a, sigma, c_a
                        )
                        for zz, ww in candidates
                    ]

                    selected = int(
                        np.argmin(fit_scores)
                    )

                    activations += int(selected != 2)
                    violations += int(
                        fit_scores[selected]
                        > min(fit_scores) + 1e-10
                    )
                    n_draws += 1

                    z_selected, w_selected = (
                        candidates[selected]
                    )

                    arms = {
                        "floor": (z0, w0),
                        "reweight": (z0, wr),
                        "raw": (z, w),
                        "ours": (z_selected, w_selected),
                        "null": (z_null, w_null),
                    }

                    for name, (zz, ww) in arms.items():
                        per_observation[name].append(
                            self.score(
                                zz, ww, bank_a, sigma, c_a
                            )
                        )

                        per_observation[
                            name + "_heldout"
                        ].append(
                            self.score(
                                zz, ww, bank_b, sigma, c_b
                            )
                        )

                for name in scores:
                    scores[name].append(
                        float(np.nanmean(per_observation[name]))
                    )

                del bank_a, bank_b

            row = {
                "M": M,
                "L": L,
                "activation": (
                    activations / max(1, n_draws)
                ),
                "n_worse_than_menu_bankA": violations,
                "n_draws": n_draws,
            }

            for name, values in scores.items():
                row[name] = median_iqr(values)

            floor = row["floor_heldout"][0]
            ours = row["ours_heldout"][0]

            row["ours_over_floor_heldout"] = (
                ours / max(floor, 1e-30)
            )

            rows.append(row)

            print(
                f"[eval] M={M} "
                f"floor={floor:.5e} "
                f"ours={ours:.5e} "
                f"ratio={row['ours_over_floor_heldout']:.4f}",
                flush=True,
            )

        return rows

    @torch.no_grad()
    def check_moments(self, n_obs=2, m_values=(32, 256)):
        """Independent check of the moved nodes, weights, mean and std.

        Read-only: loads the trained network, draws two independent DDPM banks
        per evaluation observation (A: target, B: held-out reference) and, for
        one seed set per M, recomputes every step by a separate path:
          1. the emitted nodes are the network's moved nodes net(z0);
          2. the weights sum to 1 and equal an independent numpy solve of
             min w'Kw - 2 w'mu  s.t. sum(w) = 1 at those nodes;
          3. velocity v = exp(-T(z)) per node, then mean = sum w v and
             std = sqrt(sum w (v - mean)^2), against explicit loops.
        Then reports the RMS error of the velocity mean and std against bank B
        for i.i.d., reweight-only and QField. Writes nothing.
        """
        from vspsampling import LayeredPrior, vsp_grid

        a = self.args
        ck = Path(checkpointsdir(a.experiment)) / "trained_before_evaluation.pth"
        obj = torch.load(ck, map_location="cpu", weights_only=False)
        sigma = float(obj["results"]["sigma"])
        net = self.build_net()
        net.load_state_dict(obj["state_dict"])
        net.eval()
        sampler = self.get_sampler()
        y_eval = sampler.normalize(self.y_eval_raw).to(self.dev, dtype=DTYPE)
        prior = LayeredPrior(vsp_grid().zcoords())
        L = int(a.bank_L)

        def velocity(z):
            return np.exp(-prior(z.detach().double().cpu()).numpy())

        def sqdist(X, Y):
            return torch.cdist(torch.as_tensor(X), torch.as_tensor(Y)).square().numpy()

        n_fail = 0
        for o in range(n_obs):
            gen_a = torch.Generator().manual_seed(int(a.base_seed) + 950_000 + o)
            gen_b = torch.Generator().manual_seed(int(a.base_seed) + 960_000 + o)
            bank_a = sampler.draw(y_eval[o:o + 1], L, gen_a)[0]
            bank_b = sampler.draw(y_eval[o:o + 1], L, gen_b)[0]
            v_ref = velocity(bank_b)
            ref_mean, ref_std = v_ref.mean(0), v_ref.std(0)
            for M in m_values:
                idx = torch.randint(L, (M,), generator=gen_a).to(self.dev)
                uniq, counts = torch.unique(idx, return_counts=True)
                z0 = bank_a[uniq]
                w0 = counts.to(DTYPE) / float(M)
                wr = constrained_weights(
                    z0, self.kernel_mean(z0, bank_a, sigma), sigma, a.jitter)
                z = net(z0.unsqueeze(0), y_eval[o:o + 1])[0]
                w = constrained_weights(
                    z, self.kernel_mean(z, bank_a, sigma), sigma, a.jitter)

                Z = z.double().cpu().numpy()
                W = w.double().cpu().numpy()
                A = bank_a.double().cpu().numpy()
                K = (np.exp(-sqdist(Z, Z) / (2 * sigma ** 2))
                     + float(a.jitter) * np.eye(len(Z)))
                mu = np.exp(-sqdist(Z, A) / (2 * sigma ** 2)).mean(1)
                kkt = np.block([[2 * K, np.ones((len(Z), 1))],
                                [np.ones((1, len(Z))), np.zeros((1, 1))]])
                W_ref = np.linalg.solve(kkt, np.r_[2 * mu, 1.0])[:-1]
                v = velocity(z)
                mean = W @ v
                std = np.sqrt(np.clip(W @ (v - mean) ** 2, 0.0, None))
                mean_loop = np.array([sum(W[i] * v[i, d] for i in range(len(W)))
                                      for d in range(v.shape[1])])
                var_loop = np.array([sum(W[i] * (v[i, d] - mean_loop[d]) ** 2
                                         for i in range(len(W)))
                                     for d in range(v.shape[1])])
                move = float((z - z0).norm(dim=1).median() / sigma)
                dw = float(np.abs(W - W_ref).max() / np.abs(W_ref).max())
                checks = (
                    ("nodes are the moved nodes net(z0)", move > 0,
                     f"median move {move:.3f} sigma"),
                    ("weights sum to 1", abs(W.sum() - 1) < 1e-10,
                     f"sum = {W.sum():.12f}"),
                    ("weights = independent KKT solve", dw < 1e-6,
                     f"relative diff {dw:.1e}"),
                    ("mean = sum_i w_i v_i", np.abs(mean - mean_loop).max() < 1e-6,
                     f"max diff {np.abs(mean - mean_loop).max():.1e} m/s"),
                    ("std = sqrt(sum_i w_i (v_i - mean)^2)",
                     np.abs(std - np.sqrt(np.clip(var_loop, 0, None))).max() < 1e-6,
                     f"max diff {np.abs(std - np.sqrt(np.clip(var_loop, 0, None))).max():.1e} m/s"),
                )
                print(f"\n=== obs {o}, M = {M} ===", flush=True)
                for name, ok, info in checks:
                    n_fail += not ok
                    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {info}")
                for arm, (zz, ww) in (("iid", (z0, w0)), ("reweight", (z0, wr)),
                                      ("QField", (z, w))):
                    vv, ww = velocity(zz), ww.double().cpu().numpy()
                    m_ = ww @ vv
                    s_ = np.sqrt(np.clip(ww @ (vv - m_) ** 2, 0.0, None))
                    print(f"  {arm:>8}: velocity RMS error  mean "
                          f"{np.sqrt(np.mean((m_ - ref_mean) ** 2)):6.1f} m/s   std "
                          f"{np.sqrt(np.mean((s_ - ref_std) ** 2)):6.1f} m/s")
            del bank_a, bank_b
        print(f"\n{'ALL CHECKS PASSED' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'}")

    def load_checkpoint(self):
        ck = (
            Path(checkpointsdir(self.args.experiment))
            / "checkpoint.pth"
        )

        if not ck.exists():
            raise FileNotFoundError(ck)

        obj = torch.load(
            ck,
            map_location="cpu",
            weights_only=False,
        )

        self.results = obj["results"]
        return obj

    def evaluate_checkpoint(self):
        ck_dir = Path(
            checkpointsdir(self.args.experiment)
        )

        ck = ck_dir / "trained_before_evaluation.pth"

        obj = torch.load(
            ck,
            map_location="cpu",
            weights_only=False,
        )

        net = self.build_net()
        net.load_state_dict(obj["state_dict"])

        results = obj["results"]

        results["rows"] = self.evaluate(
            net, float(results["sigma"])
        )

        self.save(
            ck_dir, net, results, "checkpoint.pth"
        )

        with open(
            ck_dir / "results.json", "w"
        ) as f:
            json.dump(results, f, indent=2)

        self.results = results
        self.visualize()

    def visualize(self):
        rows = self.results.get("rows", [])

        if not rows:
            print("[plot] No evaluation results.")
            return

        out = Path(plotsdir(self.args.experiment))
        out.mkdir(parents=True, exist_ok=True)

        apply_house_style()

        fig, ax = plt.subplots(figsize=(5.2, 4.0))

        m_values = [r["M"] for r in rows]

        labels = {
            "floor": "i.i.d. floor",
            "reweight": "reweight only",
            "ours": "QField",
            "null": "shuffled conditioner",
        }

        for name in (
            "floor", "reweight", "null", "ours"
        ):
            vals = [
                r[name + "_heldout"]
                for r in rows
            ]

            med = [v[0] for v in vals]
            lo = [max(0, v[0]-v[1]) for v in vals]
            hi = [max(0, v[2]-v[0]) for v in vals]

            ax.errorbar(
                m_values,
                med,
                yerr=[lo, hi],
                color=PALETTE[name],
                marker="o",
                capsize=2.5,
                label=labels[name],
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(m_values)
        ax.set_xticklabels([
            str(m) for m in m_values
        ])

        ax.set_xlabel("node budget M")
        ax.set_ylabel(
            r"$\mathrm{MMD}^2(Q,\rho_y)$"
        )

        despine(ax)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()

        for ext in ("png", "pdf"):
            path = out / f"dq_vsp_mmd_vs_M.{ext}"

            fig.savefig(
                path,
                dpi=300,
                bbox_inches="tight",
            )

            print(f"[plot] {path}", flush=True)

        plt.close(fig)


def parse_args():
    return setup_environment(
        CONFIG_FILE,
        ignore_arg_list=[
            "experiment_name",
            "gpu_id",
            "phase",
            "upload",
            "recon_M",
            "n_recon",
            "eval_repeats",
            "n_pcn_audit",
            "ddim_steps",
        ],
        sequence_args_and_types=[
            ("m_list", int)
        ],
    )


if __name__ == "__main__":
    args = parse_args()

    if args.phase == "train":
        experiment = DQVSP(args)
        experiment.train()

    elif args.phase == "evaluate":
        experiment = DQVSP(args)
        experiment.evaluate_checkpoint()

    elif args.phase == "check":
        experiment = DQVSP(args)
        experiment.check_moments()

    elif args.phase == "visualize":
        experiment = DQVSP(args)
        experiment.load_checkpoint()
        experiment.visualize()

    else:
        raise ValueError(
            "Supported phases: train, evaluate, visualize, check"
        )

    if args.upload:
        upload_to_cloud(args)