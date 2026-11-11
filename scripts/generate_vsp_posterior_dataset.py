"""Generate a resumable HDF5 posterior-sample bank for VSP."""

from pathlib import Path
import argparse
import time

import h5py
import numpy as np
import torch

from vspsampling.download import DATASET, REPO
from dq_vsp import VSPDDPM


BASE_SEED = 20260612
DEFAULT_N_OBS = 51200
DEFAULT_SAMPLES_PER_OBS = 2560
DEFAULT_DDIM_STEPS = 50


def generate(
    n_obs=DEFAULT_N_OBS,
    output="../data/vsp_posterior_train_51200.h5",
    samples_per_obs=DEFAULT_SAMPLES_PER_OBS,
    ddim_steps=DEFAULT_DDIM_STEPS,
):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available!")

    device = "cuda:0"
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # Load the original VSP training observations.
    source_path = Path(REPO) / DATASET

    with h5py.File(source_path, "r") as src:
        splits = src["split"][:]
        train_indices = np.flatnonzero(splits == 0)

        if n_obs > len(train_indices):
            raise ValueError(
                f"Requested {n_obs} observations, "
                f"but only {len(train_indices)} are available."
            )

        rng = np.random.default_rng(BASE_SEED + 41)

        selected = np.sort(
            rng.choice(
                train_indices,
                size=n_obs,
                replace=False,
            )
        )

        y_raw = src["y"][selected]
        xi = src["xi"][selected]

    print(f"Selected {len(selected)} training observations")

    # Load Ali's pretrained DDPM through the existing sampler.
    sampler = VSPDDPM(device, ddim_steps)

    y_normalized = sampler.normalize(
        torch.from_numpy(y_raw)
    ).cpu().numpy()

    # Create a new dataset or resume an existing one.
    if output.exists():
        print(f"Resuming existing dataset: {output}")

        dst = h5py.File(output, "r+")

        try:
            if "completed" not in dst:
                raise RuntimeError(
                    "Existing file has no completion tracking. "
                    "Refusing to overwrite it."
                )

            if not np.array_equal(dst["indices"][:], selected):
                raise RuntimeError(
                    "Existing dataset uses different observation indices."
                )

            expected_shape = (n_obs, samples_per_obs, 41)

            if dst["posterior_samples"].shape != expected_shape:
                raise RuntimeError(
                    "Existing dataset has incompatible dimensions."
                )

            if int(dst.attrs["ddim_steps"]) != ddim_steps:
                raise RuntimeError("DDIM steps do not match.")

            if int(dst.attrs["base_seed"]) != BASE_SEED:
                raise RuntimeError("Base seed does not match.")

        except Exception:
            dst.close()
            raise

    else:
        print(f"Creating new dataset: {output}")

        dst = h5py.File(output, "w")

        dst.attrs["base_seed"] = BASE_SEED
        dst.attrs["ddim_steps"] = ddim_steps
        dst.attrs["samples_per_obs"] = samples_per_obs
        dst.attrs["observations_normalized"] = True

        dst.create_dataset("indices", data=selected)
        dst.create_dataset("y", data=y_raw)
        dst.create_dataset("y_normalized", data=y_normalized)
        dst.create_dataset("xi", data=xi)

        dst.create_dataset(
            "posterior_samples",
            shape=(n_obs, samples_per_obs, 41),
            dtype="f8",
            chunks=(1, min(samples_per_obs, 256), 41),
        )

        dst.create_dataset(
            "completed",
            shape=(n_obs,),
            dtype="bool",
            data=np.zeros(n_obs, dtype=bool),
        )

        dst.flush()

    start_time = time.time()

    try:
        completed = dst["completed"]
        samples_ds = dst["posterior_samples"]

        already_done = int(np.count_nonzero(completed[:]))

        print(
            f"Completed: {already_done}/{n_obs}",
            flush=True,
        )

        for i in range(n_obs):
            if completed[i]:
                continue

            # Reproducible random stream per observation.
            generator = torch.Generator().manual_seed(
                BASE_SEED + 10000 + int(selected[i])
            )

            observation = torch.from_numpy(
                y_normalized[i:i + 1]
            )

            with torch.no_grad():
                samples = sampler.sample(
                    observation,
                    samples_per_obs,
                    generator,
                )

            sample_array = (
                samples[0].detach().cpu().numpy()
            )

            if sample_array.shape != (samples_per_obs, 41):
                raise RuntimeError(
                    f"Unexpected sample shape: {sample_array.shape}"
                )

            if not np.isfinite(sample_array).all():
                raise RuntimeError(
                    f"Non-finite samples at observation {i}"
                )

            # Write samples before marking the row completed.
            samples_ds[i] = sample_array
            dst.flush()

            completed[i] = True
            dst.flush()

            if (i + 1) % 50 == 0 or i == n_obs - 1:
                elapsed = time.time() - start_time
                done = int(np.count_nonzero(completed[:]))
                print(
                    f"Progress: {done}/{n_obs} "
                    f"({100 * done / n_obs:.2f}%) | "
                    f"Elapsed: {elapsed / 3600:.2f} hours",
                    flush=True,
                )

    finally:
        dst.close()

    print(f"Dataset generation complete: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-obs",
        type=int,
        default=DEFAULT_N_OBS,
    )

    parser.add_argument(
        "--output",
        default="../data/vsp_posterior_train_51200.h5",
    )

    parser.add_argument(
        "--samples-per-obs",
        type=int,
        default=DEFAULT_SAMPLES_PER_OBS,
    )

    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=DEFAULT_DDIM_STEPS,
    )

    args = parser.parse_args()

    generate(
        n_obs=args.n_obs,
        output=args.output,
        samples_per_obs=args.samples_per_obs,
        ddim_steps=args.ddim_steps,
    )