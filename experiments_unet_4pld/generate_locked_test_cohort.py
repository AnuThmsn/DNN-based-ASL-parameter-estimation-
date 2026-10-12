"""
Locked Evaluation Cohort Generator
==================================
Generates a new, independent, frozen test cohort of 100 2D synthetic brain phantoms
(seed 777) and pre-computes controlled SNR evaluation snapshots:
SNR in [5, 10, 15, 20, 50, inf].
This locks a clean evaluation cohort for the expanded scaling experiment,
completely separate from the historical test cohort (seed 7).
"""

import os
import sys
import hashlib
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments_unet_4pld.workspace import get_workspace
from experiments_unet_4pld.generate_spatial_data import generate_spatial_dataset
from experiments_unet_4pld.run_snr_benchmark import generate_snr_dataset


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def create_locked_test_cohort(
    data_dir: str,
    n_phantoms: int = 100,
    seed: int = 777,
    snr_levels = (5.0, 10.0, 15.0, 20.0, 50.0, float("inf"))
):
    os.makedirs(data_dir, exist_ok=True)
    base_file = os.path.join(data_dir, f"test_locked_N{n_phantoms}_s{seed}.npz")

    print(f"\n--- Generating Locked Test Cohort (N={n_phantoms}, seed={seed}) ---")
    if os.path.exists(base_file):
        print(f"Locked test cohort already exists: {base_file}")
        base_data = np.load(base_file)
        X_base = base_data["X"]
        Y_base = base_data["Y"]
        masks_base = base_data["masks"]
    else:
        # Generate base ground-truth phantoms
        X_base, Y_base, masks_base, snrs_base, sds_base = generate_spatial_dataset(
            n_phantoms=n_phantoms,
            seed=seed,
            img_size=64
        )
        np.savez_compressed(
            base_file,
            X=X_base,
            Y=Y_base,
            masks=masks_base,
            snrs=snrs_base,
            sds=sds_base
        )
        print(f"Saved locked base test cohort: {base_file}")

    base_sha256 = sha256_file(base_file)
    valid_voxels = int(np.sum(masks_base[:, 0] > 0.5))
    print(f"Base cohort SHA-256: {base_sha256}")
    print(f"Total valid brain voxels: {valid_voxels}")

    # Generate SNR snapshots
    manifest = {
        "base_file": os.path.abspath(base_file),
        "base_sha256": base_sha256,
        "n_phantoms": n_phantoms,
        "seed": seed,
        "valid_voxels": valid_voxels,
        "snr_snapshots": {}
    }

    for snr in snr_levels:
        snr_tag = "inf" if snr == float("inf") else str(int(snr))
        snr_file = os.path.join(data_dir, f"test_locked_N{n_phantoms}_s{seed}_snr_{snr_tag}.npz")
        if not os.path.exists(snr_file):
            print(f"Generating locked SNR snapshot: SNR={snr} -> {snr_file}")
            X_snr, meas_sd = generate_snr_dataset(
                Y_gt=Y_base,
                masks=masks_base,
                snr=snr,
                seed=seed + 1000
            )
            np.savez_compressed(
                snr_file,
                X=X_snr,
                Y=Y_base,
                masks=masks_base,
                snr=np.float32(snr),
                meas_sd=np.float32(meas_sd)
            )
        h = sha256_file(snr_file)
        manifest["snr_snapshots"][snr_tag] = {
            "path": os.path.abspath(snr_file),
            "snr": snr,
            "sha256": h
        }
        print(f"SNR={snr_tag}: SHA-256={h[:16]}...")

    return manifest


if __name__ == "__main__":
    ws = get_workspace(min_free_gb=5.0)
    data_dir = os.path.join(ws.root, "data")
    manifest = create_locked_test_cohort(data_dir=data_dir)
    print("\nLocked test cohort creation complete.")

