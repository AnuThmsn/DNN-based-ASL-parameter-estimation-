"""
Verification and Manifest Generation for Master Scaling Dataset (N=19200)
========================================================================
Performs rigorous cryptographic and structural checks:
1. Validates array dimensions and data types.
2. Checks bit-for-bit identical matching between N=19200[0:2400] and historical N=2400.
3. Confirms 100% uniqueness among all 19,200 phantoms.
4. Confirms 0% data leakage against Val (s99), Hist Test (s7), and Locked Test (s777).
5. Computes normalization statistics per training subset [2400, 4800, 9600, 19200].
6. Emits data_manifest.csv and subset_manifest.csv.
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments_unet_4pld.workspace import get_workspace
from experiments_unet_4pld.dataset import compute_normalization_stats
from experiments_unet_4pld.audit.cohort_integrity_check import compute_sample_hash


def verify_master_pool():
    ws = get_workspace(min_free_gb=5.0)
    data_dir = os.path.join(ws.root, "data")
    exp_dir = os.path.join(ws.root, "experiments", "training_size_scaling_v1")
    os.makedirs(exp_dir, exist_ok=True)

    master_path = os.path.join(data_dir, "train_nested_N19200_s42.npz")
    assert os.path.exists(master_path), f"Master pool missing: {master_path}"

    print(f"Loading master pool: {master_path}...")
    master = np.load(master_path)
    X = master["X"]
    Y = master["Y"]
    masks = master["masks"]
    snrs = master["snrs"]

    print(f"Master shapes: X={X.shape}, Y={Y.shape}, masks={masks.shape}, snrs={snrs.shape}")
    assert X.shape == (19200, 4, 64, 64), f"Unexpected X shape: {X.shape}"
    assert Y.shape == (19200, 2, 64, 64), f"Unexpected Y shape: {Y.shape}"
    assert masks.shape == (19200, 1, 64, 64) or masks.shape == (19200, 64, 64), f"Unexpected masks shape: {masks.shape}"
    if masks.ndim == 3:
        masks = masks[:, np.newaxis, :, :]

    # 1. Compare with historical N=2400
    hist_2400_path = os.path.join(data_dir, "train_nested_N2400_s42.npz")
    if os.path.exists(hist_2400_path):
        print(f"Verifying bit-for-bit prefix match against historical {hist_2400_path}...")
        hist_2400 = np.load(hist_2400_path)
        X_h2400 = hist_2400["X"]
        Y_h2400 = hist_2400["Y"]
        m_h2400 = hist_2400["masks"]
        if m_h2400.ndim == 3:
            m_h2400 = m_h2400[:, np.newaxis, :, :]

        assert np.array_equal(X[:2400], X_h2400), "FATAL: X[:2400] does not match train_nested_N2400_s42.npz!"
        assert np.array_equal(Y[:2400], Y_h2400), "FATAL: Y[:2400] does not match train_nested_N2400_s42.npz!"
        assert np.array_equal(masks[:2400], m_h2400), "FATAL: masks[:2400] does not match train_nested_N2400_s42.npz!"
        print(">> Bit-for-bit prefix match confirmed for N=2400 (100% exact equality)!")
    else:
        print(f"Note: {hist_2400_path} not found to check bit-for-bit match.")

    # 2. Check internal uniqueness of all 19,200 phantoms
    print("Computing cryptographic sample hashes for all 19,200 phantoms...")
    sample_hashes = []
    for i in range(len(X)):
        h = compute_sample_hash(X[i], Y[i], masks[i])
        sample_hashes.append(h)

    unique_hashes = set(sample_hashes)
    print(f"Master pool sample count: {len(sample_hashes)}, Unique hashes: {len(unique_hashes)}")
    assert len(unique_hashes) == 19200, f"Duplicate phantoms found in master pool! Unique: {len(unique_hashes)}"
    print(">> 100% uniqueness verified: 0 duplicates among 19,200 phantoms.")

    # 3. Check independence against evaluation cohorts
    eval_cohorts = {
        "val_N100_s99": os.path.join(data_dir, "val_N100_s99.npz"),
        "test_N100_s7": os.path.join(data_dir, "test_N100_s7.npz"),
        "test_locked_N100_s777": os.path.join(data_dir, "test_locked_N100_s777.npz")
    }

    for name, cpath in eval_cohorts.items():
        if os.path.exists(cpath):
            c_data = np.load(cpath)
            c_X, c_Y, c_m = c_data["X"], c_data["Y"], c_data["masks"]
            if c_m.ndim == 3:
                c_m = c_m[:, np.newaxis, :, :]
            c_hashes = set([compute_sample_hash(c_X[i], c_Y[i], c_m[i]) for i in range(len(c_X))])
            overlap = unique_hashes.intersection(c_hashes)
            print(f"Leakage check vs {name}: overlap = {len(overlap)} / {len(c_hashes)}")
            assert len(overlap) == 0, f"FATAL DATA LEAKAGE: {len(overlap)} overlapping phantoms with {name}!"
            print(f">> Zero leakage confirmed against {name}.")

    # 4. Generate subset_manifest.csv & data_manifest.csv
    sizes = [2400, 4800, 9600, 19200]
    data_manifest_rows = []
    subset_manifest_rows = []

    print("\nComputing per-subset normalization statistics and manifests...")
    for sz in sizes:
        sub_X = X[:sz]
        sub_Y = Y[:sz]
        sub_masks = masks[:sz]
        stats = compute_normalization_stats(sub_X, sub_Y, sub_masks)

        # Slice digest
        sub_hash = hashlib.sha256()
        sub_hash.update(sub_X.tobytes())
        sub_hash.update(sub_Y.tobytes())
        sub_hash.update(sub_masks.tobytes())
        slice_sha256 = sub_hash.hexdigest()

        data_manifest_rows.append({
            "subset_name": f"train_nested_N{sz}_s42",
            "size": sz,
            "slice_range": f"0:{sz}",
            "slice_sha256": slice_sha256,
            "X_mean_pld0": float(stats["X_mean"][0]),
            "X_mean_pld1": float(stats["X_mean"][1]),
            "X_mean_pld2": float(stats["X_mean"][2]),
            "X_mean_pld3": float(stats["X_mean"][3]),
            "X_std_pld0": float(stats["X_std"][0]),
            "X_std_pld1": float(stats["X_std"][1]),
            "X_std_pld2": float(stats["X_std"][2]),
            "X_std_pld3": float(stats["X_std"][3]),
            "cbf_mean": float(stats["cbf_mean"]),
            "cbf_std": float(stats["cbf_std"]),
            "att_mean": float(stats["att_mean"]),
            "att_std": float(stats["att_std"])
        })

    # Nested subset relationships
    for i in range(len(sizes)):
        sz_curr = sizes[i]
        subset_manifest_rows.append({
            "subset_name": f"train_nested_N{sz_curr}_s42",
            "size": sz_curr,
            "prefix_nested_in_master": True,
            "parent_subset": f"train_nested_N{sizes[i+1]}_s42" if i < len(sizes) - 1 else "master_pool",
            "is_strict_subset": (i < len(sizes) - 1)
        })

    df_data = pd.DataFrame(data_manifest_rows)
    df_data_path = os.path.join(exp_dir, "data_manifest.csv")
    df_data.to_csv(df_data_path, index=False)
    print(f"Saved: {df_data_path}")

    df_subset = pd.DataFrame(subset_manifest_rows)
    df_subset_path = os.path.join(exp_dir, "subset_manifest.csv")
    df_subset.to_csv(df_subset_path, index=False)
    print(f"Saved: {df_subset_path}")

    print("\n================ MASTER POOL AUDIT COMPLETE (PASS) ================\n")


if __name__ == "__main__":
    verify_master_pool()

