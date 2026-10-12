"""
Cohort Integrity, Unique Phantom ID, and Data Leakage Audit Module
==================================================================
This module provides cryptographic, sample-level audit checks for ASL synthetic
brain phantom datasets. It replaces fragile single-pixel heuristics (e.g. X[:, 0, 32, 32])
with full SHA-256 digests over (input, target, mask) raw byte arrays, persistent
phantom IDs, explicit set intersections, and nested subset verification.
"""

import os
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Set, Any


def compute_sample_hash(x_sample: np.ndarray, y_sample: np.ndarray, mask_sample: np.ndarray) -> str:
    """
    Computes a cryptographic SHA-256 digest over the complete raw bytes of a phantom.
    Includes input multi-PLD volume, ground-truth parameter maps, and tissue mask.
    """
    hasher = hashlib.sha256()
    hasher.update(x_sample.tobytes())
    hasher.update(y_sample.tobytes())
    hasher.update(mask_sample.tobytes())
    return hasher.hexdigest()


def audit_cohort_dataset(
    npz_path: str,
    split_name: str,
    seed: int
) -> Dict[str, Any]:
    """
    Loads an NPZ dataset and computes full cryptographic records for each phantom:
    - Persistent unique phantom IDs
    - Full sample SHA-256 digests
    - Input-only, target-only, and mask-only hashes
    - Valid brain voxel counts
    """
    assert os.path.exists(npz_path), f"Cohort file not found: {npz_path}"
    data = np.load(npz_path)
    X = data["X"]
    Y = data["Y"]
    masks = data["masks"]
    if masks.ndim == 3:
        masks = masks[:, np.newaxis, :, :]

    n_samples = len(X)
    records = []
    full_hashes = []
    input_hashes = []
    target_hashes = []

    for i in range(n_samples):
        phantom_id = f"{split_name}_s{seed}_idx{i:04d}"
        f_hash = compute_sample_hash(X[i], Y[i], masks[i])
        in_hash = hashlib.sha256(X[i].tobytes()).hexdigest()
        tg_hash = hashlib.sha256(Y[i].tobytes()).hexdigest()
        m_hash = hashlib.sha256(masks[i].tobytes()).hexdigest()
        n_vox = int(np.sum(masks[i, 0] > 0.5))

        records.append({
            "phantom_id": phantom_id,
            "sample_index": i,
            "full_hash": f_hash,
            "input_hash": in_hash,
            "target_hash": tg_hash,
            "mask_hash": m_hash,
            "valid_voxels": n_vox
        })
        full_hashes.append(f_hash)
        input_hashes.append(in_hash)
        target_hashes.append(tg_hash)

    total_valid_voxels = int(np.sum(masks[:, 0] > 0.5))
    unique_full = len(set(full_hashes))

    return {
        "split_name": split_name,
        "seed": seed,
        "path": os.path.abspath(npz_path),
        "n_samples": n_samples,
        "total_valid_voxels": total_valid_voxels,
        "unique_samples": unique_full,
        "is_internally_unique": (unique_full == n_samples),
        "records": records,
        "full_hashes": full_hashes,
        "input_hashes": input_hashes,
        "target_hashes": target_hashes
    }


def check_splits_independence(
    train_audit: Dict[str, Any],
    val_audit: Dict[str, Any],
    test_audit: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates mutual exclusivity among Train, Validation, and Test cohorts.
    Returns set intersection counts across full sample hashes, input hashes,
    and persistent phantom IDs.
    """
    tr_full = set(train_audit["full_hashes"])
    val_full = set(val_audit["full_hashes"])
    te_full = set(test_audit["full_hashes"])

    tr_in = set(train_audit["input_hashes"])
    val_in = set(val_audit["input_hashes"])
    te_in = set(test_audit["input_hashes"])

    overlap_tr_val = tr_full.intersection(val_full)
    overlap_tr_te = tr_full.intersection(te_full)
    overlap_val_te = val_full.intersection(te_full)

    in_overlap_tr_val = tr_in.intersection(val_in)
    in_overlap_tr_te = tr_in.intersection(te_in)
    in_overlap_val_te = val_in.intersection(te_in)

    is_independent = (
        len(overlap_tr_val) == 0 and
        len(overlap_tr_te) == 0 and
        len(overlap_val_te) == 0 and
        len(in_overlap_tr_val) == 0 and
        len(in_overlap_tr_te) == 0 and
        len(in_overlap_val_te) == 0
    )

    return {
        "is_independent": is_independent,
        "overlap_train_val": len(overlap_tr_val),
        "overlap_train_test": len(overlap_tr_te),
        "overlap_val_test": len(overlap_val_te),
        "input_overlap_train_val": len(in_overlap_tr_val),
        "input_overlap_train_test": len(in_overlap_tr_te),
        "input_overlap_val_test": len(in_overlap_val_te),
        "train_samples": train_audit["n_samples"],
        "val_samples": val_audit["n_samples"],
        "test_samples": test_audit["n_samples"]
    }


def verify_nested_subsets(
    master_train_audit: Dict[str, Any],
    subset_sizes: List[int]
) -> Dict[str, Any]:
    """
    Verifies that nested subsets D_N = D_master[:N] preserve strict prefix inclusion:
    D_100 ⊂ D_200 ⊂ D_400 ⊂ D_600 ⊂ D_1200 ⊂ D_2400.
    """
    master_hashes = master_train_audit["full_hashes"]
    results = {}
    prev_set = set()

    for size in sorted(subset_sizes):
        assert size <= len(master_hashes), f"Requested size {size} exceeds master pool {len(master_hashes)}"
        sub_hashes = master_hashes[:size]
        sub_set = set(sub_hashes)

        # Check internal uniqueness
        is_unique = (len(sub_set) == size)
        # Check strict superset of previous
        is_superset = prev_set.issubset(sub_set) if prev_set else True

        results[size] = {
            "size": size,
            "unique_count": len(sub_set),
            "is_internally_unique": is_unique,
            "contains_previous_subset": is_superset,
            "hash_prefix_start": sub_hashes[0][:16],
            "hash_prefix_end": sub_hashes[-1][:16]
        }
        prev_set = sub_set

    all_strictly_nested = all(v["contains_previous_subset"] and v["is_internally_unique"] for v in results.values())
    return {
        "all_strictly_nested": all_strictly_nested,
        "subsets": results
    }


if __name__ == "__main__":
    print("Running Cohort Integrity and Leakage Audit...")
    train_p = r"D:\ASL_WORKSPACE\data\train_nested_N2400_s42.npz"
    val_p   = r"D:\ASL_WORKSPACE\data\val_N100_s99.npz"
    test_p  = r"D:\ASL_WORKSPACE\data\test_N100_s7.npz"

    train_aud = audit_cohort_dataset(train_p, "train", 42)
    val_aud   = audit_cohort_dataset(val_p, "val", 99)
    test_aud  = audit_cohort_dataset(test_p, "test", 7)

    print(f"Train Cohort: {train_aud['n_samples']} samples, {train_aud['unique_samples']} unique, {train_aud['total_valid_voxels']} voxels.")
    print(f"Val Cohort:   {val_aud['n_samples']} samples, {val_aud['unique_samples']} unique, {val_aud['total_valid_voxels']} voxels.")
    print(f"Test Cohort:  {test_aud['n_samples']} samples, {test_aud['unique_samples']} unique, {test_aud['total_valid_voxels']} voxels.")

    ind_res = check_splits_independence(train_aud, val_aud, test_aud)
    print("Independence Results:", ind_res)
    assert ind_res["is_independent"], "DATA LEAKAGE DETECTED!"
    print("Zero Data Leakage Confirmed.")

    nest_res = verify_nested_subsets(train_aud, [100, 200, 400, 600, 1200, 2400])
    print("Nested Subset Integrity:", nest_res["all_strictly_nested"])
    assert nest_res["all_strictly_nested"], "NESTED SUBSET INTEGRITY FAILED!"
    print("Strict Subset Nesting Verified.")

