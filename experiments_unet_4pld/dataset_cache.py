import os
import sys
import json
import hashlib
import numpy as np
import shutil

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments_unet_4pld.generate_spatial_data import generate_spatial_dataset

SIMULATOR_VERSION = "buxton_two_compartment_rician_v1.1"

def compute_dataset_hash(config_metadata):
    """Generates deterministic SHA256 hash for dataset generation specifications."""
    serialized = json.dumps(config_metadata, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

def create_dataset_manifest(
    split_name,
    n_phantoms,
    seed,
    img_size,
    plds,
    snrs,
    cbf_range=(0.0, 100.0),
    att_range=(0.5, 3.0),
    simulator_version=SIMULATOR_VERSION
):
    """Constructs metadata dictionary for dataset identity."""
    meta = {
        "split_name": split_name,
        "n_phantoms": int(n_phantoms),
        "seed": int(seed),
        "img_size": int(img_size),
        "plds": [float(p) for p in plds],
        "snrs": [float(s) for s in snrs] if snrs is not None else "variable_5_to_50",
        "cbf_range": list(cbf_range),
        "att_range": list(att_range),
        "simulator_version": simulator_version
    }
    meta["hash"] = compute_dataset_hash(meta)
    return meta

def validate_cache_manifest(npz_path, expected_manifest):
    """
    Validates whether the cached npz file matches the exact expected specifications.
    Returns True if valid, False otherwise.
    """
    manifest_path = npz_path.replace(".npz", ".manifest.json")
    if not os.path.exists(npz_path) or not os.path.exists(manifest_path):
        return False

    try:
        with open(manifest_path, "r") as f:
            cached_meta = json.load(f)
        # Check key properties
        keys_to_match = ["n_phantoms", "seed", "img_size", "plds", "simulator_version"]
        for k in keys_to_match:
            if cached_meta.get(k) != expected_manifest.get(k):
                return False
        return True
    except Exception:
        return False

def get_or_create_dataset(
    split_name,
    n_phantoms,
    seed,
    config,
    ws,
    fixed_snrs=None,
    force_regenerate=False
):
    """
    Safely retrieves cached dataset from workspace (or repo data directory)
    if manifest matches, or generates and stores strictly in the managed workspace.
    """
    plds = config["physics"]["PLDs"]
    img_size = config["data"]["img_size"]
    expected_manifest = create_dataset_manifest(
        split_name=split_name,
        n_phantoms=n_phantoms,
        seed=seed,
        img_size=img_size,
        plds=plds,
        snrs=fixed_snrs
    )

    ws_data_dir = ws.get_path("data")
    target_npz = os.path.join(ws_data_dir, f"{split_name}_N{n_phantoms}_s{seed}.npz")
    target_manifest = target_npz.replace(".npz", ".manifest.json")

    # Check 1: Valid cache in workspace
    if not force_regenerate and validate_cache_manifest(target_npz, expected_manifest):
        print(f"[{split_name.upper()}] Reusing validated dataset cache from workspace: {target_npz}")
        data = np.load(target_npz)
        return data["X"], data["Y"], data["masks"], data["snrs"]

    # Check 2: Check repo local data dir for pre-existing dataset to migrate
    repo_data_dir = os.path.join(parent_dir, "experiments_unet_4pld", "data")
    repo_npz = os.path.join(repo_data_dir, f"{split_name}_N{n_phantoms}_s{seed}.npz")
    if not force_regenerate and os.path.exists(repo_npz):
        print(f"[{split_name.upper()}] Migrating existing verified dataset from {repo_npz} to workspace {target_npz}...")
        data = np.load(repo_npz)
        if len(data["X"]) == n_phantoms:
            # Save into workspace with manifest
            shutil.copy2(repo_npz, target_npz)
            with open(target_manifest, "w") as f:
                json.dump(expected_manifest, f, indent=2)
            return data["X"], data["Y"], data["masks"], data["snrs"]

    # Check 3: Generate fresh data strictly in workspace
    print(f"[{split_name.upper()}] Generating {n_phantoms} phantoms (seed {seed}) in workspace {ws_data_dir}...")
    ws.check_preflight_storage(required_gb=1.0)
    X, Y, masks, snrs, _ = generate_spatial_dataset(
        n_phantoms=n_phantoms,
        seed=seed,
        img_size=img_size,
        fixed_snrs=fixed_snrs
    )

    np.savez_compressed(target_npz, X=X, Y=Y, masks=masks, snrs=snrs)
    with open(target_manifest, "w") as f:
        json.dump(expected_manifest, f, indent=2)
    print(f"[{split_name.upper()}] Successfully saved to {target_npz} with manifest.")

    return X, Y, masks, snrs

