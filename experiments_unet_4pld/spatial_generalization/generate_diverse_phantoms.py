"""
Physically-Consistent Domain-Randomized Spatial ASL Phantom Generator.
Augments the training and validation distribution with:
1. Normal smooth anatomy (60%)
2. Focal ischemic stroke infarcts (15%)
3. Ventricular enlargement / hydrocephalus geometry (15%)
4. Sharp unsmoothed step transitions (10%)
All perfusion observations are generated strictly via the authoritative Buxton forward model
and two-channel Rician noise model from data_gen.py.
"""

import os
import sys
import json
import hashlib
import numpy as np
import scipy.ndimage as ndi

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_gen import (
    compute_signals_vec,
    PLDs, tau, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX,
    SIG_REF_SC, SD_MAX
)
from experiments_unet_4pld.generate_spatial_data import create_smooth_field

def generate_diverse_single_phantom(img_size=64, rng=None, snr_target=None):
    if rng is None:
        rng = np.random.default_rng()

    H, W = img_size, img_size
    y, x = np.mgrid[:H, :W]
    cy = H / 2.0 + rng.uniform(-1.5, 1.5)
    cx = W / 2.0 + rng.uniform(-1.5, 1.5)

    # Pick phantom subtype: 0: Normal, 1: Focal Lesion, 2: Hydrocephalus, 3: Sharp Step
    p_type = rng.choice([0, 1, 2, 3], p=[0.60, 0.15, 0.15, 0.10])

    # Outer brain contour
    ry = H * rng.uniform(0.38, 0.44)
    rx = W * rng.uniform(0.32, 0.38)
    theta = np.arctan2(y - cy, x - cx)
    f1 = rng.uniform(-0.04, 0.04) * np.cos(2 * theta)
    f2 = rng.uniform(-0.03, 0.03) * np.cos(3 * theta + rng.uniform(0, np.pi))
    f3 = rng.uniform(-0.02, 0.02) * np.cos(4 * theta)
    r_factor = 1.0 + f1 + f2 + f3
    parenchyma_mask = (np.sqrt(((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2) / r_factor) <= 1.0

    # Ventricle scaling: normal vs enlarged (hydrocephalus)
    if p_type == 2: # Hydrocephalus
        vent_scale = rng.uniform(1.8, 2.6)
    else:
        vent_scale = rng.uniform(0.8, 1.2)

    r_vent_y = H * rng.uniform(0.08, 0.12) * vent_scale
    r_vent_x = W * rng.uniform(0.04, 0.07) * vent_scale
    vent_dist = np.sqrt(((y - cy) / r_vent_y) ** 2 + ((np.abs(x - cx) - 2.0) / r_vent_x) ** 2)
    ventricle_mask = (vent_dist <= 1.0) & parenchyma_mask
    brain_mask = parenchyma_mask & (~ventricle_mask)

    # Cortical ribbon
    dist_from_border = ndi.distance_transform_edt(parenchyma_mask)
    cortex_thickness = rng.uniform(2.0, 4.0)
    cortex_gm = brain_mask & (dist_from_border <= cortex_thickness)

    # Deep GM clusters
    deep_gm1 = brain_mask & (np.sqrt(((y - (cy - 3)) / 4.0)**2 + ((x - (cx - 6)) / 3.5)**2) <= 1.0)
    deep_gm2 = brain_mask & (np.sqrt(((y - (cy - 3)) / 4.0)**2 + ((x - (cx + 6)) / 3.5)**2) <= 1.0)
    deep_gm3 = brain_mask & (np.sqrt(((y - (cy + 4)) / 3.5)**2 + ((x - (cx - 5)) / 3.0)**2) <= 1.0)
    deep_gm4 = brain_mask & (np.sqrt(((y - (cy + 4)) / 3.5)**2 + ((x - (cx + 5)) / 3.0)**2) <= 1.0)
    gm_mask = cortex_gm | deep_gm1 | deep_gm2 | deep_gm3 | deep_gm4
    wm_mask = brain_mask & (~gm_mask)

    # Physiological baselines
    base_gm_cbf = rng.uniform(55.0, 72.0)
    base_wm_cbf = rng.uniform(18.0, 28.0)
    base_gm_att = rng.uniform(0.9, 1.4)
    base_wm_att = rng.uniform(1.5, 2.2)

    cbf_field = create_smooth_field((H, W), rng, scale=5.0)
    att_field = create_smooth_field((H, W), rng, scale=6.0)
    y_norm = (y - cy) / (H / 2.0)
    att_gradient = y_norm * rng.uniform(0.1, 0.4)

    cbf_map = np.zeros((H, W), dtype=np.float64)
    att_map = np.zeros((H, W), dtype=np.float64)

    cbf_map[gm_mask] = base_gm_cbf + cbf_field[gm_mask] * rng.uniform(5.0, 10.0)
    att_map[gm_mask] = base_gm_att + att_field[gm_mask] * rng.uniform(0.1, 0.2) + att_gradient[gm_mask]

    cbf_map[wm_mask] = base_wm_cbf + cbf_field[wm_mask] * rng.uniform(3.0, 6.0)
    att_map[wm_mask] = base_wm_att + att_field[wm_mask] * rng.uniform(0.1, 0.2) + att_gradient[wm_mask]

    # Subtype-specific pathology augmentation
    if p_type == 1: # Focal stroke infarct
        # Random lesion center within brain
        valid_coords = np.argwhere(gm_mask)
        if len(valid_coords) > 0:
            c_pt = valid_coords[rng.choice(len(valid_coords))]
            l_radius = rng.uniform(4.0, 9.0)
            lesion = ((x - c_pt[1])**2 + (y - c_pt[0])**2) < (l_radius ** 2)
            cbf_map[lesion & brain_mask] = rng.uniform(3.0, 15.0) # Severe ischemia
            att_map[lesion & brain_mask] = rng.uniform(2.2, 2.9) # Collateral delay

    elif p_type == 3: # Sharp step / high-frequency pattern
        step = ((x // rng.integers(6, 12)) % 2).astype(np.float64)
        cbf_map[gm_mask] += (step[gm_mask] - 0.5) * rng.uniform(15.0, 30.0)
        att_map[gm_mask] += (step[gm_mask] - 0.5) * rng.uniform(0.3, 0.6)

    # Boundary smoothing (variable sigma: 0.0 for sharp steps, 0.8 for smooth)
    if p_type == 3:
        smooth_sigma = 0.0 # Unfiltered step
    elif p_type == 1:
        smooth_sigma = rng.uniform(0.0, 0.4) # Sharp lesion border
    else:
        smooth_sigma = rng.uniform(0.6, 1.0) # Standard partial volume

    if smooth_sigma > 0.0:
        cbf_s = ndi.gaussian_filter(cbf_map, sigma=smooth_sigma)
        att_s = ndi.gaussian_filter(att_map, sigma=smooth_sigma)
        cbf_map[brain_mask] = cbf_s[brain_mask]
        att_map[brain_mask] = att_s[brain_mask]

    cbf_map[~brain_mask] = 0.0
    att_map[~brain_mask] = 0.0
    cbf_map = np.clip(cbf_map, CBF_MIN, CBF_MAX)
    att_map = np.clip(att_map, ATT_MIN, ATT_MAX)

    # ASL Perfusion Signal Observation via Buxton Model
    sig = compute_signals_vec(cbf_map.flatten(), att_map.flatten(), ld=tau) * SCALE
    sig = sig.reshape(H, W, 4).transpose(2, 0, 1) # [4, H, W]

    # Two-Channel Rician Noise
    if snr_target is not None:
        snr = float(snr_target)
    else:
        snr = rng.uniform(5.0, 30.0)

    sd = SIG_REF_SC / snr
    e1 = rng.normal(0.0, sd, size=sig.shape)
    e2 = rng.normal(0.0, sd, size=sig.shape)
    e3 = rng.normal(0.0, sd, size=sig.shape)
    e4 = rng.normal(0.0, sd, size=sig.shape)

    mc = np.sqrt((sig + e1)**2 + e2**2)
    ml = np.sqrt((sig + e3)**2 + e4**2)
    X = (mc + ml) * brain_mask[None, ...]

    Y = np.stack([cbf_map, att_map], axis=0)
    mask = brain_mask[None, ...].astype(np.float32)

    return X.astype(np.float32), Y.astype(np.float32), mask, float(snr)

def create_diverse_dataset(n_phantoms, seed, out_path, manifest_path):
    rng = np.random.default_rng(seed)
    X_list, Y_list, mask_list, snr_list = [], [], [], []
    for _ in range(n_phantoms):
        x_p, y_p, m_p, s_p = generate_diverse_single_phantom(img_size=64, rng=rng)
        X_list.append(x_p)
        Y_list.append(y_p)
        mask_list.append(m_p)
        snr_list.append(s_p)

    X_arr = np.stack(X_list, axis=0)
    Y_arr = np.stack(Y_list, axis=0)
    mask_arr = np.stack(mask_list, axis=0)
    snr_arr = np.array(snr_list, dtype=np.float32)

    np.savez_compressed(out_path, X=X_arr, Y=Y_arr, masks=mask_arr, snrs=snr_arr)
    manifest = {
        "dataset_name": os.path.basename(out_path),
        "n_phantoms": n_phantoms,
        "seed": seed,
        "composition": "60% normal, 15% focal stroke, 15% hydrocephalus, 10% sharp step",
        "sha256": hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated {out_path} ({n_phantoms} phantoms, SHA-256: {manifest['sha256'][:16]}...)")
    return X_arr, Y_arr, mask_arr, snr_arr

if __name__ == "__main__":
    from experiments_unet_4pld.workspace import get_workspace
    ws = get_workspace()
    tr_out = ws.get_path("data", "train_diverse_N600_s42.npz")
    tr_man = ws.get_path("data", "train_diverse_N600_s42.manifest.json")
    val_out = ws.get_path("data", "val_diverse_N100_s99.npz")
    val_man = ws.get_path("data", "val_diverse_N100_s99.manifest.json")

    if not os.path.exists(tr_out):
        create_diverse_dataset(600, 42, tr_out, tr_man)
    if not os.path.exists(val_out):
        create_diverse_dataset(100, 99, val_out, val_man)

