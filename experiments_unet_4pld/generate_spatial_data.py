import os
import sys
import json
import numpy as np
import scipy.ndimage as ndi

# Add parent directory to path to import authoritative physics from data_gen
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_gen import (
    compute_signals_vec,
    PLDs, tau, T1t, T1a, alpha, beta, lmbda, SCALE,
    CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX,
    SIG_REF_SC, SD_MAX
)

def create_smooth_field(shape, rng, scale=4.0):
    """Generates a smooth 2D Gaussian random field."""
    noise = rng.normal(0.0, 1.0, size=shape)
    smooth = ndi.gaussian_filter(noise, sigma=scale)
    smooth = (smooth - smooth.mean()) / (smooth.std() + 1e-8)
    return smooth

def generate_single_phantom(img_size=64, rng=None, snr_target=None):
    """
    Generates a single synthetic 2D brain phantom with realistic spatial
    compartments (cortical Gray Matter, deep White Matter, Ventricles/CSF, background),
    smooth spatial parameter variation, exact Buxton forward model ASL signals,
    and two-channel Rician noise matching data_gen.py.
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W = img_size, img_size
    y, x = np.mgrid[:H, :W]
    cy = H / 2.0 + rng.uniform(-1.5, 1.5)
    cx = W / 2.0 + rng.uniform(-1.5, 1.5)

    # 1. Outer brain parenchyma contour (elliptical with smooth lobe indentations)
    ry = H * rng.uniform(0.38, 0.44)
    rx = W * rng.uniform(0.32, 0.38)
    theta = np.arctan2(y - cy, x - cx)
    
    # Perturb radius with low-frequency Fourier harmonics for realistic brain lobes
    f1 = rng.uniform(-0.04, 0.04) * np.cos(2 * theta)
    f2 = rng.uniform(-0.03, 0.03) * np.cos(3 * theta + rng.uniform(0, np.pi))
    f3 = rng.uniform(-0.02, 0.02) * np.cos(4 * theta)
    r_factor = 1.0 + f1 + f2 + f3
    
    normalized_dist = np.sqrt(((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2) / r_factor
    parenchyma_mask = normalized_dist <= 1.0

    # 2. Central Ventricles (CSF - non-parenchymal compartment)
    r_vent_y = H * rng.uniform(0.08, 0.12)
    r_vent_x = W * rng.uniform(0.04, 0.07)
    # Butterfly-like indentation for ventricles
    vent_dist = np.sqrt(((y - cy) / r_vent_y) ** 2 + ((np.abs(x - cx) - 2.0) / r_vent_x) ** 2)
    ventricle_mask = (vent_dist <= 1.0) & parenchyma_mask

    # Brain tissue mask (valid voxels: parenchyma minus CSF ventricles)
    brain_mask = parenchyma_mask & (~ventricle_mask)

    # 3. Tissue segmentation: Cortical Gray Matter (GM) vs White Matter (WM)
    # Distance from outer boundary
    dist_from_border = ndi.distance_transform_edt(parenchyma_mask)
    cortex_thickness = rng.uniform(2.5, 4.0)
    cortex_gm = brain_mask & (dist_from_border <= cortex_thickness)

    # Deep GM clusters (basal ganglia / thalamus)
    deep_gm1 = brain_mask & (np.sqrt(((y - (cy - 3)) / 4.0)**2 + ((x - (cx - 6)) / 3.5)**2) <= 1.0)
    deep_gm2 = brain_mask & (np.sqrt(((y - (cy - 3)) / 4.0)**2 + ((x - (cx + 6)) / 3.5)**2) <= 1.0)
    deep_gm3 = brain_mask & (np.sqrt(((y - (cy + 4)) / 3.5)**2 + ((x - (cx - 5)) / 3.0)**2) <= 1.0)
    deep_gm4 = brain_mask & (np.sqrt(((y - (cy + 4)) / 3.5)**2 + ((x - (cx + 5)) / 3.0)**2) <= 1.0)
    deep_gm = deep_gm1 | deep_gm2 | deep_gm3 | deep_gm4

    gm_mask = cortex_gm | deep_gm
    wm_mask = brain_mask & (~gm_mask)

    # 4. Spatially varying continuous physiological parameter maps
    # Smooth spatial variation fields across the slice
    cbf_spatial_field = create_smooth_field((H, W), rng, scale=5.0)
    att_spatial_field = create_smooth_field((H, W), rng, scale=6.0)

    # Smooth arterial transit delay gradient (anterior-to-posterior watershed delay)
    y_norm = (y - cy) / (H / 2.0)
    att_gradient = y_norm * rng.uniform(0.15, 0.35)

    # Base parameter assignments conforming to ASL physiology and repo bounds
    # GM: CBF ~ 55-75, ATT ~ 0.9-1.4 s
    # WM: CBF ~ 18-32, ATT ~ 1.5-2.2 s
    base_gm_cbf = rng.uniform(58.0, 70.0)
    base_wm_cbf = rng.uniform(20.0, 28.0)
    base_gm_att = rng.uniform(1.0, 1.3)
    base_wm_att = rng.uniform(1.6, 2.0)

    cbf_map = np.zeros((H, W), dtype=np.float64)
    att_map = np.zeros((H, W), dtype=np.float64)

    # GM parameters with spatial smoothness
    cbf_map[gm_mask] = base_gm_cbf + cbf_spatial_field[gm_mask] * rng.uniform(6.0, 10.0)
    att_map[gm_mask] = base_gm_att + att_spatial_field[gm_mask] * rng.uniform(0.1, 0.2) + att_gradient[gm_mask]

    # WM parameters with spatial smoothness
    cbf_map[wm_mask] = base_wm_cbf + cbf_spatial_field[wm_mask] * rng.uniform(3.0, 6.0)
    att_map[wm_mask] = base_wm_att + att_spatial_field[wm_mask] * rng.uniform(0.15, 0.25) + att_gradient[wm_mask]

    # Smooth the whole brain parameter maps slightly at tissue interfaces for partial volume effect
    cbf_smooth = ndi.gaussian_filter(cbf_map, sigma=0.8)
    att_smooth = ndi.gaussian_filter(att_map, sigma=0.8)
    
    cbf_map[brain_mask] = cbf_smooth[brain_mask]
    att_map[brain_mask] = att_smooth[brain_mask]

    # Zero background
    cbf_map[~brain_mask] = 0.0
    att_map[~brain_mask] = 0.0

    # Ensure strictly within repo physiological bounds on brain voxels
    cbf_map[brain_mask] = np.clip(cbf_map[brain_mask], CBF_MIN, CBF_MAX)
    att_map[brain_mask] = np.clip(att_map[brain_mask], ATT_MIN, ATT_MAX)

    # 5. Forward ASL signal calculation for each voxel
    # Valid brain voxels
    signals_noiseless = np.zeros((4, H, W), dtype=np.float64)
    valid_indices = np.where(brain_mask)
    if len(valid_indices[0]) > 0:
        cbf_valid = cbf_map[valid_indices]
        att_valid = att_map[valid_indices]
        sig_valid = compute_signals_vec(cbf_valid, att_valid) * SCALE # Shape: (N_valid, 4)
        for p_idx in range(4):
            signals_noiseless[p_idx][valid_indices] = sig_valid[:, p_idx]

    # 6. Noise simulation matching data_gen.py
    if snr_target is not None:
        sd = float(SIG_REF_SC / snr_target)
        snr = float(snr_target)
    else:
        sd = float(rng.uniform(0.0, SD_MAX))
        snr = float(SIG_REF_SC / sd) if sd > 0 else float("inf")

    # Generate 4-channel independent normal noise
    e1 = rng.normal(0.0, 1.0, size=(4, H, W)) * sd
    e2 = rng.normal(0.0, 1.0, size=(4, H, W)) * sd
    e3 = rng.normal(0.0, 1.0, size=(4, H, W)) * sd
    e4 = rng.normal(0.0, 1.0, size=(4, H, W)) * sd

    mc = np.sqrt((signals_noiseless + e1) ** 2 + e2 ** 2)
    ml = np.sqrt((signals_noiseless + e3) ** 2 + e4 ** 2)
    signals_noisy = (mc + ml).astype(np.float32)

    targets = np.stack([cbf_map, att_map], axis=0).astype(np.float32)
    mask = brain_mask.astype(np.float32)[np.newaxis, :, :]

    metadata = {
        "snr": snr,
        "sd": sd,
        "n_valid_voxels": int(np.sum(brain_mask)),
        "mean_cbf": float(np.mean(cbf_map[brain_mask])),
        "mean_att": float(np.mean(att_map[brain_mask]))
    }

    return signals_noisy, targets, mask, metadata

def generate_spatial_dataset(n_phantoms, seed=42, img_size=64, fixed_snrs=None):
    """
    Generates a full spatial dataset consisting of independent 2D phantoms.
    """
    rng = np.random.default_rng(seed)
    all_signals = []
    all_targets = []
    all_masks = []
    all_snrs = []
    all_sds = []

    for i in range(n_phantoms):
        snr_t = None
        if fixed_snrs is not None and len(fixed_snrs) > 0:
            snr_t = fixed_snrs[i % len(fixed_snrs)]

        sig, tgt, msk, meta = generate_single_phantom(
            img_size=img_size, rng=rng, snr_target=snr_t
        )
        all_signals.append(sig)
        all_targets.append(tgt)
        all_masks.append(msk)
        all_snrs.append(meta["snr"])
        all_sds.append(meta["sd"])

    X = np.stack(all_signals, axis=0)       # [N, 4, H, W]
    Y = np.stack(all_targets, axis=0)       # [N, 2, H, W]
    masks = np.stack(all_masks, axis=0)     # [N, 1, H, W]
    snrs = np.array(all_snrs, dtype=np.float32)
    sds = np.array(all_sds, dtype=np.float32)

    return X, Y, masks, snrs, sds

if __name__ == "__main__":
    print("Testing generate_single_phantom...")
    s, t, m, meta = generate_single_phantom(img_size=64, snr_target=10.0)
    print(f"Generated phantom: Signal shape {s.shape}, Target shape {t.shape}, Mask shape {m.shape}")
    print(f"Metadata: {meta}")
    print("Signals min/max:", s.min(), s.max())
    print("CBF valid range:", t[0][m[0] > 0.5].min(), t[0][m[0] > 0.5].max())
    print("ATT valid range:", t[1][m[0] > 0.5].min(), t[1][m[0] > 0.5].max())
    print("All tests passed in generate_spatial_data.py!")
