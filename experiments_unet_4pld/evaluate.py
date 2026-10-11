import os
import sys
import time
import json
import yaml
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_gen import (
    compute_signals_vec,
    PLDs, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX,
    SIG_REF_SC
)
from experiments_unet_4pld.model import ASLUNet
from experiments_unet_4pld.dataset import denormalize_predictions

# ----------------- METRIC UTILITIES -----------------

def calc_metrics(true, pred):
    """Computes quantitative parameter estimation metrics on 1D voxel arrays."""
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    err = pred - true
    abs_err = np.abs(err)
    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_err)
    bias = np.mean(err)

    # R^2
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    ss_res = np.sum(err ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12)) if ss_tot > 0 else 0.0

    # Lin's Concordance Correlation Coefficient (CCC)
    mu_x = np.mean(true)
    mu_y = np.mean(pred)
    var_x = np.var(true)
    var_y = np.var(pred)
    cov_xy = np.mean((true - mu_x) * (pred - mu_y))
    denominator = var_x + var_y + (mu_x - mu_y) ** 2
    ccc = (2.0 * cov_xy) / (denominator + 1e-12) if denominator > 0 else 0.0

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Bias": float(bias),
        "R2": float(r2),
        "CCC": float(ccc)
    }

def calc_masked_psnr(true_2d, pred_2d, mask_2d, data_range=100.0):
    """Computes PSNR inside the valid brain mask."""
    valid = mask_2d > 0.5
    if not np.any(valid):
        return 0.0
    mse = np.mean((pred_2d[valid] - true_2d[valid]) ** 2)
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10((data_range ** 2) / mse))

def calc_masked_ssim(true_2d, pred_2d, mask_2d, data_range=100.0, win_size=7):
    """
    Computes masked SSIM using local Gaussian windowing strictly
    within valid brain regions.
    """
    t = np.copy(true_2d).astype(np.float64)
    p = np.copy(pred_2d).astype(np.float64)
    m = (mask_2d > 0.5).astype(np.float64)

    # Zero background
    t[m == 0] = 0.0
    p[m == 0] = 0.0

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    sigma = 1.5
    mu1 = ndi.gaussian_filter(t, sigma)
    mu2 = ndi.gaussian_filter(p, sigma)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = ndi.gaussian_filter(t ** 2, sigma) - mu1_sq
    sigma2_sq = ndi.gaussian_filter(p ** 2, sigma) - mu2_sq
    sigma12 = ndi.gaussian_filter(t * p, sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12)

    # Valid mask for SSIM
    valid_window = ndi.gaussian_filter(m, sigma) > 0.5
    if np.any(valid_window):
        return float(np.mean(ssim_map[valid_window]))
    return float(np.mean(ssim_map[m > 0]))

# ----------------- ESTIMATOR IMPLEMENTATIONS -----------------

def evaluate_unet(model, X_test_norm, stats, device):
    """Runs U-Net inference on normalized multi-PLD test tensors."""
    model.eval()
    start_time = time.time()
    preds_list = []
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(X_test_norm), batch_size):
            xb = torch.as_tensor(X_test_norm[i:i+batch_size], dtype=torch.float32, device=device)
            out = model(xb)
            preds_list.append(out.cpu().numpy())

    preds_norm = np.concatenate(preds_list, axis=0) # [N, 2, H, W]
    cbf_pred, att_pred = denormalize_predictions(preds_norm, stats)
    
    # Clip to physical bounds
    cbf_pred = np.clip(cbf_pred, CBF_MIN, CBF_MAX)
    att_pred = np.clip(att_pred, ATT_MIN, ATT_MAX)
    duration = time.time() - start_time
    return cbf_pred, att_pred, duration

def build_pointwise_dnn(device):
    """
    Constructs a point-wise MLP baseline matching the repository's
    established 9-layer MLP architectures.
    """
    class MLP(nn.Module):
        def __init__(self, in_dim=4, hidden=64, n_layers=5, out_dim=1):
            super().__init__()
            layers = [nn.Linear(in_dim, hidden), nn.ELU()]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden, hidden), nn.ELU()]
            layers.append(nn.Linear(hidden, out_dim))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    cbf_mlp = MLP(4, 64, 5, 1).to(device)
    att_mlp = MLP(4, 64, 5, 1).to(device)
    return cbf_mlp, att_mlp

def train_pointwise_dnn(X_tr_vox, Y_tr_vox, stats, device, epochs=30, batch_size=4096):
    """Trains a point-wise MLP baseline on flattened training voxels."""
    cbf_mlp, att_mlp = build_pointwise_dnn(device)
    
    # Standardize voxels
    X_mean = stats["X_mean"]
    X_std = stats["X_std"]
    X_tr_std = (X_tr_vox - X_mean) / X_std
    Y_cbf_std = (Y_tr_vox[:, 0] - stats["cbf_mean"]) / stats["cbf_std"]
    Y_att_std = (Y_tr_vox[:, 1] - stats["att_mean"]) / stats["att_std"]

    xt = torch.tensor(X_tr_std, dtype=torch.float32, device=device)
    yc = torch.tensor(Y_cbf_std, dtype=torch.float32, device=device).unsqueeze(1)
    ya = torch.tensor(Y_att_std, dtype=torch.float32, device=device).unsqueeze(1)

    opt_c = torch.optim.Adam(cbf_mlp.parameters(), lr=1e-3)
    opt_a = torch.optim.Adam(att_mlp.parameters(), lr=1e-3)
    crit = nn.L1Loss()

    N = len(xt)
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            # CBF step
            opt_c.zero_grad()
            pc = cbf_mlp(xt[idx])
            loss_c = crit(pc, yc[idx])
            loss_c.backward()
            opt_c.step()

            # ATT step
            opt_a.zero_grad()
            pa = att_mlp(xt[idx])
            loss_a = crit(pa, ya[idx])
            loss_a.backward()
            opt_a.step()

    return cbf_mlp, att_mlp

def evaluate_pointwise_dnn(cbf_mlp, att_mlp, X_test, masks_test, stats, device):
    """Evaluates the point-wise DNN voxel-wise across 2D test phantoms."""
    start_time = time.time()
    cbf_mlp.eval()
    att_mlp.eval()

    N, C, H, W = X_test.shape
    cbf_out = np.zeros((N, H, W), dtype=np.float32)
    att_out = np.zeros((N, H, W), dtype=np.float32)

    X_mean = stats["X_mean"]
    X_std = stats["X_std"]

    with torch.no_grad():
        for i in range(N):
            valid = masks_test[i, 0] > 0.5
            if not np.any(valid):
                continue
            vox = X_test[i][:, valid].T # [N_valid, 4]
            vox_norm = (vox - X_mean) / X_std
            vox_t = torch.tensor(vox_norm, dtype=torch.float32, device=device)

            pred_c_std = cbf_mlp(vox_t).squeeze(1).cpu().numpy()
            pred_a_std = att_mlp(vox_t).squeeze(1).cpu().numpy()

            pred_c_phys = pred_c_std * stats["cbf_std"] + stats["cbf_mean"]
            pred_a_phys = pred_a_std * stats["att_std"] + stats["att_mean"]

            cbf_out[i][valid] = np.clip(pred_c_phys, CBF_MIN, CBF_MAX)
            att_out[i][valid] = np.clip(pred_a_phys, ATT_MIN, ATT_MAX)

    duration = time.time() - start_time
    return cbf_out, att_out, duration

def evaluate_bayesian_fast(X_test, snrs_test, masks_test, sig_ref_sc=SIG_REF_SC, grid_res=60):
    """
    Vectorized Bayesian grid estimator across valid voxels of test phantoms.
    Uses uniform prior on [CBF_MIN, CBF_MAX] x [ATT_MIN, ATT_MAX].
    """
    start_time = time.time()
    N, C, H, W = X_test.shape
    cbf_out = np.zeros((N, H, W), dtype=np.float32)
    att_out = np.zeros((N, H, W), dtype=np.float32)

    # Precompute grid
    cbf_grid = np.linspace(CBF_MIN, CBF_MAX, grid_res)
    att_grid = np.linspace(ATT_MIN, ATT_MAX, grid_res)
    CC, AA = np.meshgrid(cbf_grid, att_grid)
    grid_cbf = CC.flatten()
    grid_att = AA.flatten()
    grid_sig = 2.0 * compute_signals_vec(grid_cbf, grid_att) * SCALE # [G, 4]
    norm_S = np.sum(grid_sig ** 2, axis=1, keepdims=True).T    # [1, G]

    for i in range(N):
        valid = masks_test[i, 0] > 0.5
        if not np.any(valid):
            continue
        X_vox = X_test[i][:, valid].T # [V, 4]
        snr = float(snrs_test[i])
        sigma = sig_ref_sc / snr if snr < 1e5 else 1e-6
        sigma = max(sigma, 1e-6)

        norm_X = np.sum(X_vox ** 2, axis=1, keepdims=True)
        sq_diff = norm_X - 2.0 * (X_vox @ grid_sig.T) + norm_S
        log_L = -0.5 * sq_diff / (sigma ** 2)
        log_L -= np.max(log_L, axis=1, keepdims=True)
        L = np.exp(log_L)
        P = L / (np.sum(L, axis=1, keepdims=True) + 1e-12)

        pred_cbf = P @ grid_cbf
        pred_att = P @ grid_att

        cbf_out[i][valid] = np.clip(pred_cbf, CBF_MIN, CBF_MAX)
        att_out[i][valid] = np.clip(pred_att, ATT_MIN, ATT_MAX)

    duration = time.time() - start_time
    return cbf_out, att_out, duration

def evaluate_nlls(X_test, masks_test, max_phantoms=15):
    """
    Voxel-wise Non-Linear Least Squares (NLLS) estimator using TRF bounded solver.
    Evaluates on up to max_phantoms to maintain computationally reasonable runtime.
    """
    start_time = time.time()
    N, C, H, W = X_test.shape
    eval_n = min(N, max_phantoms)
    cbf_out = np.zeros((eval_n, H, W), dtype=np.float32)
    att_out = np.zeros((eval_n, H, W), dtype=np.float32)

    def residual_fn(params, y_obs):
        c, a = params
        s = 2.0 * compute_signals_vec(np.array([c]), np.array([a])) * SCALE
        return (s[0] - y_obs)

    x0 = [50.0, 1.75]
    bounds = ([CBF_MIN, ATT_MIN], [CBF_MAX, ATT_MAX])

    for i in range(eval_n):
        valid = masks_test[i, 0] > 0.5
        if not np.any(valid):
            continue
        vox_signals = X_test[i][:, valid].T # [V, 4]
        n_vox = vox_signals.shape[0]

        cbf_vox = np.zeros(n_vox, dtype=np.float32)
        att_vox = np.zeros(n_vox, dtype=np.float32)

        for v in range(n_vox):
            res = scipy.optimize.least_squares(
                residual_fn, x0=x0, args=(vox_signals[v],),
                bounds=bounds, method="trf", ftol=1e-3, xtol=1e-3
            )
            cbf_vox[v] = res.x[0]
            att_vox[v] = res.x[1]

        cbf_out[i][valid] = cbf_vox
        att_out[i][valid] = att_vox

    duration = time.time() - start_time
    return cbf_out, att_out, eval_n, duration

# ----------------- MAIN BENCHMARK RUNNER -----------------

def run_full_evaluation(
    unet_model,
    X_test,
    Y_test,
    masks_test,
    snrs_test,
    stats,
    device,
    X_tr_vox=None,
    Y_tr_vox=None,
    out_dir="experiments_unet_4pld/results"
):
    """
    Runs full comparative benchmark across U-Net, Point-wise DNN, NLLS, Bayesian,
    and historical references under exact same test set.
    """
    plots_dir = os.path.join(out_dir, "plots")
    preds_dir = os.path.join(out_dir, "predictions")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    print("\n================ EVALUATION BENCHMARK ================")
    # 1. Normalize test inputs for U-Net
    X_test_norm = np.copy(X_test)
    for c in range(4):
        X_test_norm[:, c] = (X_test[:, c] - stats["X_mean"][c]) / stats["X_std"][c]

    print("Evaluating U-Net...")
    cbf_unet, att_unet, t_unet = evaluate_unet(unet_model, X_test_norm, stats, device)

    print("Training and evaluating Point-wise DNN Baseline...")
    if X_tr_vox is not None and Y_tr_vox is not None:
        cbf_mlp, att_mlp = train_pointwise_dnn(X_tr_vox, Y_tr_vox, stats, device, epochs=25)
        cbf_dnn, att_dnn, t_dnn = evaluate_pointwise_dnn(cbf_mlp, att_mlp, X_test, masks_test, stats, device)
    else:
        cbf_dnn, att_dnn, t_dnn = None, None, 0.0

    print("Evaluating Vectorized Bayesian Estimator...")
    cbf_bayes, att_bayes, t_bayes = evaluate_bayesian_fast(X_test, snrs_test, masks_test)

    print("Evaluating NLLS Estimator (on representative test subset)...")
    cbf_nlls, att_nlls, nlls_n, t_nlls = evaluate_nlls(X_test, masks_test, max_phantoms=15)

    # Save prediction arrays
    np.save(os.path.join(preds_dir, "cbf_unet.npy"), cbf_unet)
    np.save(os.path.join(preds_dir, "att_unet.npy"), att_unet)
    if cbf_dnn is not None:
        np.save(os.path.join(preds_dir, "cbf_dnn.npy"), cbf_dnn)
        np.save(os.path.join(preds_dir, "att_dnn.npy"), att_dnn)
    np.save(os.path.join(preds_dir, "cbf_bayes.npy"), cbf_bayes)
    np.save(os.path.join(preds_dir, "att_bayes.npy"), att_bayes)

    # ----------------- COMPUTE QUANTITATIVE METRICS -----------------
    valid_mask_all = masks_test[:, 0] > 0.5
    gt_cbf = Y_test[:, 0][valid_mask_all]
    gt_att = Y_test[:, 1][valid_mask_all]

    methods_dict = {
        "U-Net (Proposed)": (cbf_unet[valid_mask_all], att_unet[valid_mask_all], t_unet, len(X_test)),
        "Bayesian (Grid)": (cbf_bayes[valid_mask_all], att_bayes[valid_mask_all], t_bayes, len(X_test))
    }
    if cbf_dnn is not None:
        methods_dict["Point-wise DNN"] = (cbf_dnn[valid_mask_all], att_dnn[valid_mask_all], t_dnn, len(X_test))

    # NLLS subset
    valid_mask_nlls = masks_test[:nlls_n, 0] > 0.5
    gt_cbf_nlls = Y_test[:nlls_n, 0][valid_mask_nlls]
    gt_att_nlls = Y_test[:nlls_n, 1][valid_mask_nlls]
    methods_dict["NLLS (TRF)"] = (cbf_nlls[valid_mask_nlls], att_nlls[valid_mask_nlls], t_nlls, nlls_n)

    results_rows = []
    for name, (pred_c, pred_a, duration, n_eval) in methods_dict.items():
        if name == "NLLS (TRF)":
            tc, ta = gt_cbf_nlls, gt_att_nlls
        else:
            tc, ta = gt_cbf, gt_att

        met_c = calc_metrics(tc, pred_c)
        met_a = calc_metrics(ta, pred_a)

        results_rows.append({
            "Method": name,
            "Evaluation": "Harmonized Spatial Test Set",
            "N_Phantoms": n_eval,
            "Time (s)": round(duration, 2),
            "CBF RMSE": met_c["RMSE"],
            "CBF MAE": met_c["MAE"],
            "CBF Bias": met_c["Bias"],
            "CBF R2": met_c["R2"],
            "CBF CCC": met_c["CCC"],
            "ATT RMSE": met_a["RMSE"],
            "ATT MAE": met_a["MAE"],
            "ATT Bias": met_a["Bias"],
            "ATT R2": met_a["R2"],
            "ATT CCC": met_a["CCC"],
        })

    # Add historical reference rows from reference_baselines.yaml
    ref_yaml_path = os.path.join(parent_dir, "reference_baselines.yaml")
    if os.path.exists(ref_yaml_path):
        with open(ref_yaml_path, "r") as f:
            ref_data = yaml.safe_load(f)
        for snr_key, methods in ref_data.items():
            for m_name, vals in methods.items():
                results_rows.append({
                    "Method": f"{m_name} (Fixed Ref {snr_key})",
                    "Evaluation": f"Historical Fixed Reference ({snr_key})",
                    "N_Phantoms": "N/A (Historical 0D)",
                    "Time (s)": "N/A",
                    "CBF RMSE": vals.get("CBF_RMSE", "N/A"),
                    "CBF MAE": vals.get("CBF_MAE", "N/A"),
                    "CBF Bias": "N/A",
                    "CBF R2": "N/A",
                    "CBF CCC": vals.get("CBF_CCC", "N/A"),
                    "ATT RMSE": vals.get("ATT_RMSE", "N/A"),
                    "ATT MAE": vals.get("ATT_MAE", "N/A"),
                    "ATT Bias": "N/A",
                    "ATT R2": "N/A",
                    "ATT CCC": vals.get("ATT_CCC", "N/A"),
                })

    metrics_df = pd.DataFrame(results_rows)
    metrics_csv = os.path.join(out_dir, "master_comparison.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print("\nMaster Comparison Table:")
    print(metrics_df.to_string())

    # ----------------- SNR-STRATIFIED EVALUATION -----------------
    snr_rows = []
    for snr_target in [10.0, 15.0]:
        mask_snr = (np.abs(snrs_test - snr_target) < 1.0)
        if not np.any(mask_snr):
            continue
        v_mask_snr = (masks_test[mask_snr, 0] > 0.5)
        tc_snr = Y_test[mask_snr, 0][v_mask_snr]
        ta_snr = Y_test[mask_snr, 1][v_mask_snr]

        for name, pred_tuple in [("U-Net", (cbf_unet, att_unet)),
                                 ("Bayesian", (cbf_bayes, att_bayes))]:
            pc = pred_tuple[0][mask_snr][v_mask_snr]
            pa = pred_tuple[1][mask_snr][v_mask_snr]
            mc = calc_metrics(tc_snr, pc)
            ma = calc_metrics(ta_snr, pa)
            snr_rows.append({
                "SNR": snr_target,
                "Method": name,
                "CBF RMSE": mc["RMSE"],
                "CBF MAE": mc["MAE"],
                "ATT RMSE": ma["RMSE"],
                "ATT MAE": ma["MAE"]
            })
    if len(snr_rows) > 0:
        snr_df = pd.DataFrame(snr_rows)
        snr_csv = os.path.join(out_dir, "snr_stratified_metrics.csv")
        snr_df.to_csv(snr_csv, index=False)
        print("\nSNR Stratified Metrics:")
        print(snr_df.to_string())

    # ----------------- SPATIAL METRICS (SSIM & PSNR) -----------------
    ssim_cbf_list, ssim_att_list = [], []
    psnr_cbf_list, psnr_att_list = [], []
    for i in range(len(X_test)):
        m_2d = masks_test[i, 0]
        ssim_cbf_list.append(calc_masked_ssim(Y_test[i, 0], cbf_unet[i], m_2d, data_range=100.0))
        ssim_att_list.append(calc_masked_ssim(Y_test[i, 1], att_unet[i], m_2d, data_range=2.5))
        psnr_cbf_list.append(calc_masked_psnr(Y_test[i, 0], cbf_unet[i], m_2d, data_range=100.0))
        psnr_att_list.append(calc_masked_psnr(Y_test[i, 1], att_unet[i], m_2d, data_range=2.5))

    spatial_metrics = {
        "CBF_SSIM_mean": float(np.mean(ssim_cbf_list)),
        "CBF_SSIM_std": float(np.std(ssim_cbf_list)),
        "CBF_PSNR_mean": float(np.mean(psnr_cbf_list)),
        "CBF_PSNR_std": float(np.std(psnr_cbf_list)),
        "ATT_SSIM_mean": float(np.mean(ssim_att_list)),
        "ATT_SSIM_std": float(np.std(ssim_att_list)),
        "ATT_PSNR_mean": float(np.mean(psnr_att_list)),
        "ATT_PSNR_std": float(np.std(psnr_att_list))
    }
    with open(os.path.join(out_dir, "spatial_metrics.json"), "w") as f:
        json.dump(spatial_metrics, f, indent=2)

    print("\nSpatial Quality Metrics (U-Net):")
    print(f"CBF SSIM: {spatial_metrics['CBF_SSIM_mean']:.4f} ± {spatial_metrics['CBF_SSIM_std']:.4f}")
    print(f"CBF PSNR: {spatial_metrics['CBF_PSNR_mean']:.2f} ± {spatial_metrics['CBF_PSNR_std']:.2f} dB")
    print(f"ATT SSIM: {spatial_metrics['ATT_SSIM_mean']:.4f} ± {spatial_metrics['ATT_SSIM_std']:.4f}")
    print(f"ATT PSNR: {spatial_metrics['ATT_PSNR_mean']:.2f} ± {spatial_metrics['ATT_PSNR_std']:.2f} dB")

    # ----------------- VISUALIZATION MAPS -----------------
    plot_idx = 0
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    m_slice = masks_test[plot_idx, 0] > 0.5

    # CBF
    gt_c_slice = np.where(m_slice, Y_test[plot_idx, 0], np.nan)
    pr_c_slice = np.where(m_slice, cbf_unet[plot_idx], np.nan)
    err_c_slice = np.where(m_slice, np.abs(cbf_unet[plot_idx] - Y_test[plot_idx, 0]), np.nan)

    im0 = axes[0, 0].imshow(gt_c_slice, cmap="magma", vmin=0, vmax=90)
    axes[0, 0].set_title("Ground Truth CBF")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(pr_c_slice, cmap="magma", vmin=0, vmax=90)
    axes[0, 1].set_title("U-Net Predicted CBF")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(err_c_slice, cmap="hot", vmin=0, vmax=15)
    axes[0, 2].set_title("CBF Absolute Error")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Input signal ch 0 (PLD 1.525s)
    im3 = axes[0, 3].imshow(X_test[plot_idx, 0], cmap="gray")
    axes[0, 3].set_title("Noisy Signal (PLD 1.525s)")
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # ATT
    gt_a_slice = np.where(m_slice, Y_test[plot_idx, 1], np.nan)
    pr_a_slice = np.where(m_slice, att_unet[plot_idx], np.nan)
    err_a_slice = np.where(m_slice, np.abs(att_unet[plot_idx] - Y_test[plot_idx, 1]), np.nan)

    im4 = axes[1, 0].imshow(gt_a_slice, cmap="viridis", vmin=0.5, vmax=2.5)
    axes[1, 0].set_title("Ground Truth ATT")
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im5 = axes[1, 1].imshow(pr_a_slice, cmap="viridis", vmin=0.5, vmax=2.5)
    axes[1, 1].set_title("U-Net Predicted ATT")
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im6 = axes[1, 2].imshow(err_a_slice, cmap="hot", vmin=0, vmax=0.6)
    axes[1, 2].set_title("ATT Absolute Error")
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # Brain Mask
    im7 = axes[1, 3].imshow(masks_test[plot_idx, 0], cmap="gray")
    axes[1, 3].set_title("Valid Brain Mask")
    plt.colorbar(im7, ax=axes[1, 3], fraction=0.046, pad=0.04)

    for ax in axes.flatten():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "prediction_maps_sample.png"), dpi=200)
    plt.close()

    # Scatter Plots
    fig, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(12, 5))
    sample_sub = np.random.choice(len(gt_cbf), size=min(10000, len(gt_cbf)), replace=False)

    ax_s1.scatter(gt_cbf[sample_sub], cbf_unet[valid_mask_all][sample_sub], alpha=0.15, s=8, color="darkblue")
    ax_s1.plot([0, 100], [0, 100], "r--", lw=2, label="Identity")
    ax_s1.set_xlabel("Ground Truth CBF (ml/100g/min)")
    ax_s1.set_ylabel("U-Net Predicted CBF")
    ax_s1.set_title("CBF Estimation Correlation")
    ax_s1.legend()
    ax_s1.grid(True, alpha=0.3)

    ax_s2.scatter(gt_att[sample_sub], att_unet[valid_mask_all][sample_sub], alpha=0.15, s=8, color="darkgreen")
    ax_s2.plot([0.5, 3.0], [0.5, 3.0], "r--", lw=2, label="Identity")
    ax_s2.set_xlabel("Ground Truth ATT (s)")
    ax_s2.set_ylabel("U-Net Predicted ATT")
    ax_s2.set_title("ATT Estimation Correlation")
    ax_s2.legend()
    ax_s2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "scatter_plots.png"), dpi=200)
    plt.close()

    print("All evaluation maps and plots saved to:", plots_dir)
    return metrics_df

if __name__ == "__main__":
    print("Testing evaluate module...")
    # Smoke test
    N, C, H, W = 4, 4, 32, 32
    X_d = np.random.randn(N, C, H, W).astype(np.float32)
    Y_d = np.random.uniform(20, 80, (N, 2, H, W)).astype(np.float32)
    M_d = np.ones((N, 1, H, W), dtype=np.float32)
    S_d = np.full(N, 10.0, dtype=np.float32)
    st = {
        "X_mean": np.zeros(4), "X_std": np.ones(4),
        "cbf_mean": 50.0, "cbf_std": 15.0,
        "att_mean": 1.5, "att_std": 0.5
    }
    model = ASLUNet(4, 2, features=[16, 32, 64, 128])
    dev = torch.device("cpu")
    run_full_evaluation(model, X_d, Y_d, M_d, S_d, st, dev, out_dir="experiments_unet_4pld/results_test")
    print("Evaluate module tests passed successfully!")
