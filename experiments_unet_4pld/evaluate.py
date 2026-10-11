import os
import sys
import time
import json
import yaml
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.ndimage as ndi
import scipy.stats as stats_scipy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_gen import (
    compute_signals_vec,
    PLDs, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX,
    SIG_REF_SC, tau
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

def calc_masked_ssim(true_2d, pred_2d, mask_2d, data_range=100.0):
    """
    Computes masked SSIM using local Gaussian windowing strictly
    within valid brain regions.
    """
    t = np.copy(true_2d).astype(np.float64)
    p = np.copy(pred_2d).astype(np.float64)
    m = (mask_2d > 0.5).astype(np.float64)

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
    
    # Clip to physiological bounds
    cbf_pred = np.clip(cbf_pred, CBF_MIN, CBF_MAX)
    att_pred = np.clip(att_pred, ATT_MIN, ATT_MAX)
    duration = time.time() - start_time
    return cbf_pred, att_pred, duration

def build_pointwise_dnn(in_channels=4, device=None):
    """Constructs a point-wise MLP matching established 5-layer MLP architectures."""
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

    cbf_mlp = MLP(in_channels, 64, 5, 1).to(device)
    att_mlp = MLP(in_channels, 64, 5, 1).to(device)
    return cbf_mlp, att_mlp

def train_pointwise_dnn(X_tr_vox, Y_tr_vox, stats, device, epochs=30, batch_size=4096):
    """Trains a point-wise MLP baseline on flattened training voxels."""
    in_channels = X_tr_vox.shape[1]
    cbf_mlp, att_mlp = build_pointwise_dnn(in_channels=in_channels, device=device)
    
    X_mean = stats["X_mean"][:in_channels]
    X_std = stats["X_std"][:in_channels]
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

    X_mean = stats["X_mean"][:C]
    X_std = stats["X_std"][:C]

    with torch.no_grad():
        for i in range(N):
            valid = masks_test[i, 0] > 0.5
            if not np.any(valid):
                continue
            vox = X_test[i][:, valid].T # [N_valid, C]
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

def evaluate_bayesian_calibrated(
    X_test,
    snrs_test,
    masks_test,
    device=None,
    sig_ref_sc=SIG_REF_SC,
    grid_res=100,
    batch_size=8192
):
    """
    GPU-accelerated Bayesian grid estimator on 100x100 parameter grid.
    Likelihood is calibrated to effective noise variance of sum-of-Ricians:
    sigma_eff = sqrt(2) * sig_ref_sc / SNR.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    N, C, H, W = X_test.shape
    cbf_out = np.zeros((N, H, W), dtype=np.float32)
    att_out = np.zeros((N, H, W), dtype=np.float32)

    # 100x100 dense grid
    cbf_grid = np.linspace(CBF_MIN, CBF_MAX, grid_res, dtype=np.float32)
    att_grid = np.linspace(ATT_MIN, ATT_MAX, grid_res, dtype=np.float32)
    CC, AA = np.meshgrid(cbf_grid, att_grid)
    grid_cbf_t = torch.tensor(CC.flatten(), dtype=torch.float32, device=device)
    grid_att_t = torch.tensor(AA.flatten(), dtype=torch.float32, device=device)

    # Expected signal matching Simulator A observation: 2.0 * compute_signals_vec * SCALE
    grid_sig_np = 2.0 * compute_signals_vec(CC.flatten(), AA.flatten(), ld=tau) * SCALE
    grid_sig_t = torch.tensor(grid_sig_np, dtype=torch.float32, device=device) # [G, 4]
    norm_S = torch.sum(grid_sig_t ** 2, dim=1, keepdim=True).T # [1, G]

    with torch.no_grad():
        for i in range(N):
            valid = masks_test[i, 0] > 0.5
            if not np.any(valid):
                continue
            X_vox = X_test[i][:, valid].T # [V, 4]
            V = len(X_vox)
            snr = float(snrs_test[i])
            # Calibrated effective standard deviation for sum of two Ricians:
            sigma_eff = (np.sqrt(2.0) * sig_ref_sc / snr) if (snr < 1e5 and snr > 0) else 1e-4

            cbf_preds_i = []
            att_preds_i = []

            for v_start in range(0, V, batch_size):
                v_end = min(v_start + batch_size, V)
                xb = torch.tensor(X_vox[v_start:v_end], dtype=torch.float32, device=device)
                norm_X = torch.sum(xb ** 2, dim=1, keepdim=True)
                sq_diff = norm_X - 2.0 * torch.matmul(xb, grid_sig_t.T) + norm_S
                log_L = -0.5 * sq_diff / (sigma_eff ** 2)
                log_L -= torch.max(log_L, dim=1, keepdim=True)[0]
                L = torch.exp(log_L)
                P = L / (torch.sum(L, dim=1, keepdim=True) + 1e-12)

                cb_p = torch.matmul(P, grid_cbf_t).cpu().numpy()
                at_p = torch.matmul(P, grid_att_t).cpu().numpy()
                cbf_preds_i.append(cb_p)
                att_preds_i.append(at_p)

            cbf_out[i][valid] = np.clip(np.concatenate(cbf_preds_i), CBF_MIN, CBF_MAX)
            att_out[i][valid] = np.clip(np.concatenate(att_preds_i), ATT_MIN, ATT_MAX)

    duration = time.time() - start_time
    return cbf_out, att_out, duration

def evaluate_nlls(X_test, masks_test, max_phantoms=25):
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
        s = 2.0 * compute_signals_vec(np.array([c]), np.array([a]), ld=tau) * SCALE
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
                bounds=bounds, method="trf", ftol=1e-2, xtol=1e-2
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
    n_matched_nlls=25,
    out_dir="experiments_unet_4pld/results"
):
    """
    Runs full comparative benchmark across U-Net, Point-wise DNN, Calibrated Bayesian,
    and NLLS estimators under strictly harmonized test conditions.
    """
    plots_dir = os.path.join(out_dir, "plots")
    preds_dir = os.path.join(out_dir, "predictions")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    print("\n================ EVALUATION BENCHMARK ================")
    # 1. Normalize test inputs for U-Net
    X_test_norm = np.copy(X_test)
    for c in range(X_test.shape[1]):
        X_test_norm[:, c] = (X_test[:, c] - stats["X_mean"][c]) / stats["X_std"][c]

    print("Evaluating U-Net...")
    cbf_unet, att_unet, t_unet = evaluate_unet(unet_model, X_test_norm, stats, device)

    print("Training and evaluating Point-wise DNN Baseline...")
    if X_tr_vox is not None and Y_tr_vox is not None:
        cbf_mlp, att_mlp = train_pointwise_dnn(X_tr_vox, Y_tr_vox, stats, device, epochs=30)
        cbf_dnn, att_dnn, t_dnn = evaluate_pointwise_dnn(cbf_mlp, att_mlp, X_test, masks_test, stats, device)
    else:
        cbf_dnn, att_dnn, t_dnn = None, None, 0.0

    print("Evaluating Calibrated Bayesian Estimator (GPU 100x100 grid)...")
    cbf_bayes, att_bayes, t_bayes = evaluate_bayesian_calibrated(X_test, snrs_test, masks_test, device)

    print(f"Evaluating NLLS Estimator (on matched subset of {n_matched_nlls} phantoms)...")
    cbf_nlls, att_nlls, nlls_n, t_nlls = evaluate_nlls(X_test, masks_test, max_phantoms=n_matched_nlls)

    # Save prediction arrays
    np.save(os.path.join(preds_dir, "cbf_unet.npy"), cbf_unet)
    np.save(os.path.join(preds_dir, "att_unet.npy"), att_unet)
    if cbf_dnn is not None:
        np.save(os.path.join(preds_dir, "cbf_dnn.npy"), cbf_dnn)
        np.save(os.path.join(preds_dir, "att_dnn.npy"), att_dnn)
    np.save(os.path.join(preds_dir, "cbf_bayes.npy"), cbf_bayes)
    np.save(os.path.join(preds_dir, "att_bayes.npy"), att_bayes)

    # ----------------- TABLE 1: HARMONIZED MATCHED BENCHMARK (EXACT SAME NLLS SUBSET) -----------------
    valid_mask_matched = masks_test[:nlls_n, 0] > 0.5
    gt_cbf_m = Y_test[:nlls_n, 0][valid_mask_matched]
    gt_att_m = Y_test[:nlls_n, 1][valid_mask_matched]

    matched_dict = {
        "U-Net (Proposed)": (cbf_unet[:nlls_n][valid_mask_matched], att_unet[:nlls_n][valid_mask_matched], t_unet * (nlls_n / len(X_test))),
        "Bayesian (Calibrated Grid)": (cbf_bayes[:nlls_n][valid_mask_matched], att_bayes[:nlls_n][valid_mask_matched], t_bayes * (nlls_n / len(X_test))),
        "NLLS (TRF)": (cbf_nlls[valid_mask_matched], att_nlls[valid_mask_matched], t_nlls)
    }
    if cbf_dnn is not None:
        matched_dict["Point-wise DNN"] = (cbf_dnn[:nlls_n][valid_mask_matched], att_dnn[:nlls_n][valid_mask_matched], t_dnn * (nlls_n / len(X_test)))

    matched_rows = []
    for name, (pred_c, pred_a, duration) in matched_dict.items():
        met_c = calc_metrics(gt_cbf_m, pred_c)
        met_a = calc_metrics(gt_att_m, pred_a)
        matched_rows.append({
            "Method": name,
            "Evaluation": f"Harmonized Matched Set ({nlls_n} phantoms)",
            "N_Phantoms": nlls_n,
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

    # ----------------- TABLE 2: FULL-SCALE BENCHMARK (ALL 100 PHANTOMS) -----------------
    valid_mask_all = masks_test[:, 0] > 0.5
    gt_cbf_all = Y_test[:, 0][valid_mask_all]
    gt_att_all = Y_test[:, 1][valid_mask_all]

    full_dict = {
        "U-Net (Proposed)": (cbf_unet[valid_mask_all], att_unet[valid_mask_all], t_unet, len(X_test)),
        "Bayesian (Calibrated Grid)": (cbf_bayes[valid_mask_all], att_bayes[valid_mask_all], t_bayes, len(X_test)),
    }
    if cbf_dnn is not None:
        full_dict["Point-wise DNN"] = (cbf_dnn[valid_mask_all], att_dnn[valid_mask_all], t_dnn, len(X_test))
    full_rows = []
    for name, (pred_c, pred_a, duration, n_ph) in full_dict.items():
        met_c = calc_metrics(gt_cbf_all, pred_c)
        met_a = calc_metrics(gt_att_all, pred_a)
        full_rows.append({
            "Method": name,
            "Evaluation": f"Full Test Set ({n_ph} phantoms)",
            "N_Phantoms": n_ph,
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

    # Add historical literature references
    ref_yaml_path = os.path.join(parent_dir, "reference_baselines.yaml")
    if os.path.exists(ref_yaml_path):
        with open(ref_yaml_path, "r") as f:
            ref_data = yaml.safe_load(f)
        for snr_key, methods in ref_data.items():
            for m_name, vals in methods.items():
                full_rows.append({
                    "Method": f"{m_name} (Fixed Ref {snr_key})",
                    "Evaluation": f"Historical Fixed Literature ({snr_key})",
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

    matched_df = pd.DataFrame(matched_rows)
    matched_df.to_csv(os.path.join(out_dir, "matched_benchmark_25_phantoms.csv"), index=False)

    full_df = pd.DataFrame(full_rows)
    full_df.to_csv(os.path.join(out_dir, "master_comparison.csv"), index=False)

    print("\n--- Harmonized Matched Benchmark (Exact Same Phantoms across All Methods) ---")
    print(matched_df.to_string())

    print("\n--- Full-Scale Benchmark (All 100 Phantoms) ---")
    print(full_df.to_string())

    # ----------------- PHASE 6: PAIRED STATISTICAL COMPARISONS (PER-PHANTOM) -----------------
    # Compute per-phantom metrics on the matched set
    phantom_records = []
    for p in range(nlls_n):
        m_p = masks_test[p, 0] > 0.5
        if not np.any(m_p):
            continue
        tc_p = Y_test[p, 0][m_p]
        ta_p = Y_test[p, 1][m_p]

        c_u = np.sqrt(np.mean((cbf_unet[p][m_p] - tc_p)**2))
        a_u = np.sqrt(np.mean((att_unet[p][m_p] - ta_p)**2))
        c_b = np.sqrt(np.mean((cbf_bayes[p][m_p] - tc_p)**2))
        a_b = np.sqrt(np.mean((att_bayes[p][m_p] - ta_p)**2))
        c_n = np.sqrt(np.mean((cbf_nlls[p][m_p] - tc_p)**2))
        a_n = np.sqrt(np.mean((att_nlls[p][m_p] - ta_p)**2))

        rec = {
            "phantom_id": p,
            "unet_cbf": c_u, "unet_att": a_u,
            "bayes_cbf": c_b, "bayes_att": a_b,
            "nlls_cbf": c_n, "nlls_att": a_n
        }
        if cbf_dnn is not None:
            rec["dnn_cbf"] = np.sqrt(np.mean((cbf_dnn[p][m_p] - tc_p)**2))
            rec["dnn_att"] = np.sqrt(np.mean((att_dnn[p][m_p] - ta_p)**2))
        phantom_records.append(rec)

    p_df = pd.DataFrame(phantom_records)
    p_df.to_csv(os.path.join(out_dir, "per_phantom_metrics.csv"), index=False)

    # Paired statistical comparisons (U-Net vs each baseline)
    comparisons = [
        ("Bayesian (Calibrated)", "bayes_cbf", "bayes_att"),
        ("NLLS (TRF)", "nlls_cbf", "nlls_att")
    ]
    if cbf_dnn is not None:
        comparisons.insert(0, ("Point-wise DNN", "dnn_cbf", "dnn_att"))

    stat_rows = []
    num_comparisons = len(comparisons)
    rng_boot = np.random.RandomState(42)

    for baseline, c_col, a_col in comparisons:
        diff_cbf = p_df["unet_cbf"] - p_df[c_col]
        diff_att = p_df["unet_att"] - p_df[a_col]

        t_c, p_t_c = stats_scipy.ttest_rel(p_df["unet_cbf"], p_df[c_col])
        w_c, p_w_c = stats_scipy.wilcoxon(diff_cbf)

        t_a, p_t_a = stats_scipy.ttest_rel(p_df["unet_att"], p_df[a_col])
        w_a, p_w_a = stats_scipy.wilcoxon(diff_att)

        # Standard parametric CI
        ci_c_low, ci_c_high = stats_scipy.t.interval(0.95, len(diff_cbf)-1, loc=np.mean(diff_cbf), scale=stats_scipy.sem(diff_cbf))
        ci_a_low, ci_a_high = stats_scipy.t.interval(0.95, len(diff_att)-1, loc=np.mean(diff_att), scale=stats_scipy.sem(diff_att))

        # Phantom-level Bootstrap 95% CI (1000 resamples)
        boot_c = [np.mean(rng_boot.choice(diff_cbf, size=len(diff_cbf), replace=True)) for _ in range(1000)]
        boot_a = [np.mean(rng_boot.choice(diff_att, size=len(diff_att), replace=True)) for _ in range(1000)]
        bci_c_low, bci_c_high = np.percentile(boot_c, [2.5, 97.5])
        bci_a_low, bci_a_high = np.percentile(boot_a, [2.5, 97.5])

        # Effect size (Cohen's d)
        d_cbf = float(np.mean(diff_cbf) / (np.std(diff_cbf, ddof=1) + 1e-12))
        d_att = float(np.mean(diff_att) / (np.std(diff_att, ddof=1) + 1e-12))

        # Bonferroni multiple testing correction
        p_c_bonf = min(1.0, float(p_t_c) * num_comparisons)
        p_a_bonf = min(1.0, float(p_t_a) * num_comparisons)

        stat_rows.append({
            "Comparison": f"U-Net vs {baseline}",
            "N_Phantoms": len(p_df),
            "CBF Mean Diff": round(float(np.mean(diff_cbf)), 4),
            "CBF Parametric 95% CI": f"[{ci_c_low:.4f}, {ci_c_high:.4f}]",
            "CBF Bootstrap 95% CI": f"[{bci_c_low:.4f}, {bci_c_high:.4f}]",
            "CBF Cohen d": round(d_cbf, 3),
            "CBF t-test p": f"{p_t_c:.2e}",
            "CBF Bonferroni p": f"{p_c_bonf:.2e}",
            "CBF Wilcoxon p": f"{p_w_c:.2e}",
            "CBF Significant": bool(p_c_bonf < 0.05),
            "ATT Mean Diff": round(float(np.mean(diff_att)), 4),
            "ATT Parametric 95% CI": f"[{ci_a_low:.4f}, {ci_a_high:.4f}]",
            "ATT Bootstrap 95% CI": f"[{bci_a_low:.4f}, {bci_a_high:.4f}]",
            "ATT Cohen d": round(d_att, 3),
            "ATT t-test p": f"{p_t_a:.2e}",
            "ATT Bonferroni p": f"{p_a_bonf:.2e}",
            "ATT Wilcoxon p": f"{p_w_a:.2e}",
            "ATT Significant": bool(p_a_bonf < 0.05),
        })

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(os.path.join(out_dir, "paired_statistical_tests.csv"), index=False)
    print("\n--- Paired Statistical Tests (Unit of Replication: Independent Phantoms) ---")
    print(stat_df.to_string())

    # ----------------- SNR-STRATIFIED EVALUATION -----------------
    snr_rows = []
    for snr_target in [10.0, 15.0]:
        mask_snr = (np.abs(snrs_test - snr_target) < 1.0)
        if not np.any(mask_snr):
            continue
        v_mask_snr = (masks_test[mask_snr, 0] > 0.5)
        tc_snr = Y_test[mask_snr, 0][v_mask_snr]
        ta_snr = Y_test[mask_snr, 1][v_mask_snr]

        snr_methods = [
            ("U-Net", (cbf_unet, att_unet)),
            ("Bayesian", (cbf_bayes, att_bayes))
        ]
        if cbf_dnn is not None:
            snr_methods.append(("Point-wise DNN", (cbf_dnn, att_dnn)))

        for name, pred_tuple in snr_methods:
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
        snr_df.to_csv(os.path.join(out_dir, "snr_stratified_metrics.csv"), index=False)
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

    # ----------------- PLOT GENERATION -----------------
    print("Generating benchmark plots...")
    generate_evaluation_plots(
        Y_test=Y_test,
        masks_test=masks_test,
        cbf_unet=cbf_unet,
        att_unet=att_unet,
        cbf_dnn=cbf_dnn,
        att_dnn=att_dnn,
        cbf_bayes=cbf_bayes,
        att_bayes=att_bayes,
        cbf_nlls=cbf_nlls,
        att_nlls=att_nlls,
        plots_dir=plots_dir
    )

    return full_df

def generate_evaluation_plots(
    Y_test,
    masks_test,
    cbf_unet,
    att_unet,
    cbf_dnn,
    att_dnn,
    cbf_bayes,
    att_bayes,
    cbf_nlls,
    att_nlls,
    plots_dir
):
    """Generates prediction maps, multi-method scatter plots, and Bland-Altman plots."""
    # 1. Prediction maps for sample phantoms (e.g. phantoms 0, 1, 2)
    sample_indices = [0, 1, 2] if len(Y_test) >= 3 else list(range(len(Y_test)))
    for s_idx in sample_indices:
        mask = masks_test[s_idx, 0] > 0.5
        gt_c = np.where(mask, Y_test[s_idx, 0], np.nan)
        pr_c = np.where(mask, cbf_unet[s_idx], np.nan)
        err_c = np.where(mask, np.abs(cbf_unet[s_idx] - Y_test[s_idx, 0]), np.nan)

        gt_a = np.where(mask, Y_test[s_idx, 1], np.nan)
        pr_a = np.where(mask, att_unet[s_idx], np.nan)
        err_a = np.where(mask, np.abs(att_unet[s_idx] - Y_test[s_idx, 1]), np.nan)

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        im0 = axes[0, 0].imshow(gt_c, cmap="hot", vmin=0, vmax=100)
        axes[0, 0].set_title(f"Phantom {s_idx}: True CBF")
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(pr_c, cmap="hot", vmin=0, vmax=100)
        axes[0, 1].set_title("U-Net Pred CBF")
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[0, 2].imshow(err_c, cmap="magma", vmin=0, vmax=15)
        axes[0, 2].set_title("CBF Absolute Error")
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        im3 = axes[1, 0].imshow(gt_a, cmap="viridis", vmin=0.5, vmax=3.0)
        axes[1, 0].set_title(f"Phantom {s_idx}: True ATT")
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im4 = axes[1, 1].imshow(pr_a, cmap="viridis", vmin=0.5, vmax=3.0)
        axes[1, 1].set_title("U-Net Pred ATT")
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

        im5 = axes[1, 2].imshow(err_a, cmap="magma", vmin=0, vmax=0.5)
        axes[1, 2].set_title("ATT Absolute Error")
        plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

        for ax in axes.flat:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"prediction_maps_sample_{s_idx}.png"), dpi=200)
        plt.close()

    # 2. Scatter Plots (Subsampled valid voxels from matched set)
    valid_m = masks_test[:len(cbf_nlls), 0] > 0.5
    gt_c_all = Y_test[:len(cbf_nlls), 0][valid_m]
    gt_a_all = Y_test[:len(cbf_nlls), 1][valid_m]

    methods_data = [
        ("U-Net", cbf_unet[:len(cbf_nlls)][valid_m], att_unet[:len(cbf_nlls)][valid_m], "tab:blue"),
        ("Point-wise DNN", cbf_dnn[:len(cbf_nlls)][valid_m] if cbf_dnn is not None else None, 
                           att_dnn[:len(cbf_nlls)][valid_m] if att_dnn is not None else None, "tab:orange"),
        ("Bayesian", cbf_bayes[:len(cbf_nlls)][valid_m], att_bayes[:len(cbf_nlls)][valid_m], "tab:green"),
        ("NLLS", cbf_nlls[valid_m], att_nlls[valid_m], "tab:purple")
    ]

    n_pts = min(4000, len(gt_c_all))
    sub_idx = np.random.RandomState(42).choice(len(gt_c_all), size=n_pts, replace=False)
    gt_c_sub = gt_c_all[sub_idx]
    gt_a_sub = gt_a_all[sub_idx]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for col, (name, pred_c, pred_a, colr) in enumerate(methods_data):
        if pred_c is None:
            continue
        p_c_sub = pred_c[sub_idx]
        p_a_sub = pred_a[sub_idx]

        # CBF
        axes[0, col].scatter(gt_c_sub, p_c_sub, alpha=0.25, s=8, color=colr)
        axes[0, col].plot([0, 100], [0, 100], "k--", lw=1.5, label="Identity")
        rmse_c = np.sqrt(np.mean((pred_c - gt_c_all)**2))
        axes[0, col].set_title(f"{name}\nCBF RMSE: {rmse_c:.2f} ml/100g/min", fontsize=11)
        axes[0, col].set_xlabel("True CBF")
        axes[0, col].set_ylabel("Predicted CBF" if col == 0 else "")
        axes[0, col].set_xlim(0, 100)
        axes[0, col].set_ylim(0, 100)
        axes[0, col].grid(True, alpha=0.3)

        # ATT
        axes[1, col].scatter(gt_a_sub, p_a_sub, alpha=0.25, s=8, color=colr)
        axes[1, col].plot([0.5, 3.0], [0.5, 3.0], "k--", lw=1.5, label="Identity")
        rmse_a = np.sqrt(np.mean((pred_a - gt_a_all)**2))
        axes[1, col].set_title(f"ATT RMSE: {rmse_a:.3f} s", fontsize=11)
        axes[1, col].set_xlabel("True ATT (s)")
        axes[1, col].set_ylabel("Predicted ATT (s)" if col == 0 else "")
        axes[1, col].set_xlim(0.5, 3.0)
        axes[1, col].set_ylim(0.5, 3.0)
        axes[1, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "scatter_plots.png"), dpi=200)
    plt.close()

    # 3. Bland-Altman Analysis (U-Net)
    diff_c = cbf_unet[:len(cbf_nlls)][valid_m] - gt_c_all
    mean_c = 0.5 * (cbf_unet[:len(cbf_nlls)][valid_m] + gt_c_all)
    diff_a = att_unet[:len(cbf_nlls)][valid_m] - gt_a_all
    mean_a = 0.5 * (att_unet[:len(cbf_nlls)][valid_m] + gt_a_all)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    d_c_sub = diff_c[sub_idx]
    m_c_sub = mean_c[sub_idx]
    mean_diff_c = np.mean(diff_c)
    sd_diff_c = np.std(diff_c)

    ax1.scatter(m_c_sub, d_c_sub, alpha=0.25, s=8, color="tab:blue")
    ax1.axhline(mean_diff_c, color="red", linestyle="--", label=f"Mean: {mean_diff_c:.2f}")
    ax1.axhline(mean_diff_c + 1.96 * sd_diff_c, color="gray", linestyle=":", label=f"+1.96 SD: {mean_diff_c + 1.96 * sd_diff_c:.2f}")
    ax1.axhline(mean_diff_c - 1.96 * sd_diff_c, color="gray", linestyle=":", label=f"-1.96 SD: {mean_diff_c - 1.96 * sd_diff_c:.2f}")
    ax1.set_xlabel("Mean CBF (ml/100g/min)")
    ax1.set_ylabel("Difference (Pred - True)")
    ax1.set_title("U-Net Bland-Altman: CBF")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    d_a_sub = diff_a[sub_idx]
    m_a_sub = mean_a[sub_idx]
    mean_diff_a = np.mean(diff_a)
    sd_diff_a = np.std(diff_a)

    ax2.scatter(m_a_sub, d_a_sub, alpha=0.25, s=8, color="tab:green")
    ax2.axhline(mean_diff_a, color="red", linestyle="--", label=f"Mean: {mean_diff_a:.3f}")
    ax2.axhline(mean_diff_a + 1.96 * sd_diff_a, color="gray", linestyle=":", label=f"+1.96 SD: {mean_diff_a + 1.96 * sd_diff_a:.3f}")
    ax2.axhline(mean_diff_a - 1.96 * sd_diff_a, color="gray", linestyle=":", label=f"-1.96 SD: {mean_diff_a - 1.96 * sd_diff_a:.3f}")
    ax2.set_xlabel("Mean ATT (s)")
    ax2.set_ylabel("Difference (Pred - True)")
    ax2.set_title("U-Net Bland-Altman: ATT")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "bland_altman_plots.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    print("Testing evaluate module...")
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
    run_full_evaluation(model, X_d, Y_d, M_d, S_d, st, dev, n_matched_nlls=2, out_dir="experiments_unet_4pld/results_test")
    print("Evaluate module tests passed successfully!")
