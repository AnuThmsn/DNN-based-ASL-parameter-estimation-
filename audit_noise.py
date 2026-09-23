import numpy as np
import pandas as pd
from data_gen import compute_signals_vec, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX, SD_MAX, SIG_REF_SC
def calc_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))
from baselines import fit_nlls_batch, fit_bayesian_batch
import time

def generate_test_simulators(N_total, rng, n_noise_levels=51):
    cbf_gt = rng.uniform(CBF_MIN, CBF_MAX, N_total).astype(np.float64)
    att_gt = rng.uniform(ATT_MIN, ATT_MAX, N_total).astype(np.float64)
    sd_levels = np.linspace(0.0, SD_MAX, n_noise_levels)
    sd_choice_idx = rng.integers(0, n_noise_levels, size=N_total)
    sd_arr = sd_levels[sd_choice_idx]
    
    with np.errstate(divide='ignore'):
        snr_arr = np.where(sd_arr > 0, SIG_REF_SC / sd_arr, np.inf)
        
    delta_M = compute_signals_vec(cbf_gt, att_gt) * SCALE
    sd = sd_arr[:, None]
    
    # Same noise realization for fairness
    e1 = rng.normal(0.0, 1.0, size=delta_M.shape) * sd
    e2 = rng.normal(0.0, 1.0, size=delta_M.shape) * sd
    e3 = rng.normal(0.0, 1.0, size=delta_M.shape) * sd
    e4 = rng.normal(0.0, 1.0, size=delta_M.shape) * sd
    
    # Simulator A (Current/Flawed)
    # X = sqrt((dM + e1)^2 + e2^2) + sqrt((dM + e3)^2 + e4^2)
    mc_A = np.sqrt((delta_M + e1) ** 2 + e2 ** 2)
    ml_A = np.sqrt((delta_M + e3) ** 2 + e4 ** 2)
    X_A = (mc_A + ml_A).astype(np.float32)
    
    # Simulator B (Physical Subtraction)
    # Mc = M0, Ml = M0 - dM
    M0 = 100 * SIG_REF_SC # Arbitrary large M0
    mc_B = np.sqrt((M0 + e1)**2 + e2**2)
    ml_B = np.sqrt((M0 - delta_M + e3)**2 + e4**2)
    X_B = (mc_B - ml_B).astype(np.float32)
    
    # Simulator C (Gaussian difference)
    # e_diff = (e1 - e3) ? Wait, if Mc and Ml have noise sd, diff has sqrt(2)*sd
    X_C = (delta_M + rng.normal(0.0, 1.0, size=delta_M.shape) * (np.sqrt(2)*sd)).astype(np.float32)
    
    Y = np.stack([cbf_gt, att_gt], axis=1).astype(np.float32)
    return Y, snr_arr, X_A, X_B, X_C

rng_test = np.random.default_rng(42)
N_TEST = 10_000
Y, snr_arr, X_A, X_B, X_C = generate_test_simulators(N_TEST, rng_test)

print("Running Simulator Comparison on SNR=10...")
mask10 = (snr_arr >= 9) & (snr_arr <= 11)
Y_10 = Y[mask10]
XA_10 = X_A[mask10]
XB_10 = X_B[mask10]
XC_10 = X_C[mask10]
snr_10 = snr_arr[mask10]

print(f"Testing on {Y_10.shape[0]} samples.")

def evaluate_sim(X_test, name):
    print(f"\n--- {name} ---")
    preds_nlls, t_nlls = fit_nlls_batch(X_test)
    preds_bayes, t_bayes = fit_bayesian_batch(X_test, snr_10, SIG_REF_SC)
    
    rmse_nlls_c = calc_rmse(Y_10[:,0], preds_nlls[:,0])
    rmse_nlls_a = calc_rmse(Y_10[:,1], preds_nlls[:,1])
    rmse_bayes_c = calc_rmse(Y_10[:,0], preds_bayes[:,0])
    rmse_bayes_a = calc_rmse(Y_10[:,1], preds_bayes[:,1])
    
    print(f"NLLS CBF RMSE: {rmse_nlls_c:.4f}, ATT RMSE: {rmse_nlls_a:.4f} (Time: {t_nlls:.2f}s)")
    print(f"Bayes CBF RMSE: {rmse_bayes_c:.4f}, ATT RMSE: {rmse_bayes_a:.4f} (Time: {t_bayes:.2f}s)")

evaluate_sim(XA_10, "Simulator A (Current: mc + ml around dM)")
evaluate_sim(XB_10, "Simulator B (Physical: M0 - (M0-dM))")
evaluate_sim(XC_10, "Simulator C (Gaussian: dM + noise)")
