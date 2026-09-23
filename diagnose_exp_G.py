import numpy as np
import pandas as pd
from data_gen import *
import os

os.makedirs("experiments_G/diagnostics", exist_ok=True)

print("Loading predictions...")
preds = np.load("experiments_G/baseline_G_preds_mean.npy")
y_cbf = Y_test[:, 0]
y_att = Y_test[:, 1]
p_cbf = preds[:, 0]
p_att = preds[:, 1]

# 1. Error by CBF range
cbf_bins = [0, 20, 40, 60, 80, 100, 150] # up to max CBF
cbf_results = []
for i in range(len(cbf_bins)-1):
    low, high = cbf_bins[i], cbf_bins[i+1]
    mask = (y_cbf >= low) & (y_cbf < high)
    n = mask.sum()
    if n > 0:
        rmse = np.sqrt(np.mean((y_cbf[mask] - p_cbf[mask])**2))
        mae = np.mean(np.abs(y_cbf[mask] - p_cbf[mask]))
        bias = np.mean(p_cbf[mask] - y_cbf[mask])
        cbf_results.append({'Range': f'{low}-{high}', 'N': n, 'RMSE': rmse, 'MAE': mae, 'Bias': bias})
pd.DataFrame(cbf_results).to_csv("experiments_G/diagnostics/cbf_bins.csv", index=False)

# 2. Error by ATT range
att_bins = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
att_results = []
for i in range(len(att_bins)-1):
    low, high = att_bins[i], att_bins[i+1]
    mask = (y_att >= low) & (y_att < high)
    n = mask.sum()
    if n > 0:
        rmse = np.sqrt(np.mean((y_att[mask] - p_att[mask])**2))
        mae = np.mean(np.abs(y_att[mask] - p_att[mask]))
        bias = np.mean(p_att[mask] - y_att[mask])
        att_results.append({'Range': f'{low}-{high}', 'N': n, 'RMSE': rmse, 'MAE': mae, 'Bias': bias})
pd.DataFrame(att_results).to_csv("experiments_G/diagnostics/att_bins.csv", index=False)

# 3. Error by SNR
def generate_snr_arr(N_total, rng, n_noise_levels=51):
    sd_levels = np.linspace(0.0, SD_MAX, n_noise_levels)
    sd_choice_idx = rng.integers(0, n_noise_levels, size=N_total)
    sd_arr = sd_levels[sd_choice_idx]
    with np.errstate(divide='ignore'):
        return np.where(sd_arr > 0, SIG_REF_SC / sd_arr, np.inf)

snr_test = generate_snr_arr(100_000, np.random.default_rng(7))
snr_levels = [10, 15, 20, 30, 40, 50]
snr_results = []
for s in snr_levels:
    mask = (snr_test >= s * 0.9) & (snr_test <= s * 1.1)
    n = mask.sum()
    if n > 0:
        c_rmse = np.sqrt(np.mean((y_cbf[mask] - p_cbf[mask])**2))
        c_mae = np.mean(np.abs(y_cbf[mask] - p_cbf[mask]))
        a_rmse = np.sqrt(np.mean((y_att[mask] - p_att[mask])**2))
        a_mae = np.mean(np.abs(y_att[mask] - p_att[mask]))
        snr_results.append({'SNR': s, 'N': n, 'CBF RMSE': c_rmse, 'CBF MAE': c_mae, 'ATT RMSE': a_rmse, 'ATT MAE': a_mae})
pd.DataFrame(snr_results).to_csv("experiments_G/diagnostics/snr_curves.csv", index=False)

print("Diagnostics saved to experiments_G/diagnostics/")
