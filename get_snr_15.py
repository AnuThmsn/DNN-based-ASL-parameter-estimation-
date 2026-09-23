import numpy as np
import pandas as pd
from data_gen import compute_signals_vec, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX, SD_MAX, SIG_REF_SC
import torch

def generate_test_with_snr(N_total, rng, n_noise_levels=51):
    cbf_gt = rng.uniform(CBF_MIN, CBF_MAX, N_total).astype(np.float64)
    att_gt = rng.uniform(ATT_MIN, ATT_MAX, N_total).astype(np.float64)
    sd_levels = np.linspace(0.0, SD_MAX, n_noise_levels)
    sd_choice_idx = rng.integers(0, n_noise_levels, size=N_total)
    sd_arr = sd_levels[sd_choice_idx]
    with np.errstate(divide='ignore'):
        snr_arr = np.where(sd_arr > 0, SIG_REF_SC / sd_arr, np.inf)
    Y = np.stack([cbf_gt, att_gt], axis=1).astype(np.float32)
    return Y, snr_arr

rng_test2 = np.random.default_rng(7)
Y_test_new, snr_test = generate_test_with_snr(100_000, rng_test2)

cbf_pred = np.load('experiments/exp01_baseline/cbf_pred.npy')
att_pred = np.load('experiments/exp01_baseline/att_pred.npy')

y_cbf = Y_test_new[:, 0]
y_att = Y_test_new[:, 1]

for s in [10, 15]:
    mask = (snr_test >= s * 0.9) & (snr_test <= s * 1.1)
    rmse_c = np.sqrt(np.mean((cbf_pred[mask] - y_cbf[mask])**2))
    mdae_c = np.median(np.abs(cbf_pred[mask] - y_cbf[mask]))
    mdb_c = np.median(cbf_pred[mask] - y_cbf[mask])
    
    rmse_a = np.sqrt(np.mean((att_pred[mask] - y_att[mask])**2))
    mdae_a = np.median(np.abs(att_pred[mask] - y_att[mask]))
    mdb_a = np.median(att_pred[mask] - y_att[mask])
    
    print(f'SNR={s} CBF RMSE={rmse_c:.4f} MdAE={mdae_c:.4f} MdB={mdb_c:.4f}')
    print(f'SNR={s} ATT RMSE={rmse_a:.4f} MdAE={mdae_a:.4f} MdB={mdb_a:.4f}')

