import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from data_gen import compute_signals_vec, PLDs, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX, SD_MAX, SIG_REF_SC

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
Y_test, snr_test = generate_test_with_snr(100_000, rng_test2)

os.makedirs("plots/diagnosis", exist_ok=True)

# Load baseline predictions
cbf_pred = np.load("experiments/exp01_baseline/cbf_pred.npy")
att_pred = np.load("experiments/exp01_baseline/att_pred.npy")

y_cbf = Y_test[:, 0]
y_att = Y_test[:, 1]

err_cbf = cbf_pred - y_cbf
err_att = att_pred - y_att

# 1. Error vs SNR plots
snr_bins = [5, 10, 15, 20, 30, 50, 100]
rmse_cbf, rmse_att = [], []
mdae_cbf, mdae_att = [], []
for s in snr_bins:
    mask = (snr_test >= s * 0.9) & (snr_test <= s * 1.1)
    if mask.sum() > 0:
        rmse_cbf.append(np.sqrt(np.mean(err_cbf[mask]**2)))
        rmse_att.append(np.sqrt(np.mean(err_att[mask]**2)))
        mdae_cbf.append(np.median(np.abs(err_cbf[mask])))
        mdae_att.append(np.median(np.abs(err_att[mask])))
    else:
        rmse_cbf.append(np.nan); rmse_att.append(np.nan)
        mdae_cbf.append(np.nan); mdae_att.append(np.nan)

plt.figure(figsize=(10, 4))
plt.subplot(121); plt.plot(snr_bins, rmse_cbf, '-o', label='CBF RMSE'); plt.plot(snr_bins, mdae_cbf, '-s', label='CBF MdAE'); plt.xlabel("SNR"); plt.legend(); plt.title("CBF Error vs SNR")
plt.subplot(122); plt.plot(snr_bins, rmse_att, '-o', label='ATT RMSE'); plt.plot(snr_bins, mdae_att, '-s', label='ATT MdAE'); plt.xlabel("SNR"); plt.legend(); plt.title("ATT Error vs SNR")
plt.tight_layout(); plt.savefig("plots/diagnosis/error_vs_snr.png"); plt.close()

# 2. Error vs parameter plots
plt.figure(figsize=(10, 4))
plt.subplot(121); plt.scatter(y_cbf, err_cbf, alpha=0.1, s=1); plt.xlabel("True CBF"); plt.ylabel("Error"); plt.axhline(0, color='r'); plt.title("CBF Error vs CBF")
plt.subplot(122); plt.scatter(y_att, err_att, alpha=0.1, s=1); plt.xlabel("True ATT"); plt.ylabel("Error"); plt.axhline(0, color='r'); plt.title("ATT Error vs ATT")
plt.tight_layout(); plt.savefig("plots/diagnosis/error_vs_param.png"); plt.close()

# 3. Pred vs GT
plt.figure(figsize=(10, 4))
plt.subplot(121); plt.scatter(y_cbf, cbf_pred, alpha=0.1, s=1); plt.xlabel("True CBF"); plt.ylabel("Pred CBF"); plt.plot([CBF_MIN, CBF_MAX], [CBF_MIN, CBF_MAX], 'r'); plt.title("CBF Pred vs GT")
plt.subplot(122); plt.scatter(y_att, att_pred, alpha=0.1, s=1); plt.xlabel("True ATT"); plt.ylabel("Pred ATT"); plt.plot([ATT_MIN, ATT_MAX], [ATT_MIN, ATT_MAX], 'r'); plt.title("ATT Pred vs GT")
plt.tight_layout(); plt.savefig("plots/diagnosis/pred_vs_gt.png"); plt.close()

# 5. 2D Parameter-Error Heatmap
cbf_bins = np.linspace(CBF_MIN, CBF_MAX, 11)
att_bins = np.linspace(ATT_MIN, ATT_MAX, 11)
heatmap_cbf = np.zeros((10, 10))
heatmap_att = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        mask = (y_cbf >= cbf_bins[i]) & (y_cbf < cbf_bins[i+1]) & (y_att >= att_bins[j]) & (y_att < att_bins[j+1])
        if mask.sum() > 0:
            heatmap_cbf[j, i] = np.sqrt(np.mean(err_cbf[mask]**2)) # Note: imshow plots y-axis (row) top-to-bottom. We'll map j to y (ATT)
            heatmap_att[j, i] = np.sqrt(np.mean(err_att[mask]**2))
        else:
            heatmap_cbf[j, i] = np.nan
            heatmap_att[j, i] = np.nan

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(heatmap_cbf, origin='lower', extent=[CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX], aspect='auto', cmap='viridis')
plt.colorbar(label="CBF RMSE")
plt.xlabel("True CBF"); plt.ylabel("True ATT"); plt.title("CBF Error Heatmap")

plt.subplot(122)
plt.imshow(heatmap_att, origin='lower', extent=[CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX], aspect='auto', cmap='viridis')
plt.colorbar(label="ATT RMSE")
plt.xlabel("True CBF"); plt.ylabel("True ATT"); plt.title("ATT Error Heatmap")
plt.tight_layout(); plt.savefig("plots/diagnosis/error_heatmap.png"); plt.close()

# 6. Forward model sensitivity
# Perturb CBF by 1% and ATT by 1%
cbf_ref = 50.0
att_ref = 1.5
base_sig = compute_signals_vec(np.array([cbf_ref]), np.array([att_ref])) * SCALE

sig_cbf_plus = compute_signals_vec(np.array([cbf_ref + 1.0]), np.array([att_ref])) * SCALE
ds_dcbf = (sig_cbf_plus - base_sig) / 1.0

sig_att_plus = compute_signals_vec(np.array([cbf_ref]), np.array([att_ref + 0.1])) * SCALE
ds_datt = (sig_att_plus - base_sig) / 0.1

print("Sensitivity Analysis at CBF=50, ATT=1.5:")
print(f"PLDs: {PLDs}")
print(f"Base Signal: {base_sig[0]}")
print(f"dS/dCBF: {ds_dcbf[0]}")
print(f"dS/dATT: {ds_datt[0]}")
print("Diagnosis plots generated in plots/diagnosis/")

