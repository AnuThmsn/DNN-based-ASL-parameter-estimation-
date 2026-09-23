import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data_gen import *
import os
from run_all_2 import ResidualNet, standardize, generate_snr_arr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def calc_rmse(yt, yp): return np.sqrt(np.mean((yt - yp)**2))

xtr, xte = X_tr[:200000], X_test
xtr_n, m, s = standardize(xtr)
xte_n, _, _ = standardize(xte, m, s)
xte_t = torch.tensor(xte_n, device=device, dtype=torch.float32)
xte_raw_t = torch.tensor(xte, device=device, dtype=torch.float32)

snr_test = generate_snr_arr(100_000, np.random.default_rng(7))
snr_mask_10 = (snr_test >= 9) & (snr_test <= 11)

net = ResidualNet().to(device)
net.load_state_dict(torch.load("experiments_round2/models/Exp_E_Residual.pth", map_location=device, weights_only=True))
net.eval()
with torch.no_grad():
    preds = []
    bs = 4096
    for i in range(0, xte_t.size(0), bs):
        preds.append(net(xte_t[i:i+bs], xte_raw_t[i:i+bs]).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

cbf_p = np.clip(preds[:, 0] * CBF_std + CBF_mean, CBF_MIN, CBF_MAX)
att_p = np.clip(preds[:, 1] * ATT_std + ATT_mean, ATT_MIN, ATT_MAX)

c_rmse_10 = calc_rmse(Y_test[snr_mask_10, 0], cbf_p[snr_mask_10])
a_rmse_10 = calc_rmse(Y_test[snr_mask_10, 1], att_p[snr_mask_10])

master = pd.read_csv("master_results.csv").to_dict('records')
master.append({'Method': 'DNN Experiment E (Residual)', 'CBF RMSE': c_rmse_10, 'ATT RMSE': a_rmse_10, 'Source': 'Our run'})
pd.DataFrame(master).to_csv("master_results.csv", index=False)
print(f"Exp E added. CBF: {c_rmse_10:.4f}, ATT: {a_rmse_10:.4f}")

