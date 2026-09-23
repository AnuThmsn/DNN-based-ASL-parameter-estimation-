import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data_gen import *
import os
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def standardize(X, m=None, s=None):
    if m is None:
        m = X.mean(axis=0, keepdims=True)
        s = X.std(axis=0, keepdims=True) + 1e-8
    return (X - m) / s, m, s

def generate_snr_arr(N_total, rng, n_noise_levels=51):
    sd_levels = np.linspace(0.0, SD_MAX, n_noise_levels)
    sd_choice_idx = rng.integers(0, n_noise_levels, size=N_total)
    sd_arr = sd_levels[sd_choice_idx]
    with np.errstate(divide='ignore'):
        return np.where(sd_arr > 0, SIG_REF_SC / sd_arr, np.inf)

snr_test = generate_snr_arr(100_000, np.random.default_rng(7))
snr_mask_10 = (snr_test >= 9) & (snr_test <= 11)

xtr, xte = X_tr, X_test
xtr_n, m, s = standardize(xtr)
xte_n, _, _ = standardize(xte, m, s)
xtr_t = torch.tensor(xtr_n, device=device, dtype=torch.float32)
xte_t = torch.tensor(xte_n, device=device, dtype=torch.float32)
ytr_t = Y_tr_both_t
yval_t = Y_val_both_t
xtr_raw_t = torch.tensor(xtr, device=device, dtype=torch.float32)
xte_raw_t = torch.tensor(xte, device=device, dtype=torch.float32)

def calc_rmse(yt, yp): return np.sqrt(np.mean((yt - yp)**2))

def train_eval_res(net, exp_name, epochs=40, bs=4096):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.L1Loss()
    print(f"Training {exp_name}...", flush=True)
    
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(xtr_t.size(0), device=device)
        for i in range(0, xtr_t.size(0), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = crit(net(xtr_t[idx], xtr_raw_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            
    torch.save(net.state_dict(), f"experiments_round2/models/{exp_name.replace(' ', '_')}.pth")
    net.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, xte_t.size(0), bs):
            preds.append(net(xte_t[i:i+bs], xte_raw_t[i:i+bs]).cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        
    cbf_p = np.clip(preds[:, 0] * CBF_std + CBF_mean, CBF_MIN, CBF_MAX)
    att_p = np.clip(preds[:, 1] * ATT_std + ATT_mean, ATT_MIN, ATT_MAX)
    
    c_rmse_10 = calc_rmse(Y_test[snr_mask_10, 0], cbf_p[snr_mask_10])
    a_rmse_10 = calc_rmse(Y_test[snr_mask_10, 1], att_p[snr_mask_10])
    return c_rmse_10, a_rmse_10

class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_net = nn.Sequential(nn.Linear(4, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 2))
    def forward(self, x_norm, x_raw):
        cbf_guess = x_raw.mean(dim=1, keepdim=True) * 500.0
        att_guess = (x_raw.argmax(dim=1, keepdim=True).float() * 0.5 + 1.0)
        cbf_g_n = (cbf_guess - CBF_mean) / CBF_std
        att_g_n = (att_guess - ATT_mean) / ATT_std
        base = torch.cat([cbf_g_n, att_g_n], dim=1)
        return base + self.res_net(x_norm)

results = []
c, a = train_eval_res(ResidualNet().to(device), "Exp_E_Residual")
results.append({'Method': 'DNN Experiment E (Residual)', 'CBF RMSE': c, 'ATT RMSE': a, 'Source': 'Our run'})

try:
    master = pd.read_csv("master_results.csv").to_dict('records')
    master.extend(results)
    pd.DataFrame(master).to_csv("master_results.csv", index=False)
except Exception as e:
    print("Could not update master:", e)

print("Exp E complete.")

