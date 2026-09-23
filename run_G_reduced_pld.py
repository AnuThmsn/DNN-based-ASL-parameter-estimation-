import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data_gen import *
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def calc_metrics(yt, yp):
    rmse = np.sqrt(np.mean((yt - yp)**2))
    mae = np.mean(np.abs(yt - yp))
    return rmse, mae

def standardize(X, m=None, s=None):
    if m is None:
        m = X.mean(axis=0, keepdims=True)
        s = X.std(axis=0, keepdims=True) + 1e-8
    return (X - m) / s, m, s

class AsymmetricNetG(nn.Module):
    def __init__(self, in_dim=4):
        super().__init__()
        self.cbf_net = nn.Sequential(nn.Linear(in_dim, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1))
        self.att_net = nn.Sequential(nn.Linear(in_dim, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1))
    def forward(self, x):
        return torch.cat([self.cbf_net(x), self.att_net(x)], dim=1)

ytr_t = Y_tr_both_t[:400000]

def test_plds(pld_indices, exp_name):
    print(f"Training {exp_name}...", flush=True)
    xtr_sub = X_tr[:400000, pld_indices]
    xte_sub = X_test[:, pld_indices]
    
    xtr_n, m, s = standardize(xtr_sub)
    xte_n, _, _ = standardize(xte_sub, m, s)
    
    xtr_t = torch.tensor(xtr_n, device=device, dtype=torch.float32)
    xte_t = torch.tensor(xte_n, device=device, dtype=torch.float32)
    
    net = AsymmetricNetG(in_dim=len(pld_indices)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.L1Loss()
    bs = 4096
    for ep in range(30):
        net.train()
        perm = torch.randperm(xtr_t.size(0), device=device)
        for i in range(0, xtr_t.size(0), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = crit(net(xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            
    net.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, xte_t.size(0), bs):
            preds.append(net(xte_t[i:i+bs]).cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        
    cbf_p = np.clip(preds[:, 0] * CBF_std + CBF_mean, CBF_MIN, CBF_MAX)
    att_p = np.clip(preds[:, 1] * ATT_std + ATT_mean, ATT_MIN, ATT_MAX)
    
    def generate_snr_arr(N_total, rng, n_noise_levels=51):
        sd_levels = np.linspace(0.0, SD_MAX, n_noise_levels)
        sd_choice_idx = rng.integers(0, n_noise_levels, size=N_total)
        sd_arr = sd_levels[sd_choice_idx]
        with np.errstate(divide='ignore'):
            return np.where(sd_arr > 0, SIG_REF_SC / sd_arr, np.inf)
    snr_test = generate_snr_arr(100_000, np.random.default_rng(7))
    snr_mask_10 = (snr_test >= 9) & (snr_test <= 11)
    
    y_c = Y_test[snr_mask_10, 0]
    y_a = Y_test[snr_mask_10, 1]
    
    c_rmse, c_mae = calc_metrics(y_c, cbf_p[snr_mask_10])
    a_rmse, a_mae = calc_metrics(y_a, att_p[snr_mask_10])
    
    res = {'Experiment': exp_name, 'CBF RMSE': c_rmse, 'ATT RMSE': a_rmse, 'CBF MAE': c_mae, 'ATT MAE': a_mae}
    print(res)
    return res

results = []
# [0,1,2,3] is 4-PLD
results.append(test_plds([0,1,2,3], "4-PLD (Reference)"))
# Drop S1 (first PLD)
results.append(test_plds([1,2,3], "3-PLD (Drop P1)"))
# Drop S2 (second PLD)
results.append(test_plds([0,2,3], "3-PLD (Drop P2)"))
# Drop S4 (last PLD)
results.append(test_plds([0,1,2], "3-PLD (Drop P4)"))

pd.DataFrame(results).to_csv("experiments_G/reduced_pld.csv", index=False)
