import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data_gen import *
import os
def standardize(X, m=None, s=None):
    if m is None:
        m = X.mean(axis=0, keepdims=True)
        s = X.std(axis=0, keepdims=True) + 1e-8
    return (X - m) / s, m, s

def get_data(type_name):
    xtr_sub = X_tr[:400000]
    S1_tr, S2_tr, S3_tr, S4_tr = xtr_sub[:,0:1], xtr_sub[:,1:2], xtr_sub[:,2:3], xtr_sub[:,3:4]
    S1_te, S2_te, S3_te, S4_te = X_test[:,0:1], X_test[:,1:2], X_test[:,2:3], X_test[:,3:4]
    eps = 1e-6
    f_tr = xtr_sub; f_te = X_test
    f_tr_n, m, s = standardize(f_tr)
    f_te_n, _, _ = standardize(f_te, m, s)
    return torch.tensor(f_tr_n, device=device, dtype=torch.float32), torch.tensor(f_te_n, device=device, dtype=torch.float32), f_tr_n.shape[1]

ytr_t = Y_tr_both_t[:400000]

def calc_metrics(yt, yp):
    rmse = np.sqrt(np.mean((yt - yp)**2))
    mae = np.mean(np.abs(yt - yp))
    return rmse, mae

def train_eval(net, xtr, xte, exp_name, epochs=30, bs=4096):
    print(f"Training {exp_name}...", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.L1Loss()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(xtr.size(0), device=device)
        for i in range(0, xtr.size(0), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = crit(net(xtr[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            
    torch.save(net.state_dict(), f"experiments_G/models/{exp_name}.pth")
    net.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, xte.size(0), bs):
            preds.append(net(xte[i:i+bs]).cpu().numpy())
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
    
    return {'Experiment': exp_name, 'CBF RMSE': c_rmse, 'ATT RMSE': a_rmse, 'CBF MAE': c_mae, 'ATT MAE': a_mae}

class G1_PLDAware(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2, 32), nn.ELU(), nn.Linear(32, 32), nn.ELU())
        self.plds = torch.tensor(PLDs, device=device, dtype=torch.float32)
        
        self.cbf = nn.Sequential(nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1))
        self.att = nn.Sequential(nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1))
        
    def forward(self, x):
        B = x.size(0)
        feats = [self.enc(torch.stack([self.plds[i].expand(B), x[:, i]], dim=1)) for i in range(4)]
        agg = torch.cat(feats, dim=1)
        return torch.cat([self.cbf(agg), self.att(agg)], dim=1)

class G2_SeparateReps(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2, 32), nn.ELU(), nn.Linear(32, 32), nn.ELU())
        self.plds = torch.tensor(PLDs, device=device, dtype=torch.float32)
        self.shared = nn.Sequential(nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU())
        
        # Reduced branches because shared takes up 2 layers
        self.cbf = nn.Sequential(nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1))
        self.att = nn.Sequential(nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1))
        
    def forward(self, x):
        B = x.size(0)
        feats = [self.enc(torch.stack([self.plds[i].expand(B), x[:, i]], dim=1)) for i in range(4)]
        agg = self.shared(torch.cat(feats, dim=1))
        return torch.cat([self.cbf(agg), self.att(agg)], dim=1)

class G4_TargetCap(nn.Module):
    def __init__(self, att_dim):
        super().__init__()
        self.cbf = nn.Sequential(nn.Linear(4, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1))
        self.att = nn.Sequential(nn.Linear(4, att_dim), nn.ELU(), nn.Linear(att_dim, att_dim), nn.ELU(), nn.Linear(att_dim, att_dim), nn.ELU(), nn.Linear(att_dim, att_dim), nn.ELU(), nn.Linear(att_dim, 1))
    def forward(self, x): return torch.cat([self.cbf(x), self.att(x)], dim=1)

xtr, xte, _ = get_data('raw')

results = []
res = train_eval(G1_PLDAware().to(device), xtr, xte, "G1_PLDAware")
results.append(res); print(res)

res = train_eval(G2_SeparateReps().to(device), xtr, xte, "G2_SeparateReps")
results.append(res); print(res)

for dim in [32, 96, 128]:
    res = train_eval(G4_TargetCap(dim).to(device), xtr, xte, f"G4_TargetCap_{dim}")
    results.append(res); print(res)

pd.DataFrame(results).to_csv("experiments_G/improvements_G124.csv", index=False)
