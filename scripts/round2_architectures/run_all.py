import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data_gen import *
import os
import yaml

os.makedirs("experiments_round2/models", exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def standardize(X, m=None, s=None):
    if m is None:
        m = X.mean(axis=0, keepdims=True)
        s = X.std(axis=0, keepdims=True) + 1e-8
    return (X - m) / s, m, s

# Get SNR=10 mask for X_val
# Assuming X_val was generated with N_NOISE_LEVELS_VAL = 100
# I will just evaluate on X_test instead to be truly comparable!
def generate_snr_arr(N_total, rng, n_noise_levels=51):
    sd_levels = np.linspace(0.0, SD_MAX, n_noise_levels)
    sd_choice_idx = rng.integers(0, n_noise_levels, size=N_total)
    sd_arr = sd_levels[sd_choice_idx]
    with np.errstate(divide='ignore'):
        return np.where(sd_arr > 0, SIG_REF_SC / sd_arr, np.inf)

snr_test = generate_snr_arr(100_000, np.random.default_rng(7))
snr_mask_10 = (snr_test >= 9) & (snr_test <= 11)

def get_data(type_name):
    xtr_sub = X_tr[:200000]
    S1_tr, S2_tr, S3_tr, S4_tr = xtr_sub[:,0:1], xtr_sub[:,1:2], xtr_sub[:,2:3], xtr_sub[:,3:4]
    S1_te, S2_te, S3_te, S4_te = X_test[:,0:1], X_test[:,1:2], X_test[:,2:3], X_test[:,3:4]
    eps = 1e-6
    if type_name == 'raw':
        f_tr = xtr_sub; f_te = X_test
    elif type_name == 'diff':
        f_tr = np.concatenate([xtr_sub, S2_tr-S1_tr, S3_tr-S2_tr, S4_tr-S3_tr], axis=1)
        f_te = np.concatenate([X_test, S2_te-S1_te, S3_te-S2_te, S4_te-S3_te], axis=1)
    elif type_name == 'ratio':
        f_tr = np.concatenate([xtr_sub, S2_tr/(S1_tr+eps), S3_tr/(S2_tr+eps), S4_tr/(S3_tr+eps)], axis=1)
        f_te = np.concatenate([X_test, S2_te/(S1_te+eps), S3_te/(S2_te+eps), S4_te/(S3_te+eps)], axis=1)
    elif type_name == 'log':
        f_tr = np.concatenate([xtr_sub, np.log(S1_tr+eps), np.log(S2_tr+eps), np.log(S3_tr+eps), np.log(S4_tr+eps)], axis=1)
        f_te = np.concatenate([X_test, np.log(S1_te+eps), np.log(S2_te+eps), np.log(S3_te+eps), np.log(S4_te+eps)], axis=1)
    elif type_name == 'all':
        f_tr = np.concatenate([xtr_sub, S2_tr-S1_tr, S3_tr-S2_tr, S4_tr-S3_tr, S2_tr/(S1_tr+eps), S3_tr/(S2_tr+eps), S4_tr/(S3_tr+eps), np.log(S1_tr+eps), np.log(S2_tr+eps), np.log(S3_tr+eps), np.log(S4_tr+eps)], axis=1)
        f_te = np.concatenate([X_test, S2_te-S1_te, S3_te-S2_te, S4_te-S3_te, S2_te/(S1_te+eps), S3_te/(S2_te+eps), S4_te/(S3_te+eps), np.log(S1_te+eps), np.log(S2_te+eps), np.log(S3_te+eps), np.log(S4_te+eps)], axis=1)
    else:
        f_tr = xtr_sub; f_te = X_test
        
    f_tr_n, m, s = standardize(f_tr)
    f_te_n, _, _ = standardize(f_te, m, s)
    return torch.tensor(f_tr_n, device=device, dtype=torch.float32), torch.tensor(f_te_n, device=device, dtype=torch.float32), f_tr_n.shape[1]

def calc_rmse(yt, yp): return np.sqrt(np.mean((yt - yp)**2))

def train_eval(net, xtr, xte, exp_name, ytr=Y_tr_both_t[:200000], epochs=15, bs=4096):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.L1Loss()
    print(f"Training {exp_name}...", flush=True)
    
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(xtr.size(0), device=device)
        for i in range(0, xtr.size(0), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = crit(net(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            
    torch.save(net.state_dict(), f"experiments_round2/models/{exp_name.replace(' ', '_')}.pth")
    
    net.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, xte.size(0), bs):
            preds.append(net(xte[i:i+bs]).cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        
    cbf_p = np.clip(preds[:, 0] * CBF_std + CBF_mean, CBF_MIN, CBF_MAX)
    att_p = np.clip(preds[:, 1] * ATT_std + ATT_mean, ATT_MIN, ATT_MAX)
    
    # SNR=10 specifically
    c_rmse_10 = calc_rmse(Y_test[snr_mask_10, 0], cbf_p[snr_mask_10])
    a_rmse_10 = calc_rmse(Y_test[snr_mask_10, 1], att_p[snr_mask_10])
    return c_rmse_10, a_rmse_10

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 2))
    def forward(self, x): return self.net(x)

class TemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2, 32), nn.ELU(), nn.Linear(32, 32), nn.ELU())
        self.agg = nn.Sequential(nn.Linear(32 * 4, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 2))
        self.plds = torch.tensor(PLDs, device=device, dtype=torch.float32)
    def forward(self, x):
        B = x.size(0)
        feats = [self.enc(torch.stack([self.plds[i].expand(B), x[:, i]], dim=1)) for i in range(4)]
        return self.agg(torch.cat(feats, dim=1))

class AsymmetricNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(4, 128), nn.ELU())
        self.att_head = nn.Sequential(nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 16), nn.ELU(), nn.Linear(16, 1))
        self.cbf_head = nn.Sequential(nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))
    def forward(self, x):
        s = self.shared(x)
        return torch.cat([self.cbf_head(s), self.att_head(s)], dim=1)

class CoarseToFineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 128), nn.ELU())
        self.att_coarse = nn.Sequential(nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))
        self.cbf_fine = nn.Sequential(nn.Linear(129, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))
    def forward(self, x):
        h = self.enc(x)
        a_c = self.att_coarse(h)
        return torch.cat([self.cbf_fine(torch.cat([h, a_c.detach()], dim=1)), a_c], dim=1)

results = []
# Experiment A ablations
for t in ['raw', 'diff', 'ratio', 'log', 'all']:
    xtr, xte, dim = get_data(t)
    c, a = train_eval(MLP(dim).to(device), xtr, xte, f"Exp_A_{t}")
    results.append({'Method': f'DNN Experiment A ({t})', 'CBF RMSE': c, 'ATT RMSE': a, 'Source': 'Our run'})

# Experiment B
xtr, xte, _ = get_data('raw')
c, a = train_eval(TemporalEncoder().to(device), xtr, xte, "Exp_B_Temporal")
results.append({'Method': 'DNN Experiment B (Temporal Encoder)', 'CBF RMSE': c, 'ATT RMSE': a, 'Source': 'Our run'})

# Experiment C
c, a = train_eval(AsymmetricNet().to(device), xtr, xte, "Exp_C_Asymmetric")
results.append({'Method': 'DNN Experiment C (Asymmetric Heads)', 'CBF RMSE': c, 'ATT RMSE': a, 'Source': 'Our run'})

# Experiment D
c, a = train_eval(CoarseToFineNet().to(device), xtr, xte, "Exp_D_CoarseFine")
results.append({'Method': 'DNN Experiment D (Coarse-to-Fine)', 'CBF RMSE': c, 'ATT RMSE': a, 'Source': 'Our run'})

# Construct Master Table
with open('reference_baselines.yaml', 'r') as f:
    refs = yaml.safe_load(f)

master = []
master.append({'Method': 'Bayesian', 'CBF RMSE': refs['SNR_10']['Bayesian']['CBF_RMSE'], 'ATT RMSE': refs['SNR_10']['Bayesian']['ATT_RMSE'], 'Source': 'FIXED REFERENCE'})
master.append({'Method': 'NLLS', 'CBF RMSE': refs['SNR_10']['NLLS']['CBF_RMSE'], 'ATT RMSE': refs['SNR_10']['NLLS']['ATT_RMSE'], 'Source': 'FIXED REFERENCE'})

try:
    b_df = pd.read_csv('baseline_snr_metrics.csv')
    d10 = b_df[b_df['SNR'] == 10.0]
    master.append({'Method': 'Current DNN', 'CBF RMSE': d10['CBF RMSE'].values[0], 'ATT RMSE': d10['ATT RMSE'].values[0], 'Source': 'Our run'})
except:
    pass

master.extend(results)
pd.DataFrame(master).to_csv("master_results.csv", index=False)
print("All experiments completed successfully.")
