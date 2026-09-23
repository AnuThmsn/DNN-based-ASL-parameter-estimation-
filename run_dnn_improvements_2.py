import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data_gen import *
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def standardize(X, m=None, s=None):
    if m is None:
        m = X.mean(axis=0, keepdims=True)
        s = X.std(axis=0, keepdims=True) + 1e-8
    return (X - m) / s, m, s

xtr, xval = X_tr, X_val
xtr_n, m, s = standardize(xtr)
xval_n, _, _ = standardize(xval, m, s)
xtr_t = torch.tensor(xtr_n, device=device, dtype=torch.float32)
xval_t = torch.tensor(xval_n, device=device, dtype=torch.float32)

def calc_rmse(yt, yp): return np.sqrt(np.mean((yt - yp)**2))

results = []

print("Running Exp D (Coarse-to-Fine)...")
class CoarseToFineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 128), nn.ELU())
        self.att_coarse = nn.Sequential(nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))
        # Conditioned on ATT coarse estimate
        self.cbf_fine = nn.Sequential(nn.Linear(129, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))
    def forward(self, x):
        h = self.enc(x)
        a_coarse = self.att_coarse(h)
        h_cond = torch.cat([h, a_coarse.detach()], dim=1) # Detach to prevent gradients from CBF ruining ATT
        c_fine = self.cbf_fine(h_cond)
        return torch.cat([c_fine, a_coarse], dim=1)

net = CoarseToFineNet().to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
crit = nn.L1Loss()
bs = 4096
for ep in range(40):
    net.train()
    perm = torch.randperm(xtr_t.size(0), device=device)
    for i in range(0, xtr_t.size(0), bs):
        idx = perm[i:i+bs]
        opt.zero_grad()
        loss = crit(net(xtr_t[idx]), Y_tr_both_t[idx])
        loss.backward()
        opt.step()
        
net.eval()
with torch.no_grad():
    preds = []
    for i in range(0, xval_t.size(0), bs):
        preds.append(net(xval_t[i:i+bs]).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    
cbf_p = np.clip(preds[:, 0] * CBF_std + CBF_mean, CBF_MIN, CBF_MAX)
att_p = np.clip(preds[:, 1] * ATT_std + ATT_mean, ATT_MIN, ATT_MAX)
c_rmse = calc_rmse(Y_val[:,0], cbf_p)
a_rmse = calc_rmse(Y_val[:,1], att_p)
results.append({'Experiment': 'Exp D - Coarse-to-Fine', 'CBF RMSE': c_rmse, 'ATT RMSE': a_rmse})
print(f"Exp D -> CBF: {c_rmse:.4f}, ATT: {a_rmse:.4f}")

print("Running Exp E (Residual Parameter Est)...")
# We will use a fast analytical guess for CBF: CBF ~ max(S) * constant
# And ATT ~ argmax(S) * constant. We can learn the offsets.
class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_net = nn.Sequential(nn.Linear(4, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 2))
    def forward(self, x_norm, x_raw):
        # x_raw is the original unnormalized signal
        # Fast analytical guess
        # Just use mean of S as proxy for CBF, and index of max as proxy for ATT
        cbf_guess = x_raw.mean(dim=1, keepdim=True) * 500.0
        att_guess = (x_raw.argmax(dim=1, keepdim=True).float() * 0.5 + 1.0)
        # Normalize the guesses
        cbf_g_n = (cbf_guess - CBF_mean) / CBF_std
        att_g_n = (att_guess - ATT_mean) / ATT_std
        base = torch.cat([cbf_g_n, att_g_n], dim=1)
        
        delta = self.res_net(x_norm)
        return base + delta

xtr_raw_t = torch.tensor(xtr, device=device, dtype=torch.float32)
xval_raw_t = torch.tensor(xval, device=device, dtype=torch.float32)

net = ResidualNet().to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
for ep in range(40):
    net.train()
    perm = torch.randperm(xtr_t.size(0), device=device)
    for i in range(0, xtr_t.size(0), bs):
        idx = perm[i:i+bs]
        opt.zero_grad()
        loss = crit(net(xtr_t[idx], xtr_raw_t[idx]), Y_tr_both_t[idx])
        loss.backward()
        opt.step()
        
net.eval()
with torch.no_grad():
    preds = []
    for i in range(0, xval_t.size(0), bs):
        preds.append(net(xval_t[i:i+bs], xval_raw_t[i:i+bs]).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

cbf_p = np.clip(preds[:, 0] * CBF_std + CBF_mean, CBF_MIN, CBF_MAX)
att_p = np.clip(preds[:, 1] * ATT_std + ATT_mean, ATT_MIN, ATT_MAX)
c_rmse = calc_rmse(Y_val[:,0], cbf_p)
a_rmse = calc_rmse(Y_val[:,1], att_p)
results.append({'Experiment': 'Exp E - Residual', 'CBF RMSE': c_rmse, 'ATT RMSE': a_rmse})
print(f"Exp E -> CBF: {c_rmse:.4f}, ATT: {a_rmse:.4f}")

pd.DataFrame(results).to_csv("round2_improvements_2.csv", index=False)
