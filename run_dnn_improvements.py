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

def get_data(type_name):
    S1_tr, S2_tr, S3_tr, S4_tr = X_tr[:,0:1], X_tr[:,1:2], X_tr[:,2:3], X_tr[:,3:4]
    S1_va, S2_va, S3_va, S4_va = X_val[:,0:1], X_val[:,1:2], X_val[:,2:3], X_val[:,3:4]
    
    eps = 1e-6
    if type_name == 'raw':
        f_tr = X_tr
        f_va = X_val
    elif type_name == 'diff':
        f_tr = np.concatenate([X_tr, S2_tr-S1_tr, S3_tr-S2_tr, S4_tr-S3_tr], axis=1)
        f_va = np.concatenate([X_val, S2_va-S1_va, S3_va-S2_va, S4_va-S3_va], axis=1)
    elif type_name == 'ratio':
        f_tr = np.concatenate([X_tr, S2_tr/(S1_tr+eps), S3_tr/(S2_tr+eps), S4_tr/(S3_tr+eps)], axis=1)
        f_va = np.concatenate([X_val, S2_va/(S1_va+eps), S3_va/(S2_va+eps), S4_va/(S3_va+eps)], axis=1)
    elif type_name == 'log':
        f_tr = np.concatenate([X_tr, np.log(S1_tr+eps), np.log(S2_tr+eps), np.log(S3_tr+eps), np.log(S4_tr+eps)], axis=1)
        f_va = np.concatenate([X_val, np.log(S1_va+eps), np.log(S2_va+eps), np.log(S3_va+eps), np.log(S4_va+eps)], axis=1)
    elif type_name == 'all':
        f_tr = np.concatenate([X_tr, S2_tr-S1_tr, S3_tr-S2_tr, S4_tr-S3_tr, S2_tr/(S1_tr+eps), S3_tr/(S2_tr+eps), S4_tr/(S3_tr+eps), np.log(S1_tr+eps), np.log(S2_tr+eps), np.log(S3_tr+eps), np.log(S4_tr+eps)], axis=1)
        f_va = np.concatenate([X_val, S2_va-S1_va, S3_va-S2_va, S4_va-S3_va, S2_va/(S1_va+eps), S3_va/(S2_va+eps), S4_va/(S3_va+eps), np.log(S1_va+eps), np.log(S2_va+eps), np.log(S3_va+eps), np.log(S4_va+eps)], axis=1)
    else:
        f_tr = X_tr
        f_va = X_val
        
    f_tr_n, m, s = standardize(f_tr)
    f_va_n, _, _ = standardize(f_va, m, s)
    return torch.tensor(f_tr_n, device=device, dtype=torch.float32), torch.tensor(f_va_n, device=device, dtype=torch.float32), f_tr_n.shape[1]

def calc_rmse(yt, yp): return np.sqrt(np.mean((yt - yp)**2))

def train_eval(net, xtr, xval, ytr=Y_tr_both_t, yval=Y_val_both_t, epochs=40, bs=4096):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.L1Loss()
    best_loss = float('inf')
    best_state = None
    
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(xtr.size(0), device=device)
        for i in range(0, xtr.size(0), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = crit(net(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            
        net.eval()
        with torch.no_grad():
            vl = crit(net(xval), yval).item()
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.cpu() for k, v in net.state_dict().items()}
            
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, xval.size(0), bs):
            preds.append(net(xval[i:i+bs]).cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        
    cbf_p = preds[:, 0] * CBF_std + CBF_mean
    att_p = preds[:, 1] * ATT_std + ATT_mean
    cbf_p = np.clip(cbf_p, CBF_MIN, CBF_MAX)
    att_p = np.clip(att_p, ATT_MIN, ATT_MAX)
    
    return calc_rmse(Y_val[:,0], cbf_p), calc_rmse(Y_val[:,1], att_p)

# Base MLP
class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)

results = []

print("Running Exp A (Ablation)...")
for t in ['raw', 'diff', 'ratio', 'log', 'all']:
    xtr, xval, dim = get_data(t)
    net = MLP(dim).to(device)
    c_rmse, a_rmse = train_eval(net, xtr, xval)
    results.append({'Experiment': f'Exp A - Input {t}', 'CBF RMSE': c_rmse, 'ATT RMSE': a_rmse})
    print(f"Exp A ({t}) -> CBF: {c_rmse:.4f}, ATT: {a_rmse:.4f}")

print("Running Exp B (Temporal/PLD Encoder)...")
class TemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # input: (PLD, S_i)
        self.enc = nn.Sequential(nn.Linear(2, 32), nn.ELU(), nn.Linear(32, 32), nn.ELU())
        self.agg = nn.Sequential(nn.Linear(32 * 4, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 2))
        self.plds = torch.tensor(PLDs, device=device, dtype=torch.float32)
    def forward(self, x):
        # x is (B, 4)
        B = x.size(0)
        feats = []
        for i in range(4):
            inp = torch.stack([self.plds[i].expand(B), x[:, i]], dim=1)
            feats.append(self.enc(inp))
        out = torch.cat(feats, dim=1)
        return self.agg(out)

xtr, xval, _ = get_data('raw')
net = TemporalEncoder().to(device)
c_rmse, a_rmse = train_eval(net, xtr, xval)
results.append({'Experiment': 'Exp B - Temporal Encoder', 'CBF RMSE': c_rmse, 'ATT RMSE': a_rmse})
print(f"Exp B -> CBF: {c_rmse:.4f}, ATT: {a_rmse:.4f}")

print("Running Exp C (Separate asymmetric reps)...")
class AsymmetricNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(4, 128), nn.ELU())
        # ATT needs deeper temporal processing
        self.att_head = nn.Sequential(nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 16), nn.ELU(), nn.Linear(16, 1))
        # CBF is more amplitude based, simpler
        self.cbf_head = nn.Sequential(nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))
    def forward(self, x):
        s = self.shared(x)
        return torch.cat([self.cbf_head(s), self.att_head(s)], dim=1)

net = AsymmetricNet().to(device)
c_rmse, a_rmse = train_eval(net, xtr, xval)
results.append({'Experiment': 'Exp C - Asymmetric Heads', 'CBF RMSE': c_rmse, 'ATT RMSE': a_rmse})
print(f"Exp C -> CBF: {c_rmse:.4f}, ATT: {a_rmse:.4f}")

pd.DataFrame(results).to_csv("round2_improvements.csv", index=False)
