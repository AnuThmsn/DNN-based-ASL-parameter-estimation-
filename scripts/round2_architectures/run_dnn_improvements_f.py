import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data_gen import *
import os
from run_experiments_2 import PhysicsLoss, SharedNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def calc_rmse(yt, yp): return np.sqrt(np.mean((yt - yp)**2))

xtr, xval = X_tr, X_val
m = xtr.mean(axis=0, keepdims=True)
s = xtr.std(axis=0, keepdims=True) + 1e-8
xtr_n = (xtr - m) / s
xval_n = (xval - m) / s

xtr_t = torch.tensor(xtr_n, device=device, dtype=torch.float32)
xval_t = torch.tensor(xval_n, device=device, dtype=torch.float32)
ytr_t = Y_tr_both_t
yval_t = Y_val_both_t

results = []
print("Running Exp F (Physics Consistency Sweep)...")
lambdas = [0, 1e-4, 1e-3, 1e-2, 1e-1]
for lmb in lambdas:
    net = SharedNet(4).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    if lmb == 0:
        crit = nn.L1Loss()
    else:
        crit = PhysicsLoss(lmb)
        
    bs = 4096
    for ep in range(30):
        net.train()
        perm = torch.randperm(xtr_t.size(0), device=device)
        for i in range(0, xtr_t.size(0), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            if lmb == 0:
                loss = crit(net(xtr_t[idx]), ytr_t[idx])
            else:
                loss = crit(net(xtr_t[idx]), ytr_t[idx], xtr_t[idx])
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
    results.append({'Experiment': f'Exp F - Physics Lambda {lmb}', 'CBF RMSE': c_rmse, 'ATT RMSE': a_rmse})
    print(f"Exp F (Lmb {lmb}) -> CBF: {c_rmse:.4f}, ATT: {a_rmse:.4f}")

pd.DataFrame(results).to_csv("round2_improvements_f.csv", index=False)

