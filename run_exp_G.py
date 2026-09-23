import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data_gen import *
import os

os.makedirs("experiments_G/models", exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def standardize(X, m=None, s=None):
    if m is None:
        m = X.mean(axis=0, keepdims=True)
        s = X.std(axis=0, keepdims=True) + 1e-8
    return (X - m) / s, m, s

xtr_sub = X_tr[:400000]
xtr_n, m, s = standardize(xtr_sub)
xva_n, _, _ = standardize(X_val, m, s)
xte_n, _, _ = standardize(X_test, m, s)

xtr_t = torch.tensor(xtr_n, device=device, dtype=torch.float32)
ytr_t = Y_tr_both_t[:400000]
xva_t = torch.tensor(xva_n, device=device, dtype=torch.float32)
yva_t = Y_val_both_t
xte_t = torch.tensor(xte_n, device=device, dtype=torch.float32)

class AsymmetricNetG(nn.Module):
    def __init__(self):
        super().__init__()
        self.cbf_net = nn.Sequential(
            nn.Linear(4, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            # Wait, 5 hidden layers: input -> 128 -> 128 -> 128 -> 128 -> 128 -> 1
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, 1)
        )
        self.att_net = nn.Sequential(
            nn.Linear(4, 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return torch.cat([self.cbf_net(x), self.att_net(x)], dim=1)

def calc_metrics(yt, yp):
    rmse = np.sqrt(np.mean((yt - yp)**2))
    mae = np.mean(np.abs(yt - yp))
    cc = np.corrcoef(yt, yp)[0, 1]
    r2 = cc**2
    return rmse, mae, r2

epochs = 30
bs = 4096

print("Training BASELINE-G across 3 seeds...")
all_preds = []

for seed in [42, 43, 44]:
    print(f"Seed {seed}...")
    torch.manual_seed(seed)
    net = AsymmetricNetG().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.L1Loss()
    
    best_val_loss = float('inf')
    for ep in range(epochs):
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
            vloss = crit(net(xva_t), yva_t).item()
        if vloss < best_val_loss:
            best_val_loss = vloss
            torch.save(net.state_dict(), f"experiments_G/models/baseline_G_seed{seed}.pth")
            
    net.load_state_dict(torch.load(f"experiments_G/models/baseline_G_seed{seed}.pth", weights_only=True))
    net.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, xte_t.size(0), bs):
            preds.append(net(xte_t[i:i+bs]).cpu().numpy())
        preds = np.concatenate(preds, axis=0)
    
    cbf_p = np.clip(preds[:, 0] * CBF_std + CBF_mean, CBF_MIN, CBF_MAX)
    att_p = np.clip(preds[:, 1] * ATT_std + ATT_mean, ATT_MIN, ATT_MAX)
    all_preds.append(np.stack([cbf_p, att_p], axis=1))

all_preds = np.array(all_preds) # (3, N, 2)
np.save("experiments_G/baseline_G_preds_seeds.npy", all_preds)
mean_preds = np.mean(all_preds, axis=0)
np.save("experiments_G/baseline_G_preds_mean.npy", mean_preds)

y_cbf = Y_test[:, 0]
y_att = Y_test[:, 1]

c_rmse, c_mae, c_r2 = calc_metrics(y_cbf, mean_preds[:, 0])
a_rmse, a_mae, a_r2 = calc_metrics(y_att, mean_preds[:, 1])

res = pd.DataFrame([{
    'Method': 'BASELINE-G (Mean)',
    'CBF RMSE': c_rmse, 'CBF MAE': c_mae, 'CBF R2': c_r2,
    'ATT RMSE': a_rmse, 'ATT MAE': a_mae, 'ATT R2': a_r2
}])
res.to_csv("experiments_G/baseline_G_metrics.csv", index=False)
print("BASELINE-G Training Complete.")
print(res)
