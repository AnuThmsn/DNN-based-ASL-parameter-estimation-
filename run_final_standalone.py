import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from data_gen import *

# Define network architectures
class StandardizedNet(nn.Module):
    def __init__(self, input_dim, n_hidden_layers, n_neurons, out_dim=1):
        super().__init__()
        layers = [nn.Linear(input_dim, n_neurons), nn.ELU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.ELU()]
        layers.append(nn.Linear(n_neurons, out_dim))
        self.backbone = nn.Sequential(*layers)
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.backbone(x)

class SharedNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU()
        )
        self.cbf_head = nn.Sequential(
            nn.Linear(128, 32), nn.ELU(),
            nn.Linear(32, 1)
        )
        self.att_head = nn.Sequential(
            nn.Linear(128, 32), nn.ELU(),
            nn.Linear(32, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
    def forward(self, x):
        s = self.shared(x)
        c = self.cbf_head(s)
        a = self.att_head(s)
        return torch.cat([c, a], dim=1)

df = pd.read_csv("val_results.csv")
best_overall = df.loc[df['CBF Val RMSE'].idxmin()]
print("Best Configuration:")
print(best_overall, flush=True)

arch_name = best_overall['Architecture']
inp_name = best_overall['Input']
loss_name = best_overall['Loss']
phys_lambda = float(best_overall['Physics lambda'])
opt_name = best_overall['Optimizer']
lr = float(best_overall['LR'])
use_constraint = ("Constrained" in arch_name)
wd = 1e-5 if opt_name.lower() == 'adamw' and lr != 1e-4 else 0

if "Input B" in inp_name: type_name = "B"
elif "Input C" in inp_name: type_name = "C"
elif "Input D" in inp_name: type_name = "D"
else: type_name = "A"

def make_features(X, t):
    S1, S2, S3, S4 = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4]
    eps = 1e-6
    if t == "B":
        f = [X, S2/(S1+eps), S3/(S1+eps), S4/(S1+eps)]
    elif t == "C":
        f = [X, np.log(S1+eps), np.log(S2+eps), np.log(S3+eps), np.log(S4+eps)]
    elif t == "D":
        f = [X, S2/(S1+eps), S3/(S1+eps), S4/(S1+eps), np.log(S1+eps), np.log(S2+eps), np.log(S3+eps), np.log(S4+eps), S2-S1, S3-S2, S4-S3]
    return np.concatenate(f, axis=1).astype(np.float32)

def prepare_input(t):
    Xt = make_features(X_tr, t)
    Xv = make_features(X_val, t)
    m = Xt.mean(axis=0, keepdims=True)
    s = Xt.std(axis=0, keepdims=True) + 1e-8
    Xt_n = (Xt - m) / s
    Xv_n = (Xv - m) / s
    return torch.tensor(Xt_n, device=device), torch.tensor(Xv_n, device=device), Xt.shape[1], m, s

if type_name == "A":
    X_tr_best, X_val_best, best_dim = X_tr_t, X_val_t, 4
    X_test_best = torch.tensor(X_test_n, device=device)
else:
    X_tr_best, X_val_best, best_dim, m, s = prepare_input(type_name)
    X_test_f = make_features(X_test, type_name)
    X_test_best = torch.tensor((X_test_f - m) / s, device=device)

def get_model():
    if "Model D" in arch_name:
        return SharedNet(best_dim)
    else:
        return StandardizedNet(best_dim, 4, 256, out_dim=2)

def train_seed(seed_val):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    net = get_model().to(device)
    
    if opt_name.lower() == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        
    best_val = float("inf")
    best_state = None
    no_imp = 0
    crit = nn.L1Loss()
    y_tr = Y_tr_both_t
    y_val = Y_val_both_t
    
    for epoch in range(60):
        net.train()
        permutation = torch.randperm(X_tr_best.size()[0], device=device)
        for i in range(0, X_tr_best.size()[0], 4096):
            indices = permutation[i:i+4096]
            xb, yb = X_tr_best[indices], y_tr[indices]
            optimizer.zero_grad()
            pred = net(xb)
            loss = crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            
        net.eval()
        with torch.no_grad():
            vl = 0.0
            for i in range(0, X_val_best.size()[0], 4096):
                xb, yb = X_val_best[i:i+4096], y_val[i:i+4096]
                pred = net(xb)
                loss = crit(pred, yb)
                vl += loss.item() * xb.size(0)
            vl /= X_val_best.size(0)
            
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu() for k, v in net.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 15:
                break
    net.load_state_dict(best_state)
    return net

def evaluate_test(net):
    net.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, X_test_best.size()[0], 4096):
            p = net(X_test_best[i:i+4096])
            preds.append(p.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
    
    cbf_pred = preds[:, 0] * CBF_std + CBF_mean
    att_pred = preds[:, 1] * ATT_std + ATT_mean
            
    cbf_pred = np.clip(cbf_pred, CBF_MIN, CBF_MAX)
    att_pred = np.clip(att_pred, ATT_MIN, ATT_MAX)
    
    c_rmse = calc_rmse(Y_test[:, 0], cbf_pred)
    a_rmse = calc_rmse(Y_test[:, 1], att_pred)
    return c_rmse, a_rmse, cbf_pred, att_pred

print("Training Seed 1...", flush=True)
net1 = train_seed(101)
c1, a1, cp1, ap1 = evaluate_test(net1)

print("Training Seed 2...", flush=True)
net2 = train_seed(102)
c2, a2, cp2, ap2 = evaluate_test(net2)

print("Training Seed 3...", flush=True)
net3 = train_seed(103)
c3, a3, cp3, ap3 = evaluate_test(net3)

c_mean = np.mean([c1, c2, c3])
c_std = np.std([c1, c2, c3])
a_mean = np.mean([a1, a2, a3])
a_std = np.std([a1, a2, a3])

print(f"CBF RMSE - Seed1: {c1:.4f}, Seed2: {c2:.4f}, Seed3: {c3:.4f}, Mean: {c_mean:.4f}, Std: {c_std:.4f}")
print(f"ATT RMSE - Seed1: {a1:.4f}, Seed2: {a2:.4f}, Seed3: {a3:.4f}, Mean: {a_mean:.4f}, Std: {a_std:.4f}")

with open("final_metrics.txt", "w") as f:
    f.write(f"Seed1 CBF: {c1:.4f}, ATT: {a1:.4f}\n")
    f.write(f"Seed2 CBF: {c2:.4f}, ATT: {a2:.4f}\n")
    f.write(f"Seed3 CBF: {c3:.4f}, ATT: {a3:.4f}\n")
    f.write(f"Mean CBF: {c_mean:.4f}, Std: {c_std:.4f}\n")
    f.write(f"Mean ATT: {a_mean:.4f}, Std: {a_std:.4f}\n")
    
print("Run final complete.", flush=True)

