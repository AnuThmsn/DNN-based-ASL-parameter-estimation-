import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_gen import *
from run_experiments_2 import get_best_arch, prepare_input, PhysicsLoss, ConstrainedNet

# Read val_results
df = pd.read_csv("val_results.csv")
best_overall = df.loc[df['CBF Val RMSE'].idxmin()]

print("Best Configuration:")
print(best_overall)

# Extract best config
arch_name = best_overall['Architecture']
inp_name = best_overall['Input']
loss_name = best_overall['Loss']
phys_lambda = float(best_overall['Physics lambda'])
opt_name = best_overall['Optimizer']
lr = float(best_overall['LR'])
use_constraint = ("Constrained" in arch_name)
wd = 1e-5 if opt_name.lower() == 'adamw' and lr != 1e-4 else 0

if "Input B" in inp_name:
    type_name = "B"
elif "Input C" in inp_name:
    type_name = "C"
elif "Input D" in inp_name:
    type_name = "D"
else:
    type_name = "A"

if type_name == "A":
    X_tr_best, X_val_best, best_dim = X_tr_t, X_val_t, 4
    X_test_best = torch.tensor(X_test_n, device=device)
else:
    X_tr_best, X_val_best, best_dim, m, s = prepare_input(type_name)
    X_test_f = make_features(X_test, type_name)
    X_test_best = torch.tensor((X_test_f - m) / s, device=device)
    
def get_model():
    n, a = get_best_arch(best_dim)
    if use_constraint:
        return ConstrainedNet(n)
    return n

if use_constraint:
    if phys_lambda > 0:
        crit = PhysicsLoss(phys_lambda, is_physical_pred=True)
    else:
        crit = nn.L1Loss()
else:
    if phys_lambda > 0:
        crit = PhysicsLoss(phys_lambda)
    else:
        crit = nn.L1Loss()

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
    
    y_tr = torch.tensor(Y_tr, device=device) if use_constraint else Y_tr_both_t
    y_val = torch.tensor(Y_val, device=device) if use_constraint else Y_val_both_t
    
    for epoch in range(60):
        net.train()
        permutation = torch.randperm(X_tr_best.size()[0], device=device)
        for i in range(0, X_tr_best.size()[0], 4096):
            indices = permutation[i:i+4096]
            xb, yb = X_tr_best[indices], y_tr[indices]
            optimizer.zero_grad()
            pred = net(xb)
            loss = crit(pred, yb, xb) if hasattr(crit, 'is_physics') else crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            
        net.eval()
        with torch.no_grad():
            vl = 0.0
            for i in range(0, X_val_best.size()[0], 4096):
                xb, yb = X_val_best[i:i+4096], y_val[i:i+4096]
                pred = net(xb)
                loss = crit(pred, yb, xb) if hasattr(crit, 'is_physics') else crit(pred, yb)
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
    
    if use_constraint:
        cbf_pred = preds[:, 0]
        att_pred = preds[:, 1]
    else:
        cbf_pred = preds[:, 0] * CBF_std + CBF_mean
        att_pred = preds[:, 1] * ATT_std + ATT_mean
            
    cbf_pred = np.clip(cbf_pred, CBF_MIN, CBF_MAX)
    att_pred = np.clip(att_pred, ATT_MIN, ATT_MAX)
    
    c_rmse = calc_rmse(Y_test[:, 0], cbf_pred)
    a_rmse = calc_rmse(Y_test[:, 1], att_pred)
    return c_rmse, a_rmse, cbf_pred, att_pred

print("Training Seed 1...")
net1 = train_seed(101)
c1, a1, cp1, ap1 = evaluate_test(net1)
torch.save(net1.state_dict(), "best_model_seed1.pth")

print("Training Seed 2...")
net2 = train_seed(102)
c2, a2, cp2, ap2 = evaluate_test(net2)
torch.save(net2.state_dict(), "best_model_seed2.pth")

print("Training Seed 3...")
net3 = train_seed(103)
c3, a3, cp3, ap3 = evaluate_test(net3)
torch.save(net3.state_dict(), "best_model_seed3.pth")

c_mean = np.mean([c1, c2, c3])
c_std = np.std([c1, c2, c3])
a_mean = np.mean([a1, a2, a3])
a_std = np.std([a1, a2, a3])

print(f"CBF RMSE - Seed1: {c1:.4f}, Seed2: {c2:.4f}, Seed3: {c3:.4f}, Mean: {c_mean:.4f}, Std: {c_std:.4f}")
print(f"ATT RMSE - Seed1: {a1:.4f}, Seed2: {a2:.4f}, Seed3: {a3:.4f}, Mean: {a_mean:.4f}, Std: {a_std:.4f}")

# Phase 2 Residual plots (for Baseline Model A)
cbf_net0 = StandardizedNet(4, 9, 50).to(device).eval().requires_grad_(False)
cbf_net0.load_state_dict(torch.load("asl_dnn_papermatched/CBF_papermatched_9x50_MAE_best.pth", map_location=device))
att_net0 = StandardizedNet(4, 9, 100).to(device).eval().requires_grad_(False)
att_net0.load_state_dict(torch.load("asl_dnn_papermatched/ATT_papermatched_9x100_MAE_best.pth", map_location=device))

with torch.no_grad():
    preds_c = []
    preds_a = []
    for i in range(0, X_test_n.shape[0], 4096):
        x = torch.tensor(X_test_n[i:i+4096], device=device)
        preds_c.append(cbf_net0(x).cpu().numpy())
        preds_a.append(att_net0(x).cpu().numpy())
    cbf_pred0 = np.concatenate(preds_c, axis=0).squeeze() * CBF_std + CBF_mean
    att_pred0 = np.concatenate(preds_a, axis=0).squeeze() * ATT_std + ATT_mean
    cbf_pred0 = np.clip(cbf_pred0, CBF_MIN, CBF_MAX)
    att_pred0 = np.clip(att_pred0, ATT_MIN, ATT_MAX)

cbf_bins = [(0,20), (20,40), (40,60), (60,80), (80,100)]
att_bins = [(0.5,1.0), (1.0,1.5), (1.5,2.0), (2.0,2.5), (2.5,3.0)]

cbf_res = []
for b in cbf_bins:
    mask = (Y_test[:,0] >= b[0]) & (Y_test[:,0] < b[1])
    yt = Y_test[mask, 0]
    yp = cbf_pred0[mask]
    rmse = calc_rmse(yt, yp)
    mae = calc_mae(yt, yp)
    bias = calc_bias(yt, yp)
    cbf_res.append({"Bin": f"{b[0]}-{b[1]}", "Samples": mask.sum(), "RMSE": rmse, "MAE": mae, "Bias": bias})

att_res = []
for b in att_bins:
    mask = (Y_test[:,1] >= b[0]) & (Y_test[:,1] < b[1])
    yt = Y_test[mask, 1]
    yp = att_pred0[mask]
    rmse = calc_rmse(yt, yp)
    mae = calc_mae(yt, yp)
    bias = calc_bias(yt, yp)
    att_res.append({"Bin": f"{b[0]}-{b[1]}", "Samples": mask.sum(), "RMSE": rmse, "MAE": mae, "Bias": bias})

pd.DataFrame(cbf_res).to_csv("cbf_residuals.csv", index=False)
pd.DataFrame(att_res).to_csv("att_residuals.csv", index=False)

# Make plots
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.scatter(Y_test[:,0], cbf_pred0 - Y_test[:,0], alpha=0.1, s=1)
plt.axhline(0, color='r')
plt.title("Baseline CBF Residuals")
plt.xlabel("True CBF")
plt.ylabel("Error")
plt.subplot(122)
plt.scatter(Y_test[:,1], att_pred0 - Y_test[:,1], alpha=0.1, s=1)
plt.axhline(0, color='r')
plt.title("Baseline ATT Residuals")
plt.xlabel("True ATT")
plt.ylabel("Error")
plt.savefig("asl_dnn_papermatched/plots/baseline_residuals.png")

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.scatter(Y_test[:,0], cp1 - Y_test[:,0], alpha=0.1, s=1)
plt.axhline(0, color='r')
plt.title("Best Model CBF Residuals")
plt.xlabel("True CBF")
plt.ylabel("Error")
plt.subplot(122)
plt.scatter(Y_test[:,1], ap1 - Y_test[:,1], alpha=0.1, s=1)
plt.axhline(0, color='r')
plt.title("Best Model ATT Residuals")
plt.xlabel("True ATT")
plt.ylabel("Error")
plt.savefig("asl_dnn_papermatched/plots/best_residuals.png")

with open("final_metrics.txt", "w") as f:
    f.write(f"Seed1 CBF: {c1:.4f}, ATT: {a1:.4f}\n")
    f.write(f"Seed2 CBF: {c2:.4f}, ATT: {a2:.4f}\n")
    f.write(f"Seed3 CBF: {c3:.4f}, ATT: {a3:.4f}\n")
    f.write(f"Mean CBF: {c_mean:.4f}, Std: {c_std:.4f}\n")
    f.write(f"Mean ATT: {a_mean:.4f}, Std: {a_std:.4f}\n")
    
    # Distributions
    f.write(f"\nBaseline CBF True: Mean={np.mean(Y_test[:,0]):.4f}, Std={np.std(Y_test[:,0]):.4f}, Min={np.min(Y_test[:,0]):.4f}, Max={np.max(Y_test[:,0]):.4f}\n")
    f.write(f"Baseline CBF Pred: Mean={np.mean(cbf_pred0):.4f}, Std={np.std(cbf_pred0):.4f}, Min={np.min(cbf_pred0):.4f}, Max={np.max(cbf_pred0):.4f}\n")
    
    f.write(f"\nBaseline ATT True: Mean={np.mean(Y_test[:,1]):.4f}, Std={np.std(Y_test[:,1]):.4f}, Min={np.min(Y_test[:,1]):.4f}, Max={np.max(Y_test[:,1]):.4f}\n")
    f.write(f"Baseline ATT Pred: Mean={np.mean(att_pred0):.4f}, Std={np.std(att_pred0):.4f}, Min={np.min(att_pred0):.4f}, Max={np.max(att_pred0):.4f}\n")

print("Run final complete.", flush=True)

