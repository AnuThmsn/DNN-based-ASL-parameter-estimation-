import os
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from data_gen import *

print("Starting PART 2: Clean Baseline Evaluation", flush=True)

SEED_BASELINE = 123
np.random.seed(SEED_BASELINE)
torch.manual_seed(SEED_BASELINE)

# The baseline architectures
class StandardizedNet(nn.Module):
    def __init__(self, input_dim, n_hidden_layers, n_neurons, y_mean, y_std, y_min, y_max):
        super().__init__()
        self.y_mean = y_mean
        self.y_std  = y_std
        self.y_min  = y_min
        self.y_max  = y_max
        layers = [nn.Linear(input_dim, n_neurons), nn.ELU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.ELU()]
        layers.append(nn.Linear(n_neurons, 1))
        self.backbone = nn.Sequential(*layers)
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.backbone(x)

    def predict_physical(self, x):
        with torch.no_grad():
            std_pred = self.forward(x).squeeze(-1)
        phys_pred = std_pred * self.y_std + self.y_mean
        return phys_pred # return raw physical to check failure rate before clamping

# Initialize the baseline models
cbf_net = StandardizedNet(4, 9, 50, CBF_mean, CBF_std, CBF_MIN, CBF_MAX).to(device)
att_net = StandardizedNet(4, 9, 100, ATT_mean, ATT_std, ATT_MIN, ATT_MAX).to(device)

def train_baseline(net, x_tr, y_tr, x_val, y_val, epochs=30, batch_size=4096, lr=1e-3, patience=5):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.L1Loss() # the baseline notebook used MAE
    
    best_val = float('inf')
    best_state = None
    no_imp = 0
    
    for epoch in range(epochs):
        net.train()
        permutation = torch.randperm(x_tr.size(0), device=device)
        for i in range(0, x_tr.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            xb, yb = x_tr[indices], y_tr[indices]
            optimizer.zero_grad()
            pred = net(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, x_val.size(0), batch_size):
                xb, yb = x_val[i:i+batch_size], y_val[i:i+batch_size]
                pred = net(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= x_val.size(0)
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in net.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    
    net.load_state_dict(best_state)
    return net

print("Training baseline CBF network...", flush=True)
cbf_net = train_baseline(cbf_net, X_tr_t, Y_tr_cbf_t, X_val_t, Y_val_cbf_t)

print("Training baseline ATT network...", flush=True)
att_net = train_baseline(att_net, X_tr_t, Y_tr_att_t, X_val_t, Y_val_att_t)

# Inference on Test Set
cbf_net.eval()
att_net.eval()

with torch.no_grad():
    preds_c = []
    preds_a = []
    X_test_t = torch.tensor(X_test_n, device=device)
    for i in range(0, X_test_t.size(0), 4096):
        xb = X_test_t[i:i+4096]
        preds_c.append(cbf_net.predict_physical(xb).cpu().numpy())
        preds_a.append(att_net.predict_physical(xb).cpu().numpy())
        
    cbf_pred_raw = np.concatenate(preds_c, axis=0)
    att_pred_raw = np.concatenate(preds_a, axis=0)

# Failure rate: % outside physical bounds before clamping
cbf_fails = np.mean((cbf_pred_raw < CBF_MIN) | (cbf_pred_raw > CBF_MAX)) * 100.0
att_fails = np.mean((att_pred_raw < ATT_MIN) | (att_pred_raw > ATT_MAX)) * 100.0
overall_fails = np.mean((cbf_pred_raw < CBF_MIN) | (cbf_pred_raw > CBF_MAX) | 
                        (att_pred_raw < ATT_MIN) | (att_pred_raw > ATT_MAX)) * 100.0

cbf_pred = np.clip(cbf_pred_raw, CBF_MIN, CBF_MAX)
att_pred = np.clip(att_pred_raw, ATT_MIN, ATT_MAX)

y_test_cbf = Y_test[:, 0]
y_test_att = Y_test[:, 1]

def calc_all_metrics(true, pred):
    err = pred - true
    abs_err = np.abs(err)
    mse = np.mean(err**2)
    mae = np.mean(abs_err)
    rmse = np.sqrt(mse)
    
    # normalized by range
    rng = np.max(true) - np.min(true)
    nmae = mae / rng if rng > 0 else 0
    nrmse = rmse / rng if rng > 0 else 0
    
    # MdAE and MdB
    mdae = np.median(abs_err)
    mdb = np.median(err)
    
    # RCV (Repeatability Coefficient of Variation) proxy? Wait, RCV usually involves variance of error
    # Let's use 1.96 * sqrt(2) * std(err) / mean(true) ? Or just std(err)/mean(true)
    rcv = np.std(err) / np.mean(true) * 100.0 if np.mean(true) > 0 else 0
    
    return {
        "MSE": mse, "MAE": mae, "RMSE": rmse, "NMAE": nmae, "NRMSE": nrmse,
        "MdAE": mdae, "MdB": mdb, "RCV": rcv
    }

metrics_cbf = calc_all_metrics(y_test_cbf, cbf_pred)
metrics_att = calc_all_metrics(y_test_att, att_pred)

print(f"\n--- OVERALL METRICS ---")
print(f"CBF Fails: {cbf_fails:.2f}%, ATT Fails: {att_fails:.2f}%, Overall: {overall_fails:.2f}%")
print(f"CBF RMSE: {metrics_cbf['RMSE']:.4f}, MAE: {metrics_cbf['MAE']:.4f}, MdAE: {metrics_cbf['MdAE']:.4f}")
print(f"ATT RMSE: {metrics_att['RMSE']:.4f}, MAE: {metrics_att['MAE']:.4f}, MdAE: {metrics_att['MdAE']:.4f}")

# SNR Specific
snr_bins = [5, 10, 20, 50, 100, 500]
snr_results = []

# We need to recreate test set with SNR tracking
def generate_test_with_snr(N_total, rng, n_noise_levels=51):
    cbf_gt = rng.uniform(CBF_MIN, CBF_MAX, N_total).astype(np.float64)
    att_gt = rng.uniform(ATT_MIN, ATT_MAX, N_total).astype(np.float64)
    sd_levels = np.linspace(0.0, SD_MAX, n_noise_levels)
    sd_choice_idx = rng.integers(0, n_noise_levels, size=N_total)
    sd_arr = sd_levels[sd_choice_idx]
    with np.errstate(divide='ignore'):
        snr_arr = np.where(sd_arr > 0, SIG_REF_SC / sd_arr, np.inf)
    sig = compute_signals_vec(cbf_gt, att_gt) * SCALE
    sd = sd_arr[:, None]
    e1 = rng.normal(0.0, 1.0, size=sig.shape) * sd
    e2 = rng.normal(0.0, 1.0, size=sig.shape) * sd
    e3 = rng.normal(0.0, 1.0, size=sig.shape) * sd
    e4 = rng.normal(0.0, 1.0, size=sig.shape) * sd
    mc = np.sqrt((sig + e1) ** 2 + e2 ** 2)
    ml = np.sqrt((sig + e3) ** 2 + e4 ** 2)
    X = (mc + ml).astype(np.float32)
    Y = np.stack([cbf_gt, att_gt], axis=1).astype(np.float32)
    return X, Y, snr_arr

rng_test2 = np.random.default_rng(7)
X_test_new, Y_test_new, snr_test = generate_test_with_snr(N_TEST, rng_test2)
X_test_n_new = (X_test_new - X_mean) / X_std

with torch.no_grad():
    preds_c_new = []
    preds_a_new = []
    X_test_t_new = torch.tensor(X_test_n_new, device=device)
    for i in range(0, X_test_t_new.size(0), 4096):
        xb = X_test_t_new[i:i+4096]
        preds_c_new.append(cbf_net.predict_physical(xb).cpu().numpy())
        preds_a_new.append(att_net.predict_physical(xb).cpu().numpy())
        
    cbf_pred_raw_new = np.concatenate(preds_c_new, axis=0)
    att_pred_raw_new = np.concatenate(preds_a_new, axis=0)

cbf_pred_new = np.clip(cbf_pred_raw_new, CBF_MIN, CBF_MAX)
att_pred_new = np.clip(att_pred_raw_new, ATT_MIN, ATT_MAX)
y_test_cbf_new = Y_test_new[:, 0]
y_test_att_new = Y_test_new[:, 1]

for s in snr_bins:
    mask = (snr_test >= s * 0.9) & (snr_test <= s * 1.1)
    if mask.sum() == 0:
        continue
    c_met = calc_all_metrics(y_test_cbf_new[mask], cbf_pred_new[mask])
    a_met = calc_all_metrics(y_test_att_new[mask], att_pred_new[mask])
    c_fail = np.mean((cbf_pred_raw_new[mask] < CBF_MIN) | (cbf_pred_raw_new[mask] > CBF_MAX)) * 100.0
    a_fail = np.mean((att_pred_raw_new[mask] < ATT_MIN) | (att_pred_raw_new[mask] > ATT_MAX)) * 100.0
    snr_results.append({
        "SNR": s, 
        "CBF_RMSE": c_met["RMSE"], "CBF_MdAE": c_met["MdAE"], "CBF_MdB": c_met["MdB"], "CBF_Fail": c_fail,
        "ATT_RMSE": a_met["RMSE"], "ATT_MdAE": a_met["MdAE"], "ATT_MdB": a_met["MdB"], "ATT_Fail": a_fail
    })

df_snr = pd.DataFrame(snr_results)
df_snr.to_csv("baseline_snr_metrics.csv", index=False)
print("\n--- SNR METRICS ---")
print(df_snr.to_string())

# Save baseline for future comparison
os.makedirs("experiments/exp01_baseline", exist_ok=True)
torch.save(cbf_net.state_dict(), "experiments/exp01_baseline/cbf_net.pth")
torch.save(att_net.state_dict(), "experiments/exp01_baseline/att_net.pth")
np.save("experiments/exp01_baseline/cbf_pred.npy", cbf_pred_new)
np.save("experiments/exp01_baseline/att_pred.npy", att_pred_new)

print("\nDone Phase 2.", flush=True)
