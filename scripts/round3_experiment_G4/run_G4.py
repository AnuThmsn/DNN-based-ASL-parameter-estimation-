import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from data_gen import (
    X_tr, X_val, X_test,
    Y_tr, Y_val, Y_test,
    Y_tr_both_t, Y_val_both_t,
    CBF_mean, CBF_std, ATT_mean, ATT_std,
    CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX,
    PLDs, SD_MAX, SIG_REF_SC,
    device, calc_rmse, calc_mae, calc_bias
)

# --- Configuration ---
EPS = 1e-6
N_SUB = 400_000
BS = 4096
LR = 1e-3
SEEDS = [42, 43, 44]

OUT_DIR = "experiments_G4"
MODEL_DIR = f"{OUT_DIR}/models"
DIAG_DIR = f"{OUT_DIR}/diagnostics"
PLOT_DIR = f"{OUT_DIR}/plots"

# --- Feature Preparation (G1_LOG) ---
def make_log_features(X_raw):
    return np.concatenate([X_raw, np.log(X_raw.astype(np.float64) + EPS)], axis=1).astype(np.float32)

F_tr_raw = make_log_features(X_tr[:N_SUB])
F_val_raw = make_log_features(X_val)
F_te_raw = make_log_features(X_test)

F_mu = F_tr_raw.mean(0, keepdims=True)
F_sig = F_tr_raw.std(0, keepdims=True) + EPS

F_tr_n = ((F_tr_raw - F_mu) / F_sig).astype(np.float32)
F_val_n = ((F_val_raw - F_mu) / F_sig).astype(np.float32)
F_te_n = ((F_te_raw - F_mu) / F_sig).astype(np.float32)

xtr_t = torch.tensor(F_tr_n, device=device)
ytr_t = Y_tr_both_t[:N_SUB]
xval_t = torch.tensor(F_val_n, device=device)
xte_t = torch.tensor(F_te_n, device=device)

# --- Model Definition ---
def make_mlp(dims):
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims)-2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)

class G1_LOG_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cbf = make_mlp([8, 128, 128, 128, 128, 128, 1])
        self.att = make_mlp([8, 64,  64,  64,  64,  1])
    def forward(self, x):
        return torch.cat([self.cbf(x), self.att(x)], dim=1)

def decode(preds):
    cbf = np.clip(preds[:, 0] * CBF_std + CBF_mean, CBF_MIN, CBF_MAX)
    att = np.clip(preds[:, 1] * ATT_std + ATT_mean, ATT_MIN, ATT_MAX)
    return cbf, att

# --- Evaluation Helpers ---
CBF_BINS = [(0,20), (20,40), (40,60), (60,80), (80,100)]
def evaluate_model(model, name, seed):
    model.eval()
    with torch.no_grad():
        chunks = [model(xte_t[j:j+BS]).cpu().numpy() for j in range(0, xte_t.size(0), BS)]
    raw_preds = np.concatenate(chunks, 0)
    cbf_p, att_p = decode(raw_preds)
    
    y_cbf, y_att = Y_test[:,0], Y_test[:,1]
    
    overall = {
        "cbf_rmse": calc_rmse(y_cbf, cbf_p),
        "cbf_mae": calc_mae(y_cbf, cbf_p),
        "att_rmse": calc_rmse(y_att, att_p),
        "att_mae": calc_mae(y_att, att_p),
    }
    
    # Boundary bias
    low_mask = (y_cbf >= 0) & (y_cbf < 20)
    high_mask = (y_cbf >= 80) & (y_cbf < 100)
    mid_mask = (y_cbf >= 20) & (y_cbf < 80)
    overall["low_cbf_bias"] = calc_bias(y_cbf[low_mask], cbf_p[low_mask])
    overall["high_cbf_bias"] = calc_bias(y_cbf[high_mask], cbf_p[high_mask])
    overall["mid_cbf_bias"] = calc_bias(y_cbf[mid_mask], cbf_p[mid_mask])
    
    return overall, cbf_p, att_p

def snr_analysis(cbf_p, att_p, name):
    def _snr_arr(N, rng, n=51):
        sd = np.linspace(0, SD_MAX, n)[rng.integers(0, n, N)]
        with np.errstate(divide="ignore"):
            return np.where(sd > 0, SIG_REF_SC / sd, np.inf)
    SNR_TEST = _snr_arr(100_000, np.random.default_rng(7))
    
    y_cbf, y_att = Y_test[:,0], Y_test[:,1]
    snr_rows = []
    for snr in [10, 15, 20, 30, 40]:
        mask = (SNR_TEST >= snr*0.9) & (SNR_TEST <= snr*1.1)
        if mask.sum() < 50: continue
        snr_rows.append({
            "SNR": snr,
            "CBF_RMSE": calc_rmse(y_cbf[mask], cbf_p[mask]),
            "ATT_RMSE": calc_rmse(y_att[mask], att_p[mask])
        })
    pd.DataFrame(snr_rows).to_csv(f"{DIAG_DIR}/{name}_snr.csv", index=False)

def bin_analysis(cbf_p, att_p, name):
    y_cbf, y_att = Y_test[:,0], Y_test[:,1]
    bin_rows = []
    for lo, hi in CBF_BINS:
        m = (y_cbf >= lo) & (y_cbf < hi)
        if m.sum() == 0: continue
        bin_rows.append({
            "bin": f"{lo}-{hi}", "N": int(m.sum()),
            "RMSE": calc_rmse(y_cbf[m], cbf_p[m]),
            "MAE": calc_mae(y_cbf[m], cbf_p[m]),
            "Bias": calc_bias(y_cbf[m], cbf_p[m]),
            "mean_pred": cbf_p[m].mean(),
            "mean_true": y_cbf[m].mean()
        })
    pd.DataFrame(bin_rows).to_csv(f"{DIAG_DIR}/{name}_cbf_bins.csv", index=False)

# --- Training Loop ---
def train_model(name, epochs, seed=42, lr_schedule=None):
    torch.manual_seed(seed)
    model = G1_LOG_Net().to(device)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.L1Loss()
    
    scheduler = None
    if lr_schedule == 'plateau':
        # Monitor val loss
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=False)
    elif lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    history = []
    t0 = time.time()
    
    best_val_cbf_rmse = float('inf')
    best_ep = -1
    
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(xtr_t.size(0), device=device)
        ep_loss = 0.0
        n_batches = 0
        
        for i in range(0, xtr_t.size(0), BS):
            idx = perm[i:i+BS]
            opt.zero_grad()
            out = model(xtr_t[idx])
            loss = crit(out, ytr_t[idx])
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
            
        train_loss = ep_loss / n_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(xval_t)
            val_loss = crit(val_out, Y_val_both_t).item()
            
            cbf_p, att_p = decode(val_out.cpu().numpy())
            y_val_cbf, y_val_att = Y_val[:,0], Y_val[:,1]
            
            val_cbf_rmse = calc_rmse(y_val_cbf, cbf_p)
            val_att_rmse = calc_rmse(y_val_att, att_p)
        
        current_lr = opt.param_groups[0]['lr']
        
        history.append({
            "epoch": ep + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_cbf_rmse": val_cbf_rmse,
            "val_att_rmse": val_att_rmse,
            "lr": current_lr
        })
        
        if val_cbf_rmse < best_val_cbf_rmse:
            best_val_cbf_rmse = val_cbf_rmse
            best_ep = ep + 1
            
        if lr_schedule == 'plateau':
            scheduler.step(val_loss)
        elif lr_schedule == 'cosine':
            scheduler.step()
            
    torch.save(model.state_dict(), f"{MODEL_DIR}/{name}_s{seed}.pth")
    
    elapsed = time.time() - t0
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(f"{DIAG_DIR}/{name}_s{seed}_history.csv", index=False)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss")
    axes[0].plot(hist_df["epoch"], hist_df["val_loss"], label="Val Loss")
    axes[0].axvline(best_ep, color='r', linestyle='--', alpha=0.5, label='Best Val')
    axes[0].set_title(f"{name} Loss")
    axes[0].legend()
    
    axes[1].plot(hist_df["epoch"], hist_df["val_cbf_rmse"])
    axes[1].axvline(best_ep, color='r', linestyle='--', alpha=0.5)
    axes[1].set_title("Val CBF RMSE")
    
    axes[2].plot(hist_df["epoch"], hist_df["val_att_rmse"])
    axes[2].axvline(best_ep, color='r', linestyle='--', alpha=0.5)
    axes[2].set_title("Val ATT RMSE")
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{name}_s{seed}_convergence.png")
    plt.close()
    
    ov, cbf_p, att_p = evaluate_model(model, name, seed)
    print(f"[{name:15s} s{seed}] CBF RMSE={ov['cbf_rmse']:.4f} ATT RMSE={ov['att_rmse']:.4f} "
          f"(Best Ep: {best_ep}, {elapsed:.0f}s)")
    
    return ov, cbf_p, att_p

# --- Main Execution ---
print("=== Experiment G4: Convergence and LR Study ===")
results = []

configs = [
    ("G4_A", 30, None, "30 epochs (reference)"),
    ("G4_B", 60, None, "60 epochs (fixed LR)"),
    ("G4_C", 100, None, "100 epochs (fixed LR)"),
    ("G4_E", 100, 'plateau', "100 epochs (ReduceLROnPlateau)"),
    ("G4_F", 100, 'cosine', "100 epochs (CosineAnnealingLR)")
]

best_cbf_rmse = float('inf')
best_model_name = None

for name, eps, sch, desc in configs:
    ov, cbf_p, att_p = train_model(name, eps, seed=42, lr_schedule=sch)
    results.append({"name": name, "description": desc, "seed": 42, **ov})
    
    if ov['cbf_rmse'] < best_cbf_rmse:
        best_cbf_rmse = ov['cbf_rmse']
        best_model_name = name
        # Save bins and snr for this best so far, though we might overwrite or just do it for the final winner
        bin_analysis(cbf_p, att_p, name)
        snr_analysis(cbf_p, att_p, name)

# Save screening results
master_df = pd.DataFrame(results)
master_df.to_csv(f"{OUT_DIR}/master_results_G4_screening.csv", index=False)

# 3-seed verification
print(f"\nBest candidate from screening: {best_model_name}")
# Get the config for the best model
winner_conf = next(c for c in configs if c[0] == best_model_name)

seed_rows = []
for s in SEEDS:
    # If s=42, we already have it, but we can re-train or just load. Let's re-train for cleaner code
    ov, _, _ = train_model(f"{best_model_name}_FINAL", winner_conf[1], seed=s, lr_schedule=winner_conf[2])
    seed_rows.append({"name": f"{best_model_name}_FINAL", "seed": s, **ov})

seed_df = pd.DataFrame(seed_rows)
print(f"\n3-Seed Verification of {best_model_name}:")
print(f"CBF RMSE: {seed_df['cbf_rmse'].mean():.4f} ± {seed_df['cbf_rmse'].std():.4f}")
print(f"ATT RMSE: {seed_df['att_rmse'].mean():.4f} ± {seed_df['att_rmse'].std():.4f}")
print(f"Low CBF Bias: {seed_df['low_cbf_bias'].mean():.4f} ± {seed_df['low_cbf_bias'].std():.4f}")
print(f"High CBF Bias: {seed_df['high_cbf_bias'].mean():.4f} ± {seed_df['high_cbf_bias'].std():.4f}")

# Append to master
master_df = pd.concat([master_df, seed_df], ignore_index=True)

# Add decision
def classify(row):
    if row['seed'] != 42 or 'FINAL' in row['name']: return ""
    if row['cbf_rmse'] > master_df[master_df.name == 'G4_A']['cbf_rmse'].values[0] - 0.02:
        if row['name'] in ['G4_B', 'G4_C']:
            return "NO CONVERGENCE GAIN"
        return "REJECT"
    else:
        return "CONVERGENCE EVIDENCE / PROMISING"

master_df['decision'] = master_df.apply(classify, axis=1)

# Add 3-seed mean summary row
mean_row = seed_df.mean(numeric_only=True).to_dict()
mean_row['name'] = f"{best_model_name}_3SEED_MEAN"
mean_row['description'] = "3-seed average"
mean_row['decision'] = "TRAINING-PROTOCOL IMPROVEMENT" if mean_row['cbf_rmse'] < master_df[master_df.name == 'G4_A']['cbf_rmse'].values[0] else "NO CONVERGENCE GAIN"
master_df = pd.concat([master_df, pd.DataFrame([mean_row])], ignore_index=True)

master_df.to_csv(f"{OUT_DIR}/master_results_G4.csv", index=False)
print(f"\nResults saved to {OUT_DIR}/master_results_G4.csv")

