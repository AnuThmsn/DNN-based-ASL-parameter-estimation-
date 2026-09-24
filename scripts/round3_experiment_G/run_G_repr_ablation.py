import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
"""
Experiment G — Controlled Representation Ablation Study
========================================================
Parent model: Experiment G (CBF: 4→128x5→1, ATT: 4→64x4→1)

Candidates:
  G_BASELINE  — raw [S1,S2,S3,S4]
  G1_LOG      — raw + log features (8-d input)
  G2_DIFF     — raw + consecutive differences (7-d input)
  G3_RATIO    — raw + consecutive ratios (7-d input)
  G4_COMBINED — raw + log + diff + ratio (14-d input)
  G5_PLD_AWARE — explicit (PLD_i, S_i) shared encoder + asymmetric heads

Protocol:
  - Identical simulator, test set, target normalisation
  - Training subset: 400 000 samples (speed), 30 epochs, batch 4096, Adam lr=1e-3, L1 loss
  - Evaluated at FULL test set (100 000), metrics reported in original physical units
  - SNR breakdown at 10/15/20/30/40
  - 3-seed verification for the winning candidate
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os, time

from data_gen import (
    X_tr, X_val, X_test,
    Y_tr, Y_val, Y_test,
    Y_tr_both_t, Y_val_both_t,
    CBF_mean, CBF_std, ATT_mean, ATT_std,
    CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX,
    PLDs, SD_MAX, SIG_REF_SC,
    device, calc_rmse, calc_mae, calc_bias
)

os.makedirs("experiments_G2/models", exist_ok=True)
os.makedirs("experiments_G2/diagnostics", exist_ok=True)

EPS     = 1e-6
N_SUB   = 400_000
EPOCHS  = 30
BS      = 4096
LR      = 1e-3
SEEDS   = [42, 43, 44]

# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

def _snr_arr(N, rng, n_levels=51):
    sd = np.linspace(0, SD_MAX, n_levels)[rng.integers(0, n_levels, N)]
    with np.errstate(divide="ignore"):
        return np.where(sd > 0, SIG_REF_SC / sd, np.inf)

SNR_TEST = _snr_arr(100_000, np.random.default_rng(7))

def standardize_from_train(X_raw_tr, X_raw_te):
    m = X_raw_tr.mean(0, keepdims=True)
    s = X_raw_tr.std(0,  keepdims=True) + EPS
    return (X_raw_tr - m) / s, (X_raw_te - m) / s

def to_tensor(arr):
    return torch.tensor(arr.astype(np.float32), device=device)

def check_features(X, name):
    assert not np.any(np.isnan(X)), f"{name}: NaN detected"
    assert not np.any(np.isinf(X)), f"{name}: Inf detected"
    print(f"  {name:30s} shape={X.shape}  min={X.min():.3f}  max={X.max():.3f}  OK")

# ──────────────────────────────────────────────
# feature builders
# ──────────────────────────────────────────────

def feat_raw(X):
    return X

def feat_log(X):
    return np.concatenate([X, np.log(X + EPS)], axis=1)

def feat_diff(X):
    diffs = np.diff(X, axis=1)          # shape (N, 3)
    return np.concatenate([X, diffs], axis=1)

def feat_ratio(X):
    ratios = X[:, 1:] / (X[:, :-1] + EPS)   # shape (N, 3)
    return np.concatenate([X, ratios], axis=1)

def feat_combined(X):
    return np.concatenate([
        X,
        np.log(X + EPS),
        np.diff(X, axis=1),
        X[:, 1:] / (X[:, :-1] + EPS),
    ], axis=1)

FEATURE_BUILDERS = {
    "G_BASELINE":  feat_raw,
    "G1_LOG":      feat_log,
    "G2_DIFF":     feat_diff,
    "G3_RATIO":    feat_ratio,
    "G4_COMBINED": feat_combined,
}

# ──────────────────────────────────────────────
# models
# ──────────────────────────────────────────────

def make_mlp(dims, act=nn.ELU):
    layers = []
    for i in range(len(dims)-1):
        layers += [nn.Linear(dims[i], dims[i+1])]
        if i < len(dims)-2:
            layers += [act()]
    return nn.Sequential(*layers)

class AsymmetricG(nn.Module):
    """Standard G architecture with configurable input dim."""
    def __init__(self, in_dim):
        super().__init__()
        self.cbf = make_mlp([in_dim, 128, 128, 128, 128, 128, 1])
        self.att = make_mlp([in_dim, 64,  64,  64,  64,  1])

    def forward(self, x):
        return torch.cat([self.cbf(x), self.att(x)], dim=1)


class PLDAwareEncoder(nn.Module):
    """
    Explicitly encodes each (PLD_i, S_i) pair through a shared small MLP,
    concatenates the four embeddings, then passes through asymmetric heads.
    """
    def __init__(self, enc_hidden=32):
        super().__init__()
        self.pld_vals = torch.tensor(PLDs, dtype=torch.float32, device=device)
        # shared encoder: input dim = 2 (pld, signal), output dim = enc_hidden
        self.encoder = make_mlp([2, 64, enc_hidden])
        agg_dim = 4 * enc_hidden
        self.cbf = make_mlp([agg_dim, 128, 128, 128, 128, 128, 1])
        self.att = make_mlp([agg_dim, 64,  64,  64,  64,  1])

    def forward(self, x):
        B = x.size(0)
        # x: (B, 4) — normalised signals
        feats = []
        for i in range(4):
            pld_i = self.pld_vals[i].expand(B, 1)    # (B, 1)
            s_i   = x[:, i:i+1]                       # (B, 1)
            pair  = torch.cat([pld_i, s_i], dim=1)    # (B, 2)
            feats.append(self.encoder(pair))           # (B, enc_hidden)
        z = torch.cat(feats, dim=1)                    # (B, 4*enc_hidden)
        return torch.cat([self.cbf(z), self.att(z)], dim=1)

# ──────────────────────────────────────────────
# train / evaluate
# ──────────────────────────────────────────────

def decode_predictions(raw_preds):
    cbf_p = np.clip(raw_preds[:, 0] * CBF_std + CBF_mean, CBF_MIN, CBF_MAX)
    att_p = np.clip(raw_preds[:, 1] * ATT_std + ATT_mean, ATT_MIN, ATT_MAX)
    return cbf_p, att_p

def full_metrics(cbf_p, att_p):
    y_cbf = Y_test[:, 0]
    y_att = Y_test[:, 1]
    return dict(
        cbf_rmse = calc_rmse(y_cbf, cbf_p),
        cbf_mae  = calc_mae (y_cbf, cbf_p),
        cbf_bias = calc_bias(y_cbf, cbf_p),
        att_rmse = calc_rmse(y_att, att_p),
        att_mae  = calc_mae (y_att, att_p),
        att_bias = calc_bias(y_att, att_p),
    )

def snr_metrics(cbf_p, att_p):
    y_cbf = Y_test[:, 0]
    y_att = Y_test[:, 1]
    rows = []
    for snr in [10, 15, 20, 30, 40]:
        mask = (SNR_TEST >= snr*0.9) & (SNR_TEST <= snr*1.1)
        if mask.sum() < 50:
            continue
        rows.append(dict(
            SNR      = snr,
            CBF_RMSE = calc_rmse(y_cbf[mask], cbf_p[mask]),
            ATT_RMSE = calc_rmse(y_att[mask], att_p[mask]),
        ))
    return rows

def bin_metrics(cbf_p, att_p):
    y_cbf = Y_test[:, 0]
    y_att = Y_test[:, 1]
    cbf_rows, att_rows = [], []
    for lo, hi in [(0,20),(20,40),(40,60),(60,80),(80,100)]:
        m = (y_cbf >= lo) & (y_cbf < hi)
        if m.sum() == 0: continue
        cbf_rows.append(dict(range=f"{lo}-{hi}", N=m.sum(),
            RMSE=calc_rmse(y_cbf[m],cbf_p[m]), MAE=calc_mae(y_cbf[m],cbf_p[m]),
            Bias=calc_bias(y_cbf[m],cbf_p[m])))
    for lo, hi in [(0.5,1.0),(1.0,1.5),(1.5,2.0),(2.0,2.5),(2.5,3.0)]:
        m = (y_att >= lo) & (y_att < hi)
        if m.sum() == 0: continue
        att_rows.append(dict(range=f"{lo}-{hi}", N=m.sum(),
            RMSE=calc_rmse(y_att[m],att_p[m]), MAE=calc_mae(y_att[m],att_p[m]),
            Bias=calc_bias(y_att[m],att_p[m])))
    return cbf_rows, att_rows

def train_and_eval(model, xtr_t, xte_t, name, seed=42):
    torch.manual_seed(seed)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.L1Loss()
    y_tr = Y_tr_both_t[:N_SUB]

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(xtr_t.size(0), device=device)
        for i in range(0, xtr_t.size(0), BS):
            idx = perm[i:i+BS]
            opt.zero_grad()
            loss = crit(model(xtr_t[idx]), y_tr[idx])
            loss.backward()
            opt.step()

    torch.save(model.state_dict(), f"experiments_G2/models/{name}_s{seed}.pth")

    model.eval()
    with torch.no_grad():
        chunks = [model(xte_t[i:i+BS]).cpu().numpy()
                  for i in range(0, xte_t.size(0), BS)]
    raw_preds = np.concatenate(chunks, axis=0)
    cbf_p, att_p = decode_predictions(raw_preds)
    m = full_metrics(cbf_p, att_p)
    print(f"  [{name:20s} seed={seed}]  "
          f"CBF_RMSE={m['cbf_rmse']:.4f}  ATT_RMSE={m['att_rmse']:.4f}"
          f"  ({time.time()-t0:.0f}s)")
    return cbf_p, att_p, m

# ──────────────────────────────────────────────
# PHASE 1: Diagnostic — feature statistics
# ──────────────────────────────────────────────

print("\n=== PHASE 1: Feature Diagnostics ===")
X_sub = X_tr[:N_SUB]
for fname, fn in FEATURE_BUILDERS.items():
    F = fn(X_sub)
    check_features(F, fname)
F_pld_raw = feat_raw(X_sub)
check_features(F_pld_raw, "G5_PLD_AWARE (raw input)")

# ──────────────────────────────────────────────
# PHASE 2: Train all flat-MLP feature candidates
# ──────────────────────────────────────────────

print("\n=== PHASE 2: Flat-MLP Feature Ablations (seed=42) ===")
summary_rows = []

for name, feat_fn in FEATURE_BUILDERS.items():
    F_tr_raw = feat_fn(X_tr[:N_SUB])
    F_te_raw = feat_fn(X_test)
    # normalise from training stats
    m_f = F_tr_raw.mean(0, keepdims=True).astype(np.float32)
    s_f = F_tr_raw.std(0,  keepdims=True).astype(np.float32) + EPS
    xtr_t = to_tensor((F_tr_raw - m_f) / s_f)
    xte_t = to_tensor((F_te_raw - m_f) / s_f)

    in_dim = xtr_t.shape[1]
    model  = AsymmetricG(in_dim).to(device)
    print(f"\n--- {name}  (input dim={in_dim}) ---")
    cbf_p, att_p, met = train_and_eval(model, xtr_t, xte_t, name, seed=42)

    # save predictions for diagnostics
    np.save(f"experiments_G2/diagnostics/{name}_cbf.npy", cbf_p)
    np.save(f"experiments_G2/diagnostics/{name}_att.npy", att_p)
    summary_rows.append({"Model": name, "InputDim": in_dim, **met})

# ──────────────────────────────────────────────
# PHASE 3: PLD-Aware encoder
# ──────────────────────────────────────────────

print("\n=== PHASE 3: PLD-Aware Encoder (seed=42) ===")
# PLD-aware always uses raw signals; normalisation from raw training subset
m_raw = X_tr[:N_SUB].mean(0, keepdims=True).astype(np.float32)
s_raw = X_tr[:N_SUB].std(0,  keepdims=True).astype(np.float32) + EPS
xtr_raw_t = to_tensor((X_tr[:N_SUB] - m_raw) / s_raw)
xte_raw_t = to_tensor((X_test       - m_raw) / s_raw)

for enc_h in [32, 64]:
    name = f"G5_PLD_AWARE_enc{enc_h}"
    print(f"\n--- {name} ---")
    model = PLDAwareEncoder(enc_hidden=enc_h).to(device)
    cbf_p, att_p, met = train_and_eval(model, xtr_raw_t, xte_raw_t, name, seed=42)
    np.save(f"experiments_G2/diagnostics/{name}_cbf.npy", cbf_p)
    np.save(f"experiments_G2/diagnostics/{name}_att.npy", att_p)
    summary_rows.append({"Model": name, "InputDim": 2, **met})

# ──────────────────────────────────────────────
# PHASE 4: 3-seed verification of best candidate
# ──────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("experiments_G2/phase2_3_summary.csv", index=False)
print("\n=== PHASE 2+3 Summary ===")
print(summary_df[["Model","cbf_rmse","att_rmse","cbf_mae","att_mae"]].to_string(index=False))

# Select winner by lowest CBF+ATT joint metric (sum of normalised RMSE)
cbf_ref = summary_df.loc[summary_df["Model"]=="G_BASELINE","cbf_rmse"].values[0]
att_ref = summary_df.loc[summary_df["Model"]=="G_BASELINE","att_rmse"].values[0]
summary_df["score"] = (summary_df["cbf_rmse"]/cbf_ref + summary_df["att_rmse"]/att_ref)
winner = summary_df.loc[summary_df["score"].idxmin(), "Model"]
print(f"\n>>> Candidate selected for 3-seed verification: {winner}")

# Rebuild feature/model for winner
def rebuild_winner(name, seed):
    if name.startswith("G5_PLD_AWARE"):
        enc_h = int(name.split("enc")[-1])
        model = PLDAwareEncoder(enc_hidden=enc_h).to(device)
        xtr_t = xtr_raw_t
        xte_t = xte_raw_t
        return model, xtr_t, xte_t
    else:
        feat_fn = FEATURE_BUILDERS[name]
        F_tr = feat_fn(X_tr[:N_SUB])
        F_te = feat_fn(X_test)
        m_ = F_tr.mean(0, keepdims=True).astype(np.float32)
        s_ = F_tr.std(0,  keepdims=True).astype(np.float32) + EPS
        xtr_t = to_tensor((F_tr - m_) / s_)
        xte_t = to_tensor((F_te - m_) / s_)
        model = AsymmetricG(xtr_t.shape[1]).to(device)
        return model, xtr_t, xte_t

print(f"\n=== PHASE 4: 3-Seed Verification of {winner} ===")
seed_rows = []
cbf_preds_seeds, att_preds_seeds = [], []
for seed in SEEDS:
    model, xtr_t, xte_t = rebuild_winner(winner, seed)
    cbf_p, att_p, met   = train_and_eval(model, xtr_t, xte_t, f"{winner}_FINAL", seed=seed)
    cbf_preds_seeds.append(cbf_p)
    att_preds_seeds.append(att_p)
    seed_rows.append(met)

seed_df = pd.DataFrame(seed_rows)
print("\n--- Seed results ---")
print(seed_df[["cbf_rmse","att_rmse"]].to_string(index=False))
print(f"\nCBF RMSE  {seed_df['cbf_rmse'].mean():.4f} ± {seed_df['cbf_rmse'].std():.4f}")
print(f"ATT RMSE  {seed_df['att_rmse'].mean():.4f} ± {seed_df['att_rmse'].std():.4f}")

# Mean prediction across seeds
cbf_mean = np.mean(cbf_preds_seeds, axis=0)
att_mean = np.mean(att_preds_seeds, axis=0)
np.save(f"experiments_G2/diagnostics/WINNER_cbf.npy", cbf_mean)
np.save(f"experiments_G2/diagnostics/WINNER_att.npy", att_mean)

# ──────────────────────────────────────────────
# PHASE 5: Detailed diagnostics for winner
# ──────────────────────────────────────────────

print(f"\n=== PHASE 5: Error Analysis for winner ({winner}) ===")
cbf_bin_rows, att_bin_rows = bin_metrics(cbf_mean, att_mean)
snr_rows = snr_metrics(cbf_mean, att_mean)

pd.DataFrame(cbf_bin_rows).to_csv("experiments_G2/diagnostics/winner_cbf_bins.csv", index=False)
pd.DataFrame(att_bin_rows).to_csv("experiments_G2/diagnostics/winner_att_bins.csv", index=False)
pd.DataFrame(snr_rows    ).to_csv("experiments_G2/diagnostics/winner_snr.csv",       index=False)

print("\nCBF by range:")
print(pd.DataFrame(cbf_bin_rows).to_string(index=False))
print("\nATT by range:")
print(pd.DataFrame(att_bin_rows).to_string(index=False))
print("\nSNR breakdown:")
print(pd.DataFrame(snr_rows).to_string(index=False))

# ──────────────────────────────────────────────
# FINAL: Master results table
# ──────────────────────────────────────────────

ref = {"Model":"G_BASELINE_PREV",
       "cbf_rmse":4.673,"att_rmse":0.369,
       "cbf_mae":3.386,"att_mae":0.235,
       "cbf_bias":None,"att_bias":None}

master = summary_df[["Model","cbf_rmse","att_rmse","cbf_mae","att_mae","cbf_bias","att_bias"]].copy()
winner_row = {"Model": f"{winner}_3SEED_MEAN",
              "cbf_rmse": seed_df['cbf_rmse'].mean(),
              "att_rmse": seed_df['att_rmse'].mean(),
              "cbf_mae":  seed_df['cbf_mae'].mean(),
              "att_mae":  seed_df['att_mae'].mean(),
              "cbf_bias": seed_df['cbf_bias'].mean(),
              "att_bias": seed_df['att_bias'].mean()}
master = pd.concat([master, pd.DataFrame([winner_row])], ignore_index=True)
master.to_csv("experiments_G2/master_results_G2.csv", index=False)

print("\n\n=== FINAL MASTER TABLE ===")
print(master[["Model","cbf_rmse","att_rmse"]].to_string(index=False))
print("\nWinner:", winner)
print("Done.")
