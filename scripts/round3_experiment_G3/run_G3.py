"""
Experiment G3 — CBF Boundary-Bias Reduction
============================================
Parent: G1_LOG  (raw + log features, asymmetric MLP, 4 PLDs fixed)

Candidates:
  G3_A0   — G1_LOG reproduced (baseline, seed 42)
  G3_A1   — Moderate boundary oversampling (CBF 0-20 & 80-100 @ 2x)
  G3_A2   — Strong boundary oversampling (CBF 0-20 & 80-100 @ 4x)
  G3_B1   — Mild boundary-weighted CBF loss
  G3_B2   — Strong boundary-weighted CBF loss
  G3_C    — Best oversampling + best weighting (only if both help)

Controls (ALL identical to G1_LOG):
  PLDs, simulator, noise model, test set, input repr, architecture,
  optimizer, LR, batch size, epochs, target scaling, evaluation code.

Output:
  experiments_G3/master_results_G3.csv
  experiments_G3/diagnostics/<name>_cbf_bins.csv
  experiments_G3/diagnostics/<name>_pred_vs_truth.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

from data_gen import (
    X_tr, X_val, X_test,
    Y_tr, Y_val, Y_test,
    Y_tr_both_t, Y_val_both_t,
    CBF_mean, CBF_std, ATT_mean, ATT_std,
    CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX,
    PLDs, SD_MAX, SIG_REF_SC,
    device, calc_rmse, calc_mae, calc_bias
)

# ── constants ──────────────────────────────────────────────
EPS    = 1e-6
N_SUB  = 400_000
EPOCHS = 30
BS     = 4096
LR     = 1e-3
SEEDS  = [42, 43, 44]

OUT_DIR   = "experiments_G3"
MODEL_DIR = f"{OUT_DIR}/models"
DIAG_DIR  = f"{OUT_DIR}/diagnostics"

print("=== Experiment G3: CBF Boundary-Bias Reduction ===\n")

# ── SNR array (fixed, same seed as data_gen test) ──────────
def _snr_arr(N, rng, n=51):
    sd = np.linspace(0, SD_MAX, n)[rng.integers(0, n, N)]
    with np.errstate(divide="ignore"):
        return np.where(sd > 0, SIG_REF_SC / sd, np.inf)

SNR_TEST = _snr_arr(100_000, np.random.default_rng(7))

# ── feature engineering (G1_LOG representation) ────────────
def make_log_features(X_raw):
    return np.concatenate([X_raw, np.log(X_raw.astype(np.float64) + EPS)], axis=1).astype(np.float32)

# Compute train features, derive normalisation stats from training data only
F_tr_raw = make_log_features(X_tr[:N_SUB])
F_te_raw = make_log_features(X_test)
F_mu  = F_tr_raw.mean(0, keepdims=True)
F_sig = F_tr_raw.std(0,  keepdims=True) + EPS
F_tr_n = ((F_tr_raw - F_mu) / F_sig).astype(np.float32)
F_te_n = ((F_te_raw - F_mu) / F_sig).astype(np.float32)
X_te_t = torch.tensor(F_te_n, device=device)

# ── G1_LOG architecture ────────────────────────────────────
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

# ── decode predictions to physical units ──────────────────
def decode(preds):
    cbf = np.clip(preds[:, 0] * CBF_std + CBF_mean, CBF_MIN, CBF_MAX)
    att = np.clip(preds[:, 1] * ATT_std + ATT_mean, ATT_MIN, ATT_MAX)
    return cbf, att

# ── full diagnostics ───────────────────────────────────────
CBF_BINS = [(0,20),(20,40),(40,60),(60,80),(80,100)]
ATT_BINS = [(0.5,1.0),(1.0,1.5),(1.5,2.0),(2.0,2.5),(2.5,3.0)]
SNR_VALS = [10, 15, 20, 30, 40]

def diagnostics(cbf_p, att_p, name):
    y_cbf, y_att = Y_test[:,0], Y_test[:,1]

    # overall
    overall = {
        "cbf_rmse": calc_rmse(y_cbf, cbf_p),
        "cbf_mae":  calc_mae (y_cbf, cbf_p),
        "cbf_bias": calc_bias(y_cbf, cbf_p),
        "att_rmse": calc_rmse(y_att, att_p),
        "att_mae":  calc_mae (y_att, att_p),
        "att_bias": calc_bias(y_att, att_p),
    }

    # CBF bins
    cbf_bin_rows = []
    for lo, hi in CBF_BINS:
        m = (y_cbf >= lo) & (y_cbf < hi)
        cbf_bin_rows.append({
            "bin": f"{lo}-{hi}", "N": int(m.sum()),
            "RMSE":  round(calc_rmse(y_cbf[m], cbf_p[m]), 4),
            "MAE":   round(calc_mae (y_cbf[m], cbf_p[m]), 4),
            "Bias":  round(calc_bias(y_cbf[m], cbf_p[m]), 4),
            "mean_pred": round(cbf_p[m].mean(), 4),
            "mean_true": round(y_cbf[m].mean(), 4),
        })
    bin_df = pd.DataFrame(cbf_bin_rows)
    bin_df.to_csv(f"{DIAG_DIR}/{name}_cbf_bins.csv", index=False)

    # Boundary bias summary
    low_mask  = (y_cbf >= 0)  & (y_cbf < 20)
    high_mask = (y_cbf >= 80) & (y_cbf < 100)
    mid_mask  = (y_cbf >= 20) & (y_cbf < 80)
    overall["low_cbf_bias"]  = round(calc_bias(y_cbf[low_mask],  cbf_p[low_mask]),  4)
    overall["high_cbf_bias"] = round(calc_bias(y_cbf[high_mask], cbf_p[high_mask]), 4)
    overall["mid_cbf_bias"]  = round(calc_bias(y_cbf[mid_mask],  cbf_p[mid_mask]),  4)

    # pred vs truth (subsample 5000 for CSV)
    idx = np.random.default_rng(0).choice(len(y_cbf), 5000, replace=False)
    pd.DataFrame({"true_cbf": y_cbf[idx], "pred_cbf": cbf_p[idx],
                  "true_att": y_att[idx], "pred_att": att_p[idx]}) \
      .to_csv(f"{DIAG_DIR}/{name}_pred_vs_truth.csv", index=False)

    # SNR
    snr_rows = []
    for snr in SNR_VALS:
        mask = (SNR_TEST >= snr*0.9) & (SNR_TEST <= snr*1.1)
        if mask.sum() < 50: continue
        snr_rows.append({"SNR": snr,
                         "CBF_RMSE": round(calc_rmse(y_cbf[mask], cbf_p[mask]), 4),
                         "ATT_RMSE": round(calc_rmse(y_att[mask], att_p[mask]), 4)})
    pd.DataFrame(snr_rows).to_csv(f"{DIAG_DIR}/{name}_snr.csv", index=False)

    return overall, bin_df

def print_bins(bin_df, name):
    print(f"\n  CBF bins [{name}]:")
    print(f"  {'Bin':8s} {'N':7s} {'RMSE':8s} {'MAE':8s} {'Bias':9s}")
    for _, r in bin_df.iterrows():
        print(f"  {r['bin']:8s} {r['N']:7d} {r['RMSE']:8.4f} {r['MAE']:8.4f} {r['Bias']:+9.4f}")

# ── training function ──────────────────────────────────────
def run_experiment(name, xtr_t, ytr_t, seed=42, loss_weights=None, weighted_cbf=False):
    """
    loss_weights : per-sample weight tensor (N,) for the CBF term, or None (uniform).
    weighted_cbf : if True, use sample-weighted CBF L1 loss + unweighted ATT L1 loss.
                   if False, use standard L1 on both outputs jointly.
    """
    torch.manual_seed(seed)
    model = G1_LOG_Net().to(device)
    model.apply(lambda m: m.reset_parameters() if hasattr(m,'reset_parameters') else None)
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.L1Loss(reduction="none")

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(xtr_t.size(0), device=device)
        for i in range(0, xtr_t.size(0), BS):
            idx = perm[i:i+BS]
            opt.zero_grad()
            out = model(xtr_t[idx])          # (B, 2)
            if weighted_cbf and loss_weights is not None:
                w = loss_weights[idx].unsqueeze(1)      # (B,1)
                cbf_loss = (crit(out[:,0:1], ytr_t[idx,0:1]) * w).mean()
                att_loss =  crit(out[:,1:2], ytr_t[idx,1:2]).mean()
                loss = cbf_loss + att_loss
            else:
                loss = crit(out, ytr_t[idx]).mean()
            loss.backward()
            opt.step()

    torch.save(model.state_dict(), f"{MODEL_DIR}/{name}_s{seed}.pth")

    model.eval()
    with torch.no_grad():
        chunks = [model(X_te_t[j:j+BS]).cpu().numpy()
                  for j in range(0, X_te_t.size(0), BS)]
    raw_preds = np.concatenate(chunks, 0)
    cbf_p, att_p = decode(raw_preds)
    overall, bin_df = diagnostics(cbf_p, att_p, f"{name}_s{seed}")
    elapsed = time.time() - t0
    print(f"  [{name:20s} seed={seed}]  "
          f"CBF_RMSE={overall['cbf_rmse']:.4f}  ATT_RMSE={overall['att_rmse']:.4f}  "
          f"lowBias={overall['low_cbf_bias']:+.3f}  highBias={overall['high_cbf_bias']:+.3f}  "
          f"({elapsed:.0f}s)")
    return overall, bin_df, cbf_p, att_p

# ── base training tensors (G3_A0 — standard sampling) ──────
xtr_base = torch.tensor(F_tr_n, device=device)
ytr_base = Y_tr_both_t[:N_SUB]

# ── CBF ground-truth for the training subset ───────────────
cbf_tr = Y_tr[:N_SUB, 0]   # physical units

# ─────────────────────────────────────────────────────────────
# ─── STEP 0: Reproduce G1_LOG ────────────────────────────────
# ─────────────────────────────────────────────────────────────
print("─" * 60)
print("STEP 0: Reproduce G1_LOG (G3_A0)")
print("─" * 60)
results = []

ov, bin_df, _, _ = run_experiment("G3_A0", xtr_base, ytr_base, seed=42)
print_bins(bin_df, "G3_A0")
results.append({"name":"G3_A0", "description":"G1_LOG reproduced", **ov})

# ─────────────────────────────────────────────────────────────
# ─── STEP 1: Boundary oversampling ──────────────────────────
# ─────────────────────────────────────────────────────────────
def build_oversampled(mult_boundary, seed_os=0):
    """
    Return (X_tensor, Y_tensor) where boundary samples (CBF<20 or CBF>80)
    are included `mult_boundary` times total vs 1x for the rest.
    """
    rng_os = np.random.default_rng(seed_os)
    boundary_idx = np.where((cbf_tr < 20) | (cbf_tr > 80))[0]
    interior_idx = np.where((cbf_tr >= 20) & (cbf_tr <= 80))[0]

    # how many extra copies of boundary samples?
    extra_copies = mult_boundary - 1
    aug_idx = np.tile(boundary_idx, extra_copies)
    all_idx = np.concatenate([np.arange(N_SUB), aug_idx])
    rng_os.shuffle(all_idx)

    X_aug = F_tr_n[all_idx]
    Y_aug = Y_tr_both_t[all_idx]
    return (torch.tensor(X_aug, device=device),
            Y_aug,
            len(all_idx))

print("\n" + "─" * 60)
print("STEP 1: Boundary Oversampling (G3_A1 @ 2x, G3_A2 @ 4x)")
print("─" * 60)

for mult, tag in [(2, "G3_A1"), (4, "G3_A2")]:
    xtr_os, ytr_os, n_os = build_oversampled(mult)
    desc = f"boundary oversampling {mult}x (N={n_os:,})"
    print(f"\n  {tag}: {desc}")
    ov, bin_df, _, _ = run_experiment(tag, xtr_os, ytr_os, seed=42)
    print_bins(bin_df, tag)
    results.append({"name": tag, "description": desc, **ov})

# ─────────────────────────────────────────────────────────────
# ─── STEP 2: Boundary-weighted CBF loss ─────────────────────
# ─────────────────────────────────────────────────────────────
def cbf_boundary_weight(cbf_arr, strength=2.0, boundary=20.0):
    """
    Smooth weighting: w = 1 + (strength-1)*sigmoid( -(cbf - boundary)/5 )
                         + (strength-1)*sigmoid( (cbf - (100-boundary))/5 )
    This gives weight ~`strength` at extremes and ~1 in the centre.
    """
    sig_lo = 1.0 / (1.0 + np.exp((cbf_arr - boundary) / 5.0))
    sig_hi = 1.0 / (1.0 + np.exp(-(cbf_arr - (100.0 - boundary)) / 5.0))
    w = 1.0 + (strength - 1.0) * sig_lo + (strength - 1.0) * sig_hi
    return w.astype(np.float32)

print("\n" + "─" * 60)
print("STEP 2: Weighted CBF Loss (G3_B1 mild, G3_B2 strong)")
print("─" * 60)

for strength, tag in [(2.0, "G3_B1"), (4.0, "G3_B2")]:
    w_np   = cbf_boundary_weight(cbf_tr, strength=strength)
    w_t    = torch.tensor(w_np, device=device)
    desc   = f"CBF loss weight strength={strength}"
    print(f"\n  {tag}: {desc}")
    ov, bin_df, _, _ = run_experiment(
        tag, xtr_base, ytr_base, seed=42,
        loss_weights=w_t, weighted_cbf=True)
    print_bins(bin_df, tag)
    results.append({"name": tag, "description": desc, **ov})

# ─────────────────────────────────────────────────────────────
# ─── Decision: select candidates for G3C ────────────────────
# ─────────────────────────────────────────────────────────────
res_df = pd.DataFrame(results)
baseline_cbf  = res_df.loc[res_df.name=="G3_A0","cbf_rmse"].values[0]
baseline_att  = res_df.loc[res_df.name=="G3_A0","att_rmse"].values[0]

# A candidate "helps" if:
#  - cbf_rmse <= baseline + 0.05 (not substantially worse)
#  - |low_cbf_bias| or |high_cbf_bias| < corresponding baseline value
b_low  = abs(res_df.loc[res_df.name=="G3_A0","low_cbf_bias"].values[0])
b_high = abs(res_df.loc[res_df.name=="G3_A0","high_cbf_bias"].values[0])

promising_os  = []
promising_w   = []
for _, r in res_df.iterrows():
    if r["name"] == "G3_A0": continue
    bias_improved = (abs(r["low_cbf_bias"]) < b_low or abs(r["high_cbf_bias"]) < b_high)
    rmse_ok       = r["cbf_rmse"] <= baseline_cbf + 0.05
    if bias_improved and rmse_ok:
        if r["name"].startswith("G3_A"):
            promising_os.append(r["name"])
        elif r["name"].startswith("G3_B"):
            promising_w.append(r["name"])

print(f"\n  Promising oversampling candidates: {promising_os}")
print(f"  Promising weighting  candidates: {promising_w}")

# ─────────────────────────────────────────────────────────────
# ─── STEP 3: G3C — Combined (only if both sides improved) ───
# ─────────────────────────────────────────────────────────────
run_combined = len(promising_os) > 0 and len(promising_w) > 0
if run_combined:
    best_os = promising_os[0]   # first in list
    best_w  = promising_w[0]
    mult    = 2 if "A1" in best_os else 4
    strength= 2.0 if "B1" in best_w else 4.0
    print(f"\n{'─'*60}")
    print(f"STEP 3: G3C — Combined (oversample {mult}x + weight {strength})")
    print("─" * 60)

    xtr_os, ytr_os, n_os = build_oversampled(mult)
    # recompute weights for the oversampled subset
    cbf_os = Y_tr[:N_SUB, 0][np.tile(np.arange(N_SUB), 1)]   # base
    # we apply uniform weights here (oversampling already emphasises boundary)
    w_os = torch.tensor(
        cbf_boundary_weight(
            Y_tr_both_t[np.arange(N_SUB)].cpu().numpy()[:,0] * CBF_std + CBF_mean
        ), device=device)
    # extend to oversampled size — simpler: use boundary_idx extension
    boundary_idx = np.where((Y_tr[:N_SUB,0] < 20) | (Y_tr[:N_SUB,0] > 80))[0]
    aug_idx      = np.tile(boundary_idx, mult-1)
    all_idx      = np.concatenate([np.arange(N_SUB), aug_idx])
    np.random.default_rng(0).shuffle(all_idx)
    w_comb = torch.tensor(
        cbf_boundary_weight(Y_tr[:N_SUB,0][all_idx], strength=strength), device=device)

    desc = f"combined oversample {mult}x + CBF weight {strength}"
    ov, bin_df, _, _ = run_experiment(
        "G3_C", xtr_os, ytr_os, seed=42,
        loss_weights=w_comb, weighted_cbf=True)
    print_bins(bin_df, "G3_C")
    results.append({"name":"G3_C", "description": desc, **ov})
else:
    print("\n  Skipping G3C: not both oversampling AND weighting showed independent improvement.")
    print(f"  (promising_os={promising_os}, promising_w={promising_w})")

# ─────────────────────────────────────────────────────────────
# ─── 3-Seed verification of best candidate ──────────────────
# ─────────────────────────────────────────────────────────────
res_df = pd.DataFrame(results)

# Select winner by: bias improvement AND cbf_rmse within 0.05 of baseline
candidates = res_df[res_df.name != "G3_A0"].copy()
candidates["bias_improvement"] = (
    (b_low  - candidates["low_cbf_bias"].abs()) +
    (b_high - candidates["high_cbf_bias"].abs())
)
candidates["overall_score"] = (
    candidates["cbf_rmse"] / baseline_cbf +
    candidates["att_rmse"] / baseline_att
)

# Filter to only reasonable RMSE
viable = candidates[candidates["cbf_rmse"] <= baseline_cbf + 0.05]

if len(viable) > 0:
    winner = viable.loc[viable["bias_improvement"].idxmax(), "name"]
else:
    winner = None

print(f"\n{'═'*60}")
print(f"Winner for 3-seed verification: {winner}")
print(f"{'═'*60}")

seed_rows = []
if winner is not None:
    for seed in SEEDS:
        if winner == "G3_A0":
            xtr_w, ytr_w = xtr_base, ytr_base
            w_t_ver, wc = None, False
        elif winner in ["G3_A1","G3_A2"]:
            mult = 2 if winner=="G3_A1" else 4
            xtr_w, ytr_w, _ = build_oversampled(mult, seed_os=seed)
            w_t_ver, wc = None, False
        elif winner in ["G3_B1","G3_B2"]:
            strength = 2.0 if winner=="G3_B1" else 4.0
            xtr_w, ytr_w = xtr_base, ytr_base
            w_t_ver = torch.tensor(cbf_boundary_weight(cbf_tr, strength), device=device)
            wc = True
        else:  # G3_C
            xtr_w, ytr_w, _ = build_oversampled(mult, seed_os=seed)
            w_t_ver = w_comb; wc = True

        ov, bin_df, _, _ = run_experiment(
            f"{winner}_FINAL", xtr_w, ytr_w, seed=seed,
            loss_weights=w_t_ver, weighted_cbf=wc)
        seed_rows.append(ov)

    seed_df = pd.DataFrame(seed_rows)
    print(f"\n  3-seed summary for {winner}:")
    print(f"  CBF RMSE  {seed_df['cbf_rmse'].mean():.4f} ± {seed_df['cbf_rmse'].std():.4f}")
    print(f"  ATT RMSE  {seed_df['att_rmse'].mean():.4f} ± {seed_df['att_rmse'].std():.4f}")
    print(f"  low bias  {seed_df['low_cbf_bias'].mean():.4f} ± {seed_df['low_cbf_bias'].std():.4f}")
    print(f"  high bias {seed_df['high_cbf_bias'].mean():.4f} ± {seed_df['high_cbf_bias'].std():.4f}")
    print(f"\n  G1_LOG reference: CBF 4.638±0.012, ATT 0.372±0.001")

# ─────────────────────────────────────────────────────────────
# ─── Decision table ─────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
res_df = pd.DataFrame(results)
def classify(row):
    if row["name"] == "G3_A0":
        return "BASELINE"
    bias_imp = (abs(row["low_cbf_bias"])  < b_low or
                abs(row["high_cbf_bias"]) < b_high)
    rmse_ok  = row["cbf_rmse"] <= baseline_cbf + 0.05
    att_ok   = row["att_rmse"] <= baseline_att + 0.01
    if bias_imp and rmse_ok and att_ok:
        if row["name"] == winner:
            return "PROMISING → 3-seed"
        return "PROMISING"
    elif bias_imp and not rmse_ok:
        return "KEEP AS DIAGNOSTIC"
    else:
        return "REJECT"

res_df["decision"] = res_df.apply(classify, axis=1)

# Add 3-seed mean row
if seed_rows:
    seed_df = pd.DataFrame(seed_rows)
    final_row = {
        "name": f"{winner}_3SEED_MEAN",
        "description": "3-seed verification",
        "cbf_rmse": seed_df["cbf_rmse"].mean(),
        "att_rmse": seed_df["att_rmse"].mean(),
        "cbf_mae":  seed_df["cbf_mae"].mean(),
        "att_mae":  seed_df["att_mae"].mean(),
        "cbf_bias": seed_df["cbf_bias"].mean(),
        "att_bias": seed_df["att_bias"].mean(),
        "low_cbf_bias":  seed_df["low_cbf_bias"].mean(),
        "high_cbf_bias": seed_df["high_cbf_bias"].mean(),
        "mid_cbf_bias":  seed_df["mid_cbf_bias"].mean(),
        "decision": "NEW PARENT (if verified better)" if seed_df["cbf_rmse"].mean() < baseline_cbf else "REJECT (no global improvement)",
    }
    res_df = pd.concat([res_df, pd.DataFrame([final_row])], ignore_index=True)

res_df.to_csv(f"{OUT_DIR}/master_results_G3.csv", index=False)

print(f"\n{'═'*60}")
print("FINAL MASTER TABLE")
print("═'*60")
print(res_df[["name","cbf_rmse","att_rmse","low_cbf_bias","high_cbf_bias","decision"]].to_string(index=False))
print("\nDone. Results saved to experiments_G3/master_results_G3.csv")

