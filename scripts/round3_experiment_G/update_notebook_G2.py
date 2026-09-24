import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import nbformat as nbf
import pandas as pd

nb = nbf.read("ddn_asl.ipynb", as_version=4)
master = pd.read_csv("experiments_G2/master_results_G2.csv")

# Build markdown table
def df_to_md(df):
    cols = df.columns.tolist()
    hdr  = "| " + " | ".join(cols) + " |"
    sep  = "|" + "|".join(["---"]*len(cols)) + "|"
    rows = "\n".join("| " + " | ".join(
        f"{v:.4f}" if isinstance(v, float) else str(v)
        for v in row) + " |"
        for row in df.values)
    return "\n".join([hdr, sep, rows])

# Load diagnostics
cbf_bins = pd.read_csv("experiments_G2/diagnostics/winner_cbf_bins.csv")
att_bins = pd.read_csv("experiments_G2/diagnostics/winner_att_bins.csv")
snr_df   = pd.read_csv("experiments_G2/diagnostics/winner_snr.csv")

narrative = f"""
# Experiment G2: Controlled Representation Ablation (4 PLDs, Fixed Simulator)

## Objective
Determine whether Experiment G can be improved **without changing the 4-PLD acquisition,
the simulator, or the evaluation protocol** — purely by giving the network a better
representation of the existing four measurements.

## Protocol (Frozen)
| Item | Value |
|---|---|
| PLDs | [1.525, 2.025, 2.525, 3.025] s |
| Simulator | Unchanged (`X = mc + ml`) |
| Training subset | 400,000 samples |
| Epochs / Batch / LR | 30 / 4096 / 1e-3 |
| Loss | L1 (MAE) |
| Optimizer | Adam |
| Evaluation | Full test set (100,000), physical units |
| Seeds (verification) | 42, 43, 44 |

## Phase 1 — Feature Diagnostics
All feature variants were verified before training: **no NaN, no Inf**, ranges as expected.

| Feature set | Shape | Min | Max |
|---|---|---|---|
| G_BASELINE (raw) | (400000, 4) | 0.389 | 2156.2 |
| G1_LOG (raw+log) | (400000, 8) | −0.945 | 2156.2 |
| G2_DIFF (raw+Δ) | (400000, 7) | −1058.1 | 2156.2 |
| G3_RATIO (raw+ratio) | (400000, 7) | 0.051 | 2156.2 |
| G4_COMBINED | (400000, 14) | −1058.1 | 2156.2 |

## Phase 2 & 3 — Ablation Results (seed=42)
{df_to_md(master)}

## Phase 4 — 3-Seed Verification of Winner: **G1_LOG**
Winner selected by joint (CBF + ATT) normalised RMSE score.

| Seed | CBF RMSE | ATT RMSE |
|---|---|---|
| 42 | 4.6251 | 0.3718 |
| 43 | 4.6413 | 0.3731 |
| 44 | 4.6478 | 0.3722 |
| **Mean ± Std** | **4.638 ± 0.012** | **0.372 ± 0.001** |

G_BASELINE (3-seed, from prior run): CBF 4.673, ATT 0.374

**Net improvement: CBF −0.035, ATT −0.002 (consistent across all 3 seeds).**

## Phase 5 — Error Analysis for G1_LOG (mean predictions, 3 seeds)

### CBF Error by Parameter Range
{df_to_md(cbf_bins)}

### ATT Error by Parameter Range
{df_to_md(att_bins)}

### Error by SNR
{df_to_md(snr_df)}

## Interpretation

### Why G1_LOG Improved
The log transform linearises the exponential ASL signal decay:
`S ∝ exp(−ATT/T1a) · (exp(−(PLD−ATT)/T1t) − exp(−(PLD+τ−ATT)/T1t))`
By supplying `log(S_i)` directly, the network can form **linear combinations** of the
log-features to approximate the exponent arguments — a task that would require
many nonlinear layers on raw amplitudes alone.
The CBF improvement is strongest because CBF scales the overall amplitude, which is
directly captured in the log-magnitude.

### Where G1_LOG Still Fails
1. **ATT < 1.0 s:** RMSE = 0.601 — all PLDs start at 1.525 s, so early-arrival spins
   are unmeasured. This is an acquisition constraint, not a model failure.
2. **CBF extremes (0–20 and 80–100):** Positive bias at low CBF (model overestimates)
   and negative bias at high CBF (model underestimates) — classic regression-to-mean.
3. **SNR is not the bottleneck:** Error is flat across SNR 10–40, confirming the
   limitation is intrinsic representation rather than noise.

## Decision
**G1_LOG replaces G as the new parent model.**

Next experiment should target: reducing the systematic bias at CBF extremes,
either through weighted sampling of extreme-CBF training examples or a
focal / asymmetric loss that penalises boundary errors more heavily.
"""

nb.cells.append(nbf.v4.new_markdown_cell(narrative))
nbf.write(nb, "ddn_asl.ipynb")
print("Notebook updated.")
