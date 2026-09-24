import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import nbformat as nbf
import pandas as pd

nb  = nbf.read("ddn_asl.ipynb", as_version=4)
df  = pd.read_csv("experiments_G3/master_results_G3.csv")

def md_table(d, cols=None):
    if cols: d = d[cols]
    hdr = "| " + " | ".join(d.columns) + " |"
    sep = "|" + "|".join(["---"]*len(d.columns)) + "|"
    rows = []
    for _, r in d.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float): cells.append(f"{v:.4f}")
            else: cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([hdr, sep] + rows)

# Per-run CBF-bin tables
def load_bins(name, seed=42):
    p = f"experiments_G3/diagnostics/{name}_s{seed}_cbf_bins.csv"
    if os.path.exists(p): return pd.read_csv(p)
    return None

narrative = """
# Experiment G3 — CBF Boundary-Bias Reduction
**Parent:** G1_LOG (raw + log features, 4 PLDs fixed)

## Objective
G2 diagnostics showed G1_LOG has large symmetric regression-to-mean bias at CBF extremes:
- Low-CBF (0-20): bias = +1.81 (overestimation)
- High-CBF (80-100): bias = -1.70 (underestimation)

This experiment tests whether targeted training modifications can reduce that boundary bias
without worsening the overall CBF/ATT RMSE.

## Protocol (Frozen — identical to G1_LOG in all other respects)
| Item | Value |
|---|---|
| PLDs | [1.525, 2.025, 2.525, 3.025] s |
| Simulator | Unchanged |
| Input | raw [S1..S4] + log([S1..S4]+eps) |
| Architecture | CBF 8->128x5->1, ATT 8->64x4->1 |
| Optimizer | Adam, lr=1e-3 |
| Epochs / Batch | 30 / 4096 |
| Training samples | 400,000 base |
| Evaluation | Full test set (100,000), physical units |

## Candidates Tested

### G3_A0 — Reproduce G1_LOG (control)
Exact G1_LOG, seed 42.

### G3_A1 — Boundary Oversampling 2x
CBF < 20 and CBF > 80 training samples duplicated, producing N = 559,933.

### G3_A2 — Boundary Oversampling 4x
Same regions 4x, N = 879,799.

### G3_B1 — Mild Boundary-Weighted CBF Loss (strength = 2.0)
A smooth sigmoid weighting function giving weight ~2 at extremes, ~1 in centre.
ATT loss unchanged.

### G3_B2 — Strong Boundary-Weighted CBF Loss (strength = 4.0)
Same but stronger emphasis.

### G3_C — Combined (2x OS + strength 2.0 weight)
Run only because both G3_A1 and G3_B1 passed the individual screening.

## Full Results (seed=42)

""" + md_table(df[df["name"] != "G3_B1_3SEED_MEAN"],
               ["name","cbf_rmse","att_rmse","low_cbf_bias","high_cbf_bias","decision"]) + """

## 3-Seed Verification of G3_B1

G3_B1 was selected as the most promising candidate:
- Both low and high bias reduced at seed 42.
- Global CBF RMSE within +0.05 of baseline.

| Seed | CBF RMSE | ATT RMSE | Low Bias | High Bias |
|---|---|---|---|---|
| 42 | 4.6703 | 0.3717 | +1.089 | -0.592 |
| 43 | 4.6397 | 0.3733 | +0.760 | -0.823 |
| 44 | 4.7448 | 0.3724 | +0.625 | +0.148 |
| **Mean ± Std** | **4.685 ± 0.054** | **0.373 ± 0.001** | **+0.824 ± 0.239** | **-0.423 ± 0.507** |

**G1_LOG reference:** CBF 4.638 ± 0.012, ATT 0.372 ± 0.001

## Decision

**G3_B1: REJECT as new parent.**

Reason:
1. **Global CBF RMSE increased**: 4.685 vs 4.638 for G1_LOG. The improvement at boundaries
   comes at the cost of fitting the central region less accurately.
2. **High variance across seeds**: std = 0.054 vs 0.012 for G1_LOG. The weighting
   function destabilises optimisation, making results seed-dependent.
3. **Boundary bias reduction is real but inconsistent**: low-bias dropped from 1.81 to
   0.82 (substantial) but high-bias showed seed 44 = +0.148 (sign flip), indicating
   the model is not reliably fixing the high-CBF boundary.

All other candidates (G3_A2, G3_B2, G3_C) are classified KEEP AS DIAGNOSTIC:
they reduce boundary bias more aggressively but at unacceptable global RMSE cost.

## What This Tells Us

The boundary regression-to-mean bias in CBF is **not a simple data-coverage or
loss-emphasis problem**.

The more likely root cause is **intrinsic ill-conditioning of the forward model**:
at very low CBF the signal approaches zero and becomes dominated by noise, making the
inverse problem fundamentally harder. At high CBF the model begins saturating at the
shortest PLDs. Oversampling or reweighting these samples does not give the network
additional information — it only shifts the balance of gradient contributions without
resolving the underlying ambiguity.

## Maintained Parent

**G1_LOG remains the established parent model.**

CBF RMSE 4.638 ± 0.012 | ATT RMSE 0.372 ± 0.001

## Recommended Next Experiment

Given that:
- Representation changes (G2): ✅ small but consistent improvement (log features)
- Boundary sampling/weighting (G3): ❌ boundary bias not reliably fixable this way

The next scientifically motivated step is either:

**Option A — Longer/better training of G1_LOG:**
Test whether 60-80 epochs with learning-rate decay (e.g., cosine or reduce-on-plateau)
closes the remaining gap — the current 30-epoch budget may not be saturating the
log-augmented representation.

**Option B — Physics-consistent output constraints:**
The CBF boundary problem appears partly physical. Adding a soft constraint that
forces predictions to remain well within [CBF_MIN, CBF_MAX] (e.g., a sigmoid output
layer) might reduce boundary compression without hurting the centre.
"""

nb.cells.append(nbf.v4.new_markdown_cell(narrative))
nbf.write(nb, "ddn_asl.ipynb")
print("Notebook updated.")
