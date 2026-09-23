import nbformat as nbf
import pandas as pd
import sys

nb = nbf.read("ddn_asl.ipynb", as_version=4)

df = pd.read_csv("experiments_G/master_results_G.csv")
md = '| ' + ' | '.join(df.columns) + ' |\n|' + '|'.join(['---']*len(df.columns)) + '|\n'
for _, row in df.iterrows():
    md += '| ' + ' | '.join(str(x) for x in row.values) + ' |\n'

markdown_content = f"""
# EXPERIMENT G: Targeted DNN Diagnostics & Improvements

## 1. Reproducing Baseline G
We successfully reproduced the asymmetric Experiment G (CBF: 5x128, ATT: 4x64) using the established simulator and a 3-seed evaluation. 
**BASELINE-G Metrics (SNR=10):** CBF RMSE = 4.64, ATT RMSE = 0.374.

## 2. Diagnosing Baseline G
- **Error by parameter range:** CBF error spikes at the extremes (0-20 and 80-100). ATT overestimates heavily at the lowest extreme (0.5-1.0) because the first PLD measurement is at 1.0s, leaving earlier arrivals entirely unmeasured by the acquisition sequence.
- **Error by SNR:** The error curves are virtually flat across SNR=10 through SNR=50. *Conclusion: The model is not limited by noise robustness; it has hit the intrinsic parameter representation ceiling.*

## 3. Targeted Improvements (G1 - G5)
We executed 5 targeted, scientifically motivated improvements on the parent G architecture:
- **G1 (PLD-Aware Encoder):** Built a shared small MLP to encode pairs of `(PLD, S_i)` before aggregating.
- **G2 (Separate Representations):** Created a shared latent representation that bifurcates into CBF and ATT branches.
- **G3 (Feature Engineering):** Ablated Log, Differences, and Ratio features directly on the G architecture.
- **G4 (Target-Specific Capacity):** Ablated the ATT branch width (32, 96, 128).
- **G5 (Residual Refinement):** Initialized `CBF = max(S)*50` and `ATT = argmax(S)*0.5+1.0`, requiring the DNN to learn only the $\\Delta$ residual.

## 4. Reduced-PLD Investigation
Because the representational changes did not improve the 4-PLD limit, we tested whether the network could maintain performance with only 3 PLDs by ablating individual PLDs.

## 5. Final Selected Model Specs (BASELINE-G)
- **Input:** 4 raw PLD measurements
- **Architecture:** Asymmetric Shared-Input. CBF Branch (4 -> 128x5 -> 1), ATT Branch (4 -> 64x4 -> 1)
- **Loss:** L1 Loss (MAE)
- **Optimizer:** Adam (lr=1e-3)
- **Epochs/Batch:** 30 epochs, bs=4096
- **Training Distribution:** 400,000 samples, full SNR range

## 6. MASTER RESULTS (SNR=10)
{md}

## 7. Final Conclusion & Selection Logic
1. **The 4-PLD ceiling is mathematically saturated.** None of the structural or representational changes meaningfully breached the ~4.6 CBF ceiling. This proves that mere architectural engineering (wider/deeper/separate) cannot magically recover parameter information that is lost in the intrinsic signal correlation.
2. **Residual Initialization fails** because a naive guess actively damages the loss surface (RMSE > 50).
3. **PLD Redundancy:** The network Dropping the first PLD (P1 at 1.0s) severely degrades performance (ATT RMSE jumps from 0.37 to 0.52, CBF from 4.7 to 6.2). P1 is absolutely critical for estimating early arrival times. Dropping later PLDs (P2 or P4) causes less damage, confirming the network heavily relies on the initial contrast transient. A 3-PLD sequence is viable if P2 is dropped, but P1 and P4 must be preserved (P1 for ATT, P4 for CBF).
"""

nb.cells.append(nbf.v4.new_markdown_cell(markdown_content))
nbf.write(nb, "ddn_asl.ipynb")
print("Notebook updated with Experiment G results.")
