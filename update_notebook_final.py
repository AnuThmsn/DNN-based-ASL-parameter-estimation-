import nbformat as nbf
import pandas as pd
import sys

nb = nbf.read("ddn_asl.ipynb", as_version=4)

try:
    df = pd.read_csv("master_results.csv")
    md_table = df.to_markdown(index=False)
except:
    md_table = "*Results pending...*"

markdown_content = f"""
# ROUND 2 EXPERIMENTS: DNN Diagnostic & Performance Improvement

## OBJECTIVE
Improve the ASL DNN CBF/ATT parameter estimation model and determine whether it can achieve better results than the established Bayesian and NLLS fixed reference results. The Bayesian and NLLS values are comparison targets, and remain unchanged.

## EXPERIMENTS CONDUCTED
- **Experiment A (PLD-Aware Feature Encoding):** Ablations over raw signals, consecutive differences, ratios, and logarithmic features.
- **Experiment B (Temporal/PLD Encoder):** A shared feature encoder extracting representations from (PLD, signal) pairs before aggregation.
- **Experiment C (Separate CBF/ATT Representations):** Shared low-level representation with asymmetric branches for CBF (amplitude-heavy) and ATT (temporal-heavy).
- **Experiment D (Coarse-to-Fine Estimation):** Initial coarse ATT prediction used to condition the final CBF representation.
- **Experiment E (Residual Parameter Estimation):** An analytical initial guess from which the DNN predicts the residual $\Delta$ terms.

## FINAL MASTER RESULTS (SNR = 10)
{md_table}

## CONCLUSION
(See final analysis based on the table above).
"""

nb.cells.append(nbf.v4.new_markdown_cell(markdown_content))
nbf.write(nb, "ddn_asl.ipynb")
print("Notebook updated with final Round 2 results.")
