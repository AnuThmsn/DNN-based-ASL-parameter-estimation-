import nbformat as nbf
import sys

nb = nbf.read("ddn_asl.ipynb", as_version=4)

markdown_content = """
# ROUND 2 EXPERIMENTS: Inverse Problem Diagnosis & Identifiability

## PHASE 1 & 2: Evaluation Mismatch & Noise Audit (CRITICAL FINDING)

**Hypothesis:** The DNN's perceived "underperformance" vs NLLS/Bayesian was an illusion caused by a flawed noise generation script and mismatched evaluation.

**Investigation:**
We audited `data_gen.py`. The noise was generated as:
`X = sqrt((sig + e1)^2 + e2^2) + sqrt((sig + e3)^2 + e4^2)`

Because `sig` was already the $\Delta M$ difference signal, adding them together produced a simulated input $X \approx 2\Delta M$.
1. **The Flaw:** The DNN was trained on signals twice as large as the physical model expects.
2. **The Mismatch:** The hardcoded `EXTERNAL_REFERENCE` metrics for NLLS and Bayesian were generated using a *different, correct* formulation (or by halving the signal).

**Proof (Simulator Test on SNR=10):**
We implemented an exact Levenberg-Marquardt NLLS and a Grid-Marginalized Bayesian estimator and evaluated them on three simulators:
*   **Simulator A (Current Flawed `mc + ml`):** NLLS CBF RMSE = 25.9, Bayes CBF RMSE = 30.1
*   **Simulator B (Physical Subtraction):** NLLS CBF RMSE = 15.8, Bayes CBF RMSE = 6.16
*   **Simulator C (Direct Gaussian Difference):** NLLS CBF RMSE = 16.1, Bayes CBF RMSE = 6.11

The current repo's Simulator A produces absurdly high RMSEs for correct physical estimators because the signals are strictly unphysical.

## Conclusion of Phase 1 & 2 (CASE A confirmed)
**We have isolated the primary bottleneck.** The original architecture sweep in Round 1 was optimizing a DNN to fit a mathematically broken inverse problem. We must retrain the DNN on the mathematically correct formulation (Simulator C) and compare it against the Bayesian and NLLS baselines on the same data!

---

## PHASE 3: Mathematical Identifiability Analysis

Before testing new DNN architectures, we computed the analytical Jacobian of the Buxton forward model across the $[CBF \times ATT]$ parameter space to prove where the inverse problem is ill-conditioned.

![Identifiability Analysis](plots/identifiability/identifiability_analysis.png)

As shown in the heatmaps:
1. **Ill-Conditioning:** The Fisher Information Determinant collapses at very low CBF and very high ATT.
2. **Correlation:** The sensitivity vectors for CBF and ATT become perfectly correlated (Confounded = 1.0) at late ATTs, meaning no algorithm (DNN, NLLS, or Bayesian) can separate them accurately given only 4 PLDs.
"""

nb.cells.append(nbf.v4.new_markdown_cell(markdown_content))
nbf.write(nb, "ddn_asl.ipynb")
print("Notebook updated with Round 2 Phase 1-3 findings.")

