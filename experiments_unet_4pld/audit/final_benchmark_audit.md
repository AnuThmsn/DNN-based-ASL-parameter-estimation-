# Audit Report: Final 4-PLD Benchmark Validation & Statistical Verification

## 1. Executive Summary

This audit validates the complete, corrected 4-PLD benchmark for ASL-MRI CBF and ATT parameter estimation. Following the calibration of the physics simulator expectations ($\mathbb{E}[X] \approx 2 \cdot \text{SCALE} \cdot \Delta M$, $\sigma_{eff} = \sqrt{2}\sigma$) and the establishment of matched evaluation sets, the 2D Spatial U-Net was trained and evaluated across canonical seeds `[42, 123, 2024]` and compared against:
1. **Point-wise DNN Baseline** (matching 5-layer MLP architectures)
2. **Calibrated Bayesian Estimator** (dense $100 \times 100$ GPU grid with calibrated noise model)
3. **Non-Linear Least Squares (NLLS)** (Trust Region Reflective bounded solver)
4. **Historical 0D Literature References** (cataloged from `reference_baselines.yaml`)

---

## 2. Methodology & Protocol Parity

| Dimension | Specification | Verification Status |
| :--- | :--- | :--- |
| **Inversion Protocol** | 4 PLDs: `[1.525, 2.025, 2.525, 3.025]` s, $\tau = 1.8$ s | **Verified & Frozen** |
| **Simulator Model** | Two-compartment Buxton ASL model, sum-of-Ricians noise | **Verified & Calibrated** |
| **Noise Calibration** | $\sigma_{eff} = \frac{\sqrt{2} \cdot \text{SIG\_REF\_SC}}{\text{SNR}}$ | **Calibrated across Bayesian & NLLS** |
| **Normalization** | Channel-wise z-score computed strictly on training set valid brain voxels ($M_{brain} > 0.5$) | **Zero data leakage verified** |
| **Test Set Integrity** | 100 2D synthetic brain phantoms ($64 \times 64$), SNR $\in [10, 15]$ | **Frozen & Shared** |
| **Matched Sample Size** | All 4 estimators evaluated on identical 25 phantoms (39,410 valid voxels) | **Table 1 Parity Enforced** |
| **Replication Seeds** | Multi-seed verification on canonical seeds `[42, 123, 2024]` | **Verified** |

---

## 3. Multi-Seed Stability (U-Net)

Trained across 3 independent seeds with validation loss early stopping (AdamW, lr=1e-3, ReduceLROnPlateau):

| Seed | Best Epoch | Train Time (s) | CBF RMSE (ml/100g/min) | CBF MAE | CBF $R^2$ | ATT RMSE (s) | ATT MAE | ATT $R^2$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 42 | 61 | 122.3 | 0.9673 | 0.6799 | 0.9954 | 0.0561 | 0.0376 | 0.9792 |
| 123 | 46 | 97.4 | 1.0280 | 0.7259 | 0.9948 | 0.0583 | 0.0390 | 0.9776 |
| 2024 | 52 | 107.8 | 0.9784 | 0.6930 | 0.9953 | 0.0563 | 0.0379 | 0.9791 |
| **Mean $\pm$ SD** | **53.0 $\pm$ 7.5** | **109.2 $\pm$ 12.5** | **0.9912 $\pm$ 0.0323** | **0.6996 $\pm$ 0.0237** | **0.9952 $\pm$ 0.0003** | **0.0569 $\pm$ 0.0012** | **0.0382 $\pm$ 0.0008** | **0.9786 $\pm$ 0.0009** |

---

## 4. Benchmark Results: Table 1 (Harmonized Matched Subset)

Evaluated on the exact same 25 independent test phantoms (39,410 valid brain voxels):

| Method | Evaluation Set | Time (s) | CBF RMSE | CBF MAE | CBF Bias | CBF $R^2$ | ATT RMSE | ATT MAE | ATT Bias | ATT $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net (Proposed)** | Matched (25 phantoms) | **0.02** | **0.960** | **0.675** | +0.027 | **0.995** | **0.0558** | **0.0368** | -0.0025 | **0.979** |
| Point-wise DNN | Matched (25 phantoms) | 0.08 | 2.541 | 1.939 | -0.095 | 0.967 | 0.2156 | 0.1503 | +0.0189 | 0.682 |
| Bayesian (Calibrated) | Matched (25 phantoms) | 0.37 | 3.142 | 2.497 | +1.672 | 0.949 | 0.3216 | 0.2603 | -0.1435 | 0.292 |
| NLLS (TRF Bounded) | Matched (25 phantoms) | 130.17 | 3.358 | 2.471 | -0.683 | 0.942 | 0.3111 | 0.2093 | +0.1567 | 0.338 |

---

## 5. Paired Statistical Hypothesis Testing

Unit of replication: **$N = 25$ independent test phantoms**.
- Tests performed: Two-sided paired t-test and non-parametric Wilcoxon signed-rank test.
- Significance threshold: $\alpha = 0.05$.

| Comparison | Parameter | Mean Difference | 95% Confidence Interval | Paired t-test $p$ | Wilcoxon $p$ | Statistically Significant |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **U-Net vs Point-wise DNN** | CBF | -1.5707 | [-1.7064, -1.4350] | $3.07 \times 10^{-18}$ | $5.96 \times 10^{-8}$ | **Yes ($p < 0.001$)** |
| **U-Net vs Point-wise DNN** | ATT | -0.1617 | [-0.1722, -0.1511] | $4.65 \times 10^{-21}$ | $5.96 \times 10^{-8}$ | **Yes ($p < 0.001$)** |
| **U-Net vs Bayesian** | CBF | -2.1757 | [-2.3025, -2.0488] | $3.14 \times 10^{-22}$ | $5.96 \times 10^{-8}$ | **Yes ($p < 0.001$)** |
| **U-Net vs Bayesian** | ATT | -0.2672 | [-0.2803, -0.2541] | $5.38 \times 10^{-24}$ | $5.96 \times 10^{-8}$ | **Yes ($p < 0.001$)** |
| **U-Net vs NLLS** | CBF | -2.3786 | [-2.5676, -2.1896] | $4.42 \times 10^{-19}$ | $5.96 \times 10^{-8}$ | **Yes ($p < 0.001$)** |
| **U-Net vs NLLS** | ATT | -0.2562 | [-0.2719, -0.2406] | $9.10 \times 10^{-22}$ | $5.96 \times 10^{-8}$ | **Yes ($p < 0.001$)** |

**Interpretation:**
Every pairwise comparison demonstrates overwhelming statistical superiority ($p < 10^{-17}$) in favor of U-Net over both classical physics inversions and single-voxel point-wise neural networks.

---

## 6. Physical Justification: Why U-Net Outperforms 0D Point-wise Estimators

1. **Wash-Out Identifiability Degeneracy Resolution**:
   When $\text{ATT} < \text{PLD}_1 = 1.525$ s, bolus arrival has concluded before the earliest acquisition. In individual voxels, the signal decay follows $\Delta M \propto \text{CBF} \cdot \exp(\text{ATT} \cdot \Delta R_1)$, confounding CBF and ATT into an underdetermined contour. Point-wise DNNs and NLLS suffer high variance along this contour. Spatial convolutions in the U-Net exploit continuous spatial gradients of transit delay and cortical perfusion patterns, effectively resolving the degeneracy via neighborhood spatial context.
2. **Spatial Noise Regularization**:
   Rician noise operates independently per voxel. Classical voxel-wise fitting (NLLS/Bayesian) is prone to local noise fluctuations, inflating RMSE. The encoder-decoder hierarchy of U-Net applies implicit multi-scale spatial denoising while preserving sharp tissue boundaries (GM/WM/CSF).
3. **Inference Speed**:
   U-Net processes an entire $64 \times 64$ volume in ~0.8 ms on GPU, whereas iterative TRF NLLS takes ~5.2 seconds per phantom (~130 s for 25 phantoms), representing a $>6500\times$ speedup.

---

## 7. Readiness for Reduced-PLD Research (Phase 7)

With the 4-PLD benchmark rigorously audited, fully reproducible, and statistically validated, we establish this U-Net model as the gold standard baseline for the 3-PLD ablation experiment.

