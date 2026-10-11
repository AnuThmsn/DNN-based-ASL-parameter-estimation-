# Training-Set Size Generalization Study: Experiment Plan & Protocol

**Project:** ASL-MRI Parameter Estimation (Four-PLD Spatial U-Net)  
**Date:** October 2026  
**Working Branch:** `storage-safe-4pld-final`  
**Workspace:** `D:\ASL_WORKSPACE\experiments\training_size_study`  

---

## 1. Scientific Objective

Investigate whether the baseline training cohort of **600 spatial phantoms** provides sufficient sample diversity for spatial U-Net CBF and ATT estimation, or whether the model is:
- **Underfitting:** High training and validation errors across epochs, failing to capture subtle hemodynamic patterns.
- **Overfitting:** Training loss continues to drop while validation and test error plateau or diverge, suggesting excessive capacity relative to data size.
- **Generalizing Adequately:** Both training and validation errors decrease gracefully with stable generalization gaps and strong concordance on independent tests.

---

## 2. Experimental Protocol & Nested Hierarchy

### Fixed Independent Evaluation Cohorts (Unchanged Across All Sizes)
- **Validation Cohort:** $N = 100$ phantoms, `val_seed = 99` (`D:\ASL_WORKSPACE\data\val_N100_s99.npz`).
- **Test Cohort:** $N = 100$ phantoms, `test_seed = 7` (`D:\ASL_WORKSPACE\data\test_N100_s7.npz`).
- **Target Metrics:** CBF RMSE, MAE, Bias, $R^2$, and Lin's CCC; ATT RMSE, MAE, Bias, $R^2$, and Lin's CCC (all in physical units on valid brain parenchyma).

### Strictly Nested Training Subsets
All training subsets are generated deterministically from seed 42 with sequential nesting:
$$\mathcal{D}_{\text{train}}^{100} \subset \mathcal{D}_{\text{train}}^{200} \subset \mathcal{D}_{\text{train}}^{400} \subset \mathcal{D}_{\text{train}}^{600} \subset \mathcal{D}_{\text{train}}^{1200}$$
- $N=100$: Phantoms $[0:100]$
- $N=200$: Phantoms $[0:200]$
- $N=400$: Phantoms $[0:400]$
- $N=600$: Phantoms $[0:600]$ (Verified Authoritative Baseline)
- $N=1200$: Phantoms $[0:1200]$ (Extended Scale)

### Repetition Seeds
For each training size, 3 independent random seeds are trained:
$$\text{Seeds} \in \{42, 123, 2024\}$$
Total model runs: 5 sizes $\times$ 3 seeds = **15 runs**.

---

## 3. Storage Safety & Cost Budget

| Condition | Training Phantoms | Runs (Seeds) | Estimated Train Time / Run | Est. Total GPU Time | Checkpoint Storage | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$N = 600$** | 600 | 3 (42, 123, 2024) | ~110 s | 0 s (Reused existing) | 279.8 MB (Pre-existing) | **Verified Complete** |
| **$N = 100$** | 100 | 3 (42, 123, 2024) | ~25 s | ~75 s | ~279.8 MB | **Ready to Run** |
| **$N = 200$** | 200 | 3 (42, 123, 2024) | ~45 s | ~135 s | ~279.8 MB | **Ready to Run** |
| **$N = 400$** | 400 | 3 (42, 123, 2024) | ~80 s | ~240 s | ~279.8 MB | **Ready to Run** |
| **$N = 1200$** | 1200 | 3 (42, 123, 2024) | ~210 s | ~630 s | ~279.8 MB | **Ready to Run** |
| **$N = 2400$** | 2400 | Optional staged | ~450 s | ~1350 s | ~279.8 MB | **Planned Secondary / Contingent** |

- **Total Planned Active GPU Compute:** ~18 minutes.
- **Total New Storage on D::** ~1.12 GB (Checkpoints + Histories + Manifests).
- **Free Space Available on D::** 145.10 GB (Ample, >100x margin).
- **Free Space on C::** 16.40 GB (Zero bytes written to C:).

---

## 4. Execution Staging Strategy

1. **Stage 1 (Baseline Verification):** Load and audit existing $N=600$ checkpoints (`unet_best_seed_{42,123,2024}.pth`) and logs; confirm metrics against test set.
2. **Stage 2 (Sub-sample Training):** Execute nested subsets $N=100, 200, 400$ across seeds 42, 123, 2024.
3. **Stage 3 (Scale Extension):** Generate nested phantoms up to $N=1200$ and train across seeds 42, 123, 2024.
4. **Stage 4 (Evaluation & Aggregation):** Evaluate every completed model checkpoint on the identical fixed 100-phantom test set.
5. **Stage 5 (Synthesis & Analysis):** Generate learning curves, test metric scaling curves (mean $\pm$ std), data manifest, and evidence-based overfitting/underfitting diagnostic report.

