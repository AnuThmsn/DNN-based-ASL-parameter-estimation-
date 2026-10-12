# Comprehensive Scientific Audit Report: 4-PLD ASL U-Net Parameter Estimation

**Repository:** `https://github.com/AnuThmsn/DNN-based-ASL-parameter-estimation-`  
**Branch:** `storage-safe-4pld-final`  
**Workspace:** `D:\ASL_WORKSPACE`  
**Audit Directory:** `D:\ASL_WORKSPACE\experiments\scientific_audit`  
**Date of Audit:** October 12, 2026  
**Audit Outcome:** **ALL 10 REGRESSION TESTS PASSED; DOCUMENTATION RECONCILED**

---

## 1. Executive Summary & Audit Overview

A comprehensive scientific and technical audit was conducted on the 4-PLD ASL-DNN project across all codebase assets, datasets, checkpoints, statistical evaluations, and scientific claims.

### Summary of Actions & Corrections:
1. **Decommissioned Fragile Leakage Check:** Replaced the single-coordinate check (`X[:, 0, 32, 32]`) in `verify_learning_curve_experiment.py` with full SHA-256 sample digests over `(X[i], Y[i], masks[i])` in `cohort_integrity_check.py`. Zero data leakage confirmed across all 2,400 train, 100 val, and 100 test phantoms ($100\%$ unique, $0$ overlaps).
2. **Resolved Evaluation Voxel Discrepancy:** Traced the conflicting mention of 124,534 valid brain voxels to a manual typing carryover in markdown table headers (transcript step 2163). Confirmed via array inspection that the canonical test dataset `test_N100_s7.npz` contains exactly **171,356 valid brain voxels** ($\sum \text{mask} > 0.5$). Corrected all markdown references.
3. **Reconciled Experiment Path Duality:** Demonstrated bit-for-bit identity across all 18 checkpoints (`18/18` SHA-256 matches) and all 18 training histories (`18/18` SHA-256 matches) between `D:\ASL_WORKSPACE\experiments\training_size_study` and `D:\ASL_WORKSPACE\experiments\training_size_learning_curve`.
4. **Verified Metric Integrity:** Verified normalization isolation to training data, exact denormalization scaling, physical clipping bounds (CBF: $[0, 100]$ mL/100g/min, ATT: $[0.5, 3.0]$ s), and metric consistency for RMSE, MAE, Bias, $R^2$, and Lin's CCC.
5. **Addressed Statistical Pseudoreplication:** Disclosed that voxel-level pooling ($N = 171,356$) is spatial pseudoreplication. Re-anchored inferential hypothesis testing to independent subject phantoms ($N = 100$), confirming robust statistical significance ($p < 10^{-8}$) while qualifying seed-level variation ($N=3, df=4$).
6. **Audited 3-PLD Ablation:** Confirmed input channel selections in `run_3pld_ablation.py`. Replaced claims of "proven clinical equivalence" with **"descriptive comparability within empirical bounds"** due to absence of pre-registered TOST equivalence margins.
7. **Disclosed Benchmark Timing Limitations:** Disclosed that reported inference speedup ($>6500\times$) reflects GPU-batched tensor operations (U-Net, Bayesian) vs CPU-sequential iterative solving (SciPy NLLS), rather than pure algorithmic speed on identical hardware.
8. **Automated Regression Test Suite:** Created `experiments_unet_4pld/tests/test_scientific_audit_regression.py` containing 10 rigorous unit tests covering all audit dimensions. Executed with a **100% pass rate** (10/10 passed).

---

## 2. Classification of Scientific Claims

Every major claim made in project documentation was evaluated and classified into one of four categories: **Verified**, **Qualified**, **Unsupported**, or **Unresolved**.

| Item | Topic / Claim | Original Phrasing | Audit Classification | Justification & Corrected Scientific Stance |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Cohort Independence** | "Zero data leakage between train and test" | **VERIFIED** | Full SHA-256 digests over `(X, Y, mask)` show strictly zero overlap across all 2,400 train, 100 val, and 100 test phantoms. Negative control verified. |
| **2** | **Test Voxel Count** | "124,534 valid brain voxels" | **CORRECTED** | Corrected to **171,356 valid brain voxels**. 124,534 was a typographical error in an earlier markdown draft; no 124,534 dataset ever existed. |
| **3** | **Checkpoints & Histories** | "All 18 models complete and verified" | **VERIFIED** | 18 checkpoints and 18 training histories verified bit-for-bit across both study directories ($18/18$ SHA-256 matches). |
| **4** | **Data Scaling Utility** | "Increasing training size improves test performance" | **VERIFIED** | Monotonic test error reduction from $N=100 \to 2400$ (CBF RMSE: $1.275 \to 0.813$, ATT RMSE: $0.0700 \to 0.0490$). Paired phantom $t$-test: $p < 10^{-30}$. |
| **5** | **Absence of Overfitting** | "Overfitting is eliminated" | **QUALIFIED** | Generalization gap remains stably bounded ($+0.011 \le \text{gap} \le +0.024$) on synthetic distribution, but does not guarantee generalizability to out-of-distribution clinical pathology. |
| **6** | **Universal Power Law** | "Learning curve obeys universal power law" | **QUALIFIED** | The relationship $\text{RMSE} \propto N^{-\alpha}$ is an **empirical fit over $N \in [100, 2400]$**, not an asymptotic physical law. |
| **7** | **Three-PLD Equivalence** | "Subset 1 is clinically equivalent within +10%" | **QUALIFIED** | The $\pm 10\%$ bound was an informal heuristic. Claim qualified to **"descriptive comparability within empirical bounds"**. Primary model remains strictly 4-PLD. |
| **8** | **Timing Superiority** | "U-Net is $>6500\times$ faster than NLLS" | **QUALIFIED** | Timing reflects GPU-batched tensor inference vs CPU-sequential SciPy TRF solver, not identical hardware complexity. Disclosed as an operational comparison. |
| **9** | **Physical Degeneracy** | "Spatial context eliminates physical CBF/ATT degeneracy" | **QUALIFIED** | Spatial convolutions regularize local error variance across anatomical neighbors, but do not alter Buxton signal physics or remove late-arrival non-identifiability. |
| **10** | **Historical Baselines** | "Bayesian/NLS comparison numbers preserved" | **VERIFIED** | Historical literature numbers from `ASL_MRI_Model_Comparison.pdf` were strictly preserved without retraining or re-simulation. |

---

## 3. Verified Per-Run Experimental Results Table ($N = 18$ Completed Runs)

Compiled from verified checkpoints in `D:\ASL_WORKSPACE\experiments\scientific_audit\verified_per_run_results.csv`:

| Training Size ($N$) | Random Seed | Best Epoch | Train Loss | Val Loss | Gen. Gap | Test CBF RMSE (mL/100g/min) | Test CBF CCC | Test ATT RMSE (s) | Test ATT CCC | Checkpoint SHA-256 Prefix | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 100 | 42 | 76 | 0.0897 | 0.0984 | +0.0087 | 1.2680 | 0.9960 | 0.0683 | 0.9846 | `877e31d61498...` | **VERIFIED** |
| 100 | 123 | 77 | 0.0867 | 0.0983 | +0.0116 | 1.2500 | 0.9962 | 0.0689 | 0.9843 | `d065802392d1...` | **VERIFIED** |
| 100 | 2024 | 62 | 0.0823 | 0.1039 | +0.0216 | 1.3054 | 0.9958 | 0.0728 | 0.9822 | `9c8d330cc8e0...` | **VERIFIED** |
| 200 | 42 | 61 | 0.0655 | 0.0918 | +0.0263 | 1.1330 | 0.9968 | 0.0639 | 0.9864 | `b74be4737b2e...` | **VERIFIED** |
| 200 | 123 | 63 | 0.0674 | 0.0938 | +0.0265 | 1.1687 | 0.9966 | 0.0657 | 0.9858 | `307fb7e666d3...` | **VERIFIED** |
| 200 | 2024 | 51 | 0.0730 | 0.0906 | +0.0176 | 1.1331 | 0.9968 | 0.0637 | 0.9865 | `0969450562d3...` | **VERIFIED** |
| 400 | 42 | 46 | 0.0720 | 0.0873 | +0.0153 | 1.0967 | 0.9970 | 0.0601 | 0.9879 | `be87b102fb02...` | **VERIFIED** |
| 400 | 123 | 46 | 0.0652 | 0.0861 | +0.0209 | 1.0584 | 0.9972 | 0.0597 | 0.9883 | `824b722c39d1...` | **VERIFIED** |
| 400 | 2024 | 37 | 0.0712 | 0.0862 | +0.0150 | 1.0732 | 0.9972 | 0.0614 | 0.9874 | `d7faf3719460...` | **VERIFIED** |
| 600 | 42 | 60 | 0.0627 | 0.0824 | +0.0196 | 0.9650 | 0.9977 | 0.0559 | 0.9896 | `05fdec9c3d33...` | **VERIFIED** |
| 600 | 123 | 66 | 0.0645 | 0.0837 | +0.0192 | 1.0138 | 0.9975 | 0.0581 | 0.9888 | `64c05fccd866...` | **VERIFIED** |
| 600 | 2024 | 31 | 0.0765 | 0.0836 | +0.0071 | 1.0231 | 0.9974 | 0.0579 | 0.9890 | `d5c5df9794f9...` | **VERIFIED** |
| 1200 | 42 | 41 | 0.0664 | 0.0784 | +0.0120 | 0.8979 | 0.9980 | 0.0511 | 0.9912 | `b5dc1bdaaff9...` | **VERIFIED** |
| 1200 | 123 | 28 | 0.0773 | 0.0780 | +0.0007 | 0.9403 | 0.9978 | 0.0526 | 0.9907 | `bf0a4310e74c...` | **VERIFIED** |
| 1200 | 2024 | 53 | 0.0570 | 0.0776 | +0.0206 | 0.9067 | 0.9980 | 0.0522 | 0.9910 | `caa0f557e07b...` | **VERIFIED** |
| 2400 | 42 | 57 | 0.0549 | 0.0710 | +0.0160 | 0.7970 | 0.9984 | 0.0478 | 0.9925 | `0169ab40ebb4...` | **VERIFIED** |
| 2400 | 123 | 55 | 0.0523 | 0.0719 | +0.0196 | 0.8030 | 0.9984 | 0.0489 | 0.9920 | `10810ead787e...` | **VERIFIED** |
| 2400 | 2024 | 43 | 0.0551 | 0.0716 | +0.0165 | 0.8388 | 0.9983 | 0.0502 | 0.9915 | `8434f9f230c8...` | **VERIFIED** |

---

## 4. Grouped Learning-Curve Comparison Table (Mean $\pm$ SD across 3 seeds)

Compiled in `D:\ASL_WORKSPACE\experiments\scientific_audit\verified_grouped_comparison.csv`:

| Training Size ($N$) | Val Loss | Gen. Gap | Test CBF RMSE (mL/100g/min) | Test CBF MAE | Test CBF CCC | Test ATT RMSE (s) | Test ATT MAE | Test ATT CCC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | $0.1002 \pm 0.0032$ | $+0.0140 \pm 0.0068$ | **$1.2745 \pm 0.0283$** | $0.9036 \pm 0.0221$ | $0.9960 \pm 0.0002$ | **$0.0700 \pm 0.0025$** | $0.0476 \pm 0.0018$ | $0.9837 \pm 0.0013$ |
| **200** | $0.0921 \pm 0.0016$ | $+0.0235 \pm 0.0051$ | **$1.1449 \pm 0.0206$** | $0.8111 \pm 0.0147$ | $0.9968 \pm 0.0001$ | **$0.0645 \pm 0.0011$** | $0.0439 \pm 0.0009$ | $0.9862 \pm 0.0004$ |
| **400** | $0.0865 \pm 0.0006$ | $+0.0171 \pm 0.0034$ | **$1.0761 \pm 0.0193$** | $0.7621 \pm 0.0127$ | $0.9971 \pm 0.0001$ | **$0.0604 \pm 0.0009$** | $0.0406 \pm 0.0007$ | $0.9879 \pm 0.0004$ |
| **600** | $0.0832 \pm 0.0007$ | $+0.0153 \pm 0.0071$ | **$1.0007 \pm 0.0312$** | $0.7080 \pm 0.0256$ | $0.9975 \pm 0.0001$ | **$0.0573 \pm 0.0012$** | $0.0385 \pm 0.0009$ | $0.9891 \pm 0.0004$ |
| **1200** | $0.0780 \pm 0.0004$ | $+0.0111 \pm 0.0099$ | **$0.9149 \pm 0.0224$** | $0.6495 \pm 0.0189$ | $0.9979 \pm 0.0001$ | **$0.0519 \pm 0.0008$** | $0.0348 \pm 0.0006$ | $0.9910 \pm 0.0003$ |
| **2400** | $0.0715 \pm 0.0005$ | $+0.0174 \pm 0.0020$ | **$0.8129 \pm 0.0226$** | $0.5769 \pm 0.0142$ | $0.9984 \pm 0.0001$ | **$0.0490 \pm 0.0012$** | $0.0329 \pm 0.0004$ | $0.9920 \pm 0.0005$ |

---

## 5. Automated Regression Test Execution Record

The regression test suite `experiments_unet_4pld/tests/test_scientific_audit_regression.py` was executed directly in Python:
- **Command:** `python experiments_unet_4pld/tests/test_scientific_audit_regression.py -v`
- **Result:** **10 tests ran in 18.517s — 0 failures, 0 errors (OK)**

1. `test_01_cohort_cryptographic_independence`: **PASSED** (Zero overlap across splits)
2. `test_02_leakage_detection_negative_control`: **PASSED** (Injected duplicate detected)
3. `test_03_nested_training_subset_integrity`: **PASSED** (Strict prefix inclusion verified)
4. `test_04_normalization_isolated_to_training`: **PASSED** (Statistics isolated from test data)
5. `test_05_denormalization_and_physical_clamping`: **PASSED** (Physical bounds clamping verified)
6. `test_06_checkpoint_and_history_provenance_duality`: **PASSED** ($18/18$ identical SHA-256 matches)
7. `test_07_fixed_test_cohort_hash_and_voxel_count`: **PASSED** (SHA-256 match, 171,356 valid voxels)
8. `test_08_metric_recomputation_consistency`: **PASSED** (RMSE, MAE, Lin's CCC mathematical formulas)
9. `test_09_three_pld_ablation_channel_selection_and_physics`: **PASSED** (Input channels & physical breakdown)
10. `test_10_reproducibility_manifest_completeness`: **PASSED** (18 runs completely documented)

---

## 6. Remaining Scientific Limitations

1. **Synthetic Phantom vs In-Vivo Complexity:** Data generation relies on single-compartment Buxton physics with discrete tissue boundaries and smooth perfusion gradients. Real clinical in-vivo scans feature vascular tortuosity, dispersion, magnetization transfer, and motion artifacts not modeled here.
2. **Fixed Protocol Scope:** The current network is optimized strictly for $[1.525, 2.025, 2.525, 3.025]$ s PLDs and label duration $\tau = 1.8$ s; clinical application with alternative PLD timings requires retrained or physics-adaptive networks.
3. **Seed Cardinality:** Three random seeds ($N=3$) establish stability against typical gradient descent stochasticity, but cannot rule out rare local minima or model pathological failure modes.
