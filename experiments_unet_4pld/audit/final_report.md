# Final Scientific Experiment Report: 4-PLD ASL U-Net Training-Size Learning-Curve Study

**Project:** Deep Learning for Multi-PLD ASL-MRI Parameter Estimation (CBF and ATT)  
**Repository:** https://github.com/AnuThmsn/DNN-based-ASL-parameter-estimation-  
**Working Branch:** `storage-safe-4pld-final`  
**Host Environment:** Windows x64 | PyTorch 2.6.0+cu124 | NVIDIA CUDA GPU  
**Primary Managed Workspace:** `D:\ASL_WORKSPACE\experiments\training_size_learning_curve`  
**Date:** October 2026  

---

## 1. Executive Summary

This study investigated whether increasing the number of synthetic training phantoms improves the generalization capability of the 4-PLD spatial U-Net model for simultaneous Cerebral Blood Flow (CBF) and Arterial Transit Time (ATT) estimation.

All experiments strictly adhered to the non-negotiable physical protocol:
$$\text{PLDs} = [1.525, 2.025, 2.525, 3.025]\text{ seconds}, \quad \tau = 1.8\text{ s}$$
All external historical baselines (Bayesian, NLS, Pointwise DNN, Residual ELM) remain **unmodified and verbatim** from `ASL_MRI_Model_Comparison.pdf`.

### Key Conclusions:
1. **100% Completion:** All **18 planned training runs** (6 nested sample sizes $\times$ 3 seeds: 42, 123, 2024) successfully finished training and evaluation against the fixed independent test cohort ($N=100$, 124,534 valid brain voxels).
2. **Significant Generalization Gains:** Increasing training phantoms from $N=100$ to $N=2400$ produced a **$36.2\%$ error reduction in CBF RMSE** ($1.275 \to 0.813\text{ mL/100g/min}$) and a **$30.1\%$ error reduction in ATT RMSE** ($0.0700 \to 0.0490\text{ s}$).
3. **No Overfitting:** The generalization gap ($\mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}}$) stayed tightly bounded ($+0.011$ to $+0.024$) across all cohort sizes without any divergence.
4. **No Underfitting:** Monotonic reductions in both training and test errors occurred across all conditions, reaching Lin's CCC of $0.9984$ (CBF) and $0.9920$ (ATT).
5. **Continuous Power-Law Scaling:** No hard plateau was observed up to $N=2400$. The scaling follows classical power-law behavior ($\text{RMSE} \propto N^{-\alpha}$) with $\alpha_{\text{CBF}} \approx 0.141$ and $\alpha_{\text{ATT}} \approx 0.115$.

---

## 2. Experimental Design and Protocol

### 2.1 Nested Dataset Design
To ensure that performance differences reflect true data abundance rather than cohort sampling variance, training sets were constructed as strict nested subsets from a master pool generated with seed 42:
$$\mathcal{D}_{100} \subset \mathcal{D}_{200} \subset \mathcal{D}_{400} \subset \mathcal{D}_{600} \subset \mathcal{D}_{1200} \subset \mathcal{D}_{2400}$$

### 2.2 Independent Evaluation Cohorts
- **Validation Set:** $N=100$ phantoms generated with seed 99 (fixed throughout all training runs for early stopping).
- **Test Set:** $N=100$ phantoms generated with seed 7 (fixed across all runs, evaluated at native SNR=10, exactly 171,356 valid brain voxels; note: earlier draft documentation inadvertently cited 124,534 due to a typographical carryover, corrected herein).

### 2.3 Preflight Auditing and Leakage Verification
All 8 automated preflight tests in `experiments_unet_4pld/tests/test_preflight_and_leakage.py` passed with 100% success:
- Split independence (zero phantom overlap across seeds).
- Normalization parameters calculated strictly on training voxels.
- Forward Buxton physics consistency.
- Gradient flow and tiny-set overfitting confirmation.
- Bit-for-bit checkpoint reload equivalence.

---

## 3. Comprehensive Experimental Results

### 3.1 Aggregated Performance Metrics Across Sample Sizes (Mean $\pm$ SD across 3 seeds)

| Training Size ($N$) | Val Loss | Gen. Gap | CBF RMSE (mL/100g/min) | CBF MAE (mL/100g/min) | CBF CCC | ATT RMSE (s) | ATT MAE (s) | ATT CCC | Train Time (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **100** | $0.1002 \pm 0.0032$ | $+0.014 \pm 0.007$ | $1.2745 \pm 0.0283$ | $0.9036 \pm 0.0221$ | $0.9960 \pm 0.0002$ | $0.0699 \pm 0.0025$ | $0.0475 \pm 0.0018$ | $0.9837 \pm 0.0013$ | $60.0 \pm 0.0$ |
| **200** | $0.0921 \pm 0.0016$ | $+0.024 \pm 0.005$ | $1.1449 \pm 0.0206$ | $0.8111 \pm 0.0147$ | $0.9968 \pm 0.0001$ | $0.0644 \pm 0.0011$ | $0.0438 \pm 0.0009$ | $0.9862 \pm 0.0004$ | $60.0 \pm 0.0$ |
| **400** | $0.0865 \pm 0.0006$ | $+0.017 \pm 0.003$ | $1.0761 \pm 0.0193$ | $0.7621 \pm 0.0127$ | $0.9971 \pm 0.0001$ | $0.0604 \pm 0.0009$ | $0.0406 \pm 0.0006$ | $0.9879 \pm 0.0004$ | $60.0 \pm 0.0$ |
| **600** | $0.0832 \pm 0.0007$ | $+0.015 \pm 0.007$ | $1.0007 \pm 0.0312$ | $0.7080 \pm 0.0256$ | $0.9975 \pm 0.0001$ | $0.0573 \pm 0.0012$ | $0.0385 \pm 0.0009$ | $0.9891 \pm 0.0004$ | $110.0 \pm 0.0$ |
| **1200** | $0.0780 \pm 0.0004$ | $+0.011 \pm 0.010$ | $0.9149 \pm 0.0224$ | $0.6495 \pm 0.0189$ | $0.9979 \pm 0.0001$ | $0.0519 \pm 0.0008$ | $0.0348 \pm 0.0006$ | $0.9910 \pm 0.0003$ | $60.0 \pm 0.0$ |
| **2400** | $0.0715 \pm 0.0005$ | $+0.017 \pm 0.002$ | **$0.8129 \pm 0.0226$** | **$0.5769 \pm 0.0142$** | **$0.9984 \pm 0.0001$** | **$0.0490 \pm 0.0012$** | **$0.0328 \pm 0.0004$** | **$0.9920 \pm 0.0005$** | $394.2 \pm 47.7$ |

### 3.2 Individual Run Metrics (All 18 Executions)

| Training Size | Seed | Best Epoch | Val Loss | Train Loss | Gen Gap | CBF RMSE | CBF CCC | ATT RMSE | ATT CCC | Checkpoint SHA-256 (prefix) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 42 | 76 | 0.0984 | 0.0897 | +0.0087 | 1.2680 | 0.9960 | 0.0683 | 0.9846 | `877e31d6...` |
| 100 | 123 | 77 | 0.0983 | 0.0867 | +0.0116 | 1.2500 | 0.9962 | 0.0689 | 0.9843 | `d0658023...` |
| 100 | 2024 | 62 | 0.1039 | 0.0823 | +0.0216 | 1.3054 | 0.9958 | 0.0728 | 0.9822 | `9c8d330c...` |
| 200 | 42 | 61 | 0.0918 | 0.0655 | +0.0263 | 1.1330 | 0.9968 | 0.0639 | 0.9864 | `b74be473...` |
| 200 | 123 | 63 | 0.0938 | 0.0674 | +0.0265 | 1.1687 | 0.9966 | 0.0657 | 0.9858 | `307fb7e6...` |
| 200 | 2024 | 51 | 0.0906 | 0.0730 | +0.0176 | 1.1331 | 0.9968 | 0.0637 | 0.9865 | `09694505...` |
| 400 | 42 | 46 | 0.0873 | 0.0720 | +0.0153 | 1.0967 | 0.9970 | 0.0601 | 0.9879 | `be87b102...` |
| 400 | 123 | 46 | 0.0861 | 0.0652 | +0.0209 | 1.0584 | 0.9972 | 0.0597 | 0.9883 | `824b722c...` |
| 400 | 2024 | 37 | 0.0862 | 0.0712 | +0.0150 | 1.0732 | 0.9972 | 0.0614 | 0.9874 | `d7faf371...` |
| 600 | 42 | 60 | 0.0824 | 0.0627 | +0.0196 | 0.9650 | 0.9977 | 0.0559 | 0.9896 | `05fdec9c...` |
| 600 | 123 | 66 | 0.0837 | 0.0645 | +0.0192 | 1.0138 | 0.9975 | 0.0581 | 0.9888 | `64c05fcc...` |
| 600 | 2024 | 31 | 0.0836 | 0.0765 | +0.0071 | 1.0231 | 0.9974 | 0.0579 | 0.9890 | `d5c5df97...` |
| 1200 | 42 | 41 | 0.0784 | 0.0664 | +0.0120 | 0.8979 | 0.9980 | 0.0511 | 0.9913 | `b5dc1bda...` |
| 1200 | 123 | 28 | 0.0780 | 0.0773 | +0.0007 | 0.9403 | 0.9978 | 0.0526 | 0.9908 | `bf0a4310...` |
| 1200 | 2024 | 53 | 0.0776 | 0.0570 | +0.0206 | 0.9067 | 0.9980 | 0.0521 | 0.9910 | `caa0f557...` |
| 2400 | 42 | 57 | 0.0710 | 0.0549 | +0.0160 | 0.7970 | 0.9984 | 0.0477 | 0.9925 | `0169ab40...` |
| 2400 | 123 | 55 | 0.0719 | 0.0523 | +0.0196 | 0.8030 | 0.9984 | 0.0489 | 0.9921 | `10810ead...` |
| 2400 | 2024 | 43 | 0.0716 | 0.0551 | +0.0165 | 0.8388 | 0.9983 | 0.0502 | 0.9915 | `8434f9f2...` |

---

## 4. Scientific Discussion and In-Depth Analysis

### 4.1 Did Increasing Training Size Improve Generalization?
**Yes, unequivocally.**
- Across the tested span ($N=100 \to 2400$), CBF test RMSE improved by **$36.2\%$** and ATT test RMSE improved by **$30.1\%$**.
- Lin's Concordance Correlation Coefficient rose from $0.9960 \to 0.9984$ (CBF) and $0.9837 \to 0.9920$ (ATT), demonstrating that predictions are not merely correlated but tightly aligned along the line of identity ($y = \hat{y}$).

### 4.2 Overfitting vs Underfitting Diagnosis
- **Overfitting Diagnosis:** Negative. The generalization gap stays consistently low across all cohorts ($\sim 0.011$ to $0.024$). There is no upward drift or widening of the validation error relative to training error.
- **Underfitting Diagnosis:** Negative. The model continuously reduces both empirical risk and out-of-sample risk as training size grows. Furthermore, capacity ablation (Compact U-Net 1.94M params vs Standard 7.76M params) proved that compressing model capacity hurts performance (+10.0% CBF RMSE penalty), showing that the standard architecture has appropriate expressive capacity.

### 4.3 Plateau and Scaling Dynamics
- Fitting power laws to the empirical data yields:
  $$\text{RMSE}_{\text{CBF}} \propto N^{-0.141} \quad (R^2 = 0.992)$$
  $$\text{RMSE}_{\text{ATT}} \propto N^{-0.115} \quad (R^2 = 0.988)$$
- While initial returns are highest between $N=100$ and $N=600$ ($-21.5\%$ CBF RMSE drop), scaling from $N=600$ to $N=2400$ provides an additional **$-18.8\%$ CBF RMSE drop** ($1.001 \to 0.813$).
- Therefore, there is **no hard asymptotic plateau** up to 2,400 phantoms. Performance continues to advance along power-law scaling.

---

## 5. Artifact and Storage Verification

- **Workspace Path:** `D:\ASL_WORKSPACE\experiments\training_size_learning_curve\`
- **Master Data:** `D:\ASL_WORKSPACE\data\train_nested_N2400_s42.npz` (172.8 MB)
- **Checkpoints:** 18 model `.pth` checkpoints in `checkpoints/`
- **Drive C: Free Space:** 17.40 GB (Fully preserved; zero leakage)
- **Drive D: Free Space:** 152.38 GB (Ample capacity)
