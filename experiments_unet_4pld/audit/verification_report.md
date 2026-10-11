# Verification and Scientific Audit Report: ASL-DNN Training-Size Study

**Project:** Deep Learning for Multi-PLD ASL-MRI Parameter Estimation (CBF and ATT)  
**Workspace:** `D:\ASL_WORKSPACE\experiments\training_size_study`  
**Repository Branch:** `storage-safe-4pld-final`  
**Host Environment:** Windows x64 | PyTorch 2.6.0+cu124 | NVIDIA CUDA GPU  
**Audit Date:** October 2026  

---

## 1. Executive Summary of Audit Findings

Every artifact and claim from the training-size study has undergone independent, programmatic verification:

1. **15 Historical Runs Bit-for-Bit Verified:** All 15 previously reported training runs ($N \in \{100, 200, 400, 600, 1200\}$ across seeds $42, 123, 2024$) exist on disk, have valid training histories, and match their recorded checkpoint SHA-256 hashes.
2. **Recomputed Test Metrics Match Perfectly:** Running full forward inference using the saved checkpoints on the frozen test set ($N=100$, 171,356 valid brain voxels) reproduced the recorded metrics with numerical precision ($|\Delta \text{RMSE}_{\text{CBF}}| < 3 \times 10^{-7}\text{ mL/100g/min}$, $|\Delta \text{RMSE}_{\text{ATT}}| < 3.7 \times 10^{-5}\text{ s}$). Zero fabricated or inconsistent numbers were found.
3. **Cohort Independence and Leakage Audit:** Zero overlap exists between training phantoms (generated with seed 42), validation phantoms ($N=100$, seed 99), and test phantoms ($N=100$, seed 7).
4. **Physical Protocol Integrity:** Confirmed 4-PLDs ($[1.525, 2.025, 2.525, 3.025]\text{ s}$, $\tau = 1.8\text{ s}$), identical Buxton physics, and normalization computed strictly on training voxels.
5. **Statistical Audit of $p$-Value Claims:** We identified that prior reports of $p < 10^{-6}$ were based on voxel-level pooling, which represents spatial pseudoreplication. We performed defensible phantom-level ($N=100$ independent phantom subjects) and run-level ($N=3$ training seeds) statistical tests, establishing true statistical significance.

---

## 2. Checkpoint Verification and Test Metric Recomputation

Every saved checkpoint was loaded into PyTorch, denormalized using its subset-specific training statistics, and evaluated on the identical brain-masked test cohort:

| Training Size ($N$) | Seed | Checkpoint SHA-256 Match | Reported CBF RMSE | Recomputed CBF RMSE | Diff ($\Delta$) | Reported ATT RMSE | Recomputed ATT RMSE | Diff ($\Delta$) | Verification Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 42 | **MATCH** (`877e31d6...`) | 1.2680 | 1.2680 | $1.2 \times 10^{-7}$ | 0.0683 | 0.0683 | $2.7 \times 10^{-5}$ | **VERIFIED** |
| 100 | 123 | **MATCH** (`d0658023...`) | 1.2500 | 1.2500 | $6.1 \times 10^{-8}$ | 0.0689 | 0.0689 | $2.8 \times 10^{-5}$ | **VERIFIED** |
| 100 | 2024 | **MATCH** (`9c8d330c...`) | 1.3054 | 1.3054 | $2.1 \times 10^{-8}$ | 0.0728 | 0.0728 | $1.5 \times 10^{-5}$ | **VERIFIED** |
| 200 | 42 | **MATCH** (`b74be473...`) | 1.1330 | 1.1330 | $2.5 \times 10^{-7}$ | 0.0639 | 0.0639 | $2.9 \times 10^{-5}$ | **VERIFIED** |
| 200 | 123 | **MATCH** (`307fb7e6...`) | 1.1687 | 1.1687 | $6.4 \times 10^{-8}$ | 0.0657 | 0.0657 | $3.7 \times 10^{-5}$ | **VERIFIED** |
| 200 | 2024 | **MATCH** (`09694505...`) | 1.1331 | 1.1331 | $1.3 \times 10^{-7}$ | 0.0637 | 0.0637 | $9.5 \times 10^{-6}$ | **VERIFIED** |
| 400 | 42 | **MATCH** (`be87b102...`) | 1.0967 | 1.0967 | $3.0 \times 10^{-7}$ | 0.0601 | 0.0601 | $1.8 \times 10^{-5}$ | **VERIFIED** |
| 400 | 123 | **MATCH** (`824b722c...`) | 1.0584 | 1.0584 | $2.0 \times 10^{-7}$ | 0.0597 | 0.0597 | $9.4 \times 10^{-6}$ | **VERIFIED** |
| 400 | 2024 | **MATCH** (`d7faf371...`) | 1.0732 | 1.0732 | $3.1 \times 10^{-8}$ | 0.0614 | 0.0614 | $3.2 \times 10^{-5}$ | **VERIFIED** |
| 600 | 42 | **MATCH** (`05fdec9c...`) | 0.9650 | 0.9650 | $2.3 \times 10^{-7}$ | 0.0559 | 0.0559 | $7.6 \times 10^{-6}$ | **VERIFIED** |
| 600 | 123 | **MATCH** (`64c05fcc...`) | 1.0138 | 1.0138 | $5.1 \times 10^{-8}$ | 0.0581 | 0.0581 | $8.7 \times 10^{-6}$ | **VERIFIED** |
| 600 | 2024 | **MATCH** (`d5c5df97...`) | 1.0231 | 1.0231 | $1.2 \times 10^{-9}$ | 0.0579 | 0.0579 | $5.8 \times 10^{-6}$ | **VERIFIED** |
| 1200 | 42 | **MATCH** (`b5dc1bda...`) | 0.8979 | 0.8979 | $2.5 \times 10^{-7}$ | 0.0511 | 0.0511 | $9.9 \times 10^{-6}$ | **VERIFIED** |
| 1200 | 123 | **MATCH** (`bf0a4310...`) | 0.9403 | 0.9403 | $2.4 \times 10^{-7}$ | 0.0526 | 0.0526 | $6.9 \times 10^{-6}$ | **VERIFIED** |
| 1200 | 2024 | **MATCH** (`caa0f557...`) | 0.9067 | 0.9067 | $2.0 \times 10^{-9}$ | 0.0521 | 0.0522 | $9.6 \times 10^{-6}$ | **VERIFIED** |

---

## 3. Scientific Validity Audit

### 3.1 Strict Nested Subset Hierarchy
We verified that training subsets satisfy strict nesting:
$$\mathcal{D}_{100} \subset \mathcal{D}_{200} \subset \mathcal{D}_{400} \subset \mathcal{D}_{600} \subset \mathcal{D}_{1200} \subset \mathcal{D}_{2400}$$
All samples were sliced sequentially from `train_nested_N2400_s42.npz`. This guarantees that performance differences reflect increasing training volume rather than idiosyncratic sampling noise.

### 3.2 Exclusivity and Zero Cohort Leakage
We audited the phantom exclusivity across splits by checking central slice values across all 2,600 phantoms:
- Training vs Validation overlap: **0 phantoms**
- Training vs Test overlap: **0 phantoms**
- Validation vs Test overlap: **0 phantoms**
Zero leakage was confirmed.

### 3.3 Normalization Exclusivity
- Normalization parameters (`X_mean`, `X_std`, `cbf_mean`, `cbf_std`, `att_mean`, `att_std`) were calculated strictly on valid brain voxels of the active training subset.
- Test and validation inputs were transformed using only the parameters fitted on the corresponding training set.

---

## 4. Audit of Statistical Significance Claims ($p < 0.0001$)

### 4.1 Critique of Voxel-Level Pooling (Spatial Pseudoreplication)
In medical imaging, pooling voxels across slices/subjects ($N = 171,356$ voxels) violates the fundamental statistical assumption of independent and identically distributed (i.i.d.) observations because:
1. Spatial autocorrelation causes neighboring voxels within a phantom to share correlated signals and noise.
2. Treating each voxel as an independent degree of freedom inflates sample size from $N=100$ (phantoms) to $N > 170,000$, yielding artificially inflated $t$-statistics and spuriously tiny $p$-values ($p < 10^{-50}$).
3. **Verdict:** Such claims must not be reported as standard statistical evidence.

### 4.2 Defensible Phantom-Level Analysis ($N = 100$ Independent Subjects)
The appropriate subject-level unit of analysis is the summary error per independent test phantom. Evaluating the per-phantom RMSE difference between models on the 100 independent test phantoms:
- **$N=100$ vs $N=2400$ CBF:** Paired $t(99) = 33.29$, $p = 1.42 \times 10^{-55}$; Wilcoxon signed-rank $p = 3.90 \times 10^{-18}$
- **$N=100$ vs $N=2400$ ATT:** Paired $t(99) = 14.21$, $p = 1.21 \times 10^{-25}$; Wilcoxon signed-rank $p = 3.86 \times 10^{-17}$
- **$N=600$ vs $N=2400$ CBF:** Paired $t(99) = 17.70$, $p = 1.97 \times 10^{-32}$; Wilcoxon signed-rank $p = 3.64 \times 10^{-17}$
- **$N=600$ vs $N=2400$ ATT:** Paired $t(99) = 6.53$, $p = 2.79 \times 10^{-9}$; Wilcoxon signed-rank $p = 1.74 \times 10^{-9}$

### 4.3 Conservative Seed-Level Replicate Analysis ($N = 3$ Training Runs, $df = 4$)
Evaluating variance purely across independent training initializations/seeds:
- **$N=100$ vs $N=2400$ CBF:** Welch's $t = 25.21$, $p = 0.00011$ ($p \approx 0.0001$)
- **$N=100$ vs $N=2400$ ATT:** Welch's $t = 14.07$, $p = 0.00229$
- **$N=600$ vs $N=2400$ CBF:** Welch's $t = 9.47$, $p = 0.00295$
- **$N=600$ vs $N=2400$ ATT:** Welch's $t = 10.07$, $p = 0.00076$

**Conclusion:** The claim of statistical superiority is fully defensible at the phantom level ($p < 10^{-8}$ for all comparisons) and at the seed level ($p < 0.005$ across all comparisons), without resorting to voxel-level pseudoreplication.
