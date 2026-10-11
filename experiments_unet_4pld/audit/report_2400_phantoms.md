# Technical Report: The 2,400-Training-Phantom Condition

**Experiment:** ASL-DNN Training-Set Size Generalization Study ($N = 2,400$ Phantoms)  
**Location:** `D:\ASL_WORKSPACE\experiments\training_size_study`  
**Model Architecture:** Spatial ASL U-Net (7.76M parameters)  
**Protocol:** Strictly 4-PLDs ($[1.525, 2.025, 2.525, 3.025]\text{ s}$, $\tau = 1.8\text{ s}$)  
**Date:** October 2026  

---

## 1. Experimental Setup and Execution

The 2,400-phantom condition represents a $4\times$ extension over the baseline 600-phantom training set and a $2\times$ extension over the 1,200-phantom cohort.

### 1.1 Dataset Specification
- **Master Dataset File:** `D:\ASL_WORKSPACE\data\train_nested_N2400_s42.npz` (172.8 MB, 2,400 multi-PLD phantoms, `seed = 42`).
- **Nesting Property:** Strictly contains the 100, 200, 400, 600, and 1,200 subsets as initial slices $[0:N]$.
- **Normalization:** Computed exclusively on valid brain voxels of the 2,400 training phantoms:
  - Input Mean: `[0.00342, 0.00412, 0.00451, 0.00472]`
  - Input Std: `[0.00456, 0.00512, 0.00543, 0.00558]`
  - CBF Mean: $55.02\text{ mL/100g/min}$, Std: $21.45\text{ mL/100g/min}$
  - ATT Mean: $1.498\text{ s}$, Std: $0.432\text{ s}$

---

## 2. Training Run Details Across Seeds

Each seed initialized the model weights independently and trained with Adam ($lr=0.001$, ReduceLROnPlateau, batch size 16, early stopping patience 25 on the frozen validation cohort of 100 phantoms):

| Seed | Epochs to Best | Best Val Loss | Train Loss at Best | Gen Gap | Training Duration | Checkpoint File | Checkpoint SHA-256 |
|:---:|:---:|:---:|:---:|:---:|:---:|---|---|
| **42** | 57 | 0.0710 | 0.0549 | +0.0160 | 429.1 s (~7.1 min) | `unet_N2400_seed_42.pth` | `0169ab40ebb4c4f0ba4339640b18dc61377c42505b207d8c469d413b954a9690` |
| **123** | 55 | 0.0719 | 0.0523 | +0.0196 | 413.7 s (~6.9 min) | `unet_N2400_seed_123.pth` | `10810ead787e955f56a0881de50f4d8d87218bf9fafe45a47038ad36a1123baa` |
| **2024** | 43 | 0.0716 | 0.0551 | +0.0165 | 339.9 s (~5.7 min) | `unet_N2400_seed_2024.pth` | `8434f9f230c86ebb4453550f41499d35f2877567e0b3c8f586dca9c264d86af8` |
| **Mean $\pm$ SD** | $51.7 \pm 7.6$ | **$0.0715 \pm 0.0005$** | **$0.0541 \pm 0.0016$** | **$+0.0174 \pm 0.0020$** | **$394.2 \pm 47.7\text{ s}$** | — | — |

---

## 3. Independent Test Performance ($N = 100$ Test Phantoms, 171,356 Voxels)

| Seed | CBF RMSE (mL/100g/min) | CBF MAE (mL/100g/min) | CBF $R^2$ | CBF CCC | ATT RMSE (s) | ATT MAE (s) | ATT $R^2$ | ATT CCC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | 0.7970 | 0.5654 | 0.9969 | 0.9984 | 0.0478 | 0.0324 | 0.9850 | 0.9925 |
| **123** | 0.8030 | 0.5725 | 0.9968 | 0.9984 | 0.0489 | 0.0330 | 0.9842 | 0.9921 |
| **2024** | 0.8388 | 0.5927 | 0.9965 | 0.9983 | 0.0502 | 0.0332 | 0.9834 | 0.9915 |
| **Mean $\pm$ SD** | **$0.8129 \pm 0.0226$** | **$0.5769 \pm 0.0142$** | **$0.9967 \pm 0.0002$** | **$0.9984 \pm 0.0001$** | **$0.0490 \pm 0.0012$** | **$0.0328 \pm 0.0004$** | **$0.9842 \pm 0.0008$** | **$0.9920 \pm 0.0005$** |

---

## 4. Key Performance Comparisons

### 4.1 Comparison against $N=600$ Authoritative Baseline
- **CBF RMSE:** Dropped from $1.0007 \pm 0.0312$ to **$0.8129 \pm 0.0226\text{ mL/100g/min}$** ($\mathbf{-18.8\%}$ relative error reduction).
- **ATT RMSE:** Dropped from $0.0573 \pm 0.0012$ to **$0.0490 \pm 0.0012\text{ s}$** ($\mathbf{-14.5\%}$ relative error reduction).
- **Concordance Correlation:** CBF CCC improved from $0.9975 \to 0.9984$; ATT CCC improved from $0.9891 \to 0.9920$.

### 4.2 Comparison against $N=100$ Minimal Training Set
- **CBF RMSE:** Dropped from $1.2745 \pm 0.0283$ to **$0.8129 \pm 0.0226\text{ mL/100g/min}$** ($\mathbf{-36.2\%}$ relative error reduction).
- **ATT RMSE:** Dropped from $0.0699 \pm 0.0025$ to **$0.0490 \pm 0.0012\text{ s}$** ($\mathbf{-30.1\%}$ relative error reduction).

---

## 5. Compute and Storage Impact
- Total training time for all 3 seeds: $1,182.7\text{ s}$ (~19.7 minutes).
- Peak CUDA memory: 0.88 GB allocated out of 6.0 GB total (zero VRAM strain).
- Checkpoints size: 3 checkpoints $\times$ 93.2 MB = 279.6 MB, safely located under `D:\ASL_WORKSPACE\experiments\training_size_study\checkpoints\`.
- All outputs are preserved without overwriting historical runs.
