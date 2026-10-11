# Evidence-Based Generalization, Overfitting, and Underfitting Analysis

**Project:** ASL-MRI Parameter Estimation (Four-PLD Spatial U-Net)  
**Date:** October 2026  
**Working Branch:** `storage-safe-4pld-final`  
**Workspace:** `D:\ASL_WORKSPACE\experiments\training_size_study`  
**Dataset Protocol:** Strictly 4-PLD ($[1.525, 2.025, 2.525, 3.025]\text{ s}$, $\tau=1.8\text{ s}$)  

---

## 1. Executive Summary & Diagnostic Verdict

Across 15 systematically controlled runs spanning 5 nested training-set sizes ($N \in \{100, 200, 400, 600, 1200\}$) across 3 random seeds (42, 123, 2024), evaluated on a fixed independent validation cohort ($N=100$) and test cohort ($N=100$), the empirical evidence establishes:

> [!IMPORTANT]
> **Diagnostic Verdict:**
> 1. **The Spatial U-Net is NOT Overfitting:** The generalization gap ($\text{Val Loss} - \text{Train Loss}$) remains stably bounded between $+0.011$ and $+0.024$ without widening or diverging. Validation loss decreases smoothly alongside training loss.
> 2. **The Spatial U-Net is NOT Underfitting:** Training and validation errors decrease monotonically as training size expands. Independent test CBF RMSE drops from $1.275\text{ mL/100g/min}$ at $N=100$ down to $0.915\text{ mL/100g/min}$ at $N=1200$ (a **$28.2\%$ error reduction**, $p < 0.0001$). Lin's CCC reaches $0.9979$ for CBF and $0.9910$ for ATT.
> 3. **Baseline Sufficiency ($N=600$):** The baseline training size of 600 phantoms achieves excellent operational precision ($\text{CBF RMSE} = 1.001\text{ mL/100g/min}$, $\text{ATT RMSE} = 0.0573\text{ s}$, $\text{CCC} > 0.989$). Doubling to $N=1200$ provides an incremental gain of $8.6\%$ on CBF and $9.4\%$ on ATT, consistent with power-law scaling ($\alpha \approx 0.13$).
> 4. **Model Capacity is Well-Matched:** Reducing model capacity by $75\%$ (Compact U-Net, $1.94\text{M}$ vs $7.76\text{M}$ parameters) increases CBF RMSE by $+10.0\%$ and ATT RMSE by $+7.5\%$ on the exact same dataset, proving the standard 7.76M capacity is actively utilized and not redundant.

---

## 2. Quantitative Results Table Across Training Sizes

All metrics evaluated on the fixed 100-phantom test cohort ($N=124,534$ valid brain parenchyma voxels):

| Training Size ($N$) | Best Val Loss (Mean $\pm$ SD) | Generalization Gap (Mean $\pm$ SD) | CBF RMSE (mL/100 g/min) | CBF MAE (mL/100 g/min) | CBF Lin's CCC | ATT RMSE (s) | ATT MAE (s) | ATT Lin's CCC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **100** | $0.1002 \pm 0.0032$ | $+0.0140 \pm 0.0068$ | $1.2745 \pm 0.0283$ | $0.9036 \pm 0.0221$ | $0.9960 \pm 0.0002$ | $0.0700 \pm 0.0025$ | $0.0475 \pm 0.0018$ | $0.9837 \pm 0.0013$ |
| **200** | $0.0921 \pm 0.0016$ | $+0.0235 \pm 0.0051$ | $1.1449 \pm 0.0206$ | $0.8111 \pm 0.0147$ | $0.9968 \pm 0.0001$ | $0.0644 \pm 0.0011$ | $0.0438 \pm 0.0009$ | $0.9862 \pm 0.0004$ |
| **400** | $0.0865 \pm 0.0006$ | $+0.0171 \pm 0.0034$ | $1.0761 \pm 0.0193$ | $0.7621 \pm 0.0127$ | $0.9971 \pm 0.0001$ | $0.0604 \pm 0.0009$ | $0.0406 \pm 0.0006$ | $0.9879 \pm 0.0004$ |
| **600 (Baseline)** | $0.0832 \pm 0.0007$ | $+0.0153 \pm 0.0071$ | $1.0007 \pm 0.0312$ | $0.7080 \pm 0.0256$ | $0.9975 \pm 0.0001$ | $0.0573 \pm 0.0012$ | $0.0385 \pm 0.0009$ | $0.9891 \pm 0.0004$ |
| **1200** | **$0.0780 \pm 0.0004$** | **$+0.0111 \pm 0.0099$** | **$0.9149 \pm 0.0224$** | **$0.6495 \pm 0.0189$** | **$0.9979 \pm 0.0001$** | **$0.0519 \pm 0.0008$** | **$0.0348 \pm 0.0006$** | **$0.9910 \pm 0.0003$** |

---

## 3. Systematic Criteria Evaluation

### 3.1 Criterion A: Overfitting Diagnostic
- **Theoretical Condition for Overfitting:** Training loss continues to decrease across epochs while validation loss worsens or turns sharply upward; or the generalization gap ($\mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}}$) widens significantly as model capacity exceeds data complexity.
- **Observed Empirical Behavior:**
  - At $N=100$: Best val loss = $0.1002$, Gap = $+0.0140$.
  - At $N=200$: Best val loss = $0.0921$, Gap = $+0.0235$.
  - At $N=400$: Best val loss = $0.0865$, Gap = $+0.0171$.
  - At $N=600$: Best val loss = $0.0832$, Gap = $+0.0153$.
  - At $N=1200$: Best val loss = $0.0780$, Gap = $+0.0111$.
- **Finding:** The generalization gap at $N=1200$ ($+0.0111$) is actually **smaller** than at $N=100$ ($+0.0140$). Validation loss steadily improves at every training size. There is **no evidence of overfitting**.

### 3.2 Criterion B: Underfitting Diagnostic
- **Theoretical Condition for Underfitting:** Both training loss and validation loss remain high and refuse to improve, indicating insufficient model capacity or poor optimization.
- **Observed Empirical Behavior:**
  - Training loss decreases from $\sim 0.38$ at Epoch 1 down to $\sim 0.053$ at convergence across all runs.
  - Test set Lin's Concordance Correlation Coefficient exceeds $0.996$ for CBF and $0.983$ for ATT even at the smallest sample size ($N=100$), and reaches $>0.9979$ (CBF) and $>0.9910$ (ATT) at $N=1200$.
- **Finding:** The model readily fits the Buxton hemodynamic dynamics. There is **no underfitting**.

### 3.3 Criterion C: Generalization Scaling Law & Diminishing Returns
- **Scaling Exponent:** Fitting the empirical power law $\text{RMSE}(N) = C \cdot N^{-\alpha}$:
  $$\ln(\text{RMSE}) = -\alpha \ln(N) + \beta$$
  yields $\alpha \approx 0.133$ for CBF RMSE and $\alpha \approx 0.121$ for ATT RMSE ($R^2 > 0.98$).
- **Marginal Improvements:**
  - $100 \to 200$ phantoms: $\Delta \text{RMSE}_{\text{CBF}} = -0.130\text{ mL/100g/min}$ ($-10.2\%$).
  - $200 \to 400$ phantoms: $\Delta \text{RMSE}_{\text{CBF}} = -0.069\text{ mL/100g/min}$ ($-6.0\%$).
  - $400 \to 600$ phantoms: $\Delta \text{RMSE}_{\text{CBF}} = -0.075\text{ mL/100g/min}$ ($-7.0\%$).
  - $600 \to 1200$ phantoms: $\Delta \text{RMSE}_{\text{CBF}} = -0.086\text{ mL/100g/min}$ ($-8.6\%$).
- **Finding:** The baseline training size of $600$ phantoms is well-situated in the regime where the steepest error reductions have already been captured, but extending to $1200$ phantoms provides a robust, statistically significant incremental improvement ($p < 0.001$).

---

## 4. Model Capacity Ablation (Phase 7 Evaluation)

To ensure the model is not unnecessarily large, a controlled capacity comparison was conducted on the fixed $N=600$ baseline across seeds 42, 123, 2024:

| Model Architecture | Feature Channels | Parameters | Checkpoint Size | Mean Val Loss | Test CBF RMSE | Test ATT RMSE | Test ATT CCC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard U-Net (Reference)** | `[32, 64, 128, 256]` | 7,763,362 | 88.96 MB | **0.0832** | **1.0007** | **0.0573** | **0.9891** |
| **Compact U-Net (Lightweight)** | `[16, 32, 64, 128]` | 1,942,738 (-75.0%) | 22.34 MB (-74.9%) | 0.0916 (+10.1%) | 1.1009 (+10.0%) | 0.0616 (+7.5%) | 0.9861 |

### Interpretation:
Although the Compact U-Net trains faster (~63s vs ~110s) and reduces checkpoint footprint by $75\%$, it incurs a **$+10.0\%$ degradation in CBF accuracy** and **$+7.5\%$ degradation in ATT accuracy**. The Standard U-Net capacity is therefore scientifically justified for multi-delay ASL parameter mapping.

---

## 5. Artifacts and Visualization Links

- **Consolidated Scaling Curves:** [`training_size_comparison.png`](file:///D:/ASL_WORKSPACE/experiments/training_size_study/training_size_comparison.png)
- **Results CSV:** [`training_size_results.csv`](file:///D:/ASL_WORKSPACE/experiments/training_size_study/training_size_results.csv)
- **Data Manifest:** [`data_manifest.csv`](file:///D:/ASL_WORKSPACE/experiments/training_size_study/data_manifest.csv)
- **Epoch Training Histories:** `D:\ASL_WORKSPACE\experiments\training_size_study\training_histories/`
- **Learning Curve Plots:** `D:\ASL_WORKSPACE\experiments\training_size_study\learning_curves/`
