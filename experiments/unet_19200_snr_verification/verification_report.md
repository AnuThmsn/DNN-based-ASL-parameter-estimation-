# Verified U-Net Performance at Every SNR for the 19,200-Sample Experiment

**Experiment Identifier:** `training_size_scaling_v1_snr_verification`<br>
**Source Scaling Study:** `training_size_scaling_v1` (Branch: `storage-safe-4pld-final`)<br>
**Workspace Root:** `D:\ASL_WORKSPACE`<br>
**Model Architecture:** 4-PLD Spatial U-Net (Features: `[32, 64, 128, 256]`, Bottleneck: 512, Trainable Parameters: **7,763,362**)<br>
**Canonical Acquisition Parameters:** PLDs = `[1.525, 2.025, 2.525, 3.025]` s, Labeling Duration $\tau = 1.80$ s<br>
**Audit Date:** October 13, 2026<br>
**Audit Status:** **CONFIRMED (100% Reproducible to Machine Precision)**<br>

---

## 1. Executive Summary

This audit independently evaluates and verifies the performance of the 4-PLD Spatial U-Net trained on the maximal synthetic cohort of **19,200 phantoms** across all tested signal-to-noise ratio (SNR) regimes: $\text{SNR} \in [\infty, 50, 20, 15, 10, 5]$.

### Key Findings
1. **Verified SNR Levels:** The 19,200-sample experiment evaluated exactly six discrete noise regimes: $\infty$ (noiseless Buxton kinetic simulation), 50, 20, 15, 10, and 5.
2. **Exact Reproducibility:** Fresh evaluation of all three trained checkpoints (`seed 42`, `seed 123`, `seed 2024`) matched the recorded RMSE values in `training_scaling_results.csv` bit-for-bit to machine precision ($|\Delta| < 10^{-16}$).
3. **Full Metric Recovery:** While the original experiment only recorded per-SNR RMSE in `training_scaling_results.csv`, this audit evaluated and recovered the complete metric suite (RMSE, MAE, Bias, $R^2$) for both Cerebral Blood Flow (CBF) and Arterial Transit Time (ATT) across all three random seeds.
4. **Graceful Noise Degradation:** Estimation errors degrade monotonically as SNR drops from $\infty$ down to 5:
   - At $\text{SNR} = \infty$: CBF RMSE is $0.4323 \pm 0.0083$ mL/100g/min; ATT RMSE is $0.0316 \pm 0.0006$ s ($31.6$ ms).
   - At clinical $\text{SNR} = 15$: CBF RMSE increases by $+0.2145$ mL/100g/min ($+49.61\%$) to $0.6468 \pm 0.0055$ mL/100g/min; ATT RMSE increases by $+0.0113$ s ($+35.73\%$) to $0.0429 \pm 0.0014$ s.
   - At severe noise $\text{SNR} = 5$: Thermal Gaussian/Rician noise dominates, increasing CBF RMSE to $2.4184 \pm 0.1007$ mL/100g/min ($+459.37\%$) and ATT RMSE to $0.1017 \pm 0.0037$ s ($+221.55\%$).

---

## 2. Experimental Setup & Audit Inventory

### 2.1 Model Architecture & Parameters
- **Model Class:** `ASLUNet` (`experiments_unet_4pld/model.py`)
- **Input Channels:** 4 (corresponding to PLDs `[1.525, 2.025, 2.525, 3.025]` s)
- **Output Channels:** 2 (Channel 0: CBF in mL/100g/min; Channel 1: ATT in seconds)
- **Feature Progression:** Initial 32, Down1 64, Down2 128, Down3 256, Bottleneck 512, with transposed convolution decoding (`bilinear=False`)
- **Total Trainable Parameters:** **7,763,362** (verified via `count_parameters()`)

### 2.2 Training Pool & Checkpoints
- **Master Dataset:** `D:\ASL_WORKSPACE\data\train_nested_N19200_s42.npz` (19,200 synthetic brain phantoms, seed 42, 64x64 grid).
  - SHA-256: `af09dd8f8b9787bc96e745aeedcc98093a90047ed0d0946c1183cdb3f423330d`
- **Validation Dataset:** `D:\ASL_WORKSPACE\data\val_N100_s99.npz` (100 independent phantoms, seed 99; 0 cohort overlap).
- **Checkpoints Audited:**
  1. `unet_N19200_seed_42.pth`: Best Epoch 37, Val Loss 0.060096, SHA-256: `2077ab6edb6b76da5c6e4145a98763524017c6327d44d2adc9f48358af3b276d`
  2. `unet_N19200_seed_123.pth`: Best Epoch 35, Val Loss 0.060856, SHA-256: `2a34b8184a4efd0d14b250237a95e7b936c356e223f6af1b13f258ec7e2c8f4c`
  3. `unet_N19200_seed_2024.pth`: Best Epoch 31, Val Loss 0.062242, SHA-256: `efde05dd2aa9603dcc62d472c1a2615ac6f0bba7d85d83ee4d9b371ce4a36f49`

### 2.3 Evaluation Cohort & Noise Protocol
- **Locked Base Test Cohort:** `D:\ASL_WORKSPACE\data\test_locked_N100_s777.npz` (100 phantoms, seed 777).
  - SHA-256: `90ad6d29c4bafe5c63fc7d4f910ea05c0c9cbcf7a18bb708fceb4081b37320fc`
  - Valid Brain Voxels: Exactly 171,728 valid brain voxels across all 100 phantoms.
- **Identical Phantoms & Masks:** The exact same ground-truth maps ($Y_{\text{CBF}}, Y_{\text{ATT}}$) and brain masks are shared across all SNR snapshots; only input channel observations $X$ differ by noise level.
- **Noise Simulation:** Two-coil Rician noise added to Buxton kinetic model signal $S$:
  - $\text{SNR} = \infty$: $X = 2.0 \cdot S$ (noiseless reference, $\sigma = 0$).
  - Finite SNR: $X = \sqrt{(S + \epsilon_1)^2 + \epsilon_2^2} + \sqrt{(S + \epsilon_3)^2 + \epsilon_4^2}$ with $\sigma = \text{SIG\_REF\_SC} / \text{SNR}$.
  - Snapshot files and digests:
    - $\text{SNR} = 5$: `test_locked_N100_s777_snr_5.npz` (SHA-256: `88cea96f...`)
    - $\text{SNR} = 10$: `test_locked_N100_s777_snr_10.npz` (SHA-256: `44cd021f...`)
    - $\text{SNR} = 15$: `test_locked_N100_s777_snr_15.npz` (SHA-256: `b880fa2c...`)
    - $\text{SNR} = 20$: `test_locked_N100_s777_snr_20.npz` (SHA-256: `765fb41b...`)
    - $\text{SNR} = 50$: `test_locked_N100_s777_snr_50.npz` (SHA-256: `61c722a8...`)
    - $\text{SNR} = \infty$: `test_locked_N100_s777_snr_inf.npz` (SHA-256: `40d15a6d...`)

---

## 3. Evaluation Procedure & Metric Formulations

1. **Preprocessing:** Test observations normalized using training-isolated statistics $(\mu_X, \sigma_X)$ from the 19,200 training pool:
   $$X_{\text{norm}, c} = \frac{X_c - \mu_{X, c}}{\sigma_{X, c}}$$
2. **Denormalization:** Model raw outputs denormalized using target statistics $(\mu_Y, \sigma_Y)$:
   $$\hat{c}_{\text{raw}} = \text{out}_0 \cdot \sigma_{\text{CBF}} + \mu_{\text{CBF}}, \quad \hat{a}_{\text{raw}} = \text{out}_1 \cdot \sigma_{\text{ATT}} + \mu_{\text{ATT}}$$
3. **Brain Masking:** Metrics computed strictly over brain voxels where $\text{mask} > 0.5$ (171,728 valid voxels).
4. **Physical Clamping:** Predictions clipped to physiological bounds ($[0.0, 100.0]$ mL/100g/min for CBF, $[0.5, 3.0]$ s for ATT):
   $$\hat{c} = \min(\max(\hat{c}_{\text{brain}}, 0.0), 100.0), \quad \hat{a} = \min(\max(\hat{a}_{\text{brain}}, 0.5), 3.0)$$
5. **Metric Formulas:**
   $$\text{RMSE} = \sqrt{\frac{1}{V}\sum_{i=1}^V (\hat{y}_i - y_i)^2}, \quad \text{MAE} = \frac{1}{V}\sum_{i=1}^V |\hat{y}_i - y_i|$$
   $$\text{Bias} = \frac{1}{V}\sum_{i=1}^V (\hat{y}_i - y_i), \quad R^2 = 1 - \frac{\sum_{i=1}^V (y_i - \hat{y}_i)^2}{\sum_{i=1}^V (y_i - \bar{y})^2}$$

---

## 4. Verified Results Tables

### Table A: Per-Seed Quantitative Results ($N=19,200$)

| training_samples | seed | snr | cbf_rmse | cbf_mae | cbf_bias | cbf_r2 | att_rmse | att_mae | att_bias | att_r2 | checkpoint | evaluation_status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 19200 | 42 | inf | 0.424398 | 0.274256 | 0.018327 | 0.999153 | 0.032042 | 0.017480 | -0.001857 | 0.993447 | `unet_N19200_seed_42.pth` | CONFIRMED_MATCH |
| 19200 | 42 | 50 | 0.451052 | 0.298787 | 0.034590 | 0.999043 | 0.034311 | 0.019378 | -0.001727 | 0.992486 | `unet_N19200_seed_42.pth` | CONFIRMED_MATCH |
| 19200 | 42 | 20 | 0.566319 | 0.399378 | 0.108829 | 0.998492 | 0.041358 | 0.025337 | 0.001314 | 0.989083 | `unet_N19200_seed_42.pth` | CONFIRMED_MATCH |
| 19200 | 42 | 15 | 0.641114 | 0.468043 | 0.185079 | 0.998067 | 0.043981 | 0.028365 | 0.004000 | 0.987654 | `unet_N19200_seed_42.pth` | CONFIRMED_MATCH |
| 19200 | 42 | 10 | 0.900392 | 0.686180 | 0.421386 | 0.996188 | 0.054403 | 0.037339 | 0.008561 | 0.981110 | `unet_N19200_seed_42.pth` | CONFIRMED_MATCH |
| 19200 | 42 | 5 | 2.517345 | 2.186258 | 1.990622 | 0.970201 | 0.105195 | 0.081020 | 0.051974 | 0.929371 | `unet_N19200_seed_42.pth` | CONFIRMED_MATCH |
| 19200 | 123 | inf | 0.440964 | 0.296703 | 0.095641 | 0.999086 | 0.030986 | 0.016900 | 0.000190 | 0.993872 | `unet_N19200_seed_123.pth` | CONFIRMED_MATCH |
| 19200 | 123 | 50 | 0.463542 | 0.317186 | 0.108852 | 0.998990 | 0.032843 | 0.018574 | 0.000030 | 0.993115 | `unet_N19200_seed_123.pth` | CONFIRMED_MATCH |
| 19200 | 123 | 20 | 0.576981 | 0.413374 | 0.161288 | 0.998435 | 0.038272 | 0.024182 | 0.002496 | 0.990651 | `unet_N19200_seed_123.pth` | CONFIRMED_MATCH |
| 19200 | 123 | 15 | 0.651973 | 0.479643 | 0.224714 | 0.998001 | 0.041321 | 0.026963 | 0.003963 | 0.989102 | `unet_N19200_seed_123.pth` | CONFIRMED_MATCH |
| 19200 | 123 | 10 | 0.887756 | 0.675076 | 0.414005 | 0.996294 | 0.051479 | 0.035414 | 0.008049 | 0.983086 | `unet_N19200_seed_123.pth` | CONFIRMED_MATCH |
| 19200 | 123 | 5 | 2.421704 | 2.064795 | 1.883170 | 0.972422 | 0.097814 | 0.074839 | 0.050995 | 0.938934 | `unet_N19200_seed_123.pth` | CONFIRMED_MATCH |
| 19200 | 2024 | inf | 0.431672 | 0.282862 | 0.095085 | 0.999124 | 0.031876 | 0.017290 | 0.002226 | 0.993515 | `unet_N19200_seed_2024.pth` | CONFIRMED_MATCH |
| 19200 | 2024 | 50 | 0.454557 | 0.303165 | 0.106002 | 0.999028 | 0.033530 | 0.018949 | 0.001927 | 0.992824 | `unet_N19200_seed_2024.pth` | CONFIRMED_MATCH |
| 19200 | 2024 | 20 | 0.571463 | 0.404607 | 0.169145 | 0.998464 | 0.039443 | 0.024579 | 0.003240 | 0.990070 | `unet_N19200_seed_2024.pth` | CONFIRMED_MATCH |
| 19200 | 2024 | 15 | 0.647416 | 0.470136 | 0.224455 | 0.998029 | 0.043507 | 0.027455 | 0.005550 | 0.987919 | `unet_N19200_seed_2024.pth` | CONFIRMED_MATCH |
| 19200 | 2024 | 10 | 0.891777 | 0.674027 | 0.421963 | 0.996260 | 0.053804 | 0.035978 | 0.009545 | 0.981523 | `unet_N19200_seed_2024.pth` | CONFIRMED_MATCH |
| 19200 | 2024 | 5 | 2.316116 | 1.982208 | 1.792315 | 0.974774 | 0.102153 | 0.078079 | 0.054528 | 0.933396 | `unet_N19200_seed_2024.pth` | CONFIRMED_MATCH |

*Note: Full precision values preserved in `snr_results_per_seed.csv`.*

---

### Table B: Aggregate Performance Across Seeds ($N=19,200$, Mean ± SD)

| SNR | CBF RMSE (mL/100g/min) | CBF MAE (mL/100g/min) | CBF Bias (mL/100g/min) | ATT RMSE (seconds) | ATT MAE (seconds) | ATT Bias (seconds) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\infty$** | $0.4323 \pm 0.0083$ | $0.2846 \pm 0.0113$ | $0.0697 \pm 0.0445$ | $0.0316 \pm 0.0006$ | $0.0172 \pm 0.0003$ | $0.0002 \pm 0.0020$ |
| **50** | $0.4564 \pm 0.0064$ | $0.3064 \pm 0.0096$ | $0.0831 \pm 0.0421$ | $0.0336 \pm 0.0007$ | $0.0190 \pm 0.0004$ | $0.0001 \pm 0.0018$ |
| **20** | $0.5716 \pm 0.0053$ | $0.4058 \pm 0.0071$ | $0.1464 \pm 0.0328$ | $0.0397 \pm 0.0016$ | $0.0247 \pm 0.0006$ | $0.0024 \pm 0.0010$ |
| **15** | $0.6468 \pm 0.0055$ | $0.4726 \pm 0.0062$ | $0.2114 \pm 0.0228$ | $0.0429 \pm 0.0014$ | $0.0276 \pm 0.0007$ | $0.0045 \pm 0.0009$ |
| **10** | $0.8933 \pm 0.0065$ | $0.6784 \pm 0.0067$ | $0.4191 \pm 0.0044$ | $0.0532 \pm 0.0015$ | $0.0362 \pm 0.0010$ | $0.0087 \pm 0.0008$ |
| **5** | $2.4184 \pm 0.1007$ | $2.0778 \pm 0.1026$ | $1.8887 \pm 0.0993$ | $0.1017 \pm 0.0037$ | $0.0780 \pm 0.0031$ | $0.0525 \pm 0.0018$ |

*Note: Aggregate statistics calculated directly from the three unrounded per-seed metrics; physical units strictly preserved.*

---

### Table C: Performance Degradation Across SNR Relative to Noiseless Reference ($\infty$)

| SNR | CBF RMSE (Mean) | Absolute Increase in CBF RMSE | Percentage Increase in CBF RMSE | ATT RMSE (Mean) | Absolute Increase in ATT RMSE | Percentage Increase in ATT RMSE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\infty$** | $0.4323$ mL/100g/min | Reference Condition ($0.0000$) | Reference Condition ($0.00\%$) | $0.0316$ s | Reference Condition ($0.0000$ s) | Reference Condition ($0.00\%$) |
| **50** | $0.4564$ mL/100g/min | $+0.0240$ mL/100g/min | $+5.56\%$ | $0.0336$ s | $+0.0019$ s | $+6.09\%$ |
| **20** | $0.5716$ mL/100g/min | $+0.1392$ mL/100g/min | $+32.21\%$ | $0.0397$ s | $+0.0081$ s | $+25.47\%$ |
| **15** | $0.6468$ mL/100g/min | $+0.2145$ mL/100g/min | $+49.61\%$ | $0.0429$ s | $+0.0113$ s | $+35.73\%$ |
| **10** | $0.8933$ mL/100g/min | $+0.4610$ mL/100g/min | $+106.62\%$ | $0.0532$ s | $+0.0216$ s | $+68.26\%$ |
| **5** | $2.4184$ mL/100g/min | $+1.9860$ mL/100g/min | $+459.37\%$ | $0.1017$ s | $+0.0701$ s | $+221.55\%$ |

---

## 5. Audit Reconciliation & Verification Analysis

### 5.1 Verification Status
- **Result Status:** **CONFIRMED (All 18 Conditions Verified)**
- **Discrepancy Investigation:** There is zero numerical discrepancy between the original experiment and this audit. All 18 recomputed RMSE values match `training_scaling_results.csv` with a maximum absolute difference of $0.00 \times 10^0$ (identical to floating-point representation limits).
- **Metric Expansion:** The original CSV omitted MAE, Bias, and $R^2$ for SNR-stratified snapshots, logging only RMSE. This audit successfully recovered the complete metric profile using the original model weights and cohort snapshots without retraining.

### 5.2 Independent Automated Verification
The companion verification script `verify_snr_results.py` executes 6 non-destructive automated checks:
1. Cohort sample size verification: Strictly $N=19,200$ records only.
2. Seed & SNR completeness: Exact sets $\{42, 123, 2024\}$ and $\{5, 10, 15, 20, 50, \infty\}$.
3. Uniqueness: Exactly 18 records with zero duplicate pairs.
4. Numerical parity: Aggregate mean and standard deviation match within $2.22 \times 10^{-16}$.
5. Degradation parity: Absolute and percentage degradation match within $4.55 \times 10^{-13}$.
6. Backward compatibility: Exact match against historical scaling CSV ($|\Delta| = 0.00$).

### 5.3 Reproduction Command
To re-run the complete verification suite non-destructively:
```powershell
python experiments/unet_19200_snr_verification/verify_snr_results.py
```
