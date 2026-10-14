# U-Net SNR Evaluation Audit and Completion: SNR 10, 15, and 20 dB

**Experiment Identifier:** `unet_snr_10_15_20_audit`<br>
**Target Model:** 4-PLD Spatial ASL U-Net (600 Training Phantoms, 7,763,362 Trainable Parameters)<br>
**Repository Branch:** `storage-safe-4pld-final`<br>
**Workspace Directory:** `D:\ASL_WORKSPACE`<br>
**Audit Date:** October 14, 2026<br>
**Audit Status:** **CONFIRMED (All 9 Conditions Verified, Zero Cohort Leakage)**<br>

---

## 1. Executive Summary

This scientific audit completes and rigorously validates the signal-to-noise ratio (SNR) evaluation for the canonical **600-training-phantom 4-PLD ASL U-Net** across SNR levels **10 dB, 15 dB, and 20 dB**.

### Key Findings
1. **Resolution of SNR 20 dB Data Gap:** Previous benchmark outputs (`results/snr_stratified_metrics.csv`) only evaluated SNR 10 and 15 dB; SNR 20 dB was never evaluated. This audit evaluates all three canonical model checkpoints (`seeds 42, 123, 2024`) at SNR 20 dB, establishing a complete performance curve.
2. **Resolution of Test Cohort Fairness Flaw:** In the historical test generator (`test_N100_s7.npz`), SNR targets cycled across phantom instances (`i % 2`), causing SNR 10 and SNR 15 to evaluate disjoint 50-phantom subsets with different anatomical structures and physiological distributions. This audit resolved the issue by evaluating on the locked 100-phantom test cohort where all 100 phantoms share the **exact same underlying ground-truth parameter maps and brain masks** across SNR 10, 15, and 20 dB, isolating the effect of noise.
3. **Multi-Seed Representation:** The historical `snr_stratified_metrics.csv` evaluated only a single model (Seed 42). This audit evaluates all three training seeds (`42`, `123`, `2024`) and reports mean $\pm$ standard deviation across seeds, while also quantifying phantom-level variance.
4. **Consistent Noise Robustness:**
   - **CBF Estimation:** Across the 3 training seeds, mean CBF RMSE improves monotonically from $1.1156 \pm 0.0277$ mL/100g/min at SNR 10 dB to $0.9378 \pm 0.0322$ mL/100g/min at SNR 15 dB, and $0.8799 \pm 0.0351$ mL/100g/min at SNR 20 dB.
   - **ATT Estimation:** Across the 3 training seeds, mean ATT RMSE improves monotonically from $0.0605 \pm 0.0009$ s at SNR 10 dB to $0.0545 \pm 0.0016$ s at SNR 15 dB, and $0.0536 \pm 0.0014$ s at SNR 20 dB.
   - Lin's Concordance Correlation Coefficient (CCC) exceeds $0.996$ for CBF and $0.987$ for ATT across all noise levels.

---

## 2. Physics & Data-Generation Protocol

### 2.1 Forward ASL Buxton Kinetic Simulator
Signals are generated from physiological parameters using the standard two-compartment Buxton general kinetic model with pulsed arterial spin labeling (PASL):
- **Canonical PLDs:** $t \in [1.525, 2.025, 2.525, 3.025]$ seconds.
- **Labeling Duration:** $\tau = 1.80$ seconds.
- **Physical Constants:** Blood $T_{1a} = 1.66$ s, tissue $T_{1t} = 1.20$ s, labeling efficiency $\alpha = 0.85$, inversion efficiency $\beta = 0.75$, blood-brain partition coefficient $\lambda = 0.90$ mL/g.
- **Signal Scaling:** Raw kinetic difference signals $\Delta M(t)$ are scaled by $\text{SCALE} = 100,000.0$.
- **Reference Signal:** $\text{SIG\_REF\_SC} = 334.2039$.

### 2.2 2D Synthetic Brain Phantom Construction
Phantoms are generated on a $64 \times 64$ spatial grid via `experiments_unet_4pld/generate_spatial_data.py`:
1. **Outer Brain Parenchyma:** Elliptical boundary perturbed with low-frequency Fourier angular harmonics to mimic realistic anatomical brain lobes.
2. **Ventricular CSF Compartment:** Butterfly-shaped central ventricles assigned to non-perfused cerebrospinal fluid ($\text{CBF} = 0, \text{ATT} = 0$) and excluded from brain masks.
3. **Tissue Segmentation:** Distance transforms separate cortical Gray Matter (GM, border distance $\le 2.5\text{--}4.0$ voxels) and deep GM clusters (basal ganglia/thalamus) from White Matter (WM).
4. **Physiological Parameters:**
   - Cortical GM: CBF $\sim 55\text{--}75$ mL/100g/min, ATT $\sim 0.9\text{--}1.4$ s.
   - Deep WM: CBF $\sim 18\text{--}32$ mL/100g/min, ATT $\sim 1.5\text{--}2.2$ s.
   - Spatial smoothness: Continuous 2D Gaussian random fields ($\sigma = 5.0\text{--}6.0$ voxels) plus an anterior-to-posterior watershed delay gradient ($0.15\text{--}0.35$ s).
   - Bounds: Clamped to physiological ranges $[0.0, 100.0]$ mL/100g/min for CBF and $[0.5, 3.0]$ s for ATT.

### 2.3 Rician Noise Model
For a target SNR, the noise standard deviation per channel is:
$$\sigma = \frac{\text{SIG\_REF\_SC}}{\text{SNR}}$$
Two independent Gaussian noise instances $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0, \sigma^2)$ and $\epsilon_3, \epsilon_4 \sim \mathcal{N}(0, \sigma^2)$ simulate two-coil Rician noise:
$$X = \sqrt{(S_{\text{clean}} + \epsilon_1)^2 + \epsilon_2^2} + \sqrt{(S_{\text{clean}} + \epsilon_3)^2 + \epsilon_4^2}$$
For SNR 10 dB: $\sigma = 33.42039$. For SNR 15 dB: $\sigma = 22.28026$. For SNR 20 dB: $\sigma = 16.710195$.

---

## 3. Test-Cohort Fairness Audit

| Assessment Item | Historical Test Set (`test_N100_s7.npz`) | Locked SNR Cohort (`test_N100_snr_*.npz`) | Audit Verdict |
| :--- | :--- | :--- | :--- |
| **Phantoms Evaluated** | 100 phantoms (50 at SNR 10, 50 at SNR 15) | 100 phantoms evaluated at all SNRs | Corrected |
| **Ground-Truth Identity** | Different phantoms for SNR 10 vs 15 | **Identical $Y$ and masks across SNR 10, 15, 20** | **100% Locked** |
| **Valid Brain Voxels** | $\approx 85,678$ voxels per SNR level | **171,356 voxels per SNR level** | **Harmonized** |
| **SNR 20 dB Availability** | Missing ($0$ phantoms) | Complete ($100$ phantoms, 171,356 voxels) | **Completed** |
| **Seeds Evaluated** | 1 seed (Seed 42 only) | **3 seeds (`42`, `123`, `2024`)** | **Triplicated** |
| **Cohort Leakage** | 0 / 100 overlaps | 0 / 100 overlaps vs Train & Val | **Zero Leakage** |

---

## 4. Quantitative Results Tables

### Table 1: CBF Estimation Performance Across SNR 10, 15, and 20 dB

*Units: RMSE, MAE, and Bias are reported in mL/100g/min. Seed Variation represents Mean $\pm$ SD across the 3 training seeds; Phantom Variation represents Mean $\pm$ SD across the 100 test phantoms.*

| SNR Condition | CBF RMSE (Seed Var) | CBF MAE (Seed Var) | CBF Bias (Seed Var) | CBF $R^2$ (Seed Var) | CBF Lin's CCC (Seed Var) | CBF RMSE (Phantom Var) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SNR 10 dB** | $1.1156 \pm 0.0277$ | $0.8069 \pm 0.0157$ | $+0.1883 \pm 0.0331$ | $0.9939 \pm 0.0003$ | $0.9969 \pm 0.0001$ | $1.1089 \pm 0.1381$ |
| **SNR 15 dB** | $0.9378 \pm 0.0322$ | $0.6586 \pm 0.0211$ | $+0.1037 \pm 0.0245$ | $0.9957 \pm 0.0003$ | $0.9978 \pm 0.0001$ | $0.9285 \pm 0.1430$ |
| **SNR 20 dB** | $0.8799 \pm 0.0351$ | $0.6044 \pm 0.0238$ | $+0.0867 \pm 0.0204$ | $0.9962 \pm 0.0003$ | $0.9981 \pm 0.0001$ | $0.8676 \pm 0.1555$ |

---

### Table 2: ATT Estimation Performance Across SNR 10, 15, and 20 dB

*Units: RMSE, MAE, and Bias are reported in seconds. Seed Variation represents Mean $\pm$ SD across the 3 training seeds; Phantom Variation represents Mean $\pm$ SD across the 100 test phantoms.*

| SNR Condition | ATT RMSE (Seed Var) | ATT MAE (Seed Var) | ATT Bias (Seed Var) | ATT $R^2$ (Seed Var) | ATT Lin's CCC (Seed Var) | ATT RMSE (Phantom Var) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SNR 10 dB** | $0.0605 \pm 0.0009$ | $0.0417 \pm 0.0008$ | $+0.0043 \pm 0.0042$ | $0.9758 \pm 0.0007$ | $0.9880 \pm 0.0004$ | $0.0577 \pm 0.0184$ |
| **SNR 15 dB** | $0.0545 \pm 0.0016$ | $0.0359 \pm 0.0011$ | $+0.0003 \pm 0.0034$ | $0.9805 \pm 0.0011$ | $0.9902 \pm 0.0006$ | $0.0512 \pm 0.0187$ |
| **SNR 20 dB** | $0.0536 \pm 0.0014$ | $0.0346 \pm 0.0011$ | $-0.0015 \pm 0.0032$ | $0.9810 \pm 0.0010$ | $0.9905 \pm 0.0005$ | $0.0503 \pm 0.0184$ |

---

### Table 3: Per-Seed Comprehensive Results Breakdown (9 Evaluation Runs)

| Phantoms | Seed | SNR | CBF RMSE | CBF MAE | CBF Bias | CBF $R^2$ | CBF CCC | ATT RMSE | ATT MAE | ATT Bias | ATT $R^2$ | ATT CCC | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 600 | 42 | 10 | 1.0836 | 0.7888 | +0.2196 | 0.9942 | 0.9971 | 0.0598 | 0.0408 | +0.0073 | 0.9764 | 0.9882 | `unet_best_seed_42.pth` |
| 600 | 42 | 15 | 0.9006 | 0.6342 | +0.1277 | 0.9960 | 0.9980 | 0.0527 | 0.0347 | +0.0035 | 0.9817 | 0.9909 | `unet_best_seed_42.pth` |
| 600 | 42 | 20 | 0.8399 | 0.5772 | +0.1043 | 0.9965 | 0.9983 | 0.0521 | 0.0333 | +0.0016 | 0.9821 | 0.9910 | `unet_best_seed_42.pth` |
| 600 | 123 | 10 | 1.1325 | 0.8174 | +0.1916 | 0.9937 | 0.9969 | 0.0616 | 0.0424 | +0.0060 | 0.9750 | 0.9875 | `unet_best_seed_123.pth` |
| 600 | 123 | 15 | 0.9567 | 0.6703 | +0.1045 | 0.9955 | 0.9978 | 0.0552 | 0.0362 | +0.0006 | 0.9799 | 0.9899 | `unet_best_seed_123.pth` |
| 600 | 123 | 20 | 0.9058 | 0.6210 | +0.0915 | 0.9960 | 0.9980 | 0.0549 | 0.0353 | -0.0014 | 0.9801 | 0.9900 | `unet_best_seed_123.pth` |
| 600 | 2024 | 10 | 1.1307 | 0.8144 | +0.1536 | 0.9937 | 0.9969 | 0.0602 | 0.0418 | -0.0005 | 0.9761 | 0.9881 | `unet_best_seed_2024.pth` |
| 600 | 2024 | 15 | 0.9561 | 0.6713 | +0.0788 | 0.9955 | 0.9978 | 0.0555 | 0.0369 | -0.0033 | 0.9797 | 0.9899 | `unet_best_seed_2024.pth` |
| 600 | 2024 | 20 | 0.8939 | 0.6151 | +0.0643 | 0.9961 | 0.9980 | 0.0537 | 0.0351 | -0.0047 | 0.9809 | 0.9906 | `unet_best_seed_2024.pth` |

*Note: Unrounded full-precision records are preserved in `unet_snr_metrics_per_seed.csv`.*

---

## 5. Provenance & Metric Reconciliation

### 5.1 What Existed Previously vs What Was Newly Evaluated
- **Pre-existing artifacts:** Checkpoints `unet_best_seed_42.pth`, `unet_best_seed_123.pth`, and `unet_best_seed_2024.pth` were preserved untouched. The locked test snapshots `test_N100_snr_10.npz`, `test_N100_snr_15.npz`, and `test_N100_snr_20.npz` existed in the benchmark workspace.
- **Historical CSV values:** `results/snr_stratified_metrics.csv` only recorded Seed 42 on the alternating 50-phantom split (SNR 10: CBF RMSE 1.0460, ATT RMSE 0.0577; SNR 15: CBF RMSE 0.8767, ATT RMSE 0.0541).
- **Newly evaluated values:** Full 100-phantom locked evaluation for all 3 seeds at SNR 10, 15, and 20 dB, adding MAE, Bias, $R^2$, Lin's CCC, and per-phantom variance.

### 5.2 Verification Automation
The companion script `verify_unet_snr_metrics.py` confirms:
1. Exact presence of all 3 seeds and 3 SNRs (0 missing, 0 duplicates).
2. 100% bit-for-bit identity of ground-truth maps ($Y_{\text{CBF}}, Y_{\text{ATT}}$) and brain masks across all three SNR conditions.
3. Strict parity between aggregate mean/std and raw per-seed records ($|\Delta| < 10^{-16}$).
4. Zero cohort leakage against training (600) and validation (100) phantoms.

### 5.3 Reproduction Command
To execute the automated non-destructive verification suite:
```powershell
python experiments_unet_4pld/results/snr_10_15_20_audit/verify_unet_snr_metrics.py
```
*(Verification outcome: `ALL AUDIT VERIFICATION CHECKS PASSED SUCCESSFULLY (6/6)`)*
