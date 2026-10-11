# Four-PLD Spatial U-Net and Reduced-PLD Ablation for ASL-MRI Parameter Estimation

## 1. Overview
This directory contains the audited, scientifically validated implementation of the **Spatial 2D Convolutional U-Net** and the **Reduced-PLD Ablation Study** for joint estimation of:
- **Cerebral Blood Flow (CBF)**: ml/100g/min
- **Arterial Transit Time (ATT)**: seconds

From multi-delay Arterial Spin Labeling (ASL) MRI.

---

## 2. Key Scientific Findings & Breakthroughs

1. **Resolution of Wash-Out Degeneracy via Spatial Context:**
   When $\text{ATT} < \text{PLD}_1 = 1.525$ s, bolus wash-in completes before the first measurement, creating a mathematical degeneracy along $\text{CBF} \cdot e^{\text{ATT} \cdot \Delta R_1} = \text{const}$. Single-voxel methods (Point-wise DNN, Bayesian, NLLS) cannot decouple CBF and ATT in this regime. The spatial U-Net successfully resolves this degeneracy by aggregating neighboring cortical context and transit gradients, achieving:
   - **CBF RMSE: $0.991 \pm 0.032$ ml/100g/min** ($R^2 = 0.995$, $CCC = 0.998$)
   - **ATT RMSE: $0.0569 \pm 0.0012$ s** ($R^2 = 0.979$, $CCC = 0.989$)
   - **SSIM / PSNR:** CBF SSIM = $0.995$, CBF PSNR = $40.59$ dB; ATT SSIM = $0.983$, ATT PSNR = $34.04$ dB.

2. **Rigorous Paired Statistical Significance:**
   On a strictly harmonized benchmark of 25 independent test phantoms (39,410 valid brain voxels):
   - **U-Net vs Point-wise DNN**: CBF diff $-1.57$ ml/100g/min ($p = 3.07 \times 10^{-18}$), ATT diff $-0.162$ s ($p = 4.65 \times 10^{-21}$)
   - **U-Net vs Calibrated Bayesian**: CBF diff $-2.18$ ml/100g/min ($p = 3.14 \times 10^{-22}$), ATT diff $-0.267$ s ($p = 5.38 \times 10^{-24}$)
   - **U-Net vs Bounded NLLS**: CBF diff $-2.38$ ml/100g/min ($p = 4.42 \times 10^{-19}$), ATT diff $-0.256$ s ($p = 9.10 \times 10^{-22}$)
   Every comparison shows overwhelming statistical significance ($p < 10^{-17}$).

3. **3-PLD Ablation & Acquisition Reduction:**
   - **Subset 1 (Drop P4 = 3.025s, retaining [1.525, 2.025, 2.525]s)**:
     Achieves **CBF RMSE $0.9953 \pm 0.0092$** ($+0.41\%$ change) and **ATT RMSE $0.0574 \pm 0.0012$ s** ($+0.76\%$ change).
     **Verdict: Fully Comparable (< +1% degradation, well within prespecified +10% equivalence margin).**
     Enables a **25% reduction in scan time** with virtually no loss of precision.
   - **Subset 4 (Drop P1 = 1.525s, retaining [2.025, 2.525, 3.025]s)**:
     CBF RMSE degrades by $+69.9\%$ ($1.684$) and ATT RMSE degrades by $+180.8\%$ ($0.160$ s).
     **Physical explanation:** Early PLD sampling ($P_1 = 1.525$ s) is essential to detect bolus inflow for fast transit times.

---

## 3. Harmonized 4-PLD Benchmark (Table 1: Matched 25 Phantoms)

| Method | Evaluation Set | Time (s) | CBF RMSE (ml/100g/min) | CBF MAE | CBF $R^2$ | ATT RMSE (s) | ATT MAE | ATT $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net (Proposed)** | Matched (25 phantoms) | **0.02** | **0.960** | **0.675** | **0.995** | **0.0558** | **0.0368** | **0.979** |
| Point-wise DNN | Matched (25 phantoms) | 0.08 | 2.541 | 1.939 | 0.967 | 0.2156 | 0.1503 | 0.682 |
| Bayesian (Calibrated Grid) | Matched (25 phantoms) | 0.37 | 3.142 | 2.497 | 0.949 | 0.3216 | 0.2603 | 0.292 |
| NLLS (TRF Bounded) | Matched (25 phantoms) | 130.17 | 3.358 | 2.471 | 0.942 | 0.3111 | 0.2093 | 0.338 |

---

## 4. 3-PLD Ablation Study Summary

Evaluated on the full 100 test phantoms across seeds `[42, 123, 2024]`:

| Subset | Dropped PLD | Retained PLDs (s) | U-Net CBF RMSE | CBF $\Delta\%$ | U-Net ATT RMSE | ATT $\Delta\%$ | Status (+10% margin) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4-PLD Baseline** | None | `[1.525, 2.025, 2.525, 3.025]` | $0.9912 \pm 0.0323$ | - | $0.0569 \pm 0.0012$ | - | Reference Standard |
| **Subset 1 (Best)** | P4 (3.025s) | `[1.525, 2.025, 2.525]` | **$0.9953 \pm 0.0092$** | **+0.41%** | **$0.0574 \pm 0.0012$** | **+0.76%** | **Comparable** |
| **Subset 2** | P3 (2.525s) | `[1.525, 2.025, 3.025]` | $1.0456 \pm 0.0242$ | +5.49% | $0.0594 \pm 0.0012$ | +4.39% | Comparable |
| **Subset 3** | P2 (2.025s) | `[1.525, 2.525, 3.025]` | $1.0745 \pm 0.0237$ | +8.40% | $0.0603 \pm 0.0009$ | +5.90% | Comparable |
| **Subset 4 (Worst)**| P1 (1.525s) | `[2.025, 2.525, 3.025]` | $1.6841 \pm 0.0380$ | **+69.90%** | $0.1598 \pm 0.0025$ | **+180.78%**| **Degraded** |

---

## 5. Directory Structure

```
experiments_unet_4pld/
├── config.json                     # Canonical configuration and hyperparameter settings
├── dataset.py                      # Dataset loader, dynamic normalization, denormalization
├── model.py                        # 2D ASLUNet PyTorch architecture (7.76M parameters)
├── train.py                        # Training loop with Masked L1 loss, lr schedule, early stopping
├── evaluate.py                     # Harmonized benchmark runner, GPU Bayesian, NLLS, statistics
├── generate_spatial_data.py        # Procedural 2D brain phantom generator (GM/WM/CSF)
├── run_experiment.py               # Main 4-PLD multi-seed benchmark entry point
├── run_3pld_ablation.py            # Controlled 3-PLD subset ablation experiment runner
├── audit/
│   ├── initial_audit.md            # Physics simulator discrepancy and calibration audit
│   └── final_benchmark_audit.md    # Multi-seed validation and statistical significance audit
├── checkpoints/                    # Model weights for 4-PLD and 3-PLD best epochs
├── data/                           # Cached dataset npz files and normalization statistics
├── results/
│   ├── matched_benchmark_25_phantoms.csv # Table 1: exact same 25 phantoms
│   ├── master_comparison.csv             # Full-scale 100 phantom comparison + fixed references
│   ├── per_seed_metrics.csv              # Multi-seed metrics (seeds 42, 123, 2024)
│   ├── paired_statistical_tests.csv      # Paired t-tests and Wilcoxon tests
│   ├── ablation_3pld_summary.csv         # 3-PLD ablation summary and comparability verdicts
│   ├── ablation_3pld_per_seed.csv        # 3-PLD per-seed breakdown
│   ├── plots/                            # Loss curves, scatter plots, Bland-Altman, ablation charts
│   └── predictions/                      # Saved .npy prediction arrays for all methods
└── tests/
    ├── test_audit_fixes.py         # Unit tests verifying noise scale, likelihood, and seeds
    └── test_unet_pipeline.py       # Unit tests verifying U-Net shapes, losses, and normalization
```

---

## 6. How to Reproduce

### Run Unit Tests
```bash
python -m unittest experiments_unet_4pld/tests/test_audit_fixes.py
python -m unittest experiments_unet_4pld/tests/test_unet_pipeline.py
```

### Run 4-PLD Benchmark
```bash
python -u experiments_unet_4pld/run_experiment.py
```

### Run 3-PLD Ablation Study
```bash
python -u experiments_unet_4pld/run_3pld_ablation.py
```
