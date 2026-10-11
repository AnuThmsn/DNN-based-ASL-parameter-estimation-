# Phase 1 Audit: Storage Safety, Scientific Correctness, and Baseline Inventory

## 1. Environment & Git State

- **Baseline Commit:** `47f83e00270849a0fa7144216b16731265e86077`
- **Active Branch:** `storage-safe-4pld-final`
- **Working Tree:** Clean; historical results snapshotted at `experiments_unet_4pld/historical_snapshot_47f83e0/`.
- **System Storage Inventory (Windows):**
  - **Drive C: (System):** Free: **13.21 GB** / Total: 199.37 GB (~93.4% full). **CRITICAL RISK:** Dumping datasets, PyTorch model checkpoints, or system temporary files here will exhaust disk space and crash Windows.
  - **Drive D: (Data/Workspace):** Free: **146.28 GB** / Total: 275.07 GB (Plenty of headroom).

---

## 2. Inventory of Existing Project Artifacts

### 2.1 Code & Configuration Files
- `experiments_unet_4pld/config.json`: Canonical hyperparameters and seeds `[42, 123, 2024]`.
- `experiments_unet_4pld/model.py`: Standard 2D `ASLUNet` (7.76M parameters).
- `experiments_unet_4pld/dataset.py`: Dynamic channel normalization, slice dataset loading.
- `experiments_unet_4pld/train.py`: Masked L1 loss, AdamW, `ReduceLROnPlateau`.
- `experiments_unet_4pld/evaluate.py`: Full benchmark evaluation (Bayesian, NLLS, DNN, U-Net).
- `experiments_unet_4pld/generate_spatial_data.py`: Procedural 2D brain phantom generator ($64 \times 64$).
- `experiments_unet_4pld/run_experiment.py`: Four-PLD experiment runner.
- `experiments_unet_4pld/run_3pld_ablation.py`: Secondary 3-PLD ablation study runner.

### 2.2 Datasets & Checkpoints
- `experiments_unet_4pld/data/`:
  - `train_N600_s42.npz`: 600 phantoms (seed 42).
  - `val_N100_s99.npz`: 100 phantoms (seed 99).
  - `test_N100_s7.npz`: 100 phantoms (seed 7).
- `experiments_unet_4pld/checkpoints/`:
  - 15 `.pth` checkpoints totaling ~465 MB (4-PLD and 3-PLD best and final checkpoints).

### 2.3 Snapshotted Historical Results (`historical_snapshot_47f83e0/`)
- `matched_benchmark_25_phantoms.csv`: Table 1 matched results.
- `master_comparison.csv`: Full test set comparisons and literature references.
- `per_seed_metrics.csv`: 4-PLD multi-seed results (seeds 42, 123, 2024).
- `paired_statistical_tests.csv`: Paired t-tests and Wilcoxon tests ($p < 10^{-17}$).
- `ablation_3pld_summary.csv`: 3-PLD ablation metrics and comparability verdicts.
- `ablation_3pld_per_seed.csv`: Seed-level breakdown of 3-PLD subsets.
- Plots: Loss curves, scatter plots, Bland-Altman plots, ablation bar charts, sample maps.

---

## 3. Scientific Correctness Diagnosis

### 3.1 Test-Set Leakage in Seed Selection (Must Fix)
In `run_experiment.py` (lines 186–190):
```python
if met_c["RMSE"] < best_cbf_rmse:
    best_cbf_rmse = met_c["RMSE"]
    best_overall_seed = seed
    best_model_state = eval_model
```
**Diagnosis:** The best model checkpoint used for full benchmark evaluation was selected by comparing `met_c["RMSE"]` on the **test set**! This constitutes test-set leakage.
**Remedy:** Model checkpoint selection and seed ranking must be performed strictly using **validation loss** (`val_loss`) during training. The test set must be touched only once for the final unbiased evaluation.

### 3.2 Simulator and Likelihood Matching
- Forward kinetic model in `data_gen.py` scales signals by $100,000.0$.
- Noise model produces sum-of-Ricians observation $X = m_c + m_l$.
- In the high-SNR limit, $\mathbb{E}[X] \approx 2 \cdot \text{sig} \cdot \text{SCALE}$ and effective noise variance $\text{Var}(X) \approx 2\sigma^2$.
- Calibrated Bayesian grid and TRF NLLS correctly incorporate $\sigma_{eff} = \sqrt{2}\sigma$ and $2\times$ expected signal.

---

## 4. Storage Safety Diagnosis & Mitigation Plan

1. **Risk of Filling C: Drive:**
   Windows default `TEMP` environment variable points to `C:\Users\ANU THOMSON\AppData\Local\Temp`. PyTorch cache defaults to `C:\Users\ANU THOMSON\.cache\torch`. If left unmanaged, large datasets or temp files will fill the remaining 13 GB on C:.
2. **Centralized Workspace Architecture:**
   Introduce a dedicated module `experiments_unet_4pld/workspace.py` driven by `ASL_WORK_DIR` (defaulting to `D:\ASL_WORKSPACE`).
   Subdirectories:
   - `data/`: Reusable dataset files with metadata manifests.
   - `checkpoints/`: Model weights.
   - `results/`: CSV metrics and final artifacts.
   - `logs/`: Compact training logs.
   - `plots/`: Diagnostic figures.
   - `tmp/`: Disposable temporary files (redirecting `TMPDIR`/`TEMP`).
3. **Pre-flight Free-Space & Permissions Checks:**
   Check free disk space before generating data or training. If free space is below 5 GB on the workspace drive, halt safely.
4. **Cache Manifest & Reuse:**
   Store a JSON manifest alongside cached `.npz` files containing simulator hash, image dimensions, sample counts, PLD schedule, and SNR targets. Valid caches are reused without regeneration.

