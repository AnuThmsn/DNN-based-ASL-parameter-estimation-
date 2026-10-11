# Four-PLD Spatial U-Net for ASL-MRI Parameter Estimation

## 1. Overview
This experiment implements a **Four-PLD 2D Convolutional U-Net** for joint estimation of **Cerebral Blood Flow (CBF)** and **Arterial Transit Time (ATT)** from multi-delay Arterial Spin Labeling (ASL) MRI.

Unlike previous point-wise MLPs (which treat each voxel as an isolated 4-element time series), the U-Net leverages spatial context, anatomical compartment coherence, and convolutional inductive biases to denoise low-SNR perfusion signals and predict continuous parameter maps.

## 2. Experimental Configuration
- **Acquisition Protocol (Frozen):**
  - Exactly 4 PLDs: `[1.525, 2.025, 2.525, 3.025]` seconds
  - Labeling duration ($\tau$): 1.8 s
  - Longitudinal relaxation times: $T_{1a} = 1.66$ s (blood), $T_{1t} = 1.20$ s (tissue)
  - Labeling efficiency ($\alpha$): 0.85
  - Transit factor ($\beta$): 0.75
  - Partition coefficient ($\lambda$): 0.90
  - Signal scaling: $100,000.0$
- **Physiological Bounds:**
  - CBF: $[0.0, 100.0]$ ml/100g/min
  - ATT: $[0.5, 3.0]$ s

## 3. Spatial Synthetic Phantoms
To provide meaningful spatial structure:
- **Geometry:** 2D axial brain cross-sections ($64 \times 64$) with procedural lobe perturbations and central ventricles.
- **Anatomy:** Cortical Gray Matter (GM), Deep Gray Matter (basal ganglia/thalamus), Deep White Matter (WM), and CSF/ventricles.
- **Hemodynamics:**
  - GM: CBF $\sim 55-75$, ATT $\sim 1.0-1.4$ s
  - WM: CBF $\sim 18-32$, ATT $\sim 1.6-2.1$ s
  - Continuous 2D Gaussian random fields across tissue compartments plus vascular watershed transit delay gradients.
- **Observation Model:**
  - Voxel-wise Buxton kinetic model.
  - Two-channel Rician magnitude noise matching the repository's authoritative `data_gen.py` simulator across variable SNRs ($5$ to $50$, with explicit tracking at SNR=10 and SNR=15).
  - Background/non-brain voxels are zeroed and excluded using a binary mask.

## 4. Architecture
- **Model:** Conventional 2D U-Net (`ASLUNet`)
- **Input:** $[B, 4, H, W]$ (4 PLD perfusion channels)
- **Output:** $[B, 2, H, W]$ (Continuous parameter maps for CBF and ATT)
- **Encoder:** 4 downsampling stages with DoubleConv (3x3 Conv + BatchNorm + ReLU) + MaxPool2d (features: 32, 64, 128, 256).
- **Bottleneck:** 512 channels.
- **Decoder:** 4 upsampling stages with ConvTranspose2d + Skip concatenation + DoubleConv.
- **Output Head:** 1x1 Conv with 2 channels (no activation, continuous regression).
- **Total Trainable Parameters:** 7,763,362.

## 5. Training Protocol
- **Loss:** Standardized Masked L1 Loss (computed strictly over valid brain voxels $M_{brain} = 1$).
- **Optimizer:** AdamW (initial learning rate = $1 \times 10^{-3}$, weight decay = $1 \times 10^{-4}$).
- **Scheduler:** `ReduceLROnPlateau` (factor = 0.5, patience = 5, min_lr = $1 \times 10^{-6}$).
- **Batch Size:** 32.
- **Early Stopping:** Patience = 12 epochs on validation masked loss.
- **Multi-Seed Verification:** Seeds 42, 43, 44.

## 6. Baselines & Benchmarking
All methods are evaluated on the **exact same simulated test phantoms**:
1. **U-Net (Proposed)**: 2D spatial image-to-image regression.
2. **Point-wise DNN Baseline**: Standard 9-layer MLP applied voxel-wise.
3. **Bayesian Estimator**: Fast vectorized grid marginalization over flat parameter priors.
4. **Non-Linear Least Squares (NLLS)**: Scipy TRF bounded solver.
5. **Historical Reference Values**: From `reference_baselines.yaml` (SNR=10 and SNR=15).

## 7. Metrics
- **Quantitative Parameter Accuracy:** RMSE, MAE, Mean Bias, $R^2$, and Lin's Concordance Correlation Coefficient (CCC).
- **Spatial Quality:** Masked SSIM and Masked PSNR.
- **SNR Breakdown:** Stratified at SNR=10 and SNR=15.

