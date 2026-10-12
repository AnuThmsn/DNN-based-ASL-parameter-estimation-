# Benchmark Protocol, Ablation Integrity, and Comparative Timing Audit

**Workspace Root:** `D:\ASL_WORKSPACE`  
**Target Repository:** `experiments_unet_4pld/audit/`  
**Audit Date:** October 12, 2026  
**Status:** **AUDITED, QUALIFIED, & VERIFIED**

---

## 1. Executive Summary

This protocol audit reviews the experimental and benchmark procedures across two critical areas:
1. **Four-PLD Baseline vs Three-PLD Secondary Ablation (`run_3pld_ablation.py`):** Channel indexing, data protocol parity, physiological mechanics of PLD omission, and formal qualification of previously claimed "equivalence".
2. **Matched-Method Benchmark (`final_benchmark_audit.md`):** Protocol parity across models, strict isolation of historical literature/PDF baselines, disclosure of timing methodology (GPU vs CPU), and qualification of spatial degeneracy claims.

---

## 2. Four-PLD vs Three-PLD Secondary Ablation Audit

### 2.1 Canonical Primary Protocol
The primary acquisition protocol for this project is strictly and immutably fixed at **4 Post-Labeling Delays (PLDs)**:
$$\text{PLDs} = [1.525, 2.025, 2.525, 3.025] \text{ s}, \quad \tau = 1.80 \text{ s}$$
All three-PLD experiments are treated strictly as secondary exploratory ablations to assess acquisition economy, not as replacements for the primary 4-PLD deliverable.

### 2.2 Input Channel Subset Verification
In `experiments_unet_4pld/run_3pld_ablation.py`, four 3-channel input subsets were evaluated. The channel slicing and PLD assignments were audited and verified bit-for-bit:

| Subset Key | Omitted PLD | Input Channels Selected | Physical PLDs Kept (s) | CBF Relative Error Change (%) | ATT Relative Error Change (%) | Descriptive Status |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **Reference** | None (4-PLD) | `[0, 1, 2, 3]` | `[1.525, 2.025, 2.525, 3.025]` | — | — | **Primary Baseline** |
| `drop_P4` | $P_4 = 3.025$ s | `[0, 1, 2]` | `[1.525, 2.025, 2.525]` | -1.13% | -0.51% | Descriptively Comparable |
| `drop_P3` | $P_3 = 2.525$ s | `[0, 1, 3]` | `[1.525, 2.025, 3.025]` | +4.68% | +3.37% | Descriptively Comparable |
| `drop_P2` | $P_2 = 2.025$ s | `[0, 2, 3]` | `[1.525, 2.525, 3.025]` | +6.90% | +5.59% | Descriptively Comparable |
| `drop_P1` | $P_1 = 1.525$ s | `[1, 2, 3]` | `[2.025, 2.525, 3.025]` | **+68.72%** | **+179.59%** | **Severely Degraded** |

### 2.3 Physical Mechanism of Ablation Asymmetry
The marked difference in sensitivity between dropping $P_1$ vs $P_4$ is rooted in the Buxton kinetic signal equation:
$$\Delta M(t) = 2 \alpha \beta T_{1t} \lambda^{-1} f \cdot e^{-\delta / T_{1a}} \left( e^{-\max(t - \delta, 0)/T_{1t}} - e^{-\max(\tau + t - \delta, 0)/T_{1t}} \right)$$
1. **$P_1 = 1.525$ s is indispensable for arterial arrival:** In typical gray and white matter, $\delta \in [0.7, 1.5]$ s. For boluses arriving around $1.0\text{--}1.4$ s, $P_1$ captures the initial arrival peak. Without $P_1$, the network only observes signals at $t \ge 2.025$ s, by which time the bolus has already entered the tail washout phase, destroying arrival time identifiability ($+179.59\%$ ATT RMSE increase).
2. **$P_4 = 3.025$ s contains predominantly attenuated signal:** By $3.025$ s, labeled water has cleared the microvasculature while $T_1$ decay ($T_{1t} = 1.2$ s) has attenuated the signal toward the noise floor. Thus, omitting $P_4$ incurs minimal information penalty.

### 2.4 Formal Qualification of "Equivalence" Claims
> [!WARNING]
> **Scientific Qualification:** Previous documentation asserted that Subset 1 (`drop_P4`) achieved "proven clinical equivalence within +10%".
> **Audit Finding:** The $\pm 10\%$ bound was an informal retrospective threshold rather than a pre-registered equivalence margin evaluated via formal Two One-Sided Tests (TOST) or 90% confidence intervals.
> **Corrected Stance:** Claims of "proven equivalence" are replaced with **"descriptive comparability within empirical bounds"**. The primary model remains strictly the 4-PLD network.

---

## 3. Matched-Method Benchmark Validation

### 3.1 Preservation of Historical Baselines
In accordance with research integrity standards:
- Historical Bayesian, Nonlinear Least Squares (NLS), Pointwise DNN, and Residual ELM numbers documented in `ASL_MRI_Model_Comparison.pdf` and `reference_baselines.yaml` were **neither retrained nor simulated**. Their numerical values remain exactly as originally reported in literature.
- All newly trained U-Net evaluations are explicitly labeled as newly computed results on our standardized synthetic benchmark.

### 3.2 Evaluation Parity Matrix
All comparative models evaluated in Table 1 were tested on identical data:
- **Shared Cohort:** Exactly the same 25 test phantoms (39,410 valid voxels) or 100 test phantoms (171,356 valid voxels).
- **Physical Forward Model:** Buxton single-compartment model ($\tau = 1.8$ s, $T_{1t} = 1.2$ s, $T_{1a} = 1.66$ s, $\alpha = 0.85$, $\lambda = 0.9$).
- **Noise Formulation:** Rician noise model calibrated with $\sigma_{\text{eff}} = \sqrt{2} \sigma$.
- **Parameter Bounds:** CBF $\in [0, 100]$ mL/100g/min, ATT $\in [0.5, 3.0]$ s.

### 3.3 Disclosure of Computational Timing Methodology
> [!IMPORTANT]
> **Timing Methodology Limitation:**
> In Table 1 of `final_benchmark_audit.md`, runtime was reported as:
> - U-Net: **0.02 s** (50 phantoms/s)
> - Point-wise DNN: **0.08 s**
> - Calibrated Bayesian: **0.37 s**
> - Bounded TRF NLLS: **130.17 s**
> 
> **Audit Disclosure:**
> 1. U-Net, Point-wise DNN, and Bayesian estimators were executed on an **NVIDIA GeForce GTX 1650 GPU** using batched tensor matrix operations.
> 2. Bounded TRF NLLS was executed **sequentially on CPU** (single core) using SciPy's `least_squares` solver across each voxel independently ($39,410$ sequential non-linear optimization problems).
> 3. Consequently, the reported speedup is a **GPU-parallel tensor vs CPU-sequential iterative solver comparison**, reflecting operational deployment throughput rather than pure algorithmic computational complexity on identical hardware.

### 3.4 Qualification of Spatial Degeneracy Claims
Previous drafts claimed that "spatial convolutions in U-Net eliminate the physical CBF/ATT degeneracy."  
**Audit Qualification:** Spatial convolutions exploit spatial autocorrelation and anatomical continuity of transit delays across neighboring cortical voxels. While this spatial regularization substantially narrows empirical error variance relative to single-voxel estimators, it does not physically eliminate the underlying mathematical non-identifiability of ill-conditioned arrival profiles when acquisitions lack temporal sampling. This capability should be described as **"spatial regularizing context that mitigates local variance"** rather than "eliminating physical degeneracy."
