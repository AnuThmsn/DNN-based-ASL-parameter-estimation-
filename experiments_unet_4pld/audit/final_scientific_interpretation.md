# Scientific Interpretation, Generalization Dynamics, and Methodological Limitations

**Study:** 4-PLD ASL Spatial U-Net Training-Size Generalization ($N \in \{100, 200, 400, 600, 1200, 2400\}$)  
**Location:** `D:\ASL_WORKSPACE\experiments\training_size_study`  
**Date:** October 2026  

---

## 1. Rigorous Analysis of Overfitting and Underfitting

### 1.1 What the Generalization Gap Trajectory Actually Tells Us
Previous reports informally claimed that because the generalization gap ($\mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}}$) remained small, "overfitting has been conclusively ruled out." A scientifically rigorous examination requires a more nuanced interpretation:

1. **Empirical Gap Values:**
   - $N=100$: $\text{Gap} = +0.0140 \pm 0.0068$
   - $N=200$: $\text{Gap} = +0.0235 \pm 0.0051$
   - $N=400$: $\text{Gap} = +0.0171 \pm 0.0034$
   - $N=600$: $\text{Gap} = +0.0153 \pm 0.0071$
   - $N=1200$: $\text{Gap} = +0.0111 \pm 0.0099$
   - $N=2400$: $\text{Gap} = +0.0174 \pm 0.0020$
2. **What This Supports:**
   - A classical, severe overfitting regime is characterized by **divergence**: training loss approaches zero while validation loss reverses direction and increases, causing the generalization gap to expand systematically with additional epochs.
   - Across all 18 runs, validation loss smoothly tracks training loss down to its minimum, and early stopping triggers when validation loss plateaus, before divergence occurs.
   - Furthermore, as $N$ increases from 100 to 2,400, the gap does not grow; it remains stable within a narrow band of $+0.011$ to $+0.024$.
3. **What This Does NOT Rule Out:**
   - A non-diverging generalization gap does *not* prove the absence of subtle distribution memorization. If the synthetic phantom distribution shares systematic structural features (e.g., smooth Gaussian perfusion fields, idealized vascular geometries), the network can achieve low validation and test loss within that family while still failing to generalize to clinical pathologies (e.g., acute watershed strokes, hydrocephalus, or severe focal territorial occlusions).
   - Therefore, while the spatial U-Net is **well-regularized within the synthetic ASL phantom domain**, domain-generalization to out-of-distribution clinical phenotypes remains an independent concern that cannot be settled by in-distribution test sets alone.

### 1.2 Underfitting Assessment
- **Training Loss Trajectory:** Training loss at convergence decreases monotonically from $0.0862 \pm 0.0037$ at $N=100$ to $0.0541 \pm 0.0016$ at $N=2400$.
- **Model Capacity Ablation:** Reducing the standard 7.76M parameter U-Net to a Compact U-Net (1.94M parameters, $-75\%$ size) at $N=600$ degraded CBF RMSE by $+10.0\%$ ($1.001 \to 1.101\text{ mL/100g/min}$) and ATT RMSE by $+7.5\%$ ($0.0573 \to 0.0616\text{ s}$).
- **Interpretation:** The model has sufficient capacity to learn from increased sample diversity without hitting an architectural bottleneck, and the 7.76M parameter capacity is justified.

---

## 2. Learning-Curve Dynamics and Plateau Evaluation

### 2.1 Power-Law Scaling Law
Fitting a log-linear power-law model $\text{RMSE} = C \cdot N^{-\alpha}$:
- **CBF RMSE:** $\alpha_{\text{CBF}} \approx 0.141$ ($R^2 = 0.992$)
- **ATT RMSE:** $\alpha_{\text{ATT}} \approx 0.115$ ($R^2 = 0.988$)

```
CBF RMSE (mL/100g/min):
  N=100:  1.275 ± 0.028
  N=200:  1.145 ± 0.021   (-10.2% drop)
  N=400:  1.076 ± 0.019   (-6.0% drop)
  N=600:  1.001 ± 0.031   (-7.0% drop)
  N=1200: 0.915 ± 0.022   (-8.6% drop)
  N=2400: 0.813 ± 0.023   (-11.2% drop)
```

### 2.2 Magnitude of Improvement vs Run-to-Run Variability
- At $N=600$, run-to-run standard deviation across seeds is $\sigma_{\text{CBF}} = 0.0312\text{ mL/100g/min}$.
- The gain from $N=600 \to 2400$ is $\Delta \mu = 1.0007 - 0.8129 = 0.1878\text{ mL/100g/min}$.
- The signal-to-noise ratio of this improvement is:
  $$\frac{|\Delta \mu|}{\sqrt{\sigma_{600}^2 + \sigma_{2400}^2}} = \frac{0.1878}{\sqrt{0.0312^2 + 0.0226^2}} \approx 4.88$$
- This confirms that the observed accuracy improvement of quadrupling the dataset from 600 to 2,400 phantoms exceeds seed-to-seed stochasticity by nearly $5$ standard errors.

### 2.3 Verdict on the "Plateau"
- There is **no hard mathematical plateau** up to $N=2,400$.
- However, there are **clear diminishing relative returns**: moving from $N=100 \to 600$ reduces CBF RMSE by $0.274\text{ mL/100g/min}$ (using 500 extra phantoms), whereas moving from $N=600 \to 2400$ reduces CBF RMSE by $0.188\text{ mL/100g/min}$ (requiring 1,800 extra phantoms).
- Clinically, an error of $0.81\text{ mL/100g/min}$ on CBF and $0.049\text{ s}$ on ATT represents exceptional precision against synthetic ground truth, well below typical clinical MRI physiological noise thresholds.

---

## 3. Methodological Limitations

1. **Synthetic Phantom Bias:** All phantoms were generated from synthetic geometric shapes and Buxton two-compartment hemodynamics. While noise was accurately modeled with dual-Rician physics, synthetic phantoms lack true microvascular tortuosity, subject motion artifacts, and magnetic susceptibility gradients present in in vivo human brain scans.
2. **Four-PLD Sampling Window:** The four PLDs ($[1.525, 2.025, 2.525, 3.025]\text{ s}$) do not sample arrivals earlier than 1.5 s or later than 3.0 s. In patients with delayed arterial transit (e.g., severe carotid stenosis or moyamoya disease where ATT $> 3.0\text{ s}$), late-arriving boluses cannot be detected by this 4-PLD sequence, regardless of how many training phantoms are used.
3. **Seed Cohort Size:** While 3 independent training seeds per condition ($N=3$) suffice to estimate run-to-run standard deviation, establishing narrower confidence intervals on the scaling exponent $\alpha$ would benefit from 5–10 seeds per condition if compute allowed.
