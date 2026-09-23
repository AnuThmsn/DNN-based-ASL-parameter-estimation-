# Experiment G Improvement Plan

## 1. Reproduce Experiment G
- [ ] Write `run_exp_G.py` (Asymmetric branching: CBF 5x128, ATT 4x64).
- [ ] Train with 3 seeds.
- [ ] Save models and predictions.
- [ ] Record BASELINE-G metrics (CBF/ATT RMSE, MAE, R², CCC).

## 2. Diagnose Experiment G
- [ ] Write `diagnose_exp_G.py`.
- [ ] Calculate error by CBF bins (0-20, 20-40, 40-60, 60-80, 80-100).
- [ ] Calculate error by ATT bins (0.5-1.0, 1.0-1.5, 1.5-2.0, 2.0-2.5, 2.5-3.0).
- [ ] Calculate error by SNR (10, 15, 20, 30, 40, 50).
- [ ] Analyze PLD information content (Jacobian sensitivity).

## 3. Targeted Improvements
- [ ] **G1: PLD-Aware Encoder.** Treat inputs as (PLD, S_i) pairs, shared small MLP, aggregate, then separate heads.
- [ ] **G2: Separate Representations.** Shared latent representation -> 128xCBF head, 64xATT head.
- [ ] **G3: Feature Engineering.** Ablations on Log, Diff, Ratio.
- [ ] **G4: Residual Refinement.** Initial guess -> DNN -> final estimate.
- [ ] **G5: Noise-Robust / Hard-case.** Curriculum or SNR sampling (if SNR diagnosis justifies it).

## 4. Final Evaluation & PLD Reduction
- [ ] Compare best architecture against the established fixed reference.
- [ ] Test 3-PLD vs 4-PLD for the best model.
- [ ] Produce final Master Table.
