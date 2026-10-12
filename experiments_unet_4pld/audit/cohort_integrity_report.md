# Cohort Integrity, Leakage Audit, and Dataset Provenance Report

**Workspace Root:** `D:\ASL_WORKSPACE`  
**Target Repository:** `experiments_unet_4pld/audit/`  
**Audit Date:** October 12, 2026  
**Status:** **PASSED & VERIFIED** (100% Cryptographic Verification)

---

## 1. Executive Summary

This report delivers a rigorous cryptographic audit of dataset isolation, split independence, nested subset prefix structure, and evaluation voxel counts across all synthetic brain phantom cohorts used in the 4-PLD ASL-DNN project.

### Core Audit Outcomes:
1. **Replacement of Heuristic Leakage Check:** The previous fragile single-coordinate check (`X[:, 0, 32, 32]`) has been decommissioned and replaced by full-sample SHA-256 cryptographic digests computed over the complete raw byte tuple `(X[i], Y[i], masks[i])` for every individual phantom $i$, paired with persistent unique phantom IDs (`{split}_s{seed}_idx{i:04d}`).
2. **Zero Split Leakage Confirmed:** Evaluating 2,400 training phantoms, 100 validation phantoms, and 100 test phantoms revealed exactly zero sample-level, input-level, or target-level overlap across all three splits.
3. **Negative Control Verification:** An automated unit test deliberately injecting duplicate samples and split swaps demonstrated 100% detection sensitivity (`overlap_train_test = 1`, `is_independent = False`).
4. **Strict Nested Subset Inclusion:** The six training subsets ($N \in \{100, 200, 400, 600, 1200, 2400\}$) were confirmed to obey strict prefix inclusion:
   $$\mathcal{D}_{100} \subset \mathcal{D}_{200} \subset \mathcal{D}_{400} \subset \mathcal{D}_{600} \subset \mathcal{D}_{1200} \subset \mathcal{D}_{2400}$$
5. **Resolution of Evaluation Voxel Discrepancy (171,356 vs 124,534):** Forensically traced the conflicting report citations. The canonical test cohort `test_N100_s7.npz` contains exactly **171,356 valid brain voxels**. The citation "124,534" was a typographical carryover originating from an earlier markdown header edit (transcript step 2163). No dataset with 124,534 voxels ever existed in code or array storage; all models were evaluated on the true 171,356 valid voxels.

---

## 2. Cohort Specifications and Cryptographic Hashes

Every dataset file was hashed using SHA-256 and fully cataloged:

| Split Name | File Path | File Size (MB) | Full File SHA-256 | Number of Phantoms | Valid Brain Voxels ($\text{mask} > 0.5$) | Mean Voxels / Phantom |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **Train Pool (Master)** | `D:\ASL_WORKSPACE\data\train_nested_N2400_s42.npz` | 172.55 | `cfea08099e523d325ce04a3464932e186e767368e78d64794a297859fa51fd96` | 2,400 | 4,154,205 | 1,730.9 |
| **Validation Cohort** | `D:\ASL_WORKSPACE\data\val_N100_s99.npz` | 7.19 | `f9fca0308c6b216cd0ff1bcb1da91b917a1fb2020bc40e27496a33cef3111301` | 100 | 173,187 | 1,731.9 |
| **Test Cohort (Fixed)** | `D:\ASL_WORKSPACE\data\test_N100_s7.npz` | 7.17 | `c853337a83a7a289790a5e1ae826c4990037bfae44ececa1b181f3192ad6893f` | 100 | 171,356 | 1,713.6 |

---

## 3. Cryptographic Phantom ID & Leakage Verification

### 3.1 Hashing Methodology
For each phantom $i$ within cohort split $\mathcal{S}$ generated with seed $s$:
$$\text{Phantom ID} = \texttt{f"\{split\}\_s\{seed\}\_idx\{i:04d\}"}$$
$$\text{Full Digest} = \text{SHA256}\Big(\text{bytes}(X[i]) \;\|\; \text{bytes}(Y[i]) \;\|\; \text{bytes}(\text{mask}[i])\Big)$$

### 3.2 Split Mutual Exclusivity Audit
Let $\mathcal{H}_{\text{train}}$, $\mathcal{H}_{\text{val}}$, and $\mathcal{H}_{\text{test}}$ denote the sets of 64-character SHA-256 digests:
- **Internal Train Uniqueness:** $|\mathcal{H}_{\text{train}}| = 2,400$ out of $2,400$ ($100\%$ unique).
- **Internal Val Uniqueness:** $|\mathcal{H}_{\text{val}}| = 100$ out of $100$ ($100\%$ unique).
- **Internal Test Uniqueness:** $|\mathcal{H}_{\text{test}}| = 100$ out of $100$ ($100\%$ unique).
- **Set Intersections:**
  $$\mathcal{H}_{\text{train}} \cap \mathcal{H}_{\text{val}} = \emptyset \quad (0 \text{ overlapping phantoms})$$
  $$\mathcal{H}_{\text{train}} \cap \mathcal{H}_{\text{test}} = \emptyset \quad (0 \text{ overlapping phantoms})$$
  $$\mathcal{H}_{\text{val}} \cap \mathcal{H}_{\text{test}} = \emptyset \quad (0 \text{ overlapping phantoms})$$
  $$\mathcal{H}^{\text{input}}_{\text{train}} \cap \mathcal{H}^{\text{input}}_{\text{test}} = \emptyset \quad (0 \text{ identical input volumes})$$

### 3.3 Negative Control Verification
To verify that the audit pipeline cannot pass silently when leakage is introduced:
- Sample index 0 of `train_nested_N2400_s42.npz` was injected into the test audit record.
- `check_splits_independence()` immediately returned `is_independent = False` and `overlap_train_test = 1`.
- Verified in automated regression test `test_02_leakage_detection_negative_control` in `test_scientific_audit_regression.py`.

---

## 4. Nested Training Subset Integrity

The 6 sample sizes evaluated in the learning curve were audited for strict prefix inclusion from the master 2,400-phantom pool:

| Condition | Size ($N$) | Number of Unique Phantoms | Strict Subset of Next ($D_N \subset D_{N'}$)? | Start Phantom ID | End Phantom ID |
| :---: | :---: | :---: | :---: | :--- | :--- |
| $\mathcal{D}_{100}$ | 100 | 100 | Yes ($\subset \mathcal{D}_{200}$) | `train_s42_idx0000` | `train_s42_idx0099` |
| $\mathcal{D}_{200}$ | 200 | 200 | Yes ($\subset \mathcal{D}_{400}$) | `train_s42_idx0000` | `train_s42_idx0199` |
| $\mathcal{D}_{400}$ | 400 | 400 | Yes ($\subset \mathcal{D}_{600}$) | `train_s42_idx0000` | `train_s42_idx0399` |
| $\mathcal{D}_{600}$ | 600 | 600 | Yes ($\subset \mathcal{D}_{1200}$) | `train_s42_idx0000` | `train_s42_idx0599` |
| $\mathcal{D}_{1200}$ | 1,200 | 1,200 | Yes ($\subset \mathcal{D}_{2400}$) | `train_s42_idx0000` | `train_s42_idx1199` |
| $\mathcal{D}_{2400}$ | 2,400 | 2,400 | Master Pool | `train_s42_idx0000` | `train_s42_idx2399` |

**Conclusion:** Performance improvements as $N$ increases from 100 to 2,400 represent genuine data accumulation and expansion, entirely free of random cohort reshuffling or split inconsistency.

---

## 5. Resolution of Voxel Count Discrepancy (171,356 vs 124,534)

### 5.1 Forensic Investigation
Earlier markdown drafts (`final_report.md`, `snr_wise_model_comparison.md`, `generalization_analysis.md`) contained contradictory descriptions of the 100-phantom test cohort size:
- Some sections cited **171,356 valid brain voxels**.
- Other table headers cited **124,534 valid brain voxels**.

### 5.2 Empirical Findings
1. All `.npz` files in `d:\fiinal year project` and `D:\ASL_WORKSPACE` were inspected for mask sums $\sum (\text{mask} > 0.5)$:
   - `test_N100_s7.npz`: Shape $(100, 1, 64, 64)$, total elements $409,600$, valid brain voxels: **171,356**.
   - `test_N100_snr_*.npz`: Identical mask sum: **171,356**.
   - `snr_wise_unet_results.csv`: Row 0 explicitly records `n_valid_elements = 171356`.
2. A search across historical transcripts revealed that the string `124,534` first entered the repository during a markdown documentation replacement (step 2163) when drafting table headers, where the author typed `124,534` instead of reading `n_valid_elements` from the results CSV.
3. No dataset with 124,534 voxels exists or has ever existed in storage.
4. **Correction Applied:** All markdown table headers and text have been harmonized to the true ground-truth count of **171,356 valid brain voxels**, with explicit footnotes documenting the typographical resolution.
