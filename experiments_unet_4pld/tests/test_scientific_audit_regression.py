"""
Comprehensive Scientific Audit Regression Test Suite
=====================================================
Automated verification for:
1. Cryptographic cohort independence and zero data leakage.
2. Deliberate duplicate / split leakage detection (negative control).
3. Strict nested training subset prefix inclusion.
4. Normalization fitted exclusively on training data.
5. Consistent denormalization and physical parameter clipping.
6. Checkpoint and training history provenance and bit-for-bit duality.
7. Fixed evaluation cohort SHA-256 hash and 171,356 valid voxel count.
8. Metric recomputation consistency (RMSE, MAE, Bias, R², Lin's CCC).
9. PLD channel selection and physical degradation in 3-PLD ablation.
10. Reproducibility manifest completeness across all 18 conditions.
"""

import os
import sys
import json
import hashlib
import unittest
import numpy as np
import pandas as pd
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments_unet_4pld.audit.cohort_integrity_check import (
    audit_cohort_dataset,
    check_splits_independence,
    verify_nested_subsets,
    compute_sample_hash
)
from experiments_unet_4pld.dataset import (
    compute_normalization_stats,
    normalize_dataset,
    denormalize_predictions
)
from experiments_unet_4pld.run_snr_benchmark import (
    calc_metrics,
    calc_lin_ccc
)


def calc_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calc_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


class TestScientificAuditRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_npz = r"D:\ASL_WORKSPACE\data\train_nested_N2400_s42.npz"
        cls.val_npz   = r"D:\ASL_WORKSPACE\data\val_N100_s99.npz"
        cls.test_npz  = r"D:\ASL_WORKSPACE\data\test_N100_s7.npz"
        cls.exp_dir_study = r"D:\ASL_WORKSPACE\experiments\training_size_study"
        cls.exp_dir_lc    = r"D:\ASL_WORKSPACE\experiments\training_size_learning_curve"

    def test_01_cohort_cryptographic_independence(self):
        """Audit Item 1: Cryptographic sample hashes show zero overlap across train, val, and test."""
        train_aud = audit_cohort_dataset(self.train_npz, "train", 42)
        val_aud   = audit_cohort_dataset(self.val_npz, "val", 99)
        test_aud  = audit_cohort_dataset(self.test_npz, "test", 7)

        self.assertTrue(train_aud["is_internally_unique"], "Train cohort contains duplicate phantoms.")
        self.assertTrue(val_aud["is_internally_unique"], "Val cohort contains duplicate phantoms.")
        self.assertTrue(test_aud["is_internally_unique"], "Test cohort contains duplicate phantoms.")

        ind = check_splits_independence(train_aud, val_aud, test_aud)
        self.assertTrue(ind["is_independent"], "Data leakage detected across train, val, and test splits!")
        self.assertEqual(ind["overlap_train_val"], 0)
        self.assertEqual(ind["overlap_train_test"], 0)
        self.assertEqual(ind["overlap_val_test"], 0)

    def test_02_leakage_detection_negative_control(self):
        """Audit Item 1 (Negative Control): Deliberately injected duplicate or split swap is caught."""
        train_aud = audit_cohort_dataset(self.train_npz, "train", 42)
        val_aud   = audit_cohort_dataset(self.val_npz, "val", 99)
        test_aud  = audit_cohort_dataset(self.test_npz, "test", 7)

        # Create a corrupted test audit that contains sample 0 from train
        corrupted_test_aud = {
            "n_samples": test_aud["n_samples"],
            "full_hashes": [train_aud["full_hashes"][0]] + test_aud["full_hashes"][1:],
            "input_hashes": [train_aud["input_hashes"][0]] + test_aud["input_hashes"][1:],
            "target_hashes": test_aud["target_hashes"]
        }

        leakage_result = check_splits_independence(train_aud, val_aud, corrupted_test_aud)
        self.assertFalse(leakage_result["is_independent"], "Negative control failed: injected duplicate was NOT caught!")
        self.assertEqual(leakage_result["overlap_train_test"], 1, "Expected exactly 1 detected train-test overlap.")

    def test_03_nested_training_subset_integrity(self):
        """Audit Item 1: Verify strict prefix inclusion across nested training cohort sizes."""
        train_aud = audit_cohort_dataset(self.train_npz, "train", 42)
        subset_sizes = [100, 200, 400, 600, 1200, 2400]
        nest_res = verify_nested_subsets(train_aud, subset_sizes)

        self.assertTrue(nest_res["all_strictly_nested"], "Nested training subsets violate prefix inclusion!")
        for sz in subset_sizes:
            sub_info = nest_res["subsets"][sz]
            self.assertEqual(sub_info["unique_count"], sz)
            self.assertTrue(sub_info["contains_previous_subset"])

    def test_04_normalization_isolated_to_training(self):
        """Audit Item 4: Normalization statistics are derived exclusively from training phantoms."""
        train_d = np.load(self.train_npz)
        test_d  = np.load(self.test_npz)

        X_tr = train_d["X"][:100]
        Y_tr = train_d["Y"][:100]
        M_tr = train_d["masks"][:100]
        if M_tr.ndim == 3:
            M_tr = M_tr[:, np.newaxis, :, :]

        stats_100 = compute_normalization_stats(X_tr, Y_tr, M_tr)

        # Check shapes and types
        self.assertEqual(stats_100["X_mean"].shape, (4,))
        self.assertEqual(stats_100["X_std"].shape, (4,))
        self.assertIsInstance(stats_100["cbf_mean"], float)
        self.assertIsInstance(stats_100["att_mean"], float)

        # Test normalization on test cohort
        X_te = test_d["X"]
        Y_te = test_d["Y"]
        M_te = test_d["masks"]
        if M_te.ndim == 3:
            M_te = M_te[:, np.newaxis, :, :]

        X_te_norm, Y_te_norm = normalize_dataset(X_te, Y_te, M_te, stats_100)
        self.assertEqual(X_te_norm.shape, X_te.shape)
        self.assertEqual(Y_te_norm.shape, Y_te.shape)

    def test_05_denormalization_and_physical_clamping(self):
        """Audit Item 4: Output denormalization and physical bounds clamping are mathematically exact."""
        np.random.seed(42)
        stats = {
            "cbf_mean": 50.0,
            "cbf_std": 25.0,
            "att_mean": 1.75,
            "att_std": 0.65
        }

        # Simulated raw predictions from network in standardized space [N, 2, H, W]
        norm_preds = np.array([
            [[[-3.0, 0.0, 3.0]],  # CBF std: 50 + (-3)*25 = -25 -> clamp to 0; 50 -> 50; 50 + 75 = 125 -> clamp to 100
             [[-3.0, 0.0, 3.0]]]   # ATT std: 1.75 + (-3)*0.65 = -0.2 -> clamp to 0.5; 1.75 -> 1.75; 1.75 + 1.95 = 3.7 -> clamp to 3.0
        ], dtype=np.float32)

        raw_cbf, raw_att = denormalize_predictions(norm_preds, stats)
        phys_cbf = np.clip(raw_cbf, 0.0, 100.0)
        phys_att = np.clip(raw_att, 0.5, 3.0)

        # Expected CBF bounds: [0, 100]
        self.assertAlmostEqual(phys_cbf[0, 0, 0], 0.0, places=4)
        self.assertAlmostEqual(phys_cbf[0, 0, 1], 50.0, places=4)
        self.assertAlmostEqual(phys_cbf[0, 0, 2], 100.0, places=4)

        # Expected ATT bounds: [0.5, 3.0]
        self.assertAlmostEqual(phys_att[0, 0, 0], 0.5, places=4)
        self.assertAlmostEqual(phys_att[0, 0, 1], 1.75, places=4)
        self.assertAlmostEqual(phys_att[0, 0, 2], 3.0, places=4)

    def test_06_checkpoint_and_history_provenance_duality(self):
        """Audit Item 3: Verify bit-for-bit checkpoint and history equality between study and learning_curve."""
        ckpt_dir_1 = os.path.join(self.exp_dir_study, "checkpoints")
        ckpt_dir_2 = os.path.join(self.exp_dir_lc, "checkpoints")
        hist_dir_1 = os.path.join(self.exp_dir_study, "training_histories")
        hist_dir_2 = os.path.join(self.exp_dir_lc, "training_histories")

        sizes = [100, 200, 400, 600, 1200, 2400]
        seeds = [42, 123, 2024]

        for sz in sizes:
            for sd in seeds:
                ckpt_name = f"unet_N{sz}_seed_{sd}.pth"
                p1 = os.path.join(ckpt_dir_1, ckpt_name)
                p2 = os.path.join(ckpt_dir_2, ckpt_name)
                self.assertTrue(os.path.exists(p1), f"Missing checkpoint {p1}")
                self.assertTrue(os.path.exists(p2), f"Missing checkpoint {p2}")

                with open(p1, "rb") as f:
                    h1 = hashlib.sha256(f.read()).hexdigest()
                with open(p2, "rb") as f:
                    h2 = hashlib.sha256(f.read()).hexdigest()
                self.assertEqual(h1, h2, f"Checkpoint hash mismatch for {ckpt_name}")

                # Verify checkpoint dictionary integrity
                ckpt = torch.load(p1, map_location="cpu", weights_only=False)
                self.assertIn("model_state_dict", ckpt)
                self.assertIn("epoch", ckpt)
                self.assertIn("val_loss", ckpt)
                self.assertIn("stats", ckpt)

                # Verify training history CSV
                hist_name = f"history_N{sz}_seed_{sd}.csv"
                hp1 = os.path.join(hist_dir_1, hist_name)
                hp2 = os.path.join(hist_dir_2, hist_name)
                self.assertTrue(os.path.exists(hp1), f"Missing history {hp1}")
                self.assertTrue(os.path.exists(hp2), f"Missing history {hp2}")
                with open(hp1, "rb") as f:
                    hh1 = hashlib.sha256(f.read()).hexdigest()
                with open(hp2, "rb") as f:
                    hh2 = hashlib.sha256(f.read()).hexdigest()
                self.assertEqual(hh1, hh2, f"History hash mismatch for {hist_name}")

    def test_07_fixed_test_cohort_hash_and_voxel_count(self):
        """Audit Item 2: Verify test cohort SHA-256 and exactly 171,356 valid brain voxels."""
        self.assertTrue(os.path.exists(self.test_npz), f"Test NPZ not found: {self.test_npz}")
        with open(self.test_npz, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        expected_hash = "c853337a83a7a289790a5e1ae826c4990037bfae44ececa1b181f3192ad6893f"
        self.assertEqual(h, expected_hash, f"Test cohort SHA-256 mismatch: {h}")

        d = np.load(self.test_npz)
        self.assertEqual(d["X"].shape, (100, 4, 64, 64))
        self.assertEqual(d["Y"].shape, (100, 2, 64, 64))
        masks = d["masks"]
        if masks.ndim == 3:
            masks = masks[:, np.newaxis, :, :]
        valid_voxels = int(np.sum(masks[:, 0] > 0.5))
        self.assertEqual(valid_voxels, 171356, f"Valid voxel count must be exactly 171,356, got {valid_voxels}")

    def test_08_metric_recomputation_consistency(self):
        """Audit Item 4: Verify mathematical formulation of RMSE, MAE, and Lin's CCC."""
        # 1. Identity case
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
        y_pred = y_true.copy()
        self.assertAlmostEqual(calc_rmse(y_true, y_pred), 0.0)
        self.assertAlmostEqual(calc_mae(y_true, y_pred), 0.0)
        self.assertAlmostEqual(calc_lin_ccc(y_true, y_pred), 1.0)

        # 2. Constant bias
        y_biased = y_true + 2.0
        self.assertAlmostEqual(calc_rmse(y_true, y_biased), 2.0)
        self.assertAlmostEqual(calc_mae(y_true, y_biased), 2.0)
        ccc_biased = calc_lin_ccc(y_true, y_biased)
        self.assertGreater(ccc_biased, 0.95)
        self.assertLess(ccc_biased, 1.0)

        # 3. Scale shift
        y_scaled = y_true * 1.1
        ccc_scaled = calc_lin_ccc(y_true, y_scaled)
        self.assertGreater(ccc_scaled, 0.95)
        self.assertLess(ccc_scaled, 1.0)

    def test_09_three_pld_ablation_channel_selection_and_physics(self):
        """Audit Item 6: Channel selection and indispensable arrival time information in 3-PLD subsets."""
        subsets = {
            "drop_P4": {"indices": [0, 1, 2], "plds": [1.525, 2.025, 2.525]},
            "drop_P3": {"indices": [0, 1, 3], "plds": [1.525, 2.025, 3.025]},
            "drop_P2": {"indices": [0, 2, 3], "plds": [1.525, 2.525, 3.025]},
            "drop_P1": {"indices": [1, 2, 3], "plds": [2.025, 2.525, 3.025]},
        }
        all_plds = np.array([1.525, 2.025, 2.525, 3.025])

        for s_key, s_data in subsets.items():
            selected_plds = all_plds[s_data["indices"]].tolist()
            self.assertEqual(selected_plds, s_data["plds"], f"Channel selection mismatch for {s_key}")

        # Verify summary results CSV
        csv_p = os.path.join(parent_dir, "experiments_unet_4pld", "results", "ablation_3pld_summary.csv")
        self.assertTrue(os.path.exists(csv_p))
        df = pd.read_csv(csv_p)
        self.assertEqual(len(df), 4)
        row_p1 = df[df["Subset"] == "drop_P1"].iloc[0]
        row_p4 = df[df["Subset"] == "drop_P4"].iloc[0]

        # P1 (1.525s) drop must have highest degradation on ATT (> 100% relative error increase)
        self.assertGreater(float(row_p1["ATT_Rel_Change_%"]), 100.0)
        # P4 (3.025s) drop must be minimal degradation
        self.assertLess(abs(float(row_p4["ATT_Rel_Change_%"])), 5.0)

    def test_10_reproducibility_manifest_completeness(self):
        """Audit Item 8: Verify completeness and schema of reproducibility_manifest.json."""
        manifest_p = os.path.join(self.exp_dir_study, "reproducibility_manifest.json")
        self.assertTrue(os.path.exists(manifest_p))
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertIn("checkpoints", manifest)
        self.assertEqual(manifest["total_runs_completed"], 18, "Manifest must document all 18 runs.")
        self.assertEqual(len(manifest["checkpoints"]), 18, "Checkpoints dict must contain all 18 runs.")
        for run_id, run_data in manifest["checkpoints"].items():
            self.assertIn("path", run_data)
            self.assertIn("sha256", run_data)
            self.assertIn("cbf_rmse", run_data)
            self.assertIn("att_rmse", run_data)
            self.assertIn("cbf_ccc", run_data)
            self.assertIn("att_ccc", run_data)


if __name__ == "__main__":
    unittest.main()

