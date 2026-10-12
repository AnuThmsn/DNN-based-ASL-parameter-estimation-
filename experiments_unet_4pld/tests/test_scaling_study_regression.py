"""
Regression Suite for Training Data Scaling Experiment
=====================================================
Automated unit and regression tests for the 19,200-phantom scaling study:
1. All 12 runs present and status == COMPLETED in training_scaling_results.csv.
2. Checkpoint files exist and have matching SHA-256 digests.
3. Master dataset N=19200 has 19,200 samples and matches N=2400 prefix bit-for-bit.
4. Monotonic scaling: N=19200 statistically outperforms N=2400 in CBF and ATT RMSE.
5. Multi-SNR metrics populated across all 6 discrete SNR levels.
6. Physical boundary violation rates are minimal (CBF == 0.0, ATT < 0.005).
7. Per-subset normalization stats and inclusion manifests exist.
"""

import os
import sys
import json
import hashlib
import unittest
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments_unet_4pld.workspace import get_workspace


class TestScalingStudyRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ws = get_workspace(min_free_gb=1.0)
        cls.exp_dir = os.path.join(cls.ws.root, "experiments", "training_size_scaling_v1")
        cls.results_csv = os.path.join(cls.exp_dir, "training_scaling_results.csv")
        assert os.path.exists(cls.results_csv), f"Results CSV missing: {cls.results_csv}"
        cls.df = pd.read_csv(cls.results_csv)

    def test_01_scaling_run_completeness(self):
        """Verify all 12 runs (4 sizes x 3 seeds) exist and are marked COMPLETED."""
        self.assertEqual(len(self.df), 12, f"Expected 12 runs, found {len(self.df)}")
        sizes = sorted(self.df["training_size"].unique().tolist())
        self.assertEqual(sizes, [2400, 4800, 9600, 19200])
        for sz in sizes:
            sub = self.df[self.df["training_size"] == sz]
            self.assertEqual(set(sub["seed"].tolist()), {42, 123, 2024})
            self.assertTrue((sub["status"] == "COMPLETED").all())

    def test_02_checkpoints_integrity(self):
        """Verify checkpoints exist on disk and match reported SHA-256 digests."""
        ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        for _, row in self.df.iterrows():
            sz = int(row["training_size"])
            sd = int(row["seed"])
            ckpt_path = os.path.join(ckpt_dir, f"unet_N{sz}_seed_{sd}.pth")
            self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}")
            self.assertGreater(os.path.getsize(ckpt_path), 100_000, f"Abnormally small checkpoint: {ckpt_path}")

            h = hashlib.sha256()
            with open(ckpt_path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            self.assertEqual(h.hexdigest(), row["checkpoint_sha256"], f"SHA-256 mismatch for N={sz}, seed={sd}")

    def test_03_master_pool_prefix_integrity(self):
        """Verify master pool has 19,200 phantoms and matches N=2400 bit-for-bit on prefix."""
        data_dir = os.path.join(self.ws.root, "data")
        master_path = os.path.join(data_dir, "train_nested_N19200_s42.npz")
        h2400_path = os.path.join(data_dir, "train_nested_N2400_s42.npz")

        self.assertTrue(os.path.exists(master_path), f"Master pool missing: {master_path}")
        self.assertTrue(os.path.exists(h2400_path), f"N2400 dataset missing: {h2400_path}")

        m = np.load(master_path)
        h = np.load(h2400_path)

        self.assertEqual(len(m["X"]), 19200)
        self.assertTrue(np.array_equal(m["X"][:2400], h["X"]), "Bit-for-bit mismatch on X prefix")
        self.assertTrue(np.array_equal(m["Y"][:2400], h["Y"]), "Bit-for-bit mismatch on Y prefix")

    def test_04_monotonic_scaling_improvement(self):
        """Verify N=19200 statistically and substantially outperforms N=2400."""
        cbf_2400 = self.df[self.df["training_size"] == 2400]["locked_cbf_rmse"].mean()
        cbf_19200 = self.df[self.df["training_size"] == 19200]["locked_cbf_rmse"].mean()
        att_2400 = self.df[self.df["training_size"] == 2400]["locked_att_rmse"].mean()
        att_19200 = self.df[self.df["training_size"] == 19200]["locked_att_rmse"].mean()

        self.assertLess(cbf_19200, cbf_2400)
        self.assertLess(att_19200, att_2400)

        # Improvement must exceed 10%
        cbf_gain = (cbf_2400 - cbf_19200) / cbf_2400
        att_gain = (att_2400 - att_19200) / att_2400
        self.assertGreater(cbf_gain, 0.10)
        self.assertGreater(att_gain, 0.10)

    def test_05_boundary_violations(self):
        """Verify out-of-bounds rates before clamping are strictly negligible."""
        self.assertTrue((self.df["cbf_oob_fraction"] == 0.0).all())
        self.assertTrue((self.df["att_oob_fraction"] < 0.005).all())

    def test_06_multi_snr_metrics_populated(self):
        """Verify all discrete SNR evaluation metrics are computed."""
        for snr in [5, 10, 15, 20, 50, "inf"]:
            cbf_col = f"cbf_rmse_snr_{snr}"
            att_col = f"att_rmse_snr_{snr}"
            self.assertIn(cbf_col, self.df.columns)
            self.assertIn(att_col, self.df.columns)
            self.assertFalse(self.df[cbf_col].isna().any())
            self.assertFalse(self.df[att_col].isna().any())

    def test_07_manifests_exist(self):
        """Verify data_manifest.csv and subset_manifest.csv exist and describe subsets."""
        d_man = os.path.join(self.exp_dir, "data_manifest.csv")
        s_man = os.path.join(self.exp_dir, "subset_manifest.csv")
        self.assertTrue(os.path.exists(d_man))
        self.assertTrue(os.path.exists(s_man))
        df_d = pd.read_csv(d_man)
        df_s = pd.read_csv(s_man)
        self.assertEqual(len(df_d), 4)
        self.assertEqual(len(df_s), 4)


if __name__ == "__main__":
    unittest.main()
