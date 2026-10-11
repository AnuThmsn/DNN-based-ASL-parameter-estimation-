"""
Unit Tests for SNR-Wise Metrics: RMSE, MAE, and Lin's Concordance Correlation Coefficient (CCC).
Covers:
1. Perfect predictions: RMSE = 0, MAE = 0, CCC = 1 for nonconstant ground truth.
2. A known prediction example with manually verifiable RMSE and MAE.
3. A known CCC example checked against an analytical population-moment calculation.
4. Independent calculations for CBF and ATT.
5. Constant arrays and zero-denominator handling.
6. Non-finite predictions handling.
7. Identical test samples and noise realizations for both models.
8. Correct handling of masks and invalid voxels.
9. CSV completeness: both models x all six SNR conditions.
10. Correct insertion of historical Bayesian/NLS values at SNR 10 and 15 only.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments_unet_4pld.run_snr_benchmark import (
    calc_lin_ccc,
    calc_metrics,
    generate_snr_dataset,
    HISTORICAL_BASELINES
)

class TestMetricsAndCCC(unittest.TestCase):
    def test_1_perfect_predictions(self):
        """Perfect predictions: RMSE = 0, MAE = 0, CCC = 1 for nonconstant ground truth."""
        y = np.linspace(10.0, 90.0, 100, dtype=np.float64)
        y_hat = np.copy(y)

        met = calc_metrics(y, y_hat)
        ccc = calc_lin_ccc(y, y_hat)

        assert np.isclose(met["RMSE"], 0.0), f"Expected RMSE 0, got {met['RMSE']}"
        assert np.isclose(met["MAE"], 0.0), f"Expected MAE 0, got {met['MAE']}"
        assert np.isclose(ccc, 1.0), f"Expected CCC 1.0, got {ccc}"

    def test_2_known_example_rmse_mae(self):
        """A known prediction example with manually verifiable RMSE and MAE."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        y_hat = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64)

        met = calc_metrics(y, y_hat)
        assert np.isclose(met["RMSE"], 0.5), f"Expected RMSE 0.5, got {met['RMSE']}"
        assert np.isclose(met["MAE"], 0.5), f"Expected MAE 0.5, got {met['MAE']}"
        assert np.isclose(met["Bias"], 0.5), f"Expected Bias 0.5, got {met['Bias']}"

    def test_3_known_example_ccc(self):
        """
        A known CCC example checked against analytical population-moment calculation.
        For y = [1, 2, 3, 4, 5], y_hat = [1.5, 2.5, 3.5, 4.5, 5.5]:
        mu_y = 3.0, mu_y_hat = 3.5
        var_y = 2.0, var_y_hat = 2.0, cov = 2.0
        denom = 2.0 + 2.0 + (3.0 - 3.5)^2 = 4.25
        CCC = (2 * 2.0) / 4.25 = 4.0 / 4.25 = 16 / 17 = 0.9411764705882353
        """
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        y_hat = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64)

        expected_ccc = 16.0 / 17.0
        calculated_ccc = calc_lin_ccc(y, y_hat)
        assert np.isclose(calculated_ccc, expected_ccc, atol=1e-6), (
            f"Expected CCC {expected_ccc:.8f}, got {calculated_ccc:.8f}"
        )

    def test_4_different_predictions_cbf_att(self):
        """Different predictions for CBF (scale 0-100) and ATT (scale 0.5-3.0)."""
        cbf_true = np.array([20.0, 50.0, 70.0], dtype=np.float64)
        cbf_pred = np.array([22.0, 48.0, 72.0], dtype=np.float64)

        att_true = np.array([1.0, 1.5, 2.0], dtype=np.float64)
        att_pred = np.array([1.1, 1.4, 2.1], dtype=np.float64)

        cbf_met = calc_metrics(cbf_true, cbf_pred)
        att_met = calc_metrics(att_true, att_pred)

        assert cbf_met["RMSE"] != att_met["RMSE"]
        assert cbf_met["RMSE"] > 1.0
        assert att_met["RMSE"] < 0.5

    def test_5_constant_arrays_zero_denominator(self):
        """Constant arrays and zero-denominator handling."""
        # Identical constant arrays: zero variance, identical mean -> agreement is perfect (1.0)
        y_const = np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float64)
        y_const_hat = np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float64)
        assert np.isclose(calc_lin_ccc(y_const, y_const_hat), 1.0)

        # Different constant arrays: zero variance, different mean -> 0.0
        y_const_diff = np.array([60.0, 60.0, 60.0, 60.0], dtype=np.float64)
        assert np.isclose(calc_lin_ccc(y_const, y_const_diff), 0.0)

    def test_6_non_finite_predictions(self):
        """Non-finite predictions handling (NaN or Inf raises ValueError)."""
        y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        y_nan = np.array([1.0, np.nan, 3.0], dtype=np.float64)
        y_inf = np.array([1.0, np.inf, 3.0], dtype=np.float64)

        with self.assertRaises(ValueError):
            calc_lin_ccc(y, y_nan)

        with self.assertRaises(ValueError):
            calc_lin_ccc(y, y_inf)


    def test_7_identical_samples_and_noise(self):
        """Identical test samples and noise realizations for paired comparison."""
        Y_dummy = np.ones((2, 2, 8, 8), dtype=np.float32) * 50.0
        Y_dummy[:, 1] = 1.5
        masks_dummy = np.ones((2, 1, 8, 8), dtype=np.float32)

        X1, sd1 = generate_snr_dataset(Y_dummy, masks_dummy, snr=15.0, seed=7000)
        X2, sd2 = generate_snr_dataset(Y_dummy, masks_dummy, snr=15.0, seed=7000)

        assert np.array_equal(X1, X2), "Same seed must produce identical observation tensors"
        assert sd1 == sd2

    def test_8_masks_and_invalid_voxels(self):
        """Correct handling of masks: outside voxels are zeroed in observations."""
        Y_dummy = np.ones((1, 2, 8, 8), dtype=np.float32) * 50.0
        masks_dummy = np.zeros((1, 1, 8, 8), dtype=np.float32)
        masks_dummy[0, 0, 2:6, 2:6] = 1.0  # Center valid

        X, _ = generate_snr_dataset(Y_dummy, masks_dummy, snr=20.0, seed=7000)
        # Background voxels must be zero
        assert np.all(X[0, :, :2, :] == 0.0)
        assert np.all(X[0, :, 6:, :] == 0.0)

    def test_9_csv_completeness(self):
        """CSV completeness: both models x all six SNR conditions (12 rows)."""
        csv_path = r"D:\ASL_WORKSPACE\experiments\snr_benchmark_4pld\metrics\snr_wise_unet_results.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            assert len(df) == 12, f"Expected 12 rows (2 models x 6 SNRs), got {len(df)}"
            models = set(df["Model"].unique())
            assert "Original 4-PLD U-Net" in models
            assert "Improved 4-PLD U-Net v2" in models
            snrs = set(str(s).replace(".0", "").lower() for s in df["SNR"].unique())
            assert {"inf", "50", "20", "15", "10", "5"}.issubset(snrs)

    def test_10_historical_insertion_snr10_snr15(self):
        """Historical Bayesian and NLS values must match PDF exactly at SNR 10 & 15 only."""
        csv_comb = r"D:\ASL_WORKSPACE\experiments\snr_benchmark_4pld\metrics\combined_snr10_snr15_comparison.csv"
        if os.path.exists(csv_comb):
            df = pd.read_csv(csv_comb)
            # Check Bayesian at SNR 10
            b10 = df[(df["Model"] == "Bayesian") & (df["SNR"].astype(str) == "10")].iloc[0]
            assert np.isclose(b10["CBF_RMSE"], 5.0300)
            assert np.isclose(b10["CBF_MAE"], 3.9680)
            assert np.isclose(b10["CBF_CCC"], 0.9606)
            assert np.isclose(b10["ATT_RMSE"], 0.0826)
            assert np.isclose(b10["ATT_MAE"], 0.0683)
            assert np.isclose(b10["ATT_CCC"], 0.9914)

            # Check NLS at SNR 10
            n10 = df[(df["Model"] == "NLS") & (df["SNR"].astype(str) == "10")].iloc[0]
            assert np.isclose(n10["CBF_RMSE"], 2.9820)
            assert np.isclose(n10["CBF_MAE"], 2.0480)
            assert np.isclose(n10["CBF_CCC"], 0.9913)
            assert np.isclose(n10["ATT_RMSE"], 0.5062)
            assert np.isclose(n10["ATT_MAE"], 0.3411)
            assert np.isclose(n10["ATT_CCC"], 0.6770)

            # Check Bayesian at SNR 15
            b15 = df[(df["Model"] == "Bayesian") & (df["SNR"].astype(str) == "15")].iloc[0]
            assert np.isclose(b15["CBF_RMSE"], 3.7590)
            assert np.isclose(b15["CBF_MAE"], 2.8490)
            assert np.isclose(b15["CBF_CCC"], 0.9671)
            assert np.isclose(b15["ATT_RMSE"], 0.0584)
            assert np.isclose(b15["ATT_MAE"], 0.0472)
            assert np.isclose(b15["ATT_CCC"], 0.9944)

            # Check NLS at SNR 15
            n15 = df[(df["Model"] == "NLS") & (df["SNR"].astype(str) == "15")].iloc[0]
            assert np.isclose(n15["CBF_RMSE"], 2.0620)
            assert np.isclose(n15["CBF_MAE"], 1.4140)
            assert np.isclose(n15["CBF_CCC"], 0.9958)
            assert np.isclose(n15["ATT_RMSE"], 0.4951)
            assert np.isclose(n15["ATT_MAE"], 0.3259)
            assert np.isclose(n15["ATT_CCC"], 0.6899)


if __name__ == "__main__":
    unittest.main()
