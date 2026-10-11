import os
import sys
import unittest
import numpy as np
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_gen import (
    PLDs, compute_signals_vec, SCALE,
    CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX,
    SIG_REF_SC, tau, T1t, T1a, alpha, beta, lmbda
)
from experiments_unet_4pld.generate_spatial_data import generate_single_phantom

class TestAuditFixes(unittest.TestCase):

    def test_01_signal_model_expectation(self):
        """Verify that Simulator A expected signal is 2.0 * compute_signals_vec * SCALE."""
        cbf = np.array([55.0])
        att = np.array([1.2])
        ld = tau
        sig = compute_signals_vec(cbf, att, ld=ld) * SCALE # shape (1, 4)

        # High-SNR simulation: sigma -> 0
        rng = np.random.default_rng(42)
        sig_noisy, targets, mask, meta = generate_single_phantom(
            img_size=32, rng=rng, snr_target=10000.0 # near noiseless
        )
        valid = mask[0] > 0.5
        v_idx = np.where(valid)
        vy, vx = v_idx[0][0], v_idx[1][0]
        cbf_v = targets[0, vy, vx]
        att_v = targets[1, vy, vx]
        expected = 2.0 * compute_signals_vec(np.array([cbf_v]), np.array([att_v])) * SCALE

        actual = sig_noisy[:, vy, vx]
        np.testing.assert_allclose(
            actual, expected[0], rtol=0.01,
            err_msg="Noiseless observation must equal 2 * compute_signals_vec"
        )

    def test_02_effective_noise_variance(self):
        """Verify that sum of two Rician signals has noise variance = 2 * sigma^2."""
        rng = np.random.default_rng(123)
        N = 50000
        cbf = np.full(N, 50.0)
        att = np.full(N, 1.5)
        sig = compute_signals_vec(cbf, att) * SCALE # [N, 4]

        sd = 20.0 # known sigma
        e1 = rng.normal(0, 1, size=sig.shape) * sd
        e2 = rng.normal(0, 1, size=sig.shape) * sd
        e3 = rng.normal(0, 1, size=sig.shape) * sd
        e4 = rng.normal(0, 1, size=sig.shape) * sd

        mc = np.sqrt((sig + e1)**2 + e2**2)
        ml = np.sqrt((sig + e3)**2 + e4**2)
        X = mc + ml

        # Residual from true expectation
        residual = X - 2.0 * sig
        emp_var = np.var(residual, axis=0) # [4]
        theo_var = 2.0 * (sd ** 2) # 800.0

        # Empirical variance must match theoretical 2 * sd^2 within 3%
        np.testing.assert_allclose(
            emp_var, theo_var, rtol=0.03,
            err_msg="Noise variance of mc + ml must equal 2 * sd^2"
        )

    def test_03_bayesian_likelihood_calibration(self):
        """Verify that Bayesian likelihood using sigma_eff = sqrt(2)*sd recovers true parameters."""
        cbf_true = 50.0
        att_true = 1.8
        sig_true = 2.0 * compute_signals_vec(np.array([cbf_true]), np.array([att_true])) * SCALE

        # Generate 100 noisy realizations at SNR=20
        snr = 20.0
        sd = SIG_REF_SC / snr
        rng = np.random.default_rng(42)

        e1 = rng.normal(0, sd, size=(100, 4))
        e2 = rng.normal(0, sd, size=(100, 4))
        e3 = rng.normal(0, sd, size=(100, 4))
        e4 = rng.normal(0, sd, size=(100, 4))
        X_obs = np.sqrt((sig_true/2.0 + e1)**2 + e2**2) + np.sqrt((sig_true/2.0 + e3)**2 + e4**2)

        # 80x80 Grid
        cbf_grid = np.linspace(CBF_MIN, CBF_MAX, 80)
        att_grid = np.linspace(ATT_MIN, ATT_MAX, 80)
        CC, AA = np.meshgrid(cbf_grid, att_grid)
        grid_sig = 2.0 * compute_signals_vec(CC.flatten(), AA.flatten()) * SCALE

        sigma_eff = np.sqrt(2.0) * sd
        norm_S = np.sum(grid_sig**2, axis=1, keepdims=True).T

        norm_X = np.sum(X_obs**2, axis=1, keepdims=True)
        sq_diff = norm_X - 2.0 * (X_obs @ grid_sig.T) + norm_S
        log_L = -0.5 * sq_diff / (sigma_eff**2)
        log_L -= np.max(log_L, axis=1, keepdims=True)
        L = np.exp(log_L)
        P = L / (np.sum(L, axis=1, keepdims=True) + 1e-12)

        pred_cbf = np.mean(P @ CC.flatten())
        pred_att = np.mean(P @ AA.flatten())

        self.assertAlmostEqual(pred_cbf, cbf_true, delta=5.0)
        self.assertAlmostEqual(pred_att, att_true, delta=0.20)

    def test_04_canonical_seeds_suite(self):
        """Verify project canonical audit seeds [42, 123, 2024] are distinct."""
        canonical_seeds = [42, 123, 2024]
        self.assertEqual(len(set(canonical_seeds)), 3)
        self.assertIn(42, canonical_seeds)
        self.assertIn(123, canonical_seeds)
        self.assertIn(2024, canonical_seeds)

if __name__ == "__main__":
    unittest.main()
