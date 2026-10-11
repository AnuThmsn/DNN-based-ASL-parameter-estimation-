import os
import sys
import unittest
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_gen import (
    compute_signals_vec,
    PLDs, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX,
    SIG_REF_SC, tau, alpha, beta, T1t, T1a, lmbda
)

class TestForwardModelAndNoise(unittest.TestCase):
    def test_buxton_signal_properties(self):
        """Verifies fundamental physical properties of Buxton kinetic equations."""
        cbfs = np.array([0.0, 50.0, 100.0], dtype=np.float64)
        atts = np.array([1.0, 1.8, 2.5], dtype=np.float64)

        sig = compute_signals_vec(cbfs, atts, ld=tau) # [3, 4]
        # 1. Zero CBF must yield zero signal
        np.testing.assert_allclose(sig[0], 0.0, atol=1e-12)
        # 2. Perfusion signals must be non-negative
        self.assertTrue(np.all(sig >= 0.0))
        # 3. Signals must scale linearly with CBF
        sig_half = compute_signals_vec(np.array([25.0]), np.array([1.8]), ld=tau)
        sig_full = compute_signals_vec(np.array([50.0]), np.array([1.8]), ld=tau)
        np.testing.assert_allclose(sig_half * 2.0, sig_full, rtol=1e-5)

    def test_sum_of_ricians_expectation_and_variance(self):
        """
        Simulator A generates X = mc + ml from two Ricians.
        Verifies E[X] ≈ 2 * sig * SCALE and Var(X) ≈ 2 * sigma^2.
        """
        rng = np.random.default_rng(2024)
        cbf = np.array([60.0])
        att = np.array([1.2])
        sig = compute_signals_vec(cbf, att, ld=tau)[0] * SCALE # 4 PLD values

        snr = 15.0
        sd = SIG_REF_SC / snr
        n_samples = 150_000

        # Simulate n_samples observations
        e1 = rng.normal(0, sd, size=(n_samples, len(sig)))
        e2 = rng.normal(0, sd, size=(n_samples, len(sig)))
        e3 = rng.normal(0, sd, size=(n_samples, len(sig)))
        e4 = rng.normal(0, sd, size=(n_samples, len(sig)))

        mc = np.sqrt((sig + e1)**2 + e2**2)
        ml = np.sqrt((sig + e3)**2 + e4**2)
        X = mc + ml

        # Check expectation matches 2.0 * sig
        mean_X = np.mean(X, axis=0)
        expected_signal = 2.0 * sig
        np.testing.assert_allclose(mean_X, expected_signal, rtol=0.01)

        # Check variance matches 2.0 * sd^2
        var_X = np.var(X, axis=0)
        expected_var = 2.0 * (sd ** 2)
        np.testing.assert_allclose(var_X, expected_var, rtol=0.05)

    def test_gaussian_likelihood_approximation_validity(self):
        """
        Evaluates normality of sum-of-Ricians noise across SNR 5 to 50.
        By Central Limit Theorem, the sum of two independent Rician variables
        converges toward Gaussian even at moderate SNR.
        """
        rng = np.random.default_rng(42)
        sig_val = 500.0 # arbitrary typical signal level
        for snr in [5.0, 10.0, 15.0, 50.0]:
            sd = SIG_REF_SC / snr
            e1 = rng.normal(0, sd, 50_000)
            e2 = rng.normal(0, sd, 50_000)
            e3 = rng.normal(0, sd, 50_000)
            e4 = rng.normal(0, sd, 50_000)

            X = np.sqrt((sig_val + e1)**2 + e2**2) + np.sqrt((sig_val + e3)**2 + e4**2)
            residuals = X - np.mean(X)
            # Skewness should be close to 0
            skewness = np.mean((residuals / np.std(residuals))**3)
            # Excess kurtosis should be close to 0
            kurtosis = np.mean((residuals / np.std(residuals))**4) - 3.0

            # Skewness within +/- 0.25 at SNR >= 10
            if snr >= 10.0:
                self.assertLess(abs(skewness), 0.25)
                self.assertLess(abs(kurtosis), 0.35)

if __name__ == "__main__":
    unittest.main()

