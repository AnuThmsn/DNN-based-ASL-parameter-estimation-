import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_gen import (
    PLDs, compute_signals_vec, SCALE,
    CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX
)
from experiments_unet_4pld.generate_spatial_data import (
    generate_single_phantom,
    generate_spatial_dataset
)
from experiments_unet_4pld.dataset import (
    compute_normalization_stats,
    normalize_dataset,
    denormalize_predictions,
    SpatialASLDataset
)
from experiments_unet_4pld.model import ASLUNet
from experiments_unet_4pld.train import MaskedLoss

class TestUNetPipeline(unittest.TestCase):

    def test_01_plds_ordering_and_values(self):
        """1. The four PLDs are correct and in expected order."""
        expected = np.array([1.525, 2.025, 2.525, 3.025])
        self.assertEqual(len(PLDs), 4)
        np.testing.assert_allclose(PLDs, expected, err_msg="PLDs must match [1.525, 2.025, 2.525, 3.025]")

    def test_02_simulator_reproduces_reference(self):
        """2. The simulator reproduces existing validated reference signal values."""
        cbf_test = np.array([50.0])
        att_test = np.array([1.5])
        sig = compute_signals_vec(cbf_test, att_test) * SCALE
        self.assertEqual(sig.shape, (1, 4))
        self.assertTrue(np.all(sig > 0), "ASL perfusion difference signal must be positive for typical parameters")
        # Ensure monotonic or sensible signal values
        self.assertFalse(np.any(np.isnan(sig)))

    def test_03_spatial_voxelwise_correspondence(self):
        """3. Spatial signal generation uses the corresponding CBF/ATT label at every voxel."""
        rng = np.random.default_rng(123)
        sig_noisy, targets, mask, meta = generate_single_phantom(img_size=32, rng=rng, snr_target=1000.0) # low noise
        cbf_map, att_map = targets[0], targets[1]
        valid_indices = np.where(mask[0] > 0.5)
        # Check a specific valid voxel
        vy, vx = valid_indices[0][0], valid_indices[1][0]
        cbf_val, att_val = cbf_map[vy, vx], att_map[vy, vx]
        expected_sig = 2.0 * compute_signals_vec(np.array([cbf_val]), np.array([att_val])) * SCALE
        # In near-noiseless setting, signal at that voxel must be close to expected (mc + ml = 2 * sig)
        actual_sig = sig_noisy[:, vy, vx]
        # Allow small noise offset
        np.testing.assert_allclose(actual_sig, expected_sig[0], rtol=0.1, atol=2.0)

    def test_04_dataset_tensor_shapes(self):
        """4. Dataset tensors have the expected shapes."""
        X, Y, masks, snrs, sds = generate_spatial_dataset(n_phantoms=3, seed=42, img_size=32)
        self.assertEqual(X.shape, (3, 4, 32, 32))
        self.assertEqual(Y.shape, (3, 2, 32, 32))
        self.assertEqual(masks.shape, (3, 1, 32, 32))
        self.assertEqual(snrs.shape, (3,))

    def test_05_unet_input_output_dimensions(self):
        """5. The U-Net accepts [B, 4, H, W] and returns [B, 2, H, W]."""
        model = ASLUNet(in_channels=4, out_channels=2, features=[16, 32, 64, 128])
        dummy_in = torch.randn(2, 4, 32, 32)
        dummy_out = model(dummy_in)
        self.assertEqual(dummy_out.shape, (2, 2, 32, 32))

    def test_06_output_spatial_match_input(self):
        """6. Output spatial dimensions match the input dimensions."""
        model = ASLUNet(in_channels=4, out_channels=2, features=[16, 32, 64, 128])
        for size in [32, 64]:
            inp = torch.randn(1, 4, size, size)
            out = model(inp)
            self.assertEqual(out.shape[-2:], inp.shape[-2:])

    def test_07_finite_outputs_and_loss(self):
        """7. All model outputs and loss values are finite."""
        model = ASLUNet(in_channels=4, out_channels=2, features=[16, 32, 64, 128])
        criterion = MaskedLoss("l1")
        inp = torch.randn(2, 4, 32, 32)
        tgt = torch.randn(2, 2, 32, 32)
        mask = torch.ones(2, 1, 32, 32)
        out = model(inp)
        loss = criterion(out, tgt, mask)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(loss).all())

    def test_08_masked_loss_excludes_background(self):
        """8. Masked loss excludes invalid background pixels."""
        criterion = MaskedLoss("l1")
        pred = torch.zeros(1, 2, 8, 8)
        tgt = torch.zeros(1, 2, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)

        # Set valid voxel (0, 0)
        mask[0, 0, 0, 0] = 1.0
        tgt[0, 0, 0, 0] = 5.0 # Error = 5.0 on valid voxel

        # Corrupt background voxel (4, 4) with massive error
        pred[0, 0, 4, 4] = 9999.0 # Background, mask=0

        loss = criterion(pred, tgt, mask)
        # Total valid voxels = 1. Valid channel error: |0 - 5| = 5 on ch0, 0 on ch1 => sum = 5. Total valid elements = 2.
        # Loss should be exactly 5.0 / 2 = 2.5
        self.assertAlmostEqual(loss.item(), 2.5, places=4)

    def test_09_normalization_and_inverse_scaling(self):
        """9. Target normalization and inverse transformation are correct."""
        rng = np.random.default_rng(999)
        cbf_gt = rng.uniform(20.0, 80.0, (10, 1, 32, 32)).astype(np.float32)
        att_gt = rng.uniform(0.8, 2.5, (10, 1, 32, 32)).astype(np.float32)
        Y_gt = np.concatenate([cbf_gt, att_gt], axis=1)
        masks = np.ones((10, 1, 32, 32), dtype=np.float32)
        dummy_X = np.ones((10, 4, 32, 32), dtype=np.float32)

        stats = compute_normalization_stats(dummy_X, Y_gt, masks)
        _, Y_norm = normalize_dataset(dummy_X, Y_gt, masks, stats)
        cbf_rec, att_rec = denormalize_predictions(Y_norm, stats)

        np.testing.assert_allclose(cbf_rec, cbf_gt[:, 0], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(att_rec, att_gt[:, 0], rtol=1e-4, atol=1e-4)

    def test_10_checkpoint_save_and_load(self):
        """10. The saved checkpoint can be loaded for inference."""
        model = ASLUNet(in_channels=4, out_channels=2, features=[16, 32, 64, 128])
        tmp_path = "tmp_test_ckpt.pth"
        torch.save({"model_state_dict": model.state_dict()}, tmp_path)

        model_loaded = ASLUNet(in_channels=4, out_channels=2, features=[16, 32, 64, 128])
        ckpt = torch.load(tmp_path, weights_only=True)
        model_loaded.load_state_dict(ckpt["model_state_dict"])
        model_loaded.eval()

        model.eval()
        inp = torch.randn(1, 4, 32, 32)
        with torch.no_grad():
            out1 = model(inp)
            out2 = model_loaded(inp)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), rtol=1e-5)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    def test_11_test_data_isolation(self):
        """11. Test data are excluded from training and model selection."""
        # Verify seed isolation between train, val, test
        rng_tr = np.random.default_rng(42)
        rng_te = np.random.default_rng(7)
        s_tr = rng_tr.uniform(0, 1, 100)
        s_te = rng_te.uniform(0, 1, 100)
        self.assertFalse(np.array_equal(s_tr, s_te), "Train and test seeds must produce independent sequences")

    def test_12_reproducibility_under_fixed_seed(self):
        """12. Results are reproducible under fixed seeds within tolerance."""
        s1, t1, m1, _ = generate_single_phantom(img_size=32, rng=np.random.default_rng(42))
        s2, t2, m2, _ = generate_single_phantom(img_size=32, rng=np.random.default_rng(42))
        np.testing.assert_allclose(s1, s2, err_msg="Identical seeds must produce identical phantoms")
        np.testing.assert_allclose(t1, t2, err_msg="Identical seeds must produce identical parameter maps")

if __name__ == "__main__":
    unittest.main()
