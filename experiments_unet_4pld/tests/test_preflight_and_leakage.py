"""
Preflight Auditing & Verification Test Suite for ASL U-Net:
Covers Phase 4 requirements:
1. Independence & No Leakage across Train/Val/Test splits.
2. Normalization exclusivity (computed strictly from training valid voxels).
3. Normalization/Denormalization mathematical round-trip consistency.
4. MaskedLoss strictly ignores background voxels.
5. Model input/output tensor shapes.
6. Forward Buxton model & Rician noise calibration.
7. Tiny-set overfit capability test (verifies model capacity & backprop).
8. Checkpoint save/reload bit-for-bit equivalence.
"""

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments_unet_4pld.workspace import get_workspace
from experiments_unet_4pld.model import ASLUNet
from experiments_unet_4pld.dataset import (
    SpatialASLDataset,
    compute_normalization_stats,
    normalize_dataset,
    denormalize_predictions
)
from experiments_unet_4pld.train import MaskedLoss
from experiments_unet_4pld.generate_spatial_data import (
    generate_single_phantom,
    generate_spatial_dataset
)
from data_gen import compute_signals_vec, SIG_REF_SC, SCALE

class TestPreflightAndLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = get_workspace()
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_1_split_independence_and_no_leakage(self):
        """Verify train, val, and test phantoms are generated independently with zero overlap."""
        # Generate 4 phantoms each with seeds 42 (train), 99 (val), 7 (test)
        _, Y_tr, masks_tr, _, _ = generate_spatial_dataset(n_phantoms=4, seed=42)
        _, Y_val, masks_val, _, _ = generate_spatial_dataset(n_phantoms=4, seed=99)
        _, Y_te, masks_te, _, _ = generate_spatial_dataset(n_phantoms=4, seed=7)

        # Check that target parameter maps across splits are distinct
        for i in range(4):
            for j in range(4):
                # CBF comparison
                diff_tr_val = np.max(np.abs(Y_tr[i, 0] - Y_val[j, 0]))
                diff_tr_te = np.max(np.abs(Y_tr[i, 0] - Y_te[j, 0]))
                diff_val_te = np.max(np.abs(Y_val[i, 0] - Y_te[j, 0]))

                self.assertGreater(diff_tr_val, 1.0, "Train and Val phantoms must not be identical")
                self.assertGreater(diff_tr_te, 1.0, "Train and Test phantoms must not be identical")
                self.assertGreater(diff_val_te, 1.0, "Val and Test phantoms must not be identical")

    def test_2_normalization_exclusivity(self):
        """Verify normalization statistics are computed exclusively from training valid brain voxels."""
        N = 5
        X_tr = np.ones((N, 4, 32, 32), dtype=np.float32) * 50.0
        # Background voxels have extreme values to test mask filtering
        X_tr[:, :, :10, :10] = 9999.0

        Y_tr = np.zeros((N, 2, 32, 32), dtype=np.float32)
        Y_tr[:, 0] = 60.0  # CBF
        Y_tr[:, 1] = 1.5   # ATT
        Y_tr[:, :, :10, :10] = 8888.0

        masks_tr = np.ones((N, 1, 32, 32), dtype=np.float32)
        masks_tr[:, :, :10, :10] = 0.0  # background

        stats = compute_normalization_stats(X_tr, Y_tr, masks_tr)

        # Statistics should ignore the 9999.0 and 8888.0 in the background
        self.assertAlmostEqual(stats["cbf_mean"], 60.0, places=4)
        self.assertAlmostEqual(stats["att_mean"], 1.5, places=4)
        for c in range(4):
            self.assertAlmostEqual(stats["X_mean"][c], 50.0, places=4)

    def test_3_normalization_denormalization_roundtrip(self):
        """Verify physical parameter maps accurately round-trip through normalization and denormalization."""
        cbf_gt = np.array([20.0, 45.0, 70.0, 85.0], dtype=np.float32)
        att_gt = np.array([0.8, 1.2, 1.8, 2.5], dtype=np.float32)
        
        stats = {
            "cbf_mean": 50.0,
            "cbf_std": 15.0,
            "att_mean": 1.5,
            "att_std": 0.4
        }
        
        cbf_norm = (cbf_gt - stats["cbf_mean"]) / stats["cbf_std"]
        att_norm = (att_gt - stats["att_mean"]) / stats["att_std"]
        
        pred_norm = np.stack([cbf_norm, att_norm], axis=1)[:, :, np.newaxis, np.newaxis]
        cbf_rec, att_rec = denormalize_predictions(pred_norm, stats)
        
        np.testing.assert_allclose(cbf_rec[:, 0, 0], cbf_gt, atol=1e-5)
        np.testing.assert_allclose(att_rec[:, 0, 0], att_gt, atol=1e-5)

    def test_4_masked_loss_ignores_background(self):
        """Verify MaskedLoss excludes background voxels completely."""
        loss_fn = MaskedLoss(loss_type="l1")
        
        pred = torch.ones(2, 2, 16, 16) * 10.0
        target = torch.ones(2, 2, 16, 16) * 10.0
        mask = torch.zeros(2, 1, 16, 16)
        mask[:, :, 4:12, 4:12] = 1.0  # only center is valid

        # Change prediction in background to huge values
        pred[:, :, :4, :] = 1e6
        pred[:, :, 12:, :] = 1e6

        loss = loss_fn(pred, target, mask)
        self.assertAlmostEqual(loss.item(), 0.0, places=5, msg="Masked loss must ignore background errors")

    def test_5_model_input_output_tensor_shapes(self):
        """Verify ASLUNet forward pass matches expected 4-channel input and 2-channel output."""
        model = ASLUNet(in_channels=4, out_channels=2, features=[16, 32, 64, 128]).to(self.device)
        dummy_in = torch.randn(3, 4, 64, 64, device=self.device)
        out = model(dummy_in)
        self.assertEqual(out.shape, (3, 2, 64, 64), f"Unexpected shape {out.shape}")

    def test_6_forward_physics_and_noise_consistency(self):
        """Verify ASL forward Buxton model and dual-Rician noise formula consistency."""
        cbf = np.array([60.0])
        att = np.array([1.2])
        sig = compute_signals_vec(cbf, att) * SCALE
        self.assertEqual(sig.shape, (1, 4))
        # At PLDs [1.525, 2.025, 2.525, 3.025] > ATT (1.2), signal must be strictly positive
        self.assertTrue(np.all(sig > 0.0))
        
        # Test noise level for SNR = 15
        snr_target = 15.0
        expected_sd = SIG_REF_SC / snr_target
        _, _, _, meta = generate_single_phantom(img_size=32, snr_target=snr_target)
        self.assertAlmostEqual(meta["sd"], expected_sd, places=4)

    def test_7_tiny_set_overfit(self):
        """Verify model capacity and gradient flow by overfitting a tiny 2-sample dataset."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        model = ASLUNet(in_channels=4, out_channels=2, features=[16, 32, 64, 128]).to(self.device)
        criterion = MaskedLoss(loss_type="l1")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        X = torch.randn(2, 4, 32, 32, device=self.device)
        Y = torch.randn(2, 2, 32, 32, device=self.device)
        M = torch.ones(2, 1, 32, 32, device=self.device)

        initial_loss = None
        final_loss = None

        for epoch in range(30):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y, M)
            loss.backward()
            optimizer.step()

            if epoch == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        # Training loss must drop substantially (at least 75% reduction)
        reduction = (initial_loss - final_loss) / initial_loss
        self.assertGreater(reduction, 0.75, f"Loss reduction {reduction:.1%} is insufficient (initial: {initial_loss:.4f}, final: {final_loss:.4f})")

    def test_8_checkpoint_save_reload_equivalence(self):
        """Verify saved checkpoint parameters match reloaded model exactly."""
        model1 = ASLUNet(in_channels=4, out_channels=2, features=[16, 32, 64, 128])
        save_path = os.path.join(self.ws.get_path("tmp"), "test_ckpt_reload.pth")
        
        torch.save({"model_state_dict": model1.state_dict()}, save_path)
        
        model2 = ASLUNet(in_channels=4, out_channels=2, features=[16, 32, 64, 128])
        ckpt = torch.load(save_path, weights_only=False)
        model2.load_state_dict(ckpt["model_state_dict"])

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2), "Model parameters must be bit-for-bit identical after reload")

        if os.path.exists(save_path):
            os.remove(save_path)

if __name__ == "__main__":
    unittest.main()

