import os
import sys
import unittest
import shutil
import tempfile
import json
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments_unet_4pld.workspace import WorkspaceManager, get_disk_free_gb
from experiments_unet_4pld.dataset_cache import (
    create_dataset_manifest,
    validate_cache_manifest,
    compute_dataset_hash
)

class TestStorageWorkspace(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory on current drive for isolated testing
        self.test_root = tempfile.mkdtemp(prefix="test_asl_workspace_")

    def tearDown(self):
        if os.path.exists(self.test_root):
            shutil.rmtree(self.test_root, ignore_errors=True)

    def test_workspace_directory_structure(self):
        """Verifies all expected subdirectories are created and writable."""
        ws = WorkspaceManager(custom_path=self.test_root, min_free_gb=0.1)
        for name in ["data", "checkpoints", "results", "logs", "plots", "tmp", "final_model"]:
            path = ws.get_path(name)
            self.assertTrue(os.path.exists(path), f"Subdirectory {name} does not exist")
            self.assertTrue(os.path.isdir(path))

    def test_environment_variable_redirection(self):
        """Verifies temp and library cache environment variables point to workspace."""
        ws = WorkspaceManager(custom_path=self.test_root, min_free_gb=0.1)
        tmp_dir = ws.get_path("tmp")
        self.assertEqual(os.environ.get("TMPDIR"), tmp_dir)
        self.assertEqual(os.environ.get("TEMP"), tmp_dir)
        self.assertTrue(os.environ.get("TORCH_HOME").startswith(tmp_dir))
        self.assertTrue(os.environ.get("MPLCONFIGDIR").startswith(tmp_dir))

    def test_preflight_storage_check(self):
        """Tests that pre-flight check passes normally and fails when threshold is impossible."""
        ws = WorkspaceManager(custom_path=self.test_root, min_free_gb=0.01)
        status = ws.check_preflight_storage(required_gb=0.01)
        self.assertEqual(status["status"], "PASS")

        # Set an impossible threshold (e.g., 99999 GB) to verify safety halt
        ws_high = WorkspaceManager(custom_path=self.test_root, min_free_gb=999999.0)
        with self.assertRaises(RuntimeError):
            ws_high.check_preflight_storage(required_gb=1.0)

    def test_dataset_manifest_and_cache_validation(self):
        """Verifies deterministic hash and cache validation."""
        plds = [1.525, 2.025, 2.525, 3.025]
        meta1 = create_dataset_manifest("train", 100, 42, 64, plds, [10.0, 15.0])
        meta2 = create_dataset_manifest("train", 100, 42, 64, plds, [10.0, 15.0])
        # Deterministic hash
        self.assertEqual(meta1["hash"], meta2["hash"])

        # Cache file creation
        fake_npz = os.path.join(self.test_root, "test_cache.npz")
        fake_manifest = fake_npz.replace(".npz", ".manifest.json")
        np.savez(fake_npz, X=np.zeros((5, 4, 16, 16)))
        with open(fake_manifest, "w") as f:
            json.dump(meta1, f)

        # Validation succeeds
        self.assertTrue(validate_cache_manifest(fake_npz, meta1))

        # Invalidation when parameters change
        meta_changed = create_dataset_manifest("train", 200, 42, 64, plds, [10.0, 15.0])
        self.assertFalse(validate_cache_manifest(fake_npz, meta_changed))

    def test_storage_summary(self):
        """Verifies storage accounting across workspace categories."""
        ws = WorkspaceManager(custom_path=self.test_root, min_free_gb=0.1)
        # Create dummy file in checkpoints
        ckpt_dir = ws.get_path("checkpoints")
        dummy_file = os.path.join(ckpt_dir, "dummy_model.pth")
        with open(dummy_file, "wb") as f:
            f.write(b"0" * (1024 * 1024)) # 1 MB
        summary = ws.get_storage_summary()
        self.assertGreaterEqual(summary["checkpoints"], 0.9)
        self.assertGreaterEqual(summary["total_mb"], 0.9)

if __name__ == "__main__":
    unittest.main()

