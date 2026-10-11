import os
import sys
import json
import shutil
import hashlib
import numpy as np
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments_unet_4pld.model import ASLUNet

INFERENCE_SCRIPT_CODE = '''import os
import sys
import json
import numpy as np
import torch

# Ensure local directory is in path to load model.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model import ASLUNet

class ASLParameterEstimator:
    """Production inference estimator for 4-PLD ASL MRI CBF and ATT estimation."""
    def __init__(self, model_dir=None, device=None):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = model_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(os.path.join(model_dir, "config.json"), "r") as f:
            self.config = json.load(f)
        with open(os.path.join(model_dir, "normalization_stats.json"), "r") as f:
            self.stats = json.load(f)
        with open(os.path.join(model_dir, "protocol.json"), "r") as f:
            self.protocol = json.load(f)

        self.expected_plds = self.protocol["PLDs_seconds"]
        self.expected_channels = len(self.expected_plds)

        self.model = ASLUNet(
            in_channels=self.config["model"]["in_channels"],
            out_channels=self.config["model"]["out_channels"],
            features=self.config["model"]["features"]
        ).to(self.device)

        ckpt_path = os.path.join(model_dir, "asl_unet_4pld_final.pth")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def predict(self, asl_volume, brain_mask=None):
        """
        Predicts physical CBF (ml/100g/min) and ATT (s) from 4-PLD ASL volume.
        Input:
            asl_volume: np.ndarray [B, 4, H, W] or [4, H, W]
            brain_mask: optional np.ndarray [B, 1, H, W] or [1, H, W]
        Returns:
            cbf_map: np.ndarray [B, H, W] in ml/100g/min
            att_map: np.ndarray [B, H, W] in seconds
        """
        asl_arr = np.asarray(asl_volume, dtype=np.float32)
        if asl_arr.ndim == 3:
            asl_arr = asl_arr[None, ...] # Add batch dim: [1, 4, H, W]
        if asl_arr.shape[1] != self.expected_channels:
            raise ValueError(
                f"Expected {self.expected_channels} input channels matching PLDs {self.expected_plds}, "
                f"got shape {asl_arr.shape}"
            )

        # Normalize with training stats
        norm_arr = np.copy(asl_arr)
        x_mean = np.array(self.stats["X_mean"], dtype=np.float32)
        x_std = np.array(self.stats["X_std"], dtype=np.float32)
        for c in range(self.expected_channels):
            norm_arr[:, c] = (norm_arr[:, c] - x_mean[c]) / x_std[c]

        with torch.no_grad():
            x_t = torch.tensor(norm_arr, dtype=torch.float32, device=self.device)
            preds_norm = self.model(x_t).cpu().numpy()

        cbf_pred = preds_norm[:, 0] * self.stats["cbf_std"] + self.stats["cbf_mean"]
        att_pred = preds_norm[:, 1] * self.stats["att_std"] + self.stats["att_mean"]

        cbf_pred = np.clip(cbf_pred, self.protocol["CBF_bounds"][0], self.protocol["CBF_bounds"][1])
        att_pred = np.clip(att_pred, self.protocol["ATT_bounds"][0], self.protocol["ATT_bounds"][1])

        if brain_mask is not None:
            mask_arr = np.asarray(brain_mask) > 0.5
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[None, ...]
            cbf_pred = np.where(mask_arr[:, 0], cbf_pred, 0.0)
            att_pred = np.where(mask_arr[:, 0], att_pred, 0.0)

        return cbf_pred, att_pred

if __name__ == "__main__":
    print("Testing ASLParameterEstimator self-inference...")
    estimator = ASLParameterEstimator()
    dummy_input = np.random.uniform(100, 1000, (2, 4, 64, 64)).astype(np.float32)
    cbf, att = estimator.predict(dummy_input)
    print(f"Prediction successful! Output shapes: CBF={cbf.shape}, ATT={att.shape}")
    print(f"CBF range: [{cbf.min():.2f}, {cbf.max():.2f}] ml/100g/min | ATT range: [{att.min():.3f}, {att.max():.3f}] s")
'''

def package_final_four_pld_model(
    selected_ckpt_path,
    config,
    stats,
    history_csv_path,
    test_metrics,
    output_dir
):
    """Packages all required final model artifacts and scripts for independent deployment."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Copy model architecture code
    source_model_py = os.path.join(os.path.dirname(__file__), "model.py")
    target_model_py = os.path.join(output_dir, "model.py")
    shutil.copy2(source_model_py, target_model_py)

    # 2. Copy model checkpoint
    target_ckpt = os.path.join(output_dir, "asl_unet_4pld_final.pth")
    shutil.copy2(selected_ckpt_path, target_ckpt)

    # Compute checkpoint hash
    with open(target_ckpt, "rb") as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()

    # 3. Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 4. Save normalization stats
    stats_serializable = {
        "X_mean": stats["X_mean"].tolist() if isinstance(stats["X_mean"], np.ndarray) else stats["X_mean"],
        "X_std": stats["X_std"].tolist() if isinstance(stats["X_std"], np.ndarray) else stats["X_std"],
        "cbf_mean": float(stats["cbf_mean"]),
        "cbf_std": float(stats["cbf_std"]),
        "att_mean": float(stats["att_mean"]),
        "att_std": float(stats["att_std"])
    }
    with open(os.path.join(output_dir, "normalization_stats.json"), "w") as f:
        json.dump(stats_serializable, f, indent=2)

    # 5. Save acquisition protocol metadata
    protocol = {
        "PLDs_seconds": config["physics"]["PLDs"],
        "tau_seconds": config["physics"]["tau"],
        "CBF_bounds": [config["physics"]["CBF_MIN"], config["physics"]["CBF_MAX"]],
        "ATT_bounds": [config["physics"]["ATT_MIN"], config["physics"]["ATT_MAX"]],
        "CBF_unit": "ml/100g/min",
        "ATT_unit": "seconds",
        "SCALE": config["physics"]["SCALE"]
    }
    with open(os.path.join(output_dir, "protocol.json"), "w") as f:
        json.dump(protocol, f, indent=2)

    # 6. Copy training history
    if history_csv_path and os.path.exists(history_csv_path):
        shutil.copy2(history_csv_path, os.path.join(output_dir, "training_history.csv"))

    # 7. Save final test metrics
    with open(os.path.join(output_dir, "final_test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # 8. Write inference script
    inf_script_path = os.path.join(output_dir, "inference.py")
    with open(inf_script_path, "w") as f:
        f.write(INFERENCE_SCRIPT_CODE)

    # 9. Save package manifest
    manifest = {
        "model_name": "Four-PLD Spatial ASL U-Net",
        "version": "1.0.0",
        "checkpoint_sha256": ckpt_hash,
        "input_channels": 4,
        "output_channels": 2,
        "selection_policy": "Strict validation-loss minimization across training epochs (zero test leakage)",
        "protocol": protocol,
        "test_performance": {
            "CBF_RMSE": test_metrics.get("CBF_RMSE"),
            "ATT_RMSE": test_metrics.get("ATT_RMSE"),
            "CBF_R2": test_metrics.get("CBF_R2"),
            "ATT_R2": test_metrics.get("ATT_R2"),
        }
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Model Packaging] Successfully created self-contained 4-PLD package in: {output_dir}")
    return output_dir

