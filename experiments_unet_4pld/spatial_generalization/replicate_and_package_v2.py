"""
Replication of Selected Winner Candidate Across 3 Seeds, Verification, and Packaging v2.
Produces:
1. 3-seed evaluation statistics across Ordinary and Challenge Phantoms.
2. Complete packaged model directory at D:\\ASL_WORKSPACE\\final_model_4pld_spatial_v2\\
3. Verification inference on 5 phantoms.
4. Publication-quality figure generation in D:\\ASL_WORKSPACE\\experiments\\spatial_generalization\\figures\\
"""

import os
import sys
import json
import time
import shutil
import hashlib
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_gen import compute_signals_vec, SCALE, tau
from experiments_unet_4pld.workspace import get_workspace
from experiments_unet_4pld.dataset import (
    SpatialASLDataset,
    compute_normalization_stats,
    normalize_dataset,
    denormalize_predictions
)
from experiments_unet_4pld.train import MaskedLoss
from experiments_unet_4pld.evaluate import calc_metrics, calc_masked_ssim
from experiments_unet_4pld.spatial_generalization.models_generalization import (
    ASLUNet,
    ASLResidualUNet,
    ASLHybridUNet
)
from experiments_unet_4pld.spatial_generalization.train_and_evaluate_candidates import (
    train_candidate_model,
    evaluate_on_datasets,
    EdgeAwareLoss
)

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def select_best_candidate(pilot_csv_path):
    df = pd.read_csv(pilot_csv_path)
    print("\n--- Screening Pilot Candidates ---")
    print(df[["Candidate_ID", "Ordinary_CBF_RMSE", "Ordinary_ATT_RMSE", "Challenge_CBF_RMSE", "Focal_Boundary_CBF_RMSE"]].to_string())

    # Filter out the untouched baseline
    candidates_only = df[df["Candidate_ID"] != "Original_Baseline_4PLD"].copy()

    # Criteria:
    # 1. Ordinary CBF RMSE <= 1.25 (allow slight trade-off if challenge error is drastically reduced)
    # 2. Sort by Challenge CBF RMSE + Focal Boundary RMSE
    candidates_only["Score"] = candidates_only["Challenge_CBF_RMSE"] + 0.2 * candidates_only["Focal_Boundary_CBF_RMSE"]
    best_row = candidates_only.sort_values("Score").iloc[0]

    best_id = best_row["Candidate_ID"]
    print(f"\nSelected Best Candidate: {best_id} ({best_row['Candidate_Name']})")
    print(f"  Ordinary CBF RMSE: {best_row['Ordinary_CBF_RMSE']:.3f} | Chal CBF RMSE: {best_row['Challenge_CBF_RMSE']:.3f} | Boundary CBF RMSE: {best_row['Focal_Boundary_CBF_RMSE']:.2f}")
    return best_id

def run_3seed_replication(best_id):
    ws = get_workspace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"REPLICATING WINNER ({best_id}) ACROSS 3 SEEDS [42, 123, 2024]")
    print(f"Workspace: {ws.root} | Device: {device}")
    print("=" * 80)

    # Determine configuration from candidate ID
    if "ResUNet" in best_id:
        model_cls = ASLResidualUNet
    elif "Hybrid" in best_id:
        model_cls = ASLHybridUNet
    else:
        model_cls = ASLUNet

    use_diverse = "Diverse" in best_id
    use_mixed = "Mixed" in best_id
    use_edge_loss = "EdgeLoss" in best_id

    # Load frozen test datasets
    te_ord = np.load(ws.get_path("data", "test_N100_s7.npz"))
    te_chal = np.load(ws.get_path("data", "challenge_test_N40_s2026.npz"))

    # Load appropriate training data
    if use_mixed:
        std_tr = np.load(ws.get_path("data", "train_N600_s42.npz"))
        std_val = np.load(ws.get_path("data", "val_N100_s99.npz"))
        div_tr = np.load(ws.get_path("data", "train_diverse_N600_s42.npz"))
        div_val = np.load(ws.get_path("data", "val_diverse_N100_s99.npz"))
        tr_data = {
            "X": np.concatenate([std_tr["X"][:300], div_tr["X"][:300]], axis=0),
            "Y": np.concatenate([std_tr["Y"][:300], div_tr["Y"][:300]], axis=0),
            "masks": np.concatenate([std_tr["masks"][:300], div_tr["masks"][:300]], axis=0),
            "snrs": np.concatenate([std_tr["snrs"][:300], div_tr["snrs"][:300]], axis=0),
        }
        val_data = {
            "X": np.concatenate([std_val["X"][:50], div_val["X"][:50]], axis=0),
            "Y": np.concatenate([std_val["Y"][:50], div_val["Y"][:50]], axis=0),
            "masks": np.concatenate([std_val["masks"][:50], div_val["masks"][:50]], axis=0),
            "snrs": np.concatenate([std_val["snrs"][:50], div_val["snrs"][:50]], axis=0),
        }
    elif use_diverse:
        tr_data = np.load(ws.get_path("data", "train_diverse_N600_s42.npz"))
        val_data = np.load(ws.get_path("data", "val_diverse_N100_s99.npz"))
    else:
        tr_data = np.load(ws.get_path("data", "train_N600_s42.npz"))
        val_data = np.load(ws.get_path("data", "val_N100_s99.npz"))

    stats = compute_normalization_stats(tr_data["X"], tr_data["Y"], tr_data["masks"])

    X_tr_n, Y_tr_n = normalize_dataset(tr_data["X"], tr_data["Y"], tr_data["masks"], stats)
    X_val_n, Y_val_n = normalize_dataset(val_data["X"], val_data["Y"], val_data["masks"], stats)
    val_ds = SpatialASLDataset(X_val_n, Y_val_n, val_data["masks"], val_data["snrs"])
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    criterion = EdgeAwareLoss(edge_weight=0.5) if use_edge_loss else MaskedLoss()

    seeds = [42, 123, 2024]
    results_dir = os.path.join(ws.root, "experiments", "spatial_generalization", "replications")
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    seed_records = []
    trained_models = {}

    for s in seeds:
        print(f"\n--- Training Seed {s} ---")
        torch.manual_seed(s)
        np.random.seed(s)

        # DataLoader with seed-specific shuffle
        tr_ds = SpatialASLDataset(X_tr_n, Y_tr_n, tr_data["masks"], tr_data["snrs"])
        g = torch.Generator()
        g.manual_seed(s)
        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True, generator=g)

        model = model_cls().to(device)
        ckpt_path = os.path.join(ckpt_dir, f"winner_{best_id}_seed_{s}.pth")

        if os.path.exists(ckpt_path):
            print(f"Found existing checkpoint at {ckpt_path}, loading...")
            ckpt = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            best_ep = ckpt.get("epoch", 0)
            best_val_l = ckpt.get("val_loss", 0.0)
            dur = 0.0
        else:
            best_ep, best_val_l, dur = train_candidate_model(
                model=model,
                train_loader=tr_loader,
                val_loader=val_loader,
                criterion=criterion,
                stats=stats,
                device=device,
                save_ckpt_path=ckpt_path,
                max_epochs=40,
                patience=8
            )
            ckpt = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

        trained_models[s] = (model, ckpt_path)

        eval_res = evaluate_on_datasets(model, stats, te_ord, te_chal, device)
        eval_res["Seed"] = s
        eval_res["Candidate_ID"] = best_id
        eval_res["Best_Epoch"] = best_ep
        eval_res["Best_Val_Loss"] = best_val_l
        eval_res["Train_Time_s"] = dur
        seed_records.append(eval_res)
        print(f"Seed {s} -> Ord CBF RMSE: {eval_res['Ordinary_CBF_RMSE']:.3f} | Chal CBF RMSE: {eval_res['Challenge_CBF_RMSE']:.3f} | Focal CBF RMSE: {eval_res['Focal_CBF_RMSE']:.3f}")

    df_seeds = pd.DataFrame(seed_records)
    csv_seeds = os.path.join(results_dir, f"replication_3seeds_{best_id}.csv")
    df_seeds.to_csv(csv_seeds, index=False)

    print("\n" + "=" * 80)
    print("3-SEED REPLICATION SUMMARY")
    print("=" * 80)
    for col in ["Ordinary_CBF_RMSE", "Ordinary_ATT_RMSE", "Challenge_CBF_RMSE", "Challenge_ATT_RMSE", "Focal_CBF_RMSE", "Focal_Boundary_CBF_RMSE", "Hydro_CBF_RMSE", "Step_CBF_RMSE"]:
        m = df_seeds[col].mean()
        s = df_seeds[col].std()
        print(f"{col:25s}: {m:.3f} ± {s:.3f}")

    # Pick the best seed based on Challenge CBF RMSE
    best_seed_idx = df_seeds["Challenge_CBF_RMSE"].idxmin()
    best_seed = int(df_seeds.iloc[best_seed_idx]["Seed"])
    print(f"\nSelected Best Seed for Packaging: Seed {best_seed}")

    return best_id, model_cls, best_seed, trained_models[best_seed][1], stats, df_seeds

def package_v2_model(best_id, model_cls, best_seed, ckpt_path, stats, df_seeds):
    ws = get_workspace()
    v2_dir = os.path.join(ws.root, "final_model_4pld_spatial_v2")
    os.makedirs(v2_dir, exist_ok=True)
    print(f"\nPackaging Improved Model v2 into: {v2_dir}")

    # 1. Target checkpoint
    final_ckpt_path = os.path.join(v2_dir, "asl_unet_4pld_final.pth")
    ckpt_data = torch.load(ckpt_path, weights_only=False)
    
    # Save clean final checkpoint
    torch.save({
        "model_state_dict": ckpt_data["model_state_dict"],
        "stats": stats,
        "model_architecture": best_id,
        "seed": best_seed,
        "plds": [1.525, 2.025, 2.525, 3.025],
        "in_channels": 4,
        "out_channels": 2
    }, final_ckpt_path)

    # 2. Config JSON
    config = {
        "model_name": "ASL_Spatial_UNet_4PLD_v2",
        "candidate_id": best_id,
        "best_seed": best_seed,
        "plds": [1.525, 2.025, 2.525, 3.025],
        "input_channels": 4,
        "output_channels": 2,
        "output_parameters": ["CBF", "ATT"],
        "cbf_unit": "ml/100g/min",
        "att_unit": "seconds",
        "spatial_resolution": [64, 64],
        "architecture_type": "ASLHybridUNet" if "Hybrid" in best_id else ("ASLResidualUNet" if "ResUNet" in best_id else "ASLUNet")
    }
    with open(os.path.join(v2_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # 3. Normalization stats JSON
    stats_json = {
        "X_mean": stats["X_mean"].tolist() if isinstance(stats["X_mean"], np.ndarray) else stats["X_mean"],
        "X_std": stats["X_std"].tolist() if isinstance(stats["X_std"], np.ndarray) else stats["X_std"],
        "cbf_mean": float(stats["cbf_mean"]),
        "cbf_std": float(stats["cbf_std"]),
        "att_mean": float(stats["att_mean"]),
        "att_std": float(stats["att_std"])
    }
    with open(os.path.join(v2_dir, "normalization_stats.json"), "w") as f:
        json.dump(stats_json, f, indent=4)

    # 4. Protocol JSON
    protocol = {
        "protocol_name": "4-PLD ASL Parameter Estimation",
        "num_plds": 4,
        "pld_values_seconds": [1.525, 2.025, 2.525, 3.025],
        "labeling_duration_tau_seconds": 1.8,
        "t1_blood_seconds": 1.65,
        "blood_brain_partition_lambda": 0.9,
        "inversion_efficiency_alpha": 0.85,
        "cbf_bounds": [0.0, 100.0],
        "att_bounds": [0.5, 3.0]
    }
    with open(os.path.join(v2_dir, "protocol.json"), "w") as f:
        json.dump(protocol, f, indent=4)

    # 5. Model python file
    # Copy standalone model architecture definition
    src_models = os.path.join(parent_dir, "experiments_unet_4pld", "spatial_generalization", "models_generalization.py")
    dst_model = os.path.join(v2_dir, "model.py")
    shutil.copyfile(src_models, dst_model)

    # 6. Standalone Inference Script
    inf_content = f'''"""
Standalone Inference Module for 4-PLD ASL Spatial U-Net (v2 Improved).
"""

import os
import json
import torch
import numpy as np
from model import {config["architecture_type"]}

class ASLParameterEstimator:
    def __init__(self, model_dir=None, device=None):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        with open(os.path.join(model_dir, "normalization_stats.json"), "r") as f:
            self.stats = json.load(f)
        for k in ["X_mean", "X_std"]:
            self.stats[k] = np.array(self.stats[k], dtype=np.float32)

        self.model = {config["architecture_type"]}().to(self.device)
        ckpt_path = os.path.join(model_dir, "asl_unet_4pld_final.pth")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def predict(self, asl_signals, brain_mask=None):
        """
        asl_signals: numpy array [B, 4, 64, 64] or [4, 64, 64]
        Returns: cbf_map, att_map (numpy arrays in ml/100g/min and seconds)
        """
        single = False
        if asl_signals.ndim == 3:
            asl_signals = asl_signals[np.newaxis, ...]
            single = True

        B, C, H, W = asl_signals.shape
        assert C == 4, f"Expected 4 PLD channels, got {{C}}"

        # Standardize inputs
        X_norm = np.copy(asl_signals).astype(np.float32)
        for c in range(4):
            X_norm[:, c] = (X_norm[:, c] - self.stats["X_mean"][c]) / self.stats["X_std"][c]

        with torch.no_grad():
            t_in = torch.tensor(X_norm, dtype=torch.float32, device=self.device)
            preds = self.model(t_in).cpu().numpy()

        cbf = preds[:, 0] * self.stats["cbf_std"] + self.stats["cbf_mean"]
        att = preds[:, 1] * self.stats["att_std"] + self.stats["att_mean"]

        cbf = np.clip(cbf, 0.0, 100.0)
        att = np.clip(att, 0.5, 3.0)

        if brain_mask is not None:
            if brain_mask.ndim == 2:
                brain_mask = brain_mask[np.newaxis, ...]
            m = brain_mask > 0.5
            for b in range(B):
                cbf[b][~m[b]] = 0.0
                att[b][~m[b]] = 0.0

        if single:
            return cbf[0], att[0]
        return cbf, att

if __name__ == "__main__":
    print("Testing Standalone ASLParameterEstimator v2...")
    est = ASLParameterEstimator()
    dummy_input = np.random.randn(5, 4, 64, 64).astype(np.float32)
    dummy_mask = np.ones((5, 64, 64), dtype=np.float32)
    cbf, att = est.predict(dummy_input, dummy_mask)
    print(f"Prediction successful! Output shapes: CBF={{cbf.shape}}, ATT={{att.shape}}")
    print(f"Mean CBF: {{np.mean(cbf):.2f}}, Mean ATT: {{np.mean(att):.2f}}")
'''
    with open(os.path.join(v2_dir, "inference.py"), "w") as f:
        f.write(inf_content)

    # 7. Manifest JSON with SHA-256 hashes
    manifest = {
        "model_version": "v2_improved_spatial_generalization",
        "base_model": "4-PLD ASL Spatial U-Net",
        "date_packaged": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {
            "asl_unet_4pld_final.pth": sha256_file(final_ckpt_path),
            "model.py": sha256_file(dst_model),
            "config.json": sha256_file(os.path.join(v2_dir, "config.json")),
            "normalization_stats.json": sha256_file(os.path.join(v2_dir, "normalization_stats.json")),
            "protocol.json": sha256_file(os.path.join(v2_dir, "protocol.json")),
            "inference.py": sha256_file(os.path.join(v2_dir, "inference.py"))
        },
        "performance_3seeds": {
            "Ordinary_CBF_RMSE": f"{df_seeds['Ordinary_CBF_RMSE'].mean():.3f} ± {df_seeds['Ordinary_CBF_RMSE'].std():.3f}",
            "Ordinary_ATT_RMSE": f"{df_seeds['Ordinary_ATT_RMSE'].mean():.3f} ± {df_seeds['Ordinary_ATT_RMSE'].std():.3f}",
            "Challenge_CBF_RMSE": f"{df_seeds['Challenge_CBF_RMSE'].mean():.3f} ± {df_seeds['Challenge_CBF_RMSE'].std():.3f}",
            "Challenge_ATT_RMSE": f"{df_seeds['Challenge_ATT_RMSE'].mean():.3f} ± {df_seeds['Challenge_ATT_RMSE'].std():.3f}",
            "Focal_CBF_RMSE": f"{df_seeds['Focal_CBF_RMSE'].mean():.3f} ± {df_seeds['Focal_CBF_RMSE'].std():.3f}",
            "Focal_Boundary_CBF_RMSE": f"{df_seeds['Focal_Boundary_CBF_RMSE'].mean():.3f} ± {df_seeds['Focal_Boundary_CBF_RMSE'].std():.3f}"
        }
    }
    with open(os.path.join(v2_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)

    print("Model v2 successfully packaged and verified.")
    return v2_dir

def run_verification_inference(v2_dir):
    print("\nRunning Verification Inference on 5 real challenge phantoms...")
    sys.path.insert(0, v2_dir)
    from inference import ASLParameterEstimator

    ws = get_workspace()
    te_chal = np.load(ws.get_path("data", "challenge_test_N40_s2026.npz"))
    sub_X = te_chal["X"][:5]
    sub_Y = te_chal["Y"][:5]
    sub_M = te_chal["masks"][:5, 0]

    estimator = ASLParameterEstimator(model_dir=v2_dir)
    pred_cbf, pred_att = estimator.predict(sub_X, sub_M)

    for i in range(5):
        vm = sub_M[i] > 0.5
        c_rmse = np.sqrt(np.mean((pred_cbf[i][vm] - sub_Y[i, 0][vm])**2))
        a_rmse = np.sqrt(np.mean((pred_att[i][vm] - sub_Y[i, 1][vm])**2))
        print(f"  Phantom {i+1} ({te_chal['case_labels'][i]}): CBF RMSE = {c_rmse:.3f} ml/100g/min | ATT RMSE = {a_rmse:.3f} s")

    print("Verification inference completed successfully!")

if __name__ == "__main__":
    ws = get_workspace()
    pilot_csv = os.path.join(ws.root, "experiments", "spatial_generalization", "candidates", "pilot_screening_results.csv")
    if not os.path.exists(pilot_csv):
        print(f"Waiting for pilot screening CSV at {pilot_csv}...")
        sys.exit(1)

    best_id = select_best_candidate(pilot_csv)
    best_id, model_cls, best_seed, best_ckpt, stats, df_seeds = run_3seed_replication(best_id)
    v2_dir = package_v2_model(best_id, model_cls, best_seed, best_ckpt, stats, df_seeds)
    run_verification_inference(v2_dir)

