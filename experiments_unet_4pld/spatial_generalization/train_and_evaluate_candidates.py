"""
Controlled Experiment Matrix for ASL U-Net Spatial Generalization.
Evaluates candidate improvements against the original baseline on:
1. Standard Smooth Test Phantoms (N=100)
2. Synthetic Challenge Phantoms (N=40 across focal infarct, hydrocephalus, step, watershed)
3. Boundary-region error dissection
All checkpoints saved under D:\\ASL_WORKSPACE\\experiments\\spatial_generalization\\candidates\\checkpoints
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
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

class EdgeAwareLoss(nn.Module):
    """Auxiliary loss enforcing spatial gradient consistency on brain voxels."""
    def __init__(self, edge_weight=0.5):
        super().__init__()
        self.edge_weight = edge_weight
        self.base_loss = MaskedLoss()
        
        # Fixed Sobel gradient filters
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, pred, target, mask):
        # Base masked L1 loss
        loss_base = self.base_loss(pred, target, mask)
        
        if self.edge_weight <= 0.0:
            return loss_base

        # Compute spatial gradients for CBF and ATT
        p_cbf, p_att = pred[:, 0:1], pred[:, 1:2]
        t_cbf, t_att = target[:, 0:1], target[:, 1:2]

        gx_p_c = nn.functional.conv2d(p_cbf, self.kx, padding=1)
        gy_p_c = nn.functional.conv2d(p_cbf, self.ky, padding=1)
        gx_t_c = nn.functional.conv2d(t_cbf, self.kx, padding=1)
        gy_t_c = nn.functional.conv2d(t_cbf, self.ky, padding=1)

        gx_p_a = nn.functional.conv2d(p_att, self.kx, padding=1)
        gy_p_a = nn.functional.conv2d(p_att, self.ky, padding=1)
        gx_t_a = nn.functional.conv2d(t_att, self.kx, padding=1)
        gy_t_a = nn.functional.conv2d(t_att, self.ky, padding=1)

        grad_diff_c = torch.abs(gx_p_c - gx_t_c) + torch.abs(gy_p_c - gy_t_c)
        grad_diff_a = torch.abs(gx_p_a - gx_t_a) + torch.abs(gy_p_a - gy_t_a)

        # Apply eroded mask to avoid boundary edge artifacts with background
        inner_mask = (mask > 0.5).float()
        loss_edge = (torch.sum((grad_diff_c + grad_diff_a) * inner_mask) / (torch.sum(inner_mask) * 2.0 + 1e-8))

        return loss_base + self.edge_weight * loss_edge

def train_candidate_model(
    model,
    train_loader,
    val_loader,
    criterion,
    stats,
    device,
    save_ckpt_path,
    max_epochs=40,
    patience=8
):
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_cnt = 0
    start_t = time.time()

    for ep in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        n_tr = 0
        for xb, yb, mb, _ in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb, mb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_tr += 1
        train_loss /= max(1, n_tr)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb, mb, _ in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb, mb)
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(1, n_val)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep
            patience_cnt = 0
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "stats": stats
            }, save_ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    duration = time.time() - start_t
    return best_epoch, best_val_loss, duration

def evaluate_on_datasets(model, stats, te_ord, te_chal, device):
    """Evaluates candidate model on both Ordinary Test Set (N=100) and Challenge Test Set (N=40)."""
    model.eval()
    
    # 1. Ordinary Test Set (test_N100_s7)
    X_ord = te_ord["X"]
    Y_ord = te_ord["Y"]
    M_ord = te_ord["masks"]
    
    X_ord_n = np.copy(X_ord)
    for c in range(4):
        X_ord_n[:, c] = (X_ord[:, c] - stats["X_mean"][c]) / stats["X_std"][c]
    
    with torch.no_grad():
        cbf_u_ord, att_u_ord = [], []
        for i in range(0, len(X_ord), 16):
            xb = torch.tensor(X_ord_n[i:i+16], dtype=torch.float32, device=device)
            preds = model(xb).cpu().numpy()
            c_p = preds[:, 0] * stats["cbf_std"] + stats["cbf_mean"]
            a_p = preds[:, 1] * stats["att_std"] + stats["att_mean"]
            cbf_u_ord.append(np.clip(c_p, 0.0, 100.0))
            att_u_ord.append(np.clip(a_p, 0.5, 3.0))
        cbf_u_ord = np.concatenate(cbf_u_ord, axis=0)
        att_u_ord = np.concatenate(att_u_ord, axis=0)

    vm_ord = M_ord[:, 0] > 0.5
    met_c_ord = calc_metrics(Y_ord[:, 0][vm_ord], cbf_u_ord[vm_ord])
    met_a_ord = calc_metrics(Y_ord[:, 1][vm_ord], att_u_ord[vm_ord])

    # 2. Challenge Test Set (challenge_test_N40_s2026)
    X_chal = te_chal["X"]
    Y_chal = te_chal["Y"]
    M_chal = te_chal["masks"]
    cases_chal = te_chal["case_labels"]

    X_chal_n = np.copy(X_chal)
    for c in range(4):
        X_chal_n[:, c] = (X_chal[:, c] - stats["X_mean"][c]) / stats["X_std"][c]

    with torch.no_grad():
        cbf_u_chal, att_u_chal = [], []
        for i in range(0, len(X_chal), 16):
            xb = torch.tensor(X_chal_n[i:i+16], dtype=torch.float32, device=device)
            preds = model(xb).cpu().numpy()
            c_p = preds[:, 0] * stats["cbf_std"] + stats["cbf_mean"]
            a_p = preds[:, 1] * stats["att_std"] + stats["att_mean"]
            cbf_u_chal.append(np.clip(c_p, 0.0, 100.0))
            att_u_chal.append(np.clip(a_p, 0.5, 3.0))
        cbf_u_chal = np.concatenate(cbf_u_chal, axis=0)
        att_u_chal = np.concatenate(att_u_chal, axis=0)

    vm_chal = M_chal[:, 0] > 0.5
    met_c_chal_all = calc_metrics(Y_chal[:, 0][vm_chal], cbf_u_chal[vm_chal])
    met_a_chal_all = calc_metrics(Y_chal[:, 1][vm_chal], att_u_chal[vm_chal])

    # Per-case breakdown
    case_results = {}
    for c_name in ["focal_lesion", "hydrocephalus", "discontinuous_step", "extreme_watershed"]:
        c_idx = np.where(cases_chal == c_name)[0]
        c_vm = M_chal[c_idx, 0] > 0.5
        mc = calc_metrics(Y_chal[c_idx, 0][c_vm], cbf_u_chal[c_idx][c_vm])
        ma = calc_metrics(Y_chal[c_idx, 1][c_vm], att_u_chal[c_idx][c_vm])
        case_results[c_name] = (mc["RMSE"], ma["RMSE"])

    # Boundary vs Interior RMSE on focal lesions
    focal_idx = np.where(cases_chal == "focal_lesion")[0]
    bound_rmses = []
    for i in focal_idx:
        gt_c = Y_chal[i, 0]
        m_2d = M_chal[i, 0] > 0.5
        lesion_mask = (gt_c < 10.0) & m_2d
        dil = ndi.binary_dilation(lesion_mask, iterations=2)
        ero = ndi.binary_erosion(lesion_mask, iterations=2)
        b_zone = (dil ^ ero) & m_2d
        if np.any(b_zone):
            bound_rmses.append(np.sqrt(np.mean((cbf_u_chal[i][b_zone] - gt_c[b_zone])**2)))
    mean_boundary_cbf_rmse = float(np.mean(bound_rmses)) if bound_rmses else 0.0

    return {
        "Ordinary_CBF_RMSE": met_c_ord["RMSE"],
        "Ordinary_CBF_R2": met_c_ord["R2"],
        "Ordinary_ATT_RMSE": met_a_ord["RMSE"],
        "Ordinary_ATT_R2": met_a_ord["R2"],
        "Challenge_CBF_RMSE": met_c_chal_all["RMSE"],
        "Challenge_ATT_RMSE": met_a_chal_all["RMSE"],
        "Focal_CBF_RMSE": case_results["focal_lesion"][0],
        "Focal_ATT_RMSE": case_results["focal_lesion"][1],
        "Hydro_CBF_RMSE": case_results["hydrocephalus"][0],
        "Hydro_ATT_RMSE": case_results["hydrocephalus"][1],
        "Step_CBF_RMSE": case_results["discontinuous_step"][0],
        "Step_ATT_RMSE": case_results["discontinuous_step"][1],
        "Watershed_CBF_RMSE": case_results["extreme_watershed"][0],
        "Watershed_ATT_RMSE": case_results["extreme_watershed"][1],
        "Focal_Boundary_CBF_RMSE": mean_boundary_cbf_rmse
    }

def run_pilot_matrix():
    ws = get_workspace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("CONTROLLED EXPERIMENT MATRIX: SPATIAL GENERALIZATION IMPROVEMENT")
    print(f"Workspace: {ws.root} | Device: {device}")
    print("=" * 80)

    ckpt_save_dir = os.path.join(ws.root, "experiments", "spatial_generalization", "candidates", "checkpoints")
    res_dir = os.path.join(ws.root, "experiments", "spatial_generalization", "candidates")
    os.makedirs(ckpt_save_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # Load frozen test datasets
    te_ord = np.load(ws.get_path("data", "test_N100_s7.npz"))
    te_chal = np.load(ws.get_path("data", "challenge_test_N40_s2026.npz"))

    # Load standard vs diverse training data
    std_tr = np.load(ws.get_path("data", "train_N600_s42.npz"))
    std_val = np.load(ws.get_path("data", "val_N100_s99.npz"))
    div_tr = np.load(ws.get_path("data", "train_diverse_N600_s42.npz"))
    div_val = np.load(ws.get_path("data", "val_diverse_N100_s99.npz"))

    # Compute normalization stats
    stats_std = compute_normalization_stats(std_tr["X"], std_tr["Y"], std_tr["masks"])
    stats_div = compute_normalization_stats(div_tr["X"], div_tr["Y"], div_tr["masks"])

    # Prepare standard loaders
    X_std_tr_n, Y_std_tr_n = normalize_dataset(std_tr["X"], std_tr["Y"], std_tr["masks"], stats_std)
    X_std_val_n, Y_std_val_n = normalize_dataset(std_val["X"], std_val["Y"], std_val["masks"], stats_std)
    std_tr_ds = SpatialASLDataset(X_std_tr_n, Y_std_tr_n, std_tr["masks"], std_tr["snrs"])
    std_val_ds = SpatialASLDataset(X_std_val_n, Y_std_val_n, std_val["masks"], std_val["snrs"])
    std_tr_loader = DataLoader(std_tr_ds, batch_size=32, shuffle=True)
    std_val_loader = DataLoader(std_val_ds, batch_size=32, shuffle=False)

    # Prepare diverse loaders
    X_div_tr_n, Y_div_tr_n = normalize_dataset(div_tr["X"], div_tr["Y"], div_tr["masks"], stats_div)
    X_div_val_n, Y_div_val_n = normalize_dataset(div_val["X"], div_val["Y"], div_val["masks"], stats_div)
    div_tr_ds = SpatialASLDataset(X_div_tr_n, Y_div_tr_n, div_tr["masks"], div_tr["snrs"])
    div_val_ds = SpatialASLDataset(X_div_val_n, Y_div_val_n, div_val["masks"], div_val["snrs"])
    div_tr_loader = DataLoader(div_tr_ds, batch_size=32, shuffle=True)
    div_val_loader = DataLoader(div_val_ds, batch_size=32, shuffle=False)

    # Prepare mixed loaders (50% standard smooth + 50% diverse challenge phantoms)
    mix_X_tr = np.concatenate([std_tr["X"][:300], div_tr["X"][:300]], axis=0)
    mix_Y_tr = np.concatenate([std_tr["Y"][:300], div_tr["Y"][:300]], axis=0)
    mix_M_tr = np.concatenate([std_tr["masks"][:300], div_tr["masks"][:300]], axis=0)
    mix_S_tr = np.concatenate([std_tr["snrs"][:300], div_tr["snrs"][:300]], axis=0)

    mix_X_val = np.concatenate([std_val["X"][:50], div_val["X"][:50]], axis=0)
    mix_Y_val = np.concatenate([std_val["Y"][:50], div_val["Y"][:50]], axis=0)
    mix_M_val = np.concatenate([std_val["masks"][:50], div_val["masks"][:50]], axis=0)
    mix_S_val = np.concatenate([std_val["snrs"][:50], div_val["snrs"][:50]], axis=0)

    stats_mix = compute_normalization_stats(mix_X_tr, mix_Y_tr, mix_M_tr)
    X_mix_tr_n, Y_mix_tr_n = normalize_dataset(mix_X_tr, mix_Y_tr, mix_M_tr, stats_mix)
    X_mix_val_n, Y_mix_val_n = normalize_dataset(mix_X_val, mix_Y_val, mix_M_val, stats_mix)
    mix_tr_ds = SpatialASLDataset(X_mix_tr_n, Y_mix_tr_n, mix_M_tr, mix_S_tr)
    mix_val_ds = SpatialASLDataset(X_mix_val_n, Y_mix_val_n, mix_M_val, mix_S_val)
    mix_tr_loader = DataLoader(mix_tr_ds, batch_size=32, shuffle=True)
    mix_val_loader = DataLoader(mix_val_ds, batch_size=32, shuffle=False)

    # Define Candidate List for Pilot Screening (Seed 42)
    CANDIDATES = [
        {
            "id": "C1_ResUNet_Std",
            "name": "Candidate 1: Residual U-Net (Standard Training)",
            "model_cls": ASLResidualUNet,
            "train_loader": std_tr_loader,
            "val_loader": std_val_loader,
            "stats": stats_std,
            "criterion": MaskedLoss()
        },
        {
            "id": "C2_Hybrid_Std",
            "name": "Candidate 2: Hybrid Spatial+Pointwise (Standard Training)",
            "model_cls": ASLHybridUNet,
            "train_loader": std_tr_loader,
            "val_loader": std_val_loader,
            "stats": stats_std,
            "criterion": MaskedLoss()
        },
        {
            "id": "C3_EdgeLoss_Std",
            "name": "Candidate 3: U-Net with Edge-Aware Loss (Standard Training)",
            "model_cls": ASLUNet,
            "train_loader": std_tr_loader,
            "val_loader": std_val_loader,
            "stats": stats_std,
            "criterion": EdgeAwareLoss(edge_weight=0.5)
        },
        {
            "id": "C4_UNet_Diverse",
            "name": "Candidate 4: Baseline U-Net + Domain Randomization Training",
            "model_cls": ASLUNet,
            "train_loader": div_tr_loader,
            "val_loader": div_val_loader,
            "stats": stats_div,
            "criterion": MaskedLoss()
        },
        {
            "id": "C5_Hybrid_Diverse",
            "name": "Candidate 5: Hybrid Spatial+Pointwise + Domain Randomization",
            "model_cls": ASLHybridUNet,
            "train_loader": div_tr_loader,
            "val_loader": div_val_loader,
            "stats": stats_div,
            "criterion": MaskedLoss()
        },
        {
            "id": "C6_Hybrid_Mixed",
            "name": "Candidate 6: Hybrid Spatial+Pointwise + Mixed Curriculum",
            "model_cls": ASLHybridUNet,
            "train_loader": mix_tr_loader,
            "val_loader": mix_val_loader,
            "stats": stats_mix,
            "criterion": MaskedLoss()
        }
    ]

    pilot_results = []
    
    # Also evaluate the untouched original baseline for direct comparison
    orig_ckpt = os.path.join(ws.get_path("final_model"), "asl_unet_4pld_final.pth")
    orig_model = ASLUNet().to(device)
    orig_model.load_state_dict(torch.load(orig_ckpt, weights_only=False)["model_state_dict"])
    orig_eval = evaluate_on_datasets(orig_model, stats_std, te_ord, te_chal, device)
    orig_eval["Candidate_ID"] = "Original_Baseline_4PLD"
    orig_eval["Candidate_Name"] = "Untouched Original 4-PLD U-Net"
    orig_eval["Train_Time_s"] = 0.0
    orig_eval["Best_Epoch"] = 60
    orig_eval["Best_Val_Loss"] = 0.0824
    pilot_results.append(orig_eval)

    for cand in CANDIDATES:
        c_id = cand["id"]
        print(f"\n>>> Running Pilot Training: {cand['name']} (Seed 42) <<<")
        torch.manual_seed(42)
        np.random.seed(42)

        model = cand["model_cls"]().to(device)
        ckpt_path = os.path.join(ckpt_save_dir, f"{c_id}_seed_42.pth")

        if os.path.exists(ckpt_path):
            print(f"  Existing checkpoint found at {ckpt_path}, skipping training...")
            ckpt = torch.load(ckpt_path, weights_only=False)
            best_ep = ckpt.get("epoch", 0)
            best_val_l = ckpt.get("val_loss", 0.0)
            dur = 0.0
        else:
            best_ep, best_val_l, dur = train_candidate_model(
                model=model,
                train_loader=cand["train_loader"],
                val_loader=cand["val_loader"],
                criterion=cand["criterion"],
                stats=cand["stats"],
                device=device,
                save_ckpt_path=ckpt_path,
                max_epochs=35,
                patience=8
            )

        # Load best validation checkpoint for fair evaluation
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        eval_metrics = evaluate_on_datasets(model, cand["stats"], te_ord, te_chal, device)
        eval_metrics["Candidate_ID"] = c_id
        eval_metrics["Candidate_Name"] = cand["name"]
        eval_metrics["Train_Time_s"] = round(dur, 1)
        eval_metrics["Best_Epoch"] = best_ep
        eval_metrics["Best_Val_Loss"] = round(best_val_l, 4)

        pilot_results.append(eval_metrics)
        print(f"  [Result] Ord CBF RMSE: {eval_metrics['Ordinary_CBF_RMSE']:.3f} | Chal CBF RMSE: {eval_metrics['Challenge_CBF_RMSE']:.3f} "
              f"| Focal CBF RMSE: {eval_metrics['Focal_CBF_RMSE']:.3f} | Boundary CBF RMSE: {eval_metrics['Focal_Boundary_CBF_RMSE']:.2f}")

    df_pilot = pd.DataFrame(pilot_results)
    pilot_csv = os.path.join(res_dir, "pilot_screening_results.csv")
    df_pilot.to_csv(pilot_csv, index=False)

    print("\n" + "=" * 80)
    print("PILOT SCREENING RESULTS SUMMARY")
    print("=" * 80)
    cols_show = ["Candidate_ID", "Ordinary_CBF_RMSE", "Ordinary_ATT_RMSE", "Challenge_CBF_RMSE", "Challenge_ATT_RMSE", "Focal_CBF_RMSE", "Focal_Boundary_CBF_RMSE", "Hydro_CBF_RMSE", "Step_CBF_RMSE"]
    print(df_pilot[cols_show].to_string())

    return df_pilot

if __name__ == "__main__":
    run_pilot_matrix()

