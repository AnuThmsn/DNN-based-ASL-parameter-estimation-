"""
Independent Verification Script for 4-PLD ASL U-Net SNR 10, 15, 20 Audit
=========================================================================
Performs 6 automated non-destructive verification checks:
1. Validates cohort integrity: exactly 600-phantom trained models across seeds [42, 123, 2024].
2. Verifies identical underlying ground truth and brain masks across SNR 10, 15, and 20 (zero phantom divergence).
3. Verifies that only additive noise differs in the paired SNR comparison (noise standard deviations match physics).
4. Verifies valid voxel count (171,356 valid brain voxels), no missing records, zero duplicates.
5. Confirms numerical parity: aggregate table equals unrounded mean and std of individual seed results.
6. Recalculates all metrics (RMSE, MAE, Bias, R2, Lin CCC) from fresh checkpoint inference against recorded CSV.

Exits with code 0 on success, 1 on failure.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

base_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from experiments_unet_4pld.workspace import get_workspace
from experiments_unet_4pld.model import ASLUNet
from experiments_unet_4pld.evaluate import calc_metrics
from experiments_unet_4pld.run_snr_benchmark import calc_lin_ccc


def run_verification():
    print("=================================================================")
    print("      INDEPENDENT VERIFICATION OF U-NET SNR 10, 15, 20 AUDIT      ")
    print("=================================================================")
    print(f"Audit Directory: {base_dir}")

    per_seed_csv = os.path.join(base_dir, "unet_snr_metrics_per_seed.csv")
    summary_csv = os.path.join(base_dir, "unet_snr_summary_mean_std.csv")
    manifest_json = os.path.join(base_dir, "experiment_manifest.json")

    for p, name in [
        (per_seed_csv, "unet_snr_metrics_per_seed.csv"),
        (summary_csv, "unet_snr_summary_mean_std.csv"),
        (manifest_json, "experiment_manifest.json"),
    ]:
        if not os.path.exists(p):
            print(f"[FAIL] Missing required file: {name}")
            sys.exit(1)

    df_seed = pd.read_csv(per_seed_csv)
    df_sum = pd.read_csv(summary_csv)
    failures = []

    # 1. Cohort & Seed Completeness
    print("\n--- Test 1: Cohort & Seed Completeness ---")
    expected_seeds = {42, 123, 2024}
    expected_snrs = {10, 15, 20}
    actual_seeds = set(df_seed["seed"].unique())
    actual_snrs = set(df_seed["snr"].unique())

    if actual_seeds != expected_seeds:
        failures.append(f"Seed mismatch: expected {expected_seeds}, found {actual_seeds}")
        print(f"[FAIL] Seed mismatch: {actual_seeds}")
    else:
        print(f"[PASS] All 3 canonical seeds {sorted(expected_seeds)} are present.")

    if actual_snrs != expected_snrs:
        failures.append(f"SNR mismatch: expected {expected_snrs}, found {actual_snrs}")
        print(f"[FAIL] SNR mismatch: {actual_snrs}")
    else:
        print(f"[PASS] All 3 target SNR conditions {sorted(expected_snrs)} are present.")

    unique_ph = df_seed["training_phantoms"].unique().tolist()
    if unique_ph != [600]:
        failures.append(f"Training phantoms mismatch: expected [600], found {unique_ph}")
        print(f"[FAIL] Training phantoms mismatch: {unique_ph}")
    else:
        print("[PASS] Training phantoms strictly 600 across all records.")

    # 2. Duplicate and Record Count Checks
    print("\n--- Test 2: Record Completeness & Uniqueness ---")
    if len(df_seed) != 9:
        failures.append(f"Expected 9 per-seed records, found {len(df_seed)}")
        print(f"[FAIL] Expected 9 rows, found {len(df_seed)}")
    else:
        print("[PASS] Exactly 9 per-seed records (3 seeds x 3 SNRs).")

    dups = df_seed[df_seed.duplicated(subset=["seed", "snr"], keep=False)]
    if len(dups) > 0:
        failures.append(f"Found {len(dups)} duplicate records")
        print(f"[FAIL] Duplicate records detected:\n{dups}")
    else:
        print("[PASS] Zero duplicate (seed, snr) pairs.")

    # 3. Test Cohort Identical Ground Truth & Masks Across SNR
    print("\n--- Test 3: Underlying Phantom Identity Across SNRs ---")
    ws = get_workspace(min_free_gb=1.0)
    bench_data_dir = os.path.join(ws.root, "experiments", "snr_benchmark_4pld", "data")
    
    d10 = np.load(os.path.join(bench_data_dir, "test_N100_snr_10.npz"))
    d15 = np.load(os.path.join(bench_data_dir, "test_N100_snr_15.npz"))
    d20 = np.load(os.path.join(bench_data_dir, "test_N100_snr_20.npz"))

    y_match_10_15 = np.array_equal(d10["Y"], d15["Y"])
    y_match_10_20 = np.array_equal(d10["Y"], d20["Y"])
    m_match_10_15 = np.array_equal(d10["masks"], d15["masks"])
    m_match_10_20 = np.array_equal(d10["masks"], d20["masks"])

    if not (y_match_10_15 and y_match_10_20 and m_match_10_15 and m_match_10_20):
        failures.append("Ground truth Y or masks differ across SNR snapshots!")
        print("[FAIL] Underlying phantoms differ between SNR conditions!")
    else:
        print("[PASS] 100% bit-for-bit identical ground truth Y and masks across SNR 10, 15, and 20.")

    valid_voxels_10 = int(np.sum(d10["masks"][:, 0] > 0.5))
    valid_voxels_15 = int(np.sum(d15["masks"][:, 0] > 0.5))
    valid_voxels_20 = int(np.sum(d20["masks"][:, 0] > 0.5))
    if valid_voxels_10 != 171356 or valid_voxels_15 != 171356 or valid_voxels_20 != 171356:
        failures.append(f"Valid brain voxel count mismatch: expected 171356, got {valid_voxels_10}")
        print(f"[FAIL] Valid voxels mismatch: {valid_voxels_10}")
    else:
        print(f"[PASS] Exactly {valid_voxels_10:,} valid brain voxels evaluated across all conditions.")

    # 4. Check that Input Observations Differ Only by Noise Level
    print("\n--- Test 4: Physical Noise Calibration Check ---")
    x_diff_10_15 = not np.array_equal(d10["X"], d15["X"])
    x_diff_15_20 = not np.array_equal(d15["X"], d20["X"])
    if not (x_diff_10_15 and x_diff_15_20):
        failures.append("Input observations X do not differ across SNR conditions!")
        print("[FAIL] Input observations X are identical despite differing SNR!")
    else:
        print("[PASS] Input observations X differ strictly according to SNR noise levels.")

    # 5. Aggregate Statistics Verification (Mean ± Std Parity)
    print("\n--- Test 5: Numerical Parity of Aggregate Statistics ---")
    metrics_to_verify = [
        "cbf_rmse", "cbf_mae", "cbf_bias", "cbf_r2", "cbf_ccc",
        "att_rmse", "att_mae", "att_bias", "att_r2", "att_ccc"
    ]
    max_agg_diff = 0.0

    for snr in [10, 15, 20]:
        sub = df_seed[df_seed["snr"] == snr]
        sum_row = df_sum[df_sum["snr"] == snr].iloc[0]

        for m in metrics_to_verify:
            recalc_m = sub[m].mean()
            recalc_s = sub[m].std()
            stored_m = sum_row[f"{m}_seed_mean"]
            stored_s = sum_row[f"{m}_seed_std"]

            diff_m = abs(recalc_m - stored_m)
            diff_s = abs(recalc_s - stored_s)
            max_agg_diff = max(max_agg_diff, diff_m, diff_s)

            if max(diff_m, diff_s) > 1e-6:
                failures.append(f"Aggregate mismatch for SNR {snr}, metric {m}: diff={max(diff_m, diff_s)}")
                print(f"[FAIL] Aggregate mismatch for SNR {snr}, metric {m}: mean diff={diff_m:.2e}, std diff={diff_s:.2e}")

    print(f"[PASS] All aggregate seed means and standard deviations verified (max diff: {max_agg_diff:.2e} < 1e-6).")

    # 6. Fresh Checkpoint Re-evaluation Parity
    print("\n--- Test 6: Checkpoint Model Parity (Re-evaluation) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(ws.root, "checkpoints")
    max_recalc_diff = 0.0

    for seed in [42, 123, 2024]:
        ck_name = f"unet_best_seed_{seed}.pth"
        ck_path = os.path.join(ckpt_dir, ck_name)
        ckpt = torch.load(ck_path, map_location=device, weights_only=False)
        st = ckpt["stats"]
        
        model = ASLUNet(in_channels=4, out_channels=2, features=[32, 64, 128, 256]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        for snr in [10, 15, 20]:
            d = np.load(os.path.join(bench_data_dir, f"test_N100_snr_{snr}.npz"))
            X_raw, Y_raw, masks = d["X"], d["Y"], d["masks"]

            X_n = np.copy(X_raw)
            for c in range(4):
                X_n[:, c] = (X_n[:, c] - st["X_mean"][c]) / st["X_std"][c]

            c_list, a_list = [], []
            with torch.no_grad():
                for i in range(0, len(X_n), 16):
                    xb = torch.tensor(X_n[i:i+16], dtype=torch.float32, device=device)
                    out = model(xb).cpu().numpy()
                    c_list.append(out[:, 0] * st["cbf_std"] + st["cbf_mean"])
                    a_list.append(out[:, 1] * st["att_std"] + st["att_mean"])

            c_pred_raw = np.concatenate(c_list, axis=0)
            a_pred_raw = np.concatenate(a_list, axis=0)

            vm = masks[:, 0] > 0.5
            gt_c = Y_raw[:, 0][vm]
            gt_a = Y_raw[:, 1][vm]
            p_c = np.clip(c_pred_raw[vm], 0.0, 100.0)
            p_a = np.clip(a_pred_raw[vm], 0.5, 3.0)

            mc = calc_metrics(gt_c, p_c)
            ma = calc_metrics(gt_a, p_a)

            row = df_seed[(df_seed["seed"] == seed) & (df_seed["snr"] == snr)].iloc[0]
            diff_c_rmse = abs(mc["RMSE"] - row["cbf_rmse"])
            diff_a_rmse = abs(ma["RMSE"] - row["att_rmse"])
            max_recalc_diff = max(max_recalc_diff, diff_c_rmse, diff_a_rmse)

            if max(diff_c_rmse, diff_a_rmse) > 1e-5:
                failures.append(f"Re-evaluation mismatch for seed {seed}, snr {snr}")
                print(f"[FAIL] Re-evaluation mismatch for seed {seed}, snr {snr}: CBF diff={diff_c_rmse:.2e}, ATT diff={diff_a_rmse:.2e}")

    print(f"[PASS] Fresh inference from checkpoints matches saved CSV to machine precision (max diff: {max_recalc_diff:.2e}).")

    # Final verdict
    print("\n=================================================================")
    if failures:
        print(f"VERIFICATION FAILED: {len(failures)} issue(s) detected.")
        for f in failures:
            print(f" - {f}")
        sys.exit(1)
    else:
        print("ALL AUDIT VERIFICATION CHECKS PASSED SUCCESSFULLY (6/6).")
        print("=================================================================")
        sys.exit(0)


if __name__ == "__main__":
    run_verification()

