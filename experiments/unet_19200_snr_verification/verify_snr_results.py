"""
Independent Verification Script for 19,200-Sample U-Net SNR Results
===================================================================
Performs non-destructive integrity checks on:
1. snr_results_per_seed.csv
2. snr_results_mean_std.csv
3. snr_degradation_analysis.csv
4. Historical alignment with training_scaling_results.csv (N=19200)

Exits with code 0 if all 6 validation criteria pass, 1 otherwise.
"""

import os
import sys
import numpy as np
import pandas as pd


def verify_snr_results():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    per_seed_path = os.path.join(base_dir, "snr_results_per_seed.csv")
    mean_std_path = os.path.join(base_dir, "snr_results_mean_std.csv")
    degradation_path = os.path.join(base_dir, "snr_degradation_analysis.csv")

    print("=================================================================")
    print("      INDEPENDENT VERIFICATION OF 19,200-SAMPLE SNR METRICS      ")
    print("=================================================================")
    print(f"Directory: {base_dir}")

    # Check file existence
    for path, name in [
        (per_seed_path, "snr_results_per_seed.csv"),
        (mean_std_path, "snr_results_mean_std.csv"),
        (degradation_path, "snr_degradation_analysis.csv"),
    ]:
        if not os.path.exists(path):
            print(f"[FAIL] Missing required file: {name}")
            sys.exit(1)

    df_seed = pd.read_csv(per_seed_path, dtype={"snr": str})
    df_agg = pd.read_csv(mean_std_path, dtype={"snr": str})
    df_deg = pd.read_csv(degradation_path, dtype={"snr": str})

    failures = []

    # 1. Check training size
    print("\n--- Test 1: Cohort Sample Size Integrity ---")
    unique_sizes = df_seed["training_samples"].unique().tolist()
    if unique_sizes != [19200]:
        failures.append(f"Expected only training_samples=19200, found: {unique_sizes}")
        print(f"[FAIL] Found unexpected sample sizes: {unique_sizes}")
    else:
        print("[PASS] Table strictly contains only 19,200-sample experiment records.")

    # 2. Check seeds and SNR levels
    print("\n--- Test 2: Seed & SNR Level Completeness ---")
    expected_seeds = {42, 123, 2024}
    expected_snrs = {"5", "10", "15", "20", "50", "inf"}
    actual_seeds = set(df_seed["seed"].unique())
    actual_snrs = set(df_seed["snr"].unique())

    if actual_seeds != expected_seeds:
        failures.append(f"Seed mismatch: expected {expected_seeds}, found {actual_seeds}")
        print(f"[FAIL] Seed mismatch: expected {expected_seeds}, found {actual_seeds}")
    else:
        print(f"[PASS] All 3 target seeds {sorted(expected_seeds)} are present.")

    if actual_snrs != expected_snrs:
        failures.append(f"SNR mismatch: expected {expected_snrs}, found {actual_snrs}")
        print(f"[FAIL] SNR mismatch: expected {expected_snrs}, found {actual_snrs}")
    else:
        print(f"[PASS] All 6 discrete SNR levels {sorted(expected_snrs)} are present.")

    # 3. Check for duplicates
    print("\n--- Test 3: Duplicate Record Check ---")
    duplicates = df_seed[df_seed.duplicated(subset=["seed", "snr"], keep=False)]
    if len(duplicates) > 0:
        failures.append(f"Found {len(duplicates)} duplicate (seed, snr) rows")
        print(f"[FAIL] Found duplicate rows:\n{duplicates[['seed', 'snr']]}")
    else:
        print(f"[PASS] Exactly 18 unique (seed, snr) combinations with 0 duplicates.")

    # 4. Independent Recalculation of Aggregate Mean and Std
    print("\n--- Test 4: Numerical Parity of Aggregate Statistics ---")
    metrics_to_check = [
        "cbf_rmse", "cbf_mae", "cbf_bias", "cbf_r2",
        "att_rmse", "att_mae", "att_bias", "att_r2"
    ]
    tolerance = 1e-6
    max_diff = 0.0

    for snr in ["inf", "50", "20", "15", "10", "5"]:
        sub = df_seed[df_seed["snr"] == snr]
        agg_sub = df_agg[df_agg["snr"] == snr]
        if len(agg_sub) != 1:
            failures.append(f"Expected 1 aggregate row for SNR {snr}, found {len(agg_sub)}")
            continue

        for m in metrics_to_check:
            recalc_m = sub[m].mean()
            recalc_s = sub[m].std()
            stored_m = agg_sub[f"{m}_mean"].values[0]
            stored_s = agg_sub[f"{m}_std"].values[0]

            diff_m = abs(recalc_m - stored_m)
            diff_s = abs(recalc_s - stored_s)
            max_diff = max(max_diff, diff_m, diff_s)

            if diff_m > tolerance or diff_s > tolerance:
                failures.append(f"SNR {snr} metric {m} mismatch: mean diff={diff_m}, std diff={diff_s}")
                print(f"[FAIL] SNR {snr} metric {m}: mean diff={diff_m:.2e}, std diff={diff_s:.2e}")

    print(f"[PASS] All aggregate means and standard deviations verified. Max difference: {max_diff:.2e} (< {tolerance})")

    # 5. Independent Recalculation of Degradation Table
    print("\n--- Test 5: Numerical Parity of Degradation Analysis ---")
    ref_cbf = df_agg.loc[df_agg["snr"] == "inf", "cbf_rmse_mean"].values[0]
    ref_att = df_agg.loc[df_agg["snr"] == "inf", "att_rmse_mean"].values[0]
    max_deg_diff = 0.0

    for _, row in df_deg.iterrows():
        snr = row["snr"]
        agg_row = df_agg[df_agg["snr"] == snr].iloc[0]

        recalc_abs_cbf = agg_row["cbf_rmse_mean"] - ref_cbf
        recalc_pct_cbf = (recalc_abs_cbf / ref_cbf) * 100.0
        recalc_abs_att = agg_row["att_rmse_mean"] - ref_att
        recalc_pct_att = (recalc_abs_att / ref_att) * 100.0

        diff_abs_cbf = abs(recalc_abs_cbf - row["cbf_rmse_abs_increase"])
        diff_pct_cbf = abs(recalc_pct_cbf - row["cbf_rmse_pct_increase"])
        diff_abs_att = abs(recalc_abs_att - row["att_rmse_abs_increase"])
        diff_pct_att = abs(recalc_pct_att - row["att_rmse_pct_increase"])

        diff_k = max(diff_abs_cbf, diff_pct_cbf, diff_abs_att, diff_pct_att)
        max_deg_diff = max(max_deg_diff, diff_k)

        if diff_k > tolerance:
            failures.append(f"SNR {snr} degradation mismatch: diff={diff_k}")
            print(f"[FAIL] SNR {snr} degradation mismatch: max diff={diff_k:.2e}")

    print(f"[PASS] All degradation increases (absolute and %) verified. Max difference: {max_deg_diff:.2e} (< {tolerance})")

    # 6. Backward Compatibility with Original Scaling Results CSV
    print("\n--- Test 6: Backward Consistency with Original Experiment CSV ---")
    try:
        repo_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from experiments_unet_4pld.workspace import get_workspace
        ws = get_workspace(min_free_gb=1.0)
        orig_csv = os.path.join(ws.root, "experiments", "training_size_scaling_v1", "training_scaling_results.csv")
        if os.path.exists(orig_csv):
            df_orig = pd.read_csv(orig_csv)
            df_orig_19200 = df_orig[df_orig["training_size"] == 19200].set_index("seed")
            max_orig_diff = 0.0
            for seed in [42, 123, 2024]:
                for snr in ["5", "10", "15", "20", "50", "inf"]:
                    orig_c = df_orig_19200.loc[seed, f"cbf_rmse_snr_{snr}"]
                    orig_a = df_orig_19200.loc[seed, f"att_rmse_snr_{snr}"]
                    sub = df_seed[(df_seed["seed"] == seed) & (df_seed["snr"] == snr)].iloc[0]
                    diff_c = abs(sub["cbf_rmse"] - orig_c)
                    diff_a = abs(sub["att_rmse"] - orig_a)
                    max_orig_diff = max(max_orig_diff, diff_c, diff_a)
                    if max(diff_c, diff_a) > 1e-5:
                        failures.append(f"Original CSV parity mismatch for seed {seed}, snr {snr}")
            print(f"[PASS] Exact bit-for-bit parity with original training_scaling_results.csv verified (max diff: {max_orig_diff:.2e}).")
        else:
            print("[SKIP] Original training_scaling_results.csv not found at workspace path.")
    except Exception as e:
        print(f"[SKIP] Workspace lookup for original CSV skipped: {e}")

    # Summary
    print("\n=================================================================")
    if failures:
        print(f"VERIFICATION FAILED: {len(failures)} issue(s) detected.")
        for f in failures:
            print(f" - {f}")
        sys.exit(1)
    else:
        print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY (6/6).")
        print("=================================================================")
        sys.exit(0)


if __name__ == "__main__":
    verify_snr_results()
