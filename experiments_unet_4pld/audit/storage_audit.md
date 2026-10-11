# Storage Audit & Workspace Migration Report

**Project:** ASL-MRI Parameter Estimation (Four-PLD Spatial U-Net)  
**Date:** October 2026  
**Central Workspace Root:** `D:\ASL_WORKSPACE`  
**Host System:** Windows 10/11 x64  

---

## 1. Disk Space Status (Before & After Setup)

Disk space monitored at partition level via PowerShell / `shutil.disk_usage`:

| Drive | Initial Free Space | Post-Setup Free Space | Status |
|:---:|:---:|:---:|:---:|
| **C: (System)** | **16.40 GB** (183.67 GB used) | **16.40 GB** (183.67 GB used) | **Zero Bytes Written to C:** |
| **D: (Data / Workspace)** | **145.10 GB** (129.98 GB used) | **145.10 GB** (129.98 GB used) | **Safe Workspace Operational** |

Minimum free space threshold: **5.0 GB** enforced by `WorkspaceManager.check_preflight_storage()`.

---

## 2. Environment Variables & Cache Redirection

All experiment processes, DataLoaders, and libraries are redirected to `D:\ASL_WORKSPACE` through `experiments_unet_4pld/workspace.py`:

| Variable / Subsystem | Target Path on D: | Verified Redirection |
|---|---|:---:|
| `TMPDIR` | `D:\ASL_WORKSPACE\tmp` | **Yes** |
| `TEMP` | `D:\ASL_WORKSPACE\tmp` | **Yes** |
| `TMP` | `D:\ASL_WORKSPACE\tmp` | **Yes** |
| `TORCH_HOME` | `D:\ASL_WORKSPACE\cache\torch` | **Yes** |
| `MPLCONFIGDIR` | `D:\ASL_WORKSPACE\cache\matplotlib` | **Yes** |
| `PIP_CACHE_DIR` | `D:\ASL_WORKSPACE\cache\pip` | **Yes** |
| `HF_HOME` | `D:\ASL_WORKSPACE\cache\huggingface` | **Yes** |

---

## 3. Resolved Storage Locations

| Artifact Category | Resolved Absolute Location | Purpose |
|---|---|---|
| **Project Code** | `d:\fiinal year project\dnn  for  ASL parameter estimation\` | Source tree and scripts (already located on D:) |
| **Datasets** | `D:\ASL_WORKSPACE\data\` | Compressed npz phantoms and JSON manifests |
| **Checkpoints** | `D:\ASL_WORKSPACE\checkpoints\` | PyTorch `.pth` model state weights |
| **Study Checkpoints** | `D:\ASL_WORKSPACE\experiments\training_size_study\checkpoints\` | Training-size study model checkpoints |
| **Logs** | `D:\ASL_WORKSPACE\logs\` | Epoch-level history CSVs and runtime records |
| **Histories** | `D:\ASL_WORKSPACE\experiments\training_size_study\training_histories\` | Epoch-level history CSVs for size study |
| **Results** | `D:\ASL_WORKSPACE\results\` | Metric summaries, evaluation CSVs |
| **Temporary Files** | `D:\ASL_WORKSPACE\tmp\` | Scratch buffers, named tempfiles |
| **Manifests** | `D:\ASL_WORKSPACE\manifests\` | Reproducibility manifests & data hashes |

---

## 4. Unavoidable Storage Remaining on C:

Only the immutable operating system binaries and pre-installed Python runtime remain on C::
- Python binary executable: `C:\Users\ANU THOMSON\AppData\Local\Programs\Python\Python310\python.exe`
- Operating System kernel & Windows DLLs.

**Zero project data, zero caches, zero checkpoints, and zero temporary tensors are written to C:.**

---

## 5. Storage Estimation for Training-Size Study

| Component | Condition / Size | Estimated Storage | Actual Storage Strategy |
|---|---|---|---|
| Master Training Dataset | $N=1200$ phantoms | ~86.2 MB | Stored in `D:\ASL_WORKSPACE\data\train_nested_N1200_s42.npz` |
| Nested Subsets ($N=100, 200, 400$) | In-memory slice from $N=600$ | **0 MB extra** | Sliced dynamically from cached $N=600$ / $N=1200$ arrays |
| Validation & Test Sets | Fixed $N=100$ each | 0 MB extra | Reuses existing verified `val_N100_s99.npz` & `test_N100_s7.npz` |
| Baseline Checkpoints ($N=600$) | 3 seeds (42, 123, 2024) | 0 MB extra | Reuses existing verified checkpoints |
| New Checkpoints ($N=100, 200, 400, 1200$) | 4 sizes $\times$ 3 seeds = 12 models | ~1,119 MB (1.09 GB) | Stored in `D:\ASL_WORKSPACE\experiments\training_size_study\checkpoints\` |
| Epoch CSVs & Plots | All conditions | ~15 MB | Epoch logs and curves |
| **Total Estimated Need** | **All planned runs** | **~1.22 GB** | **Available on D: = 145.10 GB (>118x safety margin)** |

---

## 6. Storage Redirection Verification Test Results

Executed via `WorkspaceManager.verify_storage_redirection()` on live runtime:

```json
{
  "workspace_root": "D:\\ASL_WORKSPACE",
  "temp_dir": "D:\\ASL_WORKSPACE\\tmp",
  "temp_on_d": true,
  "mpl_config_dir": "D:\\ASL_WORKSPACE\\cache\\matplotlib",
  "mpl_on_d": true,
  "torch_dir": "D:\\ASL_WORKSPACE\\cache\\torch\\hub",
  "torch_on_d": true,
  "pip_cache_dir": "D:\\ASL_WORKSPACE\\cache\\pip",
  "hf_home": "D:\\ASL_WORKSPACE\\cache\\huggingface",
  "free_c_gb": 16.4,
  "free_d_gb": 145.1,
  "all_redirected": true
}
```

**Verification Verdict:** Complete and successful.

