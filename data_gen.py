import os
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- SETTINGS ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./asl_dnn_papermatched"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Physics constants — UNCHANGED ----
PLDs  = np.array([1.525, 2.025, 2.525, 3.025])
tau   = 1.8
T1t   = 1.2
T1a   = 1.66
alpha = 0.85
beta  = 0.75
lmbda = 0.9
SCALE = 100_000.0

CBF_MIN, CBF_MAX = 0.0,  100.0
ATT_MIN, ATT_MAX = 0.5,    3.0

SIG_REF_SC = 334.2039 # from notebook output
SD_MAX = SIG_REF_SC / 5.0
N_NOISE_LEVELS_TRAIN = 100
N_NOISE_LEVELS_VAL   = 100
N_NOISE_LEVELS_TEST  = 51

def compute_signals_vec(cbf, att, ld=tau):
    f_per_s = (cbf / (6000.0 * lmbda))[:, None]
    delta   = att[:, None]
    pld     = PLDs[None, :]
    prefix  = 2.0 * alpha * beta * T1t * (1.0 / lmbda) * f_per_s
    e_att   = np.exp(-delta / T1a)
    t1      = np.exp(-np.maximum(pld - delta,      0.0) / T1t)
    t2      = np.exp(-np.maximum(ld  + pld - delta, 0.0) / T1t)
    return prefix * e_att * (t1 - t2)

def generate_data(N_total, rng, n_noise_levels=N_NOISE_LEVELS_TRAIN):
    cbf_gt = rng.uniform(CBF_MIN, CBF_MAX, N_total).astype(np.float64)
    att_gt = rng.uniform(ATT_MIN, ATT_MAX, N_total).astype(np.float64)

    sd_levels = np.linspace(0.0, SD_MAX, n_noise_levels)
    sd_choice_idx = rng.integers(0, n_noise_levels, size=N_total)
    sd_arr = sd_levels[sd_choice_idx]

    sig = compute_signals_vec(cbf_gt, att_gt) * SCALE

    sd = sd_arr[:, None]
    e1 = rng.normal(0.0, 1.0, size=sig.shape) * sd
    e2 = rng.normal(0.0, 1.0, size=sig.shape) * sd
    e3 = rng.normal(0.0, 1.0, size=sig.shape) * sd
    e4 = rng.normal(0.0, 1.0, size=sig.shape) * sd

    mc = np.sqrt((sig + e1) ** 2 + e2 ** 2)
    ml = np.sqrt((sig + e3) ** 2 + e4 ** 2)
    X = (mc + ml).astype(np.float32)

    Y = np.stack([cbf_gt, att_gt], axis=1).astype(np.float32)
    return X, Y

N_TRAIN, N_VAL, N_TEST = 2_000_000, 50_000, 100_000
rng_tr   = np.random.default_rng(42)
rng_val  = np.random.default_rng(99)
rng_test = np.random.default_rng(7)

print("Generating data...")
X_tr, Y_tr = generate_data(N_TRAIN, rng_tr, n_noise_levels=N_NOISE_LEVELS_TRAIN)
X_val, Y_val = generate_data(N_VAL, rng_val, n_noise_levels=N_NOISE_LEVELS_VAL)
X_test, Y_test = generate_data(N_TEST, rng_test, n_noise_levels=N_NOISE_LEVELS_TEST)

X_mean = X_tr.mean(axis=0, keepdims=True).astype(np.float32)
X_std  = X_tr.std(axis=0,  keepdims=True).astype(np.float32) + 1e-8

CBF_mean, CBF_std = float(Y_tr[:, 0].mean()), float(Y_tr[:, 0].std() + 1e-8)
ATT_mean, ATT_std = float(Y_tr[:, 1].mean()), float(Y_tr[:, 1].std() + 1e-8)

X_tr_n   = (X_tr   - X_mean) / X_std
X_val_n  = (X_val  - X_mean) / X_std
X_test_n = (X_test - X_mean) / X_std

Y_tr_std_cbf = (Y_tr[:, 0] - CBF_mean) / CBF_std
Y_val_std_cbf = (Y_val[:, 0] - CBF_mean) / CBF_std
Y_test_std_cbf = (Y_test[:, 0] - CBF_mean) / CBF_std

Y_tr_std_att = (Y_tr[:, 1] - ATT_mean) / ATT_std
Y_val_std_att = (Y_val[:, 1] - ATT_mean) / ATT_std
Y_test_std_att = (Y_test[:, 1] - ATT_mean) / ATT_std

Y_tr_std = np.stack([Y_tr_std_cbf, Y_tr_std_att], axis=1)
Y_val_std = np.stack([Y_val_std_cbf, Y_val_std_att], axis=1)

X_tr_t = torch.tensor(X_tr_n, device=device)
Y_tr_cbf_t = torch.tensor(Y_tr_std_cbf, device=device).unsqueeze(1)
Y_tr_att_t = torch.tensor(Y_tr_std_att, device=device).unsqueeze(1)
Y_tr_both_t = torch.tensor(Y_tr_std, device=device)

X_val_t = torch.tensor(X_val_n, device=device)
Y_val_cbf_t = torch.tensor(Y_val_std_cbf, device=device).unsqueeze(1)
Y_val_att_t = torch.tensor(Y_val_std_att, device=device).unsqueeze(1)
Y_val_both_t = torch.tensor(Y_val_std, device=device)

def calc_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def calc_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def calc_bias(y_true, y_pred):
    return np.mean(y_pred - y_true)

print("Data ready.")

