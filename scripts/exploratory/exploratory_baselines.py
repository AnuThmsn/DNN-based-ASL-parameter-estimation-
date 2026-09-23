import numpy as np
import scipy.optimize
from data_gen import compute_signals_vec, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX
import time

def nlls_residual(params, y_true):
    # params = [CBF, ATT]
    cbf, att = params
    sig = compute_signals_vec(np.array([cbf]), np.array([att])) * SCALE
    return (sig[0] - y_true).flatten()

def fit_nlls_batch(X_batch):
    start_time = time.time()
    n_samples = X_batch.shape[0]
    preds = np.zeros((n_samples, 2))
    
    for i in range(n_samples):
        y_true = X_batch[i]
        
        # We need a robust initialization. 
        # Mean of uniform bounds is standard for naive NLLS
        init_guess = [50.0, 1.5]
        
        res = scipy.optimize.least_squares(
            nlls_residual, 
            x0=init_guess,
            args=(y_true,),
            bounds=([CBF_MIN, ATT_MIN], [CBF_MAX, ATT_MAX]),
            method='trf'
        )
        preds[i] = res.x
        
    return preds, time.time() - start_time

def fit_bayesian_batch(X_batch, snr_batch, sig_ref_sc):
    start_time = time.time()
    n_samples = X_batch.shape[0]
    preds = np.zeros((n_samples, 2))
    
    # Precompute a dense grid of signals
    cbf_grid = np.linspace(CBF_MIN, CBF_MAX, 100)
    att_grid = np.linspace(ATT_MIN, ATT_MAX, 100)
    CC, AA = np.meshgrid(cbf_grid, att_grid)
    
    grid_cbf_flat = CC.flatten()
    grid_att_flat = AA.flatten()
    
    # Shape: (10000, 4)
    grid_sig = compute_signals_vec(grid_cbf_flat, grid_att_flat) * SCALE
    
    for i in range(n_samples):
        y_true = X_batch[i]
        snr = snr_batch[i]
        
        # Estimate noise std dev. Note: if SNR=inf, set small noise
        sigma = sig_ref_sc / snr if snr < 1e5 else 1e-6
        # To avoid division by zero or underflow in exp, add small epsilon to sigma
        sigma = max(sigma, 1e-6)
        
        # Gaussian log-likelihood approximation (ignoring constants)
        # Log L = -0.5 * sum( (y_true - grid_sig)^2 ) / sigma^2
        sq_diff = np.sum((y_true[None, :] - grid_sig)**2, axis=1)
        log_L = -0.5 * sq_diff / (sigma**2)
        
        # Shift log_L to prevent underflow in exp
        log_L -= np.max(log_L)
        L = np.exp(log_L)
        
        # Grid marginalization (flat prior)
        Z = np.sum(L)
        if Z > 0:
            p = L / Z
            cbf_pred = np.sum(p * grid_cbf_flat)
            att_pred = np.sum(p * grid_att_flat)
        else:
            cbf_pred = 50.0
            att_pred = 1.5
            
        preds[i, 0] = cbf_pred
        preds[i, 1] = att_pred
        
    return preds, time.time() - start_time

