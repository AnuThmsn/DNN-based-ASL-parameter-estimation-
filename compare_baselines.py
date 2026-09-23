import pandas as pd

snr_df = pd.read_csv("baseline_snr_metrics.csv")

EXTERNAL_REFERENCE = {
    10.0: {
        "Bayesian": {"CBF_RMSE": 5.030, "CBF_MAE": 3.968, "CBF_CCC": 0.9606, "ATT_RMSE": 0.0826, "ATT_MAE": 0.0683, "ATT_CCC": 0.9914},
        "NLS":      {"CBF_RMSE": 2.982, "CBF_MAE": 2.048, "CBF_CCC": 0.9913, "ATT_RMSE": 0.5062, "ATT_MAE": 0.3411, "ATT_CCC": 0.6770},
    },
    15.0: {
        "Bayesian": {"CBF_RMSE": 3.759, "CBF_MAE": 2.849, "CBF_CCC": 0.9671, "ATT_RMSE": 0.0584, "ATT_MAE": 0.0472, "ATT_CCC": 0.9944},
        "NLS":      {"CBF_RMSE": 2.062, "CBF_MAE": 1.414, "CBF_CCC": 0.9958, "ATT_RMSE": 0.4951, "ATT_MAE": 0.3259, "ATT_CCC": 0.6899},
    },
}

rows = []
for snr in [10.0, 15.0]:
    dnn = snr_df[snr_df['SNR'] == snr].iloc[0]
    
    rows.append({
        "SNR": snr,
        "Method": "Baseline DNN",
        "CBF RMSE": dnn['CBF_RMSE'], "CBF MAE": dnn['CBF_MdAE'], "CBF MdAE": dnn['CBF_MdAE'], "CBF MdB": dnn['CBF_MdB'],
        "ATT RMSE": dnn['ATT_RMSE'], "ATT MAE": dnn['ATT_MdAE'], "ATT MdAE": dnn['ATT_MdAE'], "ATT MdB": dnn['ATT_MdB'],
        "Failure Rate": max(dnn['CBF_Fail'], dnn['ATT_Fail'])
    })
    
    bayes = EXTERNAL_REFERENCE[snr]["Bayesian"]
    rows.append({
        "SNR": snr,
        "Method": "Bayesian",
        "CBF RMSE": bayes['CBF_RMSE'], "CBF MAE": bayes['CBF_MAE'], "CBF MdAE": "N/A", "CBF MdB": "N/A",
        "ATT RMSE": bayes['ATT_RMSE'], "ATT MAE": bayes['ATT_MAE'], "ATT MdAE": "N/A", "ATT MdB": "N/A",
        "Failure Rate": "N/A"
    })
    
    nls = EXTERNAL_REFERENCE[snr]["NLS"]
    rows.append({
        "SNR": snr,
        "Method": "NLLS",
        "CBF RMSE": nls['CBF_RMSE'], "CBF MAE": nls['CBF_MAE'], "CBF MdAE": "N/A", "CBF MdB": "N/A",
        "ATT RMSE": nls['ATT_RMSE'], "ATT MAE": nls['ATT_MAE'], "ATT MdAE": "N/A", "ATT MdB": "N/A",
        "Failure Rate": "N/A"
    })

comparison_df = pd.DataFrame(rows)
comparison_df.to_csv("baseline_comparison.csv", index=False)
print("Baseline comparison generated.")
print(comparison_df.to_string())

