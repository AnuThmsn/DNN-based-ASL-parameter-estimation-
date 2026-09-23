import pandas as pd
import glob

files = glob.glob("experiments_G/improvements_*.csv")
dfs = []
for f in files:
    dfs.append(pd.read_csv(f))

if dfs:
    res_df = pd.concat(dfs, ignore_index=True)
    
    master = pd.DataFrame(columns=['Experiment', 'CBF RMSE', 'ATT RMSE', 'CBF MAE', 'ATT MAE'])
    snr_df = pd.read_csv("experiments_G/diagnostics/snr_curves.csv")
    b_snr10 = snr_df[snr_df['SNR'] == 10].iloc[0]
    master.loc[0] = ['BASELINE-G', b_snr10['CBF RMSE'], b_snr10['ATT RMSE'], b_snr10['CBF MAE'], b_snr10['ATT MAE']]
    
    for i, row in res_df.iterrows():
        master.loc[len(master)] = [row['Experiment'], row['CBF RMSE'], row['ATT RMSE'], row['CBF MAE'], row['ATT MAE']]
    
    try:
        pld_df = pd.read_csv("experiments_G/reduced_pld.csv")
        for i, row in pld_df.iterrows():
            master.loc[len(master)] = [row['Experiment'], row['CBF RMSE'], row['ATT RMSE'], row['CBF MAE'], row['ATT MAE']]
    except:
        pass
        
    master.to_csv("experiments_G/master_results_G.csv", index=False)
    print("Master table updated.")
