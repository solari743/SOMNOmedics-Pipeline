import h5py
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os

filename = r'c:\Users\SOMNOmedics\Desktop\DATA - POSTER\DATASET - LOW RISK 02\Low_Risk02 [RAW MOVEMENT]\Low Risk 02.h5'
target_ID = '16162'  

if not os.path.exists(filename):
    print(f"File '{filename}' not found in the current directory.")
    print("Please upload your H5 file to proceed.")
    filename = None 

if filename:

    with h5py.File(filename, 'r') as f:
        if target_ID not in f['Sensors']:
            raise KeyError(f"Sensor ID {target_ID} not found. Available IDs: {list(f['Sensors'].keys())}")

        base_path = f'Sensors/{target_ID}'
        if 'Accelerometer' not in f[base_path] or 'Time' not in f[base_path]:
            raise KeyError(f"Missing Accelerometer or Time data for Sensor {target_ID}")

        acc_data = np.array(f[f'{base_path}/Accelerometer'][:], dtype=np.float64)
        time_raw = np.array(f[f'{base_path}/Time'][:], dtype=np.float64)
        time_dt = np.array([datetime.datetime.fromtimestamp(t * 1e-6) for t in time_raw])

    acc_magnitude = np.linalg.norm(acc_data, axis=1)
    df = pd.DataFrame({'timestamp': time_dt, 'acc_magnitude': acc_magnitude}).set_index('timestamp')

    epoch_stats = df['acc_magnitude'].resample('30s').agg(['mean', 'std']).dropna()

    epoch_stats['cov'] = (epoch_stats['std'] / epoch_stats['mean']).replace([np.inf, -np.inf], 0)

    cov_df = epoch_stats[['cov']].dropna()

    if cov_df.empty:
        raise ValueError("No data available to calculate Coefficient of Variation.")

    print("\n--- Coefficient of Variation (CoV) per 30s Epoch ---")
    print(cov_df.head())

    output_csv_name = f"{os.path.splitext(filename)[0]}_cov.csv"
    cov_df.to_csv(output_csv_name)
    print(f"\nSaved CoV data to {output_csv_name}")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))

    plt.plot(cov_df.index, cov_df['cov'], label='CoV', color='green', marker='o', markersize=3, linestyle='-')

    plt.title('Coefficient of Variation (CoV) Over Time (30s Epochs)', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Coefficient of Variation (Unitless)', fontsize=12)
    plt.legend()
    plt.tight_layout()

    output_plot_name = f"{os.path.splitext(filename)[0]}_cov_plot.png"
    plt.savefig(output_plot_name)
    print(f"Successfully generated and saved {output_plot_name}")