import h5py
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from scipy.stats import skew

filename = r'c:\Users\kevin\OneDrive\Desktop\DATA - POSTER\DATASET - HIGH RISK 02\High_Risk02 [RAW MOVEMENT]\highrisk02_20240820-141937.h5'
target_ID = 'XI-016162'  

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

    def compute_skew(x):
        return skew(x) if len(x) >= 3 else np.nan

    skewness_df = df['acc_magnitude'].resample('30s').apply(compute_skew).dropna().to_frame(name='skewness')

    if skewness_df.empty:
        raise ValueError("No data available to calculate skewness.")

    print("\n--- Skewness of Acceleration Magnitude ---")
    print(skewness_df.head())

    output_csv_name = f"{os.path.splitext(filename)[0]}_skewness.csv"
    skewness_df.to_csv(output_csv_name)
    print(f"\nSaved skewness data to {output_csv_name}")

    # Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))

    plt.plot(skewness_df.index, skewness_df['skewness'], label='Skewness', color='teal', marker='.', linestyle='-')

    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Symmetric (Skew = 0)')

    plt.title('Skewness of Acceleration Magnitude Over Time', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Skewness Coefficient', fontsize=12)

    # ✅ Format x-axis to only show time
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    plt.legend()
    plt.tight_layout()

    output_plot_name = f"{os.path.splitext(filename)[0]}_skewness_plot.png"
    plt.savefig(output_plot_name)
    print(f"Successfully generated and saved {output_plot_name}")
