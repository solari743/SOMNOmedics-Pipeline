import h5py
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os

filename = r'c:\Users\kevin\OneDrive\Desktop\DATA - POSTER\DATASET - LOW RISK 02\Low_Risk02 [RAW MOVEMENT]\lowrisk02.h5'
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

    def bowley_skew(x):
        """Calculates the Bowley-Galton skewness coefficient."""
        if len(x) < 4:
            return np.nan
        q1 = np.percentile(x, 25)
        q2 = np.percentile(x, 50) 
        q3 = np.percentile(x, 75)
        denominator = q3 - q1
        if denominator == 0:
            return 0  
        return (q3 + q1 - 2 * q2) / denominator

    skewness_df = df['acc_magnitude'].resample('30s').apply(bowley_skew).dropna().to_frame(name='bowley_skew')

    if skewness_df.empty:
        raise ValueError("No data available to calculate skewness.")

    print("\n--- Bowley-Galton Skewness of Acceleration Magnitude (30s Epochs) ---")
    print(skewness_df.head())

    output_csv_name = f"{os.path.splitext(filename)[0]}_bowley_skew.csv"
    skewness_df.to_csv(output_csv_name)
    print(f"\nSaved skewness data to {output_csv_name}")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))

    plt.plot(skewness_df.index, skewness_df['bowley_skew'], label='Bowley Skewness', color='purple', marker='.', linestyle='-')

    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Symmetric (Skew = 0)')

    plt.title('Bowley-Galton Skewness Over Time (30s Epochs)', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Skewness Coefficient', fontsize=12)
    plt.legend()
    plt.tight_layout()

    output_plot_name = f"{os.path.splitext(filename)[0]}_bowley_skew_plot.png"
    plt.savefig(output_plot_name)
    print(f"Successfully generated and saved {output_plot_name}")