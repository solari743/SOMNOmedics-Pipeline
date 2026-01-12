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

    diff_mag = np.diff(acc_magnitude) 

    df = pd.DataFrame({
        'timestamp': time_dt[1:], 
        'diff_mag': diff_mag
    }).set_index('timestamp')

    def zero_crossings(x):
        """Counts the number of times a signal crosses the zero-axis."""
        return np.count_nonzero(np.diff(np.sign(x)))

    zcr_df = df['diff_mag'].resample('30s').apply(zero_crossings).dropna().to_frame(name='zero_crossing_rate')

    if zcr_df.empty:
        raise ValueError("No data available to calculate Zero-Crossing Rate.")

    print("\n--- Zero-Crossing Rate of Motion Change (30s Epochs) ---")
    print(zcr_df.head())

    output_csv_name = f"{os.path.splitext(filename)[0]}_zcr.csv"
    zcr_df.to_csv(output_csv_name)
    print(f"\nSaved ZCR data to {output_csv_name}")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))

    plt.plot(zcr_df.index, zcr_df['zero_crossing_rate'], label='ZCR', color='orangered', marker='|', markersize=5, linestyle='-')

    plt.title('Zero-Crossing Rate (Restlessness) Over Time (30s Epochs)', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Zero-Crossing Count per Epoch', fontsize=12)
    plt.legend()
    plt.tight_layout()

    output_plot_name = f"{os.path.splitext(filename)[0]}_zcr_plot.png"
    plt.savefig(output_plot_name)
    print(f"Successfully generated and saved {output_plot_name}")