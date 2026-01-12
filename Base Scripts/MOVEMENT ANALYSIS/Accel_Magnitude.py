import h5py
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

filename = r'c:\Users\kevin\OneDrive\Desktop\DATA - POSTER\DATASET - HIGH RISK 02\High_Risk02 [RAW MOVEMENT]\highrisk02_20240820-141937.h5'
target_ID = 'XI-016162'  

if not os.path.exists(filename):
    print(f"File '{filename}' not found in the current directory.")
    print("Please upload your H5 file. The script will proceed assuming the file will be available.")
    filename = None 

if filename:
    with h5py.File(filename, 'r') as f:
        if target_ID not in f['Sensors']:
            raise KeyError(f"Sensor ID {target_ID} not found in the file. Available IDs: {list(f['Sensors'].keys())}")

        base_path = f'Sensors/{target_ID}'
        if 'Accelerometer' not in f[base_path] or 'Time' not in f[base_path]:
            raise KeyError(f"Missing Accelerometer or Time data for Sensor {target_ID}")

        acc_data = np.array(f[f'{base_path}/Accelerometer'][:], dtype=np.float64)
        time_raw = np.array(f[f'{base_path}/Time'][:], dtype=np.float64)

        # Convert microseconds to datetime
        time_dt = [datetime.datetime.fromtimestamp(t * 1e-6) for t in time_raw]

    # Compute acceleration magnitude
    acc_magnitude = np.linalg.norm(acc_data, axis=1)
    df = pd.DataFrame({'timestamp': time_dt, 'acc_magnitude': acc_magnitude}).set_index('timestamp')

    # Resample into 30-second epochs
    epoch_features = df['acc_magnitude'].resample('30s').agg(['mean', 'std', 'min', 'max']).dropna()

    if epoch_features.empty:
        raise ValueError("No data available to calculate epoch statistics.")

    print("\n--- Acceleration Magnitude Statistics ---")
    print(epoch_features.head())

    # Save to CSV
    output_csv_name = f"{os.path.splitext(filename)[0]}_acc_magnitude_per_epoch.csv"
    epoch_features.to_csv(output_csv_name)
    print(f"\nSaved epoch statistics to {output_csv_name}")

    # Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))

    plt.plot(epoch_features.index, epoch_features['mean'], label='Mean Acceleration', color='royalblue')

    plt.fill_between(epoch_features.index, 
                     epoch_features['mean'] - epoch_features['std'], 
                     epoch_features['mean'] + epoch_features['std'], 
                     color='lightblue', 
                     alpha=0.5, 
                     label='Standard Deviation')

    plt.title('Mean Acceleration Magnitude Over Time', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Acceleration Magnitude', fontsize=12)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    plt.legend()

    output_plot_name = f"{os.path.splitext(filename)[0]}_acceleration_magnitude_plot.png"
    plt.tight_layout()
    plt.savefig(output_plot_name)

    print(f"Successfully generated and saved {output_plot_name}")
