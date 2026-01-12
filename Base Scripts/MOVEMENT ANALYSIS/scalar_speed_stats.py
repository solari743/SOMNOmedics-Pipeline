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

    time_sec = (time_raw - time_raw[0]) * 1e-6 
    delta_t = np.gradient(time_sec)

    acc_mag = np.linalg.norm(acc_data, axis=1)

    speed_est = np.cumsum(acc_mag * delta_t)

    df = pd.DataFrame({'timestamp': time_dt, 'speed_est': speed_est}).set_index('timestamp')

    epoch_features = df['speed_est'].resample('30s').agg(['mean', 'std', 'min', 'max']).dropna()

    if epoch_features.empty:
        raise ValueError("No data available to calculate epoch statistics.")

    print("\n--- Estimated Scalar Speed Stats (Per 30s Epoch) ---")
    print(epoch_features.head())

    output_csv_name = f"{os.path.splitext(filename)[0]}_speed_stats.csv"
    epoch_features.to_csv(output_csv_name)
    print(f"\nSaved speed statistics to {output_csv_name}")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))

    plt.plot(epoch_features.index, epoch_features['mean'], label='Mean Estimated Speed', color='darkorange')

    plt.fill_between(epoch_features.index,
                     epoch_features['min'],
                     epoch_features['max'],
                     color='moccasin',
                     alpha=0.5,
                     label='Min-Max Range')

    plt.title('Estimated Scalar Speed Over Time (30s Epochs)', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Estimated Speed (m/s)', fontsize=12)
    plt.legend()
    plt.tight_layout()

    output_plot_name = f"{os.path.splitext(filename)[0]}_speed_plot.png"
    plt.savefig(output_plot_name)
    print(f"Successfully generated and saved {output_plot_name}")