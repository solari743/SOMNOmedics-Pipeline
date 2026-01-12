"""
Script to analyze head movement patterns using accelerometer data from an HDF5 file.
It processes data from a head-mounted sensor to calculate movement speed, normalize it, 
and compute statistical metrics such as Coefficient of Variation (CoV) and Skewness.

- Extracts accelerometer and timestamp data from the head sensor
- Calculates motion intensity from frame-to-frame acceleration changes
- Computes CoV and Skewness to assess variability in head movement
- Visualizes time-series plot and histogram of normalized speed
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
from scipy.stats import skew

filename = r'c:\Users\kevin\OneDrive\Desktop\DATA - POSTER\DATASET - LOW RISK 02\Low_Risk02 [RAW MOVEMENT]\lowrisk02.h5'
target_ID = '16162'  

with h5py.File(filename, 'r') as f:
    if target_ID in f['Sensors']:
        base_path = f'Sensors/{target_ID}'

        if 'Accelerometer' in f[base_path] and 'Time' in f[base_path]:
            acc_data = np.array(f[f'{base_path}/Accelerometer'][:], dtype=np.float64)
            time_raw = np.array(f[f'{base_path}/Time'][:], dtype=np.float64)

            time_dt = [datetime.datetime.fromtimestamp(t * 1e-6) for t in time_raw]

            motion_intensity = np.sqrt(np.sum(np.diff(acc_data, axis=0) ** 2, axis=1))
            time_motion = time_dt[1:]  

            mean_intensity = np.mean(motion_intensity)
            normalized_motion = motion_intensity / mean_intensity if mean_intensity != 0 else motion_intensity

            cov = np.std(normalized_motion) / np.mean(normalized_motion)
            skewness_val = skew(normalized_motion)

            print(f"\nSummary Statistics for Sensor {target_ID}:")
            print(f" Coefficient of Variation (CoV): {cov:.3f}")
            print(f" Skewness: {skewness_val:.3f}\n")

            df = pd.DataFrame({
                'timestamp': time_motion,
                'motion_intensity': normalized_motion
            })
            df.set_index('timestamp', inplace=True)

            df['smoothed'] = df['motion_intensity'].rolling(window=5, center=True).mean()

            plt.figure(figsize=(12, 4))
            plt.plot(df.index, df['smoothed'], color='red', label='Smoothed Motion (5 pt MA)')
            plt.title(f'Low Risk 02 -Smoothed Motion Intensity Over Time - Head Sensor {target_ID}')
            plt.xlabel('Time')
            plt.ylabel('Smoothed Motion Intensity')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.gcf().autofmt_xdate()
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        else:
            print(f"Missing Accelerometer or Time data for Sensor {target_ID}")
    else:
        print(f"Sensor ID {target_ID} not found in the HDF5 file.")