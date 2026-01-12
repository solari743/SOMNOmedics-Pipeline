"""
Script to visualize raw accelerometer data from a head-mounted sensor using data 
stored in an HDF5 file. It downsamples the data for clarity and plots the X, Y, 
and Z acceleration components over time.

- Loads accelerometer and timestamp data from the head sensor
- Converts microsecond timestamps to readable datetime format
- Samples data to reduce plot density
- Plots acceleration signals (X, Y, Z) over time
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates

filename = r'C:\Users\SOMNOmedics\Desktop\DATASET - HIGH RISK 02\High_Risk02 [RAW MOVEMENT]\highrisk02_20240820-141937.h5'
target_ID = 'XI-016162'

with h5py.File(filename, 'r') as f:
    if target_ID in f['Sensors']:
        base_path = f'Sensors/{target_ID}'

        if 'Accelerometer' in f[base_path] and 'Time' in f[base_path]:
            acc_data = f[f'{base_path}/Accelerometer'][:]
            time_raw = f[f'{base_path}/Time'][:]

            time_dt = [datetime.datetime.fromtimestamp(t * 1e-6) for t in time_raw]
            
            step = 500  
            time_sampled = time_dt[::step]
            acc_sampled = acc_data[::step]

            plt.figure(figsize=(12, 6))
            plt.plot(time_sampled, acc_sampled[:, 0], label='X')
            plt.plot(time_sampled, acc_sampled[:, 1], label='Y')
            plt.plot(time_sampled, acc_sampled[:, 2], label='Z')

            plt.title(f'Accelerometer Data - Sensor {target_ID}')
            plt.xlabel('Time (HH:MM:SS)')
            plt.ylabel('Acceleration (m/s²)')

            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.gcf().autofmt_xdate()

            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print(f"Missing Accelerometer or Time data for Sensor {target_ID}")
