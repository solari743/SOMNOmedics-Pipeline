# Syncs motion data with sleep stages, calculates stats (CoV, skewness), 
# and provides an interactive plot of motion intensity across sleep epochs.

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
from scipy.stats import skew, variation
import re
from matplotlib.widgets import Slider


# === 1. CONFIG ===
# Please ensure these file paths are correct for your system
sleep_file = r'c:\Users\kevin\OneDrive\Desktop\SOMNOmedics\Patient-Data\DATASET - HIGH RISK 02\High_Risk02 [CHILD TEMPLATE]\Sleep profile - HighRisk.txt'
target_ID = 'XI-016162' # Sensor ID for the Head
filename = r'c:\Users\kevin\OneDrive\Desktop\SOMNOmedics\Patient-Data\DATASET - HIGH RISK 02\High_Risk02 [RAW MOVEMENT]\highrisk02_20240820-141937.h5'


# === 2. DATA LOADING AND PROCESSING ===
date_match = re.search(r'(\d{8})', filename)
if not date_match:
    raise ValueError("Could not parse date from H5 filename.")
file_date = datetime.datetime.strptime(date_match.group(1), '%Y%m%d').date()

# Load sleep stages from the text file
sleep_data = []
with open(sleep_file, 'r') as f:
    for line in f:
        if ';' not in line: continue
        time_part, state = line.strip().split(';')
        try:
            time_obj = datetime.datetime.strptime(time_part.strip(), '%H:%M:%S,%f').time()
            full_time = datetime.datetime.combine(file_date, time_obj)
            sleep_data.append((full_time, state.strip()))
        except ValueError: continue
sleep_df = pd.DataFrame(sleep_data, columns=['timestamp', 'state']).set_index('timestamp').sort_index()

# Load motion data from the H5 file
with h5py.File(filename, 'r') as f:
    if target_ID not in f['Sensors']: raise KeyError(f"Sensor ID {target_ID} not found.")
    base_path = f'Sensors/{target_ID}'
    if 'Accelerometer' not in f[base_path] or 'Time' not in f[base_path]:
        raise KeyError(f"Missing Accelerometer or Time data for Sensor {target_ID}")

    acc_data = np.array(f[f'{base_path}/Accelerometer'][:], dtype=np.float64)
    time_raw = np.array(f[f'{base_path}/Time'][:], dtype=np.float64)
    time_dt = [datetime.datetime.fromtimestamp(t * 1e-6) for t in time_raw]
    
    motion_intensity = np.sqrt(np.sum(np.diff(acc_data, axis=0) ** 2, axis=1))
    df = pd.DataFrame({'timestamp': time_dt[1:], 'motion_intensity': motion_intensity}).set_index('timestamp')
    df['smoothed'] = df['motion_intensity'].rolling(window=5, center=True).mean().dropna()

# Create 30-second epochs and merge with sleep data
epoch_features = df['smoothed'].resample('30s').agg(['mean', 'std', 'min', 'max']).dropna()
final_df = pd.merge_asof(epoch_features, sleep_df, left_index=True, right_index=True, direction='backward')
final_df.rename(columns={'state': 'sleep_state'}, inplace=True)
final_df['sleep_state'] = final_df['sleep_state'].fillna('Unknown')

# Synchronize start time to the first available sleep stage
if not sleep_df.empty:
    first_sleep_timestamp = sleep_df.index[0]
    df = df.loc[df.index >= first_sleep_timestamp]
    final_df = final_df.loc[final_df.index >= first_sleep_timestamp]

if final_df.empty:
    raise ValueError("No overlapping data found between motion file and sleep profile after synchronization.")


# === 3. STATISTICS CALCULATION ===
# Define the start and end of the synchronized analysis window
analysis_start_time = final_df.index[0]
analysis_end_time = df.index[-1]

# --- Motion Statistics ---
motion_stats_data = final_df['mean'].dropna()
cov = variation(motion_stats_data)
skewness_val = skew(motion_stats_data)

# --- Sleep Stage Statistics ---
stats_df = sleep_df.copy()
stats_df['end_time'] = stats_df.index.to_series().shift(-1).fillna(analysis_end_time)
stats_df['clipped_start'] = stats_df.index.to_series().clip(lower=analysis_start_time)
stats_df['clipped_end'] = stats_df['end_time'].clip(upper=analysis_end_time)
stats_df['effective_duration'] = stats_df['clipped_end'] - stats_df['clipped_start']

stage_durations = stats_df.groupby('state')['effective_duration'].sum()
stage_durations = stage_durations[stage_durations > pd.Timedelta(0)]
total_analyzed_duration = stage_durations.sum()
stage_percentages = (stage_durations / total_analyzed_duration) * 100
total_hours = total_analyzed_duration.total_seconds() / 3600

# --- Print All Results --- 📊
print("\n--- Analysis Statistics ---")
print(f"Time Period: {total_hours:.2f} hours (from {analysis_start_time.strftime('%H:%M')} to {analysis_end_time.strftime('%H:%M')})")
print("\nMotion Summary (per 30s epoch):")
print(f"- Coefficient of Variation: {cov:.3f}")
print(f"- Skewness:                 {skewness_val:.3f}")
print("\nSleep Stage Distribution:")
for stage, percentage in stage_percentages.items():
    print(f"- {stage:<10}: {percentage:>5.2f}%")
print("---------------------------\n")


# === 4. INTERACTIVE PLOTTING WITH SLIDER ===
fig, ax_detail = plt.subplots(1, 1, figsize=(15, 7))
plt.subplots_adjust(bottom=0.2)

detail_line, = ax_detail.plot([], [], lw=2, color='royalblue', label='Smoothed Motion Intensity')
ax_detail.set_title("Detailed View of 30-Second Epoch with Sleep Stage")
ax_detail.set_xlabel("Time")
ax_detail.set_ylabel("Smoothed Intensity")
ax_detail.grid(True)
ax_detail.legend(loc='upper left')

cmap = {'Wake': 'orange', 'N1': 'skyblue', 'N2': 'dodgerblue', 'N3': 'navy', 'Rem': 'darkviolet', 'A': 'grey', 'Artifact': 'red', 'Unknown': 'lightgray'}
detail_stage_artists = []
sleep_df_with_end = sleep_df.copy()
sleep_df_with_end['end_time'] = sleep_df_with_end.index.to_series().shift(-1).fillna(pd.Timestamp.max)

def update(epoch_index_float):
    global detail_stage_artists
    epoch_index = int(epoch_index_float)
    
    start_time = final_df.index[epoch_index]
    end_time = start_time + pd.Timedelta('30s')
    
    epoch_data = df.loc[start_time:end_time]['smoothed']
    if not epoch_data.empty:
        detail_line.set_data(epoch_data.index, epoch_data.values)
        ax_detail.set_xlim(start_time, end_time)
        ax_detail.set_ylim(max(0, epoch_data.min() * 0.9), epoch_data.max() * 1.1)
        ax_detail.set_title(f"Detailed View - Epoch Starting {start_time.strftime('%H:%M:%S')}")
    else:
        detail_line.set_data([], [])
        ax_detail.set_title(f"Detailed View - No Data in Epoch Starting {start_time.strftime('%H:%M:%S')}")

    for artist in detail_stage_artists:
        artist.remove()
    detail_stage_artists.clear()

    overlapping_stages = sleep_df_with_end[
        (sleep_df_with_end.index < end_time) & (sleep_df_with_end['end_time'] > start_time)
    ]
    for stage_ts, stage_row in overlapping_stages.iterrows():
        stage_state = stage_row['state']
        span_start = max(start_time, stage_ts)
        span_end = min(end_time, stage_row['end_time'])
        
        span = ax_detail.axvspan(span_start, span_end, color=cmap.get(stage_state, 'lightgray'), alpha=0.3, ec='none', zorder=0)
        detail_stage_artists.append(span)

        text_x = span_start + (span_end - span_start) / 2
        text_y = ax_detail.get_ylim()[1] * 0.95
        text = ax_detail.text(text_x, text_y, stage_state, ha='center', va='top', fontsize=12, weight='bold')
        detail_stage_artists.append(text)
    
    fig.canvas.draw_idle()

ax_slider = fig.add_axes([0.15, 0.05, 0.75, 0.03])
slider = Slider(
    ax=ax_slider, label='Epoch', valmin=0,
    valmax=len(final_df) - 1,
    valinit=0, valstep=1
)
slider.on_changed(update)

update(0)
plt.show()