# ===============================================================
# CONTINUOUS + COMPRESSED SLEEP COV PLOTS
# ===============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime, re, h5py, numpy as np
from scipy.signal import butter, filtfilt
from matplotlib.patches import Patch


# Low Risk Data
filename  = r'Patient-Data/DATASET - LOW RISK 02/Low_Risk02 [RAW MOVEMENT]/Low Risk 02.h5'
targetID  = '16162'
sleep_file = r'Patient-Data/DATASET - LOW RISK 02/Low_Risk02 [CHILD TEMPLATE]/Sleep profile [EDF BASED].txt'

# High Risk Data
filename  = r'Patient-Data/DATASET - HIGH RISK 02/High_Risk02 [RAW MOVEMENT]/highrisk02_20240820-141937.h5'
targetID  = 'XI-016162'
sleep_file = r'Patient-Data/DATASET - HIGH RISK 02/High_Risk02 [CHILD TEMPLATE]/Sleep profile - HighRisk.txt'


file_date = None
start_clock_time = None

with open(sleep_file, 'r', errors='ignore') as f:
    for line in f:
        if "Start Time" in line:
            match = re.search(
                r'(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})\s*(AM|PM)',
                line
            )
            if match:
                month, day, year, hour, minute, second, meridiem = match.groups()
                hour = int(hour)
                if meridiem.upper() == 'PM' and hour != 12:
                    hour += 12
                if meridiem.upper() == 'AM' and hour == 12:
                    hour = 0
                file_date = datetime.date(int(year), int(month), int(day))
                start_clock_time = datetime.time(hour, int(minute), int(second))
                break

if file_date is None:
    name_match = re.search(r'(\d{8})', sleep_file)
    if name_match:
        file_date = datetime.datetime.strptime(name_match.group(1), '%Y%m%d').date()

if file_date is None:
    raise ValueError("Recording date not found.")

sleep_start_dt = datetime.datetime.combine(
    file_date,
    start_clock_time or datetime.time(0, 0, 0)
)

sleep_data = []

with open(sleep_file, 'r', errors='ignore') as f:
    for line in f:
        if ';' not in line:
            continue
        time_part, state = line.strip().split(';')
        try:
            time_obj = datetime.datetime.strptime(
                time_part.strip(), '%H:%M:%S,%f'
            ).time()
            full_time = datetime.datetime.combine(file_date, time_obj)
            sleep_data.append((full_time, state.strip()))
        except ValueError:
            continue

sleep_df = (
    pd.DataFrame(sleep_data, columns=['timestamp','state'])
      .set_index('timestamp')
      .sort_index()
)

if sleep_df.empty:
    raise ValueError("No sleep data found.")

sleep_df['state_norm'] = sleep_df['state'].str.lower().str.strip()

with h5py.File(filename,'r') as f:
    base = f"Sensors/{targetID}"
    acc_data = np.array(f[f'{base}/Accelerometer'][:], dtype=np.float64)
    time_raw = np.array(f[f'{base}/Time'][:], dtype=np.float64)
    time_dt  = np.array([
        datetime.datetime.fromtimestamp(t * 1e-6)
        for t in time_raw
    ])

acc_df = pd.DataFrame(
    acc_data,
    columns=['ax','ay','az'],
    index=pd.to_datetime(time_dt)
)

acc_df['mag'] = np.sqrt(
    acc_df['ax']**2 +
    acc_df['ay']**2 +
    acc_df['az']**2
)

acc_df['mag'] = acc_df['mag'].clip(
    lower=acc_df['mag'].quantile(0.01),
    upper=acc_df['mag'].quantile(0.99)
)

acc_df['mag'] -= acc_df['mag'].rolling(
    window=250, center=True, min_periods=1
).mean()

def butter_lowpass_filter(data, cutoff=3, fs=50, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)

acc_df['mag'] = butter_lowpass_filter(
    acc_df['mag'].interpolate(),
    cutoff=3,
    fs=50
)

acc_df.dropna(inplace=True)

def cov_func(x):
    mean_x = x.mean()
    return np.nan if mean_x == 0 else x.std() / mean_x

cov_30s = (
    acc_df['mag']
    .resample('30s')
    .apply(cov_func)
    .dropna()
)

# Plot 1 - Sleep States
plot1_states = [
    'wake',
    'nrem',
    'rem',
    'transitional',
    'movement',
    'a'  # file accurate abstract notation        
]

plot1_colors = {
    'wake': '#ff8c00',          
    'nrem': '#1f77b4',          
    'rem': '#2ca02c',           
    'transitional': '#9467bd',  
    'movement': '#8c564b',      
    'a': '#7f7f7f'              
}

plot1_df = sleep_df[sleep_df['state_norm'].isin(plot1_states)]


# Plot 2 - REM States only
sleep_states = ['nrem', 'rem', 'transitional']
sleep_colors = {
    'nrem':'#1f77b4',
    'rem':'#2ca02c',
    'transitional':'#9467bd'
}

sleep_only_df = sleep_df[sleep_df['state_norm'].isin(sleep_states)]

print(
    sleep_df['state']
    .str.lower()
    .value_counts()
)

print("\n=== START TIME CHECK ===")
print("Accel start:", acc_df.index.min())
print("Accel end:  ", acc_df.index.max())
print("CoV start:  ", cov_30s.index.min())
print("CoV end:    ", cov_30s.index.max())
print("Sleep start:", sleep_df.index.min())
print("Sleep end:  ", sleep_df.index.max())

offset = sleep_df.index.min() - cov_30s.index.min()
print("Sleep start - CoV start offset:", offset)

fig, ax = plt.subplots(figsize=(15,7))

sleep_df_filled = sleep_df.copy()

continuous_index = pd.date_range(
    start=max(sleep_df.index.min(), cov_30s.index.min()),
    end=min(sleep_df.index.max(), cov_30s.index.max()),
    freq='1s'
)

sleep_df_filled = sleep_df_filled.reindex(continuous_index, method='ffill')

sleep_df_filled['state_norm'] = sleep_df_filled['state_norm'].fillna(method='bfill')

for state, color in plot1_colors.items():
    state_mask = sleep_df_filled['state_norm'] == state
    state_indices = sleep_df_filled[state_mask].index
    
    if len(state_indices) == 0:
        continue
    
    state_series = pd.Series(state_indices)
    gaps = state_series.diff() > pd.Timedelta('2s')
    block_ids = gaps.cumsum()
    
    for block_id in block_ids.unique():
        block_times = state_indices[block_ids == block_id]
        ax.axvspan(
            block_times.min(),
            block_times.max() + pd.Timedelta('1s'),  # Add 1s to cover the full second
            color=color,
            alpha=0.25,
            zorder=0
        )

ax.plot(cov_30s.index, cov_30s.values, color='magenta', lw=1.5, zorder=2)

legend1 = (
    [Patch(facecolor=color, alpha=0.25, label=state.upper())
     for state, color in plot1_colors.items()] +
    [Patch(facecolor='magenta', alpha=0.5, label='CoV')]
)

ax.legend(handles=legend1, loc='upper right')
ax.set_title("Continuous Movement Variability (CoV) Across Sleep and Wake - Low Risk 02")

ax.set_xlabel("Time")
ax.set_ylabel("Coefficient of Variation")
ax.grid(True)

ax.set_xlim(cov_30s.index.min(), cov_30s.index.max())
ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

plt.tight_layout()
plt.show()


# Second Plot - Concatenated Time Series
fig2, ax2 = plt.subplots(figsize=(15,7))

sleep_labels = sleep_only_df['state'].reindex(
    cov_30s.index,
    method='nearest',
    tolerance=pd.Timedelta('40s')
)

sleep_mask = sleep_labels.str.lower().isin(sleep_states)
sleep_cov = cov_30s[sleep_mask].copy()
sleep_state_for_cov = sleep_labels[sleep_mask].str.lower()

df_blocks = pd.DataFrame({
    'state': sleep_state_for_cov,
    'original_time': sleep_cov.index
})

df_blocks['time_diff'] = df_blocks['original_time'].diff()
df_blocks['block_id'] = (
    (df_blocks['state'] != df_blocks['state'].shift()) |
    (df_blocks['time_diff'] > pd.Timedelta('40s'))
).cumsum()

time_mapping = {}
current_time = pd.Timedelta(0)

for block_id in sorted(df_blocks['block_id'].unique()):
    group = df_blocks[df_blocks['block_id'] == block_id]
    block_start = group['original_time'].min()
    block_duration = (
        group['original_time'].max() - block_start +
        pd.Timedelta('30s')
    )

    for t in group['original_time']:
        time_mapping[t] = current_time + (t - block_start)

    current_time += block_duration

compressed_times = [time_mapping[t] for t in sleep_cov.index]
sleep_cov_compressed = sleep_cov.copy()
sleep_cov_compressed.index = (
    pd.to_datetime(sleep_start_dt) +
    pd.to_timedelta(compressed_times)
)

ax2.plot(
    sleep_cov_compressed.index,
    sleep_cov_compressed.values,
    color='magenta',
    lw=1.5
)

for state in sleep_states:
    mask = sleep_state_for_cov == state
    if not mask.any():
        continue
    idx = sleep_cov_compressed.index[mask]
    ax2.axvspan(
        idx.min(),
        idx.max(),
        color=sleep_colors[state],
        alpha=0.25
    )

legend2 = (
    [Patch(facecolor=color, alpha=0.25, label=state.upper())
     for state, color in sleep_colors.items()] +
    [Patch(facecolor='magenta', alpha=0.5, label='CoV')]
)

ax2.legend(handles=legend2, loc='upper right')
ax2.set_title("REM Sleep Movement CoV Aligned to Pseudo-Timeline – Low Risk 02")
ax2.set_xlabel("Compressed Sleep Time")
ax2.set_ylabel("Coefficient of Variation")
ax2.grid(True)

ax2.set_xlim(
    sleep_cov_compressed.index.min(),
    sleep_cov_compressed.index.max()
)
ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

plt.tight_layout()
plt.show()

# Basic Analytics --> Add more analytics
print("\n========== ADVANCED CoV + SLEEP ANALYTICS ==========\n")

print("CoV STATISTICS:")
print(f"Mean:     {cov_30s.mean():.4f}")
print(f"Median:   {cov_30s.median():.4f}")
print(f"Std Dev:  {cov_30s.std():.4f}")
print(f"Variance: {cov_30s.var():.4f}")
print(f"Min:      {cov_30s.min():.4f}")
print(f"Max:      {cov_30s.max():.4f}")

sleep_start = sleep_only_df.index.min()
sleep_end   = sleep_only_df.index.max()
total_sleep_seconds = (sleep_end - sleep_start).total_seconds()
total_record_seconds = (cov_30s.index[-1] - cov_30s.index[0]).total_seconds()

print("\nOTHER METRICS:")
print(f"Total Sleep Window:   {total_sleep_seconds/3600:.2f} hrs")
print(f"Total Recording Time: {total_record_seconds/3600:.2f} hrs")
print(f"Sleep Efficiency:     {(total_sleep_seconds/total_record_seconds)*100:.1f}%")
print("\n=====================================================\n")