# ===============================================================
# CONTINUOUS SLEEP SKEWNESS PLOT (With Sleep State Legend)
# ===============================================================
# 1. Cleans accelerometer data (clip, detrend, low-pass)
# 2. Computes skewness (30s epochs)
# 3. Aligns with sleep file and isolates Sleep periods
# 4. Plots continuous timeline of sleep data:
#    - Color-coded spans = Sleep segments by state
#    - Legend = NREM / REM / Transitional
# ===============================================================

import pandas as pd, matplotlib.pyplot as plt, datetime, re, h5py, numpy as np
from scipy.stats import skew
from scipy.signal import butter, filtfilt
from matplotlib.patches import Patch

# === 1. CONFIG ===
filename  = r'Patient-Data/DATASET - HIGH RISK 02/High_Risk02 [RAW MOVEMENT]/highrisk02_20240820-141937.h5'
targetID  = 'XI-016162'
sleep_file = r'Patient-Data/DATASET - HIGH RISK 02/High_Risk02 [CHILD TEMPLATE]/Sleep profile - HighRisk.txt'


# === 2. PARSE SLEEP START TIME ===
file_date = None
start_clock_time = None
with open(sleep_file, 'r', errors='ignore') as f:
    for line in f:
        if "Start Time" in line:
            match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})\s*(AM|PM)', line)
            if match:
                month, day, year, hour, minute, second, meridiem = match.groups()
                hour = int(hour)
                if meridiem.upper() == 'PM' and hour != 12: hour += 12
                if meridiem.upper() == 'AM' and hour == 12: hour = 0
                file_date = datetime.date(int(year), int(month), int(day))
                start_clock_time = datetime.time(hour, int(minute), int(second))
                break
if file_date is None:
    name_match = re.search(r'(\d{8})', sleep_file)
    if name_match:
        file_date = datetime.datetime.strptime(name_match.group(1), '%Y%m%d').date()
if file_date is None:
    raise ValueError("Recording date not found.")
sleep_start_dt = datetime.datetime.combine(file_date, start_clock_time or datetime.time(0,0,0))


# === 3. LOAD SLEEP STATES ===
sleep_data = []
with open(sleep_file, 'r', errors='ignore') as f:
    for line in f:
        if ';' not in line: continue
        time_part, state = line.strip().split(';')
        try:
            time_obj = datetime.datetime.strptime(time_part.strip(), '%H:%M:%S,%f').time()
            full_time = datetime.datetime.combine(file_date, time_obj)
            sleep_data.append((full_time, state.strip()))
        except ValueError: continue

sleep_df = pd.DataFrame(sleep_data, columns=['timestamp','state']).set_index('timestamp').sort_index()
if sleep_df.empty: raise ValueError("No sleep data found.")


# === 4. LOAD & CLEAN ACCELEROMETER DATA ===
with h5py.File(filename,'r') as f:
    base=f"Sensors/{targetID}"
    acc_data=np.array(f[f'{base}/Accelerometer'][:],dtype=np.float64)
    time_raw=np.array(f[f'{base}/Time'][:],dtype=np.float64)
    time_dt=np.array([datetime.datetime.fromtimestamp(t*1e-6) for t in time_raw])

acc_df=pd.DataFrame(acc_data,columns=['ax','ay','az'],index=pd.to_datetime(time_dt))

# --- Cleaning Steps ---
acc_df['mag']=np.sqrt(acc_df['ax']**2+acc_df['ay']**2+acc_df['az']**2)
acc_df['mag']=acc_df['mag'].clip(lower=acc_df['mag'].quantile(0.01),
                                 upper=acc_df['mag'].quantile(0.99))
acc_df['mag']=acc_df['mag']-acc_df['mag'].rolling(window=250,center=True,min_periods=1).mean()

def butter_lowpass_filter(data, cutoff=3, fs=50, order=4):
    nyq=0.5*fs; normal_cutoff=cutoff/nyq
    b,a=butter(order,normal_cutoff,btype='low',analog=False)
    return filtfilt(b,a,data)
acc_df['mag']=butter_lowpass_filter(acc_df['mag'].interpolate(),cutoff=3,fs=50)
acc_df.dropna(inplace=True)


# === 5. COMPUTE SKEWNESS (30s Epochs) ===
skew_30s = acc_df['mag'].resample('30s').apply(lambda x: np.nan if len(x)<3 else skew(x, nan_policy='omit')).dropna()


# === 6. DETERMINE SLEEP SEGMENTS ===
sleep_states = ['nrem', 'rem', 'transitional']
sleep_colors = {'nrem':'#1f77b4', 'rem':'#2ca02c', 'transitional':'#9467bd'}  # blue, green, purple

sleep_only_df = sleep_df[sleep_df['state'].str.lower().isin(sleep_states)]
if sleep_only_df.empty:
    raise ValueError("No sleep-related states (NREM, REM, Transitional) found in sleep profile.")


# === 7. VISUALIZE ONE CONTINUOUS TIMELINE ===
fig, ax = plt.subplots(figsize=(15,7))

# Plot skewness over time
ax.plot(skew_30s.index, skew_30s.values, color='magenta', lw=1.5, label='Skewness (unitless)')

# Overlay color-coded Sleep spans
for state, color in sleep_colors.items():
    state_df = sleep_only_df[sleep_only_df['state'].str.lower() == state]
    if state_df.empty: continue
    state_df = state_df.copy()
    state_df['gap'] = (state_df.index.to_series().diff() > pd.Timedelta('40s')).cumsum()
    state_blocks = (
        state_df.groupby('gap')
                .agg(start_time=('gap', lambda x: x.index.min()),
                     end_time=('gap', lambda x: x.index.max()))
                .reset_index(drop=True)
    )
    for _, row in state_blocks.iterrows():
        ax.axvspan(row['start_time'], row['end_time'], color=color, alpha=0.25)

# Custom legend
legend_elements = [Patch(facecolor=color, alpha=0.25, label=state.upper()) for state, color in sleep_colors.items()]
legend_elements.append(Patch(facecolor='magenta', alpha=0.5, label='Skewness'))
ax.legend(handles=legend_elements, loc='upper right')

ax.set_title("Continuous Skewness of Motion Magnitude — Sleep States Highlighted", fontsize=14, weight='bold')
ax.set_xlabel("Time (HH:MM:SS)")
ax.set_ylabel("Skewness (unitless)")
ax.grid(True)

plt.tight_layout()
plt.show()


# === 8. SUMMARY ===
total_duration = (skew_30s.index[-1] - skew_30s.index[0])
print("\n--- Sleep Skewness Summary ---")
print(f"Total analyzed duration: {total_duration}")

for state in sleep_states:
    count = (sleep_only_df['state'].str.lower() == state).sum()
    print(f"{state.upper()} epochs: {count}")

print("------------------------------\n")
