# ===============================================================
# CONTINUOUS WAKE VELOCITY PLOT (No Slider)
# ===============================================================
# 1. Cleans accelerometer data (clip, detrend, low-pass)
# 2. Integrates acceleration magnitude to compute velocity
# 3. Resamples velocity into 30s means
# 4. Aligns with sleep file and isolates Wake periods
# 5. Plots continuous timeline of wake data
#    - Orange shaded spans = Wake segments
#    - Vertical dashed lines = Wake segment starts
#    - Labels = start clock times
# ===============================================================

import pandas as pd, matplotlib.pyplot as plt, datetime, re, h5py, numpy as np
from scipy.signal import butter, filtfilt

filename  = r'Patient-Data/DATASET - HIGH RISK 02/High_Risk02 [RAW MOVEMENT]/highrisk02_20240820-141937.h5'
targetID  = 'XI-016162'
sleep_file = r'Patient-Data/DATASET - HIGH RISK 02/High_Risk02 [CHILD TEMPLATE]/Sleep profile - HighRisk.txt'


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

wake_df = sleep_df[sleep_df['state'].str.lower() == 'wake']
if wake_df.empty: raise ValueError("No wake periods found in sleep profile.")


with h5py.File(filename,'r') as f:
    base=f"Sensors/{targetID}"
    acc_data=np.array(f[f'{base}/Accelerometer'][:],dtype=np.float64)
    time_raw=np.array(f[f'{base}/Time'][:],dtype=np.float64)
    time_dt=np.array([datetime.datetime.fromtimestamp(t*1e-6) for t in time_raw])

acc_df=pd.DataFrame(acc_data,columns=['ax','ay','az'],index=pd.to_datetime(time_dt))


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


dt = acc_df.index.to_series().diff().dt.total_seconds().median()
if pd.isna(dt) or dt == 0:
    dt = 1/50  

velocity = np.cumsum(acc_df['mag'] * dt)
vel_df = pd.Series(velocity, index=acc_df.index, name='velocity')

vel_30s = vel_df.resample('30s').mean().dropna()


wake_df = wake_df.copy()
wake_df['gap'] = (wake_df.index.to_series().diff() > pd.Timedelta('40s')).cumsum()
wake_blocks = (
    wake_df.groupby('gap')
           .agg(start_time=('gap', lambda x: x.index.min()),
                end_time=('gap', lambda x: x.index.max()))
           .reset_index(drop=True)
)


fig, ax = plt.subplots(figsize=(15,7))


ax.plot(vel_30s.index, vel_30s.values, color='red', lw=1.5, label='Mean Velocity (m/s)')


for _, row in wake_blocks.iterrows():
    start, end = row['start_time'], row['end_time']
    ax.axvspan(start, end, color='orange', alpha=0.2)

ax.set_title("Continuous Mean Velocity — Wake Segments Highlighted", fontsize=14, weight='bold')
ax.set_xlabel("Time (HH:MM:SS)")
ax.set_ylabel("Velocity (m/s)")
ax.legend(loc='upper right')
ax.grid(True)

plt.tight_layout()
plt.show()


total_duration = (vel_30s.index[-1] - vel_30s.index[0])
print("\n--- Wake Velocity Summary ---")
print(f"Total analyzed duration: {total_duration}")
print(f"Total Wake blocks: {len(wake_blocks)}")
for i, row in wake_blocks.iterrows():
    print(f"Block {i+1}: {row['start_time'].strftime('%H:%M:%S')} → {row['end_time'].strftime('%H:%M:%S')}")
print("------------------------------\n")
