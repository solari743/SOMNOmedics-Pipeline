import pandas as pd
import matplotlib.pyplot as plt
import datetime, re, h5py, numpy as np
from matplotlib.widgets import Slider


filename   = r'Patient-Data/DATASET - HIGH RISK 02/High_Risk02 [RAW MOVEMENT]/highrisk02_20240820-141937.h5'
targetID   = 'XI-016162'
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
    raise ValueError("Recording date could not be parsed from sleep file.")
sleep_start_dt = datetime.datetime.combine(file_date, start_clock_time or datetime.time(0, 0, 0))


sleep_data = []
with open(sleep_file, 'r', errors='ignore') as f:
    for line in f:
        if ';' not in line:
            continue
        time_part, state = line.strip().split(';')
        try:
            time_obj = datetime.datetime.strptime(time_part.strip(), '%H:%M:%S,%f').time()
            full_time = datetime.datetime.combine(file_date, time_obj)
            sleep_data.append((full_time, state.strip()))
        except ValueError:
            continue

sleep_df = pd.DataFrame(sleep_data, columns=['timestamp', 'state']).set_index('timestamp').sort_index()
if sleep_df.empty:
    raise ValueError("No sleep state data found.")

analysis_start_time, analysis_end_time = sleep_df.index[0], sleep_df.index[-1]
epoch_range = pd.date_range(start=analysis_start_time, end=analysis_end_time, freq='30s')
epoch_df = pd.DataFrame(index=epoch_range)
epoch_df = pd.merge_asof(epoch_df, sleep_df, left_index=True, right_index=True, direction='backward')
epoch_df.rename(columns={'state': 'sleep_state'}, inplace=True)
epoch_df['sleep_state'] = epoch_df['sleep_state'].fillna('Unknown')

wake_df = sleep_df[sleep_df['state'].str.lower() == 'wake'].copy()


with h5py.File(filename, 'r') as f:
    if targetID not in f['Sensors']:
        raise KeyError(f"Sensor ID {targetID} not found in file")
    base_path = f"Sensors/{targetID}"
    acc_data = np.array(f[f'{base_path}/Accelerometer'][:], dtype=np.float64)
    time_raw = np.array(f[f'{base_path}/Time'][:], dtype=np.float64)
    time_dt = np.array([datetime.datetime.fromtimestamp(t * 1e-6) for t in time_raw])

acc_df = pd.DataFrame(acc_data, columns=['ax', 'ay', 'az'], index=pd.to_datetime(time_dt))
sig = acc_df['ax']  # you can switch to 'mag' if you want combined 3-axis activity


def zcr_window(x):
    if len(x) < 2:
        return np.nan
    x = np.sign(x)
    x[x == 0] = 1
    return np.mean(x[:-1] != np.roll(x, -1)[:-1])


sr_est = 1 / sig.index.to_series().diff().dt.total_seconds().median()
if not np.isfinite(sr_est) or sr_est == 0:
    sr_est = 50  # fallback

window_s = 5  # seconds per window
samples_per_window = int(sr_est * window_s)


zcr_series = sig.rolling(samples_per_window, min_periods=2).apply(zcr_window, raw=False)
zcr_series = zcr_series.resample('1s').mean().fillna(method='ffill')


fig, ax = plt.subplots(figsize=(15, 7))
plt.subplots_adjust(bottom=0.25)
ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
slider = Slider(ax_slider, 'Epoch', 0, len(epoch_df) - 1, valinit=0, valstep=1)

def update(i):
    i = int(i)
    start, end = epoch_df.index[i], epoch_df.index[i] + pd.Timedelta('30s')
    ax.clear()
    ax.set_xlim(start, end)
    ax.set_title(f"Wake Period + Zero-Crossing Rate ({start.strftime('%H:%M:%S')})")
    ax.set_xlabel("Time")
    ax.set_ylabel("ZCR (crossings per sample)")
    ax.grid(True)

    # highlight WAKE spans
    for s, e in zip(wake_df.index, wake_df.index.to_series().shift(-1)):
        if pd.isna(e): continue
        if s < end and e > start:
            ax.axvspan(max(s, start), min(e, end), color='orange', alpha=0.3)
            ax.axvline(s, color='black', linestyle='--', lw=1)

    # plot continuous ZCR curve
    mask = (zcr_series.index >= start) & (zcr_series.index <= end)
    ax.plot(zcr_series.index[mask], zcr_series[mask], color='green', lw=1.5)
    fig.canvas.draw_idle()

slider.on_changed(update)
update(0)
plt.show()
