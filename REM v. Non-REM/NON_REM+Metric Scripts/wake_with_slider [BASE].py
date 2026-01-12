import pandas as pd
import matplotlib.pyplot as plt
import datetime
import re
from matplotlib.widgets import Slider

# === 1. CONFIG ===
sleep_file = r'Patient-Data/DATASET - HIGH RISK 02/High_Risk02 [CHILD TEMPLATE]/Sleep profile - HighRisk.txt'

# === 2. PARSE RECORDING DATE FROM "Start Time" LINE ===
file_date = None
start_clock_time = None

with open(sleep_file, 'r', errors='ignore') as f:
    for line in f:
        if "Start Time" in line:
            # Example: "Start Time: 8/20/2024 10:45:00 AM"
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

# Fallback: search filename for a date if not found
if file_date is None:
    name_match = re.search(r'(\d{8})', sleep_file)
    if name_match:
        file_date = datetime.datetime.strptime(name_match.group(1), '%Y%m%d').date()

if file_date is None:
    raise ValueError("Recording date could not be parsed from the sleep file or filename.")

# === 3. LOAD SLEEP STATE DATA ===
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
    raise ValueError("No sleep state data found in file.")

# === 4. CREATE 30-SECOND EPOCHS ===
analysis_start_time = sleep_df.index[0]
analysis_end_time = sleep_df.index[-1]
epoch_range = pd.date_range(start=analysis_start_time, end=analysis_end_time, freq='30s')
epoch_df = pd.DataFrame(index=epoch_range)

# Map states to epochs
epoch_df = pd.merge_asof(epoch_df, sleep_df, left_index=True, right_index=True, direction='backward')
epoch_df.rename(columns={'state': 'sleep_state'}, inplace=True)
epoch_df['sleep_state'] = epoch_df['sleep_state'].fillna('Unknown')

# === 5. CALCULATE SUMMARY STATS (formatted like official report) ===
sleep_df_stats = sleep_df.copy()
sleep_df_stats['end_time'] = sleep_df_stats.index.to_series().shift(-1).fillna(analysis_end_time)
sleep_df_stats['duration'] = sleep_df_stats['end_time'] - sleep_df_stats.index.to_series()

stage_durations = sleep_df_stats.groupby('state')['duration'].sum()
total_analyzed_duration = stage_durations.sum()
stage_percentages = (stage_durations / total_analyzed_duration) * 100
total_hours = total_analyzed_duration.total_seconds() / 3600

print("\n--- Analysis Statistics ---")
print(f"Time Period: {total_hours:.2f} hours (from {analysis_start_time.strftime('%H:%M')} to {analysis_end_time.strftime('%H:%M')})")
print("\nSleep Stage Distribution:")
for stage, percentage in stage_percentages.items():
    print(f"- {stage:<13}: {percentage:>6.2f}%")
print("---------------------------\n")

# === 6. FILTER FOR WAKE ONLY ===
wake_df = sleep_df[sleep_df['state'].str.lower() == 'wake'].copy()
wake_df['end_time'] = wake_df.index.to_series().shift(-1).fillna(analysis_end_time)

# === 7. PRINT ALL WAKE SEGMENTS (ONCE) ===
print("--- Wake Segments ---")
if wake_df.empty:
    print("No Wake segments found.")
else:
    for i, (start_ts, row) in enumerate(wake_df.iterrows(), start=1):
        start_time = start_ts
        end_time = row['end_time']
        duration = (end_time - start_time).total_seconds()
        print(f"{i:03d}. Start: {start_time.strftime('%H:%M:%S')} | End: {end_time.strftime('%H:%M:%S')} | Duration: {duration:>6.1f} s")
print("----------------------\n")

# === 8. SETUP PLOT ===
fig, ax_detail = plt.subplots(figsize=(15, 7))
plt.subplots_adjust(bottom=0.25)
ax_detail.set_title("Wake Stage Visualization (30-Second Epochs)")
ax_detail.set_xlabel("Time")
ax_detail.set_ylabel("State")
ax_detail.grid(True)

cmap = {'Wake': 'orange', 'Unknown': 'lightgray'}

# === 9. PREPARE SLIDER ===
ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
slider = Slider(ax_slider, 'Epoch', 0, len(epoch_df) - 1, valinit=0, valstep=1)

wake_df_with_end = wake_df.copy()
wake_df_with_end['end_time'] = wake_df_with_end.index.to_series().shift(-1).fillna(pd.Timestamp.max)

def update(epoch_index_float):
    epoch_index = int(epoch_index_float)
    start_time = epoch_df.index[epoch_index]
    end_time = start_time + pd.Timedelta('30s')

    ax_detail.clear()
    ax_detail.set_xlim(start_time, end_time)
    ax_detail.set_ylim(0, 1)
    ax_detail.set_title(f"Wake Period – Starting {start_time.strftime('%H:%M:%S')}")
    ax_detail.set_xlabel("Time")
    ax_detail.set_ylabel("State")
    ax_detail.grid(True)

    # Plot Wake segments overlapping current window
    overlapping_stages = wake_df_with_end[
        (wake_df_with_end.index < end_time) & (wake_df_with_end['end_time'] > start_time)
    ]

    for _, stage_row in overlapping_stages.iterrows():
        span_start = max(start_time, stage_row.name)
        span_end = min(end_time, stage_row['end_time'])
        ax_detail.axvspan(span_start, span_end, color=cmap['Wake'], alpha=0.3, zorder=1)
        ax_detail.axvline(span_start, color='black', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)
        text_x = span_start + (span_end - span_start) / 2
        ax_detail.text(text_x, 0.5, 'Wake', ha='center', va='center', fontsize=12, weight='bold', zorder=4)

    fig.canvas.draw_idle()

slider.on_changed(update)

# === 10. INITIAL DISPLAY ===
update(0)
plt.show()
