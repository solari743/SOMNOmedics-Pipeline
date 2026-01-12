"""
Script to parse and visualize sleep stage progression over time using line plots.
Supports Events list: NREM, Transitional, REM, Wake, Movement.

- Parses sleep stages and timestamps from the sleep profile
- Calculates time spent and percentage in each stage
- Maps stages to numerical values for visualization
- Plots sleep progression over time using a step-style line plot
"""

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import matplotlib.dates as mdates

# === CONFIG ===
# Please update this path to your actual file location
sleep_file = r'c:\Users\kevin\OneDrive\Desktop\DATA - POSTER\DATASET - HIGH RISK 02\High_Risk02 [CHILD TEMPLATE]\Sleep profile.txt'

# === PARSING ===
try:
    sleep_stages = []
    timestamps = []

    with open(sleep_file, 'r') as file:
        lines = file.readlines()

    # Skip metadata lines (header ends when the line starts with a timestamp)
    data_start = 0
    for i, line in enumerate(lines):
        if ';' in line and ',' in line and ':' in line.split(';')[0]:
            data_start = i
            break

    for line in lines[data_start:]:
        line = line.strip()
        if not line or ';' not in line:
            continue
        try:
            time_str, stage = [part.strip() for part in line.split(';')[:2]]
            if stage == 'A':
                continue  # Skip Artifact
            time_obj = datetime.strptime(time_str.split(',')[0], '%H:%M:%S')
            timestamps.append(time_obj)
            sleep_stages.append(stage)
        except Exception:
            continue

    if not timestamps:
        raise ValueError("No valid timestamps found in sleep profile.")

    print(f"Successfully parsed {len(timestamps)} data points")
    print(f"Sleep stages found: {set(sleep_stages)}")

    # === STAGE COUNTS & PERCENTAGES ===
    stage_counts = {}
    for stage in sleep_stages:
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    stage_minutes = {stage: count * 0.5 for stage, count in stage_counts.items()}
    total_epochs = len(sleep_stages)
    stage_percentages = {stage: (count / total_epochs) * 100 for stage, count in stage_counts.items()}

    print("\nSleep Statistics:")
    print("-" * 40)
    for stage in ['Wake', 'Movement', 'Transitional', 'NREM', 'REM']:
        if stage in stage_minutes:
            print(f"{stage:12}: {stage_minutes[stage]:6.1f} min ({stage_percentages[stage]:5.1f}%)")

    # === MAPPING FOR PLOTTING ===
    stage_colors = {
        'Wake': '#FF6B6B',
        'Movement': '#B8860B',
        'Transitional': '#87CEFA',
        'NREM': '#4682B4',
        'REM': '#9370DB'
    }

    stage_values = {
        'Wake': 4,
        'Movement': 3,
        'Transitional': 2,
        'NREM': 1,
        'REM': 0
    }

    # Convert timestamps to datetime for plotting (anchor to today)
    start_datetime = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    timestamp_datetimes = [
        start_datetime.replace(hour=t.hour, minute=t.minute, second=t.second)
        for t in timestamps
    ]

    y_values = [stage_values.get(stage, -1) for stage in sleep_stages]

    # === PLOTTING ===
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(timestamp_datetimes, y_values, drawstyle='steps-post', color='blue', linewidth=2)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))  # Changed to 30 minutes
    fig.autofmt_xdate()

    # Format y-axis
    ax.set_yticks(list(stage_values.values()))
    ax.set_yticklabels(list(stage_values.keys()))

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Sleep Stage', fontsize=12)
    ax.set_title('Sleep Profile Analysis - High Risk 02', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Could not find file '{sleep_file}'")
except Exception as e:
    print(f"An error occurred: {e}")