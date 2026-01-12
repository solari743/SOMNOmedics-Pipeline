import mne
import numpy as np
import matplotlib.pyplot as plt
import warnings
import datetime
import pandas as pd
from scipy.ndimage import uniform_filter1d
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

edf_file_path = r'c:\Users\kevin\OneDrive\Desktop\DATA - POSTER\DATASET - LOW RISK 02\Low_Risk02 [CHILD TEMPLATE]\LOWRISK_02_(1).edf'
raw = mne.io.read_raw_edf(edf_file_path, preload=True)

start_time = raw.info['meas_date']
if isinstance(start_time, np.datetime64):
    start_time = pd.to_datetime(str(start_time)).to_pydatetime()
elif isinstance(start_time, (tuple, list)):
    start_time = datetime.datetime.fromtimestamp(start_time[0] + start_time[1] * 1e-6)
elif not isinstance(start_time, datetime.datetime):
    start_time = None

if start_time:
    print(f"Recording start time: {start_time}")
else:
    print("Recording start time not available")

mapping = {
    "E1:M2": "eog",
    "E2:M2": "eog",
}
raw.set_channel_types(mapping)

selected_channels = [
    "F4:M1", "F3:M2", "C4:M1", "C3:M2", "O2:M1", "O1:M2",
]
available_channels = raw.info["ch_names"]
selected_channels = [ch for ch in selected_channels if ch in available_channels]
raw.pick_channels(selected_channels)

raw.filter(0.5, 30, fir_design='firwin')

ica = mne.preprocessing.ICA(n_components=min(20, len(selected_channels)), random_state=42, max_iter=300)
try:
    ica.fit(raw)
    eog_indices, _ = ica.find_bads_eog(raw)
    ecg_indices, _ = ica.find_bads_ecg(raw)
    exclude_indices = list(set(eog_indices + ecg_indices))
    ica.exclude = exclude_indices
    ica.apply(raw)
except Exception as e:
    print(f"ICA skipped due to error: {e}")

data = raw.get_data()
sfreq = raw.info['sfreq']

frequency_bands = {
    "Delta": (0.5, 3.0),
    "Theta": (4, 8),
    "Alpha": (8, 12),
}

def compute_relative_band_power(epoch_data, sfreq, bands):
    psd, freqs = mne.time_frequency.psd_array_welch(
        epoch_data, sfreq=sfreq, fmin=0.5, fmax=30, n_per_seg=256
    )
    total_power = np.sum(psd, axis=-1, keepdims=True)
    band_powers = {}
    for band_name, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        band_power = np.sum(psd[:, idx], axis=-1)
        band_powers[band_name] = band_power / total_power.flatten()
    return band_powers

epoch_length_sec = 30
num_epochs = int(raw.times[-1] // epoch_length_sec)

sleep_stages = []
epoch_times = []

if start_time:
    epoch_times = [start_time + datetime.timedelta(seconds=i * epoch_length_sec) for i in range(num_epochs)]

for i in range(num_epochs):
    start = int(i * epoch_length_sec * sfreq)
    end = int(min((i + 1) * epoch_length_sec * sfreq, data.shape[1]))

    epoch_data = data[:, start:end]

    band_powers = compute_relative_band_power(epoch_data, sfreq, frequency_bands)
    delta_mean = np.mean(band_powers["Delta"])
    theta_mean = np.mean(band_powers["Theta"])
    alpha_mean = np.mean(band_powers["Alpha"])

    if alpha_mean > 0.2 and delta_mean < 0.2:
        stage = 1  
    elif 0.1 <= alpha_mean <= 0.2 and 0.15 <= theta_mean <= 0.3:
        stage = 3  
    elif delta_mean > 0.3 and delta_mean > 1.5 * theta_mean:
        stage = 4  
    else:
        stage = 2  

    sleep_stages.append(stage)

sleep_stages_smoothed = uniform_filter1d(sleep_stages, size=3, mode='nearest')
sleep_stages_smoothed = np.round(sleep_stages_smoothed).astype(int)

stage_labels = {
    1: 'Wake',
    2: 'Movement',
    3: 'Transitional',
    4: 'NREM',
}

unique, counts = np.unique(sleep_stages_smoothed, return_counts=True)
total_epochs = len(sleep_stages_smoothed)
stage_percentages = {stage_labels.get(u, 'Unknown'): (c / total_epochs) * 100 for u, c in zip(unique, counts)}

print(f"--- Infant EEG Sleep Stage Summary ---")
print(f"Recording Duration: {raw.times[-1] / 3600:.2f} hours")
print("Sleep Stage Distribution (%):")
for stage_name, pct in stage_percentages.items():
    print(f" - {stage_name:12}: {pct:.2f}%")

plt.figure(figsize=(15,6))
plt.step(epoch_times, sleep_stages_smoothed, where='post', color='blue')

plt.yticks([1, 2, 3, 4], ['Wake', 'Movement', 'Transitional', 'NREM'])
plt.xlabel('Time')
plt.ylabel('Sleep Stage')
plt.title('Low Risk 02 - Sleep Profile Analysis', fontweight='bold')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()