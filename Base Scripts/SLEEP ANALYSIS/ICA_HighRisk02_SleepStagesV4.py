import mne
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress warnings related to frequency limits
warnings.filterwarnings("ignore")

# Load the EDF file
edf_file_path = "/Users/labuser/Desktop/SOMNOmedics Home Tests/HIghRisk02/HIghRisk02_(1).edf"
raw = mne.io.read_raw_edf(edf_file_path, preload=True)

# Define frequency bands for sleep stages (focused on Delta and Theta for infants)
frequency_bands = {
    "Delta": (0.5, 3.0),  # Delta waves
    "Theta": (4, 8),      # Theta waves
}

# Map relevant channels to appropriate types (e.g., EOG, EEG)
mapping = {
    "E1:M2": "eog",
    "E2:M2": "eog",
    # Add other mappings if needed, e.g., ECG or EMG
}
raw.set_channel_types(mapping)

# Select relevant EEG, EOG, EMG, and ECG channels for analysis
selected_channels = [
    "E1:M2", "E2:M2",  # EOG (Eye movements)
    "F4:M1", "F3:M2", "C4:M1", "C3:M2", "O2:M1", "O1:M2",  # EEG (Brain waves)
    "EMG1", "EMG2", "EMG3",  # EMG (Muscle activity)
    "ECG II"  # ECG (Heart activity)
]

# Handle missing channels
available_channels = raw.info["ch_names"]
selected_channels = [ch for ch in selected_channels if ch in available_channels]
raw.pick_channels(selected_channels)

# Apply a band-pass filter (optional)
raw.filter(l_freq=0.5, h_freq=30.0, fir_design="firwin")

# Artifact rejection using ICA
ica = mne.preprocessing.ICA(n_components=None, random_state=42, max_iter=300)
eog_artifacts = np.zeros((len(selected_channels), raw.n_times))  # Default placeholder
ecg_artifacts = np.zeros((len(selected_channels), raw.n_times))  # Default placeholder

try:
    print(f"Fitting ICA to data using {len(selected_channels)} channels.")
    ica.fit(raw)

    # Detect EOG and ECG artifacts
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw)

    # Extract artifact components
    if eog_indices:
        eog_artifacts = ica.get_sources(raw).get_data(picks=eog_indices)
    if ecg_indices:
        ecg_artifacts = ica.get_sources(raw).get_data(picks=ecg_indices)

    # Exclude artifact components
    ica.exclude = eog_indices + ecg_indices
    ica.apply(raw)
    print("ICA successfully applied.")
except Exception as e:
    print(f"Error during ICA fitting: {e}")
    print("Skipping ICA step.")

# Get the EEG data and sampling frequency
data = raw.get_data(return_times=False)  # Get data only
sfreq = raw.info["sfreq"]  # Get the sampling frequency

# Function to compute the power in a specific frequency band
def compute_band_power(data, sfreq, band):
    psd, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=sfreq, fmin=band[0], fmax=band[1], n_per_seg=256
    )
    return np.sum(psd)  # Total power in the band

# Initialize variables for sleep stage classification
time_segments = 30  # Segment length in seconds
num_segments = int(raw.times[-1] // time_segments)
sleep_stages = np.zeros(num_segments)  # Array to store sleep stages for each segment

# Loop through each time segment to compute band powers
for segment in range(num_segments):
    segment_start = int(segment * time_segments * sfreq)
    segment_end = int(min(segment_start + time_segments * sfreq, data.shape[1]))  # Ensure no out-of-bounds access

    # Compute band powers for each EEG channel in this segment
    band_powers = {band: [] for band in ["Delta", "Theta"]}
    for ch_index, ch_data in enumerate(data):
        ch_segment = ch_data[segment_start:segment_end]
        for band, freq_range in frequency_bands.items():
            band_power = compute_band_power(ch_segment, sfreq, freq_range)
            band_powers[band].append(band_power)

    # Compute average band power across all channels for this segment
    delta_power = np.mean(band_powers["Delta"])
    theta_power = np.mean(band_powers["Theta"])

    # Analyze EOG and ECG artifacts for this segment
    eog_segment = eog_artifacts[:, segment_start:segment_end]
    ecg_segment = ecg_artifacts[:, segment_start:segment_end]

    eog_activity = np.mean(np.abs(eog_segment))  # Average EOG activity
    ecg_activity = np.mean(np.abs(ecg_segment))  # Average ECG activity

    # Infant sleep stage classification logic (refined with artifacts)
    if delta_power > 1.5 * theta_power and eog_activity < 0.1:  # Quiet Sleep
        sleep_stages[segment] = 4  # N3 (Deep Sleep)
    elif theta_power > 1.5 * delta_power and eog_activity > 0.5:  # Active Sleep (REM)
        sleep_stages[segment] = 5  # REM (Active Sleep)
    elif 0.8 * delta_power < theta_power < 1.5 * delta_power:  # Intermediate stage
        sleep_stages[segment] = 3  # N2 (Light Sleep)
    elif delta_power == 0 and theta_power == 0 and ecg_activity > 0.5:  # No significant power but high ECG
        sleep_stages[segment] = 1  # Wake
    else:  # Unclear stage, default to N1
        sleep_stages[segment] = 2  # N1 (Drowsy or Transitional Sleep)

# Create a time axis for plotting
time_axis = np.arange(num_segments) * time_segments

# Plotting sleep stages
plt.figure(figsize=(15, 6))
plt.step(time_axis, sleep_stages, where="post")
plt.yticks([1, 2, 3, 4, 5], ["Wake", "N1", "N2", "N3", "REM"])
plt.xlabel("Time (seconds)")
plt.ylabel("Sleep Stages")
plt.title("Sleep Stages Over Time (Infants) with Artifact Analysis")
plt.grid()
plt.xlim(0, time_axis[-1])
plt.ylim(0.5, 5.5)
plt.show()