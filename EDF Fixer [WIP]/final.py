import mne
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import welch
from collections import Counter
import pandas as pd
import json

# =========================================================
# PATHS
# =========================================================
edf_path = (
    "c:/Users/kevin/OneDrive/Desktop/REM v. Non-REM/Patient-Data/"
    "DATASET - LOW RISK 02/Low_Risk02 [CHILD TEMPLATE]/LOWRISK_02_(1).edf"
)

output_txt = (
    "c:/Users/kevin/OneDrive/Desktop/REM [HR vs LR]/"
    "Sleep_profile_LowRisk_corrected.txt"
)

# =========================================================
# INFANT-SAFE THRESHOLDS (CONSERVATIVE, NON-DESTRUCTIVE)
# =========================================================
MOVEMENT_UV = 1500e-6
WAKE_UV = 600e-6
DELTA_NREM_RATIO = 0.35
THETA_REM_RATIO = 0.30

# =========================================================
# 1. LOAD & BASIC PREPROCESSING
# =========================================================
raw = mne.io.read_raw_edf(edf_path, preload=True, stim_channel=None)

raw.filter(0.3, 35, fir_design="firwin")
raw.set_eeg_reference("average", projection=False)

# Montage only for naming consistency — NOT for interpolation
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage, on_missing="ignore")

print(f"Loaded EDF with {len(raw.ch_names)} channels")

# =========================================================
# 2. MANUAL VISUAL CHECK (OPTIONAL)
# =========================================================
print("\nInspect EEG — close plot when finished.")
raw.plot(n_channels=20, scalings="auto", block=True)

bad_channels = raw.info["bads"].copy()
raw.info["bads"] = []   # clear before ICA
print("Bad channels (excluded from ICA):", bad_channels)

# =========================================================
# 3. ICA (TIME-DOMAIN ONLY, NO TOPO MAPS)
# =========================================================
raw_ica = raw.copy().filter(1.0, 35, fir_design="firwin")

picks_eeg = mne.pick_types(raw_ica.info, eeg=True, exclude=bad_channels)

ica = mne.preprocessing.ICA(
    n_components=0.99,
    method="picard",
    random_state=97,
    max_iter="auto"
)

print("\nFitting ICA...")
ica.fit(raw_ica, picks=picks_eeg)
print(f"ICA fitted with {ica.n_components_} components")

ica.plot_sources(raw_ica)

# Surrogate blink detection (infant PSG-safe)
for ch in ["F3", "F4", "E1", "E2"]:
    if ch in raw_ica.ch_names:
        try:
            inds, _ = ica.find_bads_eog(raw_ica, ch_name=ch)
            ica.exclude.extend(inds)
            print("Blink-like components:", inds)
        except Exception:
            pass
        break

raw_clean = raw.copy()
ica.apply(raw_clean)
print("ICA applied → raw_clean created")

# =========================================================
# 4. ***CRITICAL FIX*** — EEG-ONLY VIEW
# =========================================================
eeg_picks = mne.pick_types(raw_clean.info, eeg=True)

raw_eeg = raw_clean.copy().pick(eeg_picks)

print(f"Using {len(eeg_picks)} EEG channels for epoching")

# =========================================================
# 5. EPOCHING (NO AUTO-REJECTION)
# =========================================================
epochs = mne.make_fixed_length_epochs(
    raw_eeg,
    duration=30.0,
    preload=True
)

print(f"Total epochs: {len(epochs)}")

epoch_data = epochs.get_data()
sfreq = raw_eeg.info["sfreq"]
ch_names = epochs.ch_names

# =========================================================
# 6. FEATURE-BASED INFANT SLEEP CLASSIFIER
# =========================================================
features = []

def classify_epoch(epoch):
    # Amplitude
    ptp = np.ptp(epoch, axis=1)
    mean_ptp = ptp.mean()

    # Spectral power (Welch)
    nperseg = min(int(sfreq * 4), epoch.shape[1])
    freqs, psd = welch(epoch, fs=sfreq, nperseg=nperseg, axis=-1)
    psd_mean = psd.mean(axis=0)

    def band(lo, hi):
        idx = (freqs >= lo) & (freqs <= hi)
        return psd_mean[idx].sum()

    delta = band(0.5, 4)
    theta = band(4, 8)
    alpha = band(8, 12)
    beta  = band(12, 30)
    total = delta + theta + alpha + beta + 1e-12

    delta_r = delta / total
    theta_r = theta / total

    features.append({
        "mean_ptp": mean_ptp,
        "delta_ratio": delta_r,
        "theta_ratio": theta_r
    })

    # --- Classification logic ---
    if mean_ptp > MOVEMENT_UV:
        return "Movement"

    if delta_r >= DELTA_NREM_RATIO:
        return "NREM"

    if theta_r >= THETA_REM_RATIO and delta_r < DELTA_NREM_RATIO:
        return "REM"

    if mean_ptp > WAKE_UV:
        return "Wake"

    return "Transitional"

labels = [classify_epoch(ep) for ep in epoch_data]

print("Initial label counts:", Counter(labels))

# =========================================================
# 7. SAFETY NETS (ANTI-COLLAPSE)
# =========================================================
if labels.count("Movement") / len(labels) > 0.6:
    labels = ["Transitional" if l == "Movement" else l for l in labels]
    print("Movement collapse prevented")

# Temporal smoothing (AASM-style)
for i in range(1, len(labels) - 1):
    if labels[i-1] == labels[i+1] != labels[i]:
        labels[i] = labels[i-1]

print("Final label counts:", Counter(labels))

# =========================================================
# 8. EXPORT SCHLAFPROFIL TXT
# =========================================================
start_time = raw.info["meas_date"]
if start_time is None:
    start_time = datetime(2024, 8, 20, 10, 45, 0)

with open(output_txt, "w", encoding="utf-8") as f:
    f.write("Signal ID: SchlafProfil\\profil\n")
    f.write(f"Start Time: {start_time.strftime('%m/%d/%Y %I:%M:%S %p')}\n")
    f.write("Unit:\n")
    f.write("Signal Type: Discret\n")
    f.write("Events list: NREM,Transitional,REM,Wake,Movement\n")
    f.write("Rate: 30 s\n\n")

    t = start_time
    for lab in labels:
        f.write(f"{t.strftime('%H:%M:%S')},000; {lab}\n")
        t += timedelta(seconds=30)

print(f"✓ SchlafProfil exported to: {output_txt}")

# =========================================================
# 9. OPTIONAL: EXPORT FEATURES
# =========================================================
df = pd.DataFrame(features)
df["label"] = labels
df.to_csv(output_txt.replace(".txt", "_features.csv"), index=False)

print("✓ Feature CSV exported")

# =========================================================
# 10. FINAL VISUAL CHECK
# =========================================================
print("\nOpening final cleaned EEG...")
raw_eeg.plot(n_channels=20, scalings="auto", block=True)
