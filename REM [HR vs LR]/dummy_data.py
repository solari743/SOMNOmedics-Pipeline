from pathlib import Path
import mne


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

sample_path = mne.datasets.sample.data_path(
    path=DATA_DIR,
    force_update=False
)

RAW_FILE = sample_path / "MEG" / "sample" / "sample_audvis_raw.fif"

print("Loading raw file:")
print(RAW_FILE)

raw = mne.io.read_raw_fif(RAW_FILE, preload=True)

print("\nOriginal data:")
print(raw)

raw_eeg = raw.copy().pick_types(
    eeg=True,
    meg=False,
    eog=True,    # keep eye channels (useful for ICA)
    ecg=True,    # optional
    stim=False
)

print("\nEEG-only data:")
print(raw_eeg)

# ================================
# 5. Set EEG montage (IMPORTANT)
# ================================
montage = mne.channels.make_standard_montage("standard_1020")
raw_eeg.set_montage(montage, on_missing="ignore")

# ================================
# 6. Optional: EEG filtering (sleep / ICA friendly)
# ================================
raw_eeg.filter(
    l_freq=0.5,
    h_freq=30,
    fir_design="firwin"
)

# ================================
# 7. Save EEG-only file
# ================================
EEG_ONLY_FILE = OUTPUT_DIR / "sample_eeg_only_raw.fif"

raw_eeg.save(
    EEG_ONLY_FILE,
    overwrite=True
)

print("\nSaved EEG-only file to:")
print(EEG_ONLY_FILE)

# ================================
# 8. Quick visualization (sanity check)
# ================================
raw_eeg.plot(
    n_channels=30,
    scalings="auto",
    title="EEG-only Sample Data"
)
