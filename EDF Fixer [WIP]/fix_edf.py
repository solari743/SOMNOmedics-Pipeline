import mne
import numpy as np

edf_path = (
    "c:/Users/kevin/OneDrive/Desktop/REM v. Non-REM/Patient-Data/"
    "DATASET - LOW RISK 02/Low_Risk02 [CHILD TEMPLATE]/LOWRISK_02_(1).edf"
)

# ---------------------------------------------
# 1. Load & Preprocess
# ---------------------------------------------
raw = mne.io.read_raw_edf(
    edf_path,
    preload=True,
    stim_channel=None
)

raw.filter(0.3, 35, fir_design="firwin")
raw.set_eeg_reference("average", projection=False)

montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage, on_missing="ignore")

# ---------------------------------------------
# 2. Visual Inspection (manual bad selection)
# ---------------------------------------------
print("\nInspect channels — close plot when finished.")
raw.plot(n_channels=20, scalings='auto', block=True)
print("\nMarked bad channels:", raw.info["bads"])

print("\nSkipping interpolation (bipolar montage / missing locations)")
bad_for_ica = raw.info["bads"]
raw.info["bads"] = []  # clear for ICA
print("Bad channels excluded from ICA:", bad_for_ica)

# ---------------------------------------------
# 3. ICA (time-domain only)
# ---------------------------------------------
raw_ica = raw.copy().filter(1.0, 35, fir_design="firwin")
picks_eeg = mne.pick_types(raw_ica.info, eeg=True, exclude=bad_for_ica)

ica = mne.preprocessing.ICA(
    n_components=0.99,
    method="picard",
    random_state=97,
    max_iter="auto"
)

print("\nFitting ICA...")
ica.fit(raw_ica, picks=picks_eeg)
print("ICA fitted with", ica.n_components_, "components")

ica.plot_sources(raw_ica)

# Surrogate blink detection using frontal channels
surrogate = ["F3", "F4", "E1", "E2"]
surrogate = [ch for ch in surrogate if ch in raw_ica.ch_names]

if surrogate:
    print("Using surrogate channel for blink detection:", surrogate[0])
    try:
        eog_inds, _ = ica.find_bads_eog(raw_ica, ch_name=surrogate[0])
        print("Potential blink components:", eog_inds)
        ica.exclude.extend(eog_inds)
    except Exception as e:
        print("Surrogate blink detection failed:", e)
else:
    print("No frontal surrogate channels found. Skipping automatic blink removal.")

raw_clean = raw.copy()
ica.apply(raw_clean)
print("\nICA applied. Data stored in raw_clean")

# ---------------------------------------------
# 4. Epoch Rejection
# ---------------------------------------------
print("\nEpoching for artifact rejection...")

epochs = mne.make_fixed_length_epochs(raw, duration=30.0, preload=True)
epochs_clean = mne.make_fixed_length_epochs(raw_clean, duration=30.0, preload=True)

# Infant EEG often exceeds 150uV — start with 400uV
reject_criteria = dict(eeg=400e-6)

print("Dropping noisy epochs with reject threshold:", reject_criteria)
epochs_clean.drop_bad(reject=reject_criteria)

print(f"Remaining epochs after drop_bad: {len(epochs_clean)} / {len(epochs)}")

# ---------------------------------------------
# 5. Wake Estimation (Non-Clinical Approximation)
# ---------------------------------------------
def wake_percentage(ep, threshold=150e-6):
    """Estimate wake-like epochs based on mean peak-to-peak amplitude."""
    n = len(ep)
    if n == 0:
        return np.nan, 0, 0

    data = ep.get_data()                     # (epochs, channels, samples)
    ptp = np.ptp(data, axis=2)               # peak-to-peak on time dimension
    mean_amp = ptp.mean(axis=1)              # per-epoch
    wake_like = (mean_amp > threshold).sum()
    pct = (wake_like / n) * 100
    return pct, wake_like, n

before_pct, before_count, before_total = wake_percentage(epochs)
after_pct, after_count, after_total = wake_percentage(epochs_clean)

print("\n------------------------------------")
print("WAKE ESTIMATION REPORT (non-clinical)")
print("------------------------------------")
print(f"Total epochs before: {before_total}")
print(f"Total epochs after:  {after_total}")

if not np.isnan(before_pct):
    print(f"Wake-like before: {before_count} ({before_pct:.2f}%)")
else:
    print("Wake-like before: not computable (no epochs).")

if not np.isnan(after_pct):
    print(f"Wake-like after:  {after_count} ({after_pct:.2f}%)")
    print(f"Reduction in false Wake: {(before_pct - after_pct):.2f}%")
else:
    print("Wake-like after: not computable (no epochs).")

print("------------------------------------")

# ---------------------------------------------
# 6. Final EEG Review
# ---------------------------------------------
raw_clean.plot(n_channels=20, scalings='auto', block=True)
