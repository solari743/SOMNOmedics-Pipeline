import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.stats import skew

# ---------- File & sensor ----------
filename = r'c:\Users\kevin\OneDrive\Desktop\DATA - POSTER\DATASET - HIGH RISK 02\High_Risk02 [RAW MOVEMENT]\highrisk02_20240820-141937.h5'
target_ID = 'XI-016162'  # HEAD sensor

def load_normalized_motion(filename, target_ID):
    with h5py.File(filename, 'r') as f:
        if target_ID not in f['Sensors']:
            raise ValueError(f"Sensor {target_ID} not found.")
        base_path = f'Sensors/{target_ID}'
        if 'Accelerometer' not in f[base_path] or 'Time' not in f[base_path]:
            raise ValueError("Missing Accelerometer or Time data.")
        acc = np.array(f[f'{base_path}/Accelerometer'][:], dtype=np.float64)
        time_raw = np.array(f[f'{base_path}/Time'][:], dtype=np.float64)
    # Motion intensity (Euclidean norm of delta)
    motion = np.sqrt(np.sum(np.diff(acc, axis=0)**2, axis=1))
    mean_int = np.mean(motion)
    norm = motion / mean_int if mean_int != 0 else motion
    return norm

# ---------- Load & compute ----------
normalized_motion = load_normalized_motion(filename, target_ID)
cov = np.std(normalized_motion) / np.mean(normalized_motion)
skewness_val = skew(normalized_motion)
mean_val = np.mean(normalized_motion)

# Movement categories
cat_edges = [0, 0.5, 1, 2, 5, 10, np.max(normalized_motion) + 1e-6]
cat_labels = ["Stillness\n(0–0.5×)", "Slight\n(0.5–1×)", "Moderate\n(1–2×)",
              "Large\n(2–5×)", "Very large\n(5–10×)", "Extreme\n(>10×)"]

counts, _ = np.histogram(normalized_motion, bins=cat_edges)
percents = counts / len(normalized_motion) * 100

# ---------- Graph 1: Horizontal stacked bar with improved annotation spacing ----------
colors = plt.get_cmap('tab20').colors
plt.figure(figsize=(10, 4))  # Made figure slightly taller for better annotation space
left = 0

# Collect segments that need external annotations
external_annotations = []

for i, (pct, label) in enumerate(zip(percents, cat_labels)):
    if pct <= 0:
        continue

    # Draw bar
    plt.barh(0, pct, left=left, height=0.6,
             color=colors[i % len(colors)], edgecolor='white')
    center = left + pct / 2

    if pct >= 4:  # label inside
        plt.text(center, 0, f"{label}\n{pct:.1f}%",
                 va='center', ha='center', fontsize=9,
                 color='white', weight='bold')
    else:
        # Store info for external annotation
        external_annotations.append({
            'center': center,
            'label': label,
            'pct': pct,
            'left': left
        })

    left += pct

# Handle external annotations with proper spacing
if external_annotations:
    # Sort by position to handle spacing better
    external_annotations.sort(key=lambda x: x['center'])
    
    # Calculate annotation heights with better spacing logic
    text_height = 0.4  # Approximate height of a 2-line text annotation
    min_spacing = 0.2   # Minimum gap between text boxes
    base_height = 1.2   # Starting height for annotations
    
    annotation_heights = []
    for i, ann in enumerate(external_annotations):
        if i == 0:
            height = base_height
        else:
            # Calculate minimum height to avoid overlap
            # Previous annotation bottom = prev_height - text_height/2
            # Current annotation top should be at least min_spacing above that
            prev_height = annotation_heights[i-1]
            min_height = prev_height + text_height + min_spacing
            
            # Also ensure some progressive spacing for visual appeal
            progressive_height = base_height + i * 0.6
            
            height = max(min_height, progressive_height)
        
        annotation_heights.append(height)
    
    # Place annotations
    for ann, height in zip(external_annotations, annotation_heights):
        center = ann['center']
        label = ann['label']
        pct = ann['pct']
        left_pos = ann['left']
        
        # Determine text position (left or center aligned)
        if left_pos > 95:  # Near right edge
            text_x = center + 8
            ha_align = 'left'
        else:
            text_x = center
            ha_align = 'center'
        
        plt.annotate(f"{label}\n{pct:.1f}%",
                     xy=(center, 0),            # arrow points to segment center
                     xytext=(text_x, height),   # text position with calculated height
                     textcoords='data',
                     ha=ha_align, va='bottom',
                     fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='gray'),
                     arrowprops=dict(arrowstyle="->", color='black', lw=0.8))

# Adjust plot limits to accommodate annotations
max_annotation_height = max(annotation_heights) if external_annotations else 1.5
plt.xlim(0, 120)  # extend beyond 100% for right-side labels
plt.ylim(-0.5, max_annotation_height + 0.5)
plt.yticks([])
plt.xlabel("Percentage of time (%)")
plt.title(f"High Risk 02 - % Time in Movement Categories — Sensor {target_ID}\nCoV={cov:.2f}, Skew={skewness_val:.2f}")
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


# ---------- Graph 2: Log-scale histogram ----------
plt.figure(figsize=(8, 4))
n, bins, patches = plt.hist(normalized_motion, bins=100, color='steelblue', edgecolor='black', alpha=0.85)
plt.xscale('log')
plt.xlabel("Normalized motion intensity (log scale)")
plt.ylabel("Frequency")
plt.title(f"High Risk 02 - Motion Intensity Distribution — Sensor {target_ID}\nCoV={cov:.2f}, Skew={skewness_val:.2f}")

# Shaded reference regions
plt.axvspan(0.01, 0.5, color='green', alpha=0.08, label='Stillness')
plt.axvspan(0.5, 2, color='yellow', alpha=0.08, label='Moderate')
plt.axvspan(2, np.max(normalized_motion), color='red', alpha=0.06, label='Bursts')

# Mean line
plt.axvline(mean_val, color='red', linestyle='--', linewidth=1.2, label=f"Mean ({mean_val:.2f}×)")
plt.legend(fontsize=8)
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# ---------- Graph 3: ECDF with clearer title ----------
plt.figure(figsize=(8, 4))
sorted_vals = np.sort(normalized_motion)
y = np.arange(1, len(sorted_vals)+1) / len(sorted_vals) * 100
plt.plot(sorted_vals, y, lw=1.5)
plt.xscale('log')
plt.xlabel("Normalized motion intensity (log scale)")
plt.ylabel("Cumulative percentage of time (%)")
plt.title(f"High Risk 02 - Cumulative % of Time at or Below Motion Intensity — Head Sensor ({target_ID})\nCoV={cov:.2f}, Skew={skewness_val:.2f}")
plt.grid(True, which='both', linestyle='--', alpha=0.4)

# Threshold markers
for thr in [0.5, 1, 2, 10]:
    pct = np.searchsorted(sorted_vals, thr, side='right') / len(sorted_vals) * 100
    plt.axvline(thr, ymin=0, ymax=pct/100, color='gray', linestyle=':', linewidth=0.8)
    plt.text(thr, min(pct + 3, 97), f"{pct:.1f}% ≤ {thr}×", rotation=90, va='bottom', ha='center', fontsize=8)

plt.tight_layout()
plt.show()