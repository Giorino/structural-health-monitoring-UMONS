#!/usr/bin/env python3
"""
Visualize rolling window statistics on FBG wavelength data.
Shows rolling mean and rolling std overlaid on the original signal.
Supports both merged CSV files and raw interrogator TXT files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Configuration
WINDOW_SIZE = 5  # Rolling window size

# Data directories
MERGED_DATA_DIR = "output/20250915_135413"
RAW_DATA_DIR = "interrogator-data"

def load_merged_csv(filepath):
    """Load merged CSV file"""
    df = pd.read_csv(filepath)
    return df['WL_ch2'].dropna().values, 'WL_ch2'

def load_raw_interrogator(filepath):
    """Load raw interrogator TXT file"""
    # Raw files are tab-separated with variable columns
    # Try to parse flexibly
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header line
    header = lines[0].strip().split('\t')
    print(f"  Header columns: {header[:5]}...")  # Show first 5 columns
    
    # Parse data lines
    wavelengths = []
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            # Try to get WL2 (4th column, index 3) or fallback to WL1 (3rd column, index 2)
            try:
                if len(parts) >= 4:
                    wl = float(parts[3])  # WL2
                else:
                    wl = float(parts[2])  # WL1
                wavelengths.append(wl)
            except (ValueError, IndexError):
                # If only Timestamp, Time, and one WL value
                try:
                    wl = float(parts[2])
                    wavelengths.append(wl)
                except (ValueError, IndexError):
                    continue
    
    wl_name = 'WL_ch2' if len(header) >= 4 else 'WL_ch1'
    return pd.Series(wavelengths).dropna().values, wl_name

# Find all files
merged_files = sorted(glob.glob(os.path.join(MERGED_DATA_DIR, "merged_*.csv")))
raw_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*-interrogator.txt")))

# Display data source selection
print("\n=== Select Data Source ===")
print("  [1] Merged CSV files (processed)")
print("  [2] Raw interrogator TXT files (high-resolution)")

source_choice = input("\nEnter choice (1 or 2): ").strip()

if source_choice == "2":
    files = raw_files
    file_type = "raw"
    load_func = load_raw_interrogator
else:
    files = merged_files
    file_type = "merged"
    load_func = load_merged_csv

if not files:
    print(f"No {file_type} files found!")
    exit(1)

# Display file selection menu
print(f"\n=== Available {file_type.upper()} Files ===")
for i, f in enumerate(files):
    print(f"  [{i:2d}] {os.path.basename(f)}")

print(f"\nTotal files: {len(files)}")
choice = input("\nEnter file number (or press Enter for first file): ").strip()

if choice == "":
    file_idx = 0
else:
    try:
        file_idx = int(choice)
        if file_idx < 0 or file_idx >= len(files):
            print(f"Invalid selection. Using file 0.")
            file_idx = 0
    except ValueError:
        print(f"Invalid input. Using file 0.")
        file_idx = 0

sample_file = files[file_idx]
print(f"\nLoading: {os.path.basename(sample_file)}")

# Load data
wl, wl_name = load_func(sample_file)
time_index = range(len(wl))

print(f"Loaded {len(wl)} samples")

# Calculate rolling statistics
wl_series = pd.Series(wl)
rolling_mean = wl_series.rolling(window=WINDOW_SIZE, min_periods=1).mean()
rolling_std = wl_series.rolling(window=WINDOW_SIZE, min_periods=1).std().fillna(0)

# Create the plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Plot 1: Original signal with rolling mean
ax1 = axes[0]
ax1.plot(time_index, wl, 'b-', alpha=0.6, label=f'{wl_name} (raw)', linewidth=0.5)
ax1.plot(time_index, rolling_mean, 'r-', label=f'Rolling Mean (window={WINDOW_SIZE})', linewidth=1.5)
ax1.set_ylabel('Wavelength (nm)')
ax1.set_title(f'FBG Wavelength Signal - {os.path.basename(sample_file)} ({len(wl)} samples)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Rolling std (local variability)
ax2 = axes[1]
ax2.fill_between(time_index, 0, rolling_std, alpha=0.5, color='orange')
ax2.plot(time_index, rolling_std, 'orange', label=f'Rolling Std (window={WINDOW_SIZE})', linewidth=1)
ax2.set_ylabel('Rolling Std (nm)')
ax2.set_title('Local Variability - Higher values may indicate crack formation')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Plot 3: Signal with ±std band
ax3 = axes[2]
ax3.plot(time_index, wl, 'b-', alpha=0.4, label=wl_name, linewidth=0.5)
ax3.fill_between(time_index, 
                 rolling_mean - rolling_std, 
                 rolling_mean + rolling_std, 
                 alpha=0.3, color='green', label='±1 Std band')
ax3.plot(time_index, rolling_mean, 'r-', linewidth=1)
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Wavelength (nm)')
ax3.set_title('Signal with Rolling Mean ± Std Band')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nStatistics:")
print(f"  Total samples: {len(wl)}")
print(f"  Signal range: {wl.min():.3f} - {wl.max():.3f} nm")
print(f"  Mean rolling std: {rolling_std.mean():.4f} nm")
print(f"  Max rolling std: {rolling_std.max():.4f} nm")
