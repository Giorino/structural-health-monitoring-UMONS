#!/usr/bin/env python3

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------
# Utilities to discover outputs
# ------------------------------

def find_latest_output_directory(base_dir: Optional[str] = None) -> Optional[str]:
    """Find the most recently modified subdirectory under the base output directory.

    If base_dir is None or a relative path, it is resolved relative to this file's directory.
    Returns absolute path or None if not found.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir is None:
        resolved_base = os.path.join(script_dir, "output")
    else:
        resolved_base = base_dir if os.path.isabs(base_dir) else os.path.join(script_dir, str(base_dir))

    if not os.path.isdir(resolved_base):
        return None

    output_dirs = [
        os.path.join(resolved_base, d)
        for d in os.listdir(resolved_base)
        if os.path.isdir(os.path.join(resolved_base, d))
    ]
    if not output_dirs:
        return None
    return max(output_dirs, key=os.path.getmtime)


def list_merged_csvs(directory: str) -> List[str]:
    """Return sorted list of merged_*.csv files in the directory."""
    pattern = os.path.join(directory, "merged_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        # Fallback: accept any CSV
        files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    return files


def parse_distance_layers_from_name(name: str) -> Tuple[Optional[float], Optional[int]]:
    """Extract distance in cm and number of layers from a name like '27cm-12layers-3'."""
    base = os.path.splitext(os.path.basename(name))[0]
    # Strip leading 'merged_' and trailing timestamp if present
    if base.startswith("merged_"):
        base = base[len("merged_"):]
    base = re.sub(r"_\d{8}(_\d{4,6})?$", "", base)

    m = re.search(r"(?P<dist>\d+)cm-?(?P<layers>\d+)layers", base)
    if not m:
        return None, None
    try:
        dist_cm = float(m.group("dist"))
    except Exception:
        dist_cm = None
    try:
        layers = int(m.group("layers"))
    except Exception:
        layers = None
    return dist_cm, layers


def _derive_base_key_from_merged_filename(basename: str) -> Optional[str]:
    """From 'merged_27cm-12layers-3_YYYYMMDD_HHMM.csv' → '27cm-12layers-3'."""
    base = os.path.splitext(os.path.basename(basename))[0]
    if base.startswith("merged_"):
        base = base[len("merged_"):]
    # strip trailing _YYYYMMDD_HHMM or _YYYYMMDD_HHMMSS
    base = re.sub(r"_\d{8}(_\d{4,6})?$", "", base)
    return base or None


def filter_csvs_by_patterns(csv_paths: List[str], target_patterns: List[str]) -> List[str]:
    """Filter CSV files to only include those matching the target patterns."""
    filtered = []
    for path in csv_paths:
        base_key = _derive_base_key_from_merged_filename(os.path.basename(path))
        if base_key and any(pattern in base_key for pattern in target_patterns):
            filtered.append(path)
    return filtered


def load_and_merge_csvs(csv_paths: List[str]) -> pd.DataFrame:
    """Load CSVs and concatenate with a 'source_file' column.

    For 7-s dataset: use only first half of data
    For 8-s dataset: shift to connect with end point of truncated 7-s data
    """
    frames: List[pd.DataFrame] = []
    
    # First pass: load and process 7-s data to get its endpoint
    df_7s = None
    for path in csv_paths:
        base_key = _derive_base_key_from_merged_filename(os.path.basename(path))
        if base_key and base_key.endswith('7-s'):
            df_7s = pd.read_csv(path)
            df_7s = df_7s.head(len(df_7s) // 2)  # Use first half only
            break
    
    # Get the end point of truncated 7-s data for shifting 8-s
    shift_displacement = 0
    shift_force = 0
    if df_7s is not None and not df_7s.empty:
        last_row = df_7s.iloc[-1]
        end_displacement = last_row["Displacement (mm)"]
        end_force = last_row["Force (N)"]
        
        # Find 8-s dataset start point to calculate shift
        for path in csv_paths:
            base_key = _derive_base_key_from_merged_filename(os.path.basename(path))
            if base_key and base_key.endswith('8-s'):
                df_8s_temp = pd.read_csv(path)
                if not df_8s_temp.empty:
                    start_row = df_8s_temp.iloc[0]
                    start_displacement = start_row["Displacement (mm)"]
                    start_force = start_row["Force (N)"]
                    
                    shift_displacement = end_displacement - start_displacement
                    shift_force = end_force - start_force
                break
    
    # Second pass: load all data with modifications
    for path in csv_paths:
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)
        
        base_key = _derive_base_key_from_merged_filename(os.path.basename(path))
        
        if base_key and base_key.endswith('7-s'):
            # Use only first half for 7-s
            df = df.head(len(df) // 2)
        elif base_key and base_key.endswith('8-s'):
            # Shift 8-s data to connect with 7-s endpoint
            df["Displacement (mm)"] = df["Displacement (mm)"] + shift_displacement
            df["Force (N)"] = df["Force (N)"] + shift_force
            # Also shift strain data if it exists
            if "fbg_direct_strain [με]" in df.columns:
                pass  # Don't shift strain, it's calculated later
        
        frames.append(df)
    
    if not frames:
        raise RuntimeError("No CSV files were loaded.")
    merged = pd.concat(frames, axis=0, ignore_index=True)
    return merged


def ensure_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# -----------------------------------------
# Extrapolation utilities
# -----------------------------------------

def compute_linear_extrapolation(df: pd.DataFrame, pattern1: str, x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute linear extrapolation from single dataset matching the given pattern."""
    
    # Filter data for the pattern
    data1 = None
    
    for name, group in df.groupby("source_file", sort=False):
        base_key = _derive_base_key_from_merged_filename(str(name))
        if base_key and pattern1 in base_key:
            data1 = group
            break
    
    if data1 is None:
        raise RuntimeError(f"Could not find data for pattern: {pattern1}")
    
    # Use only the single dataset
    combined_data = data1
    
    # Ensure numeric and drop NaN values
    x_vals = pd.to_numeric(combined_data[x_col], errors="coerce")
    y_vals = pd.to_numeric(combined_data[y_col], errors="coerce")
    
    valid_mask = ~(x_vals.isna() | y_vals.isna())
    x_vals = x_vals[valid_mask].values
    y_vals = y_vals[valid_mask].values
    
    if len(x_vals) < 2:
        raise RuntimeError("Insufficient data points for linear extrapolation")
    
    # Fit linear model
    coeffs = np.polyfit(x_vals, y_vals, deg=1)
    slope, intercept = coeffs
    
    # Get the full range of x values from ALL data
    all_x = pd.to_numeric(df[x_col], errors="coerce").dropna()
    global_x_min, global_x_max = all_x.min(), all_x.max()
    
    # Create extrapolation line spanning the entire data range
    x_extended = np.linspace(global_x_min, global_x_max, 200)
    y_extended = slope * x_extended + intercept
    
    return x_extended, y_extended


# -----------------------------------------
# Plotting functions
# -----------------------------------------

def plot_selective_force_vs_displacement(df: pd.DataFrame, output_dir: str, 
                                          extrapolation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> str:
    """Plot force vs displacement for selected datasets with optional extrapolation."""
    required_cols = ["Force (N)", "Displacement (mm)"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError("Missing required columns: " + ", ".join(missing))

    df = ensure_numeric(df, required_cols)

    plt.figure(figsize=(12, 8))
    
    # Plot each dataset
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
    markers = ['o', 's', '^', 'D', 'v', '*']
    color_idx = 0
    
    # Create color and marker mapping based on dataset type
    color_marker_map = {
        "15cm-16layers-6-s": (colors[0], markers[0]),
        "15cm-16layers-7-s": (colors[1], markers[1]), 
        "15cm-16layers-8-s": (colors[1], markers[1])
    }
    
    plotted_labels = set()  # Track which labels we've already added to legend
    
    for name, group in df.groupby("source_file", sort=False):
        base_key = _derive_base_key_from_merged_filename(str(name))
        original_key = base_key if base_key else os.path.splitext(str(name))[0]
        
        # Determine the base dataset type (ignoring truncation)
        dataset_type = None
        for pattern in ["15cm-16layers-6-s", "15cm-16layers-7-s", "15cm-16layers-8-s"]:
            if pattern in original_key:
                dataset_type = pattern
                break
        
        if dataset_type:
            color, marker = color_marker_map[dataset_type]
            # Only show label for first occurrence of each dataset type, but exclude 8-s from legend
            if dataset_type == "15cm-16layers-8-s":
                label = None  # Don't show 8-s in legend
            else:
                label = original_key if original_key not in plotted_labels else None
                if label:
                    plotted_labels.add(original_key)
        else:
            color = colors[color_idx % len(colors)]
            marker = markers[color_idx % len(markers)]
            label = original_key
            color_idx += 1
        
        # Plot scatter points only (no lines)
        plt.scatter(group["Displacement (mm)"], group["Force (N)"], 
                   color=color, marker=marker, s=40, alpha=0.9, edgecolors='black', linewidth=0.5, label=label)

    # Add extrapolation line if provided
    if extrapolation_data is not None:
        x_ext, y_ext = extrapolation_data
        plt.plot(x_ext, y_ext, '--', color='red', linewidth=1.5, alpha=0.8,
                label='Linear Extrapolation (6-s)')

    plt.xlabel("Displacement (mm)", fontsize=12)
    plt.ylabel("Force (N)", fontsize=12)
    plt.title("Force vs. Displacement", fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(fontsize=10)

    out_path = os.path.join(output_dir, "selective_force_vs_displacement.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_selective_strain_vs_displacement(df: pd.DataFrame, strain_col: str, output_dir: str,
                                          extrapolation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> str:
    """Plot strain vs displacement for selected datasets with optional extrapolation."""
    required_cols = [strain_col, "Displacement (mm)"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError("Missing required columns: " + ", ".join(missing))

    plt.figure(figsize=(12, 8))
    
    # Plot each dataset
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
    markers = ['o', 's', '^', 'D', 'v', '*']
    color_idx = 0
    
    # Create color and marker mapping based on dataset type
    color_marker_map = {
        "15cm-16layers-6-s": (colors[0], markers[0]),
        "15cm-16layers-7-s": (colors[1], markers[1]), 
        "15cm-16layers-8-s": (colors[1], markers[1])
    }
    
    plotted_labels = set()  # Track which labels we've already added to legend
    
    for name, group in df.groupby("source_file", sort=False):
        base_key = _derive_base_key_from_merged_filename(str(name))
        original_key = base_key if base_key else os.path.splitext(str(name))[0]
        
        # Determine the base dataset type (ignoring truncation)
        dataset_type = None
        for pattern in ["15cm-16layers-6-s", "15cm-16layers-7-s", "15cm-16layers-8-s"]:
            if pattern in original_key:
                dataset_type = pattern
                break
        
        if dataset_type:
            color, marker = color_marker_map[dataset_type]
            # Only show label for first occurrence of each dataset type, but exclude 8-s from legend
            if dataset_type == "15cm-16layers-8-s":
                label = None  # Don't show 8-s in legend
            else:
                label = original_key if original_key not in plotted_labels else None
                if label:
                    plotted_labels.add(original_key)
        else:
            color = colors[color_idx % len(colors)]
            marker = markers[color_idx % len(markers)]
            label = original_key
            color_idx += 1
        
        # Plot scatter points only (no lines)
        plt.scatter(group["Displacement (mm)"], group[strain_col], 
                   color=color, marker=marker, s=40, alpha=0.9, edgecolors='black', linewidth=0.5, label=label)

    # Add extrapolation line if provided
    if extrapolation_data is not None:
        x_ext, y_ext = extrapolation_data
        plt.plot(x_ext, y_ext, '--', color='red', linewidth=1.5, alpha=0.8,
                label='Linear Extrapolation (6-s)')

    plt.xlabel("Displacement (mm)", fontsize=12)
    plt.ylabel("Strain [με]", fontsize=12)
    plt.title("Strain vs. Displacement", fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(fontsize=10)

    out_path = os.path.join(output_dir, "selective_strain_vs_displacement.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# -----------------------------------------
# FBG strain computation (simplified from original)
# -----------------------------------------

def _load_raw_central_baseline(raw_dir: str, base_key: str) -> Optional[float]:
    """Load the corresponding interrogator TXT and return first valid central channel wavelength (WL 2[nm])."""
    txt_path = os.path.join(raw_dir, f"{base_key}-interrogator.txt")
    if not os.path.isfile(txt_path):
        return None

    # Robust loader similar to merge_fbg.load_interrogator
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
    except Exception:
        return None

    # find first data line
    data_start = 0
    iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}T")
    for i, line in enumerate(lines[:50]):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if iso_re.match(stripped):
            if len(parts) >= 3:
                data_start = i
                break
        else:
            try:
                float(parts[0])
                if len(parts) >= 2:
                    data_start = i
                    break
            except Exception:
                pass

    # determine columns
    if data_start >= len(lines):
        return None
    n_tokens = len(lines[data_start].split())
    if n_tokens < 2:
        return None
    if iso_re.match(lines[data_start]):
        names = ["Timestamp", "Time_s"]
        wl_count = max(0, n_tokens - 2)
    else:
        names = ["Time_s"]
        wl_count = max(0, n_tokens - 1)
    names += [f"WL {i}[nm]" for i in range(1, wl_count + 1)]

    try:
        df_raw = pd.read_csv(
            txt_path,
            sep=r"\s+",
            header=None,
            names=names,
            skiprows=data_start,
            engine="python",
        )
    except Exception:
        return None

    central_col = "WL 2[nm]"
    if central_col not in df_raw.columns:
        return None
    series = pd.to_numeric(df_raw[central_col], errors="coerce")
    series = series.dropna()
    if series.empty:
        return None
    # Use the first valid as baseline λ0
    return float(series.iloc[0])


def compute_fbg_direct_microstrain(df: pd.DataFrame, pe: float = 0.22) -> pd.Series:
    """Compute microstrain using ε = (Δλ/λ0)/(1 - p_e) from the central channel per file."""
    central_col = "WL_ch2" if "WL_ch2" in df.columns else ("WL 2[nm]" if "WL 2[nm]" in df.columns else None)
    if central_col is None:
        raise KeyError("Central Bragg channel not found. Expected 'WL_ch2' or 'WL 2[nm]'.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(script_dir, "interrogator-data")

    micro_list: List[pd.Series] = []

    for fname, group in df.groupby("source_file", sort=False):
        base_key = _derive_base_key_from_merged_filename(str(fname)) or ""
        lambda0 = _load_raw_central_baseline(raw_dir, base_key) if base_key else None

        series = pd.to_numeric(group[central_col], errors="coerce")

        # Fallback λ0: first valid value in the group if raw not available
        if lambda0 is None:
            first_valid = series.dropna().iloc[0] if not series.dropna().empty else np.nan
            lambda0 = float(first_valid) if np.isfinite(first_valid) else np.nan

        # Heuristic: decide if series represents Δλ (already zero-centered) or absolute WL
        series_valid = series.dropna()
        looks_delta = False
        if not series_valid.empty:
            median_abs = float(abs(series_valid.median()))
            span = float(series_valid.max() - series_valid.min())
            # If magnitude is small (<< 100 nm) and centered near 0, treat as Δλ
            looks_delta = median_abs < 10.0 and span < 50.0

        if looks_delta:
            delta_lambda = series
        else:
            # Absolute wavelengths → subtract baseline
            delta_lambda = series - lambda0

        eps = (delta_lambda / lambda0) / (1.0 - pe)
        micro = eps * 1000000
        micro_list.append(pd.Series(micro.values, index=group.index))

    out = pd.concat(micro_list).sort_index()
    out.name = "fbg_direct_strain [με]"
    return out


# -----------------------------------------
# Main entry
# -----------------------------------------

def main(base_output_dir: Optional[str] = None) -> None:
    # Define the target patterns we want to plot
    target_patterns = ["15cm-16layers-6-s", "15cm-16layers-7-s", "15cm-16layers-8-s"]
    extrapolation_pattern = "15cm-16layers-6-s"  # Use only 6-s for linear fit
    
    latest_dir = find_latest_output_directory(base_output_dir)
    if not latest_dir:
        raise RuntimeError("No output directories found under the provided base directory.")
    print(f"Using latest output directory: {latest_dir}")

    csv_files = list_merged_csvs(latest_dir)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in: {latest_dir}")
    
    # Filter CSV files to only include our target patterns
    filtered_csvs = filter_csvs_by_patterns(csv_files, target_patterns)
    if not filtered_csvs:
        raise RuntimeError(f"No CSV files found matching patterns: {target_patterns}")
    
    print(f"Found {len(filtered_csvs)} CSV file(s) matching target patterns:")
    for csv_file in filtered_csvs:
        print(f"  - {os.path.basename(csv_file)}")

    df = load_and_merge_csvs(filtered_csvs)

    # Check required columns
    required_cols = ["Force (N)", "Displacement (mm)"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Compute extrapolation for Force vs Displacement
    try:
        force_extrapolation = compute_linear_extrapolation(
            df, extrapolation_pattern, "Displacement (mm)", "Force (N)"
        )
        print("Computed force vs displacement extrapolation")
    except Exception as e:
        print(f"Warning: Could not compute force extrapolation: {e}")
        force_extrapolation = None

    # Plot Force vs Displacement with extrapolation
    fvd_path = plot_selective_force_vs_displacement(df, latest_dir, force_extrapolation)
    print(f"Saved plot: {fvd_path}")

    # Compute FBG strain and plot with extrapolation
    try:
        df["fbg_direct_strain [με]"] = compute_fbg_direct_microstrain(df)
        
        # Compute extrapolation for strain
        strain_extrapolation = compute_linear_extrapolation(
            df, extrapolation_pattern, "Displacement (mm)", "fbg_direct_strain [με]"
        )
        
        svd_path = plot_selective_strain_vs_displacement(
            df, "fbg_direct_strain [με]", latest_dir, strain_extrapolation
        )
        print(f"Saved plot: {svd_path}")
        
    except Exception as e:
        print(f"Warning: Could not compute/plot FBG strain: {e}")

    print("\nSelective plotting completed successfully!")


if __name__ == "__main__":
    main()
