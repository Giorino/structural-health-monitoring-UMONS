#!/usr/bin/env python3
"""Plot wavelength vs time for each interrogator TXT in one tall PNG.

For each file under --input-dir (default: 'interrogator-data' next to this script),
the script:
- loads the TXT robustly via merge_fbg.load_interrogator
- detects WL columns and selects up to N channels (default 3) with the most data
- builds a time axis in seconds (from 'Time_s' or derived from 'Timestamp')
- plots WL_ch vs time for that file in its own subplot

The output is a single tall PNG placed next to the input directory.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def find_wl_columns(df: pd.DataFrame) -> List[str]:
    """Detect wavelength columns by header patterns."""
    pattern = re.compile(r"\bWL\b|\bWL\s*\d+|\bWavelength", re.I)
    candidates = [c for c in df.columns if pattern.search(str(c))]
    if not candidates:
        candidates = [c for c in df.columns if df[c].dtype.kind in "fif"]
    return candidates


def select_wl_channels(df: pd.DataFrame, wl_cols: List[str], channels_to_use: int = 3) -> List[str]:
    if not wl_cols:
        return []
    non_null_counts = {c: int(pd.Series(df[c]).notna().sum()) for c in wl_cols}
    ranked = sorted(wl_cols, key=lambda c: (-non_null_counts[c], wl_cols.index(c)))
    return ranked[:channels_to_use]


def load_interrogator(txt_path: str) -> pd.DataFrame:
    """Robust TXT loader similar to merge_fbg.load_interrogator (no SciPy dep)."""
    path = str(txt_path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}T")
    data_start = 0
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

    first_tokens = lines[data_start].split()
    n_tokens = len(first_tokens)
    if re.match(r"^\d{4}-\d{2}-\d{2}T", lines[data_start]):
        names = ["Timestamp", "Time_s"]
        wl_count = max(0, n_tokens - 2)
    else:
        names = ["Time_s"]
        wl_count = max(0, n_tokens - 1)
    names += [f"WL {i}[nm]" for i in range(1, wl_count + 1)]

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=names,
        skiprows=data_start,
        engine="python",
    )
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df


def build_time_axis_seconds(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """Return (time_seconds, label) from 'Time_s' or 'Timestamp', else index.

    If 'Timestamp' exists, convert to seconds from first valid timestamp.
    """
    # Prefer explicit numeric seconds if available
    if "Time_s" in df.columns and pd.api.types.is_numeric_dtype(df["Time_s"]):
        t = pd.to_numeric(df["Time_s"], errors="coerce").to_numpy(dtype=float)
        return t, "Time (s)"

    # Try datetime timestamp
    if "Timestamp" in df.columns:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce")
        if ts.notna().any():
            t0 = ts.dropna().iloc[0]
            dt = (ts - t0).dt.total_seconds()
            return dt.to_numpy(dtype=float), "Time (s)"

    # Fallback to sample index as seconds
    idx = np.arange(len(df), dtype=float)
    return idx, "Sample index (a.u.)"


def downsample_for_plot(x: np.ndarray, y: np.ndarray, max_points: int = 8000) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n <= max_points:
        return x, y
    step = int(np.ceil(n / max_points))
    return x[::step], y[::step]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot wavelength vs time per interrogator file into one PNG.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing '*-interrogator.txt' files (default: 'interrogator-data' next to this script).",
    )
    parser.add_argument(
        "--channels-to-use",
        type=int,
        default=3,
        help="Maximum number of WL channels to plot per file (default 3).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=8000,
        help="Downsample each series to at most this many points for plotting (default 8000).",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "interrogator-data")
    input_dir = args.input_dir or default_input

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    txt_files = sorted(glob.glob(os.path.join(input_dir, "*-interrogator.txt")))
    if not txt_files:
        raise RuntimeError(f"No '*-interrogator.txt' files found in {input_dir}")

    print(f"Found {len(txt_files)} interrogator file(s) in: {input_dir}")

    # Preload data to know how many subplots we need
    loaded: List[Tuple[str, pd.DataFrame, List[str], np.ndarray, str]] = []
    for path in txt_files:
        try:
            df = load_interrogator(path)
        except Exception as e:
            print(f"  Skipping {os.path.basename(path)}: failed to load ({e})")
            continue
        wl_cols_all = find_wl_columns(df)
        wl_cols = select_wl_channels(df, wl_cols_all, channels_to_use=args.channels_to_use) if wl_cols_all else []
        if not wl_cols:
            print(f"  Skipping {os.path.basename(path)}: no WL columns detected")
            continue
        t_s, t_label = build_time_axis_seconds(df)
        loaded.append((path, df, wl_cols, t_s, t_label))

    if not loaded:
        raise RuntimeError("No valid interrogator files could be loaded.")

    # Figure sizing: allocate ~2.6 inches per file, cap width to 14 inches
    n = len(loaded)
    fig_height = max(2.0, min(3.2, 2.6)) * n  # constant per subplot, adjustable if needed
    fig, axes = plt.subplots(n, 1, figsize=(14, fig_height), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (path, df, wl_cols, t_s, t_label) in zip(axes, loaded):
        # Choose time units (auto minutes if long)
        x = t_s.copy()
        x_label = t_label
        if np.nanmax(x) - np.nanmin(x) > 600:  # > 10 minutes
            x = x / 60.0
            x_label = "Time (min)"

        # Plot selected WL channels
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c"])        
        for i, col in enumerate(wl_cols):
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            xs, ys = downsample_for_plot(x, y, max_points=args.max_points)
            ax.plot(xs, ys, linewidth=0.8, label=col, color=colors[i % len(colors)])

        title = os.path.basename(path)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Wavelength (nm)")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        ax.legend(fontsize=8, loc="best")

    axes[-1].set_xlabel(x_label)
    fig.suptitle("Wavelength vs Time per Interrogator File", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = os.path.join(input_dir, "wavelengths_by_file.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

