#!/usr/bin/env python3
"""
Visualize median filtering on interrogator wavelength data.
Creates class-ready plots:
1) full signal: raw vs median-filtered
2) zoomed view to show spike suppression
3) residual (raw - filtered)
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import medfilt, find_peaks


FILEPATH = "interrogator-data/15cm-16layers-1-s-interrogator.txt"
OUT_DIR = "exercise_outputs"
CHANNEL = 2
KERNEL_SIZE = 7

# Zoom range in sample index for class explanation
ZOOM_START = 4200
ZOOM_END = 5200
PEAK_HALF_WINDOW = 80
DIFF_PERCENTILE = 99.7
DIFF_MIN_ABS_NM = 0.003
DIFF_REGION_HALF_WINDOW = 40
DIFF_MAX_REGIONS = 4


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\t+", engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_wl_column(df: pd.DataFrame, channel: int) -> str:
    pat = re.compile(rf"^WL\s+{channel}\[nm\]$")
    for c in df.columns:
        if pat.match(c):
            return c
    raise KeyError(f"WL {channel}[nm] not found. Columns: {list(df.columns)}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_data(FILEPATH)
    wl_col = get_wl_column(df, CHANNEL)
    raw = pd.to_numeric(df[wl_col], errors="coerce").to_numpy(dtype=float)
    x = np.arange(len(raw))

    k = KERNEL_SIZE if KERNEL_SIZE % 2 == 1 else KERNEL_SIZE + 1
    filtered = medfilt(raw, kernel_size=k)
    residual = raw - filtered

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    # Panel 1: full signal
    axes[0].plot(x, raw, linewidth=0.6, alpha=0.6, label="Raw")
    axes[0].plot(x, filtered, linewidth=1.0, alpha=0.95, label=f"Median filtered (k={k})")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Wavelength [nm]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    # Panel 2: zoomed region
    z0 = max(0, ZOOM_START)
    z1 = min(len(raw), ZOOM_END)
    axes[1].plot(x[z0:z1], raw[z0:z1], linewidth=0.8, alpha=0.65, label="Raw")
    axes[1].plot(x[z0:z1], filtered[z0:z1], linewidth=1.2, alpha=0.95, label=f"Median filtered (k={k})")
    axes[1].set_xlabel("Sample index (zoom)")
    axes[1].set_ylabel("Wavelength [nm]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    # Panel 3: residual
    axes[2].plot(x, residual, linewidth=0.6, color="tab:purple")
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
    axes[2].set_xlabel("Sample index")
    axes[2].set_ylabel("Raw - Filtered [nm]")
    axes[2].grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"exercise_median_filter_WL{CHANNEL}.png")
    plt.savefig(out_path, dpi=180)
    plt.close(fig)

    # Dedicated "most affected point" zoom figure for teaching
    # Center the zoom where filtering changed the signal the most.
    finite_mask = np.isfinite(residual)
    if np.any(finite_mask):
        peak_idx = int(np.nanargmax(np.abs(residual)))
    else:
        # Fallback if unexpected all-NaN residual
        peaks, _ = find_peaks(filtered, prominence=np.std(filtered) * 0.3, distance=30)
        if len(peaks) == 0:
            peak_idx = int(np.argmax(filtered))
        else:
            peak_idx = int(peaks[np.argmax(filtered[peaks])])

    p0 = max(0, peak_idx - PEAK_HALF_WINDOW)
    p1 = min(len(raw), peak_idx + PEAK_HALF_WINDOW)

    fig2, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(x[p0:p1], raw[p0:p1], linewidth=0.9, alpha=0.65, label="Raw")
    ax.plot(x[p0:p1], filtered[p0:p1], linewidth=1.4, alpha=0.95, label=f"Median filtered (k={k})")
    ax.axvline(peak_idx, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.set_xlabel("Sample index (single-peak zoom)")
    ax.set_ylabel("Wavelength [nm]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()

    out_peak = os.path.join(OUT_DIR, f"exercise_median_filter_WL{CHANNEL}_single_peak_zoom.png")
    plt.savefig(out_peak, dpi=220)
    plt.close(fig2)

    # Spectrum view for the same peak window (raw vs filtered)
    raw_win = raw[p0:p1]
    fil_win = filtered[p0:p1]
    n = len(raw_win)
    if n > 8:
        # Remove DC component to focus on fluctuations/noise content
        raw_d = raw_win - np.nanmean(raw_win)
        fil_d = fil_win - np.nanmean(fil_win)

        raw_d = np.nan_to_num(raw_d, nan=0.0)
        fil_d = np.nan_to_num(fil_d, nan=0.0)

        # Frequency axis in cycles/sample (normalized frequency)
        freqs = np.fft.rfftfreq(n, d=1.0)
        raw_spec = np.abs(np.fft.rfft(raw_d))
        fil_spec = np.abs(np.fft.rfft(fil_d))

        fig4, axes4 = plt.subplots(2, 1, figsize=(9, 6.2))
        axes4[0].plot(x[p0:p1], raw_win, linewidth=0.9, alpha=0.65, label="Raw")
        axes4[0].plot(x[p0:p1], fil_win, linewidth=1.3, alpha=0.95, label=f"Median filtered (k={k})")
        axes4[0].axvline(peak_idx, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.85)
        axes4[0].set_xlabel("Sample index (single-peak window)")
        axes4[0].set_ylabel("Wavelength [nm]")
        axes4[0].grid(True, alpha=0.25)
        axes4[0].legend(loc="best")

        axes4[1].plot(freqs, raw_spec, linewidth=1.0, alpha=0.8, label="Raw spectrum")
        axes4[1].plot(freqs, fil_spec, linewidth=1.2, alpha=0.9, label="Filtered spectrum")
        axes4[1].set_xlabel("Normalized frequency [cycles/sample]")
        axes4[1].set_ylabel("Magnitude")
        axes4[1].grid(True, alpha=0.25)
        axes4[1].legend(loc="best")
        plt.tight_layout()

        out_spec = os.path.join(OUT_DIR, f"exercise_median_filter_WL{CHANNEL}_single_peak_spectrum.png")
        plt.savefig(out_spec, dpi=220)
        plt.close(fig4)
        print(f"Saved: {out_spec}")

    mad_raw = np.median(np.abs(raw - np.median(raw)))
    mad_res = np.median(np.abs(residual - np.median(residual)))
    print(f"Saved: {out_path}")
    print(f"Saved: {out_peak}")
    print(f"Most-affected index used for zoom: {peak_idx}")

    # "Only where filter made a huge difference" figure
    abs_diff = np.abs(residual)
    finite_diff = abs_diff[np.isfinite(abs_diff)]
    if finite_diff.size > 0:
        thr = max(float(np.percentile(finite_diff, DIFF_PERCENTILE)), DIFF_MIN_ABS_NM)
        idx = np.where(abs_diff >= thr)[0]
    else:
        thr = DIFF_MIN_ABS_NM
        idx = np.array([], dtype=int)

    if idx.size > 0:
        # Group nearby indices into regions and keep top regions by max |diff|
        regions = []
        start = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if i - prev <= 12:
                prev = i
            else:
                regions.append((start, prev))
                start = i
                prev = i
        regions.append((start, prev))

        scored = []
        for s, e in regions:
            region_max = float(np.nanmax(abs_diff[s:e + 1]))
            scored.append((region_max, s, e))
        scored.sort(reverse=True)
        chosen = scored[:DIFF_MAX_REGIONS]

        fig3, axes3 = plt.subplots(len(chosen), 1, figsize=(10, 2.8 * len(chosen)), sharex=False)
        if len(chosen) == 1:
            axes3 = [axes3]

        for ax, (score, s, e) in zip(axes3, chosen):
            z0 = max(0, s - DIFF_REGION_HALF_WINDOW)
            z1 = min(len(raw), e + DIFF_REGION_HALF_WINDOW + 1)
            ax.plot(x[z0:z1], raw[z0:z1], linewidth=0.95, alpha=0.60, label="Raw")
            ax.plot(x[z0:z1], filtered[z0:z1], linewidth=1.35, alpha=0.95, label=f"Median filtered (k={k})")
            ax.axvspan(s, e, color="tab:red", alpha=0.15)
            ax.set_xlabel("Sample index")
            ax.set_ylabel("Wavelength [nm]")
            ax.grid(True, alpha=0.25)
            ax.text(
                0.01,
                0.93,
                f"max |raw-filtered| = {score:.4f} nm",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
            )
            ax.legend(loc="best")

        plt.tight_layout()
        out_diff = os.path.join(OUT_DIR, f"exercise_median_filter_WL{CHANNEL}_huge_diff_regions.png")
        plt.savefig(out_diff, dpi=220)
        plt.close(fig3)
        print(f"Saved: {out_diff}")
        print(f"Huge-difference threshold: |raw-filtered| >= {thr:.6f} nm (p{DIFF_PERCENTILE})")
    else:
        print("No high-difference regions found for huge-difference plot.")

    print(f"Channel: {wl_col}")
    print(f"MAD raw: {mad_raw:.6f} nm")
    print(f"MAD residual: {mad_res:.6f} nm")


if __name__ == "__main__":
    main()

