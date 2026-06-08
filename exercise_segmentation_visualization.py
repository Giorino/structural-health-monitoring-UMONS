#!/usr/bin/env python3
"""
Visualize segmentation on one interrogator file.

Outputs:
- exercise_outputs/exercise_segmentation_overview.png
- exercise_outputs/exercise_segmentation_zoom.png
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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILEPATH = os.path.join(SCRIPT_DIR, "interrogator-data/15cm-16layers-1-s-interrogator.txt")
OUT_DIR = os.path.join(SCRIPT_DIR, "exercise_outputs")

# Segmentation parameters (same idea as merge_fbg.py)
CHANNELS_FOR_REP = [1, 2, 3]
SMOOTH_KERNEL = 7
MIN_SEGMENT_LENGTH = 5
PEAK_DISTANCE = 20
PEAK_PROMINENCE = 0.1
PEAK_HEIGHT = None  # auto if None

# Zoom window for a class-friendly figure
ZOOM_START = 6500
ZOOM_END = 7800
SINGLE_SEGMENT_HALF_WINDOW = 70


def load_interrogator(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\t+", engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_time_column(df: pd.DataFrame) -> str:
    if "Time [s]" in df.columns:
        return "Time [s]"
    for c in df.columns:
        cl = c.lower()
        if "time" in cl and "s" in cl:
            return c
    return ""


def find_wl_columns(df: pd.DataFrame) -> list[str]:
    pat = re.compile(r"^WL\s+(\d+)\[nm\]$")
    wl_cols = []
    for c in df.columns:
        m = pat.match(c)
        if not m:
            continue
        idx = int(m.group(1))
        if idx in CHANNELS_FOR_REP:
            wl_cols.append((idx, c))
    wl_cols = [name for _, name in sorted(wl_cols, key=lambda t: t[0])]
    if not wl_cols:
        # fallback: use any WL columns
        wl_cols = [c for c in df.columns if pat.match(c)]
    if not wl_cols:
        raise KeyError("No WL columns found.")
    return wl_cols


def smooth_signal(signal: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return medfilt(signal, kernel_size=k)


def detect_segments(
    signal: np.ndarray,
    peak_height: float | None,
    min_segment_length: int,
    peak_distance: int,
    peak_prominence: float,
):
    if peak_height is None:
        noise_level = np.median(np.abs(signal - np.median(signal)))
        peak_height = max(0.05, 4 * 1.4826 * noise_level)

    peaks, properties = find_peaks(
        signal,
        height=peak_height,
        width=min_segment_length,
        distance=peak_distance,
        prominence=peak_prominence,
    )

    if len(peaks) == 0:
        return [], peaks, properties, peak_height

    left_edges = np.floor(properties["left_ips"]).astype(int)
    right_edges = np.ceil(properties["right_ips"]).astype(int)

    segments = []
    for l, r in zip(left_edges, right_edges):
        if r > l + min_segment_length:
            segments.append((l, r))
    segments.sort(key=lambda x: x[0])

    return segments, peaks, properties, peak_height


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_interrogator(FILEPATH)
    time_col = get_time_column(df)
    wl_cols = find_wl_columns(df)

    # Representative signal: mean of selected WL channels
    rep_signal = df[wl_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy(dtype=float)
    rep_signal = np.nan_to_num(rep_signal, nan=np.nanmedian(rep_signal))
    sm = smooth_signal(rep_signal, kernel_size=SMOOTH_KERNEL)

    segments, peaks, properties, used_thr = detect_segments(
        sm,
        peak_height=PEAK_HEIGHT,
        min_segment_length=MIN_SEGMENT_LENGTH,
        peak_distance=PEAK_DISTANCE,
        peak_prominence=PEAK_PROMINENCE,
    )

    x = np.arange(len(sm))
    if time_col:
        raw_t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        # The 'Time [s]' column only has integer precision, causing a staircase effect when plotted.
        # We reconstruct a smooth, continuous time array assuming constant sampling frequency.
        valid_t = raw_t[~np.isnan(raw_t)]
        t = np.linspace(valid_t[0], valid_t[-1], len(raw_t)) if len(valid_t) > 0 else x.astype(float)
    else:
        t = x.astype(float)

    # Overview figure
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.plot(t, rep_signal, linewidth=0.5, alpha=0.35, label="Representative signal (raw)")
    ax.plot(t, sm, linewidth=1.0, alpha=0.95, label=f"Smoothed (median k={SMOOTH_KERNEL})")
    if len(peaks) > 0:
        ax.scatter(t[peaks], sm[peaks], s=12, c="red", label="Detected peaks", zorder=4)
    for l, r in segments:
        ax.axvspan(t[l], t[r], color="green", alpha=0.10)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Time [s]" if time_col else "Sample index")
    ax.set_ylabel("Representative wavelength [nm]")
    ax.legend(loc="best")
    ax.text(
        0.01,
        0.98,
        f"Segments={len(segments)} | Peaks={len(peaks)} | height_thr={used_thr:.4f} | distance={PEAK_DISTANCE} | prominence={PEAK_PROMINENCE}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
    )
    plt.tight_layout()
    out_overview = os.path.join(OUT_DIR, "exercise_segmentation_overview.png")
    plt.savefig(out_overview, dpi=600)
    plt.close(fig)

    # Zoom figure
    z0 = max(0, ZOOM_START)
    z1 = min(len(sm), ZOOM_END)
    fig2, ax2 = plt.subplots(figsize=(12, 4.8))
    ax2.plot(t[z0:z1], rep_signal[z0:z1], linewidth=0.65, alpha=0.35, label="Raw")
    ax2.plot(t[z0:z1], sm[z0:z1], linewidth=1.15, alpha=0.95, label="Smoothed")

    peak_mask = (peaks >= z0) & (peaks < z1)
    pz = peaks[peak_mask]
    if len(pz) > 0:
        ax2.scatter(t[pz], sm[pz], s=20, c="red", label="Peaks", zorder=5)
    for l, r in segments:
        if r < z0 or l >= z1:
            continue
        ax2.axvspan(t[max(l, z0)], t[min(r, z1 - 1)], color="green", alpha=0.14)

    ax2.grid(True, alpha=0.25)
    ax2.set_xlabel("Time [s]" if time_col else "Sample index")
    ax2.set_ylabel("Representative wavelength [nm]")
    plt.tight_layout()
    out_zoom = os.path.join(OUT_DIR, "exercise_segmentation_zoom.png")
    plt.savefig(out_zoom, dpi=600)
    plt.close(fig2)

    # Single-segment figure (one peak + boundaries)
    if len(segments) > 0:
        # Pick the segment with largest peak height in smoothed signal
        seg_best_idx = 0
        seg_best_peak = -np.inf
        seg_peak_pos = None
        for i, (l, r) in enumerate(segments):
            # peaks inside this segment
            in_seg = peaks[(peaks >= l) & (peaks <= r)]
            if len(in_seg) == 0:
                local_peak = l + int(np.argmax(sm[l:r + 1]))
            else:
                local_peak = int(in_seg[np.argmax(sm[in_seg])])
            val = float(sm[local_peak])
            if val > seg_best_peak:
                seg_best_peak = val
                seg_best_idx = i
                seg_peak_pos = local_peak

        l, r = segments[seg_best_idx]
        peak_pos = int(seg_peak_pos if seg_peak_pos is not None else (l + r) // 2)
        w0 = max(0, peak_pos - SINGLE_SEGMENT_HALF_WINDOW)
        w1 = min(len(sm), peak_pos + SINGLE_SEGMENT_HALF_WINDOW + 1)

        fig3, ax3 = plt.subplots(figsize=(10, 4.5))
        ax3.plot(t[w0:w1], rep_signal[w0:w1], linewidth=0.7, alpha=0.35, label="Raw representative")
        ax3.plot(t[w0:w1], sm[w0:w1], linewidth=1.2, alpha=0.95, label="Smoothed")
        ax3.axvline(t[peak_pos], color="red", linestyle="--", linewidth=1.2, label="Peak")
        ax3.axvline(t[l], color="green", linestyle="-", linewidth=1.1, alpha=0.9, label="Segment start/end")
        ax3.axvline(t[r], color="green", linestyle="-", linewidth=1.1, alpha=0.9)
        ax3.axvspan(t[l], t[r], color="green", alpha=0.12)
        ax3.grid(True, alpha=0.25)
        ax3.set_xlabel("Time [s]" if time_col else "Sample index")
        ax3.set_ylabel("Representative wavelength [nm]")
        ax3.legend(loc="best")
        ax3.text(
            0.01,
            0.97,
            f"Segment #{seg_best_idx + 1} | start={l}, peak={peak_pos}, end={r}",
            transform=ax3.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
        )
        plt.tight_layout()
        out_single = os.path.join(OUT_DIR, "exercise_segmentation_single_peak.png")
        plt.savefig(out_single, dpi=600)
        plt.close(fig3)
    else:
        out_single = ""

    print(f"File: {FILEPATH}")
    print(f"Representative channels: {wl_cols}")
    print(f"Detected peaks: {len(peaks)}")
    print(f"Detected segments: {len(segments)}")
    print(f"Used peak height threshold: {used_thr:.6f}")
    print(f"Saved: {out_overview}")
    print(f"Saved: {out_zoom}")
    if out_single:
        print(f"Saved: {out_single}")


if __name__ == "__main__":
    main()

