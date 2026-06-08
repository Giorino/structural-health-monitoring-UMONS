#!/usr/bin/env python3
"""
Exercise 5: Channel agreement via correlation
----------------------------------------------
Plot scatter plots between raw Δλ signals from multiple Bragg channels and
compute Pearson correlation coefficients.

Inputs:
  - Raw interrogator file with columns like:
      Timestamp
      Time [s]
      WL 1[nm], WL 2[nm], WL 3[nm], ...

Outputs:
  - PNG scatter plots with regression line, saved to exercise_outputs/...
"""

from __future__ import annotations

import os
import re
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# User settings (edit if needed)
# ----------------------------
FILEPATH = "interrogator-data/15cm-16layers-1-s-interrogator.txt"
OUT_DIR = "exercise_outputs"

CHANNELS = [1, 2, 3]  # Bragg channels to compare (WL 1[nm], WL 2[nm], ...)
N_BASELINE = 2000     # baseline window from the beginning
BASELINE_MODE = "median"  # "median" or "mean"

MAX_POINTS = 6000     # downsample for scatter speed/visual clarity


def load_interrogator(path: str) -> pd.DataFrame:
    # The file is tab-separated (header uses tabs). Use regex sep to be safe.
    df = pd.read_csv(path, sep=r"\t+", engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_time_column(df: pd.DataFrame) -> str:
    # Expected column in this repo is exactly "Time [s]"
    if "Time [s]" in df.columns:
        return "Time [s]"
    # Fallback heuristics
    for c in df.columns:
        cl = c.lower()
        if "time" in cl and "s" in cl:
            return c
    raise ValueError(f"Could not find a time column. Columns: {list(df.columns)}")


def find_wl_columns(df: pd.DataFrame) -> dict[int, str]:
    # Columns are like: "WL 1[nm]"
    wl_map: dict[int, str] = {}
    pattern = re.compile(r"^WL\s+(\d+)\[nm\]$")
    for c in df.columns:
        m = pattern.match(c)
        if m:
            wl_map[int(m.group(1))] = c
    return wl_map


def compute_delta_lambda(wl: np.ndarray, n_baseline: int, baseline_mode: str) -> tuple[np.ndarray, float]:
    n = int(min(n_baseline, len(wl)))
    wl0 = wl[:n]
    wl0 = wl0[np.isfinite(wl0)]
    if wl0.size == 0:
        raise ValueError("Baseline window has no finite data.")
    lam0 = float(np.median(wl0) if baseline_mode == "median" else np.mean(wl0))
    dlam = wl - lam0
    return dlam, lam0


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # Fit y = a*x + b
    a, b = np.polyfit(x, y, deg=1)
    return float(a), float(b)


def downsample_for_plot(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return x[idx], y[idx]


def main() -> None:
    df = load_interrogator(FILEPATH)
    time_col = find_time_column(df)
    wl_map = find_wl_columns(df)

    missing = [ch for ch in CHANNELS if ch not in wl_map]
    if missing:
        raise ValueError(f"Missing requested channels {missing}. Available WL columns: {list(wl_map.keys())}")

    # Compute Δλ for each selected channel (raw signal, no filtering).
    delta_by_channel: dict[int, np.ndarray] = {}
    lambda0_by_channel: dict[int, float] = {}

    for ch in CHANNELS:
        col = wl_map[ch]
        wl = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        dlam, lam0 = compute_delta_lambda(wl, N_BASELINE, BASELINE_MODE)
        delta_by_channel[ch] = dlam
        lambda0_by_channel[ch] = lam0
        print(f"[Exercise 5] WL {ch}[nm]: lambda0({BASELINE_MODE}, first {min(N_BASELINE, len(wl))} pts) = {lam0:.6f} nm")

    os.makedirs(OUT_DIR, exist_ok=True)

    # Create scatter plots for each channel pair.
    for ci, cj in combinations(CHANNELS, 2):
        x = delta_by_channel[ci]
        y = delta_by_channel[cj]
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if x.size == 0:
            print(f"[Exercise 5] WL {ci} vs WL {cj}: no finite overlap, skipping.")
            continue

        r = float(np.corrcoef(x, y)[0, 1])

        x_plot, y_plot = downsample_for_plot(x, y, MAX_POINTS)
        a, b = fit_linear_regression(x_plot, y_plot)
        y_fit = a * x_plot + b

        plt.figure(figsize=(7.5, 5.2))
        plt.scatter(x_plot, y_plot, s=7, alpha=0.35, linewidths=0)
        plt.plot(x_plot, y_fit, color="red", linewidth=2)
        plt.grid(True, alpha=0.25)
        plt.xlabel(f"Δλ(t): WL {ci} - λ0 (nm)")
        plt.ylabel(f"Δλ(t): WL {cj} - λ0 (nm)")
        # Keep Pearson r inside the axes (no top title banner).
        plt.gca().text(
            0.05,
            0.95,
            f"Pearson r = {r:.3f}",
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, edgecolor="none"),
        )
        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, f"exercise5_corr_scatter_WL{ci}_vs_WL{cj}.png")
        plt.savefig(out_path, dpi=180)
        plt.close()

        print(f"[Exercise 5] Pair WL {ci} vs WL {cj}: Pearson r = {r:.6f}")
        print(f"            Saved: {out_path}")


if __name__ == "__main__":
    main()

