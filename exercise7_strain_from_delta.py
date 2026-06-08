#!/usr/bin/env python3
"""
Exercise 7: Convert raw Δλ to strain (physics conversion, no filtering)
----------------------------------------------------------------------------
Compute:
  ε(t) = Δλ(t) / (λ0 * (1 - p_e))
and plot strain in microstrain.

Inputs:
  - Raw interrogator file with columns:
      Time [s]
      WL i[nm]

Outputs:
  - PNG plot saved to exercise_outputs/...
"""

from __future__ import annotations

import os
import re

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

TARGET_CHANNEL = 2   # WL 2[nm] by default
N_BASELINE = 2000     # baseline window from the beginning
BASELINE_MODE = "median"  # "median" or "mean"

PE = 0.22              # photoelastic coefficient for silica fiber


def load_interrogator(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\t+", engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_time_column(df: pd.DataFrame) -> str:
    if "Time [s]" in df.columns:
        return "Time [s]"
    for c in df.columns:
        cl = c.lower()
        if "time" in cl and "s" in cl:
            return c
    raise ValueError(f"Could not find a time column. Columns: {list(df.columns)}")


def find_wl_column_for_channel(df: pd.DataFrame, channel: int) -> str:
    pattern = re.compile(rf"^WL\s+{channel}\[nm\]$")
    for c in df.columns:
        if pattern.match(c):
            return c
    raise ValueError(f"Could not find WL {channel}[nm]. Columns: {list(df.columns)}")


def compute_delta_and_lambda0(wl: np.ndarray, n_baseline: int, baseline_mode: str) -> tuple[np.ndarray, float]:
    n = int(min(n_baseline, len(wl)))
    wl0 = wl[:n]
    wl0 = wl0[np.isfinite(wl0)]
    if wl0.size == 0:
        raise ValueError("Baseline window has no finite data.")
    lam0 = float(np.median(wl0) if baseline_mode == "median" else np.mean(wl0))
    dlam = wl - lam0
    return dlam, lam0


def main() -> None:
    df = load_interrogator(FILEPATH)
    time_col = find_time_column(df)
    wl_col = find_wl_column_for_channel(df, TARGET_CHANNEL)

    time_s = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    wl = pd.to_numeric(df[wl_col], errors="coerce").to_numpy(dtype=float)

    dlam, lam0 = compute_delta_and_lambda0(wl, N_BASELINE, BASELINE_MODE)

    # ε(t) = Δλ / (λ0 * (1 - p_e))
    epsilon = dlam / (lam0 * (1.0 - PE))
    epsilon_micro = epsilon * 1e6

    mask = np.isfinite(time_s) & np.isfinite(epsilon_micro)
    time_s = time_s[mask]
    epsilon_micro = epsilon_micro[mask]

    os.makedirs(OUT_DIR, exist_ok=True)

    plt.figure(figsize=(10, 4.8))
    plt.plot(time_s, epsilon_micro, linewidth=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Strain [microstrain]")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"exercise7_strain_WL{TARGET_CHANNEL}[nm].png")
    plt.savefig(out_path, dpi=180)
    plt.close()

    print(f"[Exercise 7] Saved: {out_path}")
    print(f"[Exercise 7] WL {TARGET_CHANNEL}[nm] lambda0 = {lam0:.6f} nm (baseline_mode={BASELINE_MODE}, N_BASELINE={N_BASELINE})")
    print(f"[Exercise 7] Strain microstrain stats: min={np.min(epsilon_micro):.2f}, max={np.max(epsilon_micro):.2f}")


if __name__ == "__main__":
    main()

