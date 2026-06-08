#!/usr/bin/env python3
"""
Dataset-wide median filter effectiveness summary.

Scans all interrogator files and computes robust per-file metrics:
- median absolute residual: median(|raw - filtered|)
- p99 and p99.7 of |raw - filtered|
- fraction of points above a "huge change" threshold

Outputs:
- exercise_outputs/median_filter_dataset_summary.csv
- exercise_outputs/median_filter_dataset_summary.png
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import medfilt


DATA_DIR = Path("interrogator-data")
OUT_DIR = Path("exercise_outputs")
KERNEL_SIZE = 7
TARGET_CHANNEL = 2
HUGE_DIFF_NM = 0.04


def load_interrogator(path: Path) -> pd.DataFrame:
    # Robust loader similar to project pipeline loaders:
    # find first data row, infer token count, parse by whitespace.
    iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}T")
    data_start = 0
    first_tokens = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= 80:
                break
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if iso_re.match(s) and len(parts) >= 3:
                data_start = i
                first_tokens = parts
                break
            try:
                float(parts[0])
                if len(parts) >= 2:
                    data_start = i
                    first_tokens = parts
                    break
            except Exception:
                pass

    if first_tokens is None:
        raise ValueError("Could not detect data start.")

    n_tokens = len(first_tokens)
    if iso_re.match(first_tokens[0]):
        names = ["Timestamp", "Time_s"]
        wl_count = max(0, n_tokens - 2)
    else:
        names = ["Time_s"]
        wl_count = max(0, n_tokens - 1)
    names += [f"WL {i}[nm]" for i in range(1, wl_count + 1)]

    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=names,
            skiprows=data_start,
            engine="python",
            on_bad_lines="skip",
        )
    except TypeError:
        # Older pandas fallback
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=names,
            skiprows=data_start,
            engine="python",
            error_bad_lines=False,
            warn_bad_lines=False,
        )

    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_wl_column(df: pd.DataFrame, channel: int) -> str:
    pat = re.compile(rf"^WL\s+{channel}\[nm\]$")
    for c in df.columns:
        if pat.match(c):
            return c
    raise KeyError(f"WL {channel}[nm] not found.")


def get_best_available_wl_column(df: pd.DataFrame, preferred_channel: int) -> tuple[str, int]:
    # Prefer WL2, then nearby common channels, then any WL channel found.
    wl_cols = []
    pat = re.compile(r"^WL\s+(\d+)\[nm\]$")
    for c in df.columns:
        m = pat.match(c)
        if m:
            wl_cols.append((int(m.group(1)), c))
    if not wl_cols:
        raise KeyError("No WL channel columns found.")

    wl_map = {ch: name for ch, name in wl_cols}
    preference = [preferred_channel, 1, 3, 4, 5, 6, 7, 8]
    for ch in preference:
        if ch in wl_map:
            return wl_map[ch], ch

    # Final fallback: smallest channel index available
    ch_min = min(wl_map.keys())
    return wl_map[ch_min], ch_min


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(DATA_DIR.glob("*-interrogator.txt"))
    if not txt_files:
        raise RuntimeError(f"No interrogator files found in {DATA_DIR.resolve()}")

    k = KERNEL_SIZE if KERNEL_SIZE % 2 == 1 else KERNEL_SIZE + 1
    rows = []

    for fp in txt_files:
        try:
            df = load_interrogator(fp)
            wl_col, used_channel = get_best_available_wl_column(df, TARGET_CHANNEL)
            raw = pd.to_numeric(df[wl_col], errors="coerce").to_numpy(dtype=float)
            filtered = medfilt(raw, kernel_size=k)
            abs_diff = np.abs(raw - filtered)
            finite = abs_diff[np.isfinite(abs_diff)]
            if finite.size == 0:
                continue

            row = {
                "file": fp.name,
                "used_channel": int(used_channel),
                "n_points": int(finite.size),
                "median_abs_diff_nm": float(np.median(finite)),
                "p99_abs_diff_nm": float(np.percentile(finite, 99)),
                "p997_abs_diff_nm": float(np.percentile(finite, 99.7)),
                "max_abs_diff_nm": float(np.max(finite)),
                "huge_diff_fraction_pct": float(np.mean(finite >= HUGE_DIFF_NM) * 100.0),
            }
            rows.append(row)
        except Exception as e:
            print(f"Skipping {fp.name}: {e}")

    if not rows:
        raise RuntimeError("No files were processed successfully.")

    summary = pd.DataFrame(rows).sort_values("p997_abs_diff_nm", ascending=False).reset_index(drop=True)
    csv_path = OUT_DIR / "median_filter_dataset_summary.csv"
    summary.to_csv(csv_path, index=False)

    # Plot: p99.7 absolute diff per file (sorted), plus threshold line
    plt.figure(figsize=(13, 5.5))
    x = np.arange(len(summary))
    y = summary["p997_abs_diff_nm"].to_numpy()
    plt.bar(x, y, alpha=0.85)
    plt.axhline(HUGE_DIFF_NM, color="red", linestyle="--", linewidth=1.2, label=f"Huge-diff ref = {HUGE_DIFF_NM:.3f} nm")
    plt.xlabel("Interrogator files (sorted by p99.7 |raw-filtered|)")
    plt.ylabel("p99.7 |raw - median_filtered| [nm]")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    fig_path = OUT_DIR / "median_filter_dataset_summary.png"
    plt.savefig(fig_path, dpi=180)
    plt.close()

    print(f"Processed files: {len(summary)}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")
    print("\nTop 5 files by p99.7 |raw-filtered|:")
    print(summary[["file", "p997_abs_diff_nm", "huge_diff_fraction_pct"]].head(5).to_string(index=False))
    print("\nOverall stats:")
    print(
        f"mean p99.7={summary['p997_abs_diff_nm'].mean():.6f} nm, "
        f"median p99.7={summary['p997_abs_diff_nm'].median():.6f} nm, "
        f"max p99.7={summary['p997_abs_diff_nm'].max():.6f} nm"
    )


if __name__ == "__main__":
    main()

