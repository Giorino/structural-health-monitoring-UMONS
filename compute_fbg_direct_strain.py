#!/usr/bin/env python3

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def find_latest_output_directory(base_dir: Optional[str] = None) -> Optional[str]:
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
    files = sorted([
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.endswith('.csv') and f.startswith('merged_')
    ])
    if not files:
        files = sorted([
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith('.csv')
        ])
    return files


def _derive_base_key_from_merged_filename(basename: str) -> Optional[str]:
    base = os.path.splitext(os.path.basename(basename))[0]
    if base.startswith("merged_"):
        base = base[len("merged_"):]
    base = re.sub(r"_\d{8}(_\d{4,6})?$", "", base)
    return base or None


def _load_raw_central_baseline(raw_dir: str, base_key: str) -> Optional[float]:
    txt_path = os.path.join(raw_dir, f"{base_key}-interrogator.txt")
    if not os.path.isfile(txt_path):
        return None

    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
    except Exception:
        return None

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
    series = pd.to_numeric(df_raw[central_col], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[0])


def compute_fbg_direct_microstrain_for_df(df: pd.DataFrame, pe: float = 0.22) -> Tuple[pd.Series, Dict[str, float]]:
    central_col = "WL_ch2" if "WL_ch2" in df.columns else ("WL 2[nm]" if "WL 2[nm]" in df.columns else None)
    if central_col is None:
        raise KeyError("Central Bragg channel not found. Expected 'WL_ch2' or 'WL 2[nm]'.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(script_dir, "interrogator-data")

    micro_list: List[pd.Series] = []
    baseline_map: Dict[str, float] = {}

    for fname, group in df.groupby("source_file", sort=False):
        base_key = _derive_base_key_from_merged_filename(str(fname)) or ""
        # Try loading baseline from raw; track source for logging
        lambda0_raw = _load_raw_central_baseline(raw_dir, base_key) if base_key else None
        lambda0 = lambda0_raw

        series = pd.to_numeric(group[central_col], errors="coerce")
        if lambda0 is None:
            first_valid = series.dropna().iloc[0] if not series.dropna().empty else np.nan
            lambda0 = float(first_valid) if np.isfinite(first_valid) else np.nan

        series_valid = series.dropna()
        looks_delta = False
        if not series_valid.empty:
            median_abs = float(abs(series_valid.median()))
            span = float(series_valid.max() - series_valid.min())
            looks_delta = median_abs < 10.0 and span < 50.0

        if looks_delta:
            delta_lambda = series
        else:
            delta_lambda = series - lambda0

        eps = (delta_lambda / lambda0) / (1.0 - pe)
        micro = eps * 1e6
        micro_series = pd.Series(micro.values, index=group.index)
        micro_list.append(micro_series)
        if np.isfinite(lambda0):
            baseline_map[str(fname)] = float(lambda0)

        # Logging per source file processed
        origin = "raw TXT" if lambda0_raw is not None else "merged CSV (fallback)"
        try:
            f_min = float(pd.to_numeric(group.get("Force (N)"), errors="coerce").min())
            f_max = float(pd.to_numeric(group.get("Force (N)"), errors="coerce").max())
        except Exception:
            f_min, f_max = np.nan, np.nan
        micro_valid = pd.to_numeric(micro_series, errors="coerce").dropna()
        if micro_valid.empty:
            s_min = s_max = s_mean = np.nan
            first_vals = ""
        else:
            s_min = float(micro_valid.min())
            s_max = float(micro_valid.max())
            s_mean = float(micro_valid.mean())
            first_vals = ", ".join(f"{v:.1f}" for v in micro_valid.iloc[:5])
        print(
            f"[FBG] {base_key or fname}: points={len(group)}, lambda0={lambda0:.6f} nm, "
            f"mode={'Δλ' if looks_delta else 'abs→Δλ'}, origin={origin}, "
            f"Force≈[{f_min:.2f},{f_max:.2f}] N, strain[με] min={s_min:.1f}, max={s_max:.1f}, mean={s_mean:.1f}, first5=[{first_vals}]"
        )

    out = pd.concat(micro_list).sort_index()
    out.name = "fbg_direct_strain [\u03bcu\u03b5]"
    return out, baseline_map


def load_and_merge_csvs(csv_paths: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        raise RuntimeError("No CSV files were loaded.")
    merged = pd.concat(frames, axis=0, ignore_index=True)
    return merged


def main(base_output_dir: Optional[str] = None) -> None:
    latest_dir = find_latest_output_directory(base_output_dir)
    if not latest_dir:
        raise RuntimeError("No output directories found under the provided base directory.")

    csv_files = list_merged_csvs(latest_dir)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in: {latest_dir}")

    df = load_and_merge_csvs(csv_files)
    strain_series, baselines = compute_fbg_direct_microstrain_for_df(df)
    df_out = pd.DataFrame({
        "source_file": df["source_file"],
        "Force (N)": df.get("Force (N)"),
        "fbg_direct_strain [\u03bcu\u03b5]": strain_series,
    })
    out_csv = os.path.join(latest_dir, "fbg_direct_strain.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()


