#!/usr/bin/env python3

from __future__ import annotations

import os
import re
from typing import List, Optional

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


def parse_distance_from_name(name: str) -> Optional[float]:
    base = os.path.splitext(os.path.basename(name))[0]
    if base.startswith("merged_"):
        base = base[len("merged_"):]
    base = re.sub(r"_\d{8}(_\d{4,6})?$", "", base)
    m = re.search(r"(?P<dist>\d+)cm-?(?P<layers>\d+)layers", base)
    if not m:
        return None
    try:
        return float(m.group("dist"))
    except Exception:
        return None


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


def compute_mechanical_strain(df: pd.DataFrame) -> pd.Series:
    # Constants from the user's specification
    b_m = 34.0e-3      # width in meters (34 mm)
    h_m = 4.0e-3       # thickness in meters (4 mm)
    y_m = 1.45e-3      # neutral axis distance in meters (1.45 mm)
    E_pa = 20.0e9      # Young's modulus in Pa (20 GPa)

    # Second moment of area for rectangular cross-section
    I_m4 = (b_m * (h_m ** 3)) / 12.0
    if I_m4 == 0.0:
        return pd.Series(np.nan, index=df.index)

    # Map each row to span length (L) based on its source file name
    distances_cm_by_file = {name: parse_distance_from_name(str(name)) for name in df["source_file"].dropna().unique()}

    def span_length_m(row) -> float:
        dist_cm = distances_cm_by_file.get(str(row["source_file"]))
        return (float(dist_cm) / 100.0) if dist_cm is not None else np.nan

    L_series = df.apply(span_length_m, axis=1)

    # Force P in Newtons
    if "Force (N)" not in df.columns:
        raise KeyError("Missing 'Force (N)' column in merged CSVs.")
    P_series = pd.to_numeric(df["Force (N)"], errors="coerce")

    # Strain epsilon = (P * L * y) / (4 * E * I)
    eps = (P_series * L_series * y_m) / (4.0 * E_pa * I_m4)
    microstrain = eps * 1e6
    microstrain.name = "mechanical_strain [\u03bcu\u03b5]"

    # Logging per source file
    for name, group in df.groupby("source_file", sort=False):
        base = os.path.basename(str(name))
        try:
            dist_cm = parse_distance_from_name(base)
        except Exception:
            dist_cm = None
        L_val = (float(dist_cm) / 100.0) if dist_cm is not None else np.nan
        f_min = float(pd.to_numeric(group.get("Force (N)"), errors="coerce").min())
        f_max = float(pd.to_numeric(group.get("Force (N)"), errors="coerce").max())
        micro_group = pd.to_numeric(microstrain.loc[group.index], errors="coerce").dropna()
        if micro_group.empty:
            s_min = s_max = s_mean = np.nan
            first_vals = ""
        else:
            s_min = float(micro_group.min())
            s_max = float(micro_group.max())
            s_mean = float(micro_group.mean())
            first_vals = ", ".join(f"{v:.1f}" for v in micro_group.iloc[:5])
        print(
            f"[MECH] {base}: points={len(group)}, L={L_val:.3f} m, b={b_m*1e3:.1f} mm, h={h_m*1e3:.1f} mm, "
            f"y={y_m*1e3:.2f} mm, E={E_pa/1e9:.1f} GPa, I={I_m4:.3e} m^4, Force≈[{f_min:.2f},{f_max:.2f}] N, "
            f"strain[με] min={s_min:.1f}, max={s_max:.1f}, mean={s_mean:.1f}, first5=[{first_vals}]"
        )
    return microstrain


def main(base_output_dir: Optional[str] = None) -> None:
    latest_dir = find_latest_output_directory(base_output_dir)
    if not latest_dir:
        raise RuntimeError("No output directories found under the provided base directory.")

    csv_files = list_merged_csvs(latest_dir)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in: {latest_dir}")

    df = load_and_merge_csvs(csv_files)
    mech_series = compute_mechanical_strain(df)
    out = pd.DataFrame({
        "source_file": df["source_file"],
        "Force (N)": df.get("Force (N)"),
        "mechanical_strain [\u03bcu\u03b5]": mech_series,
    })
    out_csv = os.path.join(latest_dir, "mechanical_strain.csv")
    out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()


