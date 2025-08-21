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


def load_and_merge_csvs(csv_paths: List[str]) -> pd.DataFrame:
    """Load CSVs and concatenate with a 'source_file' column.

    Does not require a timestamp column.
    """
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)
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
# Force vs Displacement plotting
# -----------------------------------------

def plot_force_vs_displacement(df: pd.DataFrame, output_dir: str) -> str:
    required_cols = ["Force (N)", "Displacement (mm)"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError("Missing required columns: " + ", ".join(missing))

    df = ensure_numeric(df, required_cols)

    plt.figure(figsize=(10, 7))
    # Plot per file to avoid connecting unrelated segments
    for name, group in df.groupby("source_file", sort=False):
        short = _derive_base_key_from_merged_filename(str(name)) or os.path.splitext(str(name))[0]
        plt.plot(group["Displacement (mm)"], group["Force (N)"], label=short, linewidth=1.5)

    plt.xlabel("Displacement (mm)")
    plt.ylabel("Force (N)")
    plt.title("Force vs. Displacement")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(fontsize=8)

    out_path = os.path.join(output_dir, "force_vs_displacement.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# -----------------------------------------
# Wavelength → Force relation and Strain
# -----------------------------------------

@dataclass
class WLForceModel:
    wl_column: str
    slope_n_per_nm: float
    intercept_n: float
    r2: float


def pick_best_wl_column(df: pd.DataFrame, wl_candidates: List[str]) -> Optional[str]:
    best_col: Optional[str] = None
    best_score = -np.inf
    for c in wl_candidates:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        y = pd.to_numeric(df["Force (N)"], errors="coerce")
        valid = ~(x.isna() | y.isna())
        if valid.sum() < 2:
            continue
        # Use absolute Pearson correlation as score
        corr = np.corrcoef(x[valid], y[valid])[0, 1]
        score = abs(float(corr)) if np.isfinite(corr) else -np.inf
        if score > best_score:
            best_score = score
            best_col = c
    return best_col


def fit_wl_to_force_linear(df: pd.DataFrame, wl_col: str) -> WLForceModel:
    x = pd.to_numeric(df[wl_col], errors="coerce")
    y = pd.to_numeric(df["Force (N)"], errors="coerce")
    valid = ~(x.isna() | y.isna())
    if valid.sum() < 2:
        raise RuntimeError(f"Insufficient data to fit linear model for {wl_col}")
    slope, intercept = np.polyfit(x[valid], y[valid], deg=1)
    y_hat = slope * x[valid] + intercept
    ss_res = float(np.sum((y[valid] - y_hat) ** 2))
    ss_tot = float(np.sum((y[valid] - np.mean(y[valid])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return WLForceModel(wl_col, float(slope), float(intercept), float(r2))


def compute_predicted_force_from_wavelength(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, WLForceModel]]:
    """Fit per-file WL→Force models using ONLY the central Bragg channel (2nd channel).

    Preferred column names (checked in order): 'WL_ch2', 'WL 2[nm]'.
    """
    central_candidates = [c for c in ("WL_ch2", "WL 2[nm]") if c in df.columns]
    if not central_candidates:
        raise KeyError("Central Bragg channel not found. Expected 'WL_ch2' or 'WL 2[nm]' in merged CSVs.")
    central_wl_col = central_candidates[0]

    preds_list: List[pd.Series] = []
    model_by_file: Dict[str, WLForceModel] = {}

    for name, group in df.groupby("source_file", sort=False):
        if central_wl_col not in group.columns:
            preds_list.append(pd.Series([np.nan] * len(group), index=group.index))
            continue
        model = fit_wl_to_force_linear(group, central_wl_col)
        model_by_file[str(name)] = model

        x = pd.to_numeric(group[model.wl_column], errors="coerce")
        y_pred = model.slope_n_per_nm * x + model.intercept_n
        preds_list.append(pd.Series(y_pred.values, index=group.index))

    predicted_force = pd.concat(preds_list).sort_index()
    predicted_force.name = "Force_pred_from_wl (N)"
    return predicted_force, model_by_file


def compute_mechanical_strain_from_force(
    force_n: pd.Series,
    distance_cm: float,
    num_layers: int,
    beam_width_m: float = 0.025,
    layer_thickness_m: float = 0.0002,
    youngs_modulus_pa: float = 3.5e9,
) -> pd.Series:
    """Compute mechanical strain from force for a simply-supported beam with center load.

    epsilon = sigma / E,  sigma = (F * L * h) / (8 * I),  I = b * h^3 / 12
    """
    L = float(distance_cm) / 100.0
    h = float(num_layers) * float(layer_thickness_m)
    b = float(beam_width_m)
    I = (b * (h ** 3)) / 12.0

    # Avoid division by zero
    if I == 0.0:
        return pd.Series(np.nan, index=force_n.index)

    stress_pa = (force_n.astype(float) * L * h) / (8.0 * I)
    strain = stress_pa / float(youngs_modulus_pa)
    return strain


def add_distance_layers_from_source_file(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'Distance (cm)' and 'Layers' columns derived from each row's source file name if missing."""
    if "Distance (cm)" in df.columns and "Layers" in df.columns:
        return df

    distance_by_file: Dict[str, Optional[float]] = {}
    layers_by_file: Dict[str, Optional[int]] = {}
    for fname in df["source_file"].dropna().unique():
        dist, layers = parse_distance_layers_from_name(str(fname))
        distance_by_file[str(fname)] = dist
        layers_by_file[str(fname)] = layers

    def map_distance(row) -> Optional[float]:
        return distance_by_file.get(str(row["source_file"]))

    def map_layers(row) -> Optional[int]:
        return layers_by_file.get(str(row["source_file"]))

    df = df.copy()
    df["Distance (cm)"] = df.get("Distance (cm)", pd.Series(index=df.index, dtype=float))
    df["Layers"] = df.get("Layers", pd.Series(index=df.index, dtype=float))

    # Only fill where missing
    mask_missing_dist = df["Distance (cm)"].isna()
    mask_missing_layers = df["Layers"].isna()
    df.loc[mask_missing_dist, "Distance (cm)"] = df[mask_missing_dist].apply(map_distance, axis=1)
    df.loc[mask_missing_layers, "Layers"] = df[mask_missing_layers].apply(map_layers, axis=1)

    # Coerce types
    df["Distance (cm)"] = pd.to_numeric(df["Distance (cm)"], errors="coerce")
    df["Layers"] = pd.to_numeric(df["Layers"], errors="coerce").astype("Int64")
    return df


def plot_strain_vs_displacement(
    df: pd.DataFrame,
    strain_col: str,
    displacement_col: str,
    output_dir: str,
    title: str,
    filename: str,
    annotate: Optional[str] = None,
) -> str:
    required = [strain_col, displacement_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError("Missing required columns for plotting: " + ", ".join(missing))

    plt.figure(figsize=(10, 7))
    # Plot per source file
    for name, group in df.groupby("source_file", sort=False):
        short = _derive_base_key_from_merged_filename(str(name)) or os.path.splitext(str(name))[0]
        plt.plot(group[displacement_col], group[strain_col], label=short, linewidth=1.5)

    plt.xlabel("Displacement (mm)")
    plt.ylabel("Strain [με]")
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(fontsize=8)

    if annotate:
        plt.gcf().text(0.01, 0.01, annotate, fontsize=8, va="bottom", ha="left", alpha=0.8)

    out_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# -----------------------------------------
# FBG direct relation (Δλ/λ0) → strain
# -----------------------------------------

def _derive_base_key_from_merged_filename(basename: str) -> Optional[str]:
    """From 'merged_27cm-12layers-3_YYYYMMDD_HHMM.csv' → '27cm-12layers-3'."""
    base = os.path.splitext(os.path.basename(basename))[0]
    if base.startswith("merged_"):
        base = base[len("merged_"):]
    # strip trailing _YYYYMMDD_HHMM or _YYYYMMDD_HHMMSS
    base = re.sub(r"_\d{8}(_\d{4,6})?$", "", base)
    return base or None


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


def compute_fbg_direct_microstrain(
    df: pd.DataFrame,
    pe: float = 0.22,
) -> Tuple[pd.Series, Dict[str, float]]:
    """Compute microstrain using ε = (Δλ/λ0)/(1 - p_e) from the central channel per file.

    - Uses 'WL_ch2' if present, otherwise 'WL 2[nm]'.
    - If merged values look like Δλ (near zero, small range), uses them directly as Δλ.
    - If they look absolute (~1550 nm), computes Δλ = WL - λ0.
    - λ0 is read from the matching raw interrogator TXT (first valid central value). If unavailable, falls back to
      the group's first valid central value.
    """
    central_col = "WL_ch2" if "WL_ch2" in df.columns else ("WL 2[nm]" if "WL 2[nm]" in df.columns else None)
    if central_col is None:
        raise KeyError("Central Bragg channel not found. Expected 'WL_ch2' or 'WL 2[nm]'.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(script_dir, "interrogator-data")

    micro_list: List[pd.Series] = []
    baseline_map: Dict[str, float] = {}

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
        micro = eps * 1e6
        micro_list.append(pd.Series(micro.values, index=group.index))
        if np.isfinite(lambda0):
            baseline_map[str(fname)] = float(lambda0)

    out = pd.concat(micro_list).sort_index()
    out.name = "fbg_direct_strain [με]"
    return out, baseline_map

# -----------------------------------------
# Main entry
# -----------------------------------------

def main(base_output_dir: Optional[str] = None) -> None:
    latest_dir = find_latest_output_directory(base_output_dir)
    if not latest_dir:
        raise RuntimeError("No output directories found under the provided base directory.")
    print(f"Using latest output directory: {latest_dir}")

    csv_files = list_merged_csvs(latest_dir)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in: {latest_dir}")
    print(f"Found {len(csv_files)} CSV file(s)")

    df = load_and_merge_csvs(csv_files)

    # Plot Force vs Displacement
    fvd_path = plot_force_vs_displacement(df, latest_dir)
    print(f"Saved plot: {fvd_path}")

    # Ensure we have displacement for plotting the FBG direct strain
    if "Displacement (mm)" not in df.columns:
        raise KeyError("'Displacement (mm)' column is required for plotting.")

    # Ensure per-row distance/layers are available (useful for completeness in CSV, even if not used in FBG formula)
    df = add_distance_layers_from_source_file(df)

    # Also compute and plot direct FBG strain from central channel using Δλ/λ0
    try:
        df["fbg_direct_strain [με]"], baseline_map = compute_fbg_direct_microstrain(df)
        svd_fbg_path = plot_strain_vs_displacement(
            df,
            strain_col="fbg_direct_strain [με]",
            displacement_col="Displacement (mm)",
            output_dir=latest_dir,
            title="Strain vs. Displacement (FBG direct: Δλ/λ0)",
            filename="strain_vs_displacement.png",
            annotate=None,
        )
        print(f"Saved plot: {svd_fbg_path}")
    except Exception as e:
        print(f"Skipped FBG direct strain plot due to error: {e}")


if __name__ == "__main__":
    main()


