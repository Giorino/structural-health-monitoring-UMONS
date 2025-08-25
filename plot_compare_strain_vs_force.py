#!/usr/bin/env python3

from __future__ import annotations

import os
import re
from typing import List, Optional
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from compute_mechanical_strain import compute_mechanical_strain as compute_mech


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


def _parse_distance_from_name(name: str) -> Optional[float]:
    base = os.path.splitext(os.path.basename(str(name)))[0]
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


def _plot_by_distance_small_multiples(df: pd.DataFrame, latest_dir: str) -> str:
    # Gather distances per source file
    file_to_dist = {f: _parse_distance_from_name(f) for f in df["source_file"].dropna().unique()}
    dists = sorted({d for d in file_to_dist.values() if d is not None})
    if not dists:
        return ""

    # No longer computing global y limits - will use individual scaling per subplot

    # Layout
    n = len(dists)
    ncols = 3 if n >= 3 else n
    nrows = int(math.ceil(n / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols + 2, 4.2 * nrows + 1), squeeze=False)
    cmap = plt.get_cmap("tab10", n)

    for idx, dist in enumerate(dists):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        color = cmap(idx)
        files_here = [f for f, d in file_to_dist.items() if d == dist]
        sub = df[df["source_file"].isin(files_here)].copy()
        force = pd.to_numeric(sub.get("Force (N)"), errors="coerce")
        fbg = pd.to_numeric(sub.get("fbg_direct_strain [\u03bcu\u03b5]"), errors="coerce")
        mech = pd.to_numeric(sub.get("mechanical_strain [\u03bcu\u03b5]"), errors="coerce")
        mask = ~(force.isna() | fbg.isna())

        # Scatter FBG
        ax.scatter(force[mask], fbg[mask], s=14, alpha=0.75, color=color, label="FBG Δλ/λ0")

        # FBG linear fit
        if mask.sum() >= 2:
            x = force[mask].values
            y = fbg[mask].values
            slope, intercept = np.polyfit(x, y, deg=1)
            xline = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 50)
            yline = slope * xline + intercept
            # R^2
            yhat = slope * x + intercept
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            ax.plot(xline, yline, color=color, linewidth=1.8, label=f"FBG fit (R²={r2:.2f})")

        # Mechanical line fit from recomputed mechanical strain (reflects current E in compute_mechanical_strain)
        mask_mech = ~(force.isna() | mech.isna())
        if mask_mech.sum() >= 2:
            x_m = force[mask_mech].values
            y_m = mech[mask_mech].values
            slope_m, intercept_m = np.polyfit(x_m, y_m, deg=1)
            fmin, fmax = float(np.nanmin(x_m)), float(np.nanmax(x_m))
            xline = np.array([fmin, fmax])
            yline = slope_m * xline + intercept_m
            # Difference metrics using paired samples
            mask_both = ~(force.isna() | fbg.isna() | mech.isna())
            x_pts = force[mask_both].values.astype(float)
            y_fbg_pts = fbg[mask_both].values.astype(float)
            y_mech_pts = mech[mask_both].values.astype(float)
            diffs = y_mech_pts - y_fbg_pts
            mean_diff = float(np.nanmean(diffs)) if diffs.size > 0 else float('nan')
            denom = np.maximum(np.abs(y_fbg_pts), 1e-9)
            mape = float(np.nanmean(np.abs(diffs) / denom) * 100.0) if diffs.size > 0 else float('nan')
            sign = "+" if mean_diff >= 0 else ""
            diff_label_extra = f", Δmean={sign}{mean_diff:.0f} με, MAPE={mape:.1f}%"
            ax.plot(xline, yline, color=color, linestyle="--", linewidth=1.6,
                    label=f"Mechanical (P·L·y/4EI){diff_label_extra}")

        ax.set_title(f"{int(dist)} cm span")
        ax.set_xlabel("Force (N)")
        ax.set_ylabel("Strain [\u03bcu\u03b5]")
        ax.grid(True, linestyle=":", alpha=0.6)
        
        # Dynamic y-axis scaling for each subplot
        all_y_values = []
        if mask.any():
            all_y_values.extend(fbg[mask].dropna())
        if mask_mech.any():
            all_y_values.extend(mech[mask_mech].dropna())
        
        if all_y_values:
            y_min_local = float(min(all_y_values))
            y_max_local = float(max(all_y_values))
            # Add 10% padding to the range
            y_range = y_max_local - y_min_local
            if y_range > 0:
                pad_local = 0.1 * y_range
                ax.set_ylim(y_min_local - pad_local, y_max_local + pad_local)
            else:
                # If all values are the same, center around the value with some padding
                ax.set_ylim(y_min_local - 50, y_max_local + 50)
        
        ax.legend(fontsize=7, frameon=True, framealpha=0.85, loc="upper left")

    # Hide any empty axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis('off')

    fig.suptitle("Strain vs Force by Span Length", fontsize=14)
    out_path = os.path.join(latest_dir, "strain_vs_force_by_distance.png")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main(base_output_dir: Optional[str] = None) -> None:
    latest_dir = find_latest_output_directory(base_output_dir)
    if not latest_dir:
        raise RuntimeError("No output directories found under the provided base directory.")

    csv_files = list_merged_csvs(latest_dir)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in: {latest_dir}")

    df = load_and_merge_csvs(csv_files)

    # Always compute FBG and mechanical strain inline to reflect current constants and avoid stale CSVs
    from compute_fbg_direct_strain import compute_fbg_direct_microstrain_for_df
    fbg_series, _ = compute_fbg_direct_microstrain_for_df(df)
    df["fbg_direct_strain [\u03bcu\u03b5]"] = fbg_series

    df["mechanical_strain [\u03bcu\u03b5]"] = compute_mech(df)

    # Inputs
    force = pd.to_numeric(df.get("Force (N)"), errors="coerce")
    fbg = pd.to_numeric(df.get("fbg_direct_strain [\u03bcu\u03b5]"), errors="coerce")
    mech = pd.to_numeric(df.get("mechanical_strain [\u03bcu\u03b5]"), errors="coerce")

    # Build a consistent color map keyed by base file key so FBG vs Mechanical share colors
    unique_files = list(df["source_file"].dropna().unique())
    base_keys = [(_derive_base_key_from_merged_filename(str(f)) or os.path.splitext(str(f))[0]) for f in unique_files]
    # Preserve order and uniqueness
    seen = set()
    ordered_keys: List[str] = []
    for k in base_keys:
        if k not in seen:
            seen.add(k)
            ordered_keys.append(k)
    cmap = plt.get_cmap('tab20', max(2, len(ordered_keys)))
    color_by_key = {k: cmap(i) for i, k in enumerate(ordered_keys)}

    # Create the faceted plot by distance (only plot we now produce)
    try:
        out_facet = _plot_by_distance_small_multiples(df, latest_dir)
        if out_facet:
            print(f"Saved: {out_facet}")
    except Exception as e:
        print(f"Skipped faceted plot due to error: {e}")


if __name__ == "__main__":
    main()


