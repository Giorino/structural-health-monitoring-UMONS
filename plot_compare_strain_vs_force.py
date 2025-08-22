#!/usr/bin/env python3

from __future__ import annotations

import os
import re
from typing import List, Optional
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    # Constants for mechanical line
    b_m = 34.0e-3
    h_m = 4.0e-3
    y_m = 1.45e-3
    E_pa = 20.0e9
    I_m4 = (b_m * (h_m ** 3)) / 12.0

    # Compute global y limits across all facets
    ymins, ymaxs = [], []
    for dist in dists:
        files_here = [f for f, d in file_to_dist.items() if d == dist]
        sub = df[df["source_file"].isin(files_here)]
        force = pd.to_numeric(sub.get("Force (N)"), errors="coerce")
        fbg = pd.to_numeric(sub.get("fbg_direct_strain [\u03bcu\u03b5]"), errors="coerce")
        valid = ~(force.isna() | fbg.isna())
        if valid.any():
            ymins.append(float(fbg[valid].min()))
            ymaxs.append(float(fbg[valid].max()))
        # mechanical line at force range
        if not force.dropna().empty:
            L = dist / 100.0
            slope_micro_per_N = (L * y_m) / (4.0 * E_pa * I_m4) * 1e6
            fmin, fmax = float(force.min()), float(force.max())
            ymins.append(slope_micro_per_N * fmin)
            ymaxs.append(slope_micro_per_N * fmax)
    if not ymins:
        y_min, y_max = 0.0, 1.0
    else:
        y_min, y_max = min(ymins), max(ymins + ymaxs)
    # make some padding
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    y_min -= pad
    y_max += pad

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

        # Mechanical theoretical line
        if not force.dropna().empty:
            L = dist / 100.0
            slope_micro_per_N = (L * y_m) / (4.0 * E_pa * I_m4) * 1e6
            fmin, fmax = float(force.min()), float(force.max())
            xline = np.array([fmin, fmax])
            yline = slope_micro_per_N * xline
            ax.plot(xline, yline, color=color, linestyle="--", linewidth=1.6, label="Mechanical (P·L·y/4EI)")

        ax.set_title(f"{int(dist)} cm span")
        ax.set_xlabel("Force (N)")
        ax.set_ylabel("Strain [\u03bcu\u03b5]")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_ylim(y_min, y_max)
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

    # Load computed strains (if the user ran compute scripts), otherwise compute inline
    fbg_csv = os.path.join(latest_dir, "fbg_direct_strain.csv")
    mech_csv = os.path.join(latest_dir, "mechanical_strain.csv")

    if os.path.isfile(fbg_csv):
        fbg_df = pd.read_csv(fbg_csv)
        df["fbg_direct_strain [\u03bcu\u03b5]"] = fbg_df["fbg_direct_strain [\u03bcu\u03b5]"]
    else:
        from compute_fbg_direct_strain import compute_fbg_direct_microstrain_for_df
        fbg_series, _ = compute_fbg_direct_microstrain_for_df(df)
        df["fbg_direct_strain [\u03bcu\u03b5]"] = fbg_series

    if os.path.isfile(mech_csv):
        mech_df = pd.read_csv(mech_csv)
        df["mechanical_strain [\u03bcu\u03b5]"] = mech_df["mechanical_strain [\u03bcu\u03b5]"]
    else:
        from compute_mechanical_strain import compute_mechanical_strain
        df["mechanical_strain [\u03bcu\u03b5]"] = compute_mechanical_strain(df)

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

    # Plot
    plt.figure(figsize=(14, 10))
    # Plot per file to avoid zigzagging across discontinuities
    for name, group in df.groupby("source_file", sort=False):
        base_key = _derive_base_key_from_merged_filename(str(name)) or os.path.splitext(str(name))[0]
        color = color_by_key.get(base_key, None)
        mask_fbg = ~(group["Force (N)"].isna() | group["fbg_direct_strain [\u03bcu\u03b5]"].isna())
        if mask_fbg.sum() > 1:
            plt.plot(
                group.loc[mask_fbg, "Force (N)"],
                group.loc[mask_fbg, "fbg_direct_strain [\u03bcu\u03b5]"],
                label=f"FBG Δλ/λ0 ({base_key})",
                alpha=0.9,
                linewidth=1.6,
                color=color,
                linestyle='-'
            )

        mask_mech = ~(group["Force (N)"].isna() | group["mechanical_strain [\u03bcu\u03b5]"].isna())
        if mask_mech.sum() > 1:
            plt.plot(
                group.loc[mask_mech, "Force (N)"],
                group.loc[mask_mech, "mechanical_strain [\u03bcu\u03b5]"],
                label=f"Mechanical (P·L·y/4EI) ({base_key})",
                linestyle='--',
                alpha=0.9,
                linewidth=1.6,
                color=color
            )

    plt.xlabel("Force (N)")
    plt.ylabel("Strain [\u03bcu\u03b5]")
    plt.title("Strain vs Force: FBG Δλ/λ0 vs Mechanical")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(
        fontsize=6,
        ncol=3,
        handlelength=2.0,
        labelspacing=0.25,
        borderpad=0.2,
        columnspacing=0.6,
        frameon=True,
        framealpha=0.8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    out_path = os.path.join(latest_dir, "strain_vs_force_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Also create a clearer faceted plot by distance
    try:
        out_facet = _plot_by_distance_small_multiples(df, latest_dir)
        if out_facet:
            print(f"Saved: {out_facet}")
    except Exception as e:
        print(f"Skipped faceted plot due to error: {e}")


if __name__ == "__main__":
    main()


