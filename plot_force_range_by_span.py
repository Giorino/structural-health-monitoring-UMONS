import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Ensure non-interactive backend for headless environments
import matplotlib.pyplot as plt


def find_latest_output_folder(base_path: str = "output") -> Path | None:
    """Return the most recently created directory under base_path, or None if empty."""
    base = Path(base_path)
    if not base.exists():
        return None
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    # Use creation time to match prior scripts' behavior
    latest = max(subdirs, key=lambda p: p.stat().st_ctime)
    return latest


def collect_force_by_span(folder: Path) -> pd.DataFrame:
    """Scan merged_*.csv files in folder and compute min/max force per Distance (cm)."""
    csv_paths = sorted(folder.glob("merged_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No merged_*.csv files found in {folder}")

    records: List[Tuple[float, float]] = []  # (distance_cm, force_n)
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path, usecols=["Distance (cm)", "Force (N)"])
        except ValueError:
            # Columns may not be selectable by usecols in some pandas versions; read full then filter
            df = pd.read_csv(csv_path)
            missing = {c for c in ["Distance (cm)", "Force (N)"] if c not in df.columns}
            if missing:
                raise KeyError(f"Missing columns {missing} in {csv_path}")

        # Robust numeric conversion
        df["Distance (cm)"] = pd.to_numeric(df["Distance (cm)"], errors="coerce")
        df["Force (N)"] = pd.to_numeric(df["Force (N)"], errors="coerce")
        df = df.dropna(subset=["Distance (cm)", "Force (N)"])

        if df.empty:
            continue

        # Distance is constant per file; still collect all rows to cover any anomalies robustly
        records.extend(df[["Distance (cm)", "Force (N)"]].itertuples(index=False, name=None))

    if not records:
        raise RuntimeError(f"No usable force data found in {folder}")

    data = pd.DataFrame(records, columns=["Distance (cm)", "Force (N)"])
    summary = (
        data.groupby("Distance (cm)")["Force (N)"]
        .agg(force_min_N="min", force_max_N="max", force_mean_N="mean")
        .reset_index()
        .sort_values("Distance (cm)")
    )
    return summary


def plot_horizontal_force_ranges(summary: pd.DataFrame, out_png: Path) -> None:
    """Create a horizontal range plot (min→max force per span) and save it."""
    distances = summary["Distance (cm)"].astype(float).values
    force_min = summary["force_min_N"].astype(float).values
    force_max = summary["force_max_N"].astype(float).values

    n = len(distances)
    y_positions = list(range(n))

    fig_height = max(2.5, 0.6 * n + 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Draw horizontal line segments for ranges
    for idx, y in enumerate(y_positions):
        ax.hlines(y=y, xmin=force_min[idx], xmax=force_max[idx], colors="#1f77b4", linewidth=4, alpha=0.9)
        # End caps
        ax.plot([force_min[idx], force_max[idx]], [y, y], "o", color="#1f77b4", markersize=6)

    # Labels and formatting
    ax.set_xlabel("Force (N)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{int(d) if float(d).is_integer() else d} cm" for d in distances])
    ax.set_title("Applied Force Range by Span Length")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_ylim(-0.5, n - 0.5)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main(folder_arg: str | None = None) -> Path:
    folder = Path(folder_arg) if folder_arg else find_latest_output_folder("output")
    if folder is None or not folder.exists():
        raise FileNotFoundError("Could not resolve an output folder. Provide --folder or generate merged outputs first.")

    summary = collect_force_by_span(folder)

    # Persist summary alongside plot for traceability
    summary_csv = folder / "force_range_by_span.csv"
    summary.to_csv(summary_csv, index=False)

    out_png = folder / "force_range_by_span.png"
    plot_horizontal_force_ranges(summary, out_png)
    return out_png


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot horizontal force ranges per span length from merged CSVs.")
    parser.add_argument("--folder", type=str, default=None, help="Path to an output subfolder containing merged_*.csv files. Defaults to latest under 'output/'.")
    args = parser.parse_args()

    out_path = main(args.folder)
    print(f"Saved plot to: {out_path}")



