import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype


TIMESTAMPED_DIR_REGEX = re.compile(r"^\d{8}_\d{6}$")


@dataclass
class SelectedFolder:
    base_output_dir: str
    folder_name: str
    absolute_path: str
    parsed_timestamp: datetime


def parse_timestamp_from_folder_name(folder_name: str) -> Optional[datetime]:
    """Parse a folder name like 'YYYYMMDD_HHMMSS' into a datetime.

    Returns None if the name does not match the expected format.
    """
    if not TIMESTAMPED_DIR_REGEX.match(folder_name):
        return None
    try:
        return datetime.strptime(folder_name, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def list_timestamped_subdirs(base_output_dir: str) -> List[Tuple[str, datetime]]:
    """Return list of (folder_name, parsed_datetime) for timestamped subdirectories."""
    if not os.path.isdir(base_output_dir):
        raise FileNotFoundError(f"Output directory not found: {base_output_dir}")

    timestamped: List[Tuple[str, datetime]] = []
    for entry in os.listdir(base_output_dir):
        abs_path = os.path.join(base_output_dir, entry)
        if not os.path.isdir(abs_path):
            continue
        parsed = parse_timestamp_from_folder_name(entry)
        if parsed is not None:
            timestamped.append((entry, parsed))
    return timestamped


def select_folder_closest_to_now(base_output_dir: str) -> SelectedFolder:
    """Pick the timestamped subdirectory whose timestamp is closest to the current time.

    If there are multiple with the same distance, picks the one with the later timestamp.
    """
    timestamped = list_timestamped_subdirs(base_output_dir)
    if not timestamped:
        raise RuntimeError(
            f"No timestamped subfolders found in {base_output_dir}. Expected names like 'YYYYMMDD_HHMMSS'."
        )

    now = datetime.now()

    # Sort by absolute time difference, then by timestamp descending (to break ties toward later)
    folder_name, parsed_ts = sorted(
        timestamped,
        key=lambda t: (abs((t[1] - now).total_seconds()), -t[1].timestamp()),
    )[0]

    abs_path = os.path.join(base_output_dir, folder_name)
    return SelectedFolder(
        base_output_dir=base_output_dir,
        folder_name=folder_name,
        absolute_path=abs_path,
        parsed_timestamp=parsed_ts,
    )


def find_csv_files(input_dir: str) -> List[str]:
    """List CSV files in the directory (non-recursive), sorted alphabetically."""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    csv_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(".csv")
    ]
    csv_files.sort()
    if not csv_files:
        raise RuntimeError(f"No CSV files found in: {input_dir}")
    return csv_files


def load_and_concatenate_csvs(csv_paths: List[str]) -> pd.DataFrame:
    """Load CSV files, parse timestamps, add source_file column, and merge sorted by time."""
    frames: List[pd.DataFrame] = []
    file_order: List[str] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            raise KeyError(f"Missing 'timestamp' column in file: {path}")
        # Parse as datetime; pandas will respect timezone in strings when present
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        base = os.path.basename(path)
        df["source_file"] = base
        file_order.append(base)
        frames.append(df)

    if not frames:
        raise RuntimeError("No CSV data frames were loaded.")

    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged = merged.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # Preserve file ordering as encountered
    merged["source_file"] = pd.Categorical(merged["source_file"], categories=file_order, ordered=True)
    return merged


def compute_compressed_axis(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    file_col: str = "source_file",
    gap_seconds: float = 60.0,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Build a compressed x-axis (seconds) by concatenating per-file spans and removing gaps.

    Returns a tuple of (compressed_seconds_series, spans_compressed_df with columns start_s, end_s).
    """
    if file_col not in df.columns:
        raise KeyError(f"Missing '{file_col}' column in input data; cannot compress timeline")

    spans = (
        df.groupby(file_col, observed=True)[timestamp_col]
        .agg(["min", "max"])  # type: ignore[index]
        .rename(columns={"min": "start", "max": "end"})
    )

    # Preserve ordering by file category if available, else by first appearance
    if isinstance(df[file_col].dtype, CategoricalDtype):
        order = list(df[file_col].cat.categories)
    else:
        order = list(df[file_col].drop_duplicates().tolist())

    spans = spans.reindex(order)

    start_offset_by_file: dict[str, float] = {}
    base = 0.0
    for fname, row in spans.iterrows():
        start = row["start"]
        end = row["end"]
        duration_s = float((end - start).total_seconds()) if pd.notna(start) and pd.notna(end) else 0.0
        start_offset_by_file[str(fname)] = base
        base += max(duration_s, 0.0) + gap_seconds

    # Map each row's timestamp to compressed seconds
    starts = df[[file_col, timestamp_col]].copy()
    # Compute per-row offset in seconds from the file's start
    file_start_map = spans["start"].to_dict()

    def row_to_compressed_seconds(row: pd.Series) -> float:
        fname = str(row[file_col])
        t = row[timestamp_col]
        file_start = file_start_map.get(fname)
        if pd.isna(t) or pd.isna(file_start):
            return float("nan")
        return start_offset_by_file[fname] + float((t - file_start).total_seconds())

    compressed_seconds = starts.apply(row_to_compressed_seconds, axis=1)

    spans_compressed = pd.DataFrame(
        {
            "file": list(spans.index.astype(str)),
            "start_s": [start_offset_by_file[str(f)] for f in spans.index],
            "end_s": [start_offset_by_file[str(f)] + float((spans.loc[f, "end"] - spans.loc[f, "start"]).total_seconds()) for f in spans.index],
        }
    )

    return compressed_seconds, spans_compressed


def _plot_by_group(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    file_col: str,
    color: str,
    linewidth: float = 1.5,
) -> None:
    """Plot y vs x broken by file groups to avoid connecting lines across files."""
    for _, group in df.groupby(file_col, observed=True, sort=False):
        ax.plot(group[x_col], group[y_col], color=color, linewidth=linewidth, zorder=2)


def ensure_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Coerce specified columns to numeric, leaving missing values as NaN."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_time_series(
    df: pd.DataFrame,
    output_dir: str,
    timestamp_col: str = "timestamp",
    force_col: str = "Force (N)",
    displacement_col: str = "Displacement (mm)",
    wl_cols: Tuple[str, str, str] = ("WL_ch1", "WL_ch2", "WL_ch3"),
) -> str:
    """Plot Force, Displacement, and wavelength channels over time and save to PNG.

    Returns the absolute path of the saved figure.
    """
    required_columns = [timestamp_col, force_col, displacement_col, *wl_cols]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns in input data: " + ", ".join(missing)
        )

    df = ensure_numeric(df, [force_col, displacement_col, *wl_cols])

    # Build compressed x-axis (in minutes)
    compressed_seconds, spans_compressed = compute_compressed_axis(df, timestamp_col=timestamp_col)
    df = df.copy()
    df["_x_compressed_min"] = compressed_seconds / 60.0

    # Create subplots (5 rows, 1 column), shared x-axis for compressed time
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True, constrained_layout=True)

    # Plot styles
    x_vals = df["_x_compressed_min"]
    file_col = "source_file"

    # Force
    _plot_by_group(axes[0], df, "_x_compressed_min", force_col, file_col, color="#1f77b4", linewidth=1.5)
    axes[0].set_ylabel(force_col)
    axes[0].grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)

    # Displacement
    _plot_by_group(axes[1], df, "_x_compressed_min", displacement_col, file_col, color="#ff7f0e", linewidth=1.5)
    axes[1].set_ylabel(displacement_col)
    axes[1].grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)

    # WL channels
    colors = ("#2ca02c", "#d62728", "#9467bd")
    for idx, wl_col in enumerate(wl_cols, start=2):
        _plot_by_group(axes[idx], df, "_x_compressed_min", wl_col, file_col, color=colors[idx - 2], linewidth=1.2)
        axes[idx].set_ylabel(wl_col)
        axes[idx].grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)

    # Add background shading per source CSV if source_file column exists
    if file_col in df.columns:
        # Use spans on compressed axis
        spans = spans_compressed.copy()

        # Choose alternating, subtle colors for shading
        shade_colors = [
            (0.1, 0.2, 0.8, 0.06),  # light blue
            (0.8, 0.2, 0.1, 0.06),  # light red
            (0.1, 0.6, 0.2, 0.06),  # light green
            (0.6, 0.1, 0.6, 0.06),  # light purple
            (0.6, 0.6, 0.1, 0.06),  # light olive
            (0.1, 0.6, 0.6, 0.06),  # light teal
        ]

        # Map file -> color and draw spans on all axes
        import matplotlib.patches as mpatches

        legend_handles: List[mpatches.Patch] = []
        for i, row in enumerate(spans.itertuples(index=False)):
            shade_color = shade_colors[i % len(shade_colors)]
            for ax in axes:
                ax.axvspan(row.start_s / 60.0, row.end_s / 60.0, color=shade_color, zorder=0)
            legend_handles.append(mpatches.Patch(color=shade_color, label=row.file))

        # Add filename labels on top of the shaded spans in the first (top) subplot
        top_ax = axes[0]
        x_transform = top_ax.get_xaxis_transform()  # data coords for x, axes coords for y
        def shorten_label(name: str) -> str:
            # Strip extension if present
            base = os.path.splitext(name)[0]
            # Remove leading 'merged_'
            if base.startswith("merged_"):
                base = base[len("merged_"):]
            # Keep up to the pattern like '12layers-1' (or generally '<num>layers-<num>')
            m = re.search(r"(\d+layers-\d+)", base)
            if m:
                base = base[: m.end()]
            else:
                # Fallback: strip trailing timestamp like '_YYYYMMDD_HHMMSS' or '_YYYYMMDD_HHMM'
                base = re.sub(r"_\d{8}(_\d{4,6})?$", "", base)
            return base

        for row in spans.itertuples(index=False):
            mid_x = (row.start_s + row.end_s) / 120.0  # seconds -> minutes then midpoint
            label = shorten_label(row.file)
            top_ax.text(
                mid_x,
                0.98,
                label,
                transform=x_transform,
                ha="center",
                va="top",
                fontsize=8,
                color="black",
                zorder=3,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
                clip_on=False,
            )

    # X-axis formatting for compressed time (minutes), plus explicit padding to avoid clipping edges
    # Determine total span and set consistent x-limits and margins
    total_end_min = float(spans_compressed["end_s"].max()) / 60.0 if not spans_compressed.empty else float(df["_x_compressed_min"].max())
    gap_minutes = 60.0 / 60.0  # default gap_seconds=60 -> 1 minute
    pad = max(0.5, gap_minutes * 0.5)
    for ax in axes:
        ax.set_xlim(-pad, total_end_min + pad)
        ax.margins(x=0)
    axes[-1].set_xlabel("Compressed time (min)")

    # Title
    fig.suptitle("Force, Displacement, and Wavelength Shifts (compressed timeline)", fontsize=14)

    # Save figure
    output_path = os.path.join(output_dir, "time_series_over_time.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Force (N), Displacement (mm), and WL_ch1/2/3 over time from the "
            "most relevant timestamped output folder."
        )
    )
    # No required parameters: defaults to using the repository's 'output' folder next to this script and auto-selecting a timestamped subfolder
    parser.add_argument(
        "--base-output-dir",
        type=str,
        default=None,
        help=(
            "Base output directory containing timestamped subfolders (default: '<repo>/output' next to this script)."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help=(
            "Optional explicit input directory. If provided, this is used directly and timestamped-folder selection is skipped."
        ),
    )
    args = parser.parse_args([])  # ensure it runs without any CLI args when executed directly

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_base_output = os.path.join(script_dir, "output")
    base_output_dir = args.base_output_dir or default_base_output

    if args.input_dir:
        selected_dir = args.input_dir
        if not os.path.isdir(selected_dir):
            raise FileNotFoundError(f"Provided --input-dir does not exist or is not a directory: {selected_dir}")
        print(f"Using provided input directory: {selected_dir}")
    else:
        selected = select_folder_closest_to_now(base_output_dir)
        selected_dir = selected.absolute_path
        print(
            "Selected input directory (closest to now): "
            f"{selected.folder_name} (timestamp={selected.parsed_timestamp})\n"
            f"Absolute path: {selected_dir}"
        )

    csv_files = find_csv_files(selected_dir)
    print(f"Found {len(csv_files)} CSV file(s) in: {selected_dir}")

    df = load_and_concatenate_csvs(csv_files)
    print(f"Loaded {len(df)} total rows across CSVs.")

    saved_path = plot_time_series(df, selected_dir)
    print(f"Saved figure: {saved_path}")


if __name__ == "__main__":
    main()

