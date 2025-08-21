#!/usr/bin/env python3

from __future__ import annotations

import os
from typing import List

import pandas as pd


def _import_plotly():
    try:
        import plotly.graph_objects as go  # type: ignore
        import plotly.io as pio  # type: ignore
        return go, pio
    except Exception as e:
        raise RuntimeError(
            "Plotly is required for interactive 3D plotting. Please install with 'pip install plotly'."
        ) from e


def prepare_data(base_output_dir: str | None = None) -> tuple[pd.DataFrame, str]:
    """Load merged CSVs from the latest output directory and compute FBG direct strain.

    Returns (df, selected_dir) where df includes at least the following columns:
      - 'Displacement (mm)'
      - 'Force (N)'
      - 'fbg_direct_strain [με]'
      - 'source_file'
    """
    import plot_force_strain_displacement as pfsd

    latest_dir = pfsd.find_latest_output_directory(base_output_dir)
    if not latest_dir:
        raise RuntimeError("No output directories found under the provided base directory.")

    csv_files: List[str] = pfsd.list_merged_csvs(latest_dir)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in: {latest_dir}")

    df = pfsd.load_and_merge_csvs(csv_files)

    # Ensure required columns exist
    required_cols = ["Displacement (mm)", "Force (N)"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError("Missing required columns: " + ", ".join(missing))

    # Compute FBG direct strain for central channel
    df["fbg_direct_strain [με]"], _ = pfsd.compute_fbg_direct_microstrain(df)

    return df, latest_dir


def build_3d_figure(df: pd.DataFrame):
    go, pio = _import_plotly()
    import plot_force_strain_displacement as pfsd

    fig = go.Figure()

    # One trace per source file so they remain visually separated
    for name, group in df.groupby("source_file", sort=False):
        short = pfsd._derive_base_key_from_merged_filename(str(name)) or str(name)
        fig.add_trace(
            go.Scatter3d(
                x=group["Displacement (mm)"],
                y=group["Force (N)"],
                z=group["fbg_direct_strain [με]"],
                mode="lines+markers",
                name=short,
                hovertemplate=(
                    "<b>%{text}</b><br>Disp: %{x:.3f} mm" +
                    "<br>Force: %{y:.3f} N" +
                    "<br>Strain: %{z:.1f} με" +
                    "<extra></extra>"
                ),
                text=[short] * len(group),
                marker=dict(size=3),
                line=dict(width=2),
            )
        )

    fig.update_layout(
        template="plotly_white",
        scene=dict(
            xaxis_title="Displacement (mm)",
            yaxis_title="Force (N)",
            zaxis_title="Strain [με]",
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title="Force–Displacement–Strain (3D)",
        legend=dict(itemsizing="trace"),
    )

    # Enable better mouse interaction defaults
    fig.update_scenes(
        dragmode="orbit",
        camera=dict(eye=dict(x=1.7, y=1.7, z=1.3)),
    )

    return fig


def save_interactive_html(fig, output_dir: str) -> str:
    _, pio = _import_plotly()
    out_path = os.path.join(output_dir, "force_displacement_strain_3d.html")
    pio.write_html(fig, file=out_path, auto_open=False, include_plotlyjs="cdn", full_html=True)
    return out_path


def main(base_output_dir: str | None = None) -> None:
    df, folder = prepare_data(base_output_dir)
    fig = build_3d_figure(df)
    out = save_interactive_html(fig, folder)
    print(f"Saved interactive 3D plot to: {out}")


if __name__ == "__main__":
    main()



