#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


def run_pipeline() -> None:
    """Run the full pipeline:
    1) Batch merge interrogator TXT + Excel sheet into timestamped CSV outputs
    2) Generate time-series plot in the same output folder
    3) Create multi force–displacement animation video from the same folder
    """
    project_root = Path(__file__).resolve().parent
    output_base_dir = str(project_root / "output")

    # 1) Batch merge
    print("[1/3] Running batch merge ...")
    import batch_merge as batch_mod

    # Ensure the merge subprocess uses the current Python interpreter
    try:
        batch_mod.VENV_PY = Path(sys.executable)
    except Exception:
        pass

    batch_mod.main()

    # Determine the latest (most recently modified) timestamped output directory
    print("Selecting latest output directory ...")
    import multi_force_displacement_video as mfd

    latest_dir = mfd.find_latest_output_directory(base_dir=output_base_dir)
    if not latest_dir:
        raise RuntimeError(
            f"No output directory found under: {output_base_dir}. "
            "The batch merge may have produced no results."
        )
    print(f"Selected output directory: {latest_dir}")

    # 2) Plot time series using the same selected directory
    print("[2/3] Generating time-series plots ...")
    import plot_time_series as pts

    csv_files = pts.find_csv_files(latest_dir)
    df = pts.load_and_concatenate_csvs(csv_files)
    saved_plot = pts.plot_time_series(df, latest_dir)
    print(f"Saved time-series figure: {saved_plot}")

    # 3) Build multi force–displacement video using the same directory
    print("[3/3] Creating multi force–displacement video ...")
    mfd.create_multi_force_displacement_video(latest_dir)
    print("Pipeline completed.")


if __name__ == "__main__":
    run_pipeline()

