#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


def run_pipeline() -> None:
    """Run the full pipeline:
    1) Batch merge interrogator TXT + Excel sheet into timestamped CSV outputs
    2) Generate time-series plot in the same output folder
    3) Calculate strain outputs for the latest folder
    4) Plot force–displacement and FBG direct strain vs displacement
    5) Generate interactive 3D force–displacement–strain visualization
    6) Run strain wavelength analysis
    """
    project_root = Path(__file__).resolve().parent
    output_base_dir = str(project_root / "output")

    # 1) Batch merge
    print("[1/6] Running batch merge ...")
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
    print("[2/6] Generating time-series plots ...")
    import plot_time_series as pts

    csv_files = pts.find_csv_files(latest_dir)
    df = pts.load_and_concatenate_csvs(csv_files)
    saved_plot = pts.plot_time_series(df, latest_dir)
    print(f"Saved time-series figure: {saved_plot}")

    # 3) Calculate strain
    print("[3/6] Calculating strain ...")
    import calculate_strain as cs
    cs.process_directory(latest_dir)

    # 4) Build multi force–displacement video using the same directory
    # print("[4/5] Creating multi force–displacement video ...")
    # import multi_force_displacement_video as mfd
    # mfd.create_multi_force_displacement_video(latest_dir)

    # 4) Plot force–displacement and FBG direct strain vs displacement
    print("[4/6] Plotting force–displacement and FBG direct strain ...")
    import plot_force_strain_displacement as pfsd
    pfsd.main(output_base_dir)

    # 5) Generate interactive 3D force–displacement–strain visualization
    print("[5/6] Generating interactive 3D plot ...")
    import plot_3d_force_displacement_strain as p3d
    try:
        p3d.main(output_base_dir)
    except Exception as e:
        print(f"Skipping 3D visualization due to error: {e}")

    # 6) Run strain wavelength analysis
    print("[6/6] Running strain wavelength analysis ...")
    import strain_wavelength_analysis as swa
    swa.main()

    print("Pipeline completed.")


if __name__ == "__main__":
    run_pipeline()

