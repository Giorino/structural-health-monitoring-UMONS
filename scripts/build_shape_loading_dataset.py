#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from pipeline_common import ensure_dir, latest_enriched_dir_or_fail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a simple shape-plus-loading dataset from the latest enriched output.")
    parser.add_argument("--enriched-dir", type=str, default=None, help="Specific enriched dataset directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional explicit output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enriched_dir = Path(args.enriched_dir) if args.enriched_dir else latest_enriched_dir_or_fail()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else enriched_dir / f"shape_loading_dataset_{timestamp}"
    ensure_dir(output_dir)

    repetition_df = pd.read_csv(enriched_dir / "enriched_repetition_table.csv")
    raw_npz = np.load(enriched_dir / "datasets" / "dataset_E_raw_window_tensor_only.npz", allow_pickle=True)

    manifest = repetition_df.set_index("raw_segment_id").loc[raw_npz["raw_segment_id"].tolist()].reset_index()
    group_column = "source_run_id" if "source_run_id" in manifest.columns else "run_id"

    loading_columns = [
        "force_N",
        "displacement_mm",
        "air_pressure_bar",
        "loading_group_index",
        "repetition_index",
        "global_repetition_index_within_run",
        "force_increment_from_previous_group_N",
        "displacement_increment_from_previous_group_mm",
        "loading_direction",
        "time_since_run_start",
        "time_since_group_start",
    ]
    available_loading_columns = [col for col in loading_columns if col in manifest.columns]

    export_manifest = manifest[
        [
            "raw_segment_id",
            "source_file",
            group_column,
            "run_id",
            "label_damage_transition",
            "label_crack_level",
            "sample_start_index",
            "sample_center_index",
            "sample_end_index",
            "timestamp_start",
            "timestamp_center",
            "timestamp_end",
        ] + available_loading_columns
    ].copy()
    export_manifest.rename(columns={group_column: "group_id"}, inplace=True)
    export_manifest.to_csv(output_dir / "shape_loading_manifest.csv", index=False)

    x_loading_numeric_columns = [col for col in available_loading_columns if pd.api.types.is_numeric_dtype(manifest[col])]
    x_loading = manifest[x_loading_numeric_columns].to_numpy(dtype=float) if x_loading_numeric_columns else np.empty((len(manifest), 0), dtype=float)

    np.savez_compressed(
        output_dir / "shape_loading_dataset.npz",
        X_shape_resampled=np.asarray(raw_npz["X_raw"], dtype=float),
        y=np.asarray(raw_npz["y"], dtype=int),
        groups=np.asarray(export_manifest["group_id"], dtype=object),
        raw_segment_id=np.asarray(export_manifest["raw_segment_id"], dtype=object),
        X_loading=x_loading,
        loading_feature_names=np.asarray(x_loading_numeric_columns, dtype=object),
    )

    notes = {
        "created_at": datetime.now().isoformat(),
        "source_enriched_dir": str(enriched_dir),
        "dataset_intent": "One sample per detected peak, using full resampled peak shape plus a minimal loading sidecar.",
        "group_column": "group_id",
        "target_column": "label_damage_transition",
        "shape_tensor": {
            "name": "X_shape_resampled",
            "shape": list(np.asarray(raw_npz["X_raw"]).shape),
            "description": "Resampled peak waveform tensor with shape [n_samples, n_channels, n_points].",
        },
        "loading_features": x_loading_numeric_columns,
        "manifest_path": str(output_dir / "shape_loading_manifest.csv"),
        "npz_path": str(output_dir / "shape_loading_dataset.npz"),
        "recommendation": {
            "model_input": "Use X_shape_resampled as the primary input. Treat X_loading as a small auxiliary branch only.",
            "scientific_scope": "This representation is appropriate for shape-aware damage-state detection, not crack localization.",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")

    summary_lines: List[str] = [
        "# Shape Plus Loading Dataset",
        "",
        f"Source enriched directory: `{enriched_dir}`",
        f"Output directory: `{output_dir}`",
        "",
        "This dataset keeps the representation intentionally simple.",
        "",
        "Each sample corresponds to one detected peak.",
        "The primary signal is the full resampled peak shape across the available FBG channels.",
        "The side information is a minimal loading vector rather than a large handcrafted feature table.",
        "",
        f"Samples: {len(export_manifest)}",
        f"Groups: {export_manifest['group_id'].nunique()}",
        f"Positive samples: {int(export_manifest['label_damage_transition'].sum())}",
        "",
        "Loading features:",
        "",
        *(f"- {name}" for name in x_loading_numeric_columns),
    ]
    (output_dir / "README.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Shape-loading dataset written to: {output_dir}")


if __name__ == "__main__":
    main()
