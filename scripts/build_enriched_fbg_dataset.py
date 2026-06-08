#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pipeline_common import (
    ROOT,
    available_wl_columns,
    compute_channel_summary,
    detect_segments_for_run,
    dump_json,
    ensure_dir,
    infer_specimen_id,
    latest_enriched_dir_or_fail,
    list_raw_files,
    load_constants,
    load_excel_workbook,
    map_sheet_keys,
    normalize_run_key,
    parse_geometry_from_run,
    parse_interrogator_file,
    resample_segment,
    write_markdown_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build enriched per-run FBG datasets for grouped evaluation.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional explicit output directory.")
    parser.add_argument("--reuse-current-merged-boundaries", action="store_true", help="Reuse segment boundaries from the latest merged CSVs when available.")
    return parser.parse_args()


def latest_current_merged_dir() -> Optional[Path]:
    output_root = ROOT / "output"
    candidates = sorted(
        [
            path for path in output_root.iterdir()
            if path.is_dir() and not path.name.startswith("enriched_dataset_")
        ]
    ) if output_root.exists() else []
    return candidates[-1] if candidates else None


def load_current_merged_index() -> Dict[str, pd.DataFrame]:
    merged_dir = latest_current_merged_dir()
    if merged_dir is None:
        return {}
    index: Dict[str, List[pd.DataFrame]] = {}
    for path in sorted(merged_dir.glob("merged_*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        key = normalize_run_key(path.stem)
        index.setdefault(key, []).append(df)
    collapsed: Dict[str, pd.DataFrame] = {}
    for key, frames in index.items():
        merged = pd.concat(frames, ignore_index=True)
        collapsed[key] = merged
    return collapsed


def infer_sheet_name(run_key: str, sheet_lookup: Dict[str, str]) -> Optional[str]:
    if run_key in sheet_lookup:
        return sheet_lookup[run_key]
    variants = {
        run_key,
        run_key.replace("-layers", "layers"),
        run_key.replace("layers", "-layers"),
        run_key.replace("-s", ""),
    }
    for variant in variants:
        if variant in sheet_lookup:
            return sheet_lookup[variant]
    return None


def safe_float(value: object) -> float:
    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return np.nan


def segment_time_value(segment_df: pd.DataFrame, position: str) -> object:
    if "Timestamp" in segment_df.columns and segment_df["Timestamp"].notna().any():
        if position == "start":
            return segment_df["Timestamp"].iloc[0]
        if position == "end":
            return segment_df["Timestamp"].iloc[-1]
        return segment_df["Timestamp"].iloc[len(segment_df) // 2]
    if "Time_s" in segment_df.columns and segment_df["Time_s"].notna().any():
        if position == "start":
            return safe_float(segment_df["Time_s"].iloc[0])
        if position == "end":
            return safe_float(segment_df["Time_s"].iloc[-1])
        return safe_float(segment_df["Time_s"].iloc[len(segment_df) // 2])
    return np.nan


def compute_loading_direction(force_inc: float, disp_inc: float) -> str:
    if np.isnan(force_inc) and np.isnan(disp_inc):
        return "unknown"
    if force_inc > 0 or disp_inc > 0:
        return "increasing"
    if force_inc < 0 or disp_inc < 0:
        return "decreasing"
    return "hold"


def build_run_rows(
    raw_path: Path,
    raw_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    sheet_name: str,
    run_key: str,
    source_run_id: str,
    constants: Dict[str, object],
    current_merged_df: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, np.ndarray]]:
    dataset_constants = constants["dataset"]  # type: ignore[index]
    preferred_channels = int(dataset_constants["preferred_channels"])  # type: ignore[index]
    min_segment_length = int(dataset_constants["min_segment_length"])  # type: ignore[index]
    target_length = int(dataset_constants["resampled_window_length"])  # type: ignore[index]
    reps_per_group = int(dataset_constants["repetitions_per_group"])  # type: ignore[index]

    wl_cols = available_wl_columns(raw_df, limit=preferred_channels)
    segments, segment_meta = detect_segments_for_run(
        run_key=run_key,
        raw_df=raw_df,
        wl_cols=wl_cols,
        current_merged_row=current_merged_df,
        min_segment_length=min_segment_length,
    )

    if not segments:
        raise RuntimeError(f"No segments detected for {raw_path.name}")

    rows: List[Dict[str, object]] = []
    raw_segment_ids: List[str] = []
    raw_original_segments: List[np.ndarray] = []
    raw_resampled_segments: List[np.ndarray] = []
    raw_time_segments: List[np.ndarray] = []

    run_geometry = parse_geometry_from_run(run_key, safe_float(meta_df.iloc[0].get("Layers (#)")), constants)
    span_mm = safe_float(meta_df.iloc[0].get("Distance (cm)")) * 10.0
    fbg_configuration = "+".join(wl_cols) if wl_cols else "none"
    specimen_id = infer_specimen_id(run_key)
    run_start_time = segment_time_value(raw_df.iloc[segments[0][0]:segments[0][1] + 1], "start")
    group_first_time: Dict[int, float] = {}

    run_baseline_by_channel: Dict[str, float] = {}
    for channel in wl_cols:
        first_group_segments = []
        for idx, (start, end) in enumerate(segments[:reps_per_group]):
            segment_vals = raw_df.iloc[start:end + 1][channel].dropna().to_numpy(dtype=float)
            if segment_vals.size:
                first_group_segments.append(float(np.median(segment_vals)))
        run_baseline_by_channel[channel] = float(np.median(first_group_segments)) if first_group_segments else np.nan

    group_baseline_by_channel: Dict[Tuple[int, str], float] = {}
    for group_idx in range(int(np.ceil(len(segments) / reps_per_group))):
        seg_slice = segments[group_idx * reps_per_group:(group_idx + 1) * reps_per_group]
        for channel in wl_cols:
            medians = []
            for start, end in seg_slice:
                vals = raw_df.iloc[start:end + 1][channel].dropna().to_numpy(dtype=float)
                if vals.size:
                    medians.append(float(np.median(vals)))
            group_baseline_by_channel[(group_idx, channel)] = float(np.median(medians)) if medians else np.nan

    first_group_force = safe_float(meta_df.iloc[0].get("Force (N)"))
    first_group_disp = safe_float(meta_df.iloc[0].get("Displacement (mm)"))

    for global_idx, (start_idx, end_idx) in enumerate(segments):
        group_idx = global_idx // reps_per_group
        repetition_idx = global_idx % reps_per_group
        segment_df = raw_df.iloc[start_idx:end_idx + 1].copy()
        meta_row = meta_df.iloc[group_idx] if group_idx < len(meta_df) else pd.Series(dtype=object)
        force_n = safe_float(meta_row.get("Force (N)"))
        displacement_mm = safe_float(meta_row.get("Displacement (mm)"))
        air_pressure_bar = safe_float(meta_row.get("Air Pressure (bar)", meta_row.get("Unnamed: 0")))
        crack_level = int(safe_float(meta_row.get("Crack"))) if not np.isnan(safe_float(meta_row.get("Crack"))) else 0
        prev_force = safe_float(meta_df.iloc[group_idx - 1].get("Force (N)")) if group_idx > 0 and group_idx - 1 < len(meta_df) else np.nan
        prev_disp = safe_float(meta_df.iloc[group_idx - 1].get("Displacement (mm)")) if group_idx > 0 and group_idx - 1 < len(meta_df) else np.nan
        force_inc = force_n - prev_force if not np.isnan(force_n) and not np.isnan(prev_force) else np.nan
        disp_inc = displacement_mm - prev_disp if not np.isnan(displacement_mm) and not np.isnan(prev_disp) else np.nan
        force_inc_from_baseline = force_n - first_group_force if not np.isnan(force_n) and not np.isnan(first_group_force) else np.nan
        disp_inc_from_baseline = displacement_mm - first_group_disp if not np.isnan(displacement_mm) and not np.isnan(first_group_disp) else np.nan
        time_center = segment_time_value(segment_df, "center")
        time_start = segment_time_value(segment_df, "start")
        time_end = segment_time_value(segment_df, "end")
        time_values = segment_df["Time_s"].to_numpy(dtype=float) if "Time_s" in segment_df.columns else np.arange(len(segment_df), dtype=float)
        center_time_numeric = safe_float(time_center)
        if group_idx not in group_first_time:
            group_first_time[group_idx] = center_time_numeric if not np.isnan(center_time_numeric) else 0.0
        run_start_numeric = safe_float(run_start_time)
        time_since_run_start = center_time_numeric - run_start_numeric if not np.isnan(center_time_numeric) and not np.isnan(run_start_numeric) else np.nan
        time_since_group_start = center_time_numeric - group_first_time[group_idx] if not np.isnan(center_time_numeric) else np.nan

        row: Dict[str, object] = {
            "source_file": raw_path.name,
            "source_run_id": source_run_id,
            "excel_sheet_name": sheet_name,
            "specimen_id": specimen_id,
            "run_id": run_key,
            "support_span_mm": span_mm,
            "specimen_width_mm": run_geometry["beam_width_mm"],
            "specimen_thickness_mm": run_geometry["specimen_thickness_mm"],
            "number_of_layers": safe_float(meta_row.get("Layers (#)")),
            "fbg_configuration": fbg_configuration,
            "loading_group_index": group_idx,
            "repetition_index": repetition_idx,
            "global_repetition_index_within_run": global_idx,
            "timestamp_start": str(time_start) if not pd.isna(time_start) else "",
            "timestamp_center": str(time_center) if not pd.isna(time_center) else "",
            "timestamp_end": str(time_end) if not pd.isna(time_end) else "",
            "sample_start_index": int(start_idx),
            "sample_center_index": int((start_idx + end_idx) // 2),
            "sample_end_index": int(end_idx),
            "air_pressure_bar": air_pressure_bar,
            "force_N": force_n,
            "displacement_mm": displacement_mm,
            "force_group_index": group_idx,
            "force_step_order": group_idx,
            "force_increment_from_previous_group_N": force_inc,
            "force_increment_from_baseline_N": force_inc_from_baseline,
            "displacement_increment_from_previous_group_mm": disp_inc,
            "displacement_increment_from_baseline_mm": disp_inc_from_baseline,
            "loading_direction": compute_loading_direction(force_inc, disp_inc),
            "cycle_number_within_pressure_group": repetition_idx + 1,
            "normalized_position_within_run": global_idx / max(len(segments) - 1, 1),
            "normalized_position_within_loading_group": repetition_idx / max(reps_per_group - 1, 1),
            "time_since_run_start": time_since_run_start,
            "time_since_group_start": time_since_group_start,
            "label_crack_level": crack_level,
            "label_damage_transition": int(crack_level > 0),
            "segment_boundary_source": segment_meta.get("boundary_source", "unknown"),
        }

        segment_cube = np.full((preferred_channels, len(segment_df)), np.nan, dtype=float)
        resampled_cube = np.full((preferred_channels, target_length), np.nan, dtype=float)

        for channel_idx in range(preferred_channels):
            channel_name = f"WL_{channel_idx + 1}"
            csv_prefix = f"wl{channel_idx + 1}"
            if channel_idx < len(wl_cols):
                raw_channel = wl_cols[channel_idx]
                values = segment_df[raw_channel].to_numpy(dtype=float)
                segment_cube[channel_idx, :] = values
                resampled_cube[channel_idx, :] = resample_segment(values[np.isfinite(values)], target_length)
                channel_summary = compute_channel_summary(
                    values=values,
                    time_values=time_values,
                    run_baseline=run_baseline_by_channel.get(raw_channel, np.nan),
                    group_baseline=group_baseline_by_channel.get((group_idx, raw_channel), np.nan),
                )
                for name, value in channel_summary.items():
                    row[f"{csv_prefix}_{name}"] = value
                row[f"{csv_prefix}_signal_quality_missing_fraction"] = float(np.mean(~np.isfinite(values)))
                row[f"{csv_prefix}_signal_quality_available"] = 1
            else:
                for name in [
                    "mean", "median", "max", "min", "ptp", "std", "iqr", "slope", "start", "end",
                    "end_minus_start", "max_derivative", "mean_abs_derivative", "derivative_std",
                    "second_derivative_energy", "auc_baseline_subtracted", "residual_group_baseline",
                    "residual_run_baseline",
                ]:
                    row[f"{csv_prefix}_{name}"] = np.nan
                row[f"{csv_prefix}_signal_quality_missing_fraction"] = 1.0
                row[f"{csv_prefix}_signal_quality_available"] = 0

        for left_idx, right_idx in [(0, 1), (1, 2), (0, 2)]:
            left_prefix = f"wl{left_idx + 1}"
            right_prefix = f"wl{right_idx + 1}"
            row[f"{left_prefix}_minus_{right_prefix}_mean"] = row.get(f"{left_prefix}_mean", np.nan) - row.get(f"{right_prefix}_mean", np.nan)
            left_series = segment_cube[left_idx, :]
            right_series = segment_cube[right_idx, :]
            valid = np.isfinite(left_series) & np.isfinite(right_series)
            if valid.sum() >= 3:
                row[f"{left_prefix}_{right_prefix}_corr"] = float(np.corrcoef(left_series[valid], right_series[valid])[0, 1])
                left_deriv = np.diff(left_series[valid])
                right_deriv = np.diff(right_series[valid])
                if left_deriv.size >= 2 and right_deriv.size >= 2:
                    row[f"{left_prefix}_{right_prefix}_derivative_corr"] = float(np.corrcoef(left_deriv, right_deriv)[0, 1])
                else:
                    row[f"{left_prefix}_{right_prefix}_derivative_corr"] = np.nan
                row[f"{left_prefix}_{right_prefix}_std_ratio"] = float(np.std(left_series[valid]) / max(np.std(right_series[valid]), 1e-9))
                left_peak = int(np.nanargmax(left_series[valid]))
                right_peak = int(np.nanargmax(right_series[valid]))
                row[f"{left_prefix}_{right_prefix}_peak_timing_offset"] = float(left_peak - right_peak)
            else:
                row[f"{left_prefix}_{right_prefix}_corr"] = np.nan
                row[f"{left_prefix}_{right_prefix}_derivative_corr"] = np.nan
                row[f"{left_prefix}_{right_prefix}_std_ratio"] = np.nan
                row[f"{left_prefix}_{right_prefix}_peak_timing_offset"] = np.nan

        if not np.isnan(force_n) and not np.isnan(span_mm):
            bending_moment_n_mm = force_n * span_mm / 4.0
            bending_moment_n_m = force_n * (span_mm / 1000.0) / 4.0
        else:
            bending_moment_n_mm = np.nan
            bending_moment_n_m = np.nan
        row["bending_moment_N_mm"] = bending_moment_n_mm
        row["bending_moment_N_m"] = bending_moment_n_m
        row["y_fbg_mm"] = run_geometry["y_fbg_mm"]
        row["second_moment_mm4"] = run_geometry["second_moment_mm4"]
        row["youngs_modulus_gpa"] = run_geometry["youngs_modulus_gpa"]

        width_m = run_geometry["beam_width_mm"] / 1000.0
        thickness_m = run_geometry["specimen_thickness_mm"] / 1000.0
        y_m = run_geometry["y_fbg_mm"] / 1000.0
        second_moment_m4 = (width_m * (thickness_m ** 3)) / 12.0
        if not np.isnan(force_n) and not np.isnan(span_mm):
            expected_strain = (force_n * (span_mm / 1000.0) * y_m) / (4.0 * run_geometry["youngs_modulus_gpa"] * 1e9 * second_moment_m4)
            expected_strain_microstrain = expected_strain * 1e6
        else:
            expected_strain_microstrain = np.nan
        row["expected_elastic_strain_microstrain"] = expected_strain_microstrain

        if not np.isnan(row.get("wl2_residual_run_baseline", np.nan)) and not np.isnan(row.get("wl2_median", np.nan)):
            wl2_lambda0 = run_baseline_by_channel.get("WL_2", run_baseline_by_channel.get("WL_1", np.nan))
            delta_lambda_nm = row["wl2_median"] - wl2_lambda0 if not np.isnan(wl2_lambda0) else np.nan
            observed_photoelastic = (delta_lambda_nm / (wl2_lambda0 * (1.0 - run_geometry["photoelastic_coefficient"]))) * 1e6 if not np.isnan(delta_lambda_nm) and not np.isnan(wl2_lambda0) else np.nan
            observed_sensitivity = (delta_lambda_nm * 1000.0) / run_geometry["strain_sensitivity_pm_per_microstrain"] if not np.isnan(delta_lambda_nm) else np.nan
        else:
            observed_photoelastic = np.nan
            observed_sensitivity = np.nan
        row["observed_strain_microstrain_photoelastic"] = observed_photoelastic
        row["observed_strain_microstrain_sensitivity"] = observed_sensitivity
        row["observed_delta_lambda_nm_wl2"] = delta_lambda_nm if "delta_lambda_nm" in locals() else np.nan
        mechanics_residual = observed_photoelastic - expected_strain_microstrain if not np.isnan(observed_photoelastic) and not np.isnan(expected_strain_microstrain) else np.nan
        row["mechanics_residual_microstrain"] = mechanics_residual
        row["mechanics_residual_abs_microstrain"] = abs(mechanics_residual) if not np.isnan(mechanics_residual) else np.nan
        row["mechanics_residual_normalized"] = mechanics_residual / max(abs(expected_strain_microstrain), 1e-6) if not np.isnan(mechanics_residual) and not np.isnan(expected_strain_microstrain) else np.nan
        row["estimated_strain_per_force_level"] = expected_strain_microstrain / max(force_n, 1e-6) if not np.isnan(expected_strain_microstrain) and not np.isnan(force_n) else np.nan

        raw_segment_id = f"{source_run_id}__g{group_idx:02d}__r{repetition_idx:02d}"
        row["raw_segment_id"] = raw_segment_id
        raw_segment_ids.append(raw_segment_id)
        raw_original_segments.append(segment_cube)
        raw_resampled_segments.append(resampled_cube)
        raw_time_segments.append(time_values.astype(float))
        rows.append(row)

    run_df = pd.DataFrame(rows)
    run_df["mechanics_residual_change_from_previous_group"] = run_df.groupby("repetition_index")["mechanics_residual_microstrain"].diff()
    baseline_residual = run_df.loc[run_df["loading_group_index"] == run_df["loading_group_index"].min(), "mechanics_residual_microstrain"].median()
    run_df["mechanics_residual_change_from_baseline"] = run_df["mechanics_residual_microstrain"] - baseline_residual

    original_object = np.empty(len(raw_original_segments), dtype=object)
    time_object = np.empty(len(raw_time_segments), dtype=object)
    for idx, value in enumerate(raw_original_segments):
        original_object[idx] = value
    for idx, value in enumerate(raw_time_segments):
        time_object[idx] = value

    raw_payload = {
        "raw_segment_ids": np.asarray(raw_segment_ids, dtype=object),
        "original_segments": original_object,
        "resampled_segments": np.stack(raw_resampled_segments),
        "time_segments": time_object,
        "channel_names": np.asarray(wl_cols, dtype=object),
    }
    run_meta = {
        "run_id": run_key,
        "source_run_id": source_run_id,
        "sheet_name": sheet_name,
        "source_file": raw_path.name,
        "segment_count": len(segments),
        "detected_group_count": int(np.ceil(len(segments) / reps_per_group)),
        "segment_meta": segment_meta,
        "available_channels": wl_cols,
    }
    return run_df, run_meta, raw_payload


def build_data_dictionary() -> pd.DataFrame:
    rows = [
        ("source_file", "Raw interrogator filename.", "metadata"),
        ("source_run_id", "Unique source-file-derived run identity used for grouped evaluation.", "metadata"),
        ("excel_sheet_name", "Matched Excel metadata sheet.", "metadata"),
        ("specimen_id", "Inferred specimen family identifier without run suffix.", "metadata"),
        ("run_id", "Normalized workbook-linked run identifier.", "metadata"),
        ("loading_group_index", "Pressure/force group index within the run.", "index"),
        ("repetition_index", "Repetition index within the group.", "index"),
        ("global_repetition_index_within_run", "Sequential segment index within the run.", "index"),
        ("air_pressure_bar", "Excel air pressure metadata for the matched group.", "loading"),
        ("force_N", "Excel force metadata for the matched group.", "loading"),
        ("displacement_mm", "Excel displacement metadata for the matched group.", "loading"),
        ("label_crack_level", "Original Crack value from the workbook, with NaN mapped to 0.", "label"),
        ("label_damage_transition", "Binary target equal to 1 when crack level > 0.", "label"),
        ("expected_elastic_strain_microstrain", "Euler-Bernoulli expected strain using the single-source constants file.", "mechanics"),
        ("observed_strain_microstrain_photoelastic", "Observed WL2 strain using the photoelastic formula.", "mechanics"),
        ("mechanics_residual_microstrain", "Observed photoelastic strain minus expected elastic strain.", "mechanics"),
        ("raw_segment_id", "Identifier into the run-level and dataset-level NPZ payloads.", "raw"),
        ("normalized_position_within_run", "Leakage-prone normalized segment order within the run.", "leakage_prone"),
        ("normalized_position_within_loading_group", "Leakage-prone normalized repetition order within the group.", "leakage_prone"),
    ]
    for wl_idx in range(1, 4):
        prefix = f"wl{wl_idx}"
        rows.extend(
            [
                (f"{prefix}_mean", f"Mean wavelength for WL{wl_idx} over the detected segment.", "fbg_summary"),
                (f"{prefix}_median", f"Median wavelength for WL{wl_idx} over the detected segment.", "fbg_summary"),
                (f"{prefix}_std", f"Standard deviation of WL{wl_idx} over the detected segment.", "fbg_summary"),
                (f"{prefix}_residual_group_baseline", f"Segment median minus group baseline for WL{wl_idx}.", "fbg_summary"),
                (f"{prefix}_residual_run_baseline", f"Segment median minus run baseline for WL{wl_idx}.", "fbg_summary"),
            ]
        )
    return pd.DataFrame(rows, columns=["column_name", "description", "category"])


def scalar_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    leakage_columns = {
        "air_pressure_bar",
        "force_N",
        "displacement_mm",
        "force_group_index",
        "force_step_order",
        "force_increment_from_previous_group_N",
        "force_increment_from_baseline_N",
        "displacement_increment_from_previous_group_mm",
        "displacement_increment_from_baseline_mm",
        "normalized_position_within_run",
        "normalized_position_within_loading_group",
        "time_since_run_start",
        "time_since_group_start",
    }
    mechanics_columns = [
        "expected_elastic_strain_microstrain",
        "observed_strain_microstrain_photoelastic",
        "observed_strain_microstrain_sensitivity",
        "mechanics_residual_microstrain",
        "mechanics_residual_abs_microstrain",
        "mechanics_residual_normalized",
        "mechanics_residual_change_from_previous_group",
        "mechanics_residual_change_from_baseline",
        "estimated_strain_per_force_level",
        "bending_moment_N_mm",
        "bending_moment_N_m",
    ]
    meta_columns = {
        "source_file", "excel_sheet_name", "specimen_id", "run_id", "timestamp_start", "timestamp_center", "timestamp_end",
        "raw_segment_id", "loading_direction", "segment_boundary_source", "fbg_configuration",
    }
    target_columns = {"label_crack_level", "label_damage_transition"}
    index_columns = {
        "loading_group_index", "repetition_index", "global_repetition_index_within_run", "sample_start_index",
        "sample_center_index", "sample_end_index", "cycle_number_within_pressure_group",
    }
    all_numeric = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    fbg_only = [col for col in all_numeric if col not in leakage_columns and col not in mechanics_columns and col not in target_columns and col not in index_columns]
    dataset_a = sorted([col for col in fbg_only if not col.startswith("support_span_") and not col.endswith("_gpa")])
    dataset_b = sorted(dataset_a + [col for col in mechanics_columns if col in df.columns])
    dataset_c = sorted(dataset_a + [col for col in leakage_columns if col in df.columns and "normalized_position_within_run" not in col])
    dataset_d = sorted([col for col in all_numeric if col not in target_columns])
    return {
        "dataset_A_fbg_only_local_features": dataset_a,
        "dataset_B_fbg_plus_mechanics_residual": dataset_b,
        "dataset_C_fbg_plus_loading_sequence": dataset_c,
        "dataset_D_full_feature_set": dataset_d,
    }


def save_dataset_variants(output_dir: Path, all_rows: pd.DataFrame, all_resampled: np.ndarray, mechanics_columns: Sequence[str]) -> Dict[str, object]:
    datasets_dir = ensure_dir(output_dir / "datasets")
    registry: Dict[str, object] = {}
    scalar_sets = scalar_feature_sets(all_rows)
    for dataset_name, feature_columns in scalar_sets.items():
        frame = all_rows[["run_id", "source_run_id", "source_file", "raw_segment_id", "label_damage_transition", "label_crack_level"] + feature_columns].copy()
        path = datasets_dir / f"{dataset_name}.csv"
        frame.to_csv(path, index=False)
        registry[dataset_name] = {
            "type": "scalar",
            "path": str(path),
            "feature_columns": feature_columns,
            "group_column": "source_run_id",
            "target_column": "label_damage_transition",
        }

    raw_manifest = all_rows[["run_id", "source_run_id", "source_file", "raw_segment_id", "label_damage_transition", "label_crack_level"]].copy()
    raw_manifest_path = datasets_dir / "dataset_E_raw_window_tensor_only_manifest.csv"
    raw_manifest.to_csv(raw_manifest_path, index=False)
    raw_npz_path = datasets_dir / "dataset_E_raw_window_tensor_only.npz"
    np.savez_compressed(
        raw_npz_path,
        X_raw=all_resampled,
        y=all_rows["label_damage_transition"].to_numpy(dtype=int),
        groups=all_rows["source_run_id"].to_numpy(dtype=object),
        raw_segment_id=all_rows["raw_segment_id"].to_numpy(dtype=object),
    )
    registry["dataset_E_raw_window_tensor_only"] = {
        "type": "raw",
        "path": str(raw_npz_path),
        "manifest_path": str(raw_manifest_path),
        "group_column": "source_run_id",
        "target_column": "label_damage_transition",
        "channels": 3,
        "sequence_length": int(all_resampled.shape[-1]),
    }

    mechanics_matrix = all_rows[list(mechanics_columns)].to_numpy(dtype=float)
    raw_mech_manifest_path = datasets_dir / "dataset_F_raw_window_tensor_plus_mechanics_manifest.csv"
    raw_manifest.to_csv(raw_mech_manifest_path, index=False)
    raw_mech_npz_path = datasets_dir / "dataset_F_raw_window_tensor_plus_mechanics.npz"
    np.savez_compressed(
        raw_mech_npz_path,
        X_raw=all_resampled,
        X_scalar=mechanics_matrix,
        y=all_rows["label_damage_transition"].to_numpy(dtype=int),
        groups=all_rows["source_run_id"].to_numpy(dtype=object),
        raw_segment_id=all_rows["raw_segment_id"].to_numpy(dtype=object),
        scalar_feature_names=np.asarray(list(mechanics_columns), dtype=object),
    )
    registry["dataset_F_raw_window_tensor_plus_mechanics"] = {
        "type": "raw_plus_scalar",
        "path": str(raw_mech_npz_path),
        "manifest_path": str(raw_mech_manifest_path),
        "group_column": "source_run_id",
        "target_column": "label_damage_transition",
        "channels": 3,
        "sequence_length": int(all_resampled.shape[-1]),
        "scalar_feature_names": list(mechanics_columns),
    }
    return registry


def build_reports(
    output_dir: Path,
    all_rows: pd.DataFrame,
    source_inventory: pd.DataFrame,
    current_merged_columns: pd.DataFrame,
    missing_channel_table: pd.DataFrame,
    lost_positive_table: pd.DataFrame,
    unmatched_raw: pd.DataFrame,
) -> None:
    positive_runs = all_rows.groupby("source_run_id")["label_damage_transition"].max().sum()
    summary_lines = [
        "# Enriched Dataset Summary",
        "",
        f"Output directory: `{output_dir}`",
        "",
        "## File inventory",
        "",
        write_markdown_table(source_inventory, max_rows=30),
        "",
        "## Current merged CSV diagnosis",
        "",
        "The legacy merged CSVs mainly contain repetition-level peak summaries and metadata. They do not preserve raw interrogator waveforms directly, which is why the rebuilt pipeline saves per-run NPZ payloads alongside enriched repetition tables.",
        "",
        "### Current merged column inventory",
        "",
        write_markdown_table(current_merged_columns, max_rows=30),
        "",
        "### Missing-channel table",
        "",
        write_markdown_table(missing_channel_table, max_rows=30),
        "",
        "### Crack-positive files lost in the legacy WL_ch2-only path",
        "",
        write_markdown_table(lost_positive_table, max_rows=30),
        "",
        "## Rebuilt dataset summary",
        "",
        f"- Runs processed: {all_rows['source_run_id'].nunique()}",
        f"- Positive runs: {int(positive_runs)}",
        f"- Repetition rows: {len(all_rows)}",
        f"- Positive repetition rows: {int(all_rows['label_damage_transition'].sum())}",
        "",
        "## Unmatched raw files",
        "",
        write_markdown_table(unmatched_raw, max_rows=30),
    ]
    (output_dir / "dataset_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def current_merged_diagnosis(merged_index: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    column_rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []
    lost_rows: List[Dict[str, object]] = []
    for run_key, df in sorted(merged_index.items()):
        source_name = str(df.iloc[0].get("source_file", run_key)) if "source_file" in df.columns else run_key
        column_rows.append(
            {
                "run_key": run_key,
                "n_rows": len(df),
                "columns": ", ".join(df.columns.astype(str)),
            }
        )
        required = ["WL_ch1", "WL_ch2", "WL_ch3"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            missing_rows.append(
                {
                    "run_key": run_key,
                    "missing_channels": ", ".join(missing),
                    "crack_positive_rows": int(df["Crack"].notna().sum()) if "Crack" in df.columns else 0,
                }
            )
            if "Crack" in df.columns and df["Crack"].notna().sum() > 0:
                lost_rows.append(
                    {
                        "run_key": run_key,
                        "missing_channels": ", ".join(missing),
                        "crack_positive_rows": int(df["Crack"].notna().sum()),
                    }
                )
    return pd.DataFrame(column_rows), pd.DataFrame(missing_rows), pd.DataFrame(lost_rows)


def source_inventory_table() -> pd.DataFrame:
    rows = [
        {
            "path": "interrogator-data/",
            "kind": "raw data",
            "use_in_new_pipeline": "yes",
            "notes": "Raw interrogator text files used for segment extraction and raw window storage.",
        },
        {
            "path": "source/data.xlsx",
            "kind": "metadata and labels",
            "use_in_new_pipeline": "yes",
            "notes": "Workbook used for force, displacement, pressure, layers, distance, and Crack labels.",
        },
        {
            "path": "output/20260508_111324/",
            "kind": "generated data",
            "use_in_new_pipeline": "partial",
            "notes": "Used only as optional legacy segment-boundary hints and for merged CSV diagnosis.",
        },
        {
            "path": "strain_data/manual_strain_test_data_corrected.csv",
            "kind": "calibration",
            "use_in_new_pipeline": "yes",
            "notes": "Independent strain-wavelength calibration artifact; not crack ground truth.",
        },
        {
            "path": "compute_mechanical_strain.py, compute_fbg_direct_strain.py, calculate_E_and_plot.py, fbg_force_estimation.py",
            "kind": "mechanics and calibration scripts",
            "use_in_new_pipeline": "audit only",
            "notes": "Used to reconcile constants and equations before mechanics features are trusted.",
        },
        {
            "path": "main_neural_network.py, run_cnn_kfold.py, analyze_dataset_pycaret.py",
            "kind": "legacy ML scripts",
            "use_in_new_pipeline": "audit only",
            "notes": "Used to document legacy split and feature leakage issues; not reused directly.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    constants = load_constants()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "output" / f"enriched_dataset_{timestamp}"
    ensure_dir(output_dir)
    ensure_dir(output_dir / "runs")
    ensure_dir(output_dir / "raw_segments")

    workbook = load_excel_workbook()
    sheet_lookup = map_sheet_keys(workbook.keys())
    merged_index = load_current_merged_index() if args.reuse_current_merged_boundaries else {}
    raw_files = list_raw_files()

    run_frames: List[pd.DataFrame] = []
    run_manifests: List[Dict[str, object]] = []
    unmatched_rows: List[Dict[str, object]] = []
    resampled_segments_all: List[np.ndarray] = []
    mechanics_columns = [
        "expected_elastic_strain_microstrain",
        "observed_strain_microstrain_photoelastic",
        "observed_strain_microstrain_sensitivity",
        "mechanics_residual_microstrain",
        "mechanics_residual_abs_microstrain",
        "mechanics_residual_normalized",
        "mechanics_residual_change_from_previous_group",
        "mechanics_residual_change_from_baseline",
        "estimated_strain_per_force_level",
        "bending_moment_N_mm",
        "bending_moment_N_m",
    ]

    for raw_path in raw_files:
        run_key = normalize_run_key(raw_path.stem)
        source_run_id = raw_path.stem.lower().replace(" ", "").replace("_", "-")
        sheet_name = infer_sheet_name(run_key, sheet_lookup)
        if sheet_name is None:
            unmatched_rows.append(
                {
                    "raw_file": raw_path.name,
                    "normalized_key": run_key,
                    "reason": "No matching Excel sheet",
                }
            )
            continue
        raw_df = parse_interrogator_file(raw_path)
        meta_df = workbook[sheet_name]
        current_merged_df = merged_index.get(run_key)
        try:
            run_df, run_meta, raw_payload = build_run_rows(
                raw_path=raw_path,
                raw_df=raw_df,
                meta_df=meta_df,
                sheet_name=sheet_name,
                run_key=run_key,
                source_run_id=source_run_id,
                constants=constants,
                current_merged_df=current_merged_df,
            )
        except Exception as exc:
            unmatched_rows.append(
                {
                    "raw_file": raw_path.name,
                    "normalized_key": run_key,
                    "reason": f"Build failure: {exc}",
                }
            )
            continue

        run_csv_path = output_dir / "runs" / f"{source_run_id}.csv"
        run_df.to_csv(run_csv_path, index=False)
        run_npz_path = output_dir / "raw_segments" / f"{source_run_id}.npz"
        np.savez_compressed(run_npz_path, **raw_payload)

        run_meta["run_csv_path"] = str(run_csv_path)
        run_meta["run_npz_path"] = str(run_npz_path)
        run_manifests.append(run_meta)
        run_frames.append(run_df)
        resampled_segments_all.append(raw_payload["resampled_segments"])

    if not run_frames:
        raise RuntimeError("No enriched runs were created.")

    all_rows = pd.concat(run_frames, ignore_index=True)
    all_rows.sort_values(["source_run_id", "global_repetition_index_within_run"], inplace=True)
    aggregated_csv_path = output_dir / "enriched_repetition_table.csv"
    all_rows.to_csv(aggregated_csv_path, index=False)

    all_resampled = np.concatenate(resampled_segments_all, axis=0)
    dataset_registry = save_dataset_variants(output_dir, all_rows, all_resampled, mechanics_columns)
    data_dictionary = build_data_dictionary()
    data_dictionary_path = output_dir / "data_dictionary.csv"
    data_dictionary.to_csv(data_dictionary_path, index=False, quoting=csv.QUOTE_MINIMAL)

    source_inventory = source_inventory_table()
    current_columns_df, missing_channels_df, lost_positive_df = current_merged_diagnosis(merged_index)
    unmatched_df = pd.DataFrame(unmatched_rows)
    build_reports(
        output_dir=output_dir,
        all_rows=all_rows,
        source_inventory=source_inventory,
        current_merged_columns=current_columns_df,
        missing_channel_table=missing_channels_df,
        lost_positive_table=lost_positive_df,
        unmatched_raw=unmatched_df,
    )

    metadata = {
        "created_at": datetime.now().isoformat(),
        "constants": constants,
        "aggregated_csv_path": str(aggregated_csv_path),
        "run_manifest": run_manifests,
        "dataset_registry": dataset_registry,
        "counts": {
            "runs_processed": int(all_rows["source_run_id"].nunique()),
            "positive_runs": int(all_rows.groupby("source_run_id")["label_damage_transition"].max().sum()),
            "repetition_rows": int(len(all_rows)),
            "positive_repetition_rows": int(all_rows["label_damage_transition"].sum()),
        },
        "equations_used": constants.get("equations", {}),
        "units": {
            "support_span_mm": "mm",
            "specimen_width_mm": "mm",
            "specimen_thickness_mm": "mm",
            "force_N": "N",
            "displacement_mm": "mm",
            "air_pressure_bar": "bar",
            "expected_elastic_strain_microstrain": "microstrain",
            "observed_strain_microstrain_photoelastic": "microstrain",
            "mechanics_residual_microstrain": "microstrain",
        },
    }
    dump_json(output_dir / "metadata.json", metadata)
    print(f"Enriched dataset written to: {output_dir}")


if __name__ == "__main__":
    main()
