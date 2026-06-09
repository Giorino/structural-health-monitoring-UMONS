#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, medfilt
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CVPlan:
    name: str
    splits: List[Tuple[np.ndarray, np.ndarray]]


def normalize_run_key(value: str) -> str:
    cleaned = value.lower().strip()
    cleaned = re.sub(r"_\d{8}(?:_\d{4,6})?$", "", cleaned)
    cleaned = cleaned.replace("_", "-")
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace("--", "-")
    cleaned = cleaned.replace("-layers", "layers")
    cleaned = cleaned.replace("cm-", "cm-")
    cleaned = cleaned.replace("-interrogator", "")
    cleaned = cleaned.replace(".txt", "")
    cleaned = cleaned.replace(".csv", "")
    cleaned = re.sub(r"^merged-", "", cleaned)
    cleaned = re.sub(r"^merged_", "", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned.strip("-")


def infer_specimen_id(run_id: str) -> str:
    normalized = normalize_run_key(run_id)
    normalized = re.sub(r"-\d+(?:-s)?$", "", normalized)
    return normalized


def parse_simple_yaml(path: Path) -> Dict[str, object]:
    stack: List[Tuple[int, Dict[str, object]]] = [(-1, {})]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = line.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            value = parse_yaml_scalar(stripped[2:])
            parent.setdefault("__list__", []).append(value)
            continue
        if ":" not in stripped:
            continue
        key, value_text = stripped.split(":", 1)
        key = key.strip()
        value_text = value_text.strip()
        if not value_text:
            node: Dict[str, object] = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = parse_yaml_scalar(value_text)
    return cleanup_yaml_lists(stack[0][1])


def cleanup_yaml_lists(node: object) -> object:
    if isinstance(node, dict):
        if "__list__" in node and len(node) == 1:
            return node["__list__"]
        return {key: cleanup_yaml_lists(value) for key, value in node.items() if key != "__list__"}
    return node


def parse_yaml_scalar(value_text: str) -> object:
    if value_text in {"null", "None"}:
        return None
    if value_text in {"true", "True"}:
        return True
    if value_text in {"false", "False"}:
        return False
    if value_text.startswith('"') and value_text.endswith('"'):
        return value_text[1:-1]
    if value_text.startswith("'") and value_text.endswith("'"):
        return value_text[1:-1]
    try:
        if "." in value_text or "e" in value_text.lower():
            return float(value_text)
        return int(value_text)
    except ValueError:
        return value_text


def load_constants() -> Dict[str, object]:
    return parse_simple_yaml(ROOT / "config" / "mechanics_constants.yaml")


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def find_latest_existing_output() -> Optional[Path]:
    output_root = ROOT / "output"
    if not output_root.exists():
        return None
    candidates = sorted([p for p in output_root.iterdir() if p.is_dir() and p.name.startswith("enriched_dataset_")])
    return candidates[-1] if candidates else None


def list_raw_files(raw_dir: Optional[Path] = None) -> List[Path]:
    base_dir = raw_dir if raw_dir is not None else ROOT / "interrogator-data"
    return sorted(base_dir.glob("*-interrogator.txt"))


def load_excel_workbook(workbook_path: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    excel_path = workbook_path if workbook_path is not None else ROOT / "source" / "data.xlsx"
    workbook = pd.ExcelFile(excel_path)
    sheets: Dict[str, pd.DataFrame] = {}
    for sheet_name in workbook.sheet_names:
        df = pd.read_excel(workbook, sheet_name=sheet_name)
        non_crack_cols = [col for col in df.columns if str(col).strip().lower() != "crack"]
        if non_crack_cols:
            df[non_crack_cols] = df[non_crack_cols].ffill()
        sheets[sheet_name] = df
    return sheets


def map_sheet_keys(sheets: Iterable[str]) -> Dict[str, str]:
    return {normalize_run_key(name): name for name in sheets}


def parse_interrogator_file(path: Path) -> pd.DataFrame:
    first_data_tokens: Optional[List[str]] = None
    data_start = 0
    iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}T")
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for i, line in enumerate(handle):
            if i >= 100:
                break
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if iso_re.match(stripped) and len(parts) >= 3:
                first_data_tokens = parts
                data_start = i
                break
            try:
                float(parts[0])
                if len(parts) >= 2:
                    first_data_tokens = parts
                    data_start = i
                    break
            except Exception:
                continue
    if first_data_tokens is None:
        raise RuntimeError(f"Could not find data start in {path}")

    n_tokens = len(first_data_tokens)
    if iso_re.match(first_data_tokens[0]):
        names = ["Timestamp", "Time_s"]
        wl_count = max(0, n_tokens - 2)
    else:
        names = ["Time_s"]
        wl_count = max(0, n_tokens - 1)
    names += [f"WL_{i}" for i in range(1, wl_count + 1)]

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=names,
        skiprows=data_start,
        engine="python",
        on_bad_lines="skip",
    )
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if "Time_s" in df.columns:
        df["Time_s"] = pd.to_numeric(df["Time_s"], errors="coerce")
    for col in [col for col in df.columns if col.startswith("WL_")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def available_wl_columns(df: pd.DataFrame, limit: int = 3) -> List[str]:
    wl_cols = [col for col in df.columns if col.startswith("WL_")]
    valid_cols = [col for col in wl_cols if df[col].notna().sum() > 0]
    return valid_cols[:limit]


def build_representative_signal(df: pd.DataFrame, wl_cols: Sequence[str]) -> np.ndarray:
    if not wl_cols:
        raise RuntimeError("No wavelength columns available for representative signal.")
    valid_cols = [col for col in wl_cols if df[col].notna().sum() > 0]
    if not valid_cols:
        raise RuntimeError("All wavelength columns are empty.")
    variances = {col: float(df[col].std(skipna=True)) for col in valid_cols}
    best_col = max(variances, key=variances.get)
    return df[best_col].to_numpy(dtype=float)


def smooth_signal(signal: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return medfilt(signal, kernel_size=kernel)


def choose_search_params(run_key: str) -> Dict[str, List[float]]:
    match = re.match(r"(?P<dist>\d+)cm", run_key)
    dist_cm = int(match.group("dist")) if match else None
    if dist_cm is not None and dist_cm >= 23:
        return {
            "peak_heights": [0.2, 0.3, 0.15, 0.4, 0.1, 0.5],
            "prominences": [0.2, 0.15, 0.3, 0.1, 0.5],
            "kernels": [7, 9],
        }
    return {
        "peak_heights": [0.15, 0.1, 0.2, 0.08, 0.25, 0.3],
        "prominences": [0.1, 0.15, 0.08, 0.2, 0.3],
        "kernels": [7, 5],
    }


def detect_segments_with_search(
    run_key: str,
    df: pd.DataFrame,
    wl_cols: Sequence[str],
    min_segment_length: int,
) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    rep_signal = build_representative_signal(df, wl_cols)
    best_segments: List[Tuple[int, int]] = []
    best_meta: Dict[str, float] = {}
    best_distance = float("inf")
    params = choose_search_params(run_key)
    target_segments = None
    for kernel in params["kernels"]:
        smoothed = smooth_signal(rep_signal, kernel)
        for peak_height in params["peak_heights"]:
            for prominence in params["prominences"]:
                peaks, props = find_peaks(
                    smoothed,
                    height=peak_height,
                    width=min_segment_length,
                    distance=20,
                    prominence=prominence,
                )
                segments: List[Tuple[int, int]] = []
                if len(peaks) > 0:
                    left_edges = np.floor(props["left_ips"]).astype(int)
                    right_edges = np.ceil(props["right_ips"]).astype(int)
                    for start, end in zip(left_edges, right_edges):
                        if end > start + min_segment_length:
                            segments.append((int(start), int(end)))
                if target_segments is None:
                    target_segments = 120
                distance = abs(len(segments) - target_segments)
                if segments and (distance < best_distance or (distance == best_distance and len(segments) > len(best_segments))):
                    best_segments = segments
                    best_distance = distance
                    best_meta = {
                        "peak_height": float(peak_height),
                        "peak_prominence": float(prominence),
                        "smooth_kernel": float(kernel),
                    }
                if len(segments) >= 100:
                    return segments, {
                        "peak_height": float(peak_height),
                        "peak_prominence": float(prominence),
                        "smooth_kernel": float(kernel),
                    }
    return best_segments, best_meta


def detect_segments_for_run(
    run_key: str,
    raw_df: pd.DataFrame,
    wl_cols: Sequence[str],
    current_merged_row: Optional[pd.DataFrame],
    min_segment_length: int,
) -> Tuple[List[Tuple[int, int]], Dict[str, object]]:
    if current_merged_row is not None and {"segment_start_idx", "segment_end_idx"}.issubset(current_merged_row.columns):
        boundaries = []
        seen = set()
        for _, row in current_merged_row.sort_values(["segment_start_idx", "segment_end_idx"]).iterrows():
            pair = (int(row["segment_start_idx"]), int(row["segment_end_idx"]))
            if pair not in seen:
                boundaries.append(pair)
                seen.add(pair)
        if boundaries:
            return boundaries, {"boundary_source": "current_merged_csv"}
    segments, meta = detect_segments_with_search(run_key, raw_df, wl_cols, min_segment_length)
    meta["boundary_source"] = "raw_peak_search"
    return segments, meta


def parse_geometry_from_run(run_key: str, layers: Optional[float], constants: Dict[str, object]) -> Dict[str, float]:
    geometry = constants["geometry"]  # type: ignore[index]
    material = constants["material"]  # type: ignore[index]
    layers_value = int(layers) if layers and not pd.isna(layers) else None
    if layers_value is None:
        match = re.search(r"-(\d+)layers", run_key)
        layers_value = int(match.group(1)) if match else 12
    layer_thickness_mm = float(geometry["layer_thickness_mm"])  # type: ignore[index]
    thickness_mm = layers_value * layer_thickness_mm
    width_mm = float(geometry["beam_width_mm"])  # type: ignore[index]
    if run_key.endswith("-s"):
        width_mm = float(geometry["default_small_sample_width_mm"])  # type: ignore[index]
    override_y = geometry.get("default_y_fbg_mm_override")  # type: ignore[union-attr]
    if override_y is None:
        layers_below_top = float(geometry["fbg_layers_below_top_surface"])  # type: ignore[index]
        y_from_bottom_mm = thickness_mm - (layers_below_top * layer_thickness_mm)
        y_fbg_mm = y_from_bottom_mm - (thickness_mm / 2.0)
    else:
        y_fbg_mm = float(override_y)
    second_moment_mm4 = (width_mm * (thickness_mm ** 3)) / 12.0
    return {
        "beam_width_mm": width_mm,
        "specimen_thickness_mm": thickness_mm,
        "layer_thickness_mm": layer_thickness_mm,
        "y_fbg_mm": y_fbg_mm,
        "second_moment_mm4": second_moment_mm4,
        "youngs_modulus_gpa": float(material["youngs_modulus_gpa"]),  # type: ignore[index]
        "photoelastic_coefficient": float(material["photoelastic_coefficient"]),  # type: ignore[index]
        "strain_sensitivity_pm_per_microstrain": float(material["strain_sensitivity_pm_per_microstrain"]),  # type: ignore[index]
    }


def compute_channel_summary(
    values: np.ndarray,
    time_values: np.ndarray,
    run_baseline: float,
    group_baseline: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if values.size == 0 or np.all(np.isnan(values)):
        return {key: np.nan for key in [
            "mean", "median", "max", "min", "ptp", "std", "iqr", "slope",
            "start", "end", "end_minus_start", "max_derivative", "mean_abs_derivative",
            "derivative_std", "second_derivative_energy", "auc_baseline_subtracted",
            "residual_group_baseline", "residual_run_baseline",
        ]}

    valid_mask = np.isfinite(values)
    vals = values[valid_mask]
    idx = np.arange(vals.size) if time_values.size != values.size else np.asarray(time_values[valid_mask], dtype=float)
    if vals.size < 2:
        slope = 0.0
        deriv = np.array([0.0])
        second_deriv = np.array([0.0])
        auc_value = 0.0
    else:
        centered_x = idx - idx.mean()
        if np.allclose(centered_x.std(), 0.0):
            slope = 0.0
        else:
            slope = float(np.polyfit(idx, vals, 1)[0])
        step = np.diff(idx)
        step[step == 0] = 1.0
        deriv = np.diff(vals) / step
        second_deriv = np.diff(deriv)
        auc_value = float(np.trapezoid(vals - run_baseline, idx))

    out["mean"] = float(np.mean(vals))
    out["median"] = float(np.median(vals))
    out["max"] = float(np.max(vals))
    out["min"] = float(np.min(vals))
    out["ptp"] = float(np.ptp(vals))
    out["std"] = float(np.std(vals, ddof=0))
    out["iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    out["slope"] = float(slope)
    out["start"] = float(vals[0])
    out["end"] = float(vals[-1])
    out["end_minus_start"] = float(vals[-1] - vals[0])
    out["max_derivative"] = float(np.max(np.abs(deriv))) if deriv.size else 0.0
    out["mean_abs_derivative"] = float(np.mean(np.abs(deriv))) if deriv.size else 0.0
    out["derivative_std"] = float(np.std(deriv, ddof=0)) if deriv.size else 0.0
    out["second_derivative_energy"] = float(np.sum(second_deriv ** 2)) if second_deriv.size else 0.0
    out["auc_baseline_subtracted"] = auc_value
    out["residual_group_baseline"] = float(out["median"] - group_baseline)
    out["residual_run_baseline"] = float(out["median"] - run_baseline)
    return out


def resample_segment(values: np.ndarray, target_length: int) -> np.ndarray:
    if values.size == 0:
        return np.full(target_length, np.nan, dtype=float)
    if values.size == 1:
        return np.full(target_length, float(values[0]), dtype=float)
    x_old = np.linspace(0.0, 1.0, num=values.size)
    x_new = np.linspace(0.0, 1.0, num=target_length)
    return np.interp(x_new, x_old, values.astype(float))


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    positives = y_true == 1
    negatives = y_true == 0
    tpr = float((y_pred[positives] == 1).mean()) if positives.any() else 0.0
    tnr = float((y_pred[negatives] == 0).mean()) if negatives.any() else 0.0
    if positives.any() and negatives.any():
        balanced_accuracy = 0.5 * (tpr + tnr)
    elif positives.any():
        balanced_accuracy = tpr
    elif negatives.any():
        balanced_accuracy = tnr
    else:
        balanced_accuracy = np.nan
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        finite_mask = np.isfinite(y_score)
        if finite_mask.sum() >= 2 and len(np.unique(y_true[finite_mask])) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true[finite_mask], y_score[finite_mask]))
        else:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def build_group_cv(y: np.ndarray, groups: np.ndarray, max_splits: int = 5) -> CVPlan:
    group_df = pd.DataFrame({"group": groups, "label": y})
    group_labels = group_df.groupby("group")["label"].max().reset_index()
    n_groups = len(group_labels)
    positive_groups = int(group_labels["label"].sum())
    negative_groups = n_groups - positive_groups
    if positive_groups >= 2 and negative_groups >= 2:
        n_splits = min(max_splits, positive_groups, negative_groups, n_groups)
        if n_splits >= 2:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(splitter.split(np.zeros_like(y), y, groups))
            return CVPlan(name=f"StratifiedGroupKFold_{n_splits}", splits=splits)
    if n_groups >= 3:
        n_splits = min(max_splits, n_groups)
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(np.zeros_like(y), y, groups))
        return CVPlan(name=f"GroupKFold_{n_splits}", splits=splits)
    splitter = LeaveOneGroupOut()
    splits = list(splitter.split(np.zeros_like(y), y, groups))
    return CVPlan(name="LeaveOneGroupOut", splits=splits)


def choose_inner_group_split(y: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    plan = build_group_cv(y, groups, max_splits=4)
    if not plan.splits:
        indices = np.arange(len(y))
        return indices, indices
    return plan.splits[0]


def mean_ci(series: Sequence[float]) -> Tuple[float, float]:
    values = np.asarray([x for x in series if not pd.isna(x)], dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), 0.0
    mean = float(values.mean())
    stderr = float(values.std(ddof=1) / math.sqrt(values.size))
    ci95 = 1.96 * stderr
    return mean, ci95


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_enriched_dir_or_fail() -> Path:
    latest = find_latest_existing_output()
    if latest is None:
        raise RuntimeError("No enriched dataset directory found under output/.")
    return latest


def write_markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows_"
    view = df.head(max_rows).copy()
    header = "| " + " | ".join(view.columns.astype(str)) + " |"
    divider = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.itertuples(index=False, name=None)]
    extra = ""
    if len(df) > max_rows:
        extra = f"\n\nShowing first {max_rows} of {len(df)} rows."
    return "\n".join([header, divider] + rows) + extra
