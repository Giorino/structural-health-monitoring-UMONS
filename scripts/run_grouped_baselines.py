#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_common import (
    ROOT,
    build_group_cv,
    choose_inner_group_split,
    compute_binary_metrics,
    latest_enriched_dir_or_fail,
    load_constants,
    mean_ci,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grouped baselines on the enriched FBG datasets.")
    parser.add_argument("--enriched-dir", type=str, default=None, help="Path to a specific enriched dataset directory.")
    return parser.parse_args()


def load_registry(enriched_dir: Path) -> Dict[str, object]:
    return pd.read_json(enriched_dir / "metadata.json", typ="series")["dataset_registry"]  # type: ignore[index]


def load_repetition_table(enriched_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(enriched_dir / "enriched_repetition_table.csv")
    sort_group = "source_run_id" if "source_run_id" in df.columns else "run_id"
    df.sort_values([sort_group, "global_repetition_index_within_run"], inplace=True)
    return df


def make_models() -> Dict[str, object]:
    return {
        "majority_baseline": "majority",
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced")),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(random_state=42)),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "mlp_scalar": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=250, random_state=42)),
            ]
        ),
        "mechanics_residual_threshold_rule": "mechanics_threshold",
        "wl_shift_threshold_rule": "wl_shift_threshold",
    }


def threshold_candidates(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.asarray([0.0])
    quantiles = np.linspace(0.1, 0.9, 9)
    return np.unique(np.quantile(np.abs(finite), quantiles))


def fit_threshold_rule(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    groups: np.ndarray,
    feature_column: str,
) -> float:
    train_frame = frame.iloc[train_idx].reset_index(drop=True)
    inner_train_idx, inner_val_idx = choose_inner_group_split(
        train_frame["label_damage_transition"].to_numpy(dtype=int),
        train_frame["source_run_id"].to_numpy(dtype=object) if "source_run_id" in train_frame.columns else train_frame["run_id"].to_numpy(dtype=object),
    )
    values = np.nan_to_num(train_frame[feature_column].to_numpy(dtype=float), nan=0.0)
    candidates = threshold_candidates(values[inner_train_idx])
    best_threshold = float(candidates[0]) if len(candidates) else 0.0
    best_score = -1.0
    y_val = train_frame.iloc[inner_val_idx]["label_damage_transition"].to_numpy(dtype=int)
    for threshold in candidates:
        preds = (np.abs(values[inner_val_idx]) >= threshold).astype(int)
        score = precision_score(y_val, preds, zero_division=0) + recall_score(y_val, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def predict_model(
    model_name: str,
    model: object,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    X_train = train_df[list(feature_columns)].to_numpy(dtype=float)
    y_train = train_df["label_damage_transition"].to_numpy(dtype=int)
    X_test = test_df[list(feature_columns)].to_numpy(dtype=float)

    if model_name == "majority_baseline":
        majority = int(np.round(y_train.mean()) >= 0.5)
        pred = np.full(len(test_df), majority, dtype=int)
        return pred, pred.astype(float)
    if model_name == "mechanics_residual_threshold_rule":
        groups = train_df["source_run_id"].to_numpy(dtype=object) if "source_run_id" in train_df.columns else train_df["run_id"].to_numpy(dtype=object)
        threshold = fit_threshold_rule(train_df, np.arange(len(train_df)), groups, "mechanics_residual_microstrain")
        scores = np.abs(np.nan_to_num(test_df["mechanics_residual_microstrain"].to_numpy(dtype=float), nan=0.0))
        pred = (scores >= threshold).astype(int)
        return pred, scores
    if model_name == "wl_shift_threshold_rule":
        feature_name = "observed_delta_lambda_nm_wl2" if "observed_delta_lambda_nm_wl2" in train_df.columns else "wl2_residual_run_baseline"
        groups = train_df["source_run_id"].to_numpy(dtype=object) if "source_run_id" in train_df.columns else train_df["run_id"].to_numpy(dtype=object)
        threshold = fit_threshold_rule(train_df, np.arange(len(train_df)), groups, feature_name)
        scores = np.abs(np.nan_to_num(test_df[feature_name].to_numpy(dtype=float), nan=0.0))
        pred = (scores >= threshold).astype(int)
        return pred, scores

    assert hasattr(model, "fit")
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        score = 1.0 / (1.0 + np.exp(-decision))
    else:
        score = model.predict(X_test).astype(float)
    pred = (score >= 0.5).astype(int)
    return pred, score


def summarize_runs(model_name: str, dataset_name: str, fold_idx: int, test_df: pd.DataFrame, y_pred: np.ndarray) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    enriched = test_df.copy()
    enriched["y_pred"] = y_pred
    group_col = "source_run_id" if "source_run_id" in enriched.columns else "run_id"
    for run_id, run_df in enriched.groupby(group_col, sort=False):
        y_true = run_df["label_damage_transition"].to_numpy(dtype=int)
        y_hat = run_df["y_pred"].to_numpy(dtype=int)
        positives_true = int(y_true.sum())
        positives_pred = int(y_hat.sum())
        tp = int(((y_true == 1) & (y_hat == 1)).sum())
        fp = int(((y_true == 0) & (y_hat == 1)).sum())
        fn = int(((y_true == 1) & (y_hat == 0)).sum())
        first_true = run_df.loc[run_df["label_damage_transition"] == 1, "global_repetition_index_within_run"]
        first_pred = run_df.loc[run_df["y_pred"] == 1, "global_repetition_index_within_run"]
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "fold": fold_idx,
                "run_id": run_id,
                "source_file": run_df["source_file"].iloc[0],
                "true_positive_window_count": positives_true,
                "predicted_positive_window_count": positives_pred,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": float(precision_score(y_true, y_hat, zero_division=0)),
                "recall": float(recall_score(y_true, y_hat, zero_division=0)),
                "f1": float(compute_binary_metrics(y_true, y_hat)["f1"]),
                "balanced_accuracy": float(compute_binary_metrics(y_true, y_hat)["balanced_accuracy"]) if len(np.unique(y_true)) > 1 else np.nan,
                "first_true_damage_index": int(first_true.iloc[0]) if not first_true.empty else np.nan,
                "first_predicted_damage_index": int(first_pred.iloc[0]) if not first_pred.empty else np.nan,
                "detection_delay": (int(first_pred.iloc[0]) - int(first_true.iloc[0])) if (not first_true.empty and not first_pred.empty) else np.nan,
                "detected_at_least_once": int(positives_pred > 0),
            }
        )
    return rows


def evaluate_dataset(
    dataset_name: str,
    dataset_path: Path,
    repetition_df: pd.DataFrame,
    cv_plan_name: str,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    dataset_df = pd.read_csv(dataset_path)
    feature_columns = [col for col in dataset_df.columns if col not in {"run_id", "source_run_id", "source_file", "raw_segment_id", "label_damage_transition", "label_crack_level"}]
    frame = dataset_df.copy()
    if "global_repetition_index_within_run" not in frame.columns:
        meta_columns = repetition_df[["raw_segment_id", "global_repetition_index_within_run"]].copy()
        frame = frame.merge(meta_columns, on="raw_segment_id", how="left")
    models = make_models()

    fold_results: List[Dict[str, object]] = []
    window_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        train_df = frame.iloc[train_idx].reset_index(drop=True)
        test_df = frame.iloc[test_idx].reset_index(drop=True)
        for model_name, model in models.items():
            if model_name == "mechanics_residual_threshold_rule" and "mechanics_residual_microstrain" not in train_df.columns:
                continue
            if model_name == "wl_shift_threshold_rule" and not any(col in train_df.columns for col in ["observed_delta_lambda_nm_wl2", "wl2_residual_run_baseline"]):
                continue
            preds, scores = predict_model(model_name, model, train_df, test_df, feature_columns)
            metrics = compute_binary_metrics(test_df["label_damage_transition"].to_numpy(dtype=int), preds, scores)
            fold_results.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "fold": fold_idx,
                    "cv_plan": cv_plan_name,
                    "n_train_rows": len(train_df),
                    "n_test_rows": len(test_df),
                    "n_train_runs": train_df["source_run_id"].nunique() if "source_run_id" in train_df.columns else train_df["run_id"].nunique(),
                    "n_test_runs": test_df["source_run_id"].nunique() if "source_run_id" in test_df.columns else test_df["run_id"].nunique(),
                    **metrics,
                }
            )
            window_cols = ["run_id", "source_file", "raw_segment_id", "global_repetition_index_within_run", "label_damage_transition"]
            if "source_run_id" in test_df.columns:
                window_cols.insert(1, "source_run_id")
            fold_window = test_df[window_cols].copy()
            fold_window["dataset"] = dataset_name
            fold_window["model"] = model_name
            fold_window["fold"] = fold_idx
            fold_window["y_pred"] = preds
            fold_window["y_score"] = scores
            window_rows.extend(fold_window.to_dict("records"))
            run_rows.extend(summarize_runs(model_name, dataset_name, fold_idx, test_df, preds))
    return fold_results, window_rows, run_rows


def summarize_grouped_results(fold_results: pd.DataFrame, run_results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = fold_results.groupby(["dataset", "model", "cv_plan"], sort=False)
    for (dataset, model, cv_plan), group_df in grouped:
        row: Dict[str, object] = {"dataset": dataset, "model": model, "cv_plan": cv_plan}
        for metric in ["precision", "recall", "f1", "balanced_accuracy", "roc_auc"]:
            mean_value, ci = mean_ci(group_df[metric].tolist())
            row[f"window_{metric}_mean"] = mean_value
            row[f"window_{metric}_ci95"] = ci
        run_subset = run_results[(run_results["dataset"] == dataset) & (run_results["model"] == model)]
        for metric in ["precision", "recall", "f1", "balanced_accuracy", "detected_at_least_once"]:
            mean_value, ci = mean_ci(run_subset[metric].tolist())
            row[f"run_{metric}_mean"] = mean_value
            row[f"run_{metric}_ci95"] = ci
        rows.append(row)
    return pd.DataFrame(rows)


def run_ablations(repetition_df: pd.DataFrame, cv_plan_name: str, splits: Sequence[Tuple[np.ndarray, np.ndarray]], results_dir: Path) -> pd.DataFrame:
    all_feature_columns = [col for col in repetition_df.columns if pd.api.types.is_numeric_dtype(repetition_df[col]) and col not in {"label_damage_transition", "label_crack_level"}]
    feature_groups = {
        "remove_air_pressure": [col for col in all_feature_columns if col == "air_pressure_bar"],
        "remove_force": [col for col in all_feature_columns if col.startswith("force_") or col == "force_N" or col.startswith("bending_moment_")],
        "remove_displacement": [col for col in all_feature_columns if "displacement" in col],
        "remove_all_loading_sequence": [col for col in all_feature_columns if col in {"air_pressure_bar", "force_group_index", "force_step_order", "normalized_position_within_run", "normalized_position_within_loading_group", "time_since_run_start", "time_since_group_start"} or col.startswith("force_") or "displacement" in col],
        "remove_normalized_position_within_run": ["normalized_position_within_run"],
        "remove_mechanics_residuals": [col for col in all_feature_columns if "mechanics" in col or "expected_elastic_strain" in col or "observed_strain" in col],
        "remove_cross_channel_features": [col for col in all_feature_columns if "_corr" in col or "_std_ratio" in col or "_peak_timing_offset" in col or "_minus_" in col],
        "wl2_only": [col for col in all_feature_columns if not (col.startswith("wl2_") or col in {"number_of_layers", "specimen_thickness_mm", "specimen_width_mm", "support_span_mm", "y_fbg_mm"})],
        "wl123_together": [],
        "raw_waveform_only_proxy": [col for col in all_feature_columns if not (col.startswith("wl") and any(token in col for token in ["slope", "derivative", "second_derivative_energy", "auc", "start", "end", "ptp", "std", "iqr", "mean", "median"]))],
        "summary_statistics_only": [col for col in all_feature_columns if "_corr" in col or "mechanics" in col or "expected_elastic_strain" in col or "observed_strain" in col or col.startswith("force_") or "displacement" in col or col == "air_pressure_bar"],
        "mechanics_only": [col for col in all_feature_columns if not ("mechanics" in col or "expected_elastic_strain" in col or "observed_strain" in col or col.startswith("bending_moment_") or col == "estimated_strain_per_force_level")],
        "full_feature_set": [],
    }

    rows: List[Dict[str, object]] = []
    for ablation_name, removed_columns in feature_groups.items():
        feature_columns = [col for col in all_feature_columns if col not in removed_columns]
        fold_scores = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            train_df = repetition_df.iloc[train_idx].reset_index(drop=True)
            test_df = repetition_df.iloc[test_idx].reset_index(drop=True)
            model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            )
            preds, scores = predict_model("logistic_regression", model, train_df, test_df, feature_columns)
            metrics = compute_binary_metrics(test_df["label_damage_transition"].to_numpy(dtype=int), preds, scores)
            fold_scores.append(metrics["f1"])
            rows.append(
                {
                    "ablation_name": ablation_name,
                    "fold": fold_idx,
                    "cv_plan": cv_plan_name,
                    "n_features": len(feature_columns),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "roc_auc": metrics["roc_auc"],
                }
            )
    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(results_dir / "ablation_results.csv", index=False)

    summary_lines = ["# Ablation Summary", ""]
    for ablation_name, group_df in ablation_df.groupby("ablation_name", sort=False):
        mean_f1, ci_f1 = mean_ci(group_df["f1"].tolist())
        mean_bal, ci_bal = mean_ci(group_df["balanced_accuracy"].tolist())
        summary_lines.append(f"- {ablation_name}: mean F1={mean_f1:.3f} +/- {ci_f1:.3f}, mean balanced accuracy={mean_bal:.3f} +/- {ci_bal:.3f}")
    (results_dir / "ablation_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    return ablation_df


def main() -> None:
    args = parse_args()
    enriched_dir = Path(args.enriched_dir) if args.enriched_dir else latest_enriched_dir_or_fail()
    metadata = pd.read_json(enriched_dir / "metadata.json", typ="series")
    registry: Dict[str, object] = metadata["dataset_registry"]  # type: ignore[index]
    repetition_df = load_repetition_table(enriched_dir)
    y = repetition_df["label_damage_transition"].to_numpy(dtype=int)
    groups = repetition_df["source_run_id"].to_numpy(dtype=object) if "source_run_id" in repetition_df.columns else repetition_df["run_id"].to_numpy(dtype=object)
    cv_plan = build_group_cv(y, groups, max_splits=5)

    scalar_dataset_names = [
        "dataset_A_fbg_only_local_features",
        "dataset_B_fbg_plus_mechanics_residual",
        "dataset_C_fbg_plus_loading_sequence",
        "dataset_D_full_feature_set",
    ]

    fold_rows: List[Dict[str, object]] = []
    window_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []
    for dataset_name in scalar_dataset_names:
        dataset_path = Path(registry[dataset_name]["path"])  # type: ignore[index]
        dataset_fold_rows, dataset_window_rows, dataset_run_rows = evaluate_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            repetition_df=repetition_df,
            cv_plan_name=cv_plan.name,
            splits=cv_plan.splits,
        )
        fold_rows.extend(dataset_fold_rows)
        window_rows.extend(dataset_window_rows)
        run_rows.extend(dataset_run_rows)

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    fold_df = pd.DataFrame(fold_rows)
    window_df = pd.DataFrame(window_rows)
    run_df = pd.DataFrame(run_rows)
    summary_df = summarize_grouped_results(fold_df, run_df)

    summary_df.to_csv(results_dir / "grouped_cv_summary.csv", index=False)
    run_df.to_csv(results_dir / "run_level_results.csv", index=False)
    window_df.to_csv(results_dir / "window_level_results.csv", index=False)

    run_ablations(repetition_df, cv_plan.name, cv_plan.splits, results_dir)
    print(f"Grouped baseline results written to: {results_dir}")


if __name__ == "__main__":
    main()
