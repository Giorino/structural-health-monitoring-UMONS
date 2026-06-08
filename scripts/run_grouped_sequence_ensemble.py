#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_common import (
    ROOT,
    build_group_cv,
    choose_inner_group_split,
    compute_binary_metrics,
    latest_enriched_dir_or_fail,
    mean_ci,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grouped short-sequence ensemble models on enriched scalar datasets.")
    parser.add_argument("--enriched-dir", type=str, default=None)
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="dataset_C_fbg_plus_loading_sequence",
        choices=[
            "dataset_A_fbg_only_local_features",
            "dataset_B_fbg_plus_mechanics_residual",
            "dataset_C_fbg_plus_loading_sequence",
            "dataset_D_full_feature_set",
        ],
    )
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--label-mode", choices=["last", "max"], default="max")
    parser.add_argument("--model", choices=["extratrees", "randomforest", "hgb", "logreg"], default="extratrees")
    parser.add_argument("--representation", choices=["flat", "flat_plus_delta"], default="flat")
    parser.add_argument("--threshold-target", choices=["window_f1", "run_detected_f1"], default="window_f1")
    return parser.parse_args()


def compute_run_detection_f1(y_true: np.ndarray, preds: np.ndarray, groups: np.ndarray) -> float:
    run_true: List[int] = []
    run_pred: List[int] = []
    frame = pd.DataFrame({"y_true": y_true, "y_pred": preds, "group": groups})
    for _, run_df in frame.groupby("group", sort=False):
        run_true.append(int(run_df["y_true"].max()))
        run_pred.append(int(run_df["y_pred"].max()))
    return float(compute_binary_metrics(np.asarray(run_true), np.asarray(run_pred))["f1"])


def tune_threshold(y_true: np.ndarray, scores: np.ndarray, groups: np.ndarray, threshold_target: str) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 19), scores]))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        preds = (scores >= threshold).astype(int)
        if threshold_target == "run_detected_f1":
            score = compute_run_detection_f1(y_true, preds, groups)
        else:
            score = float(compute_binary_metrics(y_true, preds, scores)["f1"])
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def make_model(model_name: str):
    if model_name == "extratrees":
        return Pipeline(
            steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced", min_samples_leaf=2)),
            ]
        )
    if model_name == "randomforest":
        return Pipeline(
            steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced_subsample", min_samples_leaf=2)),
            ]
        )
    if model_name == "hgb":
        return Pipeline(
            steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(random_state=42, max_depth=6, learning_rate=0.05, max_iter=300)),
            ]
        )
    return Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )


def encode_sequence(values: np.ndarray, representation: str) -> np.ndarray:
    encoded = [values.reshape(-1)]
    if representation == "flat_plus_delta":
        encoded.append(np.diff(values, axis=0).reshape(-1))
    return np.concatenate(encoded, axis=0)


def build_flattened_sequences(
    df: pd.DataFrame,
    feature_columns: List[str],
    sequence_length: int,
    label_mode: str,
    representation: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rows: List[np.ndarray] = []
    labels: List[int] = []
    groups: List[object] = []
    manifest_rows: List[Dict[str, object]] = []
    group_col = "source_run_id" if "source_run_id" in df.columns else "run_id"
    for _, run_df in df.groupby(group_col, sort=False):
        run_df = run_df.sort_values("global_repetition_index_within_run").reset_index(drop=True)
        if len(run_df) < sequence_length:
            continue
        values = run_df[feature_columns].to_numpy(dtype=float)
        y = run_df["label_damage_transition"].to_numpy(dtype=int)
        for start in range(0, len(run_df) - sequence_length + 1):
            end = start + sequence_length
            rows.append(encode_sequence(values[start:end], representation))
            labels.append(int(y[start:end].max()) if label_mode == "max" else int(y[end - 1]))
            groups.append(run_df[group_col].iloc[-1])
            manifest_rows.append(
                {
                    "run_id": run_df["run_id"].iloc[-1],
                    "source_run_id": run_df[group_col].iloc[-1],
                    "source_file": run_df["source_file"].iloc[-1],
                    "raw_segment_id": run_df["raw_segment_id"].iloc[end - 1],
                    "label_damage_transition": int(labels[-1]),
                    "global_repetition_index_within_run": int(run_df["global_repetition_index_within_run"].iloc[end - 1]),
                    "sequence_start_index_within_run": int(run_df["global_repetition_index_within_run"].iloc[start]),
                    "sequence_end_index_within_run": int(run_df["global_repetition_index_within_run"].iloc[end - 1]),
                }
            )
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), np.asarray(groups, dtype=object), pd.DataFrame(manifest_rows)


def drop_train_empty_columns(train_X: np.ndarray, *other_arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    keep_mask = ~np.all(np.isnan(train_X), axis=0)
    filtered = [train_X[:, keep_mask]]
    for arr in other_arrays:
        filtered.append(arr[:, keep_mask])
    return tuple(filtered)


def summarize_runs(dataset_name: str, model_name: str, fold_idx: int, test_df: pd.DataFrame, y_pred: np.ndarray) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    enriched = test_df.copy()
    enriched["y_pred"] = y_pred
    for run_id, run_df in enriched.groupby("source_run_id", sort=False):
        y_true = run_df["label_damage_transition"].to_numpy(dtype=int)
        y_hat = run_df["y_pred"].to_numpy(dtype=int)
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "fold": fold_idx,
                "run_id": run_id,
                "source_file": run_df["source_file"].iloc[0],
                "precision": float(compute_binary_metrics(y_true, y_hat)["precision"]),
                "recall": float(compute_binary_metrics(y_true, y_hat)["recall"]),
                "f1": float(compute_binary_metrics(y_true, y_hat)["f1"]),
                "balanced_accuracy": float(compute_binary_metrics(y_true, y_hat)["balanced_accuracy"]) if len(np.unique(y_true)) > 1 else np.nan,
                "detected_at_least_once": int(int(y_hat.sum()) > 0),
            }
        )
    return rows


def summarize_grouped_results(fold_df: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    row: Dict[str, object] = {
        "dataset": fold_df["dataset"].iloc[0],
        "model": fold_df["model"].iloc[0],
        "cv_plan": fold_df["cv_plan"].iloc[0],
    }
    for metric in ["precision", "recall", "f1", "balanced_accuracy", "roc_auc"]:
        mean_value, ci = mean_ci(fold_df[metric].tolist())
        row[f"window_{metric}_mean"] = mean_value
        row[f"window_{metric}_ci95"] = ci
    for metric in ["precision", "recall", "f1", "balanced_accuracy", "detected_at_least_once"]:
        mean_value, ci = mean_ci(run_df[metric].tolist())
        row[f"run_{metric}_mean"] = mean_value
        row[f"run_{metric}_ci95"] = ci
    return pd.DataFrame([row])


def main() -> None:
    args = parse_args()
    enriched_dir = Path(args.enriched_dir) if args.enriched_dir else latest_enriched_dir_or_fail()
    scalar_df = pd.read_csv(enriched_dir / "datasets" / f"{args.dataset_name}.csv")
    repetition_df = pd.read_csv(enriched_dir / "enriched_repetition_table.csv")[["raw_segment_id", "global_repetition_index_within_run"]]
    if "global_repetition_index_within_run" not in scalar_df.columns:
        scalar_df = scalar_df.merge(repetition_df, on="raw_segment_id", how="left")
    feature_columns = [
        col for col in scalar_df.columns
        if col not in {"run_id", "source_run_id", "source_file", "raw_segment_id", "label_damage_transition", "label_crack_level", "global_repetition_index_within_run"}
    ]
    X, y, groups, manifest = build_flattened_sequences(
        scalar_df,
        feature_columns,
        args.sequence_length,
        args.label_mode,
        args.representation,
    )
    cv_plan = build_group_cv(y, groups, max_splits=5)

    dataset_name = f"{args.dataset_name}_seq{args.sequence_length}_{args.label_mode}_{args.representation}_{args.threshold_target}"
    model_name = args.model
    fold_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []
    window_rows: List[Dict[str, object]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_plan.splits, start=1):
        inner_train_idx, val_rel = choose_inner_group_split(y[train_idx], groups[train_idx])
        actual_train_idx = train_idx[inner_train_idx]
        actual_val_idx = train_idx[val_rel]
        X_train, X_val, X_test = drop_train_empty_columns(
            X[actual_train_idx],
            X[actual_val_idx],
            X[test_idx],
        )
        model = make_model(args.model)
        model.fit(X_train, y[actual_train_idx])
        val_scores = model.predict_proba(X_val)[:, 1]
        threshold = tune_threshold(
            y[actual_val_idx],
            val_scores,
            groups[actual_val_idx],
            args.threshold_target,
        )
        test_scores = model.predict_proba(X_test)[:, 1]
        test_pred = (test_scores >= threshold).astype(int)
        metrics = compute_binary_metrics(y[test_idx], test_pred, test_scores)
        fold_rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "fold": fold_idx,
                "cv_plan": cv_plan.name,
                "decision_threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "roc_auc": metrics["roc_auc"],
            }
        )
        fold_manifest = manifest.iloc[test_idx].copy()
        fold_manifest["dataset"] = dataset_name
        fold_manifest["model"] = model_name
        fold_manifest["fold"] = fold_idx
        fold_manifest["y_pred"] = test_pred
        fold_manifest["y_score"] = test_scores
        window_rows.extend(fold_manifest.to_dict("records"))
        run_rows.extend(summarize_runs(dataset_name, model_name, fold_idx, fold_manifest, test_pred))

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset_name}_seq{args.sequence_length}_{args.label_mode}_{args.representation}_{args.threshold_target}_{args.model}"
    fold_df = pd.DataFrame(fold_rows)
    run_df = pd.DataFrame(run_rows)
    window_df = pd.DataFrame(window_rows)
    summary_df = summarize_grouped_results(fold_df, run_df)
    fold_df.to_csv(results_dir / f"{tag}_fold_results.csv", index=False)
    run_df.to_csv(results_dir / f"{tag}_run_level_results.csv", index=False)
    window_df.to_csv(results_dir / f"{tag}_window_level_results.csv", index=False)
    summary_df.to_csv(results_dir / f"{tag}_summary.csv", index=False)
    (results_dir / f"{tag}_metadata.json").write_text(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "sequence_length": args.sequence_length,
                "label_mode": args.label_mode,
                "representation": args.representation,
                "threshold_target": args.threshold_target,
                "model": args.model,
                "feature_count": len(feature_columns),
                "model_input_dim": int(X.shape[1]) if X.size else 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Grouped sequence ensemble results written to: {results_dir}")


if __name__ == "__main__":
    main()
