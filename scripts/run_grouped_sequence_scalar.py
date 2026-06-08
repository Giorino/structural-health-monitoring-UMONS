#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pipeline_common import (
    ROOT,
    build_group_cv,
    choose_inner_group_split,
    compute_binary_metrics,
    latest_enriched_dir_or_fail,
    mean_ci,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grouped sequence models on enriched scalar datasets.")
    parser.add_argument("--enriched-dir", type=str, default=None, help="Path to a specific enriched dataset directory.")
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
        help="Scalar dataset variant to use as the base trajectory.",
    )
    parser.add_argument("--sequence-length", type=int, default=5, help="Number of consecutive repetitions per sample.")
    parser.add_argument("--label-mode", choices=["last", "max"], default="max", help="How to assign the sequence label.")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--decision-metric", choices=["f1", "balanced_accuracy"], default="f1", help="Validation metric used for threshold tuning.")
    return parser.parse_args()


class SequenceScalarDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class ScalarSequenceModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.temporal = nn.GRU(
            input_size=n_features,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.temporal(x)
        pooled = out[:, -1, :]
        return self.classifier(pooled).squeeze(1)


def load_scalar_dataset(enriched_dir: Path, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sequence_df = pd.read_csv(enriched_dir / "datasets" / f"{dataset_name}.csv")
    repetition_df = pd.read_csv(enriched_dir / "enriched_repetition_table.csv")
    return sequence_df, repetition_df


def build_sequence_samples(
    scalar_df: pd.DataFrame,
    repetition_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    feature_columns = [
        col for col in scalar_df.columns
        if col not in {"run_id", "source_run_id", "source_file", "raw_segment_id", "label_damage_transition", "label_crack_level"}
    ]
    merged = scalar_df.merge(
        repetition_df[["raw_segment_id", "global_repetition_index_within_run"]],
        on="raw_segment_id",
        how="left",
    )
    group_col = "source_run_id" if "source_run_id" in merged.columns else "run_id"

    xs: List[np.ndarray] = []
    ys: List[int] = []
    rows: List[Dict[str, object]] = []
    for _, run_df in merged.groupby(group_col, sort=False):
        run_df = run_df.sort_values("global_repetition_index_within_run").reset_index(drop=True)
        if len(run_df) < args.sequence_length:
            continue
        run_features = run_df[feature_columns].to_numpy(dtype=float)
        run_labels = run_df["label_damage_transition"].to_numpy(dtype=int)
        for start in range(0, len(run_df) - args.sequence_length + 1):
            end = start + args.sequence_length
            chunk = run_df.iloc[start:end]
            chunk_features = run_features[start:end]
            xs.append(chunk_features)
            ys.append(int(run_labels[start:end].max()) if args.label_mode == "max" else int(run_labels[end - 1]))
            rows.append(
                {
                    "run_id": chunk["run_id"].iloc[-1],
                    "source_run_id": chunk[group_col].iloc[-1],
                    "source_file": chunk["source_file"].iloc[-1],
                    "raw_segment_id": chunk["raw_segment_id"].iloc[-1],
                    "global_repetition_index_within_run": int(chunk["global_repetition_index_within_run"].iloc[-1]),
                    "label_damage_transition": int(ys[-1]),
                    "sequence_start_index_within_run": int(chunk["global_repetition_index_within_run"].iloc[0]),
                    "sequence_end_index_within_run": int(chunk["global_repetition_index_within_run"].iloc[-1]),
                }
            )
    return np.stack(xs), np.asarray(ys, dtype=int), np.asarray([row["source_run_id"] for row in rows], dtype=object), pd.DataFrame(rows), feature_columns


def compute_normalization(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x_train, axis=(0, 1), keepdims=True)
    std = np.nanstd(x_train, axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_normalization(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    fill_value = np.where(np.isnan(mean), 0.0, mean)
    x_filled = np.where(np.isnan(x), fill_value, x)
    return ((x_filled - fill_value) / std).astype(np.float32)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(SequenceScalarDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, threshold: float = 0.5) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    logits_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * len(y)
            logits_list.append(logits.cpu().numpy())
            y_list.append(y.cpu().numpy())
    logits_all = np.concatenate(logits_list)
    y_all = np.concatenate(y_list).astype(int)
    scores = 1.0 / (1.0 + np.exp(-logits_all))
    preds = (scores >= threshold).astype(int)
    metrics = compute_binary_metrics(y_all, preds, scores)
    metrics["loss"] = total_loss / max(len(y_all), 1)
    metrics["y_true"] = y_all
    metrics["y_pred"] = preds
    metrics["y_score"] = scores
    metrics["threshold"] = threshold
    return metrics


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, metric_name: str) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.1, 0.9, 17), scores]))
    best_threshold = 0.5
    best_value = -1.0
    for threshold in candidates:
        preds = (scores >= threshold).astype(int)
        value = float(compute_binary_metrics(y_true, preds, scores)[metric_name])
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold


def train_one_fold(
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: np.ndarray,
    n_features: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, float, List[Dict[str, float]]]:
    model = ScalarSequenceModel(n_features=n_features).to(device)
    negatives = max(int((y_train == 0).sum()), 1)
    positives = max(int((y_train == 1).sum()), 1)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_metric = -1.0
    best_threshold = 0.5
    best_state: Dict[str, torch.Tensor] | None = None
    patience_left = args.patience
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(y_batch)

        val_eval_raw = evaluate_loader(model, val_loader, device, criterion, threshold=0.5)
        tuned_threshold = choose_threshold(val_eval_raw["y_true"], val_eval_raw["y_score"], args.decision_metric)
        val_eval = evaluate_loader(model, val_loader, device, criterion, threshold=tuned_threshold)
        train_eval = evaluate_loader(model, train_loader, device, criterion, threshold=tuned_threshold)
        train_eval["loss"] = train_loss / max(len(train_loader.dataset), 1)

        history.append(
            {
                "epoch": float(epoch),
                "threshold": float(tuned_threshold),
                "train_loss": float(train_eval["loss"]),
                "val_loss": float(val_eval["loss"]),
                "train_f1": float(train_eval["f1"]),
                "val_f1": float(val_eval["f1"]),
                "train_recall": float(train_eval["recall"]),
                "val_recall": float(val_eval["recall"]),
                "train_balanced_accuracy": float(train_eval["balanced_accuracy"]),
                "val_balanced_accuracy": float(val_eval["balanced_accuracy"]),
                "train_roc_auc": float(train_eval["roc_auc"]) if not pd.isna(train_eval["roc_auc"]) else np.nan,
                "val_roc_auc": float(val_eval["roc_auc"]) if not pd.isna(val_eval["roc_auc"]) else np.nan,
            }
        )

        current_value = float(val_eval[args.decision_metric])
        if current_value > best_metric:
            best_metric = current_value
            best_threshold = float(tuned_threshold)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_threshold, history


def summarize_runs(dataset_name: str, model_name: str, fold_idx: int, test_df: pd.DataFrame, y_pred: np.ndarray) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    enriched = test_df.copy()
    enriched["y_pred"] = y_pred
    for run_id, run_df in enriched.groupby("source_run_id", sort=False):
        y_true = run_df["label_damage_transition"].to_numpy(dtype=int)
        y_hat = run_df["y_pred"].to_numpy(dtype=int)
        first_true = run_df.loc[run_df["label_damage_transition"] == 1, "global_repetition_index_within_run"]
        first_pred = run_df.loc[run_df["y_pred"] == 1, "global_repetition_index_within_run"]
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "fold": fold_idx,
                "run_id": run_id,
                "source_file": run_df["source_file"].iloc[0],
                "true_positive_window_count": int(y_true.sum()),
                "predicted_positive_window_count": int(y_hat.sum()),
                "true_positives": int(((y_true == 1) & (y_hat == 1)).sum()),
                "false_positives": int(((y_true == 0) & (y_hat == 1)).sum()),
                "false_negatives": int(((y_true == 1) & (y_hat == 0)).sum()),
                "precision": float(precision_score(y_true, y_hat, zero_division=0)),
                "recall": float(recall_score(y_true, y_hat, zero_division=0)),
                "f1": float(compute_binary_metrics(y_true, y_hat)["f1"]),
                "balanced_accuracy": float(compute_binary_metrics(y_true, y_hat)["balanced_accuracy"]) if len(np.unique(y_true)) > 1 else np.nan,
                "first_true_damage_index": int(first_true.iloc[0]) if not first_true.empty else np.nan,
                "first_predicted_damage_index": int(first_pred.iloc[0]) if not first_pred.empty else np.nan,
                "detection_delay": (int(first_pred.iloc[0]) - int(first_true.iloc[0])) if (not first_true.empty and not first_pred.empty) else np.nan,
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
    scalar_df, repetition_df = load_scalar_dataset(enriched_dir, args.dataset_name)
    x, y, groups, manifest_df, feature_columns = build_sequence_samples(scalar_df, repetition_df, args)

    dataset_name = f"{args.dataset_name}_seq{args.sequence_length}_{args.label_mode}"
    model_name = "gru_sequence_scalar"
    cv_plan = build_group_cv(y, groups, max_splits=5)
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_rows: List[Dict[str, object]] = []
    window_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_plan.splits, start=1):
        inner_train_idx, val_idx_rel = choose_inner_group_split(y[train_idx], groups[train_idx])
        actual_train_idx = train_idx[inner_train_idx]
        actual_val_idx = train_idx[val_idx_rel]

        mean, std = compute_normalization(x[actual_train_idx])
        x_train = apply_normalization(x[actual_train_idx], mean, std)
        x_val = apply_normalization(x[actual_val_idx], mean, std)
        x_test = apply_normalization(x[test_idx], mean, std)

        train_loader = make_loader(x_train, y[actual_train_idx], args.batch_size, shuffle=True)
        val_loader = make_loader(x_val, y[actual_val_idx], args.batch_size, shuffle=False)
        test_loader = make_loader(x_test, y[test_idx], args.batch_size, shuffle=False)

        model, threshold, _history = train_one_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            y_train=y[actual_train_idx],
            n_features=x.shape[2],
            args=args,
            device=device,
        )

        negatives = max(int((y[actual_train_idx] == 0).sum()), 1)
        positives = max(int((y[actual_train_idx] == 1).sum()), 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device))
        test_metrics = evaluate_loader(model, test_loader, device, criterion, threshold=threshold)

        fold_rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "fold": fold_idx,
                "cv_plan": cv_plan.name,
                "n_train_rows": len(actual_train_idx),
                "n_val_rows": len(actual_val_idx),
                "n_test_rows": len(test_idx),
                "n_train_runs": len(np.unique(groups[actual_train_idx])),
                "n_val_runs": len(np.unique(groups[actual_val_idx])),
                "n_test_runs": len(np.unique(groups[test_idx])),
                "n_features": len(feature_columns),
                "sequence_length": args.sequence_length,
                "decision_threshold": threshold,
                **{key: test_metrics[key] for key in ["precision", "recall", "f1", "balanced_accuracy", "roc_auc"]},
            }
        )

        fold_window = manifest_df.iloc[test_idx].copy()
        fold_window["dataset"] = dataset_name
        fold_window["model"] = model_name
        fold_window["fold"] = fold_idx
        fold_window["y_pred"] = test_metrics["y_pred"]
        fold_window["y_score"] = test_metrics["y_score"]
        window_rows.extend(fold_window.to_dict("records"))
        run_rows.extend(summarize_runs(dataset_name, model_name, fold_idx, fold_window, test_metrics["y_pred"]))

    fold_df = pd.DataFrame(fold_rows)
    window_df = pd.DataFrame(window_rows)
    run_df = pd.DataFrame(run_rows)
    summary_df = summarize_grouped_results(fold_df, run_df)

    tag = f"{args.dataset_name}_seq{args.sequence_length}_{args.label_mode}"
    fold_df.to_csv(results_dir / f"{tag}_fold_results.csv", index=False)
    window_df.to_csv(results_dir / f"{tag}_window_level_results.csv", index=False)
    run_df.to_csv(results_dir / f"{tag}_run_level_results.csv", index=False)
    summary_df.to_csv(results_dir / f"{tag}_summary.csv", index=False)
    metadata = {
        "dataset": dataset_name,
        "model": model_name,
        "feature_columns": feature_columns,
        "sequence_length": args.sequence_length,
        "label_mode": args.label_mode,
        "decision_metric": args.decision_metric,
    }
    (results_dir / f"{tag}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Grouped scalar-sequence results written to: {results_dir}")


if __name__ == "__main__":
    main()
