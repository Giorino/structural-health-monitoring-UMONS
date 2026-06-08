#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    parser = argparse.ArgumentParser(description="Run grouped CNN plus mechanics experiments on dataset F.")
    parser.add_argument("--enriched-dir", type=str, default=None, help="Path to a specific enriched dataset directory.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience on validation loss.")
    parser.add_argument("--sequence-length", type=int, default=1, help="Number of consecutive repetitions per sample. Use 1 for single-window CNN.")
    parser.add_argument("--label-mode", choices=["last", "max"], default="max", help="How to assign sequence labels when sequence-length > 1.")
    parser.add_argument("--decision-metric", choices=["f1", "balanced_accuracy"], default="f1", help="Validation metric used for threshold tuning.")
    return parser.parse_args()


class TensorWithScalarDataset(Dataset):
    def __init__(self, x_raw: np.ndarray, x_scalar: np.ndarray, y: np.ndarray):
        self.x_raw = torch.as_tensor(x_raw, dtype=torch.float32)
        self.x_scalar = torch.as_tensor(x_scalar, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_raw[idx], self.x_scalar[idx], self.y[idx]


class SimplePhysicsCNN(nn.Module):
    def __init__(self, n_channels: int, n_scalar: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.scalar_head = nn.Sequential(
            nn.Linear(n_scalar, 16),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(48 + 16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x_raw: torch.Tensor, x_scalar: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x_raw).squeeze(-1)
        s = self.scalar_head(x_scalar)
        return self.classifier(torch.cat([x, s], dim=1)).squeeze(1)


@dataclass
class NormalizationState:
    raw_mean: np.ndarray
    raw_std: np.ndarray
    scalar_median: np.ndarray
    scalar_mean: np.ndarray
    scalar_std: np.ndarray


def load_dataset_f(enriched_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    dataset_path = enriched_dir / "datasets" / "dataset_F_raw_window_tensor_plus_mechanics.npz"
    data = np.load(dataset_path, allow_pickle=True)
    x_raw = np.asarray(data["X_raw"], dtype=float)
    x_scalar = np.asarray(data["X_scalar"], dtype=float)
    y = np.asarray(data["y"], dtype=int)
    groups = np.asarray(data["groups"], dtype=object)
    raw_segment_id = np.asarray(data["raw_segment_id"], dtype=object)
    scalar_feature_names = [str(x) for x in data["scalar_feature_names"].tolist()]
    return x_raw, x_scalar, y, groups, raw_segment_id, scalar_feature_names


def build_sequence_samples(
    manifest_df: pd.DataFrame,
    x_raw: np.ndarray,
    x_scalar: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    if args.sequence_length <= 1:
        return x_raw, x_scalar, y, manifest_df.reset_index(drop=True)

    seq_raw: List[np.ndarray] = []
    seq_scalar: List[np.ndarray] = []
    seq_y: List[int] = []
    seq_rows: List[Dict[str, object]] = []
    ordered = manifest_df.copy().reset_index(drop=True)
    ordered["row_idx"] = np.arange(len(ordered))

    group_col = "source_run_id" if "source_run_id" in ordered.columns else "run_id"
    for _, run_df in ordered.groupby(group_col, sort=False):
        run_df = run_df.sort_values("global_repetition_index_within_run").reset_index(drop=True)
        if len(run_df) < args.sequence_length:
            continue
        for start in range(0, len(run_df) - args.sequence_length + 1):
            chunk = run_df.iloc[start:start + args.sequence_length]
            row_indices = chunk["row_idx"].to_numpy(dtype=int)
            raw_concat = np.concatenate([x_raw[idx] for idx in row_indices], axis=1)
            scalar_window = x_scalar[row_indices]
            scalar_last = scalar_window[-1]
            scalar_mean = np.nanmean(scalar_window, axis=0)
            scalar_delta = scalar_window[-1] - scalar_window[0]
            seq_raw.append(raw_concat)
            seq_scalar.append(np.concatenate([scalar_last, scalar_mean, scalar_delta]))
            labels = y[row_indices]
            seq_y.append(int(labels.max()) if args.label_mode == "max" else int(labels[-1]))
            seq_rows.append(
                {
                    "run_id": chunk["run_id"].iloc[-1] if "run_id" in chunk.columns else chunk[group_col].iloc[-1],
                    "source_run_id": chunk[group_col].iloc[-1],
                    "source_file": chunk["source_file"].iloc[-1],
                    "raw_segment_id": chunk["raw_segment_id"].iloc[-1],
                    "global_repetition_index_within_run": int(chunk["global_repetition_index_within_run"].iloc[-1]),
                    "label_damage_transition": int(seq_y[-1]),
                    "sequence_start_index_within_run": int(chunk["global_repetition_index_within_run"].iloc[0]),
                    "sequence_end_index_within_run": int(chunk["global_repetition_index_within_run"].iloc[-1]),
                }
            )

    seq_manifest_df = pd.DataFrame(seq_rows)
    return np.stack(seq_raw), np.stack(seq_scalar), np.asarray(seq_y, dtype=int), seq_manifest_df


def compute_normalization(x_raw: np.ndarray, x_scalar: np.ndarray) -> NormalizationState:
    raw_mean = np.nanmean(x_raw, axis=(0, 2), keepdims=True)
    raw_std = np.nanstd(x_raw, axis=(0, 2), keepdims=True)
    raw_std = np.where(raw_std < 1e-6, 1.0, raw_std)
    scalar_median = np.nanmedian(x_scalar, axis=0)
    safe_scalar = np.where(np.isnan(x_scalar), scalar_median[None, :], x_scalar)
    scalar_mean = np.mean(safe_scalar, axis=0)
    scalar_std = np.std(safe_scalar, axis=0)
    scalar_std = np.where(scalar_std < 1e-6, 1.0, scalar_std)
    return NormalizationState(
        raw_mean=raw_mean,
        raw_std=raw_std,
        scalar_median=scalar_median,
        scalar_mean=scalar_mean,
        scalar_std=scalar_std,
    )


def apply_normalization(x_raw: np.ndarray, x_scalar: np.ndarray, state: NormalizationState) -> Tuple[np.ndarray, np.ndarray]:
    filled_raw = np.where(np.isnan(x_raw), state.raw_mean, x_raw)
    norm_raw = (filled_raw - state.raw_mean) / state.raw_std
    filled_scalar = np.where(np.isnan(x_scalar), state.scalar_median[None, :], x_scalar)
    norm_scalar = (filled_scalar - state.scalar_mean[None, :]) / state.scalar_std[None, :]
    return norm_raw.astype(np.float32), norm_scalar.astype(np.float32)


def make_loader(x_raw: np.ndarray, x_scalar: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorWithScalarDataset(x_raw, x_scalar, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, threshold: float = 0.5) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    logits_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    with torch.no_grad():
        for x_raw, x_scalar, y in loader:
            x_raw = x_raw.to(device)
            x_scalar = x_scalar.to(device)
            y = y.to(device)
            logits = model(x_raw, x_scalar)
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
    candidate_grid = np.unique(np.concatenate([np.linspace(0.1, 0.9, 17), scores]))
    best_threshold = 0.5
    best_metric = -1.0
    for threshold in candidate_grid:
        preds = (scores >= threshold).astype(int)
        metrics = compute_binary_metrics(y_true, preds, scores)
        metric_value = float(metrics[metric_name])
        if metric_value > best_metric:
            best_metric = metric_value
            best_threshold = float(threshold)
    return best_threshold


def train_one_fold(
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: np.ndarray,
    n_channels: int,
    n_scalar: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, float, List[Dict[str, float]]]:
    model = SimplePhysicsCNN(n_channels=n_channels, n_scalar=n_scalar).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    negatives = max(int((y_train == 0).sum()), 1)
    positives = max(int((y_train == 1).sum()), 1)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history: List[Dict[str, float]] = []
    best_metric = -1.0
    best_state: Dict[str, torch.Tensor] | None = None
    best_threshold = 0.5
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x_raw, x_scalar, y_batch in train_loader:
            x_raw = x_raw.to(device)
            x_scalar = x_scalar.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_raw, x_scalar)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(y_batch)

        train_eval = evaluate_loader(model, train_loader, device, criterion, threshold=0.5)
        val_eval_raw = evaluate_loader(model, val_loader, device, criterion, threshold=0.5)
        tuned_threshold = choose_threshold(val_eval_raw["y_true"], val_eval_raw["y_score"], args.decision_metric)
        val_eval = evaluate_loader(model, val_loader, device, criterion, threshold=tuned_threshold)
        train_eval_tuned = evaluate_loader(model, train_loader, device, criterion, threshold=tuned_threshold)
        train_eval_tuned["loss"] = train_loss / max(len(train_loader.dataset), 1)

        history.append(
            {
                "epoch": float(epoch),
                "threshold": float(tuned_threshold),
                "train_loss": float(train_eval_tuned["loss"]),
                "val_loss": float(val_eval["loss"]),
                "train_f1": float(train_eval_tuned["f1"]),
                "val_f1": float(val_eval["f1"]),
                "train_recall": float(train_eval_tuned["recall"]),
                "val_recall": float(val_eval["recall"]),
                "train_balanced_accuracy": float(train_eval_tuned["balanced_accuracy"]),
                "val_balanced_accuracy": float(val_eval["balanced_accuracy"]),
                "train_roc_auc": float(train_eval_tuned["roc_auc"]) if not pd.isna(train_eval_tuned["roc_auc"]) else np.nan,
                "val_roc_auc": float(val_eval["roc_auc"]) if not pd.isna(val_eval["roc_auc"]) else np.nan,
            }
        )

        current_metric = float(val_eval[args.decision_metric])
        if current_metric > best_metric:
            best_metric = current_metric
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


def plot_history(history: Sequence[Dict[str, float]], out_dir: Path, fold_label: str) -> None:
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / f"{fold_label}_history.csv", index=False)

    plots = [
        ("loss", "Loss"),
        ("f1", "F1"),
        ("recall", "Recall"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("roc_auc", "ROC-AUC"),
    ]
    for metric_key, label in plots:
        plt.figure(figsize=(8, 5))
        plt.plot(hist_df["epoch"], hist_df[f"train_{metric_key}"], label="train")
        plt.plot(hist_df["epoch"], hist_df[f"val_{metric_key}"], label="validation")
        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.title(f"{fold_label} {label}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{fold_label}_{metric_key}.png", dpi=200)
        plt.close()


def summarize_runs(model_name: str, dataset_name: str, fold_idx: int, test_df: pd.DataFrame, y_pred: np.ndarray) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    enriched = test_df.copy()
    enriched["y_pred"] = y_pred
    group_col = "source_run_id" if "source_run_id" in enriched.columns else "run_id"
    for run_id, run_df in enriched.groupby(group_col, sort=False):
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


def summarize_grouped_results(fold_df: pd.DataFrame, run_df: pd.DataFrame, dataset_name: str, model_name: str) -> pd.DataFrame:
    row: Dict[str, object] = {
        "dataset": dataset_name,
        "model": model_name,
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
    repetition_df = pd.read_csv(enriched_dir / "enriched_repetition_table.csv")
    base_cols = ["run_id", "source_file", "raw_segment_id", "global_repetition_index_within_run", "label_damage_transition"]
    if "source_run_id" in repetition_df.columns:
        base_cols.insert(1, "source_run_id")
    repetition_df = repetition_df[base_cols].copy()

    x_raw, x_scalar, y, groups, raw_segment_id, scalar_feature_names = load_dataset_f(enriched_dir)
    manifest_df = pd.DataFrame(
        {
            "raw_segment_id": raw_segment_id,
            "label_damage_transition": y,
            "source_run_id": groups,
        }
    )
    repetition_key_cols = ["source_run_id", "source_file", "raw_segment_id", "global_repetition_index_within_run"]
    if "run_id" in repetition_df.columns:
        repetition_key_cols.insert(0, "run_id")
    manifest_df = manifest_df.merge(repetition_df[repetition_key_cols], on=["source_run_id", "raw_segment_id"], how="left")

    x_raw, x_scalar, y, manifest_df = build_sequence_samples(manifest_df, x_raw, x_scalar, y, groups, args)
    groups = manifest_df["source_run_id"].to_numpy(dtype=object)
    dataset_name = "dataset_F_raw_window_tensor_plus_mechanics" if args.sequence_length == 1 else f"dataset_F_seq{args.sequence_length}_{args.label_mode}"
    model_name = "simple_physics_cnn"

    if args.sequence_length > 1:
        expanded_names: List[str] = []
        for prefix in ["last", "mean", "delta"]:
            expanded_names.extend([f"{prefix}_{name}" for name in scalar_feature_names])
        scalar_feature_names = expanded_names

    cv_plan = build_group_cv(y, groups, max_splits=5)
    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures_revision" / "learning_curves" / ("grouped_cnn" if args.sequence_length == 1 else f"grouped_cnn_seq{args.sequence_length}_{args.label_mode}")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_rows: List[Dict[str, object]] = []
    window_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_plan.splits, start=1):
        train_groups = groups[train_idx]
        inner_train_idx, val_idx_rel = choose_inner_group_split(y[train_idx], train_groups)
        actual_train_idx = train_idx[inner_train_idx]
        actual_val_idx = train_idx[val_idx_rel]

        norm_state = compute_normalization(x_raw[actual_train_idx], x_scalar[actual_train_idx])
        x_train_raw, x_train_scalar = apply_normalization(x_raw[actual_train_idx], x_scalar[actual_train_idx], norm_state)
        x_val_raw, x_val_scalar = apply_normalization(x_raw[actual_val_idx], x_scalar[actual_val_idx], norm_state)
        x_test_raw, x_test_scalar = apply_normalization(x_raw[test_idx], x_scalar[test_idx], norm_state)

        train_loader = make_loader(x_train_raw, x_train_scalar, y[actual_train_idx], args.batch_size, shuffle=True)
        val_loader = make_loader(x_val_raw, x_val_scalar, y[actual_val_idx], args.batch_size, shuffle=False)
        test_loader = make_loader(x_test_raw, x_test_scalar, y[test_idx], args.batch_size, shuffle=False)

        model, best_threshold, history = train_one_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            y_train=y[actual_train_idx],
            n_channels=x_raw.shape[1],
            n_scalar=x_scalar.shape[1],
            args=args,
            device=device,
        )
        plot_history(history, figures_dir, f"fold_{fold_idx}")

        negatives = max(int((y[actual_train_idx] == 0).sum()), 1)
        positives = max(int((y[actual_train_idx] == 1).sum()), 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device))
        test_metrics = evaluate_loader(model, test_loader, device, criterion, threshold=best_threshold)
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
                "n_scalar_features": len(scalar_feature_names),
                "sequence_length": args.sequence_length,
                "decision_threshold": best_threshold,
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
        run_rows.extend(summarize_runs(model_name, dataset_name, fold_idx, fold_window, test_metrics["y_pred"]))

    fold_df = pd.DataFrame(fold_rows)
    window_df = pd.DataFrame(window_rows)
    run_df = pd.DataFrame(run_rows)
    summary_df = summarize_grouped_results(fold_df, run_df, dataset_name, model_name)

    tag = "grouped_cnn" if args.sequence_length == 1 else f"grouped_cnn_seq{args.sequence_length}_{args.label_mode}"
    fold_df.to_csv(results_dir / f"{tag}_fold_results.csv", index=False)
    summary_df.to_csv(results_dir / f"{tag}_summary.csv", index=False)
    run_df.to_csv(results_dir / f"{tag}_run_level_results.csv", index=False)
    window_df.to_csv(results_dir / f"{tag}_window_level_results.csv", index=False)

    metadata = {
        "dataset": dataset_name,
        "model": model_name,
        "cv_plan": cv_plan.name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "sequence_length": args.sequence_length,
        "label_mode": args.label_mode,
        "decision_metric": args.decision_metric,
        "scalar_feature_names": scalar_feature_names,
        "learning_curve_dir": str(figures_dir),
    }
    (results_dir / f"{tag}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Grouped CNN results written to: {results_dir}")


if __name__ == "__main__":
    main()
