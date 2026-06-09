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
    parser = argparse.ArgumentParser(description="Run grouped CNN with loading traces and physics regularization on dataset G.")
    parser.add_argument("--enriched-dir", type=str, default=None, help="Path to a specific enriched dataset directory.")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on validation metric.")
    parser.add_argument("--max-splits", type=int, default=5, help="Maximum number of grouped CV splits.")
    parser.add_argument("--decision-metric", choices=["f1", "balanced_accuracy", "recall"], default="f1", help="Validation metric used for threshold tuning and model selection.")
    parser.add_argument("--physics-loss-weight", type=float, default=0.2, help="Weight applied to the auxiliary physics consistency loss.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout used in the classifier head.")
    return parser.parse_args()


class TraceDataset(Dataset):
    def __init__(self, x_raw: np.ndarray, x_loading: np.ndarray, y: np.ndarray, target_physics: np.ndarray):
        self.x_raw = torch.as_tensor(x_raw, dtype=torch.float32)
        self.x_loading = torch.as_tensor(x_loading, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.target_physics = torch.as_tensor(target_physics, dtype=torch.float32)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_raw[idx], self.x_loading[idx], self.y[idx], self.target_physics[idx]


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, widths: Sequence[int]):
        super().__init__()
        layers: List[nn.Module] = []
        current = in_channels
        kernels = [5, 5, 3]
        for width, kernel in zip(widths, kernels):
            layers.extend(
                [
                    nn.Conv1d(current, width, kernel_size=kernel, padding=kernel // 2),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                ]
            )
            current = width
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LoadingTracePhysicsCNN(nn.Module):
    def __init__(self, raw_channels: int, loading_channels: int, dropout: float):
        super().__init__()
        self.raw_encoder = ConvEncoder(raw_channels, [32, 64, 64])
        self.loading_encoder = ConvEncoder(loading_channels, [16, 32, 32])
        self.physics_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_raw: torch.Tensor, x_loading: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_feat = self.raw_encoder(x_raw)
        loading_feat = self.loading_encoder(x_loading)
        logits = self.classifier(torch.cat([raw_feat, loading_feat], dim=1)).squeeze(1)
        predicted_physics = self.physics_head(raw_feat).squeeze(1)
        return logits, predicted_physics


@dataclass
class NormalizationState:
    raw_mean: np.ndarray
    raw_std: np.ndarray
    loading_mean: np.ndarray
    loading_std: np.ndarray


def load_dataset_g(enriched_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    dataset_path = enriched_dir / "datasets" / "dataset_G_raw_window_tensor_plus_loading_traces.npz"
    data = np.load(dataset_path, allow_pickle=True)
    x_raw = np.asarray(data["X_raw"], dtype=float)
    x_loading = np.asarray(data["X_loading"], dtype=float)
    y = np.asarray(data["y"], dtype=int)
    groups = np.asarray(data["groups"], dtype=object)
    raw_segment_id = np.asarray(data["raw_segment_id"], dtype=object)
    loading_trace_names = [str(x) for x in data["loading_trace_names"].tolist()]
    return x_raw, x_loading, y, groups, raw_segment_id, loading_trace_names


def compute_normalization(x_raw: np.ndarray, x_loading: np.ndarray) -> NormalizationState:
    raw_mean = np.nanmean(x_raw, axis=(0, 2), keepdims=True)
    raw_std = np.nanstd(x_raw, axis=(0, 2), keepdims=True)
    raw_std = np.where(raw_std < 1e-6, 1.0, raw_std)
    loading_mean = np.nanmean(x_loading, axis=(0, 2), keepdims=True)
    loading_std = np.nanstd(x_loading, axis=(0, 2), keepdims=True)
    loading_std = np.where(loading_std < 1e-6, 1.0, loading_std)
    return NormalizationState(
        raw_mean=raw_mean,
        raw_std=raw_std,
        loading_mean=loading_mean,
        loading_std=loading_std,
    )


def apply_normalization(x_raw: np.ndarray, x_loading: np.ndarray, state: NormalizationState) -> Tuple[np.ndarray, np.ndarray]:
    safe_raw = np.where(np.isnan(x_raw), state.raw_mean, x_raw)
    safe_loading = np.where(np.isnan(x_loading), state.loading_mean, x_loading)
    norm_raw = (safe_raw - state.raw_mean) / state.raw_std
    norm_loading = (safe_loading - state.loading_mean) / state.loading_std
    return norm_raw.astype(np.float32), norm_loading.astype(np.float32)


def make_loader(
    x_raw: np.ndarray,
    x_loading: np.ndarray,
    y: np.ndarray,
    target_physics: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(TraceDataset(x_raw, x_loading, y, target_physics), batch_size=batch_size, shuffle=shuffle)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    physics_criterion: nn.Module,
    physics_loss_weight: float,
    threshold: float = 0.5,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    logits_list: List[np.ndarray] = []
    physics_pred_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    physics_target_list: List[np.ndarray] = []
    with torch.no_grad():
        for x_raw, x_loading, y, physics_target in loader:
            x_raw = x_raw.to(device)
            x_loading = x_loading.to(device)
            y = y.to(device)
            physics_target = physics_target.to(device)
            logits, physics_pred = model(x_raw, x_loading)
            cls_loss = criterion(logits, y)
            aux_loss = physics_criterion(physics_pred, physics_target)
            loss = cls_loss + physics_loss_weight * aux_loss
            total_loss += float(loss.item()) * len(y)
            logits_list.append(logits.cpu().numpy())
            physics_pred_list.append(physics_pred.cpu().numpy())
            y_list.append(y.cpu().numpy())
            physics_target_list.append(physics_target.cpu().numpy())
    logits_all = np.concatenate(logits_list)
    y_all = np.concatenate(y_list).astype(int)
    physics_pred_all = np.concatenate(physics_pred_list)
    physics_target_all = np.concatenate(physics_target_list)
    scores = 1.0 / (1.0 + np.exp(-logits_all))
    preds = (scores >= threshold).astype(int)
    metrics = compute_binary_metrics(y_all, preds, scores)
    metrics["loss"] = total_loss / max(len(y_all), 1)
    metrics["physics_mae"] = float(np.mean(np.abs(physics_pred_all - physics_target_all)))
    metrics["y_true"] = y_all
    metrics["y_pred"] = preds
    metrics["y_score"] = scores
    metrics["physics_pred"] = physics_pred_all
    metrics["physics_target"] = physics_target_all
    metrics["threshold"] = threshold
    return metrics


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, metric_name: str) -> float:
    candidate_grid = np.unique(np.concatenate([np.linspace(0.05, 0.95, 37), scores]))
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
    raw_channels: int,
    loading_channels: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, float, List[Dict[str, float]]]:
    model = LoadingTracePhysicsCNN(raw_channels=raw_channels, loading_channels=loading_channels, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    negatives = max(int((y_train == 0).sum()), 1)
    positives = max(int((y_train == 1).sum()), 1)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    physics_criterion = nn.SmoothL1Loss()

    history: List[Dict[str, float]] = []
    best_metric = -1.0
    best_state: Dict[str, torch.Tensor] | None = None
    best_threshold = 0.5
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x_raw, x_loading, y_batch, physics_target in train_loader:
            x_raw = x_raw.to(device)
            x_loading = x_loading.to(device)
            y_batch = y_batch.to(device)
            physics_target = physics_target.to(device)
            optimizer.zero_grad()
            logits, physics_pred = model(x_raw, x_loading)
            cls_loss = criterion(logits, y_batch)
            aux_loss = physics_criterion(physics_pred, physics_target)
            loss = cls_loss + args.physics_loss_weight * aux_loss
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(y_batch)

        train_eval = evaluate_loader(model, train_loader, device, criterion, physics_criterion, args.physics_loss_weight, threshold=0.5)
        val_eval_raw = evaluate_loader(model, val_loader, device, criterion, physics_criterion, args.physics_loss_weight, threshold=0.5)
        tuned_threshold = choose_threshold(val_eval_raw["y_true"], val_eval_raw["y_score"], args.decision_metric)
        val_eval = evaluate_loader(model, val_loader, device, criterion, physics_criterion, args.physics_loss_weight, threshold=tuned_threshold)
        train_eval_tuned = evaluate_loader(model, train_loader, device, criterion, physics_criterion, args.physics_loss_weight, threshold=tuned_threshold)
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
                "train_physics_mae": float(train_eval_tuned["physics_mae"]),
                "val_physics_mae": float(val_eval["physics_mae"]),
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
        ("physics_mae", "Physics MAE"),
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
    for metric in ["precision", "recall", "f1", "balanced_accuracy", "roc_auc", "physics_mae"]:
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
    repetition_cols = ["run_id", "source_run_id", "source_file", "raw_segment_id", "global_repetition_index_within_run", "label_damage_transition"]
    repetition_df = repetition_df[repetition_cols].copy()

    x_raw, x_loading, y, groups, raw_segment_id, loading_trace_names = load_dataset_g(enriched_dir)
    manifest_df = pd.DataFrame(
        {
            "raw_segment_id": raw_segment_id,
            "label_damage_transition": y,
            "source_run_id": groups,
        }
    )
    manifest_df = manifest_df.merge(
        repetition_df[["run_id", "source_run_id", "source_file", "raw_segment_id", "global_repetition_index_within_run"]],
        on=["source_run_id", "raw_segment_id"],
        how="left",
    )

    dataset_name = "dataset_G_raw_window_tensor_plus_loading_traces"
    model_name = "loading_trace_physics_cnn"
    cv_plan = build_group_cv(y, groups, max_splits=args.max_splits)
    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures_revision" / "learning_curves" / "grouped_loading_trace_cnn"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        expected_idx = loading_trace_names.index("expected_elastic_strain_microstrain")
    except ValueError as exc:
        raise RuntimeError("expected_elastic_strain_microstrain was not found in loading_trace_names.") from exc

    physics_target = np.nanmean(x_loading[:, expected_idx, :], axis=1).astype(np.float32)
    finite_physics = np.isfinite(physics_target)
    if finite_physics.any():
        physics_fill = float(np.mean(physics_target[finite_physics]))
    else:
        physics_fill = 0.0
    physics_target = np.where(np.isfinite(physics_target), physics_target, physics_fill).astype(np.float32)
    fold_rows: List[Dict[str, object]] = []
    window_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_plan.splits, start=1):
        train_groups = groups[train_idx]
        inner_train_idx, val_idx_rel = choose_inner_group_split(y[train_idx], train_groups)
        actual_train_idx = train_idx[inner_train_idx]
        actual_val_idx = train_idx[val_idx_rel]

        norm_state = compute_normalization(x_raw[actual_train_idx], x_loading[actual_train_idx])
        x_train_raw, x_train_loading = apply_normalization(x_raw[actual_train_idx], x_loading[actual_train_idx], norm_state)
        x_val_raw, x_val_loading = apply_normalization(x_raw[actual_val_idx], x_loading[actual_val_idx], norm_state)
        x_test_raw, x_test_loading = apply_normalization(x_raw[test_idx], x_loading[test_idx], norm_state)

        physics_train = physics_target[actual_train_idx]
        physics_val = physics_target[actual_val_idx]
        physics_test = physics_target[test_idx]
        physics_mean = float(np.mean(physics_train))
        physics_std = float(np.std(physics_train))
        if physics_std < 1e-6:
            physics_std = 1.0
        physics_train = ((physics_train - physics_mean) / physics_std).astype(np.float32)
        physics_val = ((physics_val - physics_mean) / physics_std).astype(np.float32)
        physics_test = ((physics_test - physics_mean) / physics_std).astype(np.float32)

        train_loader = make_loader(x_train_raw, x_train_loading, y[actual_train_idx], physics_train, args.batch_size, shuffle=True)
        val_loader = make_loader(x_val_raw, x_val_loading, y[actual_val_idx], physics_val, args.batch_size, shuffle=False)
        test_loader = make_loader(x_test_raw, x_test_loading, y[test_idx], physics_test, args.batch_size, shuffle=False)

        model, best_threshold, history = train_one_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            y_train=y[actual_train_idx],
            raw_channels=x_raw.shape[1],
            loading_channels=x_loading.shape[1],
            args=args,
            device=device,
        )
        plot_history(history, figures_dir, f"fold_{fold_idx}")

        negatives = max(int((y[actual_train_idx] == 0).sum()), 1)
        positives = max(int((y[actual_train_idx] == 1).sum()), 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device))
        physics_criterion = nn.SmoothL1Loss()
        test_metrics = evaluate_loader(
            model,
            test_loader,
            device,
            criterion,
            physics_criterion,
            args.physics_loss_weight,
            threshold=best_threshold,
        )

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
                "raw_channels": x_raw.shape[1],
                "loading_channels": x_loading.shape[1],
                "decision_threshold": best_threshold,
                **{key: test_metrics[key] for key in ["precision", "recall", "f1", "balanced_accuracy", "roc_auc", "physics_mae"]},
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

    tag = "grouped_loading_trace_cnn"
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
        "max_splits": args.max_splits,
        "decision_metric": args.decision_metric,
        "physics_loss_weight": args.physics_loss_weight,
        "dropout": args.dropout,
        "loading_trace_names": loading_trace_names,
        "learning_curve_dir": str(figures_dir),
    }
    (results_dir / f"{tag}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Grouped loading-trace CNN results written to: {results_dir}")


if __name__ == "__main__":
    main()
