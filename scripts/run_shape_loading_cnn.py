#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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
    parser = argparse.ArgumentParser(description="Run grouped CNN on simple shape-plus-loading dataset.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Path to a shape_loading_dataset_* directory.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--decision-metric", choices=["f1", "balanced_accuracy"], default="f1")
    return parser.parse_args()


class ShapeLoadingDataset(Dataset):
    def __init__(self, x_shape: np.ndarray, x_loading: np.ndarray, y: np.ndarray):
        self.x_shape = torch.as_tensor(x_shape, dtype=torch.float32)
        self.x_loading = torch.as_tensor(x_loading, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        return self.x_shape[idx], self.x_loading[idx], self.y[idx]


class ShapeLoadingCNN(nn.Module):
    def __init__(self, n_shape_channels: int, n_loading: int):
        super().__init__()
        self.shape_branch = nn.Sequential(
            nn.Conv1d(n_shape_channels, 24, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(24, 48, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.loading_branch = nn.Sequential(
            nn.Linear(n_loading, 16),
            nn.ReLU(),
        ) if n_loading > 0 else None
        head_in = 64 + (16 if n_loading > 0 else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x_shape: torch.Tensor, x_loading: torch.Tensor) -> torch.Tensor:
        shape_feat = self.shape_branch(x_shape).squeeze(-1)
        if self.loading_branch is not None:
            loading_feat = self.loading_branch(x_loading)
            feat = torch.cat([shape_feat, loading_feat], dim=1)
        else:
            feat = shape_feat
        return self.head(feat).squeeze(1)


def resolve_dataset_dir(arg_value: str | None) -> Path:
    if arg_value:
        return Path(arg_value)
    enriched_dir = latest_enriched_dir_or_fail()
    candidates = sorted([path for path in enriched_dir.iterdir() if path.is_dir() and path.name.startswith("shape_loading_dataset_")])
    if not candidates:
        raise RuntimeError("No shape_loading_dataset_* directory found.")
    return candidates[-1]


def load_dataset(dataset_dir: Path):
    payload = np.load(dataset_dir / "shape_loading_dataset.npz", allow_pickle=True)
    manifest = pd.read_csv(dataset_dir / "shape_loading_manifest.csv")
    x_shape = np.asarray(payload["X_shape_resampled"], dtype=float)
    x_loading = np.asarray(payload["X_loading"], dtype=float)
    y = np.asarray(payload["y"], dtype=int)
    groups = np.asarray(payload["groups"], dtype=object)
    loading_feature_names = [str(x) for x in payload["loading_feature_names"].tolist()]
    return x_shape, x_loading, y, groups, manifest, loading_feature_names


def transform_shape(x_shape: np.ndarray) -> np.ndarray:
    centered = x_shape - np.nanmean(x_shape, axis=2, keepdims=True)
    scale = np.nanstd(centered, axis=2, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    normalized = centered / scale
    derivative = np.diff(normalized, axis=2, prepend=normalized[:, :, :1])
    enriched = np.concatenate([normalized, derivative], axis=1)
    return np.nan_to_num(enriched, nan=0.0).astype(np.float32)


def normalize_loading(train_x: np.ndarray, x: np.ndarray) -> np.ndarray:
    if train_x.shape[1] == 0:
        return x.astype(np.float32)
    median = np.nanmedian(train_x, axis=0)
    train_filled = np.where(np.isnan(train_x), median[None, :], train_x)
    mean = np.mean(train_filled, axis=0)
    std = np.std(train_filled, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    filled = np.where(np.isnan(x), median[None, :], x)
    return ((filled - mean[None, :]) / std[None, :]).astype(np.float32)


def make_loader(x_shape: np.ndarray, x_loading: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ShapeLoadingDataset(x_shape, x_loading, y), batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, threshold: float = 0.5) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    logits_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    with torch.no_grad():
        for x_shape, x_loading, y in loader:
            x_shape = x_shape.to(device)
            x_loading = x_loading.to(device)
            y = y.to(device)
            logits = model(x_shape, x_loading)
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
    return metrics


def tune_threshold(y_true: np.ndarray, scores: np.ndarray, metric_name: str) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 19), scores]))
    best_threshold = 0.5
    best_metric = -1.0
    for threshold in candidates:
        preds = (scores >= threshold).astype(int)
        metric = float(compute_binary_metrics(y_true, preds, scores)[metric_name])
        if metric > best_metric:
            best_metric = metric
            best_threshold = float(threshold)
    return best_threshold


def train_fold(
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: np.ndarray,
    n_shape_channels: int,
    n_loading: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, float]:
    model = ShapeLoadingCNN(n_shape_channels=n_shape_channels, n_loading=n_loading).to(device)
    negatives = max(int((y_train == 0).sum()), 1)
    positives = max(int((y_train == 1).sum()), 1)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_metric = -1.0
    best_threshold = 0.5
    best_state = None
    patience_left = args.patience

    for _epoch in range(args.epochs):
        model.train()
        for x_shape, x_loading, y in train_loader:
            x_shape = x_shape.to(device)
            x_loading = x_loading.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x_shape, x_loading)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_raw = evaluate(model, val_loader, device, criterion, threshold=0.5)
        threshold = tune_threshold(val_raw["y_true"], val_raw["y_score"], args.decision_metric)
        val_eval = evaluate(model, val_loader, device, criterion, threshold=threshold)
        metric_value = float(val_eval[args.decision_metric])
        if metric_value > best_metric:
            best_metric = metric_value
            best_threshold = threshold
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_threshold


def summarize_runs(dataset_name: str, model_name: str, fold_idx: int, test_df: pd.DataFrame, y_pred: np.ndarray) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    enriched = test_df.copy()
    enriched["y_pred"] = y_pred
    for run_id, run_df in enriched.groupby("group_id", sort=False):
        y_true = run_df["label_damage_transition"].to_numpy(dtype=int)
        y_hat = run_df["y_pred"].to_numpy(dtype=int)
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "fold": fold_idx,
                "run_id": run_id,
                "source_file": run_df["source_file"].iloc[0],
                "precision": float(precision_score(y_true, y_hat, zero_division=0)),
                "recall": float(recall_score(y_true, y_hat, zero_division=0)),
                "f1": float(compute_binary_metrics(y_true, y_hat)["f1"]),
                "balanced_accuracy": float(compute_binary_metrics(y_true, y_hat)["balanced_accuracy"]) if len(np.unique(y_true)) > 1 else np.nan,
                "detected_at_least_once": int(int(y_hat.sum()) > 0),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    x_shape, x_loading, y, groups, manifest, loading_feature_names = load_dataset(dataset_dir)
    x_shape = transform_shape(x_shape)
    cv_plan = build_group_cv(y, groups, max_splits=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = "shape_loading_dataset"
    model_name = "shape_loading_cnn"
    fold_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_plan.splits, start=1):
        inner_train_idx, val_rel = choose_inner_group_split(y[train_idx], groups[train_idx])
        actual_train_idx = train_idx[inner_train_idx]
        actual_val_idx = train_idx[val_rel]

        train_loading = normalize_loading(x_loading[actual_train_idx], x_loading[actual_train_idx])
        val_loading = normalize_loading(x_loading[actual_train_idx], x_loading[actual_val_idx])
        test_loading = normalize_loading(x_loading[actual_train_idx], x_loading[test_idx])

        train_loader = make_loader(x_shape[actual_train_idx], train_loading, y[actual_train_idx], args.batch_size, shuffle=True)
        val_loader = make_loader(x_shape[actual_val_idx], val_loading, y[actual_val_idx], args.batch_size, shuffle=False)
        test_loader = make_loader(x_shape[test_idx], test_loading, y[test_idx], args.batch_size, shuffle=False)

        model, threshold = train_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            y_train=y[actual_train_idx],
            n_shape_channels=x_shape.shape[1],
            n_loading=train_loading.shape[1],
            args=args,
            device=device,
        )
        negatives = max(int((y[actual_train_idx] == 0).sum()), 1)
        positives = max(int((y[actual_train_idx] == 1).sum()), 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device))
        test_metrics = evaluate(model, test_loader, device, criterion, threshold=threshold)
        fold_rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "fold": fold_idx,
                "cv_plan": cv_plan.name,
                "decision_threshold": threshold,
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "balanced_accuracy": test_metrics["balanced_accuracy"],
                "roc_auc": test_metrics["roc_auc"],
            }
        )
        fold_manifest = manifest.iloc[test_idx].copy()
        run_rows.extend(summarize_runs(dataset_name, model_name, fold_idx, fold_manifest, test_metrics["y_pred"]))

    fold_df = pd.DataFrame(fold_rows)
    run_df = pd.DataFrame(run_rows)
    summary: Dict[str, object] = {"dataset": dataset_name, "model": model_name, "cv_plan": cv_plan.name}
    for metric in ["precision", "recall", "f1", "balanced_accuracy", "roc_auc"]:
        mean_value, ci = mean_ci(fold_df[metric].tolist())
        summary[f"window_{metric}_mean"] = mean_value
        summary[f"window_{metric}_ci95"] = ci
    for metric in ["precision", "recall", "f1", "balanced_accuracy", "detected_at_least_once"]:
        mean_value, ci = mean_ci(run_df[metric].tolist())
        summary[f"run_{metric}_mean"] = mean_value
        summary[f"run_{metric}_ci95"] = ci

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(results_dir / "shape_loading_cnn_fold_results.csv", index=False)
    run_df.to_csv(results_dir / "shape_loading_cnn_run_level_results.csv", index=False)
    pd.DataFrame([summary]).to_csv(results_dir / "shape_loading_cnn_summary.csv", index=False)
    (results_dir / "shape_loading_cnn_metadata.json").write_text(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "loading_feature_names": loading_feature_names,
                "shape_channels_after_transform": int(x_shape.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Shape-loading CNN results written to: {results_dir}")


if __name__ == "__main__":
    main()
