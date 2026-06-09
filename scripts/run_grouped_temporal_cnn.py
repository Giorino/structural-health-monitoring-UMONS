#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
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
    parser = argparse.ArgumentParser(description="Run grouped temporal CNN experiments on dataset F.")
    parser.add_argument("--enriched-dir", type=str, default=None)
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--label-mode", choices=["last", "max"], default="max")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--decision-metric", choices=["f1", "balanced_accuracy"], default="f1")
    parser.add_argument("--dropout", type=float, default=0.25)
    return parser.parse_args()


class TemporalSequenceDataset(Dataset):
    def __init__(self, x_raw: np.ndarray, x_scalar: np.ndarray, y: np.ndarray):
        self.x_raw = torch.as_tensor(x_raw, dtype=torch.float32)
        self.x_scalar = torch.as_tensor(x_scalar, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_raw[idx], self.x_scalar[idx], self.y[idx]


class WindowEncoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=5, padding=2),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Conv1d(24, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TemporalPhysicsCNN(nn.Module):
    def __init__(self, n_channels: int, n_scalar: int, dropout: float):
        super().__init__()
        self.window_encoder = WindowEncoder(n_channels)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(n_scalar, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Conv1d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )

    def forward(self, x_raw: torch.Tensor, x_scalar: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, window_len = x_raw.shape
        raw_flat = x_raw.reshape(batch_size * seq_len, channels, window_len)
        raw_embed = self.window_encoder(raw_flat).reshape(batch_size, seq_len, 64)
        scalar_embed = self.scalar_encoder(x_scalar.reshape(batch_size * seq_len, -1)).reshape(batch_size, seq_len, 32)
        fused = torch.cat([raw_embed, scalar_embed], dim=-1).transpose(1, 2)
        temporal = self.temporal_conv(fused).squeeze(-1)
        return self.head(temporal).squeeze(1)


@dataclass
class NormalizationState:
    raw_mean: np.ndarray
    raw_std: np.ndarray
    scalar_median: np.ndarray
    scalar_mean: np.ndarray
    scalar_std: np.ndarray


def load_dataset_f(enriched_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    payload = np.load(enriched_dir / "datasets" / "dataset_F_raw_window_tensor_plus_mechanics.npz", allow_pickle=True)
    x_raw = np.asarray(payload["X_raw"], dtype=float)
    x_scalar = np.asarray(payload["X_scalar"], dtype=float)
    y = np.asarray(payload["y"], dtype=int)
    groups = np.asarray(payload["groups"], dtype=object)
    raw_segment_id = np.asarray(payload["raw_segment_id"], dtype=object)
    scalar_feature_names = [str(x) for x in payload["scalar_feature_names"].tolist()]
    return x_raw, x_scalar, y, groups, raw_segment_id, scalar_feature_names


def build_sequence_manifest(enriched_dir: Path, raw_segment_id: np.ndarray, y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    repetition_df = pd.read_csv(enriched_dir / "enriched_repetition_table.csv")
    base_cols = ["run_id", "source_file", "raw_segment_id", "global_repetition_index_within_run", "label_damage_transition"]
    if "source_run_id" in repetition_df.columns:
        base_cols.insert(1, "source_run_id")
    repetition_df = repetition_df[base_cols].copy()
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
    return manifest_df.merge(repetition_df[repetition_key_cols], on=["source_run_id", "raw_segment_id"], how="left")


def build_sequences(
    manifest_df: pd.DataFrame,
    x_raw: np.ndarray,
    x_scalar: np.ndarray,
    y: np.ndarray,
    sequence_length: int,
    label_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    seq_raw: List[np.ndarray] = []
    seq_scalar: List[np.ndarray] = []
    seq_y: List[int] = []
    seq_groups: List[object] = []
    seq_rows: List[Dict[str, object]] = []
    ordered = manifest_df.copy().reset_index(drop=True)
    ordered["row_idx"] = np.arange(len(ordered))
    group_col = "source_run_id" if "source_run_id" in ordered.columns else "run_id"

    for _, run_df in ordered.groupby(group_col, sort=False):
        run_df = run_df.sort_values("global_repetition_index_within_run").reset_index(drop=True)
        if len(run_df) < sequence_length:
            continue
        for start in range(0, len(run_df) - sequence_length + 1):
            chunk = run_df.iloc[start:start + sequence_length]
            row_indices = chunk["row_idx"].to_numpy(dtype=int)
            seq_raw.append(x_raw[row_indices])
            seq_scalar.append(x_scalar[row_indices])
            labels = y[row_indices]
            seq_y.append(int(labels.max()) if label_mode == "max" else int(labels[-1]))
            seq_groups.append(chunk[group_col].iloc[-1])
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

    return (
        np.stack(seq_raw).astype(float),
        np.stack(seq_scalar).astype(float),
        np.asarray(seq_y, dtype=int),
        np.asarray(seq_groups, dtype=object),
        pd.DataFrame(seq_rows),
    )


def compute_normalization(x_raw: np.ndarray, x_scalar: np.ndarray) -> NormalizationState:
    raw_mean = np.nanmean(x_raw, axis=(0, 1, 3), keepdims=True)
    raw_std = np.nanstd(x_raw, axis=(0, 1, 3), keepdims=True)
    raw_std = np.where(raw_std < 1e-6, 1.0, raw_std)
    scalar_median = np.nanmedian(x_scalar.reshape(-1, x_scalar.shape[-1]), axis=0)
    safe_scalar = np.where(np.isnan(x_scalar), scalar_median[None, None, :], x_scalar)
    scalar_mean = np.mean(safe_scalar, axis=(0, 1))
    scalar_std = np.std(safe_scalar, axis=(0, 1))
    scalar_std = np.where(scalar_std < 1e-6, 1.0, scalar_std)
    return NormalizationState(raw_mean, raw_std, scalar_median, scalar_mean, scalar_std)


def apply_normalization(x_raw: np.ndarray, x_scalar: np.ndarray, state: NormalizationState) -> Tuple[np.ndarray, np.ndarray]:
    filled_raw = np.where(np.isnan(x_raw), state.raw_mean, x_raw)
    norm_raw = (filled_raw - state.raw_mean) / state.raw_std
    filled_scalar = np.where(np.isnan(x_scalar), state.scalar_median[None, None, :], x_scalar)
    norm_scalar = (filled_scalar - state.scalar_mean[None, None, :]) / state.scalar_std[None, None, :]
    return norm_raw.astype(np.float32), norm_scalar.astype(np.float32)


def make_loader(x_raw: np.ndarray, x_scalar: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TemporalSequenceDataset(x_raw, x_scalar, y), batch_size=batch_size, shuffle=shuffle)


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
    return metrics


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, metric_name: str) -> float:
    candidate_grid = np.unique(np.concatenate([np.linspace(0.1, 0.9, 17), scores]))
    best_threshold = 0.5
    best_metric = -1.0
    for threshold in candidate_grid:
        preds = (scores >= threshold).astype(int)
        metric_value = float(compute_binary_metrics(y_true, preds, scores)[metric_name])
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
    model = TemporalPhysicsCNN(n_channels=n_channels, n_scalar=n_scalar, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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

        val_raw = evaluate_loader(model, val_loader, device, criterion, threshold=0.5)
        tuned_threshold = choose_threshold(val_raw["y_true"], val_raw["y_score"], args.decision_metric)
        val_eval = evaluate_loader(model, val_loader, device, criterion, threshold=tuned_threshold)
        history.append(
            {
                "epoch": float(epoch),
                "threshold": float(tuned_threshold),
                "train_loss": train_loss / max(len(train_loader.dataset), 1),
                "val_loss": float(val_eval["loss"]),
                "val_f1": float(val_eval["f1"]),
                "val_balanced_accuracy": float(val_eval["balanced_accuracy"]),
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


def summarize_grouped_results(fold_df: pd.DataFrame, run_df: pd.DataFrame, dataset_name: str, model_name: str) -> pd.DataFrame:
    row: Dict[str, object] = {"dataset": dataset_name, "model": model_name, "cv_plan": fold_df["cv_plan"].iloc[0]}
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
    x_raw, x_scalar, y, groups, raw_segment_id, scalar_feature_names = load_dataset_f(enriched_dir)
    manifest_df = build_sequence_manifest(enriched_dir, raw_segment_id, y, groups)
    x_raw_seq, x_scalar_seq, y_seq, groups_seq, seq_manifest = build_sequences(
        manifest_df, x_raw, x_scalar, y, args.sequence_length, args.label_mode
    )

    cv_plan = build_group_cv(y_seq, groups_seq, max_splits=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = f"dataset_F_temporal_seq{args.sequence_length}_{args.label_mode}"
    model_name = "temporal_physics_cnn"
    fold_rows: List[Dict[str, object]] = []
    window_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_plan.splits, start=1):
        inner_train_idx, val_rel = choose_inner_group_split(y_seq[train_idx], groups_seq[train_idx])
        actual_train_idx = train_idx[inner_train_idx]
        actual_val_idx = train_idx[val_rel]

        norm_state = compute_normalization(x_raw_seq[actual_train_idx], x_scalar_seq[actual_train_idx])
        x_train_raw, x_train_scalar = apply_normalization(x_raw_seq[actual_train_idx], x_scalar_seq[actual_train_idx], norm_state)
        x_val_raw, x_val_scalar = apply_normalization(x_raw_seq[actual_val_idx], x_scalar_seq[actual_val_idx], norm_state)
        x_test_raw, x_test_scalar = apply_normalization(x_raw_seq[test_idx], x_scalar_seq[test_idx], norm_state)

        train_loader = make_loader(x_train_raw, x_train_scalar, y_seq[actual_train_idx], args.batch_size, shuffle=True)
        val_loader = make_loader(x_val_raw, x_val_scalar, y_seq[actual_val_idx], args.batch_size, shuffle=False)
        test_loader = make_loader(x_test_raw, x_test_scalar, y_seq[test_idx], args.batch_size, shuffle=False)

        model, best_threshold, history = train_one_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            y_train=y_seq[actual_train_idx],
            n_channels=x_raw_seq.shape[2],
            n_scalar=x_scalar_seq.shape[2],
            args=args,
            device=device,
        )
        negatives = max(int((y_seq[actual_train_idx] == 0).sum()), 1)
        positives = max(int((y_seq[actual_train_idx] == 1).sum()), 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device))
        test_metrics = evaluate_loader(model, test_loader, device, criterion, threshold=best_threshold)
        fold_rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "fold": fold_idx,
                "cv_plan": cv_plan.name,
                "sequence_length": args.sequence_length,
                "decision_threshold": best_threshold,
                **{key: test_metrics[key] for key in ["precision", "recall", "f1", "balanced_accuracy", "roc_auc"]},
            }
        )
        fold_window = seq_manifest.iloc[test_idx].copy()
        fold_window["dataset"] = dataset_name
        fold_window["model"] = model_name
        fold_window["fold"] = fold_idx
        fold_window["y_pred"] = test_metrics["y_pred"]
        fold_window["y_score"] = test_metrics["y_score"]
        window_rows.extend(fold_window.to_dict("records"))
        run_rows.extend(summarize_runs(dataset_name, model_name, fold_idx, fold_window, test_metrics["y_pred"]))

    tag = f"temporal_cnn_seq{args.sequence_length}_{args.label_mode}"
    fold_df = pd.DataFrame(fold_rows)
    run_df = pd.DataFrame(run_rows)
    window_df = pd.DataFrame(window_rows)
    summary_df = summarize_grouped_results(fold_df, run_df, dataset_name, model_name)
    fold_df.to_csv(results_dir / f"{tag}_fold_results.csv", index=False)
    run_df.to_csv(results_dir / f"{tag}_run_level_results.csv", index=False)
    window_df.to_csv(results_dir / f"{tag}_window_level_results.csv", index=False)
    summary_df.to_csv(results_dir / f"{tag}_summary.csv", index=False)
    (results_dir / f"{tag}_metadata.json").write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "model": model_name,
                "sequence_length": args.sequence_length,
                "label_mode": args.label_mode,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "patience": args.patience,
                "decision_metric": args.decision_metric,
                "scalar_feature_names": scalar_feature_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Temporal CNN results written to: {results_dir}")


if __name__ == "__main__":
    main()
