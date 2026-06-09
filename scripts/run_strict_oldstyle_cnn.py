#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pipeline_common import (
    ROOT,
    build_group_cv,
    choose_inner_group_split,
    compute_binary_metrics,
    latest_enriched_dir_or_fail,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a strict grouped approximation of the old 30-seq physics-filter CNN.")
    parser.add_argument("--enriched-dir", type=str, default=None, help="Path to a specific enriched dataset directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training and fold results.")
    parser.add_argument(
        "--feature-set",
        choices=["full", "shape_loading", "shape_only", "mechanics_core", "mechanics_plus"],
        default="full",
        help="Feature preset for ablations. Mechanics presets append Euler-Bernoulli-driven terms.",
    )
    parser.add_argument("--sequence-length", type=int, default=30, help="Window length.")
    parser.add_argument("--prediction-horizon", type=int, default=5, help="Prediction horizon after the input window.")
    parser.add_argument("--stride", type=int, default=1, help="Stride between windows.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience.")
    parser.add_argument("--positive-weight-scale", type=float, default=1.0, help="Scale factor applied to the default class-imbalance positive weight.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional fixed decision threshold. If omitted, threshold is tuned on validation folds.")
    parser.add_argument("--decision-metric", choices=["f1", "balanced_accuracy", "recall"], default="f1", help="Validation metric for threshold tuning.")
    parser.add_argument("--use-physics-filter", action="store_true", help="Apply the hard force-slope / max-force filter from the legacy notes.")
    parser.add_argument("--force-slope-threshold", type=float, default=-0.1, help="Mean force slope threshold for the physics filter.")
    parser.add_argument("--min-force-threshold", type=float, default=100.0, help="Minimum max-force threshold for the physics filter.")
    parser.add_argument("--cv-mode", choices=["grouped", "loocv"], default="grouped", help="Grouped K-fold or leave-one-run-out evaluation.")
    parser.add_argument("--max-splits", type=int, default=5, help="Maximum grouped CV splits.")
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Beta(alpha, alpha) mixup strength. Set 0 to disable.")
    parser.add_argument("--mixup-prob", type=float, default=0.0, help="Probability of applying mixup to each training batch.")
    parser.add_argument("--start-fold", type=int, default=1, help="1-based fold index to start from.")
    parser.add_argument("--max-folds", type=int, default=None, help="Optional number of folds to process from start-fold.")
    return parser.parse_args()


def get_feature_columns(feature_set: str) -> List[str]:
    if feature_set == "shape_loading":
        return [
            "wl2_median",
            "wl2_std",
            "delta_wl_ch2",
            "force_N",
            "displacement_mm",
        ]
    if feature_set == "shape_only":
        return [
            "wl2_median",
            "wl2_std",
            "delta_wl_ch2",
        ]
    if feature_set == "mechanics_core":
        return [
            "wl2_median",
            "wl2_std",
            "delta_wl_ch2",
            "force_N",
            "displacement_mm",
            "air_pressure_bar",
            "delta_wl_rate",
            "delta_disp_rate",
            "expected_elastic_strain_microstrain",
            "observed_strain_microstrain_sensitivity",
            "mechanics_residual_microstrain",
            "mechanics_residual_abs_microstrain",
            "mechanics_residual_normalized",
            "force_increment_from_baseline_N",
            "displacement_increment_from_baseline_mm",
            "is_small_sample",
        ]
    if feature_set == "mechanics_plus":
        return [
            "wl2_median",
            "wl2_std",
            "delta_wl_ch2",
            "force_N",
            "displacement_mm",
            "air_pressure_bar",
            "delta_wl_rate",
            "delta_disp_rate",
            "expected_elastic_strain_microstrain",
            "observed_strain_microstrain_sensitivity",
            "mechanics_residual_microstrain",
            "mechanics_residual_abs_microstrain",
            "mechanics_residual_normalized",
            "mechanics_residual_change_from_baseline",
            "mechanics_residual_change_from_previous_group",
            "force_increment_from_baseline_N",
            "force_increment_from_previous_group_N",
            "displacement_increment_from_baseline_mm",
            "displacement_increment_from_previous_group_mm",
            "bending_moment_N_mm",
            "estimated_strain_per_force_level",
            "support_span_mm",
            "specimen_thickness_mm",
            "specimen_width_mm",
            "number_of_layers",
            "is_small_sample",
        ]
    return [
        "wl2_median",
        "wl2_std",
        "delta_wl_ch2",
        "force_N",
        "displacement_mm",
        "air_pressure_bar",
        "delta_wl_rate",
        "delta_disp_rate",
        "is_small_sample",
    ]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    positive = logits >= 0.0
    neg_exp = np.exp(np.where(positive, -logits, logits))
    return np.where(positive, 1.0 / (1.0 + neg_exp), neg_exp / (1.0 + neg_exp))


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class OldStyleSequenceCNN(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=12, padding=6),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=12, padding=6),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=12, padding=6),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        feat = self.cnn(x).squeeze(-1)
        return self.classifier(feat).squeeze(1)


@dataclass
class WindowSample:
    x: np.ndarray
    y: int
    group: str
    row: Dict[str, object]


def load_repetition_table(enriched_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(enriched_dir / "enriched_repetition_table.csv")
    df = df.sort_values(["source_run_id", "global_repetition_index_within_run"]).reset_index(drop=True)
    return df


def add_oldstyle_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["is_small_sample"] = enriched["source_file"].astype(str).str.contains("-s-", regex=False).astype(int)
    enriched["delta_wl_ch2"] = enriched["wl2_residual_run_baseline"]
    enriched["delta_wl_rate"] = enriched.groupby("source_run_id")["delta_wl_ch2"].diff().fillna(0.0)
    enriched["delta_disp_rate"] = enriched.groupby("source_run_id")["displacement_mm"].diff().fillna(0.0)
    return enriched


def build_windows(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    feature_columns = get_feature_columns(args.feature_set)
    samples: List[WindowSample] = []

    for run_id, run_df in df.groupby("source_run_id", sort=False):
        run_df = run_df.sort_values("global_repetition_index_within_run").reset_index(drop=True)
        if len(run_df) < args.sequence_length + args.prediction_horizon:
            continue
        features = run_df[feature_columns].to_numpy(dtype=float)
        labels = run_df["label_damage_transition"].to_numpy(dtype=int)
        for start in range(0, len(run_df) - args.sequence_length - args.prediction_horizon + 1, args.stride):
            end = start + args.sequence_length
            label_idx = end + args.prediction_horizon - 1
            x_window = features[start:end]
            future_label = int(labels[label_idx])
            current_max = int(labels[start:end].max())
            y = max(future_label, current_max)
            window_force = run_df["force_N"].iloc[start:end].to_numpy(dtype=float)
            force_diffs = np.diff(window_force) if len(window_force) > 1 else np.array([0.0])
            mean_force_slope = float(np.mean(force_diffs)) if force_diffs.size else 0.0
            max_force = float(np.nanmax(window_force)) if len(window_force) else np.nan
            row = {
                "run_id": run_df["run_id"].iloc[-1],
                "source_run_id": run_id,
                "source_file": run_df["source_file"].iloc[-1],
                "raw_segment_id": run_df["raw_segment_id"].iloc[end - 1],
                "window_start_index": int(run_df["global_repetition_index_within_run"].iloc[start]),
                "window_end_index": int(run_df["global_repetition_index_within_run"].iloc[end - 1]),
                "label_index": int(run_df["global_repetition_index_within_run"].iloc[label_idx]),
                "label_damage_transition": int(y),
                "mean_force_slope": mean_force_slope,
                "max_force_N": max_force,
            }
            samples.append(WindowSample(x=x_window, y=y, group=run_id, row=row))

    x = np.stack([sample.x for sample in samples]).astype(float)
    y = np.asarray([sample.y for sample in samples], dtype=int)
    groups = np.asarray([sample.group for sample in samples], dtype=object)
    manifest_df = pd.DataFrame([sample.row for sample in samples])
    return x, y, groups, manifest_df, feature_columns


def compute_norm(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x_train, axis=(0, 1), keepdims=True)
    std = np.nanstd(x_train, axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_mean = np.where(np.isnan(mean), 0.0, mean)
    x_filled = np.where(np.isnan(x), safe_mean, x)
    return ((x_filled - safe_mean) / std).astype(np.float32)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(WindowDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def sample_mixup(x_batch: torch.Tensor, y_batch: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0 or len(x_batch) < 2:
        return x_batch, y_batch, y_batch, 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(len(x_batch), device=x_batch.device)
    mixed_x = lam * x_batch + (1.0 - lam) * x_batch[index]
    y_a = y_batch
    y_b = y_batch[index]
    return mixed_x, y_a, y_b, lam


def build_cv_plan(y: np.ndarray, groups: np.ndarray, args: argparse.Namespace) -> Tuple[str, List[Tuple[np.ndarray, np.ndarray]]]:
    if args.cv_mode == "loocv":
        splitter = LeaveOneGroupOut()
        splits = list(splitter.split(np.zeros_like(y), y, groups))
        return "LeaveOneGroupOut", splits
    plan = build_group_cv(y, groups, max_splits=args.max_splits)
    return plan.name, plan.splits


def apply_physics_filter(scores: np.ndarray, manifest_df: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    if not args.use_physics_filter:
        return scores
    keep_mask = ~(
        (manifest_df["mean_force_slope"].to_numpy(dtype=float) < args.force_slope_threshold)
        | (manifest_df["max_force_N"].to_numpy(dtype=float) < args.min_force_threshold)
    )
    filtered = scores.copy()
    filtered[~keep_mask] = 0.0
    return filtered


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    threshold: float,
    manifest_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    logits_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += float(loss.item()) * len(y_batch)
            logits_list.append(logits.cpu().numpy())
            y_list.append(y_batch.cpu().numpy())
    logits_all = np.concatenate(logits_list)
    y_all = np.concatenate(y_list).astype(int)
    raw_scores = stable_sigmoid(logits_all)
    scores = apply_physics_filter(raw_scores, manifest_df, args)
    preds = (scores >= threshold).astype(int)
    metrics = compute_binary_metrics(y_all, preds, scores)
    metrics["loss"] = total_loss / max(len(y_all), 1)
    metrics["y_true"] = y_all
    metrics["y_score"] = scores
    metrics["y_pred"] = preds
    return metrics


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, metric_name: str) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 19), scores]))
    best_threshold = 0.5
    best_value = -1.0
    for threshold in candidates:
        preds = (scores >= threshold).astype(int)
        metrics = compute_binary_metrics(y_true, preds, scores)
        value = float(metrics[metric_name])
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold


def train_one_fold(
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: np.ndarray,
    x_val_manifest: pd.DataFrame,
    n_features: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, float]:
    model = OldStyleSequenceCNN(n_features=n_features).to(device)
    negatives = max(int((y_train == 0).sum()), 1)
    positives = max(int((y_train == 1).sum()), 1)
    pos_weight_value = (negatives / positives) * max(float(args.positive_weight_scale), 1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_metric = -1.0
    best_threshold = 0.5 if args.threshold is None else float(args.threshold)
    best_state: Dict[str, torch.Tensor] | None = None
    patience_left = args.patience

    for _epoch in range(args.epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            if args.mixup_alpha > 0.0 and args.mixup_prob > 0.0 and np.random.rand() < args.mixup_prob:
                mixed_x, y_a, y_b, lam = sample_mixup(x_batch, y_batch, args.mixup_alpha)
                logits = model(mixed_x)
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        raw_val = evaluate_loader(model, val_loader, device, criterion, 0.5, x_val_manifest, args)
        threshold = float(args.threshold) if args.threshold is not None else choose_threshold(raw_val["y_true"], raw_val["y_score"], args.decision_metric)
        tuned_val = evaluate_loader(model, val_loader, device, criterion, threshold, x_val_manifest, args)
        current_value = float(tuned_val[args.decision_metric])
        if current_value > best_metric:
            best_metric = current_value
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


def summarize_run_level(window_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for run_id, run_df in window_df.groupby("source_run_id", sort=False):
        y_true = run_df["label_damage_transition"].to_numpy(dtype=int)
        y_pred = run_df["y_pred"].to_numpy(dtype=int)
        rows.append(
            {
                "source_run_id": run_id,
                "source_file": run_df["source_file"].iloc[0],
                "y_true_run": int(y_true.max()),
                "y_pred_run": int(y_pred.max()),
            }
        )
    return pd.DataFrame(rows)


def confusion_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def build_summary(window_df: pd.DataFrame, run_df: pd.DataFrame, cv_plan_name: str, args: argparse.Namespace) -> Dict[str, object]:
    window_metrics = compute_binary_metrics(
        window_df["label_damage_transition"].to_numpy(dtype=int),
        window_df["y_pred"].to_numpy(dtype=int),
        window_df["y_score"].to_numpy(dtype=float),
    )
    run_metrics = compute_binary_metrics(
        run_df["y_true_run"].to_numpy(dtype=int),
        run_df["y_pred_run"].to_numpy(dtype=int),
    )
    window_conf = confusion_dict(
        window_df["label_damage_transition"].to_numpy(dtype=int),
        window_df["y_pred"].to_numpy(dtype=int),
    )
    run_conf = confusion_dict(
        run_df["y_true_run"].to_numpy(dtype=int),
        run_df["y_pred_run"].to_numpy(dtype=int),
    )
    return {
        "dataset": "strict_oldstyle_rebuilt",
        "model": "oldstyle_sequence_cnn",
        "feature_set": args.feature_set,
        "cv_plan": cv_plan_name,
        "sequence_length": args.sequence_length,
        "prediction_horizon": args.prediction_horizon,
        "stride": args.stride,
        "use_physics_filter": args.use_physics_filter,
        "window_precision": float(window_metrics["precision"]),
        "window_recall": float(window_metrics["recall"]),
        "window_f1": float(window_metrics["f1"]),
        "window_balanced_accuracy": float(window_metrics["balanced_accuracy"]),
        "window_roc_auc": float(window_metrics["roc_auc"]) if not pd.isna(window_metrics["roc_auc"]) else np.nan,
        "window_tn": window_conf["tn"],
        "window_fp": window_conf["fp"],
        "window_fn": window_conf["fn"],
        "window_tp": window_conf["tp"],
        "experiment_precision": float(run_metrics["precision"]),
        "experiment_recall": float(run_metrics["recall"]),
        "experiment_f1": float(run_metrics["f1"]),
        "experiment_balanced_accuracy": float(run_metrics["balanced_accuracy"]),
        "experiment_tn": run_conf["tn"],
        "experiment_fp": run_conf["fp"],
        "experiment_fn": run_conf["fn"],
        "experiment_tp": run_conf["tp"],
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    enriched_dir = Path(args.enriched_dir) if args.enriched_dir else latest_enriched_dir_or_fail()
    repetition_df = add_oldstyle_features(load_repetition_table(enriched_dir))
    x, y, groups, manifest_df, feature_columns = build_windows(repetition_df, args)

    cv_plan_name, cv_splits = build_cv_plan(y, groups, args)
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_rows: List[Dict[str, object]] = []
    window_rows: List[pd.DataFrame] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits, start=1):
        if fold_idx < args.start_fold:
            continue
        if args.max_folds is not None and fold_idx >= args.start_fold + args.max_folds:
            break
        inner_train_idx, val_idx_rel = choose_inner_group_split(y[train_idx], groups[train_idx])
        actual_train_idx = train_idx[inner_train_idx]
        actual_val_idx = train_idx[val_idx_rel]

        mean, std = compute_norm(x[actual_train_idx])
        x_train = apply_norm(x[actual_train_idx], mean, std)
        x_val = apply_norm(x[actual_val_idx], mean, std)
        x_test = apply_norm(x[test_idx], mean, std)

        train_loader = make_loader(x_train, y[actual_train_idx], args.batch_size, shuffle=True)
        val_loader = make_loader(x_val, y[actual_val_idx], args.batch_size, shuffle=False)
        test_loader = make_loader(x_test, y[test_idx], args.batch_size, shuffle=False)

        model, threshold = train_one_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            y_train=y[actual_train_idx],
            x_val_manifest=manifest_df.iloc[actual_val_idx].reset_index(drop=True),
            n_features=x.shape[2],
            args=args,
            device=device,
        )

        negatives = max(int((y[actual_train_idx] == 0).sum()), 1)
        positives = max(int((y[actual_train_idx] == 1).sum()), 1)
        pos_weight_value = (negatives / positives) * max(float(args.positive_weight_scale), 1e-6)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))
        test_manifest = manifest_df.iloc[test_idx].reset_index(drop=True)
        test_metrics = evaluate_loader(model, test_loader, device, criterion, threshold, test_manifest, args)
        fold_rows.append(
            {
                "fold": fold_idx,
                "cv_plan": cv_plan_name,
                "decision_threshold": threshold,
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "balanced_accuracy": test_metrics["balanced_accuracy"],
                "roc_auc": test_metrics["roc_auc"],
                "n_test_windows": len(test_idx),
                "n_test_runs": len(np.unique(groups[test_idx])),
            }
        )

        fold_window = test_manifest.copy()
        fold_window["fold"] = fold_idx
        fold_window["y_score"] = test_metrics["y_score"]
        fold_window["y_pred"] = test_metrics["y_pred"]
        window_rows.append(fold_window)

    if not fold_rows or not window_rows:
        print("No eligible folds were processed for the requested fold range.")
        return

    fold_df = pd.DataFrame(fold_rows)
    window_df = pd.concat(window_rows, ignore_index=True)
    run_df = summarize_run_level(window_df)

    threshold_tag = "tuned" if args.threshold is None else f"t{str(args.threshold).replace('.', 'p')}"
    base_tag = "strict_oldstyle_cnn_physfilter" if args.use_physics_filter else "strict_oldstyle_cnn"
    dataset_tag = Path(enriched_dir).name.replace("enriched_dataset_", "").replace("dataset-new", "dataset-new")
    mixup_tag = "mixup" if args.mixup_alpha > 0.0 and args.mixup_prob > 0.0 else "nomix"
    cv_tag = "loocv" if args.cv_mode == "loocv" else "grouped"
    feature_tag = args.feature_set
    weight_tag = f"pw{str(args.positive_weight_scale).replace('.', 'p')}"
    tag = f"{base_tag}_{dataset_tag}_{cv_tag}_{mixup_tag}_{feature_tag}_{weight_tag}_{threshold_tag}"
    fold_path = results_dir / f"{tag}_fold_results.csv"
    window_path = results_dir / f"{tag}_window_level_results.csv"
    experiment_path = results_dir / f"{tag}_experiment_level_results.csv"
    summary_path = results_dir / f"{tag}_summary.csv"

    if args.start_fold > 1 and fold_path.exists() and window_path.exists():
        prev_fold_df = pd.read_csv(fold_path)
        prev_window_df = pd.read_csv(window_path)
        fold_df = pd.concat([prev_fold_df, fold_df], ignore_index=True).drop_duplicates(subset=["fold"], keep="last").sort_values("fold")
        window_df = pd.concat([prev_window_df, window_df], ignore_index=True).drop_duplicates(
            subset=["fold", "source_run_id", "window_start_index", "window_end_index", "label_index"],
            keep="last",
        ).sort_values(["fold", "source_run_id", "window_start_index"])
        run_df = summarize_run_level(window_df)

    summary = build_summary(window_df, run_df, cv_plan_name, args)
    fold_df.to_csv(fold_path, index=False)
    window_df.to_csv(window_path, index=False)
    run_df.to_csv(experiment_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    metadata = {
        "feature_columns": feature_columns,
        "args": vars(args),
    }
    (results_dir / f"{tag}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Strict oldstyle CNN results written to: {results_dir}")


if __name__ == "__main__":
    main()
