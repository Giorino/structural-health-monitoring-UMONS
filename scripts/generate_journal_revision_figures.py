#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
IMG_DIR = ROOT / "latex-journal-revision" / "img"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG_DIR / f"{stem}.png", dpi=400)
    fig.savefig(IMG_DIR / f"{stem}.pdf")
    plt.close(fig)


def draw_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str, face: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0,
        edgecolor="#233142",
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + 0.02, y + h - 0.055, title, ha="left", va="top", fontsize=9.2, fontweight="bold", color="#10212b")
    ax.text(x + 0.02, y + h - 0.11, body, ha="left", va="top", fontsize=7.0, color="#233142", linespacing=1.22)


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.0,
        color="#425466",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def make_pipeline_figure() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(11, 5.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.95, "Method Overview", fontsize=14, fontweight="bold", color="#10212b", va="top")
    ax.text(
        0.03,
        0.90,
        "Peak-aligned FBG responses are converted into compact loading-aware sequences and classified under strict run-level validation.",
        fontsize=9.2,
        color="#425466",
        va="top",
    )

    y = 0.56
    h = 0.22
    positions = [0.03, 0.27, 0.51, 0.75]
    titles = [
        "1. Alignment",
        "2. Feature construction",
        "3. Sequence CNN",
        "4. Run-level decision",
    ]
    bodies = [
        "Raw interrogator traces are\nmatched with loading groups\nand segmented into repeated\npeak-centered responses.",
        "Each response becomes a\n9-feature vector with wavelength\nstatistics, residualized shift,\nand synchronized loading terms.",
        "Thirty consecutive responses\nare processed by a simple 1D CNN\nwith temporal convolutions,\nglobal pooling, and an MLP head.",
        "Scores are thresholded at 0.75,\nfiltered by loading consistency,\nand aggregated into a\nrun-level crack-event decision.",
    ]
    fills = ["#e8f1fb", "#eef7ee", "#fff4df", "#fdecef"]
    for x, title, body, fill in zip(positions, titles, bodies, fills):
        draw_box(ax, x, y, 0.20, h, title, body, fill)

    draw_arrow(ax, (0.23, y + h / 2), (0.27, y + h / 2))
    draw_arrow(ax, (0.47, y + h / 2), (0.51, y + h / 2))
    draw_arrow(ax, (0.71, y + h / 2), (0.75, y + h / 2))

    footer = FancyBboxPatch(
        (0.03, 0.16),
        0.94,
        0.20,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.9,
        edgecolor="#c7d3df",
        facecolor="#f7f9fb",
    )
    ax.add_patch(footer)
    ax.text(0.05, 0.33, "Retained operating point", fontsize=10, fontweight="bold", color="#10212b", va="top")
    ax.text(
        0.05,
        0.285,
        "Leave-one-run-out validation; sequence length = 30; prediction horizon = 5; seed = 42; weighted BCE training; "
        "physics-consistency filter with mean force slope and maximum force thresholds.",
        fontsize=8.1,
        color="#233142",
        va="top",
        linespacing=1.25,
    )
    ax.text(0.05, 0.205, "Final 9-feature set", fontsize=9.2, fontweight="bold", color="#10212b", va="top")
    ax.text(
        0.05,
        0.168,
        "wl2_median, wl2_std, delta_wl_ch2, force_N, displacement_mm, air_pressure_bar, delta_wl_rate, delta_disp_rate, is_small_sample",
        fontsize=7.9,
        color="#425466",
        va="top",
    )

    save_figure(fig, "simple_sequence_cnn_pipeline")


def make_results_figure() -> None:
    apply_style()
    threshold_df = pd.read_csv(RESULTS_DIR / "strict_oldstyle_cnn_20260609_091055_loocv_nomix_physfilter_threshold_sweep.csv")
    baseline_df = pd.read_csv(RESULTS_DIR / "strict_oldstyle_cnn_20260609_091055_loocv_nomix_tuned_summary.csv")
    filtered_df = pd.read_csv(RESULTS_DIR / "strict_oldstyle_cnn_physfilter_newraw_mergedhint_20260609_loocv_nomix_full_t0p75_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), gridspec_kw={"width_ratios": [1.25, 1.0]})

    colors = {
        "precision": "#1b6ca8",
        "recall": "#2d936c",
        "f1": "#c03a2b",
    }
    ax = axes[0]
    for key, label in [("experiment_precision", "Precision"), ("experiment_recall", "Recall"), ("experiment_f1", "F1 score")]:
        series_key = key.split("_")[1]
        ax.plot(
            threshold_df["threshold"],
            threshold_df[key],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            color=colors[series_key],
            label=label,
        )
    selected = threshold_df.loc[threshold_df["threshold"] == 0.75].iloc[0]
    ax.axvline(0.75, color="#425466", linestyle="--", linewidth=1.0)
    ax.scatter([0.75], [selected["experiment_f1"]], s=45, color=colors["f1"], zorder=5)
    ax.annotate(
        "Selected threshold\n0.75",
        xy=(0.75, selected["experiment_f1"]),
        xytext=(0.78, 0.81),
        textcoords="data",
        fontsize=8.3,
        arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": "#425466"},
        color="#233142",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Experiment-level metric")
    ax.set_ylim(0.35, 0.9)
    ax.set_xlim(0.48, 0.97)
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Threshold sweep with physics-consistency filter", loc="left", pad=8)
    ax.text(-0.13, 1.04, "A", transform=ax.transAxes, fontsize=13, fontweight="bold", color="#10212b")

    ax = axes[1]
    categories = ["TP runs", "FP runs", "FN runs"]
    baseline_counts = [
        int(baseline_df["experiment_tp"].iloc[0]),
        int(baseline_df["experiment_fp"].iloc[0]),
        int(baseline_df["experiment_fn"].iloc[0]),
    ]
    filtered_counts = [
        int(filtered_df["experiment_tp"].iloc[0]),
        int(filtered_df["experiment_fp"].iloc[0]),
        int(filtered_df["experiment_fn"].iloc[0]),
    ]
    xpos = range(len(categories))
    width = 0.34
    ax.bar([x - width / 2 for x in xpos], baseline_counts, width=width, color="#9fb3c8", label="Baseline")
    ax.bar([x + width / 2 for x in xpos], filtered_counts, width=width, color="#c03a2b", label="Filtered @ 0.75")
    for x, y in zip([x - width / 2 for x in xpos], baseline_counts):
        ax.text(x, y + 0.18, str(y), ha="center", va="bottom", fontsize=8)
    for x, y in zip([x + width / 2 for x in xpos], filtered_counts):
        ax.text(x, y + 0.18, str(y), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(xpos), categories)
    ax.set_ylabel("Number of runs")
    ax.set_ylim(0, max(baseline_counts + filtered_counts) + 2)
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Run-level error profile", loc="left", pad=8)
    ax.text(-0.16, 1.04, "B", transform=ax.transAxes, fontsize=13, fontweight="bold", color="#10212b")

    save_figure(fig, "strict_loocv_results_summary")


def main() -> None:
    make_pipeline_figure()
    make_results_figure()


if __name__ == "__main__":
    main()
