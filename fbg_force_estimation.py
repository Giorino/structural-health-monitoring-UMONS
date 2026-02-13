#!/usr/bin/env python3
"""
FBG-Based Force Estimation from Raw Interrogator Data
=====================================================

Implements the full physics chain to estimate applied force purely from
FBG wavelength measurements, with NO external sensors (no load cell,
no LVDT, no pressure gauge):

    Δλ  →  ε = Δλ / (λ₀ × (1 - pₑ))  →  σ = E × ε  →  F = (σ × b × h³) / (3 × L × y_FBG)

Physics background (Euler-Bernoulli 3-point bending):
    ε = (3 × F × L × y_FBG) / (E × b × h³)
    Solving for F:  F = (ε × E × b × h³) / (3 × L × y_FBG)
    Which is equivalent to:  F = (σ × b × h³) / (3 × L × y_FBG)  since σ = E × ε

Parameters from the UMONS experimental paper:
    E       = 18.6 GPa      (effective Young's Modulus, measured from FBG data)
    pₑ      = 0.22           (photoelastic coefficient for silica optical fiber)
    b       = 34 mm          (beam width, constant for all specimens)
    layer_t = 0.333 mm/layer (ply thickness, from 12-layer specimen: 4.0mm / 12)
    FBG placement: 2 layers below top surface for all specimens
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from pathlib import Path
from datetime import datetime


# =============================================================================
# Material & Geometric Constants
# =============================================================================

E_GPA = 8.6                    # Young's Modulus (GPa), from paper
E_MPA = E_GPA * 1e3             # Young's Modulus (MPa)
PE = 0.22                       # Photoelastic coefficient for silica fiber
BEAM_WIDTH_MM = 34.0            # b: beam width (mm), constant for all specimens
LAYER_THICKNESS_MM = 1.0 / 3.0  # ~0.333 mm/layer (from 12-layer: h=4.0mm / 12)

# Small samples may have a different width — set here (update if measured)
SMALL_SAMPLE_WIDTH_MM = 34.0    # same default; update if different

# Baseline estimation: use first N points to compute a stable λ₀
BASELINE_POINTS = 200

# Median filter kernel size for noise reduction on raw wavelength signal
MEDIAN_KERNEL = 7

# Central Bragg channel column name in interrogator data
CENTRAL_CHANNEL_IDX = 2  # WL 2 (0-indexed: col index 3 in the file → WL 2[nm])

# Output directory
OUTPUT_DIR = "fbg_force_estimation_output"


# =============================================================================
# Filename Parsing
# =============================================================================

def parse_interrogator_filename(filename: str) -> dict:
    """
    Parse interrogator filenames like:
        23cm-12layers-1-interrogator.txt         (regular)
        15cm-16layers-2-s-interrogator.txt        (small sample)
        19 cm-16 layers-7-s-interrogator.txt      (with spaces)
        15cm-16-layers-2-s-interrogator.txt       (with extra hyphen)
        27cm-12-layers-5-interrogator.txt         (with extra hyphen)

    Returns dict with: span_cm, n_layers, sample_number, is_small
    """
    base = os.path.basename(filename).replace("-interrogator.txt", "").replace("-interrogator", "")

    # Normalize: remove spaces around hyphens and units
    base_clean = base.replace(" ", "")

    # Detect small sample
    is_small = "-s" in base_clean
    base_clean = base_clean.replace("-s", "")

    # Try to extract: {span}cm-{layers}layers-{number}
    # Handle variants like "15cm-16-layers-2" and "15cm-16layers-2"
    m = re.match(r"(\d+)cm-?(\d+)-?layers-?(\d+)", base_clean)
    if m:
        return {
            "span_cm": int(m.group(1)),
            "n_layers": int(m.group(2)),
            "sample_number": int(m.group(3)),
            "is_small": is_small,
        }

    print(f"  WARNING: Could not parse filename '{filename}', skipping.")
    return None


def compute_geometry(n_layers: int, is_small: bool = False) -> dict:
    """
    Compute beam geometry for a given specimen configuration.

    Returns dict with: h_mm, y_fbg_mm, b_mm, I_mm4
    """
    h = n_layers * LAYER_THICKNESS_MM           # total thickness (mm)
    b = SMALL_SAMPLE_WIDTH_MM if is_small else BEAM_WIDTH_MM

    # FBG is placed 2 layers below the top surface
    fbg_layer_from_bottom = n_layers - 2        # layer index from bottom
    y_from_bottom = fbg_layer_from_bottom * LAYER_THICKNESS_MM
    y_fbg = y_from_bottom - (h / 2.0)          # distance from neutral axis (mm)

    # Moment of inertia I = b × h³ / 12
    I = (b * h**3) / 12.0

    return {
        "h_mm": h,
        "y_fbg_mm": y_fbg,
        "b_mm": b,
        "I_mm4": I,
    }


# =============================================================================
# Data Loading
# =============================================================================

def load_interrogator_data(filepath: str) -> pd.DataFrame:
    """Load raw FBG interrogator text file. Returns DataFrame with wavelength columns."""
    iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}T")
    data_start = 0
    first_data_tokens = None

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= 50:
                break
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if iso_re.match(stripped) and len(parts) >= 3:
                data_start = i
                first_data_tokens = parts
                break
            else:
                try:
                    float(parts[0])
                    if len(parts) >= 2:
                        data_start = i
                        first_data_tokens = parts
                        break
                except (ValueError, IndexError):
                    pass

    if first_data_tokens is None:
        raise ValueError(f"Could not find data start in {filepath}")

    n_tokens = len(first_data_tokens)
    if iso_re.match(first_data_tokens[0]):
        names = ["Timestamp", "Time_s"]
        wl_count = max(0, n_tokens - 2)
    else:
        names = ["Time_s"]
        wl_count = max(0, n_tokens - 1)
    names += [f"WL_{i}" for i in range(1, wl_count + 1)]

    try:
        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            names=names,
            skiprows=data_start,
            engine="python",
            on_bad_lines="skip",
        )
    except TypeError:
        # Older pandas versions use error_bad_lines instead
        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            names=names,
            skiprows=data_start,
            engine="python",
            error_bad_lines=False,
            warn_bad_lines=False,
        )

    # Convert wavelength columns to numeric
    wl_cols = [c for c in df.columns if c.startswith("WL_")]
    for c in wl_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convert Time_s to numeric
    if "Time_s" in df.columns:
        df["Time_s"] = pd.to_numeric(df["Time_s"], errors="coerce")

    df = df.dropna(subset=wl_cols)
    return df, wl_cols


# =============================================================================
# Physics Chain: Δλ → ε → σ → F
# =============================================================================

def compute_force_from_wavelength(
    wavelength_nm: np.ndarray,
    baseline_nm: float,
    E_mpa: float,
    b_mm: float,
    h_mm: float,
    L_mm: float,
    y_fbg_mm: float,
    pe: float = PE,
) -> dict:
    """
    Full physics chain from FBG wavelength to estimated force.

    Steps:
        1. Δλ = λ - λ₀                             (wavelength shift, nm)
        2. ε  = Δλ / (λ₀ × (1 - pₑ))              (strain, dimensionless)
        3. σ  = E × ε                               (stress, MPa)
        4. F  = (σ × b × h³) / (3 × L × y_FBG)    (force, N)

    Parameters
    ----------
    wavelength_nm : array of absolute wavelengths (nm)
    baseline_nm   : baseline Bragg wavelength λ₀ (nm)
    E_mpa         : Young's modulus (MPa)
    b_mm          : beam width (mm)
    h_mm          : beam thickness (mm)
    L_mm          : support span (mm)
    y_fbg_mm      : FBG distance from neutral axis (mm)
    pe            : photoelastic coefficient

    Returns
    -------
    dict with arrays: delta_lambda_nm, strain, stress_mpa, force_N
    """
    # Step 1: Wavelength shift
    delta_lambda = wavelength_nm - baseline_nm  # nm

    # Step 2: Strain (dimensionless)
    strain = delta_lambda / (baseline_nm * (1.0 - pe))

    # Step 3: Stress (MPa)
    stress = E_mpa * strain

    # Step 4: Force (N) from Euler-Bernoulli 3-point bending
    # ε = (3 × F × L × y_FBG) / (E × b × h³)
    # → F = (ε × E × b × h³) / (3 × L × y_FBG)
    # → F = (σ × b × h³) / (3 × L × y_FBG)
    force = (stress * b_mm * h_mm**3) / (3.0 * L_mm * y_fbg_mm)

    return {
        "delta_lambda_nm": delta_lambda,
        "strain": strain,
        "strain_microstrain": strain * 1e6,
        "stress_mpa": stress,
        "force_N": force,
    }


# =============================================================================
# Processing Pipeline
# =============================================================================

def process_single_file(filepath: str, output_dir: str) -> dict:
    """Process a single interrogator file through the full physics chain."""
    filename = os.path.basename(filepath)
    print(f"\n{'='*70}")
    print(f"Processing: {filename}")
    print(f"{'='*70}")

    # 1. Parse filename
    info = parse_interrogator_filename(filename)
    if info is None:
        return None

    span_cm = info["span_cm"]
    n_layers = info["n_layers"]
    is_small = info["is_small"]
    sample_num = info["sample_number"]
    L_mm = span_cm * 10.0  # convert cm to mm

    print(f"  Span: {span_cm} cm ({L_mm} mm)")
    print(f"  Layers: {n_layers}")
    print(f"  Sample #: {sample_num}")
    print(f"  Small sample: {is_small}")

    # 2. Compute geometry
    geom = compute_geometry(n_layers, is_small)
    h_mm = geom["h_mm"]
    y_fbg_mm = geom["y_fbg_mm"]
    b_mm = geom["b_mm"]
    I_mm4 = geom["I_mm4"]

    print(f"  Thickness (h): {h_mm:.3f} mm")
    print(f"  Width (b): {b_mm:.1f} mm")
    print(f"  y_FBG (from neutral axis): {y_fbg_mm:.3f} mm")
    print(f"  I (moment of inertia): {I_mm4:.3f} mm^4")

    # 3. Load data
    df, wl_cols = load_interrogator_data(filepath)
    if len(wl_cols) < CENTRAL_CHANNEL_IDX:
        if len(wl_cols) >= 1:
            print(f"  WARNING: File has only {len(wl_cols)} WL channels, using WL_1 as fallback")
            central_col = wl_cols[0]
        else:
            print(f"  ERROR: No WL channels found, skipping.")
            return None
    else:
        central_col = wl_cols[CENTRAL_CHANNEL_IDX - 1]  # WL_2 (1-indexed)
    print(f"  Central channel: {central_col}")
    print(f"  Total data points: {len(df)}")

    # 4. Get raw wavelength signal
    wl_raw = df[central_col].values

    # 5. Apply median filter for noise reduction
    kernel = MEDIAN_KERNEL if MEDIAN_KERNEL % 2 == 1 else MEDIAN_KERNEL + 1
    wl_smooth = medfilt(wl_raw, kernel_size=kernel)

    # 6. Compute baseline λ₀ from first N stable points
    baseline_points = min(BASELINE_POINTS, len(wl_smooth))
    baseline_nm = np.median(wl_smooth[:baseline_points])
    print(f"  Baseline λ₀: {baseline_nm:.5f} nm (median of first {baseline_points} points)")

    # 7. Apply physics chain
    results = compute_force_from_wavelength(
        wavelength_nm=wl_smooth,
        baseline_nm=baseline_nm,
        E_mpa=E_MPA,
        b_mm=b_mm,
        h_mm=h_mm,
        L_mm=L_mm,
        y_fbg_mm=y_fbg_mm,
    )

    # 8. Build output dataframe
    time_col = "Time_s" if "Time_s" in df.columns else None
    out_df = pd.DataFrame({
        "sample_index": np.arange(len(df)),
        "wavelength_raw_nm": wl_raw,
        "wavelength_filtered_nm": wl_smooth,
        "delta_lambda_nm": results["delta_lambda_nm"],
        "strain_microstrain": results["strain_microstrain"],
        "stress_MPa": results["stress_mpa"],
        "estimated_force_N": results["force_N"],
    })
    if time_col:
        out_df.insert(0, "time_s", df[time_col].values)

    # 9. Summary statistics
    # Only consider positive force values (loading phase)
    positive_mask = results["force_N"] > 0
    if positive_mask.any():
        f_pos = results["force_N"][positive_mask]
        print(f"\n  --- Estimated Force Summary (loading phases only) ---")
        print(f"  Min force:  {f_pos.min():.2f} N")
        print(f"  Max force:  {f_pos.max():.2f} N")
        print(f"  Mean force: {f_pos.mean():.2f} N")
        print(f"  Std force:  {f_pos.std():.2f} N")
    else:
        print(f"\n  WARNING: No positive force values found. Check baseline/signal.")

    max_strain = np.abs(results["strain_microstrain"]).max()
    max_stress = np.abs(results["stress_mpa"]).max()
    print(f"  Max |strain|: {max_strain:.1f} microstrain")
    print(f"  Max |stress|: {max_stress:.2f} MPa")

    # 10. Save per-file CSV
    safe_name = filename.replace("-interrogator.txt", "")
    csv_path = os.path.join(output_dir, f"force_estimation_{safe_name}.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return {
        "filename": filename,
        "span_cm": span_cm,
        "n_layers": n_layers,
        "sample_number": sample_num,
        "is_small": is_small,
        "h_mm": h_mm,
        "b_mm": b_mm,
        "y_fbg_mm": y_fbg_mm,
        "L_mm": L_mm,
        "baseline_nm": baseline_nm,
        "n_datapoints": len(df),
        "max_force_N": float(results["force_N"].max()),
        "min_force_N": float(results["force_N"].min()),
        "max_strain_ue": float(max_strain),
        "max_stress_MPa": float(max_stress),
        "results_df": out_df,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_single_file(result: dict, output_dir: str):
    """Generate a 4-panel plot for a single file showing the full physics chain."""
    df = result["results_df"]
    filename = result["filename"].replace("-interrogator.txt", "")
    x = df["sample_index"].values

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f"FBG Force Estimation: {filename}\n"
        f"Span={result['span_cm']}cm, Layers={result['n_layers']}, "
        f"λ₀={result['baseline_nm']:.3f}nm, "
        f"{'Small' if result['is_small'] else 'Regular'} sample",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: Wavelength & Δλ
    ax = axes[0]
    ax.plot(x, df["wavelength_filtered_nm"], "b-", linewidth=0.4, alpha=0.8, label="λ (filtered)")
    ax.axhline(result["baseline_nm"], color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"λ₀ = {result['baseline_nm']:.3f} nm")
    ax.set_ylabel("Wavelength (nm)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Step 1: FBG Wavelength (Δλ = λ - λ₀)", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Strain
    ax = axes[1]
    ax.plot(x, df["strain_microstrain"], "g-", linewidth=0.4, alpha=0.8)
    ax.set_ylabel("Strain (μɛ)")
    ax.set_title("Step 2: ε = Δλ / (λ₀ × (1 - pₑ))", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 3: Stress
    ax = axes[2]
    ax.plot(x, df["stress_MPa"], "m-", linewidth=0.4, alpha=0.8)
    ax.set_ylabel("Stress (MPa)")
    ax.set_title(f"Step 3: σ = E × ε   (E = {E_GPA} GPa)", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 4: Force
    ax = axes[3]
    ax.plot(x, df["estimated_force_N"], "r-", linewidth=0.5, alpha=0.8)
    ax.set_ylabel("Estimated Force (N)")
    ax.set_xlabel("Sample Index")
    ax.set_title(
        f"Step 4: F = (σ × b × h³) / (3 × L × y_FBG)   "
        f"(b={result['b_mm']}mm, h={result['h_mm']:.2f}mm, L={result['L_mm']}mm, y_FBG={result['y_fbg_mm']:.2f}mm)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"force_chain_{filename}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")


def plot_summary(all_results: list, output_dir: str):
    """Generate summary comparison plots across all files."""
    if not all_results:
        return

    # --- Plot 1: Max estimated force per file, grouped by span ---
    fig, ax = plt.subplots(figsize=(14, 7))

    # Group by span
    spans = sorted(set(r["span_cm"] for r in all_results))
    colors = plt.cm.tab10(np.linspace(0, 1, len(spans)))
    span_color = {s: c for s, c in zip(spans, colors)}

    for r in all_results:
        color = span_color[r["span_cm"]]
        marker = "s" if r["is_small"] else "o"
        label_key = f"{r['span_cm']}cm-{r['n_layers']}L"
        ax.bar(
            r["filename"].replace("-interrogator.txt", ""),
            r["max_force_N"],
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
        )

    ax.set_ylabel("Max Estimated Force (N)", fontsize=12)
    ax.set_title("Maximum FBG-Estimated Force per Specimen", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    # Custom legend for spans
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=span_color[s], edgecolor="black", label=f"Span: {s} cm") for s in spans]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "summary_max_force.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary plot saved: {plot_path}")

    # --- Plot 2: Force vs Span Length (physics relationship) ---
    fig, ax = plt.subplots(figsize=(10, 6))

    layer_groups = sorted(set(r["n_layers"] for r in all_results))
    markers = {12: "o", 16: "s", 19: "^"}
    layer_colors = {12: "#2196F3", 16: "#4CAF50", 19: "#FF5722"}

    for n_lay in layer_groups:
        subset = [r for r in all_results if r["n_layers"] == n_lay and not r["is_small"]]
        if not subset:
            continue
        spans_here = [r["span_cm"] for r in subset]
        forces_here = [r["max_force_N"] for r in subset]
        ax.scatter(
            spans_here, forces_here,
            marker=markers.get(n_lay, "o"),
            color=layer_colors.get(n_lay, "gray"),
            s=100, edgecolors="black", linewidth=0.8,
            label=f"{n_lay} layers", alpha=0.8, zorder=5,
        )

    ax.set_xlabel("Span Length (cm)", fontsize=12)
    ax.set_ylabel("Max Estimated Force (N)", fontsize=12)
    ax.set_title("Estimated Force vs Span Length (Regular Samples Only)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "force_vs_span.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Force vs span plot saved: {plot_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("FBG-Based Force Estimation Pipeline")
    print("Δλ → ε → σ → F  (purely from FBG sensor data)")
    print("=" * 70)
    print(f"\nMaterial parameters:")
    print(f"  E = {E_GPA} GPa")
    print(f"  pₑ = {PE}")
    print(f"  b = {BEAM_WIDTH_MM} mm (regular), {SMALL_SAMPLE_WIDTH_MM} mm (small)")
    print(f"  Layer thickness = {LAYER_THICKNESS_MM:.4f} mm")

    # Locate interrogator data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "interrogator-data")

    if not os.path.isdir(data_dir):
        print(f"\nERROR: Interrogator data directory not found: {data_dir}")
        sys.exit(1)

    # Find all interrogator files
    txt_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith("-interrogator.txt")
    ])

    if not txt_files:
        print(f"\nERROR: No interrogator files found in {data_dir}")
        sys.exit(1)

    print(f"\nFound {len(txt_files)} interrogator files.")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(script_dir, OUTPUT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Process all files
    all_results = []
    for filepath in txt_files:
        try:
            result = process_single_file(filepath, output_dir)
            if result is not None:
                # Generate per-file plot
                plot_single_file(result, output_dir)
                all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR processing {os.path.basename(filepath)}: {e}")
            continue

    # Generate summary CSV
    if all_results:
        summary_rows = []
        for r in all_results:
            summary_rows.append({
                "filename": r["filename"],
                "span_cm": r["span_cm"],
                "n_layers": r["n_layers"],
                "sample_number": r["sample_number"],
                "is_small": r["is_small"],
                "h_mm": r["h_mm"],
                "b_mm": r["b_mm"],
                "y_fbg_mm": r["y_fbg_mm"],
                "L_mm": r["L_mm"],
                "baseline_nm": r["baseline_nm"],
                "n_datapoints": r["n_datapoints"],
                "max_estimated_force_N": r["max_force_N"],
                "min_estimated_force_N": r["min_force_N"],
                "max_strain_microstrain": r["max_strain_ue"],
                "max_stress_MPa": r["max_stress_MPa"],
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(output_dir, "force_estimation_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n{'='*70}")
        print(f"Summary saved: {summary_csv}")
        print(f"{'='*70}")
        print(summary_df.to_string(index=False))

        # Generate summary plots
        plot_summary(all_results, output_dir)

    print(f"\n{'='*70}")
    print(f"Pipeline complete. Processed {len(all_results)}/{len(txt_files)} files.")
    print(f"All outputs in: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
