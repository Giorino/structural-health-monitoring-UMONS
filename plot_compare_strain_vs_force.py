#!/usr/bin/env python3

from __future__ import annotations

import os
import re
from typing import List, Optional, Dict, Tuple
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from compute_mechanical_strain import compute_mechanical_strain as compute_mech


def find_latest_output_directory(base_dir: Optional[str] = None) -> Optional[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir is None:
        resolved_base = os.path.join(script_dir, "output")
    else:
        resolved_base = base_dir if os.path.isabs(base_dir) else os.path.join(script_dir, str(base_dir))

    if not os.path.isdir(resolved_base):
        return None

    output_dirs = [
        os.path.join(resolved_base, d)
        for d in os.listdir(resolved_base)
        if os.path.isdir(os.path.join(resolved_base, d))
    ]
    if not output_dirs:
        return None
    return max(output_dirs, key=os.path.getmtime)


def list_merged_csvs(directory: str) -> List[str]:
    files = sorted([
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.endswith('.csv') and f.startswith('merged_')
    ])
    if not files:
        files = sorted([
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith('.csv')
        ])
    return files


def _derive_base_key_from_merged_filename(basename: str) -> Optional[str]:
    base = os.path.splitext(os.path.basename(basename))[0]
    if base.startswith("merged_"):
        base = base[len("merged_"):]
    base = re.sub(r"_\d{8}(_\d{4,6})?$", "", base)
    return base or None


def load_and_merge_csvs(csv_paths: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        raise RuntimeError("No CSV files were loaded.")
    merged = pd.concat(frames, axis=0, ignore_index=True)
    return merged


def _parse_distance_from_name(name: str) -> Optional[float]:
    base = os.path.splitext(os.path.basename(str(name)))[0]
    if base.startswith("merged_"):
        base = base[len("merged_"):]
    base = re.sub(r"_\d{8}(_\d{4,6})?$", "", base)
    m = re.search(r"(?P<dist>\d+)cm-?(?P<layers>\d+)layers", base)
    if not m:
        return None
    try:
        return float(m.group("dist"))
    except Exception:
        return None


def _plot_by_distance_small_multiples(df: pd.DataFrame, latest_dir: str) -> str:
    # Gather distances per source file
    file_to_dist = {f: _parse_distance_from_name(f) for f in df["source_file"].dropna().unique()}
    dists = sorted({d for d in file_to_dist.values() if d is not None})
    if not dists:
        return ""

    # Calculate global x-axis limits (force range) for consistent scaling across all subplots
    global_force = pd.to_numeric(df.get("Force (N)"), errors="coerce")
    global_force_clean = global_force.dropna()
    if len(global_force_clean) > 0:
        global_x_min = float(global_force_clean.min())
        global_x_max = float(global_force_clean.max())
        # Add 5% padding to the global range
        x_range = global_x_max - global_x_min
        if x_range > 0:
            x_pad = 0.05 * x_range
            global_x_min -= x_pad
            global_x_max += x_pad
    else:
        global_x_min, global_x_max = 0, 100  # fallback

    # Layout
    n = len(dists)
    ncols = 3 if n >= 3 else n
    nrows = int(math.ceil(n / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols + 2, 4.2 * nrows + 1), squeeze=False)
    cmap = plt.get_cmap("tab10", n)

    for idx, dist in enumerate(dists):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        color = cmap(idx)
        files_here = [f for f, d in file_to_dist.items() if d == dist]
        sub = df[df["source_file"].isin(files_here)].copy()
        force = pd.to_numeric(sub.get("Force (N)"), errors="coerce")
        fbg = pd.to_numeric(sub.get("fbg_direct_strain [\u03bcu\u03b5]"), errors="coerce")
        mech = pd.to_numeric(sub.get("mechanical_strain [\u03bcu\u03b5]"), errors="coerce")
        mask = ~(force.isna() | fbg.isna())

        # Scatter FBG
        ax.scatter(force[mask], fbg[mask], s=14, alpha=0.75, color=color, label="FBG Δλ/λ0")

        # FBG linear fit
        if mask.sum() >= 2:
            x = force[mask].values
            y = fbg[mask].values
            slope, intercept = np.polyfit(x, y, deg=1)
            xline = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 50)
            yline = slope * xline + intercept
            # R^2
            yhat = slope * x + intercept
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            ax.plot(xline, yline, color=color, linewidth=1.8, label=f"FBG fit (R²={r2:.2f})")

        # Mechanical line fit from recomputed mechanical strain (reflects current E in compute_mechanical_strain)
        mask_mech = ~(force.isna() | mech.isna())
        if mask_mech.sum() >= 2:
            x_m = force[mask_mech].values
            y_m = mech[mask_mech].values
            slope_m, intercept_m = np.polyfit(x_m, y_m, deg=1)
            fmin, fmax = float(np.nanmin(x_m)), float(np.nanmax(x_m))
            xline = np.array([fmin, fmax])
            yline = slope_m * xline + intercept_m
            # Difference metrics using paired samples
            mask_both = ~(force.isna() | fbg.isna() | mech.isna())
            x_pts = force[mask_both].values.astype(float)
            y_fbg_pts = fbg[mask_both].values.astype(float)
            y_mech_pts = mech[mask_both].values.astype(float)
            diffs = y_mech_pts - y_fbg_pts
            mean_diff = float(np.nanmean(diffs)) if diffs.size > 0 else float('nan')
            denom = np.maximum(np.abs(y_fbg_pts), 1e-9)
            mape = float(np.nanmean(np.abs(diffs) / denom) * 100.0) if diffs.size > 0 else float('nan')
            sign = "+" if mean_diff >= 0 else ""
            diff_label_extra = f", Δmean={sign}{mean_diff:.0f} με, MAPE={mape:.1f}%"
            ax.plot(xline, yline, color=color, linestyle="--", linewidth=1.6,
                    label=f"Mechanical (P·L·y/4EI){diff_label_extra}")

        ax.set_title(f"{int(dist)} cm span")
        ax.set_xlabel("Force (N)")
        ax.set_ylabel("Strain [\u03bcu\u03b5]")
        ax.grid(True, linestyle=":", alpha=0.6)
        
        # Force regular number formatting instead of scientific notation
        ax.ticklabel_format(style='plain', axis='both')
        
        # Set consistent x-axis limits across all subplots
        ax.set_xlim(global_x_min, global_x_max)
        
        # Dynamic y-axis scaling for each subplot
        all_y_values = []
        if mask.any():
            all_y_values.extend(fbg[mask].dropna())
        if mask_mech.any():
            all_y_values.extend(mech[mask_mech].dropna())
        
        if all_y_values:
            y_min_local = float(min(all_y_values))
            y_max_local = float(max(all_y_values))
            # Add 10% padding to the range
            y_range = y_max_local - y_min_local
            if y_range > 0:
                pad_local = 0.1 * y_range
                ax.set_ylim(y_min_local - pad_local, y_max_local + pad_local)
            else:
                # If all values are the same, center around the value with some padding
                ax.set_ylim(y_min_local - 50, y_max_local + 50)
        
        ax.legend(fontsize=7, frameon=True, framealpha=0.85, loc="upper left")

    # Hide any empty axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis('off')

    fig.suptitle("Strain vs Force by Span Length", fontsize=14)
    out_path = os.path.join(latest_dir, "strain_vs_force_by_distance.png")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_overlay_all_spans(df: pd.DataFrame, latest_dir: str) -> str:
    """Create a single overlay plot showing all span lengths on the same axes for parallelism comparison."""
    # Gather distances per source file
    file_to_dist = {f: _parse_distance_from_name(f) for f in df["source_file"].dropna().unique()}
    dists = sorted({d for d in file_to_dist.values() if d is not None})
    if not dists:
        return ""

    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Use a colormap that provides good contrast for different spans
    cmap = plt.get_cmap("tab10", len(dists))
    
    # Track all force and strain values for global scaling
    all_forces = []
    all_fbg_strains = []
    all_mech_strains = []
    
    for idx, dist in enumerate(dists):
        color = cmap(idx)
        files_here = [f for f, d in file_to_dist.items() if d == dist]
        sub = df[df["source_file"].isin(files_here)].copy()
        force = pd.to_numeric(sub.get("Force (N)"), errors="coerce")
        fbg = pd.to_numeric(sub.get("fbg_direct_strain [\u03bcu\u03b5]"), errors="coerce")
        mech = pd.to_numeric(sub.get("mechanical_strain [\u03bcu\u03b5]"), errors="coerce")
        
        # FBG data
        mask_fbg = ~(force.isna() | fbg.isna())
        if mask_fbg.any():
            # Scatter plot for FBG measurements
            ax.scatter(force[mask_fbg], fbg[mask_fbg], s=12, alpha=0.6, color=color, 
                      label=f"FBG {int(dist)}cm", marker='o')
            
            # FBG linear fit
            if mask_fbg.sum() >= 2:
                x_fbg = force[mask_fbg].values
                y_fbg = fbg[mask_fbg].values
                slope_fbg, intercept_fbg = np.polyfit(x_fbg, y_fbg, deg=1)
                xline_fbg = np.linspace(float(np.nanmin(x_fbg)), float(np.nanmax(x_fbg)), 50)
                yline_fbg = slope_fbg * xline_fbg + intercept_fbg
                # R^2 calculation
                yhat_fbg = slope_fbg * x_fbg + intercept_fbg
                ss_res_fbg = float(np.sum((y_fbg - yhat_fbg) ** 2))
                ss_tot_fbg = float(np.sum((y_fbg - np.mean(y_fbg)) ** 2))
                r2_fbg = 1.0 - ss_res_fbg / ss_tot_fbg if ss_tot_fbg > 0 else 0.0
                ax.plot(xline_fbg, yline_fbg, color=color, linewidth=2.0, linestyle='-',
                       label=f"FBG {int(dist)}cm fit (R²={r2_fbg:.2f})")
                
                all_forces.extend(x_fbg)
                all_fbg_strains.extend(y_fbg)
        
        # Mechanical data
        mask_mech = ~(force.isna() | mech.isna())
        if mask_mech.any():
            # Use lighter shade and different marker for mechanical
            lighter_color = (*color[:3], 0.7) if len(color) == 4 else color
            ax.scatter(force[mask_mech], mech[mask_mech], s=8, alpha=0.4, color=lighter_color,
                      label=f"Mech {int(dist)}cm", marker='s')
            
            # Mechanical linear fit
            if mask_mech.sum() >= 2:
                x_mech = force[mask_mech].values
                y_mech = mech[mask_mech].values
                slope_mech, intercept_mech = np.polyfit(x_mech, y_mech, deg=1)
                xline_mech = np.linspace(float(np.nanmin(x_mech)), float(np.nanmax(x_mech)), 50)
                yline_mech = slope_mech * xline_mech + intercept_mech
                ax.plot(xline_mech, yline_mech, color=lighter_color, linewidth=1.5, linestyle='--',
                       label=f"Mech {int(dist)}cm fit")
                
                all_mech_strains.extend(y_mech)
    
    # Set labels and formatting
    ax.set_xlabel("Force (N)", fontsize=12)
    ax.set_ylabel("Strain [μɛ]", fontsize=12)
    ax.set_title("Strain vs Force - All Span Lengths Overlay\n(Parallel Behavior Comparison)", fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.4)
    
    # Force regular number formatting instead of scientific notation
    ax.ticklabel_format(style='plain', axis='both')
    
    # Set axis limits with some padding
    if all_forces:
        force_min, force_max = min(all_forces), max(all_forces)
        force_range = force_max - force_min
        if force_range > 0:
            force_pad = 0.05 * force_range
            ax.set_xlim(force_min - force_pad, force_max + force_pad)
    
    if all_fbg_strains or all_mech_strains:
        all_strains = []
        if all_fbg_strains:
            all_strains.extend(all_fbg_strains)
        if all_mech_strains:
            all_strains.extend(all_mech_strains)
        
        strain_min, strain_max = min(all_strains), max(all_strains)
        strain_range = strain_max - strain_min
        if strain_range > 0:
            strain_pad = 0.1 * strain_range
            ax.set_ylim(strain_min - strain_pad, strain_max + strain_pad)
    
    # Create a custom legend with better organization
    handles, labels = ax.get_legend_handles_labels()
    
    # Organize legend: FBG scatter, FBG fits, Mechanical scatter, Mechanical fits
    fbg_scatter_handles = [h for h, l in zip(handles, labels) if "FBG" in l and "fit" not in l]
    fbg_fit_handles = [h for h, l in zip(handles, labels) if "FBG" in l and "fit" in l]
    mech_scatter_handles = [h for h, l in zip(handles, labels) if "Mech" in l and "fit" not in l]
    mech_fit_handles = [h for h, l in zip(handles, labels) if "Mech" in l and "fit" in l]
    
    fbg_scatter_labels = [l for l in labels if "FBG" in l and "fit" not in l]
    fbg_fit_labels = [l for l in labels if "FBG" in l and "fit" in l]
    mech_scatter_labels = [l for l in labels if "Mech" in l and "fit" not in l]
    mech_fit_labels = [l for l in labels if "Mech" in l and "fit" in l]
    
    # Create two-column legend
    legend1 = ax.legend(fbg_scatter_handles + fbg_fit_handles, 
                       fbg_scatter_labels + fbg_fit_labels, 
                       loc='upper left', fontsize=9, frameon=True, framealpha=0.9,
                       title="FBG Measurements", title_fontsize=10)
    legend1.get_frame().set_linewidth(0.8)
    
    # Add second legend for mechanical
    if mech_scatter_handles or mech_fit_handles:
        legend2 = ax.legend(mech_scatter_handles + mech_fit_handles, 
                           mech_scatter_labels + mech_fit_labels, 
                           loc='lower right', fontsize=9, frameon=True, framealpha=0.9,
                           title="Mechanical Theory", title_fontsize=10)
        legend2.get_frame().set_linewidth(0.8)
        ax.add_artist(legend1)  # Add back the first legend
    
    # Save the plot
    out_path = os.path.join(latest_dir, "strain_vs_force_overlay_all_spans.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ============================================================================
# NEW CROSS-CHECK ANALYSES
# ============================================================================

def extract_slope_and_stats(force: np.ndarray, strain: np.ndarray) -> Tuple[float, float, float]:
    """Extract slope, intercept, and R² from linear fit."""
    if len(force) < 2 or len(strain) < 2:
        return np.nan, np.nan, np.nan
    
    # Remove any NaN/inf values
    mask = np.isfinite(force) & np.isfinite(strain)
    if np.sum(mask) < 2:
        return np.nan, np.nan, np.nan
    
    x_clean = force[mask]
    y_clean = strain[mask]
    
    # Linear fit
    slope, intercept = np.polyfit(x_clean, y_clean, deg=1)
    
    # R² calculation
    y_pred = slope * x_clean + intercept
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return float(slope), float(intercept), float(r2)


def analyze_slope_vs_span(df: pd.DataFrame, latest_dir: str) -> Tuple[str, pd.DataFrame]:
    """1. Slope vs. Span Length Test - Check if k_L scales linearly with span."""
    # Get span lengths and group data by span
    file_to_dist = {f: _parse_distance_from_name(f) for f in df["source_file"].dropna().unique()}
    dists = sorted({d for d in file_to_dist.values() if d is not None})
    
    if len(dists) < 3:
        print("Warning: Need at least 3 different span lengths for slope analysis")
        return "", pd.DataFrame()
    
    # Extract slopes for each span
    slope_data = []
    
    for dist in dists:
        files_here = [f for f, d in file_to_dist.items() if d == dist]
        sub = df[df["source_file"].isin(files_here)].copy()
        
        force = pd.to_numeric(sub.get("Force (N)"), errors="coerce")
        fbg_strain = pd.to_numeric(sub.get("fbg_direct_strain [\u03bcu\u03b5]"), errors="coerce")
        
        # Clean data and convert to numpy arrays
        mask = ~(force.isna() | fbg_strain.isna())
        if not mask.any():
            continue
            
        force_clean = force[mask].values
        fbg_strain_clean = fbg_strain[mask].values
        
        # Get slope k_L = dε/dF
        slope, intercept, r2 = extract_slope_and_stats(force_clean, fbg_strain_clean)
        
        slope_data.append({
            'span_length_cm': dist,
            'span_length_m': dist / 100.0,
            'slope_k_L': slope,  # με/N
            'intercept': intercept,
            'r2': r2,
            'n_points': len(sub)
        })
    
    slope_df = pd.DataFrame(slope_data)
    
    # Plot k_L vs L
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Slope vs span length
    valid_mask = np.isfinite(slope_df['slope_k_L'])
    if valid_mask.sum() >= 2:
        # Separate 27cm span from others
        outlier_mask = slope_df['span_length_cm'] == 27.0
        normal_mask = valid_mask & ~outlier_mask
        
        # Plot normal spans in blue
        if normal_mask.sum() > 0:
            x_normal = slope_df.loc[normal_mask, 'span_length_m'].values
            y_normal = slope_df.loc[normal_mask, 'slope_k_L'].values
            ax1.scatter(x_normal, y_normal, s=80, alpha=0.8, color='blue', edgecolors='navy', label='Normal spans')
            
            # Fit regression through origin using only normal spans
            if len(x_normal) >= 2:
                a_origin = np.sum(x_normal * y_normal) / np.sum(x_normal ** 2)
                x_line = np.linspace(0, max(slope_df.loc[valid_mask, 'span_length_m']) * 1.1, 100)
                y_line = a_origin * x_line
                ax1.plot(x_line, y_line, 'r--', linewidth=2, 
                        label=f'k_L = {a_origin:.2e}·L (normal spans only)')
                
                # R² for fit through origin (normal spans only)
                y_pred_origin = a_origin * x_normal
                ss_res_origin = np.sum((y_normal - y_pred_origin) ** 2)
                ss_tot_origin = np.sum((y_normal - np.mean(y_normal)) ** 2)
                r2_origin = 1.0 - ss_res_origin / ss_tot_origin if ss_tot_origin > 0 else 0.0
                
                ax1.text(0.05, 0.95, f'R² = {r2_origin:.3f} (excl. 27cm)', transform=ax1.transAxes, 
                        fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # Plot 27cm span in light grey
        if (valid_mask & outlier_mask).sum() > 0:
            x_outlier = slope_df.loc[valid_mask & outlier_mask, 'span_length_m'].values
            y_outlier = slope_df.loc[valid_mask & outlier_mask, 'slope_k_L'].values
            ax1.scatter(x_outlier, y_outlier, s=80, alpha=0.6, color='lightgray', edgecolors='gray', 
                       label='27cm span (excluded)')
    
    ax1.set_xlabel('Span Length L (m)')
    ax1.set_ylabel('Slope k_L (με/N)')
    ax1.set_title('FBG Strain-Force Slope vs Span Length\n(Theory predicts linear relationship)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right plot: R² values for each span's strain-force fit
    colors = ['lightgray' if span == 27.0 else 'green' for span in slope_df['span_length_cm']]
    edgecolors = ['gray' if span == 27.0 else 'darkgreen' for span in slope_df['span_length_cm']]
    
    ax2.bar(slope_df['span_length_cm'], slope_df['r2'], alpha=0.7, color=colors, edgecolor=edgecolors)
    ax2.set_xlabel('Span Length (cm)')
    ax2.set_ylabel('R² for Strain vs Force Fit')
    ax2.set_title('Quality of Linear Fits by Span')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Add R² values as text
    for _, row in slope_df.iterrows():
        if not np.isnan(row['r2']):
            ax2.text(row['span_length_cm'], row['r2'] + 0.02, f'{row["r2"]:.3f}', 
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    out_path = os.path.join(latest_dir, "slope_vs_span_analysis.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    
    return out_path, slope_df

    # EI cross-validation removed per user request


def generate_summary_table(slope_df: pd.DataFrame, latest_dir: str) -> str:
    """Generate summary table for slope vs span only (27cm shown but excluded from fits)."""
    summary = slope_df.copy() if not slope_df.empty else pd.DataFrame()
    if summary.empty:
        return ""

    # Keep only slope-related columns
    keep_cols = ['span_length_cm', 'span_length_m', 'slope_k_L', 'r2', 'n_points']
    summary = summary[keep_cols]

    # Rename for presentation
    summary = summary.rename(columns={
        'span_length_cm': 'Span (cm)',
        'span_length_m': 'Span (m)',
        'slope_k_L': 'k_L (με/N)',
        'r2': 'R² (ε-F)',
        'n_points': 'Data Points',
    })

    # Format
    if 'Span (m)' in summary.columns:
        summary['Span (m)'] = summary['Span (m)'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else 'N/A')
    if 'k_L (με/N)' in summary.columns:
        summary['k_L (με/N)'] = summary['k_L (με/N)'].apply(lambda x: f"{x:.2e}" if pd.notna(x) else 'N/A')
    if 'R² (ε-F)' in summary.columns:
        summary['R² (ε-F)'] = summary['R² (ε-F)'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else 'N/A')

    # Save and print
    csv_path = os.path.join(latest_dir, "cross_check_analysis_summary.csv")
    summary.to_csv(csv_path, index=False)

    print("\n" + "="*60)
    print("SLOPE VS SPAN SUMMARY")
    print("="*60)
    print("Note: 27cm span excluded from linear fits but shown in grey")
    print("="*60)
    print(summary.to_string(index=False))
    print("="*60)
    print(f"\nDetailed results saved to: {csv_path}")

    return csv_path


def main(base_output_dir: Optional[str] = None) -> None:
    latest_dir = find_latest_output_directory(base_output_dir)
    if not latest_dir:
        raise RuntimeError("No output directories found under the provided base directory.")

    csv_files = list_merged_csvs(latest_dir)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in: {latest_dir}")

    df = load_and_merge_csvs(csv_files)

    # Always compute FBG and mechanical strain inline to reflect current constants and avoid stale CSVs
    from compute_fbg_direct_strain import compute_fbg_direct_microstrain_for_df
    fbg_series, _ = compute_fbg_direct_microstrain_for_df(df)
    df["fbg_direct_strain [\u03bcu\u03b5]"] = fbg_series

    df["mechanical_strain [\u03bcu\u03b5]"] = compute_mech(df)

    # Inputs
    force = pd.to_numeric(df.get("Force (N)"), errors="coerce")
    fbg = pd.to_numeric(df.get("fbg_direct_strain [\u03bcu\u03b5]"), errors="coerce")
    mech = pd.to_numeric(df.get("mechanical_strain [\u03bcu\u03b5]"), errors="coerce")

    # Build a consistent color map keyed by base file key so FBG vs Mechanical share colors
    unique_files = list(df["source_file"].dropna().unique())
    base_keys = [(_derive_base_key_from_merged_filename(str(f)) or os.path.splitext(str(f))[0]) for f in unique_files]
    # Preserve order and uniqueness
    seen = set()
    ordered_keys: List[str] = []
    for k in base_keys:
        if k not in seen:
            seen.add(k)
            ordered_keys.append(k)
    cmap = plt.get_cmap('tab20', max(2, len(ordered_keys)))
    color_by_key = {k: cmap(i) for i, k in enumerate(ordered_keys)}

    # Create the faceted plot by distance 
    try:
        out_facet = _plot_by_distance_small_multiples(df, latest_dir)
        if out_facet:
            print(f"Saved: {out_facet}")
    except Exception as e:
        print(f"Skipped faceted plot due to error: {e}")

    # Create the overlay plot showing all spans on single axes for parallelism analysis
    try:
        out_overlay = _plot_overlay_all_spans(df, latest_dir)
        if out_overlay:
            print(f"Saved: {out_overlay}")
    except Exception as e:
        print(f"Skipped overlay plot due to error: {e}")

    # ============================================================================
    # NEW CROSS-CHECK ANALYSES
    # ============================================================================
    
    print("\n" + "="*60)
    print("PERFORMING CROSS-CHECK ANALYSES")
    print("="*60)
    print("Note: 27cm span treated as outlier (shown in grey, excluded from fits)")
    print("="*60)
    
    # Initialize result dataframes
    slope_df = pd.DataFrame()
    
    # 1. Slope vs. Span Length Test
    try:
        print("1. Analyzing slope vs span length...")
        out_slope, slope_df = analyze_slope_vs_span(df, latest_dir)
        if out_slope:
            print(f"   Saved: {os.path.basename(out_slope)}")
    except Exception as e:
        print(f"   Skipped slope analysis due to error: {e}")
    
    # 2. Generate Summary Table
    try:
        print("2. Generating summary table...")
        summary_csv = generate_summary_table(slope_df, latest_dir)
        if summary_csv:
            print(f"   Saved: {os.path.basename(summary_csv)}")
    except Exception as e:
        print(f"   Failed to generate summary table: {e}")
    
    print("="*60)
    print("CROSS-CHECK ANALYSES COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()


