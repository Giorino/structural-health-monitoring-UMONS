#!/usr/bin/env python3

"""
Script to plot interrogator data from TXT files containing wavelength measurements over time.
Reads tab-separated files with timestamp, time, and wavelength columns.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def read_interrogator_data(file_path: str) -> pd.DataFrame:
    """Read interrogator data from tab-separated text file.
    
    Args:
        file_path: Path to the interrogator txt file
        
    Returns:
        DataFrame with parsed timestamps and wavelength data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Reading data from: {file_path}")
    
    # First, read just the header to understand the expected structure
    with open(file_path, 'r') as f:
        header_line = f.readline().strip()
    
    header_cols = [col.strip() for col in header_line.split('\t')]
    print(f"Header columns: {header_cols}")
    
    # Some files have variable number of columns per row, so we need to handle this
    # Read line by line and pad with NaN for missing columns
    rows = []
    max_cols = 0
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the maximum number of columns in any row
    for i, line in enumerate(lines[1:], start=2):  # Skip header
        if line.strip():
            cols = len(line.strip().split('\t'))
            max_cols = max(max_cols, cols)
    
    print(f"Maximum columns found in data: {max_cols}")
    
    # Generate column names for any extra columns
    extended_cols = header_cols.copy()
    if max_cols > len(header_cols):
        for i in range(len(header_cols), max_cols):
            extended_cols.append(f'WL {i-1}[nm]')  # Continue wavelength numbering
    
    # Now read the data with consistent column structure
    for i, line in enumerate(lines[1:], start=2):  # Skip header
        if line.strip():
            values = line.strip().split('\t')
            # Pad with empty strings if fewer columns
            while len(values) < max_cols:
                values.append('')
            rows.append(values)
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=extended_cols[:max_cols])
    
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Parse timestamp column to datetime
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    else:
        raise ValueError("No 'Timestamp' column found in data")
    
    # Ensure Time column is numeric
    if 'Time [s]' in df.columns:
        df['Time [s]'] = pd.to_numeric(df['Time [s]'], errors='coerce')
    
    # Convert wavelength columns to numeric
    wl_cols = [col for col in df.columns if col.startswith('WL ') and '[nm]' in col]
    for col in wl_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Found {len(wl_cols)} wavelength columns: {wl_cols}")
    
    return df


def identify_active_wavelength_columns(df: pd.DataFrame) -> List[str]:
    """Identify wavelength columns that contain actual data (not all NaN).
    
    Args:
        df: DataFrame containing the wavelength data
        
    Returns:
        List of column names that have actual wavelength measurements
    """
    wl_cols = [col for col in df.columns if col.startswith('WL ') and '[nm]' in col]
    active_cols = []
    
    for col in wl_cols:
        if not df[col].isna().all():
            non_na_count = df[col].notna().sum()
            print(f"Column {col}: {non_na_count} non-null values")
            active_cols.append(col)
    
    return active_cols


def plot_interrogator_data(df: pd.DataFrame, output_path: str, use_time_seconds: bool = False) -> str:
    """Create plots of wavelength measurements over time.
    
    Args:
        df: DataFrame containing the interrogator data
        output_path: Path where to save the plot
        use_time_seconds: If True, use Time[s] column instead of Timestamp for x-axis
        
    Returns:
        Path to the saved plot
    """
    # Identify active wavelength columns
    active_wl_cols = identify_active_wavelength_columns(df)
    
    if not active_wl_cols:
        raise ValueError("No active wavelength columns found in the data")
    
    # Set up the figure
    n_plots = len(active_wl_cols)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), 
                            sharex=True, constrained_layout=True)
    
    # Handle case where there's only one subplot
    if n_plots == 1:
        axes = [axes]
    
    # Choose x-axis data
    if use_time_seconds and 'Time [s]' in df.columns:
        x_data = df['Time [s]']
        x_label = 'Time (s)'
        x_col = 'Time [s]'
    else:
        x_data = df['Timestamp']
        x_label = 'Time'
        x_col = 'Timestamp'
    
    # Define colors for different wavelength channels
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot each active wavelength column
    for i, col in enumerate(active_wl_cols):
        ax = axes[i]
        
        # Filter out NaN values for cleaner plotting
        mask = df[col].notna() & x_data.notna()
        x_clean = x_data[mask]
        y_clean = df[col][mask]
        
        # Plot the data
        color = colors[i % len(colors)]
        ax.plot(x_clean, y_clean, color=color, linewidth=1.0, alpha=0.8)
        
        # Formatting
        ax.set_ylabel(f'{col}', fontsize=10)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        
        # Add statistics in legend
        if len(y_clean) > 0:
            mean_val = y_clean.mean()
            std_val = y_clean.std()
            ax.set_title(f'{col} (μ={mean_val:.4f} nm, σ={std_val:.4f} nm)', fontsize=9)
    
    # Set x-axis label on bottom plot
    axes[-1].set_xlabel(x_label, fontsize=12)
    
    # Format x-axis for datetime if using timestamps
    if not use_time_seconds:
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
    
    # Overall title
    filename = os.path.basename(output_path).replace('.png', '')
    fig.suptitle(f'Wavelength Measurements: {filename}', fontsize=14, fontweight='bold')
    
    # Save the plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    # Also display plot if running interactively
    if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        plt.show()
    else:
        plt.close(fig)
    
    return output_path


def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics for wavelength data.
    
    Args:
        df: DataFrame containing the interrogator data
        
    Returns:
        DataFrame with summary statistics
    """
    active_wl_cols = identify_active_wavelength_columns(df)
    
    stats = []
    for col in active_wl_cols:
        data = df[col].dropna()
        if len(data) > 0:
            stats.append({
                'Channel': col,
                'Count': len(data),
                'Mean (nm)': data.mean(),
                'Std (nm)': data.std(),
                'Min (nm)': data.min(),
                'Max (nm)': data.max(),
                'Range (nm)': data.max() - data.min()
            })
    
    return pd.DataFrame(stats)


def main():
    """Main function to handle command line arguments and execute plotting."""
    parser = argparse.ArgumentParser(
        description='Plot wavelength measurements from interrogator data files'
    )
    parser.add_argument('input_file', help='Path to the interrogator txt file')
    parser.add_argument('--output', '-o', help='Output path for the plot (default: auto-generated)')
    parser.add_argument('--use-time-seconds', action='store_true', 
                       help='Use Time[s] column instead of Timestamp for x-axis')
    parser.add_argument('--show-stats', action='store_true',
                       help='Print summary statistics')
    
    # If no arguments provided, use the file from user query
    if len(sys.argv) == 1:
        # Default to the user's specified file
        script_dir = Path(__file__).parent
        input_file = script_dir / 'interrogator-data' / '15cm-12layers-9-interrogator.txt'
        output_file = script_dir / 'interrogator_plot_15cm-12layers-9.png'
        use_time_seconds = False
        show_stats = True
    else:
        args = parser.parse_args()
        input_file = Path(args.input_file)
        
        # Auto-generate output filename if not provided
        if args.output:
            output_file = Path(args.output)
        else:
            stem = input_file.stem
            output_file = input_file.parent / f'plot_{stem}.png'
        
        use_time_seconds = args.use_time_seconds
        show_stats = args.show_stats
    
    # Read and plot data
    try:
        df = read_interrogator_data(str(input_file))
        
        if show_stats:
            print("\n" + "="*50)
            print("SUMMARY STATISTICS")
            print("="*50)
            stats_df = create_summary_statistics(df)
            print(stats_df.to_string(index=False))
            print("="*50)
        
        plot_path = plot_interrogator_data(df, str(output_file), use_time_seconds)
        print(f"\nPlot saved successfully: {plot_path}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()