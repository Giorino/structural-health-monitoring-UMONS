#!/usr/bin/env python3
"""
PyCaret-based analysis of the structural health monitoring dataset.
"""

import os
import glob
import pandas as pd
import numpy as np
from pycaret.classification import *

def load_and_preprocess_data(data_dir="./"):
    """Load CSV files and preprocess data for PyCaret analysis."""
    
    # Look for merged CSV files in current directory first
    csv_files = glob.glob(os.path.join(data_dir, "merged_*.csv"))
    
    # Function to find latest directory in output folder
    def find_latest_output_dir(output_base_dir):
        if not os.path.exists(output_base_dir):
            return None
            
        subdirs = []
        for item in os.listdir(output_base_dir):
            subdir_path = os.path.join(output_base_dir, item)
            if os.path.isdir(subdir_path):
                subdirs.append((item, subdir_path))
        
        if not subdirs:
            return None
            
        # Sort by directory name (assumes timestamp format like 20250825_124924)
        latest_subdir = sorted(subdirs, key=lambda x: x[0], reverse=True)[0]
        return latest_subdir[1]
    
    # Search in output directory - only the latest one
    output_dir = os.path.join(data_dir, "output")
    latest_dir = find_latest_output_dir(output_dir)
    if latest_dir:
        latest_csv_files = glob.glob(os.path.join(latest_dir, "merged_*.csv"))
        csv_files.extend(latest_csv_files)
        print(f"Using latest output directory: {os.path.basename(latest_dir)}")
        print(f"Found {len(latest_csv_files)} CSV files in latest directory")
    
    if not csv_files:
        print("No merged CSV files found.")
        return None
    
    print(f"Found {len(csv_files)} CSV files")
    
    all_dfs = []
    
    for file_path in csv_files:
        print(f"Processing {file_path}")
        try:
            df = pd.read_csv(file_path)
            
            filename = os.path.basename(file_path)
            is_small_sample = "-s" in filename
            
            base_required_cols = ['group_index', 'repetition_index', 'WL_ch2', 'WL_ch2_std', 
                               'Force (N)', 'Displacement (mm)', 'Crack']
            
            if is_small_sample:
                if 'Air Pressure (bar)' not in df.columns:
                    df['Air Pressure (bar)'] = 1.0
            
            required_cols = base_required_cols + ['Air Pressure (bar)']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols} in {file_path}")
                continue

            # Calculate baseline and delta values
            df_out = pd.DataFrame()
            groups = df.groupby(['group_index', 'repetition_index'])
            for (group_idx, rep_idx), group_df in groups:
                 if len(group_df) < 1:
                    continue
                 group_df = group_df.copy()
                 baseline_wl = group_df['WL_ch2'].iloc[0]
                 group_df['delta_wl_ch2'] = group_df['WL_ch2'] - baseline_wl
                 
                 # Calculate rates
                 group_df['delta_wl_rate'] = group_df['delta_wl_ch2'].diff().fillna(0)
                 group_df['delta_disp_rate'] = group_df['Displacement (mm)'].diff().fillna(0)
                 
                 group_df['is_small_sample'] = 1 if is_small_sample else 0
                 df_out = pd.concat([df_out, group_df])

            all_dfs.append(df_out)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not all_dfs:
        print("No valid data found in CSV files.")
        return None
    
    # Combine all dataframes
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Clean up the Crack column
    final_df['Crack'] = final_df['Crack'].fillna(0).astype(int)
    
    return final_df

def main():
    """Main execution pipeline for PyCaret analysis."""
    print("Starting PyCaret analysis...")
    
    # 1. Load data
    dataset = load_and_preprocess_data()
    
    if dataset is None:
        print("Data loading failed. Exiting.")
        return
        
    print("Data loaded successfully.")
    print(dataset.head())
    print(f"Dataset shape: {dataset.shape}")
    print("\nCrack distribution:")
    print(dataset['Crack'].value_counts())
    
    # 2. Setup PyCaret
    features = ['WL_ch2', 'WL_ch2_std', 'delta_wl_ch2', 'Force (N)', 
                'Displacement (mm)', 'Air Pressure (bar)', 'delta_wl_rate', 
                'delta_disp_rate', 'is_small_sample']
    target = 'Crack'
    
    # Select only the columns to be used for the analysis
    analysis_df = dataset[features + [target]]

    print("\nSetting up PyCaret Classification experiment...")
    s = setup(data=analysis_df, target=target, session_id=123,
              normalize=True, transformation=True, train_size=0.8)
              
    # 3. Compare models
    print("\nComparing models...")
    best_model = compare_models()
    
    print("\nBest model found:")
    print(best_model)
    
    # 4. Save results
    print("\nSaving experiment results...")
    save_model(best_model, 'pycaret_best_model')
    print("Best model saved as pycaret_best_model.pkl")

    print("\n" + "="*80)
    print("DETAILED 10-FOLD BREAKDOWN (Standard Deviation Calculation)")
    print("="*80)
    # create_model prints the 10-fold grid by default
    rf = create_model('rf')

if __name__ == "__main__":
    main()
