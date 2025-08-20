import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from io import StringIO
import re

def preprocess_and_load_data(csv_path):
    """
    Preprocesses the malformed CSV data to handle multi-line headers and records,
    then loads it into a pandas DataFrame.
    """
    with open(csv_path, 'r') as f:
        content = f.read()

    # Join lines that are continuations of previous data lines.
    # This regex finds a newline that is NOT followed by a timestamp (e.g., '2025-')
    # and removes it, effectively merging the line with the previous one.
    content = re.sub(r'\n(?!\s*20\d{2}-\d{2}-\d{2}T)', '', content)

    # Split content into lines and remove any that are now empty
    lines = [line for line in content.split('\n') if line.strip()]

    # Find the starting index of the actual data
    data_start_index = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('20'):
            data_start_index = i
            break
    
    # Isolate the data content
    data_content = "\n".join(lines[data_start_index:])
    
    # Define column names manually for consistency
    column_names = ['Timestamp', 'Time [s]'] + [f'WL {i}[nm]' for i in range(1, 9)]

    # Read the cleaned data into a DataFrame
    df = pd.read_csv(StringIO(data_content), sep=r'\s+', names=column_names, index_col=False)
    
    # Convert data columns to numeric types, coercing errors
    for col in df.columns:
        if col != 'Timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def analyze_strain_data(csv_path, output_dir='strain_plots'):
    """
    Loads strain data, plots wavelength and strain over time for the first 3 channels.

    Args:
        csv_path (str): The path to the CSV file.
        output_dir (str): The directory to save the plots in.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        data = preprocess_and_load_data(csv_path)
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return

    base_filename = os.path.splitext(os.path.basename(csv_path))[0]
    time_col = 'Time [s]'
    channels = ['WL 1[nm]', 'WL 2[nm]', 'WL 3[nm]']
    
    # --- Plot Wavelength vs. Time ---
    plt.figure(figsize=(12, 6))
    for i, channel in enumerate(channels):
        if channel in data.columns:
            plt.plot(data[time_col], data[channel], label=f'Channel {i+1}')
    
    plt.xlabel('Time [s]')
    plt.ylabel('Wavelength [nm]')
    plt.title(f'Wavelength vs. Time for {base_filename}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{base_filename}_wavelength.png'))
    plt.close()

    # --- Plot Delta Wavelength vs. Time ---
    plt.figure(figsize=(12, 6))
    for i, channel in enumerate(channels):
        if channel in data.columns:
            wavelengths = data[channel].dropna()
            if not wavelengths.empty:
                lambda_0 = wavelengths.iloc[0]
                delta_lambda = wavelengths - lambda_0
                plt.plot(data.loc[wavelengths.index, time_col], delta_lambda, label=f'Channel {i+1} Δλ')

    plt.xlabel('Time [s]')
    plt.ylabel('Delta Wavelength (Δλ) [nm]')
    plt.title(f'Delta Wavelength vs. Time for {base_filename}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{base_filename}_delta_wavelength.png'))
    plt.close()

    # --- Calculate and Plot Strain vs. Time ---
    p_e = 0.22  # Photo-elastic coefficient

    plt.figure(figsize=(12, 6))
    for i, channel in enumerate(channels):
        if channel in data.columns:
            wavelengths = data[channel].dropna()
            if not wavelengths.empty:
                lambda_0 = wavelengths.iloc[0]
                delta_lambda = wavelengths - lambda_0
                strain = (delta_lambda / lambda_0) / (1 - p_e)
                plt.plot(data.loc[wavelengths.index, time_col], strain * 1e6, label=f'Channel {i+1} Strain')

    plt.xlabel('Time [s]')
    plt.ylabel('Strain [με]')
    plt.title(f'Strain vs. Time for {base_filename}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{base_filename}_strain.png'))
    plt.close()

def main():
    """Main function to process all strain data files."""
    strain_data_dir = 'strain_data'
    csv_files = glob.glob(os.path.join(strain_data_dir, 'STRAIN_BATCH_*.csv'))

    if not csv_files:
        print(f"No 'STRAIN_BATCH_*.csv' files found in {strain_data_dir}.")
        return

    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        analyze_strain_data(csv_file)
    
    print("Processing complete.")

if __name__ == '__main__':
    main()
