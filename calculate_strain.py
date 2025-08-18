import pandas as pd
import os
import glob
import re

def parse_filename_details(filename):
    """Parses distance and layers from the filename."""
    match = re.search(r'(\d+)cm-(\d+)layers', filename)
    if match:
        return float(match.group(1)), int(match.group(2))
    return None, None

def calculate_mechanical_strain(df, distance_cm, num_layers=12):
    """Calculates mechanical strain and adds it to the DataFrame."""
    # Constants
    E_PLA = 3.5e9  # Young's modulus for PLA in Pa
    b = 0.025  # width of the beam in meters
    h = num_layers * 0.0002  # height of the beam (200 microns per layer)
    L = distance_cm / 100  # span length in meters
    
    # Moment of inertia for a rectangular cross-section
    I = (b * h**3) / 12
    
    # Calculate bending stress (sigma = M*c/I)
    # M = F * L / 4 (for a point load at the center)
    # c = h / 2
    # sigma = (F * L / 4) * (h / 2) / I
    force_n = df['Force (N)']
    stress = (force_n * L * h) / (8 * I)
    
    # Mechanical strain (epsilon = sigma / E)
    df['mechanical_strain'] = stress / E_PLA
    df['Distance (cm)'] = distance_cm
    df['Layers'] = num_layers
    return df

def process_directory(directory_path):
    """
    Processes all merged CSV files in a directory to calculate strain.
    """
    csv_files = glob.glob(os.path.join(directory_path, 'merged_*.csv'))
    
    if not csv_files:
        print(f"No 'merged_*.csv' files found in {directory_path}.")
        return

    all_strains = []

    for f in csv_files:
        filename = os.path.basename(f)
        distance, layers = parse_filename_details(filename)
        
        if distance is None:
            print(f"Could not parse distance/layers from {filename}, skipping.")
            continue
            
        df = pd.read_csv(f)
        df_strain = calculate_mechanical_strain(df, distance, layers)
        all_strains.append(df_strain)

    if all_strains:
        combined_df = pd.concat(all_strains, ignore_index=True)
        output_path = os.path.join(directory_path, 'strain_analysis_results.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"Strain analysis complete. Results saved to {output_path}")

def find_latest_output_folder(base_path='output'):
    """Finds the most recently created folder in the base_path."""
    list_of_dirs = glob.glob(os.path.join(base_path, '*'))
    if not list_of_dirs:
        return None
    latest_dir = max(list_of_dirs, key=os.path.getctime)
    return latest_dir

if __name__ == '__main__':
    latest_output_dir = find_latest_output_folder()
    if latest_output_dir:
        process_directory(latest_output_dir)
    else:
        print("No output directory found.")
