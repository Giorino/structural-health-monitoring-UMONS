import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import os
import glob

def find_latest_output_folder(base_path='output'):
    """Finds the most recently created folder in the base_path."""
    list_of_dirs = glob.glob(os.path.join(base_path, '*'))
    if not list_of_dirs:
        return None
    latest_dir = max(list_of_dirs, key=os.path.getctime)
    return latest_dir

def plot_strain_vs_wavelength_by_distance(csv_path, output_dir='strain_wavelength_analysis_plots'):
    """
    Loads data and creates a separate Bragg Wavelength vs. Strain plot with subplots 
    for each channel, for each span distance.

    Args:
        csv_path (str): The path to the CSV file.
        output_dir (str): The directory to save the plots in.
    """
    # Load the data
    data = pd.read_csv(csv_path)
    
    distances = sorted(data['Distance (cm)'].unique())
    channels = ['WL_ch1', 'WL_ch2', 'WL_ch3']
    colors = ['r', 'g', 'b']

    for distance in distances:
        fig, axes = plt.subplots(len(channels), 1, figsize=(10, 15), sharex=True)
        fig.suptitle(f'Bragg Wavelength vs. Strain at {distance} cm Span', fontsize=16)

        subset = data[data['Distance (cm)'] == distance]
        
        for i, (channel, color) in enumerate(zip(channels, colors)):
            ax = axes[i]
            strain = subset['mechanical_strain'] * 1e6  # to microstrain
            wavelength = subset[channel]
            
            valid_data = ~np.isnan(strain) & ~np.isnan(wavelength)
            strain = strain[valid_data]
            wavelength = wavelength[valid_data]

            if len(strain) > 1:
                slope, intercept, r_value, p_value, std_err = linregress(strain, wavelength)
                line = slope * strain + intercept
                
                ax.scatter(strain, wavelength, label=f'{channel} data', s=15, color=color, alpha=0.7)
                ax.plot(strain, line, color=color, label=f'{channel} fit (R²={r_value**2:.4f})')
                
                # Zoom in on the y-axis
                y_min, y_max = wavelength.min(), wavelength.max()
                y_padding = (y_max - y_min) * 0.05  # Zoom in more
                ax.set_ylim(y_min - y_padding, y_max + y_padding)

            ax.set_ylabel('Bragg Wavelength [nm]')
            ax.set_title(f'Channel {i+1}')
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel('Strain [με]')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_filename = os.path.join(output_dir, f'strain_vs_wavelength_{distance}cm.png')
        plt.savefig(output_filename)
        plt.close()

    print(f"Generated {len(distances)} plots in {output_dir}")

def main():
    """Main function to find the latest results and generate plots."""
    latest_folder = find_latest_output_folder()

    if latest_folder:
        csv_file_path = os.path.join(latest_folder, 'strain_analysis_results.csv')
        if os.path.exists(csv_file_path):
            output_dir = 'strain_wavelength_analysis_plots'
            os.makedirs(output_dir, exist_ok=True)
            plot_strain_vs_wavelength_by_distance(csv_file_path, output_dir)
        else:
            print(f"Error: 'strain_analysis_results.csv' not found in {latest_folder}")
    else:
        print("Error: No output folder found.")

if __name__ == '__main__':
    main()
