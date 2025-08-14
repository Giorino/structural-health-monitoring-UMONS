#!/usr/bin/env python3
"""
Power Spectral Density and Spectrogram Analysis for Fiber Bragg Grating Strain Monitoring
Similar to the frequency domain analysis in wind turbine tower monitoring papers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import os
import glob
from pathlib import Path

class FBG_PSD_Analyzer:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_interrogator_data(self, filename):
        """Load raw interrogator data from txt file"""
        file_path = self.data_dir / filename
        
        try:
            # Read the data, skipping the header
            data = pd.read_csv(file_path, sep='\t', skiprows=1, 
                             names=['Timestamp', 'Time_s', 'WL1_nm', 'WL2_nm', 'WL3_nm'])
            
            # Remove any rows with missing data
            data = data.dropna()
            
            # Convert wavelengths to float (handle any string issues)
            for col in ['WL1_nm', 'WL2_nm', 'WL3_nm']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove any rows where conversion failed
            data = data.dropna()
            
            print(f"Loaded {filename}: {len(data)} samples")
            print(f"Time range: {data['Time_s'].min():.1f} - {data['Time_s'].max():.1f} seconds")
            print(f"WL1 range: {data['WL1_nm'].min():.3f} - {data['WL1_nm'].max():.3f} nm")
            print(f"WL2 range: {data['WL2_nm'].min():.3f} - {data['WL2_nm'].max():.3f} nm")
            print(f"WL3 range: {data['WL3_nm'].min():.3f} - {data['WL3_nm'].max():.3f} nm")
            
            return data
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    
    def calculate_wavelength_shifts(self, data, baseline_points=100):
        """Calculate wavelength shifts relative to baseline (initial values)"""
        # Use first N points as baseline
        baseline_wl1 = data['WL1_nm'].head(baseline_points).mean()
        baseline_wl2 = data['WL2_nm'].head(baseline_points).mean()
        baseline_wl3 = data['WL3_nm'].head(baseline_points).mean()
        
        # Calculate shifts in picometers (pm)
        data['Delta_WL1_pm'] = (data['WL1_nm'] - baseline_wl1) * 1000  # nm to pm
        data['Delta_WL2_pm'] = (data['WL2_nm'] - baseline_wl2) * 1000  # nm to pm
        data['Delta_WL3_pm'] = (data['WL3_nm'] - baseline_wl3) * 1000  # nm to pm
        
        print(f"Baseline wavelengths: WL1={baseline_wl1:.3f}nm, WL2={baseline_wl2:.3f}nm, WL3={baseline_wl3:.3f}nm")
        
        return data
    
    def calculate_psd(self, time_series, time_array, nperseg=None):
        """Calculate Power Spectral Density"""
        # Calculate sampling frequency
        dt = np.diff(time_array).mean()
        fs = 1.0 / dt
        
        # Use Welch's method for PSD estimation
        if nperseg is None:
            nperseg = min(len(time_series) // 4, 1024)
            
        frequencies, psd = signal.welch(time_series, fs, nperseg=nperseg, 
                                       window='hann', noverlap=None, 
                                       scaling='density', detrend='linear')
        
        return frequencies, psd
    
    def find_peaks_in_psd(self, frequencies, psd, prominence=None, height=None):
        """Find significant peaks in PSD"""
        if prominence is None:
            prominence = np.max(psd) * 0.1  # 10% of max as default
            
        peaks, properties = signal.find_peaks(psd, prominence=prominence, height=height)
        
        peak_freqs = frequencies[peaks]
        peak_powers = psd[peaks]
        
        # Sort by power (descending)
        sorted_indices = np.argsort(peak_powers)[::-1]
        
        return peak_freqs[sorted_indices], peak_powers[sorted_indices], peaks[sorted_indices]
    
    def plot_psd_analysis(self, data, filename, max_freq=3.0):
        """Create PSD plots similar to the reference figure"""
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle(f'Bragg Wavelength Shift PSD Analysis - {filename}', fontsize=14, fontweight='bold')
        
        # Colors for each grating
        colors = ['blue', 'red', 'green']
        grating_names = ['FBG1 (~1555nm)', 'FBG2 (~1538nm)', 'FBG3 (~1525nm)']
        shift_columns = ['Delta_WL1_pm', 'Delta_WL2_pm', 'Delta_WL3_pm']
        
        for i, (shift_col, color, name) in enumerate(zip(shift_columns, colors, grating_names)):
            # Left column: Time domain (wavelength shift vs time)
            ax_time = axes[i, 0]
            time_data = data['Time_s'] - data['Time_s'].min()  # Start from 0
            ax_time.plot(time_data, data[shift_col], color=color, linewidth=0.8)
            ax_time.set_xlabel('Time [s]')
            ax_time.set_ylabel('Bragg wavelength shift [pm]')
            ax_time.set_title(f'({chr(97+i)}) {name}')
            ax_time.grid(True, alpha=0.3)
            
            # Right column: Frequency domain (PSD)
            ax_psd = axes[i, 1]
            
            # Calculate PSD
            frequencies, psd = self.calculate_psd(data[shift_col].values, time_data.values)
            
            # Convert to pm²/Hz (ensure proper units)
            psd_pm2_hz = psd  # Already in correct units since input was in pm
            
            # Plot PSD
            ax_psd.plot(frequencies, psd_pm2_hz, color=color, linewidth=1.2)
            ax_psd.set_xlabel('Frequency [Hz]')
            ax_psd.set_ylabel('PSD [pm²/Hz]')
            ax_psd.set_xlim(-0.5, max_freq)
            ax_psd.grid(True, alpha=0.3)
            
            # Find and label significant peaks
            peak_freqs, peak_powers, peak_indices = self.find_peaks_in_psd(
                frequencies, psd_pm2_hz, prominence=np.max(psd_pm2_hz) * 0.05)
            
            # Label top 3 peaks
            for j, (freq, power) in enumerate(zip(peak_freqs[:3], peak_powers[:3])):
                if freq <= max_freq:  # Only show peaks within our frequency range
                    ax_psd.plot(freq, power, 'ro', markersize=8, markerfacecolor='red', 
                              markeredgecolor='darkred', markeredgewidth=1)
                    ax_psd.annotate(f'P{j+1}', xy=(freq, power), xytext=(5, 5), 
                                  textcoords='offset points', fontsize=10, 
                                  fontweight='bold', color='red',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                          edgecolor='red', alpha=0.8))
            
            # Set y-axis to log scale if there's a wide range of values
            if np.max(psd_pm2_hz) / np.min(psd_pm2_hz[psd_pm2_hz > 0]) > 100:
                ax_psd.set_yscale('log')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / f'psd_analysis_{filename.replace(".txt", ".png")}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"PSD analysis saved to: {output_file}")
        
        return fig
    
    def analyze_multiple_files(self, pattern="*-interrogator.txt", max_freq=3.0):
        """Analyze multiple interrogator files"""
        files = list(self.data_dir.glob(pattern))
        files.sort()
        
        print(f"Found {len(files)} files matching pattern: {pattern}")
        
        results = {}
        
        for file_path in files:
            filename = file_path.name
            print(f"\n{'='*60}")
            print(f"Analyzing: {filename}")
            print('='*60)
            
            # Load data
            data = self.load_interrogator_data(filename)
            if data is None:
                continue
                
            # Calculate wavelength shifts
            data = self.calculate_wavelength_shifts(data)
            
            # Create PSD analysis plot
            fig = self.plot_psd_analysis(data, filename, max_freq)
            plt.close(fig)  # Close to save memory
            
            # Store results for summary
            results[filename] = {
                'data_points': len(data),
                'duration_s': data['Time_s'].max() - data['Time_s'].min(),
                'max_shifts_pm': {
                    'WL1': data['Delta_WL1_pm'].abs().max(),
                    'WL2': data['Delta_WL2_pm'].abs().max(),
                    'WL3': data['Delta_WL3_pm'].abs().max()
                }
            }
        
        # Create summary report
        self.create_summary_report(results)
        
        return results
    
    def create_summary_report(self, results):
        """Create a summary report of all analyses"""
        summary_file = self.output_dir / 'psd_analysis_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("Fiber Bragg Grating PSD Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for filename, data in results.items():
                f.write(f"File: {filename}\n")
                f.write(f"  Data points: {data['data_points']:,}\n")
                f.write(f"  Duration: {data['duration_s']:.1f} seconds\n")
                f.write(f"  Max wavelength shifts (pm):\n")
                f.write(f"    FBG1 (1555nm): {data['max_shifts_pm']['WL1']:.2f}\n")
                f.write(f"    FBG2 (1538nm): {data['max_shifts_pm']['WL2']:.2f}\n")
                f.write(f"    FBG3 (1525nm): {data['max_shifts_pm']['WL3']:.2f}\n")
                f.write("-" * 40 + "\n")
        
        print(f"\nSummary report saved to: {summary_file}")
    
    def create_spectrogram_analysis(self, data, filename, max_freq=3.0):
        """Create spectrogram plots showing frequency evolution over time"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle(f'Spectrogram Analysis - {filename}', fontsize=14, fontweight='bold')
        
        # Colors and names for each grating
        grating_names = ['FBG1 (~1555nm)', 'FBG2 (~1538nm)', 'FBG3 (~1525nm)']
        shift_columns = ['Delta_WL1_pm', 'Delta_WL2_pm', 'Delta_WL3_pm']
        
        for i, (shift_col, name) in enumerate(zip(shift_columns, grating_names)):
            ax = axes[i]
            
            # Prepare time series data
            time_data = data['Time_s'] - data['Time_s'].min()  # Start from 0
            wavelength_shifts = data[shift_col].values
            
            # Calculate sampling frequency
            dt = np.diff(time_data).mean()
            fs = 1.0 / dt
            
            # Calculate spectrogram
            # Use shorter segments for better time resolution
            nperseg = min(len(wavelength_shifts) // 8, 256)  # Shorter segments than PSD
            noverlap = nperseg // 2  # 50% overlap
            
            frequencies, times, Sxx = signal.spectrogram(
                wavelength_shifts, fs,
                nperseg=nperseg,
                noverlap=noverlap,
                window='hann',
                scaling='density',
                detrend='linear'
            )
            
            # Convert to dB scale for better visualization
            Sxx_db = 10 * np.log10(Sxx + 1e-12)  # Add small value to avoid log(0)
            
            # Create spectrogram plot
            im = ax.pcolormesh(times, frequencies, Sxx_db, 
                              shading='gouraud', cmap='viridis')
            
            ax.set_ylabel('Frequency [Hz]')
            ax.set_ylim(0, min(max_freq, fs/2))
            ax.set_title(f'({chr(97+i)}) {name}')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('PSD [dB]')
            
            # Only add x-label to bottom plot
            if i == 2:
                ax.set_xlabel('Time [s]')
        
        plt.tight_layout()
        
        # Save the spectrogram plot
        output_file = self.output_dir / f'spectrogram_{filename.replace(".txt", ".png")}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Spectrogram saved to: {output_file}")
        
        return fig
    
    def analyze_loading_phases(self, data, merged_data_file=None):
        """Analyze how frequency content changes with loading phases"""
        if merged_data_file is None:
            return None
            
        try:
            # Load merged data to get loading information
            merged_data = pd.read_csv(merged_data_file)
            
            # Get unique pressure levels
            pressure_levels = sorted(merged_data['Air Pressure (bar)'].unique())
            force_levels = []
            
            for pressure in pressure_levels:
                force = merged_data[merged_data['Air Pressure (bar)'] == pressure]['Force (N)'].iloc[0]
                force_levels.append(force)
            
            print(f"Loading phases detected:")
            for p, f in zip(pressure_levels, force_levels):
                print(f"  {p:.1f} bar → {f:.0f} N")
                
            return pressure_levels, force_levels
            
        except Exception as e:
            print(f"Could not analyze loading phases: {e}")
            return None
    
    def create_combined_spectrogram_analysis(self, pattern="*-interrogator.txt", max_freq=3.0):
        """Create spectrogram analysis for multiple files"""
        files = list(self.data_dir.glob(pattern))
        files.sort()
        
        print(f"\nCreating Spectrogram Analysis")
        print("=" * 50)
        print(f"Found {len(files)} files matching pattern: {pattern}")
        
        for file_path in files:  # Process all files
            filename = file_path.name
            print(f"\nCreating spectrogram for: {filename}")
            
            # Load data
            data = self.load_interrogator_data(filename)
            if data is None:
                continue
                
            # Calculate wavelength shifts
            data = self.calculate_wavelength_shifts(data)
            
            # Create spectrogram
            fig = self.create_spectrogram_analysis(data, filename, max_freq)
            plt.close(fig)  # Close to save memory
            
            # Try to find corresponding merged data
            base_name = filename.replace('-interrogator.txt', '')
            merged_pattern = f"output/*/merged_{base_name}_*.csv"
            merged_files = list(Path('.').glob(merged_pattern))
            
            if merged_files:
                merged_file = merged_files[0]  # Take first match
                print(f"Found merged data: {merged_file}")
                loading_info = self.analyze_loading_phases(data, merged_file)

    def create_comprehensive_psd_summary(self, pattern="*-interrogator.txt", max_freq=3.0):
        """Create a comprehensive summary plot with all PSD analyses"""
        files = list(self.data_dir.glob(pattern))
        files.sort()
        
        n_files = len(files)
        print(f"\nCreating Comprehensive PSD Summary")
        print("=" * 50)
        print(f"Generating summary for {n_files} files")
        
        # Create large figure: 3 columns (FBG1, FBG2, FBG3) x n_files rows
        fig, axes = plt.subplots(n_files, 3, figsize=(18, 4*n_files))
        fig.suptitle('Comprehensive Power Spectral Density Analysis - All Files', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Ensure axes is always 2D
        if n_files == 1:
            axes = axes.reshape(1, -1)
        
        grating_names = ['FBG1 (~1555nm)', 'FBG2 (~1538nm)', 'FBG3 (~1525nm)']
        shift_columns = ['Delta_WL1_pm', 'Delta_WL2_pm', 'Delta_WL3_pm']
        colors = ['blue', 'red', 'green']
        
        for file_idx, file_path in enumerate(files):
            filename = file_path.name
            print(f"  Processing {filename}...")
            
            # Load and process data
            data = self.load_interrogator_data(filename)
            if data is None:
                continue
            data = self.calculate_wavelength_shifts(data)
            
            # Extract key info for title
            parts = filename.replace('-interrogator.txt', '').split('-')
            span_info = f"{parts[0]} {parts[1]}"
            
            for grating_idx, (shift_col, color, grating_name) in enumerate(zip(shift_columns, colors, grating_names)):
                ax = axes[file_idx, grating_idx]
                
                # Calculate PSD
                time_data = data['Time_s'] - data['Time_s'].min()
                frequencies, psd = self.calculate_psd(data[shift_col].values, time_data.values)
                
                # Plot PSD
                ax.plot(frequencies, psd, color=color, linewidth=1.0)
                ax.set_xlim(0, max_freq)
                ax.grid(True, alpha=0.3)
                
                # Find and mark peaks
                peak_freqs, peak_powers, peak_indices = self.find_peaks_in_psd(
                    frequencies, psd, prominence=np.max(psd)*0.05)
                
                # Mark top 3 peaks
                for j, (freq, power) in enumerate(zip(peak_freqs[:3], peak_powers[:3])):
                    if freq <= max_freq:
                        ax.plot(freq, power, 'o', color='red', markersize=4)
                        ax.annotate(f'P{j+1}', xy=(freq, power), xytext=(2, 2), 
                                  textcoords='offset points', fontsize=8, color='red')
                
                # Set labels and title
                if file_idx == n_files - 1:  # Bottom row
                    ax.set_xlabel('Frequency [Hz]', fontsize=10)
                if grating_idx == 0:  # Left column
                    ax.set_ylabel('PSD [pm²/Hz]', fontsize=10)
                
                # Title for each subplot
                if file_idx == 0:  # Top row
                    ax.set_title(f'{grating_name}', fontsize=12, fontweight='bold')
                if grating_idx == 0:  # Left column  
                    ax.text(-0.15, 0.5, span_info, rotation=90, transform=ax.transAxes, 
                           fontsize=11, fontweight='bold', va='center', ha='center')
                
                # Set y-axis to log scale if needed
                if np.max(psd) / np.min(psd[psd > 0]) > 100:
                    ax.set_yscale('log')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.97)
        
        # Save comprehensive PSD summary
        summary_file = self.output_dir / 'comprehensive_psd_summary.png'
        plt.savefig(summary_file, dpi=200, bbox_inches='tight')
        print(f"\nComprehensive PSD summary saved to: {summary_file}")
        plt.close(fig)
        
        return summary_file
    
    def create_comprehensive_spectrogram_summary(self, pattern="*-interrogator.txt", max_freq=3.0):
        """Create a comprehensive summary plot with all spectrograms"""
        files = list(self.data_dir.glob(pattern))
        files.sort()
        
        n_files = len(files)
        print(f"\nCreating Comprehensive Spectrogram Summary")
        print("=" * 50)
        print(f"Generating summary for {n_files} files")
        
        # Create large figure: 3 columns (FBG1, FBG2, FBG3) x n_files rows
        fig, axes = plt.subplots(n_files, 3, figsize=(18, 3*n_files))
        fig.suptitle('Comprehensive Spectrogram Analysis - All Files', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Ensure axes is always 2D
        if n_files == 1:
            axes = axes.reshape(1, -1)
        
        grating_names = ['FBG1 (~1555nm)', 'FBG2 (~1538nm)', 'FBG3 (~1525nm)']
        shift_columns = ['Delta_WL1_pm', 'Delta_WL2_pm', 'Delta_WL3_pm']
        
        for file_idx, file_path in enumerate(files):
            filename = file_path.name
            print(f"  Processing {filename}...")
            
            # Load and process data
            data = self.load_interrogator_data(filename)
            if data is None:
                continue
            data = self.calculate_wavelength_shifts(data)
            
            # Extract key info for title
            parts = filename.replace('-interrogator.txt', '').split('-')
            span_info = f"{parts[0]} {parts[1]}"
            
            for grating_idx, (shift_col, grating_name) in enumerate(zip(shift_columns, grating_names)):
                ax = axes[file_idx, grating_idx]
                
                # Prepare time series data
                time_data = data['Time_s'] - data['Time_s'].min()
                wavelength_shifts = data[shift_col].values
                
                # Calculate sampling frequency
                dt = np.diff(time_data).mean()
                fs = 1.0 / dt
                
                # Calculate spectrogram
                nperseg = min(len(wavelength_shifts) // 8, 256)
                noverlap = nperseg // 2
                
                frequencies, times, Sxx = signal.spectrogram(
                    wavelength_shifts, fs,
                    nperseg=nperseg, noverlap=noverlap,
                    window='hann', scaling='density', detrend='linear'
                )
                
                # Convert to dB scale
                Sxx_db = 10 * np.log10(Sxx + 1e-12)
                
                # Create spectrogram plot
                im = ax.pcolormesh(times, frequencies, Sxx_db, 
                                  shading='gouraud', cmap='viridis')
                
                ax.set_ylim(0, min(max_freq, fs/2))
                
                # Set labels and title
                if file_idx == n_files - 1:  # Bottom row
                    ax.set_xlabel('Time [s]', fontsize=10)
                if grating_idx == 0:  # Left column
                    ax.set_ylabel('Frequency [Hz]', fontsize=10)
                
                # Title for each subplot
                if file_idx == 0:  # Top row
                    ax.set_title(f'{grating_name}', fontsize=12, fontweight='bold')
                if grating_idx == 0:  # Left column  
                    ax.text(-0.12, 0.5, span_info, rotation=90, transform=ax.transAxes, 
                           fontsize=11, fontweight='bold', va='center', ha='center')
        
        # Add a single colorbar for the entire plot
        plt.tight_layout()
        plt.subplots_adjust(top=0.97, right=0.95)
        
        # Create colorbar
        cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('PSD [dB]', fontsize=12)
        
        # Save comprehensive spectrogram summary
        summary_file = self.output_dir / 'comprehensive_spectrogram_summary.png'
        plt.savefig(summary_file, dpi=150, bbox_inches='tight')
        print(f"\nComprehensive spectrogram summary saved to: {summary_file}")
        plt.close(fig)
        
        return summary_file

def main():
    """Main function to run PSD analysis"""
    # Configuration
    script_dir = Path(__file__).parent
    interrogator_data_dir = script_dir / "interrogator-data"
    output_dir = script_dir / "psd_output"
    
    print("Fiber Bragg Grating Power Spectral Density Analysis")
    print("=" * 60)
    print(f"Input directory: {interrogator_data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create analyzer
    analyzer = FBG_PSD_Analyzer(interrogator_data_dir, output_dir)
    
    # Run PSD analysis on all interrogator files
    results = analyzer.analyze_multiple_files(
        pattern="*-interrogator.txt",
        max_freq=3.0  # Similar to reference figure
    )
    
    print(f"\nPSD Analysis complete! Results saved to: {output_dir}")
    print(f"Analyzed {len(results)} files successfully.")
    
    # Also run spectrogram analysis
    print(f"\nStarting Spectrogram Analysis...")
    analyzer.create_combined_spectrogram_analysis(
        pattern="*-interrogator.txt",
        max_freq=3.0
    )
    
    print(f"\nAll analyses complete! Results saved to: {output_dir}")
    
    # Generate comprehensive summary plots
    print(f"\nGenerating comprehensive summary plots...")
    psd_summary = analyzer.create_comprehensive_psd_summary(
        pattern="*-interrogator.txt", max_freq=3.0
    )
    spectrogram_summary = analyzer.create_comprehensive_spectrogram_summary(
        pattern="*-interrogator.txt", max_freq=3.0
    )
    
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"✅ Individual PSD plots: 14 files")
    print(f"✅ Individual spectrograms: 14 files") 
    print(f"✅ Comprehensive PSD summary: {psd_summary.name}")
    print(f"✅ Comprehensive spectrogram summary: {spectrogram_summary.name}")
    print(f"\nAll analyses complete! Check: {output_dir}")

def spectrogram_only():
    """Run only spectrogram analysis"""
    # Configuration
    script_dir = Path(__file__).parent
    interrogator_data_dir = script_dir / "interrogator-data"
    output_dir = script_dir / "psd_output"
    
    print("Fiber Bragg Grating Spectrogram Analysis")
    print("=" * 60)
    print(f"Input directory: {interrogator_data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create analyzer
    analyzer = FBG_PSD_Analyzer(interrogator_data_dir, output_dir)
    
    # Run only spectrogram analysis
    print(f"\nStarting Spectrogram Analysis for ALL files...")
    analyzer.create_combined_spectrogram_analysis(
        pattern="*-interrogator.txt",
        max_freq=3.0
    )
    
    print(f"\nSpectrogram analysis complete! Results saved to: {output_dir}")

def summary_only():
    """Generate only the comprehensive summary plots"""
    # Configuration
    script_dir = Path(__file__).parent
    interrogator_data_dir = script_dir / "interrogator-data"
    output_dir = script_dir / "psd_output"
    
    print("Comprehensive Summary Generation")
    print("=" * 60)
    print(f"Input directory: {interrogator_data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create analyzer
    analyzer = FBG_PSD_Analyzer(interrogator_data_dir, output_dir)
    
    # Generate comprehensive summary plots
    print(f"\nGenerating comprehensive summary plots...")
    psd_summary = analyzer.create_comprehensive_psd_summary(
        pattern="*-interrogator.txt", max_freq=3.0
    )
    spectrogram_summary = analyzer.create_comprehensive_spectrogram_summary(
        pattern="*-interrogator.txt", max_freq=3.0
    )
    
    print(f"\n🎯 SUMMARY GENERATION COMPLETE:")
    print(f"✅ Comprehensive PSD summary: {psd_summary.name}")
    print(f"✅ Comprehensive spectrogram summary: {spectrogram_summary.name}")
    print(f"\nFiles saved to: {output_dir}")

if __name__ == "__main__":
    main()