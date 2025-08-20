import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Channel constants to avoid scoping issues
CHANNELS = ['Lambda_00_nm', 'Lambda_01_nm', 'Lambda_02_nm']
CHANNEL_NAMES = ['Channel 0', 'Channel 1', 'Channel 2']

def analyze_manual_strain_data():
    """
    Analyze manual strain test data with corrected strain calculation.
    """
    # Read the manual strain test data
    data_path = 'strain_data/manual_strain_test_data.csv'
    df = pd.read_csv(data_path)
    
    # Constants
    L0 = 37650  # Initial length in mm
    
    # Calculate corrected strain using ε = ΔL / L₀
    # ΔL = L₀ - Current_Length = 37650 - Total_Length_mm
    df['Delta_L'] = L0 - df['Total_Length_mm']
    df['Strain_corrected'] = df['Delta_L'] / L0
    df['Strain_corrected_microstrain'] = df['Strain_corrected'] * 1e6  # Convert to microstrain
    
    # Calculate delta wavelength for each channel (relative to initial wavelength)
    channels = CHANNELS
    channel_names = CHANNEL_NAMES
    initial_wavelengths = df.iloc[0][CHANNELS].values
    
    for i, channel in enumerate(channels):
        df[f'Delta_{channel}'] = df[channel] - initial_wavelengths[i]
    
    # Create per-direction delta wavelength columns so each path starts at Δλ=0
    per_direction_delta_cols = []
    for channel in channels:
        forward_start_vals = df.loc[df['Direction'] == 'Forward', channel].dropna()
        reverse_start_vals = df.loc[df['Direction'] == 'Reverse', channel].dropna()
        if len(forward_start_vals) == 0 or len(reverse_start_vals) == 0:
            continue
        forward_start = forward_start_vals.iloc[0]
        reverse_start = reverse_start_vals.iloc[0]
        col_name = f'DeltaPD_{channel}'
        df[col_name] = np.where(
            df['Direction'] == 'Forward',
            df[channel] - forward_start,
            df[channel] - reverse_start
        )
        per_direction_delta_cols.append(col_name)

    # Create output directory
    output_dir = 'manual_strain_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Strain vs Wavelength for each channel
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['red', 'blue']
    
    for i, (channel, name) in enumerate(zip(channels, channel_names)):
        ax = axes[i]
        
        # Separate forward and reverse data
        forward_data = df[df['Direction'] == 'Forward'].dropna(subset=[channel])
        reverse_data = df[df['Direction'] == 'Reverse'].dropna(subset=[channel])
        
        # Plot forward direction
        if not forward_data.empty:
            ax.scatter(forward_data['Strain_corrected_microstrain'], forward_data[channel], 
                      color=colors[0], label='Forward', alpha=0.7, s=50)
            ax.plot(forward_data['Strain_corrected_microstrain'], forward_data[channel], 
                   color=colors[0], alpha=0.5, linestyle='-')
        
        # Plot reverse direction
        if not reverse_data.empty:
            ax.scatter(reverse_data['Strain_corrected_microstrain'], reverse_data[channel], 
                      color=colors[1], label='Reverse', alpha=0.7, s=50)
            ax.plot(reverse_data['Strain_corrected_microstrain'], reverse_data[channel], 
                   color=colors[1], alpha=0.5, linestyle='--')
        
        ax.set_xlabel('Strain [με]')
        ax.set_ylabel('Wavelength [nm]')
        ax.set_title(f'{name} - Wavelength vs Strain')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strain_vs_wavelength_all_channels.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Delta Wavelength vs Strain
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    delta_channels = ['Delta_Lambda_00_nm', 'Delta_Lambda_01_nm', 'Delta_Lambda_02_nm']
    
    for i, (delta_channel, name) in enumerate(zip(delta_channels, channel_names)):
        ax = axes[i]
        
        # Separate forward and reverse data
        forward_data = df[df['Direction'] == 'Forward'].dropna(subset=[delta_channel])
        reverse_data = df[df['Direction'] == 'Reverse'].dropna(subset=[delta_channel])
        
        # Plot forward direction
        if not forward_data.empty:
            ax.scatter(forward_data['Strain_corrected_microstrain'], forward_data[delta_channel], 
                      color=colors[0], label='Forward', alpha=0.7, s=50)
            ax.plot(forward_data['Strain_corrected_microstrain'], forward_data[delta_channel], 
                   color=colors[0], alpha=0.5, linestyle='-')
        
        # Plot reverse direction
        if not reverse_data.empty:
            ax.scatter(reverse_data['Strain_corrected_microstrain'], reverse_data[delta_channel], 
                      color=colors[1], label='Reverse', alpha=0.7, s=50)
            ax.plot(reverse_data['Strain_corrected_microstrain'], reverse_data[delta_channel], 
                   color=colors[1], alpha=0.5, linestyle='--')
        
        ax.set_xlabel('Strain [με]')
        ax.set_ylabel('Δλ [nm]')
        ax.set_title(f'{name} - Delta Wavelength vs Strain')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delta_wavelength_vs_strain_all_channels.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Hysteresis loops for all channels combined (closed loop, per-direction Δλ)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, (channel, delta_channel, name) in enumerate(zip(channels, per_direction_delta_cols, channel_names)):
        # Keep original CSV order (no 'Position' dependency)
        forward_data = df[df['Direction'] == 'Forward'].dropna(subset=[channel])
        reverse_data = df[df['Direction'] == 'Reverse'].dropna(subset=[channel])
        
        if not forward_data.empty and not reverse_data.empty:
            # Create complete closed loop: forward + reverse (reversed order)
            # Forward path: from low strain to high strain
            forward_strain = forward_data['Strain_corrected_microstrain'].values
            forward_wl = forward_data[delta_channel].values
            
            # Reverse path: use recorded order so it starts from where forward ended (Delta_L_9)
            reverse_strain = reverse_data['Strain_corrected_microstrain'].values
            reverse_wl = reverse_data[delta_channel].values
            
            # Combine to create closed loop
            complete_strain = np.concatenate([forward_strain, reverse_strain])
            complete_wl = np.concatenate([forward_wl, reverse_wl])
            
            # Plot the complete hysteresis loop
            ax.plot(complete_wl, complete_strain, 'o-', label=name, alpha=0.8, markersize=4, linewidth=2)
            
            # Add arrows to show direction
            # Arrow for forward direction (middle of forward path)
            mid_forward = len(forward_strain) // 2
            if mid_forward < len(forward_strain) - 1:
                ax.annotate('', xy=(forward_wl[mid_forward+1], forward_strain[mid_forward+1]), 
                           xytext=(forward_wl[mid_forward], forward_strain[mid_forward]),
                           arrowprops=dict(arrowstyle='->', color=ax.lines[-1].get_color(), lw=2))
            
            # Arrow for reverse direction (middle of reverse path)
            mid_reverse = len(reverse_strain) // 2
            if mid_reverse < len(reverse_strain) - 1:
                total_mid = len(forward_strain) + mid_reverse
                if total_mid < len(complete_strain) - 1:
                    ax.annotate('', xy=(complete_wl[total_mid+1], complete_strain[total_mid+1]), 
                               xytext=(complete_wl[total_mid], complete_strain[total_mid]),
                               arrowprops=dict(arrowstyle='->', color=ax.lines[-1].get_color(), lw=2))
    
    ax.set_xlabel('Δλ [nm]')
    ax.set_ylabel('Strain [με]')
    ax.set_title('Hysteresis Loops - All Channels')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hysteresis_loops_all_channels.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Strain comparison (original vs corrected)
    plt.figure(figsize=(12, 6))
    valid_data = df.dropna(subset=['Strain_microstrain'])
    
    plt.subplot(1, 2, 1)
    plt.plot(valid_data.index, valid_data['Strain_microstrain'], 'o-', label='Original Strain', color='red')
    plt.plot(valid_data.index, valid_data['Strain_corrected_microstrain'], 'o-', label='Corrected Strain (ΔL/L₀)', color='blue')
    plt.xlabel('Measurement Point')
    plt.ylabel('Strain [με]')
    plt.title('Strain Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(valid_data['Strain_microstrain'], valid_data['Strain_corrected_microstrain'], alpha=0.7)
    plt.xlabel('Original Strain [με]')
    plt.ylabel('Corrected Strain [με]')
    plt.title('Original vs Corrected Strain')
    plt.plot([0, 1200], [0, 1200], 'r--', alpha=0.5, label='1:1 line')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strain_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save corrected data to new CSV
    output_csv = 'strain_data/manual_strain_test_data_corrected.csv'
    df.to_csv(output_csv, index=False)
    
    # Print summary statistics
    print("Manual Strain Test Analysis Complete!")
    print(f"Plots saved to: {output_dir}/")
    print(f"Corrected data saved to: {output_csv}")
    print("\nSummary Statistics:")
    print(f"Strain range: {df['Strain_corrected_microstrain'].min():.1f} to {df['Strain_corrected_microstrain'].max():.1f} με")
    print(f"Number of measurement points: {len(df)}")
    
    # Calculate strain sensitivity for each channel
    print("\nStrain Sensitivity (Δλ/Δε):")
    for i, (delta_channel, name) in enumerate(zip(delta_channels, channel_names)):
        valid_data = df.dropna(subset=[delta_channel])
        if len(valid_data) > 1:
            # Linear fit to get sensitivity
            strain_data = valid_data['Strain_corrected_microstrain'].values
            wavelength_data = valid_data[delta_channel].values
            sensitivity = np.polyfit(strain_data, wavelength_data, 1)[0]  # pm/με
            print(f"{name}: {sensitivity*1000:.3f} pm/με")
    
    return df

if __name__ == '__main__':
    analyze_manual_strain_data()
