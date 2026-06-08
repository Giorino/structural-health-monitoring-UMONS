import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# =================================================================
# WORKSHOP EXERCISE: THE KERNEL PLAYGROUND
# =================================================================
# Goal: Experiment with different numbers in your filter (kernel)
# to see how they transform raw sensor data into "Features".
# =================================================================

def load_raw_data(file_path, num_lines=500):
    """Loads a segment of raw FBG sensor data."""
    df = pd.read_csv(file_path, sep='\t', skiprows=0)
    signal = df['WL 2[nm]'].values[:num_lines]
    return signal[~np.isnan(signal)]

def apply_filter(signal, kernel):
    """Applies your custom kernel using 1D Convolution."""
    x = torch.tensor(signal, dtype=torch.float32).view(1, 1, -1)
    k = torch.tensor(kernel, dtype=torch.float32).view(1, 1, -1)
    with torch.no_grad():
        conv = nn.Conv1d(1, 1, kernel_size=k.shape[-1], padding='same', padding_mode='replicate', bias=False)
        conv.weight.data = k
        output = conv(x)
    return output.view(-1).numpy()

def main():
    # Load raw data from the interrogator
    raw_file = "interrogator-data/11cm-16layers-1-interrogator.txt"
    signal = load_raw_data(raw_file)
    
    # -------------------------------------------------------------
    # EXERCISE: EXPERIMENT HERE
    # -------------------------------------------------------------
    # Modify the list of numbers below. Try different lengths, 
    # positive vs negative numbers, and different patterns.
    #
    # DISCOVERY QUESTIONS:
    # 1. How do you make the spikes larger?
    # 2. Can you design a filter that makes the signal look "smoother"?
    # 3. What happens if the sum of your numbers is zero? (e.g. [-1, 1])
    # 4. What happens if the sum is one? (e.g. [0.2, 0.2, 0.2, 0.2, 0.2])
    
    my_kernel = [1] # <--- PLAY WITH THESE NUMBERS!
    
    # -------------------------------------------------------------
    
    # Apply your filter
    feature_map = apply_filter(signal, my_kernel)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top: Original Data
    ax1.plot(signal, color='black', label='Raw Sensor Signal')
    ax1.set_ylabel("Wavelength [nm]")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Your Feature Map
    ax2.plot(feature_map, color='red', label=f'Feature Map (Kernel: {my_kernel})')
    ax2.set_ylabel("Intensity")
    ax2.set_xlabel("Time Steps")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
