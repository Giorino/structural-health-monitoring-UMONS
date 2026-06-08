import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def load_raw_data(file_path, start_line=0, num_lines=500):
    df = pd.read_csv(file_path, sep='\t', skiprows=0)
    signal = df['WL 2[nm]'].values[start_line:start_line+num_lines]
    signal = signal[~np.isnan(signal)]
    return signal

def apply_filter(signal, kernel):
    x = torch.tensor(signal, dtype=torch.float32).view(1, 1, -1)
    k = torch.tensor(kernel, dtype=torch.float32).view(1, 1, -1)
    with torch.no_grad():
        conv = nn.Conv1d(1, 1, kernel_size=k.shape[-1], padding='same', padding_mode='replicate', bias=False)
        conv.weight.data = k
        output = conv(x)
    return output.view(-1).numpy()

def main():
    raw_file = "interrogator-data/11cm-16layers-1-interrogator.txt"
    signal = load_raw_data(raw_file)
    
    # Define the kernels
    kernels = [
        [-1, 0, 1],
        [-1, 0, 0, 0, 1],
        [1, -2, 1]
    ]
    
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Raw Signal
    plt.subplot(2, 1, 1)
    plt.plot(signal, color='black')
    plt.ylabel("Wavelength [nm]")
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Feature Maps comparison
    plt.subplot(2, 1, 2)
    colors = ['red', 'blue', 'green']
    
    for k, color in zip(kernels, colors):
        feature_map = apply_filter(signal, k)
        plt.plot(feature_map, color=color, label=str(k), alpha=0.8)
    
    plt.ylabel("Filter Output Intensity")
    plt.xlabel("Time Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the result
    output_plot = "exercise_outputs/filter_comparison_results.png"
    os.makedirs("exercise_outputs", exist_ok=True)
    plt.savefig(output_plot)
    print(f"\nResults saved to {output_plot}")
    
    plt.show()

if __name__ == "__main__":
    main()
