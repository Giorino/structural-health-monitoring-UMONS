import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class CNNModel(nn.Module):
    def __init__(self, input_size=9, num_classes=4, dropout=0.2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

def main():
    print("Visualizing CNN Feature Maps...")
    
    # 1. Setup the Model
    # We will use the trained model if available, else a random one.
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neural_network_results", "best_CNNModel.pth")
    model = CNNModel(input_size=9, num_classes=4)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
            print("Loaded trained model weights.")
        except:
            print("Could not load weights, using untrained model for demonstration.")
    model.eval()
    
    # 2. Create an intuitive synthetic signal to clearly show how feature maps react
    # We create a 50-step sequence with 9 channels.
    sequence = np.zeros((50, 9))
    
    # Channel 0: Wavelength (Add a sharp peak)
    sequence[20:25, 0] = [0.2, 0.8, 2.5, 0.8, 0.2] 
    
    # Channel 3: Force (Add a step function/drop)
    sequence[:30, 3] = 5.0
    sequence[30:, 3] = 1.0
    
    # Channel 4: Displacement (Gradual increase)
    sequence[:, 4] = np.linspace(0, 3, 50)
    
    # Add slight noise to everything
    sequence += np.random.normal(0, 0.05, sequence.shape)
    
    # 3. Pass data through the first Convolutional Layer
    # Input shape needs to be (batch_size, input_size, sequence_length) -> (1, 9, 50)
    x = torch.FloatTensor([sequence]).transpose(1, 2)
    
    with torch.no_grad():
        feature_maps = F.relu(model.conv1(x))
        
    fmaps = feature_maps.numpy()[0] # Shape: (32, 50)
    
    # 4. Plot the results
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    
    # Plot Original Data
    time_steps = np.arange(50)
    axes[0].plot(time_steps, sequence[:, 0], label='Wavelength Peak', color='blue', linewidth=2)
    axes[0].plot(time_steps, sequence[:, 3], label='Force Drop', color='red', linestyle='--')
    axes[0].set_title("1. Original Raw Input (50 Time Steps)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    
    # We pick the 3 filters that had the strongest maximum activation
    # This ensures we plot the most interesting feature maps!
    max_activations = np.max(fmaps, axis=1)
    best_filters = np.argsort(max_activations)[-3:][::-1] # Top 3
    
    colors = ['purple', 'orange', 'teal']
    for i, filter_idx in enumerate(best_filters):
        axes[i+1].plot(time_steps, fmaps[filter_idx], color=colors[i], linewidth=2)
        axes[i+1].fill_between(time_steps, fmaps[filter_idx], alpha=0.3, color=colors[i])
        axes[i+1].set_title(f"2. Feature Map from Filter #{filter_idx}")
        axes[i+1].set_ylabel("Activation")
        axes[i+1].grid(True, alpha=0.3)
    
    axes[3].set_xlabel("Time Step (Index)")
    plt.tight_layout()
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exercise_outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "feature_map_visualization.png")
    plt.savefig(out_file, dpi=200)
    print(f"Saved feature map visualization to {out_file}")

if __name__ == "__main__":
    main()
