import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# STUDENT EXERCISE: BUILD YOUR FIRST 1D CNN
# ==========================================
# Your task is to build a simple Convolutional Neural Network to predict 
# structural cracks based on 9 sensor channels over 50 time steps.

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # TASK 1: Define your layers!
        # Our input data has 9 channels (features).
        # Create a 1D Convolutional layer (nn.Conv1d) that takes 9 input channels 
        # and outputs 16 feature maps. Use a kernel_size of 3 and padding of 1.
        self.conv1 = None  # <-- FILL THIS IN
        
        # Create a second 1D Convolutional layer that takes the 16 channels from 
        # the previous layer and outputs 32 channels. (kernel_size=3, padding=1)
        self.conv2 = None  # <-- FILL THIS IN
        
        # We need a way to condense the sequence down to a single prediction.
        # This layer will average out the 50 time steps into just 1 time step.
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Finally, a Fully Connected (Linear) layer to make the final prediction.
        # It should take the 32 channels from conv2 and output 4 classes 
        # (0: No Crack, 1: Small, 2: Medium, 3: Large).
        self.fc = None     # <-- FILL THIS IN
        
    def forward(self, x):
        # x is your input data. Shape: (Batch Size, 9 channels, 50 time steps)
        
        # TASK 2: Pass the data through the network!
        # 1. Pass 'x' through conv1, then apply the ReLU activation function (F.relu)
        # x = ...          # <-- FILL THIS IN
        
        # 2. Pass 'x' through conv2, then apply the ReLU activation function
        # x = ...          # <-- FILL THIS IN
        
        # 3. Pool the data to squash the time steps
        x = self.global_pool(x)
        
        # 4. Remove the extra empty dimension (this is done for you)
        x = x.squeeze(-1) 
        
        # 5. Pass 'x' through your final fully connected layer (fc) to get the prediction
        # output = ...     # <-- FILL THIS IN
        
        return output

# ==========================================
# TEST YOUR MODEL
# ==========================================
if __name__ == "__main__":
    print("Testing your CNN Architecture...")
    try:
        # Create a fake batch of sensor data: 
        # 5 examples, 9 sensor channels, 50 time steps
        fake_data = torch.randn(5, 9, 50) 
        
        # Initialize your model
        model = SimpleCNN()
        
        # Make a prediction!
        predictions = model(fake_data)
        
        print("\nSUCCESS! Your model successfully processed the data.")
        print(f"Output shape: {predictions.shape} (Expected: 5 examples, 4 classes)")
        
    except Exception as e:
        print("\nOops! Something went wrong in your architecture:")
        print(e)
