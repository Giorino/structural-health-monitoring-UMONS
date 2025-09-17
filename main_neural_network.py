#!/usr/bin/env python3
"""
PyTorch-based Neural Network Pipeline for Crack Prediction in Composite Materials
Focus on WL_ch2 sensor data with mechanical features for structural health monitoring.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings('ignore')

# Create results directory
RESULTS_DIR = "neural_network_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SequenceDataset(Dataset):
    """Custom dataset for sequence data with crack prediction labels"""
    
    def __init__(self, sequences, labels, sequence_length=25):
        self.sequences = sequences
        self.labels = labels
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.LongTensor([self.labels[idx]])
        
        # Pad or truncate sequence to fixed length
        if len(sequence) < self.sequence_length:
            padding = torch.zeros(self.sequence_length - len(sequence), sequence.shape[1])
            sequence = torch.cat([sequence, padding], dim=0)
        elif len(sequence) > self.sequence_length:
            sequence = sequence[:self.sequence_length]
            
        return sequence, label.squeeze()

class GRUModel(nn.Module):
    """GRU-based model for crack prediction"""
    
    def __init__(self, input_size=9, hidden_size=64, num_classes=4, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        gru_out, _ = self.gru(x)
        # Use the last output for classification
        output = gru_out[:, -1, :]  # (batch_size, hidden_size)
        output = self.dropout(output)
        output = self.fc(output)
        return output

class LSTMModel(nn.Module):
    """LSTM-based model for crack prediction"""
    
    def __init__(self, input_size=9, hidden_size=64, num_classes=4, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Use the last output for classification
        output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        output = self.dropout(output)
        output = self.fc(output)
        return output

class CNNGRUModel(nn.Module):
    """CNN-GRU hybrid model for crack prediction"""
    
    def __init__(self, input_size=9, hidden_size=64, num_classes=4, dropout=0.2):
        super(CNNGRUModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.gru = nn.GRU(32, hidden_size, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        # Transpose for CNN: (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        
        # CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Transpose back for GRU: (batch_size, sequence_length, features)
        x = x.transpose(1, 2)
        
        # GRU layer
        gru_out, _ = self.gru(x)
        output = gru_out[:, -1, :]  # Use last output
        output = self.dropout(output)
        output = self.fc(output)
        return output

class TransformerModel(nn.Module):
    """Transformer Encoder model for crack prediction"""
    
    def __init__(self, input_size=9, d_model=64, num_heads=2, num_layers=2, num_classes=4, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))  # Max sequence length
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Project to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer encoder
        transformer_out = self.transformer(x)
        
        # Global average pooling
        output = transformer_out.mean(dim=1)  # (batch_size, d_model)
        output = self.dropout(output)
        output = self.fc(output)
        return output

class CNNModel(nn.Module):
    """Pure CNN model for crack prediction"""
    
    def __init__(self, input_size=9, num_classes=4, dropout=0.2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        # Transpose for CNN: (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        
        # CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global average pooling
        x = self.global_pool(x) # shape: (batch_size, 128, 1)
        x = x.squeeze(-1) # shape: (batch_size, 128)
        
        # Final classification
        output = self.dropout(x)
        output = self.fc(output)
        return output

def generate_synthetic_data(num_samples=1000, sequence_length=25):
    """Generate synthetic data for testing when no real CSV files are available"""
    print("Generating synthetic data for demonstration...")
    
    data = []
    for i in range(num_samples):
        # Simulate group and repetition indices
        group_idx = np.random.randint(0, 10)
        rep_idx = np.random.randint(0, 5)
        
        # Generate sequence
        sequence = []
        baseline_wl = 1550.0 + np.random.normal(0, 0.1)  # Baseline wavelength
        
        for t in range(sequence_length):
            # Simulate increasing strain/displacement over time
            time_factor = t / sequence_length
            
            # WL_ch2 increases with strain (realistic for FBG sensors)
            wl_ch2 = baseline_wl + time_factor * np.random.uniform(0.1, 2.0) + np.random.normal(0, 0.01)
            wl_ch2_std = np.random.uniform(0.001, 0.01)
            
            # Mechanical features
            force = time_factor * np.random.uniform(50, 500) + np.random.normal(0, 5)
            displacement = time_factor * np.random.uniform(1, 10) + np.random.normal(0, 0.1)
            air_pressure = np.random.uniform(1, 6)  # Bar
            
            # Calculate delta WL
            delta_wl_ch2 = wl_ch2 - baseline_wl
            
            # Calculate rates (simple differences)
            if t > 0:
                prev_delta = sequence[-1][2]  # Previous delta_wl_ch2
                prev_disp = sequence[-1][4]   # Previous displacement
                delta_wl_rate = delta_wl_ch2 - prev_delta
                delta_disp_rate = displacement - prev_disp
            else:
                delta_wl_rate = 0
                delta_disp_rate = 0
            
            # Randomly assign small sample status for synthetic data
            is_small_sample = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% chance of small sample
            
            sequence.append([wl_ch2, wl_ch2_std, delta_wl_ch2, force, displacement, air_pressure, 
                           delta_wl_rate, delta_disp_rate, is_small_sample])
        
        # Assign crack label based on maximum displacement/strain
        max_displacement = max([s[4] for s in sequence])
        if max_displacement < 2:
            crack_label = 0  # No crack
        elif max_displacement < 5:
            crack_label = 1  # Small crack
        elif max_displacement < 8:
            crack_label = 2  # Medium crack
        else:
            crack_label = 3  # Large crack
        
        data.append({
            'group_index': group_idx,
            'repetition_index': rep_idx,
            'sequence': np.array(sequence),
            'crack_label': crack_label
        })
    
    return data

def load_and_preprocess_data(data_dir="./", prediction_horizon=5):
    """Load CSV files and preprocess data for neural network training"""
    
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
    
    # If running from neural_network_results directory, look one level up
    parent_output_dir = os.path.join("..", "output")
    latest_parent_dir = find_latest_output_dir(parent_output_dir)
    if latest_parent_dir and not latest_dir:  # Only if we didn't find any in current dir
        latest_csv_files = glob.glob(os.path.join(latest_parent_dir, "merged_*.csv"))
        csv_files.extend(latest_csv_files)
        print(f"Using latest parent output directory: {os.path.basename(latest_parent_dir)}")
        print(f"Found {len(latest_csv_files)} CSV files in latest directory")
    
    if not csv_files:
        print("No merged CSV files found. Using synthetic data for demonstration.")
        return generate_synthetic_data()
    
    print(f"Found {len(csv_files)} CSV files")
    
    all_data = []
    
    for file_path in csv_files:
        print(f"Processing {file_path}")
        try:
            df = pd.read_csv(file_path)
            
            # Detect small samples from filename (contains "-s")
            filename = os.path.basename(file_path)
            is_small_sample = "-s" in filename
            print(f"{'Small sample' if is_small_sample else 'Regular sample'} detected: {filename}")
            
            # Base required columns (common to all files)
            base_required_cols = ['group_index', 'repetition_index', 'WL_ch2', 'WL_ch2_std', 
                                'Force (N)', 'Displacement (mm)', 'Crack']
            
            # Handle Air Pressure column differences between regular and small samples
            if is_small_sample:
                # Small samples may not have Air Pressure column, use default value
                if 'Air Pressure (bar)' not in df.columns:
                    print("  → Adding default Air Pressure value for small sample")
                    df['Air Pressure (bar)'] = 1.0  # Default air pressure for small samples
            
            required_cols = base_required_cols + ['Air Pressure (bar)']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols} in {file_path}")
                print(f"Available columns: {list(df.columns)}")
                continue
            
            print(f"✓ All required columns found. Shape: {df.shape}")
            
            # Group by group_index and repetition_index
            groups = df.groupby(['group_index', 'repetition_index'])
            
            processed_count = 0
            for (group_idx, rep_idx), group_df in groups:
                if len(group_df) < 1:  # Accept single measurements
                    continue
                
                processed_count += 1
                
                # Sort by timestamp or segment index
                if 'timestamp' in group_df.columns:
                    group_df = group_df.sort_values('timestamp')
                elif 'segment_start_idx' in group_df.columns:
                    group_df = group_df.sort_values('segment_start_idx')
                
                # Calculate baseline and delta values
                baseline_wl = group_df['WL_ch2'].iloc[0]
                group_df = group_df.copy()
                group_df['delta_wl_ch2'] = group_df['WL_ch2'] - baseline_wl
                
                # Calculate rates
                group_df['delta_wl_rate'] = group_df['delta_wl_ch2'].diff().fillna(0)
                group_df['delta_disp_rate'] = group_df['Displacement (mm)'].diff().fillna(0)
                
                # Add small sample indicator as a feature (binary: 1 for small sample, 0 for regular)
                group_df['is_small_sample'] = 1 if is_small_sample else 0
                
                # Create feature sequence (using actual CSV column names + small sample indicator)
                features = ['WL_ch2', 'WL_ch2_std', 'delta_wl_ch2', 'Force (N)', 
                           'Displacement (mm)', 'Air Pressure (bar)', 'delta_wl_rate', 'delta_disp_rate', 'is_small_sample']
                
                sequence = group_df[features].values
                
                # Handle crack labels
                crack_values = group_df['Crack'].fillna(0).astype(int)
                # Use the maximum crack level in the sequence as the label
                crack_label = crack_values.max()
                
                all_data.append({
                    'group_index': group_idx,
                    'repetition_index': rep_idx,
                    'sequence': sequence,
                    'crack_label': crack_label
                })
                
            print(f"✓ Processed {processed_count} measurements from {file_path}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not all_data:
        print("No valid data found in CSV files. Using synthetic data.")
        return generate_synthetic_data()
    
    return all_data

def create_sequences_and_labels(data, sequence_length=50, prediction_horizon=5):
    """Create fixed-length sequences and labels for training using a sliding window approach."""

    sequences = []
    labels = []

    # Group single measurements by air pressure to create sequences
    pressure_groups = {}
    for item in data:
        # Each item['sequence'] is currently a single measurement (shape (1, num_features))
        if item['sequence'].shape[0] > 0:
            measurement = item['sequence'][0]
            air_pressure = measurement[5]  # Air pressure is at index 5
            crack_label = item['crack_label']

            if air_pressure not in pressure_groups:
                pressure_groups[air_pressure] = []
            
            pressure_groups[air_pressure].append({
                'measurement': measurement,
                'crack_label': int(crack_label) # Ensure label is int
            })

    print(f"Found {len(pressure_groups)} pressure groups.")

    # For each pressure group, create sliding window sequences
    total_sequences_created = 0
    for pressure, measurements in pressure_groups.items():
        # Sort by displacement to create a logical progression for the test
        measurements.sort(key=lambda x: x['measurement'][4])  # Displacement is at index 4

        num_measurements = len(measurements)
        print(f"Pressure {pressure:.1f} bar: {num_measurements} measurements.")

        # Create sliding windows if there are enough measurements
        if num_measurements >= sequence_length + prediction_horizon:
            sequences_from_group = 0
            for i in range(num_measurements - sequence_length - prediction_horizon + 1):
                # Define the window for features and the point for the label
                feature_window = measurements[i : i + sequence_length]
                label_index = i + sequence_length + prediction_horizon - 1

                # Extract feature data
                sequence_data = [m['measurement'] for m in feature_window]
                
                # The label is the crack state at the prediction horizon
                label = measurements[label_index]['crack_label']
                
                # To prevent predicting a lower crack state than what already exists
                max_crack_in_sequence = max(m['crack_label'] for m in feature_window)
                final_label = max(label, max_crack_in_sequence)

                sequences.append(np.array(sequence_data))
                labels.append(final_label)
                sequences_from_group += 1

            if sequences_from_group > 0:
                print(f"  → Created {sequences_from_group} sequences.")
            total_sequences_created += sequences_from_group

    print(f"\nTotal sequences created across all groups: {total_sequences_created}")
    return sequences, labels

def normalize_features(sequences):
    """Normalize features using StandardScaler"""
    
    # Flatten all sequences to compute statistics
    all_features = np.concatenate([seq for seq in sequences], axis=0)
    
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    # Normalize each sequence
    normalized_sequences = []
    for seq in sequences:
        normalized_seq = scaler.transform(seq)
        normalized_sequences.append(normalized_seq)
    
    return normalized_sequences, scaler

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train a neural network model with early stopping"""
    
    model = model.to(device)
    
    # Calculate class weights for imbalanced data
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.cpu().numpy())
    
    # Get unique classes and their counts
    unique_classes = sorted(set(all_labels))
    num_classes = 4  # Fixed number of classes as defined in model
    
    # Initialize class counts array for all classes
    class_counts = np.ones(num_classes)  # Start with 1 to avoid division by zero
    for class_idx in unique_classes:
        if class_idx < num_classes:
            class_counts[class_idx] = all_labels.count(class_idx)
    
    total_samples = len(all_labels)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 50)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f'best_{model.__class__.__name__}.pth'))
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, class_names=None, training_percentage=100):
    """Evaluate model performance on test set"""
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate overall accuracy
    total_samples = len(all_labels)
    correct_predictions = sum(1 for true, pred in zip(all_labels, all_predictions) if true == pred)
    overall_accuracy = (correct_predictions / total_samples) * 100
    
    print(f"\n{model.__class__.__name__} Test Results Summary:")
    print("=" * 60)
    print(f"Total Test Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")
    print("=" * 60)
    
    # Determine class names based on actual data
    unique_classes = sorted(set(all_labels + all_predictions))
    if class_names is None:
        default_names = ['No Crack', 'Small', 'Medium', 'Large']
        class_names = [default_names[i] if i < len(default_names) else f'Class {i}' for i in unique_classes]
    
    # Detailed per-class results with percentages
    print(f"\nDetailed Per-Class Test Results:")
    print("-" * 60)
    
    # Calculate per-class statistics
    for i, class_idx in enumerate(unique_classes):
        class_name = class_names[i] if i < len(class_names) else f'Class {class_idx}'
        
        # True positives, false positives, false negatives
        true_samples = [j for j, label in enumerate(all_labels) if label == class_idx]
        predicted_samples = [j for j, pred in enumerate(all_predictions) if pred == class_idx]
        
        true_count = len(true_samples)
        predicted_count = len(predicted_samples)
        correct_class = sum(1 for j in true_samples if all_predictions[j] == class_idx)
        
        if true_count > 0:
            class_recall = (correct_class / true_count) * 100
        else:
            class_recall = 0.0
            
        if predicted_count > 0:
            class_precision = (correct_class / predicted_count) * 100
        else:
            class_precision = 0.0
        
        print(f"{class_name} (Class {class_idx}):")
        print(f"  - True samples: {true_count} ({(true_count/total_samples)*100:.1f}% of total)")
        print(f"  - Correctly predicted: {correct_class}")
        print(f"  - Recall (Sensitivity): {class_recall:.2f}%")
        print(f"  - Precision: {class_precision:.2f}%")
        print()
    
    print(classification_report(all_labels, all_predictions, target_names=class_names[:len(unique_classes)]))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names[:len(unique_classes)], yticklabels=class_names[:len(unique_classes)])
    plt.title(f'{model.__class__.__name__} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model.__class__.__name__}_confusion_matrix_{training_percentage}pct.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    # Log final test accuracy for easy reference
    print(f"\n{'='*60}")
    print(f"FINAL TEST ACCURACY: {overall_accuracy:.2f}%")
    print(f"{'='*60}")
    
    return all_predictions, all_labels, all_probabilities, overall_accuracy

def plot_training_history(train_losses, val_losses, model_name):
    """Plot training and validation loss curves"""
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

def main(training_data_percentage=100):
    """Main execution pipeline
    
    Args:
        training_data_percentage (int): Percentage of training data to actually use for training (0-100).
                                      For example, if 50, only 50% of the allocated training data will be used.
                                      Test and validation set sizes remain constant.
    """
    
    print("Structural Health Monitoring - Crack Prediction Pipeline")
    print("=" * 60)
    print(f"Training Data Usage: {training_data_percentage}% of allocated training data will be used")
    print("=" * 60)
    
    # 1. Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    data = load_and_preprocess_data()
    print(f"Loaded {len(data)} sequences")
    
    # 2. Create sequences and labels
    print("\n2. Creating sequences and labels...")
    sequence_length = 50  # Define sequence length here
    sequences, labels = create_sequences_and_labels(data, sequence_length=sequence_length)
    print(f"Created {len(sequences)} training sequences")
    
    if not sequences:
        print("Could not create any sequences. Exiting.")
        return None, None

    # Check label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Label distribution: {dict(zip(unique_labels, counts))}")
    
    # 3. Normalize features
    print("\n3. Normalizing features...")
    sequences, scaler = normalize_features(sequences)
    
    # 4. Split data
    print("\n4. Splitting data...")
    
    # Check if we have enough samples for stratified splitting
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = min(counts)
    
    if min_count >= 2 and len(sequences) > 10:
        # Use stratified split if we have enough samples
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, labels, test_size=0.15, stratify=labels, random_state=42
        )
        
        # Check if we can stratify the validation split too
        unique_temp, temp_counts = np.unique(y_temp, return_counts=True)
        if min(temp_counts) >= 2:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.176, random_state=42
            )
    else:
        # Use simple random split for small datasets
        print("Warning: Small dataset detected, using random split instead of stratified split")
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, labels, test_size=0.2, random_state=42  # Slightly larger test set
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42  # 25% of 80% = 20% total for val
        )
    
    print(f"Initial Split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 4.1 Apply training data percentage reduction
    if training_data_percentage < 100:
        print(f"\n4.1 Reducing training data to {training_data_percentage}%...")
        original_train_size = len(X_train)
        target_train_size = int((training_data_percentage / 100) * original_train_size)
        
        if target_train_size > 0:
            # Use stratified sampling if possible for consistent class distribution
            unique_train_labels, train_counts = np.unique(y_train, return_counts=True)
            min_train_count = min(train_counts)
            
            if min_train_count >= 2 and target_train_size >= len(unique_train_labels):
                # Use stratified sampling
                X_train_reduced, _, y_train_reduced, _ = train_test_split(
                    X_train, y_train, 
                    train_size=target_train_size, 
                    stratify=y_train, 
                    random_state=42
                )
            else:
                # Use random sampling
                X_train_reduced, _, y_train_reduced, _ = train_test_split(
                    X_train, y_train, 
                    train_size=target_train_size, 
                    random_state=42
                )
            
            X_train = X_train_reduced
            y_train = y_train_reduced
            
            print(f"Reduced training set from {original_train_size} to {len(X_train)} samples")
            print(f"Unused training data: {original_train_size - len(X_train)} samples")
        else:
            print("Warning: Training data percentage too low, keeping at least 1 sample")
            X_train = X_train[:1] if len(X_train) > 0 else X_train
            y_train = y_train[:1] if len(y_train) > 0 else y_train
    
    print(f"Final Split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Total data distribution: Train {(len(X_train)/(len(X_train)+len(X_val)+len(X_test)))*100:.1f}%, Val {(len(X_val)/(len(X_train)+len(X_val)+len(X_test)))*100:.1f}%, Test {(len(X_test)/(len(X_train)+len(X_val)+len(X_test)))*100:.1f}%")
    
    # 5. Create data loaders
    batch_size = min(8, len(X_train)) if len(X_train) > 0 else 1 # Smaller batch size for small dataset
    
    train_dataset = SequenceDataset(X_train, y_train, sequence_length)
    val_dataset = SequenceDataset(X_val, y_val, sequence_length)
    test_dataset = SequenceDataset(X_test, y_test, sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 6. Define models
    input_size = 9  # WL_ch2, WL_ch2_std, delta_wl_ch2, Force, Displacement, Air Pressure, rates, is_small_sample
    num_classes = 4  # 0, 1, 2, 3
    
    models = {
        #'GRU': GRUModel(input_size=input_size, num_classes=num_classes),
        #'LSTM': LSTMModel(input_size=input_size, num_classes=num_classes),
        #'CNN_GRU': CNNGRUModel(input_size=input_size, num_classes=num_classes),
        #'Transformer': TransformerModel(input_size=input_size, num_classes=num_classes),
        'CNN': CNNModel(input_size=input_size, num_classes=num_classes)
    }
    
    # 7. Train and evaluate models
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name} Model")
        print(f"{'='*60}")
        
        # Train model
        train_losses, val_losses = train_model(model, train_loader, val_loader)
        plot_training_history(train_losses, val_losses, model_name)
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f'best_{model.__class__.__name__}.pth')))
        model = model.to(device)
        
        # Evaluate model
        predictions, true_labels, probabilities, test_accuracy = evaluate_model(model, test_loader, training_percentage=training_data_percentage)
        
        results[model_name] = {
            'model': model,
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities,
            'test_accuracy': test_accuracy
        }
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Models saved in '{RESULTS_DIR}/' as 'best_<ModelName>.pth'")
    print(f"Training history and confusion matrix plots saved in '{RESULTS_DIR}/' as PNG files")
    print(f"{'='*60}")
    
    # Create a summary with training information
    training_info = {
        'training_data_percentage': training_data_percentage,
        'training_set_size': len(X_train),
        'validation_set_size': len(X_val),
        'test_set_size': len(X_test),
        'total_sequences': len(sequences)
    }
    
    return results, scaler, training_info

if __name__ == "__main__":
    # Run the complete pipeline
    # You can modify the training_data_percentage parameter here
    # For example: main(50) will use only 50% of the allocated training data
    training_percentage = 90  # Default: use 100% of training data
    results, scaler, training_info = main(training_data_percentage=training_percentage)
