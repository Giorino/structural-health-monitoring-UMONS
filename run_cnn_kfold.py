
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# Import your existing pipeline components
from main_neural_network import (
    load_and_preprocess_data, 
    create_sequences_and_labels, 
    SequenceDataset, 
    CNNModel, 
    train_model, 
    evaluate_model,
    device
)

def run_cnn_kfold():
    print("="*60)
    print("STARTING CNN 5-FOLD CROSS-VALIDATION")
    print("="*60)

    # 1. Load Data
    print("\n1. Loading Data...")
    real_data = load_and_preprocess_data(allow_synthetic_fallback=True)
    if not real_data:
        print("No data found!")
        return

    # 2. Create Sequences
    print("\n2. Creating Sequences...")
    sequence_length = 50
    sequences, labels = create_sequences_and_labels(real_data, sequence_length=sequence_length)
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    print(f"Total Sequences: {len(sequences)}")

    # 3. Setup K-Fold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    accuracies = []
    
    # 4. Run Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        
        # Split data
        X_train, X_val = sequences[train_idx], sequences[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # Create DataLoaders
        train_dataset = SequenceDataset(X_train, y_train)
        val_dataset = SequenceDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize FRESH model
        model = CNNModel(input_size=9, num_classes=4, dropout=0.2).to(device)
        
        # Train
        print(f"Training on {len(X_train)} samples, Validating on {len(X_val)} samples...")
        # Reduced epochs for speed, typically sufficient for fine-tuning demonstration
        train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001)
        
        # Evaluate
        _, _, _, accuracy = evaluate_model(model, val_loader)
        accuracies.append(accuracy)
        print(f"Fold {fold+1} Accuracy: {accuracy:.2f}%")

    # 5. Report Results
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print("\n" + "="*60)
    print("CNN 5-FOLD CV RESULTS (For Presentation)")
    print("="*60)
    print(f"Mean Accuracy: {mean_acc:.2f}%")
    print(f"Standard Deviation: +/- {std_acc:.2f}%")
    print("-" * 30)
    for i, acc in enumerate(accuracies):
        print(f"Fold {i+1}: {acc:.2f}%")
    print("="*60)

if __name__ == "__main__":
    run_cnn_kfold()
