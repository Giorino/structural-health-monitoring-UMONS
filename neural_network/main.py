#!/usr/bin/env python3
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import load_and_preprocess_data, create_sequences_and_labels, normalize_features
from .datasets import SequenceDataset
from .models import GRUModel, LSTMModel, CNNGRUModel, TransformerModel, CNNModel
from .train import train_model, get_device
from .eval_utils import evaluate_model, plot_training_history, visualize_predictions


def run_pipeline():
    print("Structural Health Monitoring - Crack Prediction Pipeline")
    print("=" * 60)

    print("\n1. Loading and preprocessing data...")
    data = load_and_preprocess_data()
    print(f"Loaded {len(data)} sequences")

    print("\n2. Creating sequences and labels...")
    sequence_length = 50
    sequences, labels = create_sequences_and_labels(data, sequence_length=sequence_length)
    print(f"Created {len(sequences)} training sequences")
    if not sequences:
        print("Could not create any sequences. Exiting.")
        return None, None

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Label distribution: {dict(zip(unique_labels, counts))}")

    print("\n3. Normalizing features...")
    sequences, scaler = normalize_features(sequences)

    print("\n4. Splitting data...")
    from sklearn.model_selection import train_test_split
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = min(counts)
    if min_count >= 2 and len(sequences) > 10:
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, labels, test_size=0.15, stratify=labels, random_state=42
        )
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
        print("Warning: Small dataset detected, using random split instead of stratified split")
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42
        )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    batch_size = min(8, len(X_train)) if len(X_train) > 0 else 1
    train_dataset = SequenceDataset(X_train, y_train, sequence_length)
    val_dataset = SequenceDataset(X_val, y_val, sequence_length)
    test_dataset = SequenceDataset(X_test, y_test, sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = 8
    num_classes = 4
    models = {
        'GRU': GRUModel(input_size=input_size, num_classes=num_classes),
        'LSTM': LSTMModel(input_size=input_size, num_classes=num_classes),
        'CNN_GRU': CNNGRUModel(input_size=input_size, num_classes=num_classes),
        'Transformer': TransformerModel(input_size=input_size, num_classes=num_classes),
        'CNN': CNNModel(input_size=input_size, num_classes=num_classes),
    }

    results = {}
    device = get_device()
    for model_name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Training {model_name} Model")
        print(f"{'=' * 60}")
        train_losses, val_losses = train_model(model, train_loader, val_loader)
        plot_training_history(train_losses, val_losses, model_name)
        model.load_state_dict(torch.load(f'best_{model.__class__.__name__}.pth'))
        model = model.to(device)
        predictions, true_labels, probabilities = evaluate_model(model, test_loader)
        test_data = [data[i] for i in range(len(data)) if i < len(predictions)]
        if test_data:
            visualize_predictions(test_data, predictions, true_labels, model_name)
        results[model_name] = {
            'model': model,
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities,
        }

    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print("Models saved as 'best_<ModelName>.pth'")
    print("Visualizations saved as PNG files")
    print(f"{'=' * 60}")
    return results, scaler


if __name__ == "__main__":
    results, scaler = run_pipeline()


