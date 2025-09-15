import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, test_loader, class_names=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    print(f"\n{model.__class__.__name__} Performance:")
    print("=" * 50)
    unique_classes = sorted(set(all_labels + all_predictions))
    if class_names is None:
        default_names = ['No Crack', 'Small', 'Medium', 'Large']
        class_names = [default_names[i] if i < len(default_names) else f'Class {i}' for i in unique_classes]
    print(classification_report(all_labels, all_predictions, target_names=class_names[:len(unique_classes)]))

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names[:len(unique_classes)], yticklabels=class_names[:len(unique_classes)])
    plt.title(f'{model.__class__.__name__} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{model.__class__.__name__}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    return all_predictions, all_labels, all_probabilities


def plot_training_history(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_predictions(data, predictions, labels, model_name, num_samples=5):
    plt.figure(figsize=(15, 10))
    for i in range(min(num_samples, len(data))):
        plt.subplot(num_samples, 1, i + 1)
        sequence = data[i]['sequence']
        true_label = labels[i]
        pred_label = predictions[i]
        delta_wl = sequence[:, 2]
        plt.plot(delta_wl, 'b-', linewidth=2, label='ΔWL_ch2 (Strain proxy)')
        colors = ['green', 'yellow', 'orange', 'red']
        default_names = ['No Crack', 'Small', 'Medium', 'Large']
        unique_classes = sorted(set(labels + predictions))
        class_names = [default_names[i] if i < len(default_names) else f'Class {i}' for i in unique_classes]
        true_name = class_names[true_label] if true_label < len(class_names) else f'Class {true_label}'
        pred_name = class_names[pred_label] if pred_label < len(class_names) else f'Class {pred_label}'
        plt.title(f'Sample {i + 1}: True={true_name}, Pred={pred_name}')
        plt.ylabel('ΔWL_ch2 (nm)')
        plt.xlabel('Time Step')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.suptitle(f'{model_name} - Predictions vs Actual')
    plt.tight_layout()
    plt.savefig(f'{model_name}_predictions_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()



