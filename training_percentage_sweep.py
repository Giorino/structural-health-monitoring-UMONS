#!/usr/bin/env python3
"""
Training Percentage Sweep Analysis
Runs neural network training with different training data percentages from 100% down to 5% in steps of 5%.
Collects training sequence numbers and final test accuracy results.
Saves results to CSV and creates analysis plots.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import sys

# Import the main neural network pipeline
from main_neural_network import main

# Results directory
RESULTS_DIR = "neural_network_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_training_percentage_sweep():
    """Run training with different percentages and collect results"""
    
    # Parameters for the sweep
    percentages = list(range(100, 0, -5))  # [100, 95, 90, ..., 10, 5]
    results_data = []
    
    print("="*80)
    print("TRAINING PERCENTAGE SWEEP ANALYSIS")
    print("="*80)
    print(f"Running {len(percentages)} experiments with training percentages: {percentages}")
    print("="*80)
    
    for i, percentage in enumerate(percentages, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(percentages)}: Training with {percentage}% of training data")
        print(f"{'='*80}")
        
        try:
            # Run the neural network training
            results, scaler, training_info = main(training_data_percentage=percentage)
            
            # Extract CNN results (assuming CNN is the model being used)
            cnn_results = results.get('CNN', {})
            test_accuracy = cnn_results.get('test_accuracy', 0.0)
            
            # Collect the data
            experiment_data = {
                'training_percentage': percentage,
                'training_set_size': training_info['training_set_size'],
                'validation_set_size': training_info['validation_set_size'],
                'test_set_size': training_info['test_set_size'],
                'total_sequences': training_info['total_sequences'],
                'test_accuracy': test_accuracy,
                'experiment_number': i
            }
            
            results_data.append(experiment_data)
            
            print(f"\n✓ EXPERIMENT {i} COMPLETED:")
            print(f"  - Training Percentage: {percentage}%")
            print(f"  - Training Sequences Used: {training_info['training_set_size']}")
            print(f"  - Test Accuracy: {test_accuracy:.2f}%")
            print(f"  - Confusion matrix saved as: CNNModel_confusion_matrix_{percentage}pct.png")
            
        except Exception as e:
            print(f"❌ EXPERIMENT {i} FAILED: {e}")
            # Still record the failure
            experiment_data = {
                'training_percentage': percentage,
                'training_set_size': 0,
                'validation_set_size': 0,
                'test_set_size': 0,
                'total_sequences': 0,
                'test_accuracy': 0.0,
                'experiment_number': i,
                'error': str(e)
            }
            results_data.append(experiment_data)
            continue
    
    return results_data

def save_results_to_csv(results_data):
    """Save results to CSV file with timestamp"""
    
    # Create DataFrame
    df = pd.DataFrame(results_data)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(RESULTS_DIR, f"training_percentage_sweep_{timestamp}.csv")
    
    # Save to CSV
    df.to_csv(csv_filename, index=False)
    print(f"\n✓ Results saved to: {csv_filename}")
    
    return csv_filename, df

def create_analysis_plots(df):
    """Create analysis plots showing the relationship between training data and performance"""
    
    # Filter out failed experiments
    valid_data = df[df['test_accuracy'] > 0].copy()
    
    if len(valid_data) == 0:
        print("❌ No valid data to plot!")
        return
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Test Accuracy vs Training Percentage
    ax1.plot(valid_data['training_percentage'], valid_data['test_accuracy'], 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Training Data Percentage (%)')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Test Accuracy vs Training Data Percentage')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 105)
    
    # Add accuracy values as annotations
    for _, row in valid_data.iterrows():
        ax1.annotate(f"{row['test_accuracy']:.1f}%", 
                    (row['training_percentage'], row['test_accuracy']),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # Plot 2: Test Accuracy vs Training Set Size
    ax2.plot(valid_data['training_set_size'], valid_data['test_accuracy'], 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Training Set Size (sequences)')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy vs Training Set Size')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Set Size vs Training Percentage
    ax3.plot(valid_data['training_percentage'], valid_data['training_set_size'], 'g-o', linewidth=2, markersize=6)
    ax3.set_xlabel('Training Data Percentage (%)')
    ax3.set_ylabel('Training Set Size (sequences)')
    ax3.set_title('Training Set Size vs Training Data Percentage')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 105)
    
    # Plot 4: Efficiency Analysis (Accuracy per Training Sample)
    efficiency = valid_data['test_accuracy'] / valid_data['training_set_size']
    ax4.plot(valid_data['training_percentage'], efficiency, 'm-o', linewidth=2, markersize=6)
    ax4.set_xlabel('Training Data Percentage (%)')
    ax4.set_ylabel('Test Accuracy per Training Sample')
    ax4.set_title('Training Efficiency (Accuracy/Training Size)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 105)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(RESULTS_DIR, f"training_percentage_analysis_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Analysis plots saved to: {plot_filename}")
    
    plt.show()
    
    return plot_filename

def print_summary_statistics(df):
    """Print summary statistics from the sweep"""
    
    valid_data = df[df['test_accuracy'] > 0].copy()
    
    if len(valid_data) == 0:
        print("❌ No valid data for summary!")
        return
    
    print(f"\n{'='*80}")
    print("SWEEP SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total experiments: {len(df)}")
    print(f"Successful experiments: {len(valid_data)}")
    print(f"Failed experiments: {len(df) - len(valid_data)}")
    
    print(f"\nACCURACY STATISTICS:")
    print(f"Highest accuracy: {valid_data['test_accuracy'].max():.2f}% (at {valid_data.loc[valid_data['test_accuracy'].idxmax(), 'training_percentage']}% training data)")
    print(f"Lowest accuracy: {valid_data['test_accuracy'].min():.2f}% (at {valid_data.loc[valid_data['test_accuracy'].idxmin(), 'training_percentage']}% training data)")
    print(f"Average accuracy: {valid_data['test_accuracy'].mean():.2f}%")
    print(f"Accuracy std deviation: {valid_data['test_accuracy'].std():.2f}%")
    
    print(f"\nTRAINING SET SIZE STATISTICS:")
    print(f"Largest training set: {valid_data['training_set_size'].max()} sequences (at {valid_data.loc[valid_data['training_set_size'].idxmax(), 'training_percentage']}%)")
    print(f"Smallest training set: {valid_data['training_set_size'].min()} sequences (at {valid_data.loc[valid_data['training_set_size'].idxmin(), 'training_percentage']}%)")
    print(f"Average training set size: {valid_data['training_set_size'].mean():.0f} sequences")
    
    # Find the optimal efficiency point
    if len(valid_data) > 1:
        efficiency = valid_data['test_accuracy'] / valid_data['training_set_size']
        optimal_idx = efficiency.idxmax()
        optimal_row = valid_data.loc[optimal_idx]
        print(f"\nOPTIMAL EFFICIENCY:")
        print(f"Most efficient training: {optimal_row['training_percentage']}% ({optimal_row['training_set_size']} sequences)")
        print(f"Efficiency score: {efficiency.loc[optimal_idx]:.6f} accuracy per sequence")
        print(f"Accuracy at optimal efficiency: {optimal_row['test_accuracy']:.2f}%")
    
    print(f"{'='*80}")

def main_sweep():
    """Main function to run the complete sweep analysis"""
    
    print("Starting Training Percentage Sweep Analysis...")
    
    # Run the sweep
    results_data = run_training_percentage_sweep()
    
    # Save results to CSV
    csv_filename, df = save_results_to_csv(results_data)
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Create analysis plots
    plot_filename = create_analysis_plots(df)
    
    print(f"\n{'='*80}")
    print("SWEEP ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"✓ Results CSV: {csv_filename}")
    if 'plot_filename' in locals():
        print(f"✓ Analysis plots: {plot_filename}")
    print(f"✓ Individual confusion matrices: CNNModel_confusion_matrix_*pct.png")
    print(f"✓ All files saved in: {RESULTS_DIR}/")
    print(f"{'='*80}")

if __name__ == "__main__":
    main_sweep()
