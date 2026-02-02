#!/usr/bin/env python3
"""
Recreate the Training Accuracy plot with Training Samples on X-axis instead of Percentage.
This addresses reviewer feedback requesting the number of test trials/samples.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Data from the sweep results
data = {
    'training_percentage': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    'training_set_size': [169, 338, 508, 677, 846, 1016, 1185, 1354, 1524, 1693, 1862, 2032, 2201, 2370, 2540, 2709, 2878, 3048, 3217, 3387],
    'test_accuracy': [90.91, 92.42, 93.66, 95.45, 96.28, 96.56, 98.07, 97.80, 98.35, 98.35, 98.62, 98.62, 99.04, 99.04, 98.76, 98.62, 98.76, 98.90, 98.76, 98.62]
}

df = pd.DataFrame(data)

# Sort by training set size
df = df.sort_values('training_set_size')

# Create figure with professional styling
fig, ax = plt.subplots(figsize=(10, 6))

# Plot main line
ax.plot(df['training_set_size'], df['test_accuracy'], 
        'o-', 
        color='#2E86AB', 
        linewidth=2, 
        markersize=8,
        markerfacecolor='#2E86AB',
        markeredgecolor='white',
        markeredgewidth=1.5)

# Add annotations for each point
for idx, row in df.iterrows():
    ax.annotate(f"{row['test_accuracy']:.1f}%", 
                (row['training_set_size'], row['test_accuracy']),
                textcoords="offset points", 
                xytext=(0, 10), 
                ha='center', 
                fontsize=8,
                color='#333333')

# Labels and title
ax.set_xlabel('Number of Training Samples', fontsize=12, fontweight='medium')
ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='medium')
ax.set_title('Test Accuracy vs Training Sample Size', fontsize=14, fontweight='bold')

# Grid
ax.grid(True, linestyle=':', alpha=0.6)

# Set axis limits
ax.set_xlim(0, 3600)
ax.set_ylim(90, 100)

# Add secondary y-axis on the right (same scale, for readability)
ax2 = ax.twinx()
ax2.set_ylim(90, 100)
ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='medium')

# Tight layout
plt.tight_layout()

# Save to latex/img folder
output_path = os.path.join(os.path.dirname(__file__), 'latex', 'img', 'training_samples_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved figure to: {output_path}")

# Also show
plt.show()
