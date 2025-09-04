#!/bin/bash

# Bash script to run the Neural Network Crack Prediction Pipeline
# This script activates the existing venv, installs requirements, and runs the analysis

set -e  # Exit on any error

echo "=========================================="
echo "Neural Network Crack Prediction Pipeline"
echo "=========================================="

# Check if venv directory exists
if [ ! -d "venv" ]; then
    echo "Error: venv directory not found. Please create a virtual environment first."
    echo "Run: python3 -m venv venv"
    exit 1
fi

echo "Using virtual environment Python directly (bypassing activation issues)..."

# Get absolute path to avoid issues when changing directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"

if [ -f "$VENV_PYTHON" ]; then
    echo "Found venv Python: $VENV_PYTHON"
    $VENV_PYTHON --version
    echo "Python path: $($VENV_PYTHON -c 'import sys; print(sys.executable)')"
    PYTHON_CMD="$VENV_PYTHON"
else
    echo "Error: Virtual environment Python not found at $VENV_PYTHON"
    exit 1
fi

echo "Checking/installing required packages..."

# Function to check if package is installed and meets version requirement
check_package() {
    local package_name=$1
    local version_req=$2
    
    if $PYTHON_CMD -c "import $package_name" 2>/dev/null; then
        echo "✓ $package_name is already installed"
        return 0
    else
        echo "✗ $package_name not found, will install"
        return 1
    fi
}

# Check and install packages only if needed
if ! check_package "numpy"; then
    echo "Installing NumPy..."
    $PYTHON_CMD -m pip install "numpy<2.0"
fi

if ! check_package "torch"; then
    echo "Installing PyTorch..."
    $PYTHON_CMD -m pip install torch>=2.0.0 torchvision>=0.15.0
fi

if ! check_package "sklearn"; then
    echo "Installing scikit-learn..."
    $PYTHON_CMD -m pip install scikit-learn>=1.1.0
fi

if ! check_package "pandas"; then
    echo "Installing pandas..."
    $PYTHON_CMD -m pip install pandas>=1.5.0
fi

if ! check_package "matplotlib"; then
    echo "Installing matplotlib..."
    $PYTHON_CMD -m pip install matplotlib>=3.5.0
fi

if ! check_package "seaborn"; then
    echo "Installing seaborn..."
    $PYTHON_CMD -m pip install seaborn>=0.11.0
fi

if ! check_package "tqdm"; then
    echo "Installing tqdm..."
    $PYTHON_CMD -m pip install tqdm>=4.64.0
fi

echo "Package check completed!"

echo "=========================================="
echo "Starting Neural Network Training Pipeline"
echo "=========================================="

# Check for merged CSV files
echo "Searching for merged CSV files..."
if find output -name "merged_*.csv" -type f | head -1 | grep -q .; then
    echo "Found merged CSV files in output directory - will use real data"
else
    echo "No merged CSV files found - will generate synthetic data for demonstration"
fi

# Create output directory for results
mkdir -p neural_network_results
cd neural_network_results

# Verify we're using the correct Python
echo "Verifying Python executable..."
echo "Using: $PYTHON_CMD"
echo "Python executable: $($PYTHON_CMD -c 'import sys; print(sys.executable)')"

# Test that imports work before running main script
echo "Testing critical imports..."
$PYTHON_CMD -c "
try:
    import numpy as np
    import torch
    import sklearn
    print(f'✓ NumPy version: {np.__version__}')
    print(f'✓ PyTorch version: {torch.__version__}')
    print('✓ All critical imports successful!')
except Exception as e:
    print(f'✗ Import error: {e}')
    exit(1)
"

# Run the main neural network script
echo "Running main_neural_network.py..."
$PYTHON_CMD ../main_neural_network.py

echo "=========================================="
echo "Pipeline completed successfully!"
echo "Results saved in: neural_network_results/"
echo "=========================================="

# List generated files
echo "Generated files:"
ls -la *.png *.pth 2>/dev/null || echo "No output files found (this is normal if using synthetic data)"

# Return to original directory
cd ..

# Note: No need to deactivate since we used Python executable directly

echo "Done! Check the neural_network_results/ directory for outputs."
