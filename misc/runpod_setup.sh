#!/bin/bash
# RunPod Setup Script for PyTorch BFO Optimizer Testing

echo "=================================================="
echo "PyTorch BFO Optimizer - RunPod Setup"
echo "=================================================="

# Check if we're in a conda environment or virtual environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Detected Conda environment: $CONDA_DEFAULT_ENV"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "Detected virtual environment: $VIRTUAL_ENV"
else
    echo "No virtual environment detected. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Upgrade pip
echo -e "\n1. Upgrading pip..."
pip install --upgrade pip

# Install PyTorch 2.8.0 with CUDA support
echo -e "\n2. Installing PyTorch 2.8.0 with CUDA support..."
# Note: Adjust the CUDA version based on your RunPod instance
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu124

# Install the package in development mode
echo -e "\n3. Installing pytorch_bfo_optimizer in development mode..."
pip install -e ".[dev]"

# Verify installation
echo -e "\n4. Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import pytorch_bfo_optimizer; print('BFO Optimizer imported successfully')"

# Run quick tests to ensure everything works
echo -e "\n5. Running quick functionality test..."
python -c "
import torch
import torch.nn as nn
from pytorch_bfo_optimizer import BFO

# Quick test
model = nn.Linear(10, 1)
if torch.cuda.is_available():
    model = model.cuda()
    print('Testing on GPU...')
else:
    print('Testing on CPU...')

optimizer = BFO(model.parameters(), population_size=5)
print('BFO Optimizer initialized successfully!')
"

echo -e "\n=================================================="
echo "Setup complete! You can now run tests with:"
echo "  python runpod_test.py --all"
echo "Or for quick tests:"
echo "  python runpod_test.py --quick"
echo "=================================================="