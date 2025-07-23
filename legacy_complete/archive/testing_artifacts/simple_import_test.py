#!/usr/bin/env python3
"""
Simple import test to verify basic functionality
"""

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test basic import
try:
    from pytorch_bfo_optimizer import BFO
    print("✅ BFO import successful")
except Exception as e:
    print(f"❌ BFO import failed: {e}")
    exit(1)

# Test initialization with even population size
try:
    import torch.nn as nn
    model = nn.Linear(10, 1)
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = BFO(
        model.parameters(),
        population_size=4,  # Even number
        compile_mode=False  # Disable compile
    )
    print("✅ BFO initialization successful")
    print(f"   Device: {optimizer.device}")
    print(f"   Population size: {optimizer.defaults['population_size']}")
    print(f"   Number of parameters: {optimizer.num_params}")
    
except Exception as e:
    print(f"❌ BFO initialization failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")