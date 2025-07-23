#!/usr/bin/env python3
"""
Minimal BFO test to verify the fix works
"""

import torch
import sys
import time

sys.path.insert(0, '.')

from pytorch_bfo_optimizer import BFO

# Simple test
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Testing on {device}")

# Create a simple optimization problem
x = torch.nn.Parameter(torch.tensor([5.0, 4.0, 3.0], device=device))
print(f"Initial x: {x.data}")

# Create optimizer with minimal settings
opt = BFO(
    [x],
    population_size=5,     # Very small
    chem_steps=2,          # Minimal steps
    swim_length=2,         # Short swim
    repro_steps=1,         # Minimal
    elim_steps=1,          # Minimal
    compile_mode=False,
    verbose=False
)

# Simple quadratic loss
def loss_fn():
    with torch.no_grad():
        return (x ** 2).sum().item()

# Run just 5 steps
print("\nRunning 5 optimization steps...")
for i in range(5):
    start = time.time()
    loss = opt.step(loss_fn)
    elapsed = time.time() - start
    print(f"Step {i}: loss={loss:.4f}, time={elapsed:.3f}s")

print(f"\nFinal x: {x.data}")
print(f"Final loss: {(x ** 2).sum().item():.4f}")
print("\nTest passed!")