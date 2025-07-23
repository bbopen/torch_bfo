#!/usr/bin/env python3
"""
Minimal working example for PyTorch BFO on RunPod
Avoids known issues with PyTorch 2.8.0.dev
"""

import torch
import torch.nn as nn
from pytorch_bfo_optimizer import BFO, HybridBFO
import torch._dynamo as dynamo

# Fix graph breaks
dynamo.config.capture_scalar_outputs = True

# Create model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Linear(100, 10).to(device)

# Use even population size to avoid bug
optimizer = BFO(
    model.parameters(),
    population_size=6,  # EVEN number
    compile_mode=False  # Disable compile due to dev version issue
)

# Or use HybridBFO for better GPU performance
# optimizer = HybridBFO(
#     model.parameters(),
#     population_size=6,
#     gradient_weight=0.5,
#     compile_mode=False
# )

# Training data
data = torch.randn(512, 100, device=device)
target = torch.randn(512, 10, device=device)

# Optimization loop
def closure():
    with torch.no_grad():
        output = model(data)
        loss = nn.functional.mse_loss(output, target)
    return loss.item()

# For HybridBFO, use this closure instead:
# def closure():
#     optimizer.zero_grad()
#     output = model(data)
#     loss = nn.functional.mse_loss(output, target)
#     loss.backward()
#     return loss.item()

print("Running optimization...")
for i in range(10):
    loss = optimizer.step(closure)
    print(f"Step {i+1}: Loss = {loss:.6f}")

print("Done!")
