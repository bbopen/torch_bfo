#!/usr/bin/env python
"""
Basic usage example for BFO-Torch optimizer.

This example demonstrates how to use the BFO optimizer for training
a simple neural network on synthetic data.

Author: Brett G. Bonner
Repository: https://github.com/bbopen/torch_bfo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from bfo_torch import BFO, AdaptiveBFO, HybridBFO


def generate_data(n_samples=1000):
    """Generate synthetic regression data."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, 2)
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.1 * torch.randn(n_samples)
    return X, y


class SimpleModel(nn.Module):
    """Simple neural network for regression."""
    
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_with_bfo():
    """Train model using standard BFO optimizer."""
    print("Training with BFO optimizer...")
    
    # Generate data and model
    X, y = generate_data()
    model = SimpleModel()
    
    # Initialize BFO optimizer
    optimizer = BFO(
        model.parameters(),
        lr=0.01,
        population_size=30,
        chemotaxis_steps=5,
        early_stopping=True,
        convergence_patience=5
    )
    
    # Training loop
    losses = []
    for epoch in range(50):
        def closure():
            optimizer.zero_grad()
            output = model(X).squeeze()
            loss = F.mse_loss(output, y)
            return loss.item()
        
        loss = optimizer.step(closure)
        losses.append(loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}: Loss = {loss:.6f}")
    
    return losses


def train_with_adaptive_bfo():
    """Train model using Adaptive BFO optimizer."""
    print("\nTraining with AdaptiveBFO optimizer...")
    
    # Generate data and model
    X, y = generate_data()
    model = SimpleModel()
    
    # Initialize Adaptive BFO optimizer
    optimizer = AdaptiveBFO(
        model.parameters(),
        lr=0.01,
        population_size=20,
        adaptation_rate=0.1,
        early_stopping=True
    )
    
    # Training loop
    losses = []
    for epoch in range(50):
        def closure():
            optimizer.zero_grad()
            output = model(X).squeeze()
            loss = F.mse_loss(output, y)
            return loss.item()
        
        loss = optimizer.step(closure)
        losses.append(loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}: Loss = {loss:.6f}")
    
    return losses


def train_with_hybrid_bfo():
    """Train model using Hybrid BFO optimizer."""
    print("\nTraining with HybridBFO optimizer...")
    
    # Generate data and model
    X, y = generate_data()
    model = SimpleModel()
    
    # Initialize Hybrid BFO optimizer
    optimizer = HybridBFO(
        model.parameters(),
        lr=0.01,
        population_size=25,
        gradient_weight=0.3,
        early_stopping=True
    )
    
    # Training loop
    losses = []
    for epoch in range(50):
        def closure():
            # For hybrid optimizer, we can compute gradients
            optimizer.zero_grad()
            output = model(X).squeeze()
            loss = F.mse_loss(output, y)
            loss.backward()  # Compute gradients for hybrid approach
            return loss.item()
        
        loss = optimizer.step(closure)
        losses.append(loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}: Loss = {loss:.6f}")
    
    return losses


def compare_optimizers():
    """Compare all three BFO variants."""
    print("=" * 60)
    print("Comparing BFO Optimizers")
    print("=" * 60)
    
    # Train with each optimizer
    bfo_losses = train_with_bfo()
    adaptive_losses = train_with_adaptive_bfo() 
    hybrid_losses = train_with_hybrid_bfo()
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    epochs = range(len(bfo_losses))
    plt.plot(epochs, bfo_losses, 'b-', label='BFO', linewidth=2)
    plt.plot(epochs, adaptive_losses, 'r--', label='AdaptiveBFO', linewidth=2)
    plt.plot(epochs, hybrid_losses, 'g-.', label='HybridBFO', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('BFO Optimizer Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('bfo_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved as 'bfo_comparison.png'")
    
    # Print final results
    print("\nFinal Results:")
    print(f"BFO:         {bfo_losses[-1]:.6f}")
    print(f"AdaptiveBFO: {adaptive_losses[-1]:.6f}")
    print(f"HybridBFO:   {hybrid_losses[-1]:.6f}")


if __name__ == "__main__":
    compare_optimizers()