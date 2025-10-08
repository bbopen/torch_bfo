"""
Neural network training example.

Demonstrates training a simple regression network with BFO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from bfo_torch import BFO, HybridBFO


class SimpleNet(nn.Module):
    """Simple feedforward network for regression."""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_data(n_samples=100, input_dim=10, noise=0.1):
    """Generate synthetic regression data."""
    X = torch.randn(n_samples, input_dim)
    # True function: linear combination + nonlinearity
    true_weights = torch.randn(input_dim, 1)
    y = torch.mm(X, true_weights) + torch.sin(X[:, 0:1]) * 2
    y = y + torch.randn(n_samples, 1) * noise
    return X, y


def train_with_bfo():
    """Train network with standard BFO."""
    print("Training with BFO (Black-box Optimization)")
    print("=" * 50)
    
    # Generate data
    X_train, y_train = generate_data(n_samples=100, input_dim=10)
    X_test, y_test = generate_data(n_samples=50, input_dim=10, noise=0.0)
    
    # Create model
    model = SimpleNet(input_dim=10, hidden_dim=20, output_dim=1)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create optimizer
    optimizer = BFO(
        model.parameters(),
        lr=0.01,
        population_size=30,
        chemotaxis_steps=5,
        swim_length=4,
        reproduction_steps=2,
        elimination_steps=2,
        seed=42,
    )
    
    # Training loop
    def closure():
        output = model(X_train)
        loss = F.mse_loss(output, y_train)
        return loss.item()
    
    print("\nTraining...")
    for epoch in range(20):
        train_loss = optimizer.step(closure)
        
        # Evaluate on test set
        with torch.no_grad():
            test_output = model(X_test)
            test_loss = F.mse_loss(test_output, y_test).item()
        
        if epoch % 5 == 0 or epoch == 19:
            print(f"Epoch {epoch:2d}: train_loss = {train_loss:.6f}, test_loss = {test_loss:.6f}")
    
    print(f"\nFinal test loss: {test_loss:.6f}")
    return test_loss


def train_with_hybrid_bfo():
    """Train network with HybridBFO (uses gradients)."""
    print("\n\nTraining with HybridBFO (BFO + Gradients)")
    print("=" * 50)
    
    # Generate data
    X_train, y_train = generate_data(n_samples=100, input_dim=10)
    X_test, y_test = generate_data(n_samples=50, input_dim=10, noise=0.0)
    
    # Create model
    model = SimpleNet(input_dim=10, hidden_dim=20, output_dim=1)
    
    # Create HybridBFO optimizer
    optimizer = HybridBFO(
        model.parameters(),
        lr=0.01,
        population_size=30,
        chemotaxis_steps=5,
        swim_length=4,
        reproduction_steps=2,
        elimination_steps=2,
        gradient_weight=0.5,  # Balance BFO and gradient information
        seed=42,
    )
    
    # Training loop with gradients
    def closure():
        optimizer.zero_grad()
        output = model(X_train)
        loss = F.mse_loss(output, y_train)
        loss.backward()  # Compute gradients
        return loss.item()
    
    print("\nTraining...")
    for epoch in range(20):
        train_loss = optimizer.step(closure)
        
        # Evaluate on test set
        with torch.no_grad():
            test_output = model(X_test)
            test_loss = F.mse_loss(test_output, y_test).item()
        
        if epoch % 5 == 0 or epoch == 19:
            print(f"Epoch {epoch:2d}: train_loss = {train_loss:.6f}, test_loss = {test_loss:.6f}")
    
    print(f"\nFinal test loss: {test_loss:.6f}")
    return test_loss


def main():
    """Run both training methods."""
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Train with standard BFO
    bfo_loss = train_with_bfo()
    
    # Train with HybridBFO
    torch.manual_seed(42)  # Reset for fair comparison
    hybrid_loss = train_with_hybrid_bfo()
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  BFO test loss:       {bfo_loss:.6f}")
    print(f"  HybridBFO test loss: {hybrid_loss:.6f}")
    print("\nNote: HybridBFO typically converges faster by leveraging gradients")


if __name__ == "__main__":
    main()

