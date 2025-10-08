"""
Hyperparameter tuning example.

Demonstrates using BFO to optimize hyperparameters of another optimizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from bfo_torch import BFO


class SimpleNet(nn.Module):
    """Simple network for demonstration."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def generate_data():
    """Generate synthetic regression data."""
    torch.manual_seed(123)  # Fixed seed for data
    X_train = torch.randn(100, 10)
    y_train = torch.randn(100, 1)
    X_val = torch.randn(50, 10)
    y_val = torch.randn(50, 1)
    return X_train, y_train, X_val, y_val


def train_model_with_hyperparams(lr, weight_decay, X_train, y_train, X_val, y_val, epochs=10):
    """Train a model with given hyperparameters and return validation loss."""
    # Ensure hyperparameters are valid
    lr = max(1e-5, min(1e-1, lr))
    weight_decay = max(0.0, min(1e-2, weight_decay))
    
    # Create fresh model
    model = SimpleNet()
    
    # Create optimizer with current hyperparameters
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Train for a few epochs
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = F.mse_loss(output, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate on validation set
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = F.mse_loss(val_output, y_val).item()
    
    return val_loss


def main():
    print("Hyperparameter Optimization with BFO")
    print("=" * 60)
    print("Searching for optimal learning rate and weight decay")
    print("for Adam optimizer on a simple regression task\n")
    
    # Generate data once
    X_train, y_train, X_val, y_val = generate_data()
    
    # Hyperparameters to optimize (in log space for numerical stability)
    log_lr = nn.Parameter(torch.tensor([-3.0]))  # log10(lr)
    log_wd = nn.Parameter(torch.tensor([-4.0]))  # log10(weight_decay)
    
    # Create BFO optimizer for hyperparameter search
    meta_optimizer = BFO(
        [log_lr, log_wd],
        lr=0.1,
        population_size=20,
        chemotaxis_steps=5,
        swim_length=3,
        reproduction_steps=2,
        elimination_steps=2,
        domain_bounds=(-5.0, -1.0),  # Search in log space
        seed=42,
    )
    
    # Keep track of best hyperparameters
    best_val_loss = float('inf')
    best_lr = None
    best_wd = None
    
    # Define closure for hyperparameter optimization
    def closure():
        # Convert from log space
        lr = 10 ** log_lr.item()
        wd = 10 ** log_wd.item()
        
        # Train model and get validation loss
        val_loss = train_model_with_hyperparams(
            lr, wd, X_train, y_train, X_val, y_val, epochs=10
        )
        
        return val_loss
    
    # Run hyperparameter search
    print("Searching...\n")
    print(f"{'Trial':>5} | {'log(lr)':>8} | {'log(wd)':>8} | {'lr':>10} | {'wd':>10} | {'Val Loss':>10}")
    print("-" * 75)
    
    for trial in range(15):
        val_loss = meta_optimizer.step(closure)
        
        # Convert to actual hyperparameter values
        lr = 10 ** log_lr.item()
        wd = 10 ** log_wd.item()
        
        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lr = lr
            best_wd = wd
        
        # Print progress
        print(f"{trial:5d} | {log_lr.item():8.3f} | {log_wd.item():8.3f} | "
              f"{lr:10.6f} | {wd:10.6f} | {val_loss:10.6f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Search Complete!")
    print(f"\nBest hyperparameters found:")
    print(f"  Learning rate:  {best_lr:.6f}")
    print(f"  Weight decay:   {best_wd:.6f}")
    print(f"  Validation loss: {best_val_loss:.6f}")
    
    # Train final model with best hyperparameters
    print(f"\nTraining final model with best hyperparameters...")
    final_val_loss = train_model_with_hyperparams(
        best_lr, best_wd, X_train, y_train, X_val, y_val, epochs=20
    )
    print(f"Final validation loss: {final_val_loss:.6f}")
    
    # Compare with default hyperparameters
    print(f"\nComparison with default Adam hyperparameters (lr=0.001, wd=0.0):")
    default_val_loss = train_model_with_hyperparams(
        0.001, 0.0, X_train, y_train, X_val, y_val, epochs=20
    )
    print(f"Default validation loss: {default_val_loss:.6f}")
    
    improvement = ((default_val_loss - final_val_loss) / default_val_loss) * 100
    print(f"\nImprovement: {improvement:.1f}%")


if __name__ == "__main__":
    main()

