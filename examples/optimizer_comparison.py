#!/usr/bin/env python
"""
Optimizer Comparison Example

This example compares the performance of BFO-Torch optimizers with
standard PyTorch optimizers on various optimization problems.

Author: Brett G. Bonner
Repository: https://github.com/bbopen/torch_bfo
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Callable

from bfo_torch import BFO, AdaptiveBFO, HybridBFO


def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    """Rosenbrock function - a classic non-convex optimization benchmark."""
    return torch.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    """Rastrigin function - highly multimodal with many local minima."""
    A = 10
    n = len(x)
    return A * n + torch.sum(x**2 - A * torch.cos(2 * np.pi * x))


def ackley(x: torch.Tensor) -> torch.Tensor:
    """Ackley function - many local minima with a global minimum at origin."""
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    sum1 = torch.sum(x**2)
    sum2 = torch.sum(torch.cos(c * x))
    return -a * torch.exp(-b * torch.sqrt(sum1 / n)) - torch.exp(sum2 / n) + a + np.e


def neural_network_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Simple neural network regression loss."""
    output = model(x)
    return nn.functional.mse_loss(output, y)


class SimpleNN(nn.Module):
    """Simple neural network for regression."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)


def optimize_function(
    func: Callable,
    optimizer_class,
    optimizer_kwargs: Dict,
    dim: int = 10,
    num_steps: int = 100,
    initial_point: torch.Tensor = None
) -> List[float]:
    """Optimize a function and return loss history."""
    # Initialize parameter
    if initial_point is None:
        x = torch.randn(dim, requires_grad=True) * 5  # Random initialization
    else:
        x = initial_point.clone().detach().requires_grad_(True)
    
    # Create optimizer
    if optimizer_class in [torch.optim.SGD, torch.optim.Adam]:
        # Standard PyTorch optimizers
        optimizer = optimizer_class([x], **optimizer_kwargs)
    else:
        # BFO-based optimizers
        optimizer = optimizer_class([x], **optimizer_kwargs)
    
    # Optimization loop
    losses = []
    for step in range(num_steps):
        if optimizer_class in [BFO, AdaptiveBFO, HybridBFO]:
            # BFO-style optimization
            def closure():
                optimizer.zero_grad()
                loss = func(x)
                return loss.item()
            
            loss = optimizer.step(closure)
        else:
            # Standard optimization
            optimizer.zero_grad()
            loss = func(x)
            loss.backward()
            optimizer.step()
            loss = loss.item()
        
        losses.append(loss)
        
        # Early stopping for BFO
        if hasattr(optimizer, 'converged') and optimizer.converged:
            losses.extend([loss] * (num_steps - step - 1))
            break
    
    return losses


def compare_optimizers_on_function(
    func: Callable,
    func_name: str,
    dim: int = 10,
    num_steps: int = 200,
    num_runs: int = 5
):
    """Compare different optimizers on a given function."""
    optimizers = {
        'SGD': (torch.optim.SGD, {'lr': 0.01, 'momentum': 0.9}),
        'Adam': (torch.optim.Adam, {'lr': 0.01}),
        'BFO': (BFO, {'lr': 0.01, 'population_size': 30}),
        'AdaptiveBFO': (AdaptiveBFO, {'lr': 0.01, 'population_size': 30}),
        'HybridBFO': (HybridBFO, {'lr': 0.01, 'gradient_weight': 0.3})
    }
    
    results = {name: [] for name in optimizers}
    
    # Run multiple times for statistical significance
    for run in range(num_runs):
        # Same initial point for fair comparison
        initial_point = torch.randn(dim) * 5
        
        for name, (opt_class, opt_kwargs) in optimizers.items():
            losses = optimize_function(
                func, opt_class, opt_kwargs, dim, num_steps, initial_point
            )
            results[name].append(losses)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    for name, all_losses in results.items():
        # Calculate mean and std across runs
        all_losses = np.array(all_losses)
        mean_losses = np.mean(all_losses, axis=0)
        std_losses = np.std(all_losses, axis=0)
        
        plt.plot(mean_losses, label=name, linewidth=2)
        plt.fill_between(
            range(len(mean_losses)),
            mean_losses - std_losses,
            mean_losses + std_losses,
            alpha=0.2
        )
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Optimizer Comparison on {func_name} Function')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'comparison_{func_name.lower()}.png', dpi=150)
    plt.show()
    
    # Print final results
    print(f'\n{func_name} Function Results (final loss ± std):')
    print('-' * 50)
    for name, all_losses in results.items():
        final_losses = [losses[-1] for losses in all_losses]
        print(f'{name:12s}: {np.mean(final_losses):.6f} ± {np.std(final_losses):.6f}')


def compare_on_neural_network():
    """Compare optimizers on neural network training."""
    # Generate synthetic data
    torch.manual_seed(42)
    X = torch.randn(1000, 10)
    true_weights = torch.randn(10, 1)
    y = X @ true_weights + 0.1 * torch.randn(1000, 1)
    
    # Split data
    train_X, test_X = X[:800], X[800:]
    train_y, test_y = y[:800], y[800:]
    
    optimizers = {
        'SGD': (torch.optim.SGD, {'lr': 0.01, 'momentum': 0.9}),
        'Adam': (torch.optim.Adam, {'lr': 0.001}),
        'BFO': (BFO, {'lr': 0.01, 'population_size': 20}),
        'HybridBFO': (HybridBFO, {'lr': 0.01, 'gradient_weight': 0.5})
    }
    
    results = {}
    
    for name, (opt_class, opt_kwargs) in optimizers.items():
        print(f'\nTraining with {name}...')
        model = SimpleNN()
        
        if opt_class in [torch.optim.SGD, torch.optim.Adam]:
            optimizer = opt_class(model.parameters(), **opt_kwargs)
        else:
            optimizer = opt_class(model.parameters(), **opt_kwargs)
        
        train_losses = []
        test_losses = []
        
        # Training loop
        for epoch in range(50):
            # Batch training
            batch_size = 32
            epoch_loss = 0
            
            for i in range(0, len(train_X), batch_size):
                batch_X = train_X[i:i+batch_size]
                batch_y = train_y[i:i+batch_size]
                
                if opt_class in [BFO, AdaptiveBFO, HybridBFO]:
                    def closure():
                        optimizer.zero_grad()
                        loss = neural_network_loss(model, batch_X, batch_y)
                        return loss.item()
                    
                    loss = optimizer.step(closure)
                else:
                    optimizer.zero_grad()
                    loss = neural_network_loss(model, batch_X, batch_y)
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()
                
                epoch_loss += loss
            
            # Evaluate
            with torch.no_grad():
                train_loss = neural_network_loss(model, train_X, train_y).item()
                test_loss = neural_network_loss(model, test_X, test_y).item()
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}')
        
        results[name] = {
            'train': train_losses,
            'test': test_losses
        }
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for name, losses in results.items():
        ax1.plot(losses['train'], label=name, linewidth=2)
        ax2.plot(losses['test'], label=name, linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Test Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('comparison_neural_network.png', dpi=150)
    plt.show()
    
    # Print final results
    print('\nNeural Network Training Results:')
    print('-' * 50)
    for name, losses in results.items():
        print(f'{name:12s}: Train = {losses["train"][-1]:.6f}, Test = {losses["test"][-1]:.6f}')


def main():
    """Run all comparisons."""
    print('BFO-Torch Optimizer Comparison')
    print('=' * 50)
    
    # Test on different optimization problems
    test_functions = [
        (rosenbrock, 'Rosenbrock', 10),
        (rastrigin, 'Rastrigin', 10),
        (ackley, 'Ackley', 10)
    ]
    
    for func, name, dim in test_functions:
        print(f'\nTesting on {name} function...')
        compare_optimizers_on_function(func, name, dim)
    
    # Test on neural network
    print('\nTesting on Neural Network...')
    compare_on_neural_network()
    
    print('\nComparison complete! Check the generated plots.')


if __name__ == '__main__':
    main()