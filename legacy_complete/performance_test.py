#!/usr/bin/env python3
"""
Performance demonstration of PyTorch BFO Optimizer V3.
Tests optimization on real mathematical functions.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pytorch_bfo_optimizer.optimizer_v3_fixed import BFOv2, AdaptiveBFOv2, HybridBFOv2


def performance_test():
    """Run performance tests on various optimization problems"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch BFO Optimizer V3 - Performance Test")
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    tests = [
        ("Rosenbrock 2D", test_rosenbrock_2d),
        ("Sphere Function", test_sphere_function), 
        ("Rastrigin Function", test_rastrigin_function),
        ("Neural Network Training", test_neural_network),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n=== {test_name} ===")
        try:
            result = test_func(device)
            results.append((test_name, result))
            print(f"✅ {test_name}: {result}")
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, f"Failed: {e}"))
    
    print("\n" + "=" * 60)
    print("Performance Test Summary:")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅" if not result.startswith("Failed") else "❌"
        print(f"{test_name:.<25} {status}")
        if not result.startswith("Failed"):
            print(f"  Result: {result}")
    
    print(f"\nTotal: {sum(1 for _, r in results if not r.startswith('Failed'))}/{len(results)} tests successful")


def test_rosenbrock_2d(device):
    """Test on 2D Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²"""
    a, b = 1.0, 100.0
    x = nn.Parameter(torch.tensor([-2.0, 2.0], device=device))
    
    optimizer = BFOv2([x], population_size=30, compile_mode='false', 
                     convergence_patience=10, verbose=False)
    
    def closure():
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    
    start_time = time.time()
    initial_loss = closure().item()
    
    for i in range(50):
        loss = optimizer.step(closure)
        if loss < 0.01:  # Good enough convergence
            break
    
    final_loss = loss
    elapsed = time.time() - start_time
    
    return f"Loss: {initial_loss:.4f} → {final_loss:.4f} in {elapsed:.2f}s ({i+1} iters)"


def test_sphere_function(device):
    """Test on N-dimensional sphere function: f(x) = Σx²"""
    n_dims = 10
    x = nn.Parameter(torch.randn(n_dims, device=device) * 5)  # Start far from optimum
    
    optimizer = AdaptiveBFOv2([x], population_size=25, compile_mode='false',
                             convergence_patience=8, verbose=False)
    
    def closure():
        return (x**2).sum()
    
    start_time = time.time()
    initial_loss = closure().item()
    
    for i in range(30):
        loss = optimizer.step(closure)
        if loss < 0.001:
            break
    
    final_loss = loss
    elapsed = time.time() - start_time
    
    return f"Loss: {initial_loss:.4f} → {final_loss:.4f} in {elapsed:.2f}s ({i+1} iters)"


def test_rastrigin_function(device):
    """Test on Rastrigin function (multimodal): f(x) = A*n + Σ(x² - A*cos(2πx))"""
    n_dims = 5
    A = 10.0
    x = nn.Parameter(torch.randn(n_dims, device=device) * 3)
    
    optimizer = BFOv2([x], population_size=40, compile_mode='false',
                     convergence_patience=15, verbose=False)
    
    def closure():
        return A * n_dims + ((x**2) - A * torch.cos(2 * np.pi * x)).sum()
    
    start_time = time.time()
    initial_loss = closure().item()
    
    for i in range(60):
        loss = optimizer.step(closure)
        if loss < 1.0:  # Rastrigin is harder to optimize
            break
    
    final_loss = loss
    elapsed = time.time() - start_time
    
    return f"Loss: {initial_loss:.4f} → {final_loss:.4f} in {elapsed:.2f}s ({i+1} iters)"


def test_neural_network(device):
    """Test optimization of a small neural network"""
    # Create synthetic data
    torch.manual_seed(42)
    X = torch.randn(100, 5, device=device)
    y = torch.sum(X[:, :3], dim=1) + 0.1 * torch.randn(100, device=device)
    
    # Simple neural network
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    ).to(device)
    
    # Use HybridBFOv2 to combine BFO with gradient information
    optimizer = HybridBFOv2(model.parameters(), 
                           gradient_weight=0.3, population_size=20,
                           compile_mode='false', convergence_patience=8)
    
    def closure():
        # Gradient-free closure for BFO evaluation
        with torch.no_grad():
            pred = model(X).squeeze()
            loss = nn.MSELoss()(pred, y)
            return loss.item()
    
    start_time = time.time()
    initial_loss = closure()
    
    for i in range(25):
        loss = optimizer.step(closure)
        if loss < 0.1:
            break
    
    final_loss = loss
    elapsed = time.time() - start_time
    
    return f"MSE: {initial_loss:.4f} → {final_loss:.4f} in {elapsed:.2f}s ({i+1} iters)"


if __name__ == "__main__":
    performance_test()