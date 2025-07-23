#!/usr/bin/env python3
"""
GPU Benchmark for V2 Improvements
Tests the key improvements on GPU hardware
"""

import torch
import torch.nn as nn
import time
import sys

sys.path.insert(0, '.')

from pytorch_bfo_optimizer import BFO  # V1
from pytorch_bfo_optimizer.optimizer_v2 import BFOv2, HybridBFOv2  # V2


def benchmark_convergence():
    """Compare convergence speed V1 vs V2"""
    print("\n" + "="*60)
    print("Convergence Benchmark (Rosenbrock Function)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Rosenbrock function
    def rosenbrock(x):
        with torch.no_grad():
            n = len(x) - 1
            return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2).item()
    
    dim = 10
    iterations = 30
    
    # Test V1
    print("\nTesting V1 (Original BFO)...")
    x_v1 = nn.Parameter(torch.ones(dim, device=device) * 2.0)
    opt_v1 = BFO([x_v1], population_size=20, compile_mode=False, verbose=False)
    
    losses_v1 = []
    start = time.time()
    for i in range(iterations):
        loss = opt_v1.step(lambda: rosenbrock(x_v1))
        losses_v1.append(loss)
        if i % 10 == 0:
            print(f"  Iteration {i}: {loss:.4f}")
    time_v1 = time.time() - start
    
    # Test V2
    print("\nTesting V2 (Improved BFO)...")
    x_v2 = nn.Parameter(torch.ones(dim, device=device) * 2.0)
    opt_v2 = BFOv2([x_v2], population_size=20, device_type='auto', early_stopping=True)
    
    losses_v2 = []
    start = time.time()
    for i in range(iterations):
        loss = opt_v2.step(lambda: rosenbrock(x_v2))
        losses_v2.append(loss)
        if i % 10 == 0:
            print(f"  Iteration {i}: {loss:.4f}")
        if opt_v2.stagnation_count >= opt_v2.convergence_patience:
            print(f"  Early stopping at iteration {i}")
            break
    time_v2 = time.time() - start
    
    # Results
    print("\nResults:")
    print(f"V1: Final loss = {losses_v1[-1]:.6f}, Time = {time_v1:.2f}s")
    print(f"V2: Final loss = {losses_v2[-1]:.6f}, Time = {time_v2:.2f}s")
    print(f"V2 Improvement: {(losses_v1[-1] - losses_v2[-1]) / losses_v1[-1] * 100:.1f}%")
    print(f"V2 Speedup: {time_v1 / time_v2:.2f}x")


def benchmark_gradient_handling():
    """Test HybridBFOv2 gradient handling"""
    print("\n" + "="*60)
    print("Gradient Handling Benchmark")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Simple neural network
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    ).to(device)
    
    # Generate dummy data
    X = torch.randn(100, 10, device=device)
    y = torch.randn(100, 1, device=device)
    
    # Test with gradients
    print("\nTesting HybridBFOv2 with gradients...")
    opt = HybridBFOv2(model.parameters(), gradient_weight=0.5, population_size=10)
    
    def closure_with_grad():
        opt.zero_grad()
        output = model(X)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()
        return loss.item()
    
    losses = []
    start = time.time()
    for i in range(20):
        loss = opt.step(closure_with_grad)
        losses.append(loss)
        if i % 5 == 0:
            print(f"  Iteration {i}: {loss:.6f}")
    
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Time: {time.time() - start:.2f}s")
    
    # Test without gradients (should still work)
    print("\nTesting HybridBFOv2 without gradients...")
    for p in model.parameters():
        p.requires_grad = False
    
    def closure_no_grad():
        with torch.no_grad():
            output = model(X)
            loss = nn.functional.mse_loss(output, y)
            return loss.item()
    
    try:
        loss = opt.step(closure_no_grad)
        print(f"✓ Works without gradients! Loss: {loss:.6f}")
    except Exception as e:
        print(f"✗ Failed: {e}")


def benchmark_population_sizes():
    """Test odd population sizes"""
    print("\n" + "="*60)
    print("Population Size Benchmark")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sizes = [3, 5, 7, 10, 15, 20]
    
    for size in sizes:
        x = nn.Parameter(torch.randn(10, device=device))
        opt = BFOv2([x], population_size=size, device_type='auto')
        
        start = time.time()
        loss = opt.step(lambda: (x ** 2).sum().item())
        elapsed = time.time() - start
        
        print(f"Population {size:2d}: Loss = {loss:8.4f}, Time = {elapsed*1000:6.1f}ms")


def benchmark_batch_sizes():
    """Test different batch sizes for performance"""
    print("\n" + "="*60)
    print("Batch Size Performance Benchmark")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim = 100
    batch_sizes = [4, 8, 16, 32] if device == 'cuda' else [2, 4, 8]
    
    for batch_size in batch_sizes:
        x = nn.Parameter(torch.randn(dim, device=device))
        opt = BFOv2([x], population_size=50, batch_size=batch_size, parallel_eval=True)
        
        # Time 5 steps
        start = time.time()
        for _ in range(5):
            opt.step(lambda: (x ** 2).sum().item())
        elapsed = time.time() - start
        
        print(f"Batch size {batch_size:2d}: {elapsed/5*1000:6.1f}ms per step")


def main():
    print("PyTorch BFO V2 GPU Benchmark")
    print("=" * 60)
    
    # Check environment
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: Running on CPU")
    
    print(f"PyTorch Version: {torch.__version__}")
    
    # Run benchmarks
    benchmark_convergence()
    benchmark_gradient_handling()
    benchmark_population_sizes()
    benchmark_batch_sizes()
    
    print("\n" + "="*60)
    print("Benchmark Complete!")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()