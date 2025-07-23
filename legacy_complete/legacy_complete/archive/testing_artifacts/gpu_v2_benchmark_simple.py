#!/usr/bin/env python3
"""
Simple GPU Benchmark for V2 Improvements
Focus on key improvements without torch.compile
"""

import torch
import torch.nn as nn
import time
import sys

sys.path.insert(0, '.')

from pytorch_bfo_optimizer import BFO  # V1
from pytorch_bfo_optimizer.optimizer_v2 import BFOv2, HybridBFOv2  # V2


def test_gradient_handling():
    """Test that HybridBFOv2 handles gradients safely"""
    print("\n" + "="*60)
    print("TEST 1: Gradient Handling (HybridBFOv2)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test with gradients
    print("\nWith gradients (requires_grad=True):")
    x1 = nn.Parameter(torch.tensor([5.0, 4.0, 3.0], device=device, requires_grad=True))
    opt1 = HybridBFOv2([x1], gradient_weight=0.5, compile_mode=False)
    
    def closure_grad():
        opt1.zero_grad()
        loss = (x1 ** 2).sum()
        loss.backward()
        return loss.item()
    
    try:
        loss = opt1.step(closure_grad)
        print(f"✓ SUCCESS: Loss = {loss:.4f}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
    
    # Test without gradients
    print("\nWithout gradients (requires_grad=False):")
    x2 = nn.Parameter(torch.tensor([5.0, 4.0, 3.0], device=device, requires_grad=False))
    opt2 = HybridBFOv2([x2], gradient_weight=0.5, compile_mode=False)
    
    def closure_no_grad():
        with torch.no_grad():
            return (x2 ** 2).sum().item()
    
    try:
        loss = opt2.step(closure_no_grad)
        print(f"✓ SUCCESS: Falls back to pure BFO, Loss = {loss:.4f}")
    except Exception as e:
        print(f"✗ FAILED: {e}")


def test_convergence_speed():
    """Compare convergence V1 vs V2"""
    print("\n" + "="*60)
    print("TEST 2: Convergence Speed Comparison")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim = 10
    iterations = 20
    
    # Simple quadratic function
    def quadratic(x):
        return (x ** 2).sum().item()
    
    # V1 Test
    x_v1 = nn.Parameter(torch.randn(dim, device=device) * 3)
    opt_v1 = BFO([x_v1], population_size=20, compile_mode=False)
    
    print("\nV1 (Original BFO):")
    losses_v1 = []
    start = time.time()
    for i in range(iterations):
        loss = opt_v1.step(lambda: quadratic(x_v1))
        losses_v1.append(loss)
        if i % 5 == 0:
            print(f"  Iteration {i}: {loss:.4f}")
    time_v1 = time.time() - start
    
    # V2 Test
    x_v2 = nn.Parameter(torch.randn(dim, device=device) * 3)
    opt_v2 = BFOv2([x_v2], population_size=20, compile_mode=False, early_stopping=True)
    
    print("\nV2 (Improved BFO):")
    losses_v2 = []
    start = time.time()
    stopped_early = False
    for i in range(iterations):
        loss = opt_v2.step(lambda: quadratic(x_v2))
        losses_v2.append(loss)
        if i % 5 == 0:
            print(f"  Iteration {i}: {loss:.4f}")
        if opt_v2.stagnation_count >= opt_v2.convergence_patience:
            print(f"  Early stopping triggered at iteration {i}")
            stopped_early = True
            break
    time_v2 = time.time() - start
    
    print("\n--- Results ---")
    print(f"V1: Final loss = {losses_v1[-1]:.6f}, Time = {time_v1:.2f}s")
    print(f"V2: Final loss = {losses_v2[-1]:.6f}, Time = {time_v2:.2f}s")
    if losses_v1[-1] > 0:
        print(f"V2 Improvement: {(losses_v1[-1] - losses_v2[-1]) / losses_v1[-1] * 100:.1f}%")
    if stopped_early:
        print(f"V2 saved {iterations - len(losses_v2)} iterations via early stopping")


def test_odd_populations():
    """Test odd population sizes work"""
    print("\n" + "="*60)
    print("TEST 3: Odd Population Sizes")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    odd_sizes = [3, 5, 7, 11]
    
    print("Testing V2 with odd population sizes:")
    for size in odd_sizes:
        x = nn.Parameter(torch.randn(5, device=device))
        opt = BFOv2([x], population_size=size, compile_mode=False)
        
        try:
            loss = opt.step(lambda: (x ** 2).sum().item())
            print(f"  Size {size}: ✓ SUCCESS (loss = {loss:.4f})")
        except Exception as e:
            print(f"  Size {size}: ✗ FAILED - {e}")


def test_performance():
    """Test batch size performance"""
    print("\n" + "="*60)
    print("TEST 4: Batch Size Performance")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim = 100
    pop_size = 50
    
    batch_sizes = [4, 8, 16] if device == 'cuda' else [2, 4]
    
    for batch_size in batch_sizes:
        x = nn.Parameter(torch.randn(dim, device=device))
        opt = BFOv2([x], population_size=pop_size, batch_size=batch_size, 
                   compile_mode=False, parallel_eval=True)
        
        # Time 3 steps
        start = time.time()
        for _ in range(3):
            opt.step(lambda: (x ** 2).sum().item())
        elapsed = time.time() - start
        
        print(f"Batch size {batch_size}: {elapsed/3*1000:.1f}ms per step")


def main():
    print("PyTorch BFO V2 GPU Benchmark (Simple)")
    print("=" * 60)
    
    # Environment info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("Running on CPU")
    print(f"PyTorch: {torch.__version__}")
    
    # Run tests
    test_gradient_handling()
    test_convergence_speed()
    test_odd_populations()
    test_performance()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()