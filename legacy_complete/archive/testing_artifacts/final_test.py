#!/usr/bin/env python3
"""
Final test for BFO optimizer - testing the fixed version
"""

import torch
import torch.nn as nn
import time
import sys

sys.path.insert(0, '.')

from pytorch_bfo_optimizer import BFO


def test_rosenbrock():
    """Test on Rosenbrock function"""
    print("\n" + "="*60)
    print("Testing BFO on Rosenbrock Function")
    print("="*60)
    
    dim = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Initialize parameters far from optimum
    x = nn.Parameter(torch.ones(dim, device=device) * 3.0)
    print(f"Initial x: {x.data[:5]}... (showing first 5)")
    
    # Rosenbrock function
    def closure():
        with torch.no_grad():
            loss = 0
            for i in range(dim - 1):
                loss += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
            return loss.item()
    
    initial_loss = closure()
    print(f"Initial loss: {initial_loss:.4f}")
    
    # Create optimizer
    optimizer = BFO(
        [x],
        population_size=20,
        chem_steps=10,
        swim_length=4,
        repro_steps=4,
        elim_steps=2,
        compile_mode=False,
        verbose=False
    )
    
    # Optimize
    print("\nOptimizing...")
    losses = []
    start_time = time.time()
    
    for i in range(50):
        loss = optimizer.step(closure)
        losses.append(loss)
        
        if i % 10 == 0:
            print(f"  Iteration {i}: Loss = {loss:.6f}")
    
    elapsed = time.time() - start_time
    
    print(f"\nOptimization complete!")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Improvement: {(initial_loss - losses[-1]) / initial_loss * 100:.1f}%")
    print(f"Time: {elapsed:.2f}s ({elapsed/50*1000:.1f}ms per step)")
    print(f"Final x: {x.data[:5]}... (showing first 5)")


def test_simple_quadratic():
    """Test on simple quadratic function"""
    print("\n" + "="*60)
    print("Testing BFO on Simple Quadratic Function")
    print("="*60)
    
    dim = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Initialize parameters randomly
    x = nn.Parameter(torch.randn(dim, device=device) * 5.0)
    initial_x = x.data.clone()
    
    # Simple quadratic: sum of x^2
    def closure():
        with torch.no_grad():
            return (x ** 2).sum().item()
    
    initial_loss = closure()
    print(f"Initial loss: {initial_loss:.4f}")
    
    # Create optimizer with smaller population for faster convergence
    optimizer = BFO(
        [x],
        population_size=10,
        chem_steps=5,
        swim_length=4,
        repro_steps=2,
        elim_steps=1,
        compile_mode=False,
        verbose=False
    )
    
    # Optimize
    print("\nOptimizing...")
    losses = []
    start_time = time.time()
    
    for i in range(20):
        loss = optimizer.step(closure)
        losses.append(loss)
        
        if i % 5 == 0:
            print(f"  Iteration {i}: Loss = {loss:.6f}")
    
    elapsed = time.time() - start_time
    
    print(f"\nOptimization complete!")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Improvement: {(initial_loss - losses[-1]) / initial_loss * 100:.1f}%")
    print(f"Time: {elapsed:.2f}s ({elapsed/20*1000:.1f}ms per step)")
    
    # Check how close we got to zero
    distance_from_zero = torch.norm(x.data).item()
    print(f"Distance from zero: {distance_from_zero:.4f}")


def main():
    print("PyTorch BFO Optimizer - Final Test")
    print("=" * 60)
    
    # Check environment
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Running on CPU")
    
    print(f"PyTorch Version: {torch.__version__}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run tests
    test_simple_quadratic()
    test_rosenbrock()
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()