#!/usr/bin/env python3
"""
Reduced benchmark test for GPU server
Tests only essential optimizers with minimal iterations
"""

import torch
import torch.nn as nn
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, '.')

from pytorch_bfo_optimizer import BFO, HybridBFO
from pytorch_bfo_optimizer.optimizer_v2 import BFOv2, HybridBFOv2


def rosenbrock_test():
    """Simple Rosenbrock function test"""
    dim = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nTesting on {device}")
    
    # Test function
    x = nn.Parameter(torch.randn(dim, device=device))
    
    def closure():
        loss = 0
        for i in range(dim - 1):
            loss += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return loss
    
    # Test optimizers
    optimizers = {
        'Adam': torch.optim.Adam([x], lr=0.01),
        'BFO_original': BFO([x], population_size=4, compile_mode=False, verbose=True),
        'BFOv2': BFOv2([x], device_type='auto', parallel_eval=True, verbose=True),
        'HybridBFOv2': HybridBFOv2([x], gradient_weight=0.5, device_type='auto', verbose=True),
    }
    
    results = {}
    iterations = 20
    
    for name, optimizer in optimizers.items():
        print(f"\n{'='*60}")
        print(f"Testing {name}")
        print(f"{'='*60}")
        
        # Reset parameters
        x.data = torch.randn_like(x.data)
        
        start_time = time.time()
        losses = []
        
        for i in range(iterations):
            if 'BFO' in name and 'Hybrid' not in name:
                # Pure BFO doesn't need gradients
                def no_grad_closure():
                    with torch.no_grad():
                        return closure().item()
                loss = optimizer.step(no_grad_closure)
            elif 'Hybrid' in name:
                # Hybrid needs gradients
                def grad_closure():
                    optimizer.zero_grad()
                    loss = closure()
                    loss.backward()
                    return loss.item()
                loss = optimizer.step(grad_closure)
            else:
                # Standard optimizers
                optimizer.zero_grad()
                loss = closure()
                loss.backward()
                optimizer.step()
                loss = loss.item()
            
            losses.append(loss)
            if i % 5 == 0:
                print(f"  Iteration {i}: Loss = {loss:.6f}")
        
        elapsed = time.time() - start_time
        
        results[name] = {
            'final_loss': losses[-1],
            'best_loss': min(losses),
            'time': elapsed,
            'time_per_step': elapsed / iterations
        }
        
        print(f"\nResults for {name}:")
        print(f"  Final loss: {results[name]['final_loss']:.6f}")
        print(f"  Best loss: {results[name]['best_loss']:.6f}")
        print(f"  Total time: {results[name]['time']:.2f}s")
        print(f"  Time per step: {results[name]['time_per_step']*1000:.1f}ms")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Optimizer':<15} {'Best Loss':<12} {'Time/Step (ms)':<15}")
    print(f"{'-'*42}")
    
    for name, res in sorted(results.items(), key=lambda x: x[1]['best_loss']):
        print(f"{name:<15} {res['best_loss']:<12.6f} {res['time_per_step']*1000:<15.1f}")


def main():
    print("PyTorch BFO Optimizer - Reduced Benchmark Test")
    print("=" * 60)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Running on CPU")
    
    print(f"PyTorch Version: {torch.__version__}")
    
    # Set random seed
    torch.manual_seed(42)
    
    # Run tests
    rosenbrock_test()
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()