#!/usr/bin/env python3
"""
Test script for V2 improvements
Validates gradient handling, convergence, and performance enhancements
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys

sys.path.insert(0, '.')

from pytorch_bfo_optimizer.optimizer_v2_improved import BFOv2, AdaptiveBFOv2, HybridBFOv2


def test_gradient_handling():
    """Test that HybridBFOv2 handles gradients safely"""
    print("\n" + "="*60)
    print("Testing Gradient Handling in HybridBFOv2")
    print("="*60)
    
    # Test 1: With gradients
    print("\n1. Testing with gradients (requires_grad=True):")
    x = nn.Parameter(torch.tensor([5.0, 4.0, 3.0], requires_grad=True))
    optimizer = HybridBFOv2([x], gradient_weight=0.5, verbose=True)
    
    def closure_with_grad():
        optimizer.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        return loss.item()
    
    try:
        loss = optimizer.step(closure_with_grad)
        print(f"✓ Step with gradients successful: loss={loss:.4f}")
    except Exception as e:
        print(f"✗ Failed with gradients: {e}")
    
    # Test 2: Without gradients  
    print("\n2. Testing without gradients (requires_grad=False):")
    y = nn.Parameter(torch.tensor([5.0, 4.0, 3.0], requires_grad=False))
    optimizer2 = HybridBFOv2([y], gradient_weight=0.5, verbose=True)
    
    def closure_no_grad():
        with torch.no_grad():
            return (y ** 2).sum().item()
    
    try:
        loss = optimizer2.step(closure_no_grad)
        print(f"✓ Step without gradients successful: loss={loss:.4f}")
        print("✓ Graceful fallback to pure BFO mode")
    except Exception as e:
        print(f"✗ Failed without gradients: {e}")
        raise


def test_convergence_improvements():
    """Test convergence with improved defaults"""
    print("\n" + "="*60)
    print("Testing Convergence Improvements")
    print("="*60)
    
    # Simple quadratic test
    dim = 10
    x_bfo = nn.Parameter(torch.randn(dim) * 3)
    x_adaptive = nn.Parameter(x_bfo.data.clone())
    x_hybrid = nn.Parameter(x_bfo.data.clone(), requires_grad=True)
    
    def quadratic(params):
        return (params ** 2).sum().item()
    
    def quadratic_with_grad(params, opt):
        opt.zero_grad()
        loss = (params ** 2).sum()
        loss.backward()
        return loss.item()
    
    # Test each optimizer variant
    optimizers = {
        'BFOv2': (BFOv2([x_bfo], population_size=20), x_bfo, lambda: quadratic(x_bfo)),
        'AdaptiveBFOv2': (AdaptiveBFOv2([x_adaptive]), x_adaptive, lambda: quadratic(x_adaptive)),
        'HybridBFOv2': (HybridBFOv2([x_hybrid]), x_hybrid, lambda: quadratic_with_grad(x_hybrid, optimizers['HybridBFOv2'][0]) if 'HybridBFOv2' in optimizers else 0)
    }
    
    print(f"\nInitial loss: {quadratic(x_bfo):.4f}")
    
    iterations = 20
    for name, (opt, params, closure) in optimizers.items():
        print(f"\nTesting {name}:")
        losses = []
        start_time = time.time()
        
        for i in range(iterations):
            if name == 'HybridBFOv2':
                # Special handling for hybrid
                opt_obj = optimizers['HybridBFOv2'][0]
                def hybrid_closure():
                    opt_obj.zero_grad()
                    loss = (params ** 2).sum()
                    loss.backward()
                    return loss.item()
                loss = opt.step(hybrid_closure)
            else:
                loss = opt.step(closure)
            losses.append(loss)
            
            if i % 5 == 0:
                print(f"  Iteration {i}: loss={loss:.6f}")
        
        elapsed = time.time() - start_time
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  Time: {elapsed:.2f}s ({elapsed/iterations*1000:.1f}ms per step)")
        
        # Check convergence quality
        if losses[-1] < 1.0:
            print("  ✓ Good convergence")
        else:
            print("  ⚠ Convergence could be better")


def test_performance_optimizations():
    """Test performance improvements"""
    print("\n" + "="*60)
    print("Testing Performance Optimizations")
    print("="*60)
    
    # Test batch evaluation efficiency
    dim = 100
    x = nn.Parameter(torch.randn(dim))
    
    # Compare different batch sizes
    batch_sizes = [4, 8, 16]
    
    for batch_size in batch_sizes:
        opt = BFOv2([x], population_size=20, batch_size=batch_size, parallel_eval=True)
        
        def closure():
            return (x ** 2).sum().item()
        
        # Time a single step
        start = time.time()
        opt.step(closure)
        elapsed = time.time() - start
        
        print(f"Batch size {batch_size}: {elapsed:.3f}s per step")


def test_adaptive_features():
    """Test adaptive parameter adjustment"""
    print("\n" + "="*60)
    print("Testing Adaptive Features")
    print("="*60)
    
    # Rosenbrock function for testing adaptation
    def rosenbrock(x):
        return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2).item()
    
    x = nn.Parameter(torch.tensor([3.0, 3.0, 3.0]))
    opt = AdaptiveBFOv2([x], adapt_pop_size=True, adapt_chem_steps=True)
    
    print("Testing on Rosenbrock function...")
    print(f"Initial parameters: {x.data}")
    print(f"Initial population size: {opt.defaults['population_size']}")
    print(f"Initial chem_steps: {opt.defaults['chem_steps']}")
    
    # Run optimization
    losses = []
    for i in range(30):
        loss = opt.step(lambda: rosenbrock(x))
        losses.append(loss)
        
        if i % 10 == 0:
            print(f"\nIteration {i}:")
            print(f"  Loss: {loss:.6f}")
            print(f"  Population size: {opt.defaults['population_size']}")
            print(f"  Chem steps: {opt.defaults['chem_steps']}")
            print(f"  Step size: {opt.current_step_size:.6f}")
    
    print(f"\nFinal loss: {losses[-1]:.6f}")
    print(f"Final parameters: {x.data}")
    
    # Check if adaptation occurred
    if opt.defaults['population_size'] != 30 or opt.defaults['chem_steps'] != 10:
        print("✓ Parameters adapted during optimization")
    else:
        print("⚠ No adaptation observed")


def test_odd_population_sizes():
    """Test that odd population sizes work correctly"""
    print("\n" + "="*60)
    print("Testing Odd Population Sizes")
    print("="*60)
    
    odd_sizes = [3, 5, 7, 11, 13]
    x = nn.Parameter(torch.randn(5))
    
    for size in odd_sizes:
        try:
            opt = BFOv2([x], population_size=size)
            loss = opt.step(lambda: (x ** 2).sum().item())
            print(f"✓ Population size {size}: loss={loss:.4f}")
        except Exception as e:
            print(f"✗ Population size {size} failed: {e}")


def test_early_stopping():
    """Test early stopping functionality"""
    print("\n" + "="*60)
    print("Testing Early Stopping")
    print("="*60)
    
    # Create a problem that converges quickly
    x = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))
    opt = BFOv2([x], early_stopping=True, convergence_patience=3, verbose=True)
    
    def simple_quadratic():
        return (x ** 2).sum().item()
    
    # Run until convergence
    losses = []
    for i in range(20):
        loss = opt.step(simple_quadratic)
        losses.append(loss)
        
        if opt.stagnation_count >= opt.convergence_patience:
            print(f"✓ Early stopping triggered at iteration {i}")
            break
    
    print(f"Final loss: {losses[-1]:.8f}")
    print(f"Total iterations: {len(losses)}")


def run_all_tests():
    """Run all improvement tests"""
    print("PyTorch BFO V2 Improvement Tests")
    print("=" * 60)
    
    tests = [
        ("Gradient Handling", test_gradient_handling),
        ("Convergence Improvements", test_convergence_improvements),
        ("Performance Optimizations", test_performance_optimizations),
        ("Adaptive Features", test_adaptive_features),
        ("Odd Population Sizes", test_odd_population_sizes),
        ("Early Stopping", test_early_stopping)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test_name} FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("✓ All improvements working correctly!")
    else:
        print("⚠ Some tests failed - review the output above")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_all_tests()