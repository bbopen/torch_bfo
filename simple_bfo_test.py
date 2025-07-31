#!/usr/bin/env python3
"""
Simple BFO Verification Test
===========================

A simplified test to verify the bfo_torch implementation works correctly.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import json

# Import our BFO implementation
from src.bfo_torch.optimizer import BFO, AdaptiveBFO, HybridBFO


def test_sphere_function():
    """Test BFO on the sphere function."""
    print("Testing BFO on Sphere function...")
    
    # Sphere function: f(x) = sum(x_i^2), global min at x=0
    x = nn.Parameter(torch.randn(5) * 2.0)
    optimizer = BFO([x], population_size=20, lr=0.01)
    
    def closure():
        return torch.sum(x**2).item()
    
    initial_loss = closure()
    print(f"Initial loss: {initial_loss:.6f}")
    
    # Run optimization
    for step in range(20):
        loss = optimizer.step(closure)
        if step % 5 == 0:
            print(f"Step {step}: loss = {loss:.6f}")
    
    final_loss = closure()
    print(f"Final loss: {final_loss:.6f}")
    print(f"Improvement: {initial_loss - final_loss:.6f}")
    
    return {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'improvement': initial_loss - final_loss,
        'success': final_loss < 0.1  # Should be close to 0
    }


def test_rosenbrock_function():
    """Test BFO on the Rosenbrock function."""
    print("\nTesting BFO on Rosenbrock function...")
    
    # Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    x = nn.Parameter(torch.randn(2) * 2.0)
    optimizer = BFO([x], population_size=30, lr=0.01)
    
    def closure():
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    initial_loss = closure().item()
    print(f"Initial loss: {initial_loss:.6f}")
    
    # Run optimization
    for step in range(30):
        loss = optimizer.step(closure)
        if step % 10 == 0:
            print(f"Step {step}: loss = {loss:.6f}")
    
    final_loss = closure().item()
    print(f"Final loss: {final_loss:.6f}")
    print(f"Improvement: {initial_loss - final_loss:.6f}")
    
    return {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'improvement': initial_loss - final_loss,
        'success': final_loss < 1.0  # Should be close to 0
    }


def test_optimizer_variants():
    """Test different BFO variants."""
    print("\nTesting BFO variants...")
    
    x = nn.Parameter(torch.randn(3) * 2.0)
    
    optimizers = {
        'BFO': BFO([x], population_size=20, lr=0.01),
        'AdaptiveBFO': AdaptiveBFO([x], population_size=20, lr=0.01, adaptation_rate=0.1),
        'HybridBFO': HybridBFO([x], population_size=20, lr=0.01, gradient_weight=0.5)
    }
    
    results = {}
    
    for name, optimizer in optimizers.items():
        print(f"\nTesting {name}...")
        
        # Reset parameter
        with torch.no_grad():
            x.data = torch.randn(3) * 2.0
        
        def closure():
            return torch.sum(x**2).item()
        
        initial_loss = closure()
        print(f"  Initial loss: {initial_loss:.6f}")
        
        # Run optimization
        for step in range(15):
            loss = optimizer.step(closure)
            if step % 5 == 0:
                print(f"  Step {step}: loss = {loss:.6f}")
        
        final_loss = closure()
        print(f"  Final loss: {final_loss:.6f}")
        
        results[name] = {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'improvement': initial_loss - final_loss,
            'success': final_loss < 0.1
        }
    
    return results


def test_against_other_optimizers():
    """Compare BFO against other PyTorch optimizers."""
    print("\nComparing BFO against other optimizers...")
    
    x = nn.Parameter(torch.randn(3) * 2.0)
    
    optimizers = {
        'BFO': BFO([x], population_size=20, lr=0.01),
        'SGD': torch.optim.SGD([x], lr=0.01),
        'Adam': torch.optim.Adam([x], lr=0.01)
    }
    
    results = {}
    
    for name, optimizer in optimizers.items():
        print(f"\nTesting {name}...")
        
        # Reset parameter
        with torch.no_grad():
            x.data = torch.randn(3) * 2.0
        
        def closure():
            return torch.sum(x**2).item()
        
        initial_loss = closure()
        print(f"  Initial loss: {initial_loss:.6f}")
        
        start_time = time.time()
        
        # Run optimization
        for step in range(20):
            if name == 'BFO':
                loss = optimizer.step(closure)
            else:
                optimizer.zero_grad()
                loss_tensor = torch.sum(x**2)
                loss_tensor.backward()
                optimizer.step()
                loss = loss_tensor.item()
            
            if step % 5 == 0:
                print(f"  Step {step}: loss = {loss:.6f}")
        
        end_time = time.time()
        final_loss = closure()
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Time: {end_time - start_time:.3f}s")
        
        results[name] = {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'improvement': initial_loss - final_loss,
            'time': end_time - start_time,
            'success': final_loss < 0.1
        }
    
    return results


def main():
    """Run all tests."""
    print("=" * 50)
    print("BFO IMPLEMENTATION VERIFICATION")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    sphere_result = test_sphere_function()
    rosenbrock_result = test_rosenbrock_function()
    variants_result = test_optimizer_variants()
    comparison_result = test_against_other_optimizers()
    
    # Compile results
    all_results = {
        'sphere_function': sphere_result,
        'rosenbrock_function': rosenbrock_result,
        'optimizer_variants': variants_result,
        'optimizer_comparison': comparison_result
    }
    
    # Save results
    with open('simple_bfo_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    print(f"Sphere function test: {'✓ PASS' if sphere_result['success'] else '✗ FAIL'}")
    print(f"Rosenbrock function test: {'✓ PASS' if rosenbrock_result['success'] else '✗ FAIL'}")
    
    print("\nOptimizer variants:")
    for name, result in variants_result.items():
        status = '✓ PASS' if result['success'] else '✗ FAIL'
        print(f"  {name}: {status}")
    
    print("\nOptimizer comparison:")
    for name, result in comparison_result.items():
        status = '✓ PASS' if result['success'] else '✗ FAIL'
        print(f"  {name}: {status} (time: {result['time']:.3f}s)")
    
    # Overall success
    all_tests = [sphere_result, rosenbrock_result] + list(variants_result.values()) + list(comparison_result.values())
    success_count = sum(1 for test in all_tests if test['success'])
    total_count = len(all_tests)
    
    print(f"\nOverall: {success_count}/{total_count} tests passed")
    
    if success_count == total_count:
        print("🎉 All tests passed! BFO implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")
    
    print(f"\nResults saved to: simple_bfo_test_results.json")


if __name__ == "__main__":
    main()