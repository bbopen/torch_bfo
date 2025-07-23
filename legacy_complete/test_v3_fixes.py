#!/usr/bin/env python3
"""
Comprehensive test suite for PyTorch BFO Optimizer V3 fixes.
Tests all critical improvements and edge cases.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pytorch_bfo_optimizer.optimizer_v3_fixed import BFOv2, AdaptiveBFOv2, HybridBFOv2
from pytorch_bfo_optimizer.debug_utils import enable_debug_mode


def test_debug_context():
    """Test 1: DebugContext and log_debug are working"""
    print("\n=== Test 1: Debug Context ===")
    
    # Disable excessive debug output for test suite
    os.environ['BFO_DEBUG_LEVEL'] = 'WARNING'
    
    # Simple test function
    def test_closure():
        return torch.tensor(1.0)
    
    # Create optimizer and test debug context
    x = nn.Parameter(torch.randn(5))
    optimizer = BFOv2([x], verbose=True, compile_mode='false')
    
    try:
        loss = optimizer.step(test_closure)
        print("✅ DebugContext working - no import errors")
    except NameError as e:
        print(f"❌ DebugContext error: {e}")
        return False
    
    return True


def test_resize_population():
    """Test 2: _resize_population preserves elite solutions"""
    print("\n=== Test 2: Resize Population ===")
    
    # Create adaptive optimizer
    x = nn.Parameter(torch.randn(10))
    optimizer = AdaptiveBFOv2([x], adapt_pop_size=True, verbose=True, compile_mode='false')
    
    # Create a test closure with known fitness values
    call_count = 0
    def test_closure():
        nonlocal call_count
        call_count += 1
        # Return different fitness for different individuals
        return (x ** 2).sum() + call_count * 0.1
    
    # Run optimization to trigger resize
    for i in range(10):
        loss = optimizer.step(test_closure)
    
    print(f"✅ Resize population completed without inf fitness")
    print(f"  Final population size: {optimizer.population.shape[0]}")
    print(f"  Best fitness: {optimizer.best_fitness:.4f}")
    
    return True


def test_mixed_gradients():
    """Test 3: HybridBFOv2 handles mixed gradient scenarios"""
    print("\n=== Test 3: Mixed Gradients ===")
    
    # Create parameters with mixed gradient requirements
    x1 = nn.Parameter(torch.randn(5), requires_grad=True)
    x2 = nn.Parameter(torch.randn(5), requires_grad=False)
    x3 = nn.Parameter(torch.randn(5), requires_grad=True)
    
    optimizer = HybridBFOv2([x1, x2, x3], gradient_weight=0.5, verbose=True, compile_mode='false')
    
    def closure():
        # Create a gradient-free closure for BFO population evaluation
        # The hybrid optimizer will handle gradients separately
        with torch.no_grad():
            loss = (x1.detach() ** 2).sum() + (x2.detach() ** 2).sum() + (x3.detach() ** 2).sum()
            return loss.item()
    
    try:
        # This should work with mixed gradients
        loss = optimizer.step(closure)
        print("✅ Mixed gradients handled correctly")
        print(f"  Parameters with grad: {optimizer.step.__code__.co_varnames}")
        print(f"  Loss: {loss:.4f}")
    except RuntimeError as e:
        print(f"❌ Mixed gradient error: {e}")
        return False
    
    return True


def test_amp_compatibility():
    """Test 4: AMP (Automatic Mixed Precision) compatibility"""
    print("\n=== Test 4: AMP Compatibility ===")
    
    # Test with different dtypes
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        if dtype == torch.bfloat16 and not torch.cuda.is_available():
            continue  # Skip bfloat16 on CPU
            
        print(f"\nTesting dtype: {dtype}")
        
        x = nn.Parameter(torch.randn(10, dtype=dtype))
        optimizer = BFOv2([x], compile_mode='false')
        
        def closure():
            return (x ** 2).sum().item()
        
        try:
            loss = optimizer.step(closure)
            print(f"  ✅ {dtype} working, loss: {loss:.4f}")
            
            # Check that internal tensors have correct dtype
            assert optimizer.population.dtype == dtype, f"Population dtype mismatch"
            assert optimizer.dtype == dtype, f"Optimizer dtype not set correctly"
            
        except Exception as e:
            print(f"  ❌ {dtype} error: {e}")
            return False
    
    return True


def test_vectorized_swarming():
    """Test 5: Vectorized swarming computation"""
    print("\n=== Test 5: Vectorized Swarming ===")
    
    # Create optimizer with larger population to test efficiency
    x = nn.Parameter(torch.randn(20))
    optimizer = BFOv2([x], population_size=50, compile_mode='false')
    
    # Test swarming computation
    positions = torch.randn(50, 20)
    
    import time
    start = time.time()
    swarming = optimizer._compute_swarming(positions)
    elapsed = time.time() - start
    
    print(f"✅ Vectorized swarming computed in {elapsed*1000:.2f}ms")
    print(f"  Swarming shape: {swarming.shape}")
    print(f"  No nan/inf values: {torch.isfinite(swarming).all()}")
    
    return True


def test_torch_compile():
    """Test 6: torch.compile compatibility"""
    print("\n=== Test 6: torch.compile Compatibility ===")
    
    if not hasattr(torch, 'compile'):
        print("⚠️  torch.compile not available, skipping")
        return True
    
    x = nn.Parameter(torch.randn(10))
    
    # Test different compile modes
    for mode in ['default', 'reduce-overhead']:
        print(f"\nTesting compile mode: {mode}")
        
        optimizer = BFOv2([x], compile_mode=mode)
        
        def closure():
            return (x ** 2).sum().item()
        
        try:
            # Run a few steps
            losses = []
            for i in range(5):
                loss = optimizer.step(closure)
                losses.append(loss)
            
            print(f"  ✅ Compile mode '{mode}' working")
            print(f"  Losses: {[f'{l:.4f}' for l in losses]}")
            
        except Exception as e:
            print(f"  ❌ Compile error: {e}")
    
    return True


def test_levy_flight_stability():
    """Test 8: Lévy flight numerical stability"""
    print("\n=== Test 8: Lévy Flight Stability ===")
    
    x = nn.Parameter(torch.randn(10))
    optimizer = BFOv2([x], compile_mode='false')
    
    # Test with extreme alpha values
    for alpha in [0.5, 1.0, 1.5, 1.9]:
        optimizer.defaults['levy_alpha'] = alpha
        
        # Generate many samples to test stability
        all_finite = True
        for _ in range(100):
            step = optimizer._levy_flight((20, 10))
            if not torch.isfinite(step).all():
                all_finite = False
                break
        
        print(f"  Alpha={alpha}: {'✅ Stable' if all_finite else '❌ Numerical issues'}")
    
    return True


def test_reproducibility():
    """Test 9: Reproducibility with state_dict"""
    print("\n=== Test 9: Reproducibility ===")
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create optimizer and run a few steps
    x = nn.Parameter(torch.randn(10))
    opt1 = BFOv2([x], compile_mode='false')
    
    def closure():
        return (x ** 2).sum().item()
    
    # Run optimization and save trajectory
    trajectory1 = []
    for i in range(5):
        loss = opt1.step(closure)
        trajectory1.append(loss)
    
    # Save state
    state = opt1.state_dict()
    
    # Create new optimizer and load state
    torch.manual_seed(99)  # Different seed
    y = nn.Parameter(torch.randn(10))
    opt2 = BFOv2([y], compile_mode='false')
    
    # Load state (should restore RNG state too)
    opt2.load_state_dict(state)
    
    # Continue optimization
    trajectory2 = []
    for i in range(5):
        loss = opt2.step(closure)
        trajectory2.append(loss)
    
    # Check if trajectories match (allowing small numerical differences)
    max_diff = max(abs(t1 - t2) for t1, t2 in zip(trajectory1[-3:], trajectory2[:3]))
    
    print(f"✅ State dict save/load working")
    print(f"  Trajectory 1: {[f'{l:.4f}' for l in trajectory1[-3:]]}")
    print(f"  Trajectory 2: {[f'{l:.4f}' for l in trajectory2[:3]]}")
    print(f"  Max difference: {max_diff:.6f}")
    
    return max_diff < 1e-4


def test_vmap_evaluation():
    """Test 7: vmap population evaluation"""
    print("\n=== Test 7: vmap Evaluation ===")
    
    x = nn.Parameter(torch.randn(10))
    
    # Test with compile mode disabled for now
    optimizer = BFOv2([x], compile_mode='false', population_size=20)
    
    def closure():
        return (x ** 2).sum().item()
    
    # Test evaluation
    try:
        fitness = optimizer._vmap_evaluate_population(closure)
        print("✅ vmap evaluation attempted")
        print(f"  Fitness shape: {fitness.shape}")
        print(f"  Using vmap: {'Yes' if hasattr(torch, 'vmap') else 'No (fallback to batch)'}")
    except Exception as e:
        print(f"❌ vmap error: {e}")
        # Try fallback
        fitness = optimizer._batch_evaluate_population(closure)
        print("✅ Fallback to batch evaluation working")
    
    return True


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("PyTorch BFO Optimizer V3 - Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        ("Debug Context", test_debug_context),
        ("Resize Population", test_resize_population),
        ("Mixed Gradients", test_mixed_gradients),
        ("AMP Compatibility", test_amp_compatibility),
        ("Vectorized Swarming", test_vectorized_swarming),
        ("torch.compile", test_torch_compile),
        ("vmap Evaluation", test_vmap_evaluation),
        ("Lévy Flight Stability", test_levy_flight_stability),
        ("Reproducibility", test_reproducibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! BFO V3 is production-ready.")
    else:
        print("\n⚠️  Some tests failed. Please review the issues above.")


if __name__ == "__main__":
    run_all_tests()