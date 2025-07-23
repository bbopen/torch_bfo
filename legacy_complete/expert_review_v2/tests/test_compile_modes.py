#!/usr/bin/env python3
"""
Test cases for torch.compile compatibility
Demonstrates issues and validates solutions for compilation
"""

import torch
import torch.nn as nn
import time
import sys

sys.path.append('../src')


def test_graph_breaks():
    """Demonstrate torch.compile graph breaks"""
    print("Testing torch.compile Graph Breaks")
    print("=" * 60)
    
    # Simple function with .item() calls
    def problematic_function(x):
        """Function that causes graph breaks"""
        result = torch.sum(x ** 2)
        # This .item() call breaks the graph
        scalar_result = result.item()
        return scalar_result
    
    # Test compilation
    print("\nTest 1: Compiling function with .item() calls")
    x = torch.randn(10)
    
    try:
        compiled_fn = torch.compile(problematic_function, mode='reduce-overhead')
        result = compiled_fn(x)
        print(f"✓ Compiled and executed (with graph breaks), result = {result:.4f}")
        print("  Note: Check warnings for graph break messages")
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
    
    # Better version without .item()
    def graph_friendly_function(x):
        """Function without graph breaks"""
        return torch.sum(x ** 2)
    
    print("\nTest 2: Compiling graph-friendly function")
    try:
        compiled_fn2 = torch.compile(graph_friendly_function, mode='reduce-overhead', fullgraph=True)
        result2 = compiled_fn2(x)
        print(f"✓ Compiled without graph breaks, result = {result2:.4f}")
    except Exception as e:
        print(f"✗ Compilation failed: {e}")


def test_population_evaluation_compilation():
    """Test compilation of population evaluation patterns"""
    print("\n\nTesting Population Evaluation Compilation")
    print("=" * 60)
    
    # Current pattern (not compile-friendly)
    def evaluate_population_current(population, closure):
        """Current evaluation pattern with loops"""
        fitness = torch.zeros(population.shape[0])
        for i in range(population.shape[0]):
            # This pattern is hard to compile efficiently
            x = population[i]
            fitness[i] = closure(x)
        return fitness
    
    # Compile-friendly pattern using vmap
    def evaluate_population_vmap(population, closure):
        """Vectorized evaluation using vmap"""
        # vmap automatically vectorizes the closure
        return torch.vmap(closure)(population)
    
    # Test setup
    pop_size = 20
    dim = 10
    population = torch.randn(pop_size, dim)
    
    def simple_closure(x):
        return torch.sum(x ** 2)
    
    # Test current pattern
    print("\nTest 1: Current loop-based pattern")
    start = time.time()
    result1 = evaluate_population_current(population, simple_closure)
    time1 = time.time() - start
    print(f"  Non-compiled time: {time1*1000:.2f}ms")
    
    try:
        compiled_current = torch.compile(evaluate_population_current, mode='reduce-overhead')
        start = time.time()
        result1_compiled = compiled_current(population, simple_closure)
        time1_compiled = time.time() - start
        print(f"  Compiled time: {time1_compiled*1000:.2f}ms")
        print(f"  Speedup: {time1/time1_compiled:.2f}x")
    except Exception as e:
        print(f"  Compilation failed: {e}")
    
    # Test vmap pattern
    print("\nTest 2: Vectorized vmap pattern")
    start = time.time()
    result2 = evaluate_population_vmap(population, simple_closure)
    time2 = time.time() - start
    print(f"  Non-compiled time: {time2*1000:.2f}ms")
    
    try:
        compiled_vmap = torch.compile(evaluate_population_vmap, mode='reduce-overhead')
        start = time.time()
        result2_compiled = compiled_vmap(population, simple_closure)
        time2_compiled = time.time() - start
        print(f"  Compiled time: {time2_compiled*1000:.2f}ms")
        print(f"  Speedup: {time2/time2_compiled:.2f}x")
    except Exception as e:
        print(f"  Compilation failed: {e}")


def test_optimization_step_compilation():
    """Test compilation of optimization step"""
    print("\n\nTesting Optimization Step Compilation")
    print("=" * 60)
    
    class SimpleOptimizer:
        def __init__(self, population_size=10, dim=5):
            self.population = torch.randn(population_size, dim)
            self.best_fitness = float('inf')
            self.best_params = torch.zeros(dim)
        
        def step_with_item_calls(self, closure):
            """Step with .item() calls (causes graph breaks)"""
            fitness = torch.zeros(self.population.shape[0])
            
            for i in range(self.population.shape[0]):
                fitness[i] = closure(self.population[i])
            
            # These .item() calls break the graph
            min_idx = torch.argmin(fitness).item()
            min_fitness = fitness[min_idx].item()
            
            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                self.best_params = self.population[min_idx].clone()
            
            return self.best_fitness
        
        def step_compile_friendly(self, closure):
            """Step without .item() calls"""
            fitness = torch.vmap(closure)(self.population)
            
            # Keep everything as tensors
            min_fitness, min_idx = torch.min(fitness, dim=0)
            
            # Use tensor operations instead of Python conditionals
            improved = min_fitness < self.best_fitness
            self.best_fitness = torch.where(improved, min_fitness, torch.tensor(self.best_fitness))
            self.best_params = torch.where(
                improved.unsqueeze(-1),
                self.population[min_idx],
                self.best_params
            )
            
            return self.best_fitness
    
    # Test both versions
    def test_closure(x):
        return torch.sum(x ** 2)
    
    print("\nTest 1: Step with .item() calls")
    opt1 = SimpleOptimizer()
    
    # Non-compiled
    start = time.time()
    for _ in range(10):
        opt1.step_with_item_calls(test_closure)
    time1 = time.time() - start
    print(f"  Non-compiled time: {time1*1000:.2f}ms")
    
    # Try to compile (will have graph breaks)
    opt1_compiled = SimpleOptimizer()
    try:
        compiled_step = torch.compile(opt1_compiled.step_with_item_calls, mode='reduce-overhead')
        start = time.time()
        for _ in range(10):
            compiled_step(test_closure)
        time1_compiled = time.time() - start
        print(f"  Compiled time: {time1_compiled*1000:.2f}ms")
        print("  Note: Likely has graph breaks")
    except Exception as e:
        print(f"  Compilation failed: {e}")
    
    print("\nTest 2: Compile-friendly step")
    opt2 = SimpleOptimizer()
    
    # Non-compiled
    start = time.time()
    for _ in range(10):
        opt2.step_compile_friendly(test_closure)
    time2 = time.time() - start
    print(f"  Non-compiled time: {time2*1000:.2f}ms")
    
    # Compiled
    opt2_compiled = SimpleOptimizer()
    try:
        compiled_step2 = torch.compile(opt2_compiled.step_compile_friendly, 
                                      mode='reduce-overhead',
                                      fullgraph=True)
        start = time.time()
        for _ in range(10):
            compiled_step2(test_closure)
        time2_compiled = time.time() - start
        print(f"  Compiled time: {time2_compiled*1000:.2f}ms")
        print(f"  Speedup: {time2/time2_compiled:.2f}x")
    except Exception as e:
        print(f"  Compilation failed: {e}")


def test_compile_modes():
    """Test different torch.compile modes"""
    print("\n\nTesting Different Compile Modes")
    print("=" * 60)
    
    def optimizer_kernel(population, directions):
        """Simple optimizer kernel for testing"""
        # Chemotaxis step
        new_positions = population + 0.01 * directions
        # Simple boundary
        new_positions = torch.clamp(new_positions, -5, 5)
        return new_positions
    
    # Test data
    pop_size = 100
    dim = 50
    population = torch.randn(pop_size, dim)
    directions = torch.randn(pop_size, dim)
    
    modes = ['default', 'reduce-overhead', 'max-autotune']
    
    for mode in modes:
        print(f"\nTesting mode: {mode}")
        
        try:
            compiled_kernel = torch.compile(optimizer_kernel, mode=mode)
            
            # Warm up
            _ = compiled_kernel(population, directions)
            
            # Time execution
            start = time.time()
            for _ in range(100):
                result = compiled_kernel(population, directions)
            elapsed = time.time() - start
            
            print(f"  ✓ Success: {elapsed*1000:.2f}ms for 100 iterations")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # Compare with non-compiled
    print("\nNon-compiled baseline:")
    start = time.time()
    for _ in range(100):
        result = optimizer_kernel(population, directions)
    baseline = time.time() - start
    print(f"  Time: {baseline*1000:.2f}ms for 100 iterations")


def test_dynamic_shapes():
    """Test compilation with dynamic shapes"""
    print("\n\nTesting Dynamic Shapes")
    print("=" * 60)
    
    def process_batch(x):
        """Function that works with dynamic batch sizes"""
        return torch.nn.functional.softmax(x, dim=-1)
    
    # Test with different shapes
    shapes = [(10, 5), (20, 5), (15, 5)]
    
    print("\nTest 1: Without dynamic shapes")
    try:
        compiled_static = torch.compile(process_batch, mode='reduce-overhead')
        
        for shape in shapes:
            x = torch.randn(shape)
            result = compiled_static(x)
            print(f"  ✓ Shape {shape}: Success")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\nTest 2: With dynamic shapes")
    try:
        compiled_dynamic = torch.compile(process_batch, 
                                       mode='reduce-overhead',
                                       dynamic=True)
        
        for shape in shapes:
            x = torch.randn(shape)
            result = compiled_dynamic(x)
            print(f"  ✓ Shape {shape}: Success")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def run_all_tests():
    """Run all compilation tests"""
    print("torch.compile Compatibility Tests")
    print("=" * 60)
    
    tests = [
        ("Graph Breaks", test_graph_breaks),
        ("Population Evaluation", test_population_evaluation_compilation),
        ("Optimization Step", test_optimization_step_compilation),
        ("Compile Modes", test_compile_modes),
        ("Dynamic Shapes", test_dynamic_shapes),
    ]
    
    for test_name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n✗ {test_name} FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("Compilation testing complete!")


if __name__ == "__main__":
    # Set up for better torch.compile messages
    import os
    os.environ['TORCH_LOGS'] = '+dynamo'
    
    run_all_tests()