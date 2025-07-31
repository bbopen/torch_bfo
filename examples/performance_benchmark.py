#!/usr/bin/env python
"""
Performance benchmark for BFO optimizers.

This script compares the performance of the vectorized BFO implementation
against a simple test function to measure optimization speed.
"""

import time
import torch
import torch.nn as nn
import argparse
from bfo_torch import BFO, AdaptiveBFO, HybridBFO


def rosenbrock(x):
    """Rosenbrock function for benchmarking."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def sphere(x):
    """Sphere function for benchmarking."""
    return (x**2).sum()


def benchmark_optimizer(optimizer_class, optimizer_name, test_function, initial_params, **kwargs):
    """Benchmark an optimizer on a test function."""
    # Reset parameters
    params = nn.Parameter(initial_params.clone())
    
    # Create optimizer
    optimizer = optimizer_class([params], **kwargs)
    
    # Time the optimization
    start_time = time.time()
    losses = []
    
    def closure():
        optimizer.zero_grad()
        loss = test_function(params)
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward()
        return loss.item() if isinstance(loss, torch.Tensor) else loss
    
    # Run optimization
    for i in range(20):
        loss = optimizer.step(closure)
        losses.append(loss)
    
    end_time = time.time()
    
    return {
        'optimizer': optimizer_name,
        'time': end_time - start_time,
        'final_loss': losses[-1],
        'losses': losses,
        'final_params': params.detach().clone()
    }


def main():
    """Run performance benchmarks."""
    parser = argparse.ArgumentParser(description="BFO Performance Benchmark")
    parser.add_argument(
        '--no-plot', 
        action='store_true',
        help="Disable plotting and saving the benchmark figure."
    )
    args = parser.parse_args()
    
    print("BFO Performance Benchmark")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        {
            'function': rosenbrock,
            'function_name': 'Rosenbrock',
            'initial': torch.tensor([-1.2, 1.0]),
            'optimal': torch.tensor([1.0, 1.0])
        },
        {
            'function': sphere,
            'function_name': 'Sphere (5D)',
            'initial': torch.randn(5) * 2,
            'optimal': torch.zeros(5)
        }
    ]
    
    # Optimizer configurations
    optimizer_configs = [
        {
            'class': BFO,
            'name': 'BFO',
            'kwargs': {
                'lr': 0.01,
                'population_size': 20,
                'chemotaxis_steps': 5,
                'reproduction_steps': 2,
                'elimination_steps': 1
            }
        },
        {
            'class': AdaptiveBFO,
            'name': 'AdaptiveBFO',
            'kwargs': {
                'lr': 0.01,
                'population_size': 15,
                'adaptation_rate': 0.2,
                'min_population_size': 10,
                'max_population_size': 30
            }
        },
        {
            'class': HybridBFO,
            'name': 'HybridBFO',
            'kwargs': {
                'lr': 0.01,
                'population_size': 15,
                'gradient_weight': 0.5,
                'enable_momentum': True
            }
        }
    ]
    
    # Run benchmarks
    results = []
    for test_config in test_configs:
        print(f"\nTesting on {test_config['function_name']} function:")
        print("-" * 40)
        
        for opt_config in optimizer_configs:
            result = benchmark_optimizer(
                opt_config['class'],
                opt_config['name'],
                test_config['function'],
                test_config['initial'],
                **opt_config['kwargs']
            )
            
            # Calculate distance to optimum
            distance = torch.norm(result['final_params'] - test_config['optimal']).item()
            
            print(f"{opt_config['name']:15s}: Time={result['time']:6.3f}s, "
                  f"Final Loss={result['final_loss']:10.6f}, "
                  f"Distance to Optimum={distance:8.6f}")
            
            result['test_function'] = test_config['function_name']
            results.append(result)
    
    # Plot results
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            for i, test_name in enumerate(['Rosenbrock', 'Sphere (5D)']):
                ax = axes[i]
                test_results = [r for r in results if r['test_function'] == test_name]
                
                for result in test_results:
                    ax.semilogy(result['losses'], label=result['optimizer'])
                
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Loss (log scale)')
                ax.set_title(f'{test_name} Function')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('bfo_performance_benchmark.png', dpi=150)
            print(f"\nBenchmark plot saved to 'bfo_performance_benchmark.png'")
        
        except ImportError:
            print("\nPlotting disabled: matplotlib not found. Please install it to generate plots.")

    # Performance summary
    print("\n" + "=" * 50)
    print("Performance Summary:")
    print("-" * 50)
    
    # Average time per optimizer
    for opt_name in ['BFO', 'AdaptiveBFO', 'HybridBFO']:
        opt_results = [r for r in results if r['optimizer'] == opt_name]
        avg_time = sum(r['time'] for r in opt_results) / len(opt_results)
        avg_loss = sum(r['final_loss'] for r in opt_results) / len(opt_results)
        print(f"{opt_name:15s}: Avg Time={avg_time:6.3f}s, Avg Final Loss={avg_loss:10.6f}")


if __name__ == "__main__":
    main()