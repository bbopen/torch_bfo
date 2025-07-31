#!/usr/bin/env python3
"""
Rigorous benchmark script to check BFO convergence to global optima within accepted error margins.
"""
import math
import torch
import torch.nn as nn
import time
from bfo_torch import BFO

def run_optimizer(fn, bounds, dim, steps=50, pop=20):
    """Run optimizer with more iterations for better convergence."""
    torch.manual_seed(0)
    x = nn.Parameter(torch.empty(dim).uniform_(bounds[0], bounds[1]))
    optimizer = BFO(
        [x],
        lr=0.05,
        population_size=pop,
        chemotaxis_steps=3,
        swim_length=3,
        reproduction_steps=2,
        elimination_steps=1,
        step_size_max=0.1 * (bounds[1] - bounds[0]),
        domain_bounds=bounds,
        seed=0,
        early_stopping=False,
    )

    def closure():
        return fn(x).item()

    initial_loss = closure()
    start_time = time.time()
    
    losses = [initial_loss]
    for _ in range(steps):
        loss = optimizer.step(closure)
        losses.append(loss)
    
    end_time = time.time()
    final_loss = losses[-1]
    
    return {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'time': end_time - start_time,
        'final_params': x.detach().clone(),
        'losses': losses
    }

# Benchmark functions with their known global optima
def rastrigin(x):
    n = x.numel()
    return 10 * n + torch.sum(x**2 - 10 * torch.cos(2 * math.pi * x))

def ackley(x):
    n = x.numel()
    return (
        -20 * torch.exp(-0.2 * torch.sqrt((x**2).mean()))
        - torch.exp(torch.cos(2 * math.pi * x).mean())
        + 20
        + math.e
    )

def griewank(x):
    part1 = torch.sum(x**2) / 4000
    i = torch.arange(1, x.numel() + 1, device=x.device, dtype=x.dtype)
    part2 = torch.prod(torch.cos(x / torch.sqrt(i)))
    return part1 - part2 + 1

def schwefel(x):
    n = x.numel()
    return 418.9829 * n - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))))

def rosenbrock(x):
    return torch.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)

def sphere(x):
    return torch.sum(x**2)

def michalewicz(x, m=10):
    i = torch.arange(1, x.numel() + 1, device=x.device, dtype=x.dtype)
    return -torch.sum(torch.sin(x) * torch.sin(i * x**2 / math.pi) ** (2 * m))

def levy(x):
    w = 1 + (x - 1) / 4
    term1 = torch.sin(math.pi * w[0]) ** 2
    term2 = ((w[:-1] - 1) ** 2 * (1 + 10 * torch.sin(math.pi * w[:-1] + 1) ** 2)).sum()
    term3 = (w[-1] - 1) ** 2 * (1 + torch.sin(2 * math.pi * w[-1]) ** 2)
    return term1 + term2 + term3

def step(x):
    return torch.sum(torch.floor(x + 0.5) ** 2)

def drop_wave(x):
    r2 = torch.sum(x**2)
    return -(1 + torch.cos(12 * torch.sqrt(r2))) / (0.5 * r2 + 2)

def easom(x):
    return (
        -torch.cos(x[0])
        * torch.cos(x[1])
        * torch.exp(-((x[0] - math.pi) ** 2 + (x[1] - math.pi) ** 2))
    )

# Benchmark configurations with known global optima and accepted error tolerances
benchmarks = [
    # (name, function, bounds, dim, global_minimum, optimal_point, tolerance)
    ("Rastrigin", rastrigin, (-5.12, 5.12), 5, 0.0, [0.0]*5, 1.0),
    ("Ackley", ackley, (-32.768, 32.768), 5, 0.0, [0.0]*5, 1e-2),
    ("Griewank", griewank, (-600, 600), 5, 0.0, [0.0]*5, 1e-2),
    ("Schwefel", schwefel, (-500, 500), 5, 0.0, [420.9687]*5, 50.0),  # Very difficult
    ("Rosenbrock", rosenbrock, (-30, 30), 5, 0.0, [1.0]*5, 1e-1),
    ("Sphere", sphere, (-5.12, 5.12), 5, 0.0, [0.0]*5, 1e-3),
    ("Michalewicz", michalewicz, (0, math.pi), 5, -4.687, None, 0.5),  # Complex optimum
    ("Levy", levy, (-10, 10), 5, 0.0, [1.0]*5, 1e-2),
    ("Step", step, (-100, 100), 5, 0.0, [0.0]*5, 1e-6),
    ("Drop Wave", drop_wave, (-5.12, 5.12), 2, -1.0, [0.0, 0.0], 1e-3),
    ("Easom", easom, (-100, 100), 2, -1.0, [math.pi, math.pi], 1e-3),
]

def main():
    print("BFO Convergence Analysis - Global Optima Achievement")
    print("=" * 90)
    print(f"{'Function':<12} {'Dim':<4} {'Target':<10} {'Achieved':<12} {'Error':<12} {'Tolerance':<10} {'Status':<10} {'Time(s)':<8}")
    print("-" * 90)
    
    total_time = 0
    converged_count = 0
    results = []
    
    for name, fn, bounds, dim, global_min, optimal_point, tolerance in benchmarks:
        result = run_optimizer(fn, bounds, dim, steps=100 if name == "Schwefel" else 50)
        total_time += result['time']
        
        final_loss = result['final_loss']
        error = abs(final_loss - global_min)
        converged = error <= tolerance
        
        if converged:
            converged_count += 1
        
        status = "✓ PASS" if converged else "✗ FAIL" 
        
        results.append({
            'name': name,
            'converged': converged,
            'error': error,
            'tolerance': tolerance,
            'final_loss': final_loss,
            'target': global_min,
            'time': result['time']
        })
        
        print(f"{name:<12} {dim:<4} {global_min:<10.3f} {final_loss:<12.6f} {error:<12.6f} {tolerance:<10.3f} {status:<10} {result['time']:<8.3f}")
    
    print("-" * 90)
    print(f"Convergence Summary:")
    print(f"Functions converged to global optimum: {converged_count}/{len(benchmarks)} ({converged_count/len(benchmarks)*100:.1f}%)")
    print(f"Total execution time: {total_time:.3f}s")
    print(f"Average time per function: {total_time/len(benchmarks):.3f}s")
    
    # Detailed analysis
    print(f"\nDetailed Analysis:")
    print("-" * 50)
    
    converged_functions = [r for r in results if r['converged']]
    failed_functions = [r for r in results if not r['converged']]
    
    if converged_functions:
        print(f"Successfully converged functions:")
        for r in converged_functions:
            print(f"  • {r['name']}: Error = {r['error']:.6f} (tolerance: {r['tolerance']:.6f})")
    
    if failed_functions:
        print(f"\nFunctions that did not converge:")
        for r in failed_functions:
            error_ratio = r['error'] / r['tolerance']
            print(f"  • {r['name']}: Error = {r['error']:.6f}, {error_ratio:.1f}x over tolerance")
    
    # Performance categories
    excellent = [r for r in results if r['converged'] and r['error'] < r['tolerance'] * 0.1]
    good = [r for r in results if r['converged'] and r['tolerance'] * 0.1 <= r['error'] <= r['tolerance'] * 0.5]
    acceptable = [r for r in results if r['converged'] and r['error'] > r['tolerance'] * 0.5]
    
    print(f"\nPerformance Categories:")
    print(f"Excellent (error < 10% of tolerance): {len(excellent)}")
    print(f"Good (error 10-50% of tolerance): {len(good)}")
    print(f"Acceptable (error 50-100% of tolerance): {len(acceptable)}")
    print(f"Failed (error > tolerance): {len(failed_functions)}")

if __name__ == "__main__":
    main()
