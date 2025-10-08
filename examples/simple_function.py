"""
Simple function optimization example.

Demonstrates basic BFO usage on the Rosenbrock function.
"""

import torch
import torch.nn as nn
from bfo_torch import BFO


def rosenbrock(x):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    
    Global minimum at (1, 1) with f(1,1) = 0
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def main():
    print("Optimizing Rosenbrock Function with BFO")
    print("=" * 50)
    
    # Initialize parameters
    x = nn.Parameter(torch.tensor([0.0, 0.0]))
    print(f"Initial point: x = {x.data.tolist()}")
    print(f"Initial value: f(x) = {rosenbrock(x).item():.6f}\n")
    
    # Create optimizer
    optimizer = BFO(
        [x],
        lr=0.1,
        population_size=50,
        chemotaxis_steps=5,
        swim_length=4,
        reproduction_steps=3,
        elimination_steps=2,
        step_size_max=0.3,
        domain_bounds=(-2.0, 2.0),
        seed=42,
    )
    
    # Define closure
    def closure():
        return rosenbrock(x).item()
    
    # Optimize
    print("Optimizing...")
    for step in range(20):
        loss = optimizer.step(closure)
        if step % 5 == 0 or step == 19:
            print(f"Step {step:2d}: loss = {loss:.6f}, x = [{x[0].item():.4f}, {x[1].item():.4f}]")
    
    print("\n" + "=" * 50)
    print(f"Final point: x = [{x[0].item():.4f}, {x[1].item():.4f}]")
    print(f"Final value: f(x) = {rosenbrock(x).item():.6f}")
    print(f"Optimal point: x = [1.0000, 1.0000]")
    print(f"Optimal value: f(x) = 0.000000")
    print(f"\nDistance from optimum: {torch.norm(x - torch.tensor([1.0, 1.0])).item():.6f}")


if __name__ == "__main__":
    main()

