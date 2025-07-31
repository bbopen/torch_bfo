#!/usr/bin/env python3
"""Quick P1 test with fewer runs to debug issues."""

import torch
import numpy as np
from src.bfo_torch.chaotic_bfo import ChaoticBFO
import torch.nn as nn


def schwefel_function(x):
    """Schwefel function implementation."""
    n = len(x)
    return 418.9829 * n - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))))


def test_chaotic_bfo():
    """Test ChaoticBFO with a single run."""
    print("Testing ChaoticBFO on 2D Schwefel...")
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize
    x = nn.Parameter(torch.rand(2) * 1000 - 500)  # [-500, 500]
    
    # Create optimizer
    optimizer = ChaoticBFO(
        [x],
        population_size=50,
        lr=0.01,
        chemotaxis_steps=10,
        reproduction_steps=5,
        elimination_steps=2,
        elimination_prob=0.4,
        step_size_max=1.0,
        levy_alpha=1.8,
        enable_swarming=True,
        enable_chaos=True,
        chaos_strength=0.5,
        diversity_trigger_ratio=0.5,
        enable_crossover=True
    )
    
    def closure():
        loss = schwefel_function(x)
        return loss.item()
    
    # Initial loss
    initial_loss = closure()
    print(f"Initial loss: {initial_loss:.2f}")
    print(f"Initial position: {x.data.numpy()}")
    
    # Run optimization with budget
    max_fe = 10000
    best_loss = initial_loss
    
    for step in range(100):  # Max 100 steps
        current_fe = optimizer.get_function_evaluations()
        if current_fe >= max_fe:
            print(f"Reached FE budget at step {step}")
            break
            
        loss = optimizer.step(closure, max_fe=max_fe)
        
        # Bounds
        with torch.no_grad():
            x.data = torch.clamp(x.data, -500, 500)
        
        if loss < best_loss:
            best_loss = loss
            
        if (step + 1) % 10 == 0:
            fe_used = optimizer.get_function_evaluations()
            print(f"Step {step+1}: Loss = {loss:.2f}, FE = {fe_used}")
    
    # Final results
    final_fe = optimizer.get_function_evaluations()
    print(f"\nFinal loss: {best_loss:.2f}")
    print(f"Final position: {x.data.numpy()}")
    print(f"Function evaluations: {final_fe}")
    print(f"Target position: [420.9687, 420.9687]")
    print(f"Distance to optimum: {torch.norm(x.data - torch.tensor([420.9687, 420.9687])).item():.2f}")


if __name__ == "__main__":
    test_chaotic_bfo()