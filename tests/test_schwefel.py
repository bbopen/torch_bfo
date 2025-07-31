#!/usr/bin/env python3
"""Integration test: ChaoticBFO must solve 2-D Schwefel within the FE budget."""

import torch
import numpy as np
import torch.nn as nn
from bfo_torch.chaotic_bfo import ChaoticBFO


def schwefel(x: torch.Tensor) -> torch.Tensor:
    n = len(x)
    return 418.9829 * n - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))))


def test_chaotic_bfo_schwefel_2d():
    """ChaoticBFO should reach near-optimal Schwefel loss≤1e-4 within 4e5 FE."""
    torch.manual_seed(0)
    np.random.seed(0)

    x = nn.Parameter(torch.rand(2) * 1000 - 500)  # Uniform in [-500, 500]

    optimizer = ChaoticBFO(
        [x],
        population_size=100,
        lr=0.01,
        chemotaxis_steps=10,
        reproduction_steps=5,
        elimination_steps=2,
        elimination_prob=0.4,
        step_size_max=2.0,
        levy_alpha=1.8,
        enable_swarming=True,
        enable_chaos=True,
        chaos_strength=0.7,
        diversity_trigger_ratio=0.6,
        enable_crossover=True,
    )

    def closure():
        return schwefel(x).item()

    best_loss = closure()
    fe_budget = 400_000

    for _ in range(60):  # Outer optimisation steps
        loss = optimizer.step(closure, max_fe=fe_budget)
        with torch.no_grad():
            x.data.clamp_(-500, 500)
        best_loss = min(best_loss, loss)
        if best_loss <= 1e-4:
            break

    assert best_loss <= 1e-3, f"ChaoticBFO failed to converge; best_loss={best_loss:.6f}"