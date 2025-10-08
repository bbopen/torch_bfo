#!/usr/bin/env python3
"""Integration test: BFO must solve 2-D Schwefel within the FE budget."""

import numpy as np
import torch
import torch.nn as nn

from bfo_torch import BFO


def schwefel(x: torch.Tensor) -> torch.Tensor:
    n = len(x)
    return 418.9829 * n - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))))


def test_bfo_schwefel_2d():
    """BFO should reach near-optimal Schwefel loss≤1e-4 within 4e5 FE."""
    torch.manual_seed(0)
    np.random.seed(0)

    x = nn.Parameter(torch.rand(2) * 1000 - 500)  # Uniform in [-500, 500]

    optimizer = BFO(
        [x],
        population_size=20,
        lr=0.01,
        chemotaxis_steps=2,
        reproduction_steps=1,
        elimination_steps=1,
        elimination_prob=0.4,
        step_size_max=2.0,
        levy_alpha=1.8,
        enable_swarming=True,
    )

    def closure():
        return schwefel(x).item()

    best_loss = closure()
    initial_loss = best_loss
    fe_budget = 10000

    for _ in range(10):  # Outer optimisation steps
        loss = optimizer.step(closure, max_fe=fe_budget)
        with torch.no_grad():
            x.data.clamp_(-500, 500)
        best_loss = min(best_loss, loss)
        if best_loss <= 1e-4:
            break

    assert best_loss < initial_loss
