"""
PyTorch Bacterial Foraging Optimizer (BFO)
==========================================

A PyTorch implementation of Bacterial Foraging Optimization algorithm
for deep learning and scientific computing applications.

Classes:
    BFO: Standard Bacterial Foraging Optimizer
    AdaptiveBFO: BFO with adaptive parameter adjustment
    HybridBFO: BFO with gradient information integration

Example:
    >>> import torch
    >>> import torch.nn as nn
    >>> from bfo_torch import BFO
    >>>
    >>> model = nn.Linear(10, 1)
    >>> optimizer = BFO(model.parameters(), lr=0.01)
    >>>
    >>> def closure():
    >>>     optimizer.zero_grad()
    >>>     output = model(data)
    >>>     loss = criterion(output, target)
    >>>     return loss
    >>>
    >>> optimizer.step(closure)
"""

from .optimizer import BFO, AdaptiveBFO, HybridBFO

__version__ = "0.1.0"
__author__ = "Brett G. Bonner"
__url__ = "https://github.com/bbopen/torch_bfo"

__all__ = ["BFO", "AdaptiveBFO", "HybridBFO"]
