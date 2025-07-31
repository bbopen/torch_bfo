# bfo-torch

[![PyPI version](https://badge.fury.io/py/bfo-torch.svg)](https://badge.fury.io/py/bfo-torch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of the Bacterial Foraging Optimization (BFO) algorithm.

## Installation

You can install `bfo-torch` from PyPI:

```bash
pip install bfo-torch
```

## Basic Usage

Here's a simple example of how to use the `BFO` optimizer with a PyTorch model:

```python
import torch
import torch.nn as nn
from bfo_torch import BFO

# Define model and data
model = nn.Linear(10, 1).cuda()
optimizer = BFO(model.parameters(), population_size=50)

# Define closure for BFO
def closure():
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    return loss.item()

# Optimization step
optimizer.step(closure)
```

The optimizer ships with aggressive defaults tuned from convergence analysis.
`step_size_max` is enlarged to `1.0` and the Lévy exponent is heavier at
`levy_alpha=1.9`. Additional reproduction and swim cycles are used and an
optional local search phase can further refine the best solution after each
iteration.

## Citation

If you use this work in your research, please consider citing it:

```bibtex
@software{bfo-torch,
  author = {Brett G. Bonner},
  title = {bfo-torch: Bacterial Foraging Optimization for PyTorch},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/bbopen/torch_bfo}},
}
```