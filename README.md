# bfo-torch

[![PyPI version](https://badge.fury.io/py/bfo-torch.svg)](https://badge.fury.io/py/bfo-torch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyTorch implementation of Bacterial Foraging Optimization (BFO) algorithm with enhancements including Lévy flights, adaptive mechanisms, and gradient integration.

## Features

**Device Support**
- CPU, CUDA, and MPS (Apple Silicon)
- Mixed precision: FP16, BF16, FP32, FP64

**Functionality**
- State checkpointing
- Progress callbacks
- Function evaluation budgets
- Reproducible results with seed control

**Algorithm Enhancements**
- Lévy flight-based exploration (Mantegna 1994)
- Adaptive step sizing
- Diversity-based elimination probability
- Optional gradient integration (HybridBFO variant)

## Installation

```bash
pip install bfo-torch
```

## Quick Start

### Basic Function Optimization

```python
import torch
import torch.nn as nn
from bfo_torch import BFO

# Define parameters to optimize
x = nn.Parameter(torch.tensor([5.0, 5.0]))

# Create optimizer
optimizer = BFO([x], lr=0.1, population_size=30)

# Define objective function
def closure():
    loss = (x ** 2).sum()  # Minimize x^2 + y^2
    return loss.item()

# Optimize
for step in range(10):
    loss = optimizer.step(closure)
    print(f"Step {step}: loss = {loss:.6f}")
```

### Neural Network Training

```python
import torch.nn as nn
from bfo_torch import BFO

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

optimizer = BFO(model.parameters(), lr=0.01, population_size=50)

def closure():
    output = model(X_train)
    loss = nn.functional.mse_loss(output, y_train)
    return loss.item()

for epoch in range(20):
    loss = optimizer.step(closure)
    print(f"Epoch {epoch}: loss = {loss:.4f}")
```

### With Gradients (HybridBFO)

```python
from bfo_torch import HybridBFO

optimizer = HybridBFO(
    model.parameters(),
    lr=0.01,
    population_size=30,
    gradient_weight=0.5  # Balance BFO and gradients
)

def closure():
    optimizer.zero_grad()
    loss = model(X_train).pow(2).sum()
    loss.backward()  # Compute gradients
    return loss.item()

optimizer.step(closure)
```

## Documentation

**Guides:**
- [Quick Start Guide](docs/QUICKSTART.md)
- [Algorithm Overview](docs/ALGORITHM.md)
- [Hyperparameter Tuning](docs/HYPERPARAMETERS.md)

**Examples:**
- [Simple Function Optimization](examples/simple_function.py)
- [Neural Network Training](examples/neural_network.py)
- [Hyperparameter Search](examples/hyperparameter_tuning.py)

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 50 | Number of bacteria (20-100 typical) |
| `chemotaxis_steps` | 10 | Local search steps |
| `swim_length` | 4 | Consecutive moves in good direction |
| `levy_alpha` | 1.5 | Lévy flight parameter (1.0-2.0) |
| `elimination_prob` | 0.25 | Probability of random restart |

See [HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md) for detailed guidance.

## Optimizer Variants

### BFO
Standard implementation without gradient information.

```python
from bfo_torch import BFO
optimizer = BFO(params, population_size=50)
```

### AdaptiveBFO
Adjusts population size during optimization based on convergence behavior.

```python
from bfo_torch import AdaptiveBFO
optimizer = AdaptiveBFO(params, min_population_size=20, max_population_size=100)
```

### HybridBFO
Incorporates gradient information alongside BFO mechanisms.

```python
from bfo_torch import HybridBFO
optimizer = HybridBFO(params, gradient_weight=0.5)
```

## Applicability

**Suitable for:**
- Black-box optimization problems
- Non-differentiable or discontinuous objectives
- Noisy objective functions
- Hyperparameter optimization
- Small-scale neural architecture search

**Limitations:**
- Requires many function evaluations per optimization step
- Computational cost increases with population size
- May be inefficient for high-dimensional problems (>1000 parameters) without careful tuning

## Algorithm Overview

The implementation follows the canonical BFO algorithm (Passino 2002) with four main mechanisms:

1. **Chemotaxis**: Bacterial movement through tumble (random direction) and swim (continuation in improving direction)
2. **Swarming**: Attraction-repulsion forces between bacteria based on cell-to-cell signaling
3. **Reproduction**: Elimination of least-fit bacteria and replication of most-fit bacteria
4. **Elimination-Dispersal**: Random elimination and repositioning of bacteria with specified probability

**Implementation modifications:**
- Lévy flight distribution for tumble directions instead of uniform random
- Direction vector normalization for dimension-independent step sizes
- Adaptive step size schedules (cosine, linear, performance-based)
- Adaptive elimination probability based on population diversity

See [ALGORITHM.md](docs/ALGORITHM.md) for mathematical formulations and implementation details.

## Benchmark Results

Standard test functions with default parameters:
- Sphere (10D, unimodal): Mean final loss <1.0 after 10 steps
- Rosenbrock (2D, narrow valley): Mean final loss <10.0 after 15 steps
- Rastrigin (5D, multimodal): Convergence to local minimum
- Ackley (5D, many local minima): Mean final loss <5.0 after 15 steps

## Citation

If you use this work in your research, please consider citing it:

```bibtex
@software{bfo-torch,
  author = {Brett G. Bonner},
  title = {bfo-torch: Bacterial Foraging Optimization for PyTorch},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bbopen/torch_bfo}},
}
```

## References

- **Passino, K. M. (2002)**: "Biomimicry of Bacterial Foraging for Distributed Optimization and Control" - IEEE Control Systems Magazine (canonical BFO algorithm)
- **Mantegna, R. N. (1994)**: "Fast, accurate algorithm for numerical simulation of Levy stable stochastic processes" (Lévy flight generation)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions may be submitted via pull request or issue on the [GitHub repository](https://github.com/bbopen/torch_bfo).

## Links

- Documentation: [docs/](docs/)
- Examples: [examples/](examples/)
- Issues: https://github.com/bbopen/torch_bfo/issues
- PyPI: https://pypi.org/project/bfo-torch/