# bfo-torch Documentation

Welcome to the official documentation for `bfo-torch`, a PyTorch-native Bacterial Foraging Optimizer.

`bfo-torch` is designed for performance, leveraging `torch.compile` and vectorized operations for GPU acceleration. It offers a simple, intuitive API that integrates seamlessly into any PyTorch workflow.

## Key Features

- **High-Performance**: Optimized with `torch.compile` and vectorized operations.
- **Adaptive Variants**: Includes `AdaptiveBFO` for automatic parameter tuning and `HybridBFO` for leveraging gradient information.
- **Easy Integration**: Drop-in replacement for any `torch.optim.Optimizer`.

## Getting Started

To get started, first install the library:

```bash
pip install bfo-torch
```

Then, you can use it in your project like any other PyTorch optimizer:

```python
from bfo_torch import BFO

optimizer = BFO(model.parameters())
```