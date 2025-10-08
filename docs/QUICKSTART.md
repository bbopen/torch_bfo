# BFO-Torch Quick Start Guide

Get started with Bacterial Foraging Optimization in PyTorch in 5 minutes.

## Installation

```bash
pip install bfo-torch
```

## Basic Usage

### 1. Optimize a Simple Function

```python
import torch
import torch.nn as nn
from bfo_torch import BFO

# Define a parameter to optimize
x = nn.Parameter(torch.tensor([5.0, 5.0]))

# Create optimizer
optimizer = BFO([x], lr=0.1, population_size=30)

# Define objective function (minimize x^2 + y^2)
def closure():
    loss = (x ** 2).sum()
    return loss.item()

# Run optimization
for step in range(10):
    best_loss = optimizer.step(closure)
    print(f"Step {step}: loss = {best_loss:.6f}, x = {x.data}")
```

Output:
```
Step 0: loss = 50.000000, x = tensor([5.0000, 5.0000])
Step 1: loss = 12.234567, x = tensor([2.3456, 2.9876])
...
Step 9: loss = 0.001234, x = tensor([0.0234, 0.0123])
```

### 2. Train a Neural Network

```python
import torch
import torch.nn as nn
from bfo_torch import BFO

# Create a simple model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Create optimizer
optimizer = BFO(model.parameters(), lr=0.01, population_size=50)

# Training data
X_train = torch.randn(100, 10)
y_train = torch.randn(100, 1)

# Training loop
def closure():
    output = model(X_train)
    loss = nn.functional.mse_loss(output, y_train)
    return loss.item()

for epoch in range(20):
    loss = optimizer.step(closure)
    print(f"Epoch {epoch}: loss = {loss:.4f}")
```

### 3. With Gradients (HybridBFO)

For differentiable objectives, use `HybridBFO` to leverage both gradient information and BFO's exploration:

```python
from bfo_torch import HybridBFO

model = nn.Linear(10, 1)
optimizer = HybridBFO(
    model.parameters(),
    lr=0.01,
    population_size=30,
    gradient_weight=0.5  # Balance BFO and gradient descent
)

def closure():
    optimizer.zero_grad()
    output = model(X_train)
    loss = nn.functional.mse_loss(output, y_train)
    loss.backward()  # Compute gradients
    return loss.item()

for epoch in range(20):
    loss = optimizer.step(closure)
    print(f"Epoch {epoch}: loss = {loss:.4f}")
```

### 4. Adaptive Population (AdaptiveBFO)

For problems where you're unsure about population size, use `AdaptiveBFO`:

```python
from bfo_torch import AdaptiveBFO

x = nn.Parameter(torch.randn(20))
optimizer = AdaptiveBFO(
    [x],
    lr=0.01,
    population_size=30,
    min_population_size=10,
    max_population_size=100,
    adaptation_rate=0.2
)

def closure():
    return (x ** 2).sum().item()

for step in range(20):
    loss = optimizer.step(closure)
    pop_size = optimizer.param_groups[0]['population_size']
    print(f"Step {step}: loss = {loss:.4f}, population = {pop_size}")
```

## Common Patterns

### Hyperparameter Search

Use BFO to find optimal hyperparameters for another optimizer:

```python
from bfo_torch import BFO
import torch
import torch.nn as nn

# Hyperparameters to search
lr_param = nn.Parameter(torch.tensor([0.001]))  # Learning rate
wd_param = nn.Parameter(torch.tensor([0.0001]))  # Weight decay

# BFO to optimize hyperparameters
meta_optimizer = BFO(
    [lr_param, wd_param],
    lr=0.1,
    population_size=20,
    domain_bounds=(1e-5, 1e-1)  # Constrain search space
)

def evaluate_hyperparameters():
    # Train model with current hyperparameters
    model = create_model()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr_param.item(),
        weight_decay=wd_param.item()
    )
    
    # Train for a few epochs
    val_loss = train_model(model, optimizer, epochs=5)
    return val_loss

# Search for best hyperparameters
for trial in range(10):
    val_loss = meta_optimizer.step(evaluate_hyperparameters)
    print(f"Trial {trial}: lr={lr_param.item():.5f}, wd={wd_param.item():.5f}, loss={val_loss:.4f}")
```

### Progress Monitoring with Callback

Track optimization progress with a callback:

```python
from bfo_torch import BFO

x = nn.Parameter(torch.randn(10))
optimizer = BFO([x], lr=0.1, population_size=30)

def closure():
    return (x ** 2).sum().item()

def progress_callback(info):
    print(f"Iteration {info['iteration']}: "
          f"best_fitness={info['best_fitness']:.6e}, "
          f"diversity={info['population_diversity']:.4f}, "
          f"stagnation={info['stagnation_count']}")

# Run with callback
for step in range(10):
    loss = optimizer.step(closure, callback=progress_callback)
```

### Checkpointing

Save and restore optimizer state:

```python
from bfo_torch import BFO
import torch

# Create and train optimizer
model = nn.Linear(10, 1)
optimizer = BFO(model.parameters(), population_size=30, seed=42)

def closure():
    return model(torch.randn(10, 10)).pow(2).sum().item()

# Train for a while
for _ in range(5):
    optimizer.step(closure)

# Save checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(checkpoint, 'checkpoint.pt')

# Later: restore and continue training
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Continue training
for _ in range(5):
    optimizer.step(closure)
```

## Key Parameters

| Parameter | Default | Description | When to Adjust |
|-----------|---------|-------------|----------------|
| `lr` | 0.01 | Learning rate | Increase for faster convergence, decrease for stability |
| `population_size` | 50 | Number of bacteria | Increase for complex problems (20-100 typical) |
| `chemotaxis_steps` | 10 | Local search steps | Increase for fine-grained exploration |
| `swim_length` | 4 | Consecutive moves in good direction | Increase to exploit good gradients |
| `step_size_max` | 0.1 | Maximum step size | Adjust based on problem scale |
| `levy_alpha` | 1.5 | Lévy flight parameter | 1.5 balanced, 2.0 more local |
| `elimination_prob` | 0.25 | Probability of random restart | Increase if stuck in local minima |

## Troubleshooting

### Problem: Slow convergence

**Solutions:**
- Increase `population_size` (30-100)
- Increase `chemotaxis_steps` (10-20)
- Try `HybridBFO` if gradients available
- Adjust `step_size_max` to match problem scale

### Problem: Gets stuck in local minima

**Solutions:**
- Increase `elimination_prob` (0.3-0.5)
- Increase `elimination_steps` (3-5)
- Use `levy_alpha=1.5` (default) for better exploration
- Try `AdaptiveBFO` for automatic population adjustment

### Problem: Too many function evaluations

**Solutions:**
- Decrease `population_size` (10-30)
- Decrease `chemotaxis_steps` (3-5)
- Set `max_fe` budget in `step()`: `optimizer.step(closure, max_fe=10000)`
- Enable `early_stopping=True` (default)

### Problem: Unstable or diverging

**Solutions:**
- Add `domain_bounds` to constrain search space
- Decrease `step_size_max` (0.01-0.05)
- Decrease `lr` (0.001-0.01)
- Check closure returns finite values

## Next Steps

- Read [HYPERPARAMETERS.md](HYPERPARAMETERS.md) for detailed tuning guide
- Read [ALGORITHM.md](ALGORITHM.md) to understand how BFO works
- Check [examples/](../examples/) for more use cases
- Review API documentation for advanced features

## Getting Help

- GitHub Issues: https://github.com/bbopen/torch_bfo/issues
- Documentation: https://github.com/bbopen/torch_bfo/wiki

