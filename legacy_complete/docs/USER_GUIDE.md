# PyTorch BFO Optimizer User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Choosing the Right Optimizer](#choosing-the-right-optimizer)
4. [Best Practices](#best-practices)
5. [Advanced Usage](#advanced-usage)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Examples](#examples)

## Introduction

The PyTorch BFO (Bacterial Foraging Optimization) Optimizer implements a nature-inspired optimization algorithm based on the foraging behavior of E. coli bacteria. This guide will help you effectively use the optimizer for various machine learning tasks.

### When to Use BFO

BFO is particularly effective for:
- Non-convex optimization problems
- Problems with many local minima
- Black-box optimization where gradients are expensive
- Hyperparameter optimization
- Neural architecture search
- Reinforcement learning policy optimization

### When NOT to Use BFO

Consider traditional optimizers (Adam, SGD) for:
- Large-scale deep learning with millions of parameters
- Problems requiring fast convergence
- Real-time training scenarios
- When computational budget is limited

## Getting Started

### Installation

```bash
pip install pytorch-bfo-optimizer
```

Or install from source:
```bash
git clone https://github.com/yourusername/pytorch-bfo-optimizer.git
cd pytorch-bfo-optimizer
pip install -e .
```

### Basic Example

```python
import torch
import torch.nn as nn
from pytorch_bfo_optimizer import BFO

# Define model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# Create optimizer
optimizer = BFO(model.parameters(), population_size=30)

# Define closure function
def closure():
    optimizer.zero_grad()
    output = model(input_data)
    loss = nn.MSELoss()(output, target_data)
    return loss.item()

# Optimize
for step in range(100):
    loss = optimizer.step(closure)
    print(f'Step {step}: Loss = {loss:.4f}')
```

## Choosing the Right Optimizer

### BFO (Base Optimizer)
Best for:
- General non-convex optimization
- Problems with unknown landscape
- When exploration is important

```python
optimizer = BFO(
    model.parameters(),
    population_size=50,
    use_swarming=True  # Enable for better exploration
)
```

### AdaptiveBFO
Best for:
- Long training runs
- When optimal hyperparameters unknown
- Dynamic optimization landscapes

```python
optimizer = AdaptiveBFO(
    model.parameters(),
    population_size=50,
    adaptation_rate=0.1
)
```

### HybridBFO
Best for:
- Differentiable optimization problems
- Faster convergence needed
- GPU acceleration

```python
optimizer = HybridBFO(
    model.parameters(),
    population_size=20,
    gradient_weight=0.5  # Balance between BFO and gradient
)
```

## Best Practices

### 1. Population Size Selection

The population size is the most critical hyperparameter:

```python
# CPU Optimization
optimizer = BFO(params, population_size=50)  # Good default

# GPU Optimization
optimizer = BFO(params, population_size=10)  # Smaller for efficiency

# Complex Problems
optimizer = BFO(params, population_size=100) # More exploration
```

### 2. Closure Function Design

Always use `torch.no_grad()` for pure BFO:

```python
def closure():
    with torch.no_grad():
        output = model(data)
        loss = criterion(output, target)
    return loss.item()  # Return scalar
```

For HybridBFO, include gradient computation:

```python
def closure():
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()  # Compute gradients
    return loss.item()
```

### 3. Batch Size Optimization

Larger batches improve GPU utilization:

```python
# Good for GPU
batch_size = 512
data_loader = DataLoader(dataset, batch_size=batch_size)

# Smaller for CPU
batch_size = 64
```

### 4. Learning Rate Scheduling

Adjust BFO parameters over time:

```python
scheduler = {
    'step_size_max': lambda epoch: 0.1 * (0.9 ** epoch),
    'elim_prob': lambda epoch: 0.25 * (1.1 ** epoch)
}

for epoch in range(num_epochs):
    # Update optimizer parameters
    optimizer.defaults['step_size_max'] = scheduler['step_size_max'](epoch)
    optimizer.defaults['elim_prob'] = scheduler['elim_prob'](epoch)
```

## Advanced Usage

### Multi-Objective Optimization

```python
def multi_objective_closure():
    output = model(data)
    loss1 = criterion1(output, target1)
    loss2 = criterion2(output, target2)
    
    # Weighted combination
    total_loss = 0.7 * loss1 + 0.3 * loss2
    return total_loss.item()
```

### Constrained Optimization

```python
def constrained_closure():
    output = model(data)
    loss = criterion(output, target)
    
    # Add penalty for constraint violation
    constraint_penalty = 0
    if model.weight.norm() > max_norm:
        constraint_penalty = 100 * (model.weight.norm() - max_norm)
    
    return (loss + constraint_penalty).item()
```

### Custom Swarming Behavior

```python
optimizer = BFO(
    model.parameters(),
    use_swarming=True,
    swarming_params=(
        0.5,   # d_attract: attraction depth
        0.2,   # w_attract: attraction width
        0.5,   # h_repel: repulsion height
        5.0    # w_repel: repulsion width
    )
)
```

### Checkpointing

```python
# Save optimizer state
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'population': optimizer.population,
    'best_params': optimizer.best_params,
    'best_fitness': optimizer.best_fitness,
    'epoch': epoch
}
torch.save(checkpoint, 'checkpoint.pth')

# Load optimizer state
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
optimizer.population = checkpoint['population']
optimizer.best_params = checkpoint['best_params']
optimizer.best_fitness = checkpoint['best_fitness']
```

## Performance Optimization

### GPU Optimization

```python
# 1. Fix torch.compile issues (PyTorch 2.8+)
import torch._dynamo as dynamo
dynamo.config.capture_scalar_outputs = True

# 2. Optimal GPU configuration
optimizer = HybridBFO(
    model.parameters(),
    population_size=8,      # Small population
    compile_mode=True,      # Enable if stable
    device='cuda'
)

# 3. Large batch processing
batch_size = 1024
data = data.to('cuda')
target = target.to('cuda')

# 4. Mixed precision training
with torch.cuda.amp.autocast():
    def closure():
        output = model(data)
        loss = criterion(output, target)
        return loss.item()
```

### CPU Optimization

```python
# 1. Parallel evaluation (if possible)
torch.set_num_threads(8)

# 2. Optimal CPU configuration  
optimizer = BFO(
    model.parameters(),
    population_size=50,
    compile_mode=False  # Often slower on CPU
)

# 3. Efficient data loading
data_loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    pin_memory=True
)
```

### Memory Optimization

```python
# 1. Gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def forward(self, x):
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x

# 2. Clear cache periodically
if step % 100 == 0:
    torch.cuda.empty_cache()

# 3. Reduce population for memory-constrained scenarios
optimizer = BFO(params, population_size=10)
```

## Troubleshooting Guide

### Issue: RuntimeError with Population Split

**Error**: `RuntimeError: shape mismatch in reproduction`

**Solution**: Use even population sizes
```python
# Good
optimizer = BFO(params, population_size=10)  # Even number

# Avoid
optimizer = BFO(params, population_size=11)  # Odd number
```

### Issue: torch.compile Errors

**Error**: `CppCompileError` or graph breaks

**Solution**: 
```python
# Option 1: Disable compile
optimizer = BFO(params, compile_mode=False)

# Option 2: Fix graph breaks
import torch._dynamo as dynamo
dynamo.config.capture_scalar_outputs = True

# Option 3: Use different backend
optimizer = BFO(params, compile_kwargs={'backend': 'eager'})
```

### Issue: Slow GPU Performance

**Solution**: Optimize configuration
```python
# Use HybridBFO for better GPU utilization
optimizer = HybridBFO(
    params,
    population_size=5,      # Very small
    gradient_weight=0.7,    # More gradient
    compile_mode=False      # If issues
)

# Increase batch size
batch_size = 2048

# Profile to find bottlenecks
with torch.profiler.profile() as prof:
    optimizer.step(closure)
prof.export_chrome_trace("trace.json")
```

### Issue: Not Converging

**Solution**: Adjust parameters
```python
# Increase exploration
optimizer = AdaptiveBFO(
    params,
    population_size=100,    # Larger population
    step_size_max=0.5,      # Larger steps
    levy_alpha=1.2,         # More exploration
    use_swarming=True
)

# Or try different optimizer
optimizer = HybridBFO(params, gradient_weight=0.8)
```

## Examples

### Example 1: Image Classification

```python
import torchvision.models as models
from pytorch_bfo_optimizer import HybridBFO

# Load pretrained model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 10)  # 10 classes

# Optimize only final layer
optimizer = HybridBFO(
    model.fc.parameters(),
    population_size=20,
    gradient_weight=0.6
)

# Training loop
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            return loss.item()
        
        loss = optimizer.step(closure)
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}')
```

### Example 2: Hyperparameter Optimization

```python
from pytorch_bfo_optimizer import AdaptiveBFO

class HyperparameterModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable hyperparameters
        self.lr = nn.Parameter(torch.tensor(0.01))
        self.momentum = nn.Parameter(torch.tensor(0.9))
        self.weight_decay = nn.Parameter(torch.tensor(0.0001))
    
    def forward(self):
        # Return hyperparameters constrained to valid ranges
        return {
            'lr': torch.sigmoid(self.lr) * 0.1,
            'momentum': torch.sigmoid(self.momentum),
            'weight_decay': torch.sigmoid(self.weight_decay) * 0.01
        }

# Optimize hyperparameters
hyper_model = HyperparameterModel()
hyper_optimizer = AdaptiveBFO(hyper_model.parameters())

def hyper_closure():
    # Get current hyperparameters
    hparams = hyper_model()
    
    # Train model with these hyperparameters
    val_loss = train_with_hyperparams(hparams)
    
    return val_loss

# Find best hyperparameters
for step in range(50):
    loss = hyper_optimizer.step(hyper_closure)
    print(f'Step {step}: Validation Loss = {loss:.4f}')
```

### Example 3: Reinforcement Learning

```python
from pytorch_bfo_optimizer import BFO

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# Initialize policy
policy = PolicyNetwork(state_dim=10, action_dim=4)
optimizer = BFO(
    policy.parameters(),
    population_size=30,
    use_swarming=True
)

def rl_closure():
    # Evaluate policy
    total_reward = 0
    for episode in range(10):
        state = env.reset()
        done = False
        
        while not done:
            with torch.no_grad():
                action = policy(torch.tensor(state))
            next_state, reward, done, _ = env.step(action.numpy())
            total_reward += reward
            state = next_state
    
    # BFO minimizes, so return negative reward
    return -total_reward / 10

# Optimize policy
for generation in range(100):
    avg_reward = -optimizer.step(rl_closure)
    print(f'Generation {generation}: Avg Reward = {avg_reward:.2f}')
```

## Tips for Success

1. **Start Simple**: Begin with default parameters and adjust based on results
2. **Monitor Progress**: Track loss over iterations to ensure convergence
3. **Use Validation**: Always validate on held-out data
4. **Profile Performance**: Use PyTorch profiler to identify bottlenecks
5. **Experiment**: BFO has many parameters - experiment to find optimal settings

## Further Resources

- [API Reference](API_REFERENCE.md)
- [Examples Directory](../examples/)
- [Research Papers](RESEARCH.md)
- [Contributing Guide](../CONTRIBUTING.md)

## Getting Help

- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share experiences
- Documentation: Check the full API reference

Remember that BFO is a population-based optimizer with unique characteristics. It may require different thinking compared to gradient-based optimizers, but can be very effective for the right problems!