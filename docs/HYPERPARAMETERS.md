# BFO Hyperparameter Tuning Guide

Comprehensive guide for selecting and tuning BFO optimizer hyperparameters.

## Quick Reference Table

| Parameter | Default | Range | Problem Type → Value |
|-----------|---------|-------|---------------------|
| **population_size** | 50 | 10-200 | Small/simple: 20-30 <br> Medium: 50-100 <br> Large/complex: 100-200 |
| **chemotaxis_steps** | 10 | 3-20 | Unimodal: 5-10 <br> Multimodal: 10-20 |
| **swim_length** | 4 | 0-10 | Smooth: 6-10 <br> Noisy: 2-4 <br> Discontinuous: 0-2 |
| **reproduction_steps** | 4 | 2-10 | Fast: 2-3 <br> Balanced: 4-6 <br> Thorough: 8-10 |
| **elimination_steps** | 2 | 1-5 | Simple: 1-2 <br> Multimodal: 3-5 |
| **elimination_prob** | 0.25 | 0.1-0.5 | Unimodal: 0.1-0.2 <br> Multimodal: 0.3-0.5 |
| **step_size_max** | 0.1 | 1e-3 to 1.0 | Match problem scale |
| **step_size_min** | 1e-4 | 1e-6 to 1e-2 | Usually 0.01× step_size_max |
| **levy_alpha** | 1.5 | 1.0-2.0 | Exploration: 1.3-1.5 <br> Exploitation: 1.8-2.0 |
| **lr** | 0.01 | 1e-4 to 1.0 | Standard learning rate |

---

## Core Parameters

### population_size

**What it does**: Number of bacteria (candidate solutions) in the population.

**Impact**:
- **Larger**: Better exploration, more robust, but slower (more function evaluations)
- **Smaller**: Faster, but may miss good solutions

**Guidelines**:
```python
# Problem complexity → population size
dimensions = parameter_count

if dimensions <= 10:
    population_size = 20-30
elif dimensions <= 100:
    population_size = 50-100
else:
    population_size = max(100, dimensions // 2)
```

**Examples**:
- Simple 2D optimization: `population_size=20`
- Small neural network (1000 params): `population_size=50`
- Hyperparameter search (5-10 dims): `population_size=30`
- Complex landscape: `population_size=100`

**Memory Usage**: `O(population_size × num_parameters)`
- 50 bacteria × 10K parameters × 4 bytes (FP32) = ~2 MB

---

### chemotaxis_steps

**What it does**: Number of tumble-and-swim iterations per reproduction cycle.

**Impact**:
- **More steps**: Finer local search, better exploitation
- **Fewer steps**: Faster, more exploration

**Guidelines**:
```python
# Landscape type → chemotaxis steps
if landscape_is_smooth:
    chemotaxis_steps = 15-20  # Exploit gradient-like structure
elif landscape_is_multimodal:
    chemotaxis_steps = 10-15  # Balance exploration/exploitation
elif landscape_is_noisy:
    chemotaxis_steps = 5-10   # Don't over-optimize noise
```

**Trade-off**: Each step costs `population_size` evaluations
- 10 steps × 50 bacteria = 500 evaluations per reproduction cycle

---

### swim_length

**What it does**: Maximum consecutive moves in a good direction after improvement.

**Impact**:
- **Longer**: Exploits good directions more aggressively
- **Shorter** or **0**: More exploration, less exploitation

**Guidelines**:
```python
# Function smoothness → swim length
if function_is_smooth_and_continuous:
    swim_length = 6-10  # Exploit gradients
elif function_is_noisy:
    swim_length = 2-4   # Don't follow noise
elif function_is_discontinuous:
    swim_length = 0-2   # Minimal swimming
```

**Examples**:
- Sphere function (smooth): `swim_length=10`
- Rastrigin (multimodal): `swim_length=4`
- Noisy sensor data: `swim_length=2`

**Cost**: Adds `0` to `swim_length` evaluations per bacterium per chemotaxis step (average: `swim_length/2`)

---

### reproduction_steps & elimination_steps

**What they do**:
- **reproduction_steps**: How many chemotaxis cycles before elite selection
- **elimination_steps**: How many reproduction cycles before random restart

**Impact**:
- **More reproduction steps**: More exploitation before culling weak solutions
- **More elimination steps**: More exploration via random restarts

**Guidelines**:
```python
# Problem characteristics → steps
if problem_is_simple_unimodal:
    reproduction_steps = 2-3
    elimination_steps = 1-2
elif problem_is_complex_multimodal:
    reproduction_steps = 4-6
    elimination_steps = 3-5
```

**Total evaluations per `step()` call**:
```
FEs ≈ population_size × chemotaxis_steps × reproduction_steps × elimination_steps × (1 + swim_length/2)
```

Example: 50 × 10 × 4 × 2 × (1 + 4/2) = 12,000 evaluations

---

### elimination_prob

**What it does**: Probability each bacterium is randomly reinitialized.

**Impact**:
- **Higher**: More exploration, escapes local minima
- **Lower**: More exploitation, refines current solutions

**Guidelines**:
```python
# Landscape modality → elimination probability
if landscape_is_unimodal:
    elimination_prob = 0.1-0.15  # Minimal disruption
elif landscape_is_multimodal:
    elimination_prob = 0.25-0.35  # Default works well
elif landscape_has_many_local_minima:
    elimination_prob = 0.4-0.5   # Aggressive exploration
```

**Adaptive behavior**: The optimizer automatically increases `elimination_prob` when:
- Population diversity drops below threshold
- Stagnation detected (no improvement for several steps)

---

### step_size_max & step_size_min

**What they do**: Control the magnitude of chemotaxis movements.

**Impact**:
- **Larger steps**: Faster exploration, may overshoot
- **Smaller steps**: Finer refinement, may converge slowly

**Guidelines**:
```python
# Problem scale → step size
problem_range = max_value - min_value

# Rule of thumb: max step ≈ 10% of search range
step_size_max = 0.1 * problem_range
step_size_min = 0.01 * step_size_max

# Examples:
# Problem in [-1, 1]: step_size_max = 0.2
# Problem in [-100, 100]: step_size_max = 20.0
# Problem in [0, 1]: step_size_max = 0.1
```

**With domain bounds**:
```python
optimizer = BFO(
    params,
    step_size_max=0.1 * (upper - lower),
    domain_bounds=(lower, upper)
)
```

**Adaptive schedules** (default: `step_schedule='adaptive'`):
- Step size automatically decreases when converging
- Step size increases when exploring

---

### levy_alpha

**What it does**: Controls the Lévy flight distribution for exploration.

**Impact**:
- **α = 1.0**: Cauchy distribution (heavy tails, long jumps)
- **α = 1.5**: Balanced (default, good for most problems)
- **α = 2.0**: Gaussian-like (shorter jumps, local search)

**Guidelines**:
```python
# Exploration need → levy_alpha
if need_global_exploration:
    levy_alpha = 1.3-1.5  # Long-range exploration
elif problem_is_local:
    levy_alpha = 1.8-2.0  # Fine local search
```

**Advanced**: Combine with `levy_schedule='linear-decrease'` (default) for automatic exploration → exploitation transition.

---

## Variant-Specific Parameters

### AdaptiveBFO

Automatically adjusts population size based on progress.

```python
from bfo_torch import AdaptiveBFO

optimizer = AdaptiveBFO(
    params,
    population_size=50,          # Initial size
    min_population_size=20,      # Lower bound
    max_population_size=150,     # Upper bound
    adaptation_rate=0.1,         # Growth/shrink rate
    diversity_threshold=1e-3,    # Trigger elimination if diversity < threshold
)
```

**When to use**:
- Uncertain about problem difficulty
- Problem difficulty changes during optimization
- Want automatic parameter tuning

---

### HybridBFO

Combines BFO with gradient information.

```python
from bfo_torch import HybridBFO

optimizer = HybridBFO(
    params,
    gradient_weight=0.5,    # Balance: 0=pure BFO, 1=pure gradient
    momentum=0.9,           # Momentum for gradient updates
    enable_momentum=True,   # Use momentum buffer
)
```

**Guidelines**:
- **gradient_weight=0.3-0.5**: Balanced (recommended for noisy gradients)
- **gradient_weight=0.7-0.9**: Trust gradients (smooth, well-behaved functions)
- **gradient_weight=0.1-0.2**: Mostly BFO (unreliable gradients)

**When to use**:
- Gradients available but noisy
- Want exploration + gradient information
- Escaping local minima in gradient descent

---

## Problem-Specific Recommendations

### Unimodal Smooth Functions (Sphere, Quadratic)

```python
optimizer = BFO(
    params,
    population_size=20,
    chemotaxis_steps=10,
    swim_length=8,
    reproduction_steps=3,
    elimination_steps=1,
    elimination_prob=0.1,
    step_schedule='linear',  # Smooth decay
)
```

---

### Multimodal Functions (Rastrigin, Ackley)

```python
optimizer = BFO(
    params,
    population_size=50,
    chemotaxis_steps=10,
    swim_length=4,
    reproduction_steps=4,
    elimination_steps=3,
    elimination_prob=0.3,
    levy_alpha=1.5,  # Good exploration
    levy_schedule='linear-decrease',
)
```

---

### Narrow Valleys (Rosenbrock)

```python
optimizer = BFO(
    params,
    population_size=50,
    chemotaxis_steps=15,
    swim_length=6,
    reproduction_steps=5,
    elimination_steps=2,
    step_size_max=0.3,  # Larger steps
    step_schedule='adaptive',  # Adjust to valley
)
```

---

### High-Dimensional (>100 dims)

```python
optimizer = BFO(
    params,
    population_size=min(200, num_dims),  # Scale with dimensions
    chemotaxis_steps=5,  # Keep reasonable FE budget
    swim_length=4,
    reproduction_steps=2,
    elimination_steps=2,
    normalize_directions=True,  # Essential for high-dim
)
```

---

### Noisy Objectives

```python
optimizer = BFO(
    params,
    population_size=80,  # Larger to average noise
    chemotaxis_steps=5,  # Don't over-fit noise
    swim_length=2,  # Don't chase noise
    reproduction_steps=6,  # More averaging
    elimination_steps=2,
    step_schedule='adaptive',
)
```

---

### Hyperparameter Search

```python
# Searching over N hyperparameters
optimizer = BFO(
    hyperparams,
    population_size=max(20, 3 * N),  # Rule: 3-5× dims
    chemotaxis_steps=8,
    swim_length=4,
    reproduction_steps=3,
    elimination_steps=2,
    domain_bounds=(lower, upper),  # Always constrain
    step_size_max=0.2 * (upper - lower),
)
```

---

## Budget Constraints

If you have limited function evaluations:

```python
# Calculate expected FEs per step
fe_per_step = (
    population_size *
    chemotaxis_steps *
    reproduction_steps *
    elimination_steps *
    (1 + swim_length / 2)
)

# Example: 10,000 FE budget, want 10 steps
target_fe_per_step = 10000 / 10  # 1000 FEs

# Option 1: Reduce population
population_size = 20
chemotaxis_steps = 10
# → 20 × 10 × 4 × 2 × 3 = 4,800 FEs

# Option 2: Reduce cycles
population_size = 50
chemotaxis_steps = 5
reproduction_steps = 2
elimination_steps = 1
# → 50 × 5 × 2 × 1 × 3 = 1,500 FEs

# Option 3: Use max_fe parameter
optimizer.step(closure, max_fe=1000)  # Hard limit per step
```

---

## Debugging Checklist

**Poor convergence?**
1. Is `step_size_max` appropriate for problem scale?
2. Is `population_size` large enough (>= 20)?
3. Try `HybridBFO` if gradients available
4. Increase `chemotaxis_steps` (more local search)

**Stuck in local minimum?**
1. Increase `elimination_prob` (0.3-0.5)
2. Increase `elimination_steps` (3-5)
3. Use `levy_alpha=1.5` (more exploration)

**Too slow?**
1. Decrease `population_size`
2. Decrease `chemotaxis_steps`
3. Set `max_fe` budget
4. Use `early_stopping=True`

**Diverging?**
1. Add `domain_bounds`
2. Decrease `step_size_max`
3. Decrease `lr`

---

## Further Resources

- See [ALGORITHM.md](ALGORITHM.md) for implementation details
- See [QUICKSTART.md](QUICKSTART.md) for usage examples
- Check [examples/](../examples/) for complete scripts

