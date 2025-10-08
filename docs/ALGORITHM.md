# Bacterial Foraging Optimization: Algorithm & Enhancements

## Overview

This document explains the BFO implementation, its core mechanisms, and the enhancements made for improved performance and PyTorch integration.

## Core BFO Algorithm (Passino 2002)

Bacterial Foraging Optimization mimics the foraging behavior of E. coli bacteria in search of nutrients. The algorithm consists of four main mechanisms:

### 1. Chemotaxis (Tumble and Swim)

**Purpose**: Local search through movement in random directions, continuing if fitness improves.

**Canonical equation**: 
```
θ(i, j+1, k, l) = θ(i, j, k, l) + C(i) × Δ(i)
```

Where:
- `θ(i, j, k, l)` = position of bacterium i at chemotaxis step j, reproduction step k, elimination step l
- `C(i)` = step size for bacterium i
- `Δ(i)` = random direction vector (tumble)

**Swimming**: If fitness improves after a tumble, continue moving in the same direction for up to `N_s` (swim_length) steps.

### 2. Swarming (Cell-to-Cell Attraction/Repulsion)

**Purpose**: Coordinate bacteria to converge toward promising regions while maintaining diversity.

**Formula**:
```
J_cc(θ, P) = Σ_i [J_cc^attract(θ, θ^i) + J_cc^repel(θ, θ^i)]

Where:
  J_cc^attract = -d_attract × exp(-w_attract × ||θ - θ^i||²)
  J_cc^repel = h_repel × exp(-w_repel × ||θ - θ^i||²)
```

This creates attraction at medium distances and repulsion at close range, promoting exploration while preventing premature convergence.

### 3. Reproduction

**Purpose**: Elite selection that favors successful bacteria.

**Process**:
1. Compute health of each bacterium (cumulative fitness over chemotaxis steps)
2. Sort bacteria by health
3. Keep top 50% (Sr = S/2)
4. Duplicate them to replace bottom 50%

### 4. Elimination-Dispersal

**Purpose**: Escape local optima through random restart with low probability.

**Process**: With probability P_ed, randomly eliminate each bacterium and place it at a new random location in the search space.

---

## Enhancements in This Implementation

Our implementation includes several modern enhancements to improve convergence and robustness:

### Enhancement 1: Lévy Flights for Exploration

**What**: Replace uniform random tumbles with Lévy flight steps.

**Why**: Lévy flights have heavy-tailed distributions, enabling occasional large jumps while maintaining local search. This improves exploration in complex landscapes.

**Implementation**: Mantegna (1994) algorithm for generating Lévy-stable random variables with α ∈ [1, 2].

**Evidence**: Lévy flights have been successfully applied to various metaheuristic algorithms, improving their ability to escape local optima while maintaining convergence. The heavy-tailed distribution allows for both fine-grained local search and occasional long-distance exploration.

**Formula**:
```python
Lévy(α) ~ u / |v|^(1/α)
where u ~ N(0, σ_u²), v ~ N(0, 1)
```

### Enhancement 2: Normalized Chemotaxis Directions

**What**: Normalize all direction vectors to unit length before applying step size.

**Why**: Mathematical necessity for high-dimensional problems.

**Rationale**: Without normalization, the expected distance traveled in n dimensions is proportional to √n, making step_size parameter dimension-dependent. Normalization ensures:
```
||θ(i, j+1) - θ(i, j)|| = C(i) × levy_scale
```
regardless of dimensionality.

**Implementation**:
```python
direction = levy_steps / (||levy_steps|| + ε)
θ_new = θ_old + step_size × levy_scale × direction
```

### Enhancement 3: Adaptive Step Sizing

**What**: Dynamically adjust step size based on optimization progress.

**Why**: Large steps early (exploration) → small steps later (exploitation).

**Three schedules available**:

1. **Adaptive** (default): Performance-based
   - Improvement → increase step size by 1.05
   - Stagnation → decrease step size by 0.95

2. **Cosine**: Smooth annealing
   - `C(t) = C_min + 0.5 × (C_max - C_min) × (1 + cos(π × t/T))`

3. **Linear**: Linear decay
   - `C(t) = C_max - (C_max - C_min) × t/T`

**Evidence**: Adaptive parameter control is standard in modern optimization (e.g., AdaGrad, Adam). Many BFO variants have shown improved convergence with adaptive step sizes.

### Enhancement 4: Diversity-Based Elimination

**What**: Adjust elimination probability based on population diversity and stagnation.

**Why**: Trigger more elimination when population converges prematurely or gets stuck.

**Implementation**:
```python
diversity = mean(||bacterium_i - population_mean||)
if stagnation_count > 5:
    P_ed *= 1.5
elif diversity < threshold:
    P_ed *= 3.0
```

**Evidence**: Diversity maintenance is crucial in evolutionary algorithms. Triggering exploration when diversity drops prevents premature convergence.

### Enhancement 5: Smart Reinitialization

**What**: Eliminated bacteria respawn near the current best solution (with noise).

**Why**: Don't lose progress by dispersing completely randomly.

**Implementation**:
```python
new_position = best_params + randn() × current_step_size
```

**Rationale**: This is a heuristic that balances exploration (via noise) with exploitation (near best solution). Unlike pure random dispersal, it maintains information about good regions while still allowing escape from local optima.

---

## PyTorch-Specific Features

### GPU Acceleration
All tensor operations are device-agnostic and work on CPU, CUDA, and MPS (Apple Silicon).

### Mixed Precision Support
Automatically handles FP16, BF16, and FP32 dtypes. Swarming computation upcasts to FP32 for numerical stability when needed.

### Vectorized Operations
- Parallel fitness evaluation for all bacteria
- Vectorized swimming (continues only for improving bacteria)
- Batch norm/distance computations

### State Checkpointing
Full optimizer state (population, fitness history, RNG state) can be saved and restored, enabling:
- Training interruption/resumption
- Hyperparameter continuation
- Reproducible experiments

---

## Design Decisions

### Why Not Multiple Modes?

We maintain **one focused implementation** with the enhancements as defaults because:

1. **Simplicity**: Users don't need to choose between "canonical" vs "enhanced" modes
2. **Testing**: One code path is easier to test and maintain
3. **Performance**: The enhancements generally improve convergence across problem types

If canonical Passino 2002 behavior is needed, users can:
- Set `levy_alpha=2.0` (Gaussian, not Lévy)
- Set `normalize_directions=False`
- Set `step_schedule='linear'` with fixed range
- Adjust `elimination_prob` to a fixed value

### Why These Enhancements?

Each enhancement addresses a known limitation:

- **Lévy flights**: Better exploration in multimodal landscapes
- **Normalized directions**: Dimension-independent behavior
- **Adaptive steps**: Automatic exploration-exploitation balance
- **Diversity-based elimination**: Prevents premature convergence
- **Smart reinitialization**: Retains useful information

---

## Convergence Properties

### Expected Behavior

- **Unimodal functions** (e.g., Sphere): Fast convergence, linear/quadratic rate
- **Multimodal functions** (e.g., Rastrigin): Slower but reliable, benefits from Lévy flights
- **Narrow valleys** (e.g., Rosenbrock): Benefits from adaptive step sizing

### Typical Function Evaluations

For a problem with `D` dimensions and `S` bacteria:

```
FE per step() = S × N_c × N_rep × N_ed × (1 + N_swim/2)
```

Where:
- N_c = chemotaxis_steps
- N_rep = reproduction_steps  
- N_ed = elimination_steps
- N_swim = swim_length (averaged)

Default (D=10, S=50): ~10,000-20,000 FEs per step()

---

## References

### Canonical Algorithm
- **Passino, K. M. (2002)**: "Biomimicry of Bacterial Foraging for Distributed Optimization and Control" - IEEE Control Systems Magazine

### Enhancements
- **Mantegna, R. N. (1994)**: "Fast, accurate algorithm for numerical simulation of Levy stable stochastic processes" - Physical Review E
- **Yang, X. S. (2008)**: "Nature-Inspired Metaheuristic Algorithms" - Lévy flight applications
- **Adaptive BFO variants**: Multiple papers (2010-2024) show adaptive parameter control improves convergence
- **Diversity maintenance**: Standard practice in evolutionary computation (Eiben & Smith, 2015)

### PyTorch Integration
- Follows `torch.optim.Optimizer` interface conventions
- Compatible with standard PyTorch training workflows
- Supports modern features (AMP, DDP, checkpointing)

---

## When to Use BFO

**Good for**:
- Black-box optimization (gradient-free)
- Non-differentiable objectives
- Noisy or discontinuous functions
- Hyperparameter search
- Neural architecture search (small search spaces)

**Not ideal for**:
- Large-scale deep learning (gradient-based methods better)
- Real-time applications (many function evaluations needed)
- Very high dimensions (>1000) without proper tuning

**Compared to alternatives**:
- vs Adam/SGD: Use when gradients unavailable or unreliable
- vs PSO: Similar performance, different exploration strategy
- vs CMA-ES: BFO has better multimodal exploration via Lévy flights
- vs Random Search: Much more efficient, structured exploration

