# Implementation Examples for Proposed Solutions

## Example 1: Dual Closure Pattern Implementation

### Full Implementation
```python
class HybridBFOv2(BFOv2):
    """
    Hybrid optimizer with separate gradient and non-gradient closures.
    This pattern provides explicit control and avoids gradient errors.
    """
    
    def __init__(self, params, gradient_weight: float = 0.5, **kwargs):
        super().__init__(params, **kwargs)
        self.gradient_weight = gradient_weight
        self._gradient_direction = None
        
    def step(self, closure: Callable, grad_closure: Optional[Callable] = None) -> float:
        """
        Performs optimization step with optional gradient information.
        
        Args:
            closure: Non-gradient closure that returns scalar loss
            grad_closure: Optional gradient closure for hybrid mode
            
        Returns:
            Best fitness value found
        """
        # Handle gradient information if provided
        if grad_closure is not None and self.gradient_weight > 0:
            self._process_gradients(grad_closure)
        
        # Perform BFO step with non-gradient closure
        return super().step(closure)
    
    def _process_gradients(self, grad_closure: Callable):
        """Process gradient information for hybrid optimization"""
        # Check if parameters support gradients
        params_with_grad = [p for p in self.param_groups[0]["params"] if p.requires_grad]
        
        if not params_with_grad:
            logger.warning("No parameters require gradients, skipping gradient processing")
            return
        
        # Compute gradients
        try:
            loss = grad_closure()
            
            # Extract gradient direction
            grads = []
            for p in self.param_groups[0]["params"]:
                if p.grad is not None:
                    grads.append(p.grad.view(-1).clone())
                else:
                    grads.append(torch.zeros_like(p.view(-1)))
            
            self._gradient_direction = torch.cat(grads)
            
            # Apply gradient bias to population
            self._bias_population_toward_gradient()
            
        except Exception as e:
            logger.warning(f"Gradient computation failed: {e}, continuing with pure BFO")
    
    def _bias_population_toward_gradient(self):
        """Bias population toward gradient direction"""
        if self._gradient_direction is None:
            return
        
        # Normalize gradient direction
        grad_norm = torch.norm(self._gradient_direction)
        if grad_norm > 0:
            grad_direction = self._gradient_direction / grad_norm
            
            # Bias each population member
            for i in range(self.population.shape[0]):
                # Weighted combination
                self.population[i] = (
                    (1 - self.gradient_weight) * self.population[i] +
                    self.gradient_weight * (self.param_vector - self.current_step_size * grad_direction)
                )
```

### Usage Example
```python
# Model setup
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
criterion = nn.CrossEntropyLoss()
optimizer = HybridBFOv2(model.parameters(), gradient_weight=0.3)

# Non-gradient closure (always required)
def eval_closure():
    with torch.no_grad():
        output = model(data)
        loss = criterion(output, target)
        return loss.item()

# Gradient closure (optional)
def grad_closure():
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    return loss.item()

# Training loop
for epoch in range(num_epochs):
    for data, target in dataloader:
        # Hybrid mode with both closures
        loss = optimizer.step(eval_closure, grad_closure)
        
        # Or pure BFO mode
        # loss = optimizer.step(eval_closure)
```

## Example 2: torch.compile Compatible Implementation

### Vectorized Population Evaluation
```python
import torch
from torch import vmap
from functools import partial

class CompiledBFOv2(BFOv2):
    """BFO implementation optimized for torch.compile"""
    
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        
        # Pre-compile critical functions
        if self.compile_mode:
            self._compiled_evaluate = torch.compile(
                self._vectorized_evaluate,
                mode=self.compile_mode,
                fullgraph=True
            )
    
    @staticmethod
    def _evaluate_single(params_flat, param_shapes, closure_fn):
        """Evaluate a single population member"""
        # Unflatten parameters
        offset = 0
        params = []
        for shape in param_shapes:
            numel = shape.numel()
            param = params_flat[offset:offset + numel].view(shape)
            params.append(param)
            offset += numel
        
        # Evaluate closure
        return closure_fn(*params)
    
    def _vectorized_evaluate(self, population, closure_fn):
        """Vectorized population evaluation for torch.compile"""
        # Use vmap for efficient vectorization
        param_shapes = [p.shape for p in self.param_groups[0]["params"]]
        
        # Create vectorized evaluation function
        eval_vmap = vmap(partial(self._evaluate_single, 
                                 param_shapes=param_shapes,
                                 closure_fn=closure_fn))
        
        # Evaluate entire population at once
        return eval_vmap(population)
    
    def _parallel_evaluate_population(self, closure: Callable) -> torch.Tensor:
        """Compiled population evaluation"""
        if hasattr(self, '_compiled_evaluate'):
            # Use compiled version
            return self._compiled_evaluate(self.population, closure)
        else:
            # Fallback to original
            return super()._parallel_evaluate_population(closure)
```

### Avoiding Graph Breaks
```python
class GraphFriendlyBFO(BFOv2):
    """Implementation that avoids torch.compile graph breaks"""
    
    def _optimization_step(self, closure, population, best_params, best_fitness, current_iter):
        """Optimization step without .item() calls"""
        # Keep everything as tensors
        best_fitness_tensor = torch.tensor(best_fitness, device=self.device)
        
        # Population evaluation
        fitness = self._parallel_evaluate_population(closure)
        
        # Find minimum without .item()
        min_fitness, min_idx = torch.min(fitness, dim=0)
        
        # Update best without graph break
        improved = min_fitness < best_fitness_tensor
        best_fitness_tensor = torch.where(improved, min_fitness, best_fitness_tensor)
        best_params = torch.where(
            improved.unsqueeze(-1),
            population[min_idx],
            best_params
        )
        
        # Continue optimization...
        # Only convert to Python scalar at the very end
        return population, best_params, best_fitness_tensor.item()
```

## Example 3: Configuration-Based Solution

### Implementation with Config Class
```python
from dataclasses import dataclass
from enum import Enum

class GradientMode(Enum):
    AUTO = "auto"
    ALWAYS = "always"
    NEVER = "never"
    HYBRID = "hybrid"

@dataclass
class HybridConfig:
    """Configuration for HybridBFOv2 behavior"""
    gradient_mode: GradientMode = GradientMode.AUTO
    gradient_weight: float = 0.5
    fallback_on_error: bool = True
    compile_mode: str = "none"
    verbose_errors: bool = False
    
class ConfigurableHybridBFO(BFOv2):
    """Hybrid optimizer with configuration-based behavior"""
    
    def __init__(self, params, config: HybridConfig = None, **kwargs):
        super().__init__(params, **kwargs)
        self.config = config or HybridConfig()
        
        # Configure gradient behavior
        self._setup_gradient_mode()
    
    def _setup_gradient_mode(self):
        """Configure gradient handling based on mode"""
        if self.config.gradient_mode == GradientMode.NEVER:
            # Disable gradients for all parameters
            for p in self.param_groups[0]["params"]:
                p.requires_grad_(False)
        elif self.config.gradient_mode == GradientMode.ALWAYS:
            # Enable gradients for all parameters
            for p in self.param_groups[0]["params"]:
                p.requires_grad_(True)
    
    def step(self, closure: Callable) -> float:
        """Step with configuration-based behavior"""
        try:
            if self.config.gradient_mode == GradientMode.AUTO:
                return self._auto_step(closure)
            elif self.config.gradient_mode == GradientMode.HYBRID:
                return self._hybrid_step(closure)
            elif self.config.gradient_mode == GradientMode.ALWAYS:
                return self._gradient_step(closure)
            else:  # NEVER
                return self._pure_bfo_step(closure)
        except Exception as e:
            if self.config.fallback_on_error:
                logger.warning(f"Error in step: {e}, falling back to pure BFO")
                return self._pure_bfo_step(closure)
            raise
    
    def _auto_step(self, closure):
        """Automatically detect and handle gradient mode"""
        # Try to detect if closure uses gradients
        with torch.no_grad():
            # Test evaluation
            try:
                test_result = closure()
                # If successful without gradients, use pure BFO
                return self._pure_bfo_step(closure)
            except Exception:
                # Might need gradients
                return self._hybrid_step(closure)
```

### Usage with Configuration
```python
# Pure BFO mode
bfo_config = HybridConfig(gradient_mode=GradientMode.NEVER)
optimizer = ConfigurableHybridBFO(params, config=bfo_config)

# Always use gradients
grad_config = HybridConfig(gradient_mode=GradientMode.ALWAYS)
optimizer = ConfigurableHybridBFO(params, config=grad_config)

# Auto-detect mode
auto_config = HybridConfig(
    gradient_mode=GradientMode.AUTO,
    fallback_on_error=True,
    verbose_errors=True
)
optimizer = ConfigurableHybridBFO(params, config=auto_config)
```

## Example 4: Minimal Fix Implementation

### Quick Fix for Immediate Use
```python
def create_fixed_hybridbfo():
    """Factory function that patches HybridBFOv2"""
    
    class FixedHybridBFO(HybridBFOv2):
        def _evaluate_closure(self, closure: Callable) -> float:
            """Fixed closure evaluation that handles gradient errors"""
            try:
                # First try with gradients if available
                if any(p.requires_grad for p in self.param_groups[0]["params"]):
                    try:
                        result = closure()
                        if isinstance(result, torch.Tensor):
                            result = result.item()
                        return float(result)
                    except RuntimeError as e:
                        if "does not require grad" in str(e):
                            # Fall through to no-grad evaluation
                            pass
                        else:
                            raise
                
                # Evaluate without gradients
                with torch.no_grad():
                    # Temporarily get closure result
                    result = closure()
                    if isinstance(result, torch.Tensor):
                        result = result.item()
                    return float(result)
                    
            except Exception as e:
                if self.verbose:
                    logger.error(f"Closure evaluation failed: {e}")
                return float('inf')
    
    return FixedHybridBFO

# Usage
HybridBFOv2Fixed = create_fixed_hybridbfo()
optimizer = HybridBFOv2Fixed(model.parameters())
```

## Testing the Solutions

### Test Suite for Dual Closure Pattern
```python
def test_dual_closure_pattern():
    """Test the dual closure implementation"""
    
    # Test 1: With gradients
    x = torch.nn.Parameter(torch.randn(10), requires_grad=True)
    opt = HybridBFOv2([x], gradient_weight=0.5)
    
    def closure():
        return (x ** 2).sum().item()
    
    def grad_closure():
        opt.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        return loss.item()
    
    # Should work with both closures
    loss = opt.step(closure, grad_closure)
    assert loss < float('inf')
    
    # Test 2: Without gradients
    y = torch.nn.Parameter(torch.randn(10), requires_grad=False)
    opt2 = HybridBFOv2([y], gradient_weight=0.5)
    
    # Should work with just non-gradient closure
    loss2 = opt2.step(lambda: (y ** 2).sum().item())
    assert loss2 < float('inf')
    
    print("Dual closure pattern tests passed!")

# Run test
test_dual_closure_pattern()
```

These implementation examples provide concrete code for each proposed solution, making it easier to evaluate and implement the best approach for fixing HybridBFOv2.