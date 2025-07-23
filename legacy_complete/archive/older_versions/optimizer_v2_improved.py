"""
PyTorch BFO Optimizer V2 Implementation - Improved Version
Fixes gradient handling, convergence, and performance issues identified in testing.
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Callable, List, Dict, Any, Tuple
import numpy as np
from .debug_utils import log_debug, DebugContext
import logging

logger = logging.getLogger(__name__)


class BFOv2(Optimizer):
    """
    Enhanced Bacterial Foraging Optimization (BFO) for PyTorch 2.8+.
    
    This version includes improvements for:
    - Better convergence with adaptive parameters
    - Efficient parallel evaluation
    - Robust error handling
    - CPU/GPU auto-detection
    """
    
    def __init__(
        self,
        params,
        population_size: int = 20,  # Increased from 10 for better convergence
        chem_steps: int = 10,
        swim_length: int = 4,
        repro_steps: int = 4,
        elim_steps: int = 2,
        elim_prob: float = 0.25,
        attract_factor: float = 0.1,
        repel_factor: float = 0.1,
        levy_alpha: float = 1.5,
        step_size_min: float = 1e-4,
        step_size_max: float = 0.01,
        device_type: str = 'auto',
        compile_mode: str = 'default',
        parallel_eval: bool = True,
        batch_size: int = 8,  # Reduced for CPU efficiency
        early_stopping: bool = True,
        convergence_tol: float = 1e-6,
        convergence_patience: int = 5,  # Reduced for faster testing
        verbose: bool = False,
        **kwargs
    ):
        if not 0 < elim_prob <= 1:
            raise ValueError(f"elim_prob must be in (0, 1], got {elim_prob}")
        
        defaults = dict(
            population_size=population_size,
            chem_steps=chem_steps,
            swim_length=swim_length,
            repro_steps=repro_steps,
            elim_steps=elim_steps,
            elim_prob=elim_prob,
            attract_factor=attract_factor,
            repel_factor=repel_factor,
            levy_alpha=levy_alpha,
            step_size_min=step_size_min,
            step_size_max=step_size_max,
            **kwargs
        )
        
        super().__init__(params, defaults)
        
        # Device setup with auto-detection
        if device_type == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_type)
        
        # Configuration
        self.compile_mode = compile_mode
        self.parallel_eval = parallel_eval
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.convergence_tol = convergence_tol
        self.convergence_patience = convergence_patience
        self.verbose = verbose
        
        # State initialization
        self._initialize_state()
        
        # Compile optimization step if requested
        if compile_mode and compile_mode != 'false':
            try:
                self._compiled_step = torch.compile(
                    self._optimization_step,
                    mode=compile_mode,
                    backend='inductor'
                )
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}, falling back to eager mode")
                self._compiled_step = self._optimization_step
        else:
            self._compiled_step = self._optimization_step
    
    def _initialize_state(self):
        """Initialize optimizer state"""
        # Flatten parameters
        self.param_vector = self._flatten_params()
        param_dim = self.param_vector.shape[0]
        pop_size = self.defaults["population_size"]
        
        # Detect dtype from parameters for AMP compatibility
        if len(self.param_groups) > 0 and len(self.param_groups[0]['params']) > 0:
            self.dtype = self.param_groups[0]['params'][0].dtype
        else:
            self.dtype = torch.float32
        
        # Initialize population around current parameters with correct dtype
        self.population = torch.randn(
            pop_size, param_dim, device=self.device, dtype=self.dtype
        ) * self.defaults["step_size_max"]
        self.population[0] = self.param_vector.clone()
        
        # Best solution tracking
        self.best_params = self.param_vector.clone()
        self.best_fitness = float('inf')
        
        # Convergence tracking
        self.fitness_history = []
        self.stagnation_count = 0
        self.current_iter = 0
        
        # Adaptive parameters
        self.current_step_size = self.defaults["step_size_max"]
        
        if self.verbose:
            logger.info(f"BFOv2 initialized on {self.device} with population_size={pop_size}, dtype={self.dtype}")
    
    def _flatten_params(self) -> torch.Tensor:
        """Flatten all parameters into a single tensor"""
        params = []
        for group in self.param_groups:
            for p in group['params']:
                params.append(p.view(-1))
        return torch.cat(params)
    
    def _unflatten_params(self, flat_params: torch.Tensor) -> List[torch.Tensor]:
        """Convert flat tensor back to parameter shapes"""
        params = []
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.numel()
                params.append(flat_params[offset:offset + numel].view_as(p))
                offset += numel
        return params
    
    def _evaluate_closure(self, closure: Callable) -> float:
        """Safely evaluate the closure function"""
        try:
            with DebugContext("closure_evaluation"):
                result = closure()
                if isinstance(result, torch.Tensor):
                    result = result.item()
                return float(result)
        except Exception as e:
            logger.error(f"Closure evaluation failed: {e}")
            return float('inf')
    
    def _levy_flight(self, size: torch.Size) -> torch.Tensor:
        """Generate Lévy flight steps with improved numerical stability"""
        alpha = self.defaults["levy_alpha"]
        
        # Compute sigma_u more stably
        gamma1 = torch.exp(torch.lgamma(torch.tensor(1 + alpha, device=self.device, dtype=self.dtype)))
        gamma2 = torch.exp(torch.lgamma(torch.tensor((1 + alpha) / 2, device=self.device, dtype=self.dtype)))
        sin_term = torch.sin(torch.tensor(np.pi * alpha / 2, device=self.device, dtype=self.dtype))
        pow_term = torch.pow(torch.tensor(2.0, device=self.device, dtype=self.dtype), (alpha - 1) / 2)
        
        sigma_u = (gamma1 * sin_term / (gamma2 * alpha * pow_term)) ** (1 / alpha)
        
        # Generate stable random samples with retry logic
        max_retries = 3
        for retry in range(max_retries):
            u = torch.randn(size, device=self.device, dtype=self.dtype) * sigma_u
            v = torch.randn(size, device=self.device, dtype=self.dtype)
            
            # Compute step with numerical safeguards
            v_abs = torch.abs(v) + 1e-10
            v_power = v_abs ** (1 / alpha)
            
            # Check for numerical issues
            if torch.isfinite(v_power).all():
                step = u / v_power
                
                # Check if result is finite
                if torch.isfinite(step).all():
                    # Soft clamp using tanh for smoother gradients
                    step = 10 * torch.tanh(step / 10)
                    return step
        
        # Fallback to standard normal if Lévy flight fails
        logger.warning("Lévy flight numerical issues, falling back to normal distribution")
        return torch.randn(size, device=self.device, dtype=self.dtype)
    
    def _compute_swarming(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute attraction-repulsion swarming behavior using vectorized operations"""
        pop_size = positions.shape[0]
        
        # Compute pairwise distances using cdist for efficiency
        # distances[i,j] = ||positions[i] - positions[j]||
        distances = torch.cdist(positions, positions, p=2) + 1e-10
        
        # Create masks to exclude self-interactions
        mask = ~torch.eye(pop_size, dtype=torch.bool, device=positions.device)
        
        # Compute differences: diff[i,j] = positions[j] - positions[i]
        # Shape: (pop_size, pop_size, dim)
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Normalize differences by distances
        # Shape: (pop_size, pop_size, dim)
        normalized_diff = diff / distances.unsqueeze(-1)
        
        # Compute attraction and repulsion factors
        # Shape: (pop_size, pop_size)
        attract_factor = -self.defaults["attract_factor"] * torch.exp(-distances**2)
        repel_factor = self.defaults["repel_factor"] * torch.exp(-distances)
        
        # Combined factor
        combined_factor = (attract_factor + repel_factor) * mask.float()
        
        # Apply factors and sum over all other bacteria
        # Shape: (pop_size, dim)
        swarming = (combined_factor.unsqueeze(-1) * diff).sum(dim=1)
        
        return swarming
    
    def _vmap_evaluate_population(self, closure: Callable) -> torch.Tensor:
        """
        Vectorized population evaluation using vmap for torch.compile optimization.
        Falls back to batch evaluation if vmap is not available.
        """
        try:
            # Try to use vmap for vectorized evaluation
            from torch import vmap
            
            # Create a functional closure that takes flattened parameters
            def eval_individual(flat_params):
                # Unflatten and update parameters
                params_list = self._unflatten_params(flat_params)
                with torch.no_grad():
                    for p, new_p in zip(self.param_groups[0]["params"], params_list):
                        p.data.copy_(new_p)
                # Evaluate
                return self._evaluate_closure(closure)
            
            # Vectorize the evaluation
            vectorized_eval = vmap(eval_individual)
            fitness = vectorized_eval(self.population)
            
            return fitness
            
        except (ImportError, RuntimeError) as e:
            # Fallback to original batch evaluation
            log_debug("vmap not available or failed, using batch evaluation", error=str(e))
            return self._batch_evaluate_population(closure)
    
    def _batch_evaluate_population(self, closure: Callable) -> torch.Tensor:
        """
        Original batch evaluation method for compatibility.
        """
        pop_size = self.population.shape[0]
        fitness = torch.zeros(pop_size, device=self.device, dtype=self.dtype)
        
        if self.parallel_eval and self.device.type == 'cuda':
            # GPU parallel evaluation
            batch_size = self.batch_size
        else:
            # CPU sequential with smaller batches
            batch_size = min(4, self.batch_size)
        
        for i in range(0, pop_size, batch_size):
            batch_end = min(i + batch_size, pop_size)
            
            # Process batch
            for j in range(i, batch_end):
                params_list = self._unflatten_params(self.population[j])
                # Update parameters
                for p, new_p in zip(self.param_groups[0]["params"], params_list):
                    p.data.copy_(new_p)
                # Evaluate
                fitness[j] = self._evaluate_closure(closure)
        
        return fitness
    
    def _parallel_evaluate_population(self, closure: Callable) -> torch.Tensor:
        """
        Main evaluation method that chooses between vmap and batch evaluation.
        """
        if self.compile_mode and self.compile_mode != 'false':
            # Try vmap for compiled mode
            return self._vmap_evaluate_population(closure)
        else:
            # Use batch evaluation for non-compiled mode
            return self._batch_evaluate_population(closure)
    
    def _optimization_step(
        self,
        closure: Callable,
        population: torch.Tensor,
        best_params: torch.Tensor,
        best_fitness: float,
        current_iter: int
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Core optimization step with improvements"""
        
        pop_size = population.shape[0]
        
        for elim_iter in range(self.defaults["elim_steps"]):
            for repro_iter in range(self.defaults["repro_steps"]):
                for chem_iter in range(self.defaults["chem_steps"]):
                    
                    # Evaluate fitness with improved batching
                    fitness = self._parallel_evaluate_population(closure)
                    
                    # Update best solution - keep as tensor for compile
                    min_idx = torch.argmin(fitness)
                    min_fitness = fitness[min_idx]
                    best_fitness_tensor = torch.tensor(best_fitness, device=self.device, dtype=self.dtype)
                    
                    # Use tensor comparison to avoid graph break
                    improved_solution = min_fitness < best_fitness_tensor
                    if improved_solution:
                        best_fitness_tensor = min_fitness
                        best_params = population[min_idx].clone()
                    
                    # Chemotaxis with adaptive step size
                    directions = self._levy_flight(population.shape)
                    swarming = self._compute_swarming(population)
                    
                    # Adaptive step size based on improvement (keep as tensor)
                    if len(self.fitness_history) > 1:
                        last_fitness = torch.tensor(self.fitness_history[-1], device=self.device, dtype=self.dtype)
                        improvement_rate = torch.abs(last_fitness - best_fitness_tensor) / (torch.abs(last_fitness) + 1e-10)
                        
                        # Create step size as tensor and update
                        step_size_tensor = torch.tensor(self.current_step_size, device=self.device, dtype=self.dtype)
                        step_size_tensor = torch.where(
                            improvement_rate < 0.001,
                            step_size_tensor * 0.95,
                            step_size_tensor * 1.05
                        )
                        step_size_tensor = torch.clamp(
                            step_size_tensor,
                            self.defaults["step_size_min"],
                            self.defaults["step_size_max"]
                        )
                        self.current_step_size = step_size_tensor.item()  # Only convert at end
                    
                    new_positions = population + self.current_step_size * directions + 0.1 * swarming
                    
                    # Swimming with fixed logic
                    new_fitness = self._parallel_evaluate_population(closure)
                    improved = new_fitness < fitness
                    
                    # Update positions for improved solutions
                    population[improved] = new_positions[improved]
                    
                # Reproduction - fixed for odd populations
                if repro_iter < self.defaults["repro_steps"] - 1:
                    sorted_idx = fitness.argsort()
                    half = pop_size // 2
                    if half > 0:
                        num_to_replace = pop_size - half
                        population[sorted_idx[half:]] = population[sorted_idx[:num_to_replace]].clone()
            
            # Elimination-dispersal with adaptive probability
            if elim_iter < self.defaults["elim_steps"] - 1:
                # Increase elimination probability if stagnating
                elim_prob = self.defaults["elim_prob"]
                if self.stagnation_count > 3:
                    elim_prob = min(0.5, elim_prob * 1.5)
                
                eliminate = torch.rand(pop_size, device=self.device) < elim_prob
                if eliminate.any():
                    population[eliminate] = (
                        best_params + 
                        torch.randn((eliminate.sum(), population.shape[1]), device=self.device, dtype=self.dtype) * 
                        self.current_step_size * 10
                    )
        
        # Convert best_fitness back to float at the very end
        if 'best_fitness_tensor' in locals() and isinstance(best_fitness_tensor, torch.Tensor):
            best_fitness = best_fitness_tensor.item()
        
        return population, best_params, best_fitness
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> float:
        """Perform optimization step with improved error handling"""
        if closure is None:
            raise ValueError("BFOv2 requires a closure")
        
        # Run optimization
        self.population, self.best_params, self.best_fitness = self._compiled_step(
            closure,
            self.population,
            self.best_params,
            self.best_fitness,
            self.current_iter
        )
        
        # Update parameters with best solution
        params_list = self._unflatten_params(self.best_params)
        for p, new_p in zip(self.param_groups[0]["params"], params_list):
            p.data.copy_(new_p)
        
        # Track convergence
        self.fitness_history.append(self.best_fitness)
        if len(self.fitness_history) > 1:
            if abs(self.fitness_history[-1] - self.fitness_history[-2]) < self.convergence_tol:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
        
        # Early stopping check
        if self.early_stopping and self.stagnation_count >= self.convergence_patience:
            if self.verbose:
                logger.info(f"Early stopping triggered at iteration {self.current_iter}")
        
        self.current_iter += 1
        
        return self.best_fitness
    
    def state_dict(self) -> dict:
        """Save optimizer state including RNG state for reproducibility"""
        state = super().state_dict()
        
        # Add BFO-specific state
        state['bfo_state'] = {
            'population': self.population.clone(),
            'best_params': self.best_params.clone(),
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history.copy(),
            'stagnation_count': self.stagnation_count,
            'current_iter': self.current_iter,
            'current_step_size': self.current_step_size,
        }
        
        # Save RNG states for reproducibility
        state['rng_state'] = {
            'torch_rng': torch.get_rng_state(),
            'numpy_rng': np.random.get_state(),
        }
        
        # Save device-specific RNG state if on GPU
        if self.device.type == 'cuda':
            state['rng_state']['cuda_rng'] = torch.cuda.get_rng_state(self.device)
        
        return state
    
    def load_state_dict(self, state_dict: dict):
        """Load optimizer state including RNG state"""
        # Restore RNG states first for deterministic behavior
        if 'rng_state' in state_dict:
            torch.set_rng_state(state_dict['rng_state']['torch_rng'])
            np.random.set_state(state_dict['rng_state']['numpy_rng'])
            
            if 'cuda_rng' in state_dict['rng_state'] and self.device.type == 'cuda':
                torch.cuda.set_rng_state(state_dict['rng_state']['cuda_rng'], self.device)
        
        # Restore BFO-specific state
        if 'bfo_state' in state_dict:
            bfo_state = state_dict['bfo_state']
            self.population = bfo_state['population'].to(self.device)
            self.best_params = bfo_state['best_params'].to(self.device)
            self.best_fitness = bfo_state['best_fitness']
            self.fitness_history = bfo_state['fitness_history']
            self.stagnation_count = bfo_state['stagnation_count']
            self.current_iter = bfo_state['current_iter']
            self.current_step_size = bfo_state['current_step_size']
        
        # Restore parent state
        super().load_state_dict({k: v for k, v in state_dict.items() if k not in ['bfo_state', 'rng_state']})


class AdaptiveBFOv2(BFOv2):
    """
    Adaptive BFO with dynamic parameter adjustment.
    Improved convergence through adaptive strategies.
    """
    
    def __init__(
        self,
        params,
        adapt_pop_size: bool = True,
        adapt_chem_steps: bool = True,
        adapt_step_size: bool = True,
        min_pop_size: int = 10,
        max_pop_size: int = 50,
        **kwargs
    ):
        # Better defaults for adaptive version
        kwargs.setdefault('population_size', 30)
        kwargs.setdefault('early_stopping', True)
        
        super().__init__(params, **kwargs)
        
        self.adapt_pop_size = adapt_pop_size
        self.adapt_chem_steps = adapt_chem_steps
        self.adapt_step_size = adapt_step_size
        self.min_pop_size = min_pop_size
        self.max_pop_size = max_pop_size
        
        # Adaptive tracking
        self.improvement_history = []
        self.current_fitness = None  # Track current population fitness
        
        # Ensure dtype is inherited
        if not hasattr(self, 'dtype'):
            self.dtype = torch.float32
    
    def _optimization_step(
        self,
        closure: Callable,
        population: torch.Tensor,
        best_params: torch.Tensor,
        best_fitness: float,
        current_iter: int
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Override to track fitness for adaptive resizing"""
        # Call parent optimization step
        population, best_params, best_fitness = super()._optimization_step(
            closure, population, best_params, best_fitness, current_iter
        )
        
        # Store current fitness if we're adapting population size
        if self.adapt_pop_size and hasattr(self, '_last_fitness'):
            self.current_fitness = self._last_fitness
        
        return population, best_params, best_fitness
    
    def _parallel_evaluate_population(self, closure: Callable) -> torch.Tensor:
        """Override to store fitness for adaptive resizing"""
        fitness = super()._parallel_evaluate_population(closure)
        self._last_fitness = fitness.clone()  # Store for potential resizing
        return fitness
    
    def step(self, closure: Optional[Callable] = None) -> float:
        """Step with adaptive parameter updates"""
        old_fitness = self.best_fitness
        fitness = super().step(closure)
        
        # Track improvement
        improvement = old_fitness - fitness
        self.improvement_history.append(improvement)
        
        # Adapt parameters based on performance
        if len(self.improvement_history) >= 5:
            recent_improvement = np.mean(self.improvement_history[-5:])
            
            # Adapt population size
            if self.adapt_pop_size:
                if recent_improvement < 1e-6:  # Stagnating
                    new_size = min(self.max_pop_size, self.defaults["population_size"] + 5)
                elif recent_improvement > 1e-3:  # Good progress
                    new_size = max(self.min_pop_size, self.defaults["population_size"] - 2)
                else:
                    new_size = self.defaults["population_size"]
                
                if new_size != self.defaults["population_size"]:
                    # Pass current fitness if available
                    self._resize_population(new_size, self.current_fitness)
            
            # Adapt chemotactic steps
            if self.adapt_chem_steps:
                if recent_improvement < 1e-6:
                    self.defaults["chem_steps"] = min(20, self.defaults["chem_steps"] + 2)
                elif recent_improvement > 1e-3:
                    self.defaults["chem_steps"] = max(5, self.defaults["chem_steps"] - 1)
        
        return fitness
    
    def _resize_population(self, new_size: int, current_fitness: Optional[torch.Tensor] = None):
        """Resize population while preserving best solutions"""
        current_size = self.population.shape[0]
        
        if new_size > current_size:
            # Add new random individuals
            new_individuals = torch.randn(
                new_size - current_size,
                self.population.shape[1],
                device=self.device,
                dtype=self.dtype
            ) * self.current_step_size
            self.population = torch.cat([self.population, new_individuals], dim=0)
        elif new_size < current_size:
            # Keep best individuals based on current fitness
            if current_fitness is None:
                # If no fitness provided, use random selection but log warning
                logger.warning("Resizing population without fitness values - selecting randomly")
                keep_idx = torch.randperm(current_size, device=self.device)[:new_size]
            else:
                # Sort by fitness and keep the best
                keep_idx = torch.argsort(current_fitness)[:new_size]
            self.population = self.population[keep_idx]
        
        self.defaults["population_size"] = new_size


class HybridBFOv2(BFOv2):
    """
    Refined Hybrid optimizer combining BFOv2 with gradient information.
    Fixes: Gradient checks to avoid errors; improved convergence defaults; basic batching.
    """
    
    def __init__(
        self,
        params,
        gradient_weight: float = 0.5,
        use_momentum: bool = True,
        momentum: float = 0.9,
        population_size: int = 20,  # Increased default for better convergence
        **kwargs
    ):
        # Override for better defaults
        kwargs['population_size'] = population_size
        kwargs['early_stopping'] = True
        kwargs['convergence_patience'] = 5  # Shorter patience for faster tests
        
        super().__init__(params, **kwargs)
        
        self.gradient_weight = gradient_weight
        self.use_momentum = use_momentum
        self.momentum = momentum
        
        # Initialize momentum buffer safely
        if use_momentum:
            self.momentum_buffer = torch.zeros_like(self.param_vector)
        
        if self.verbose:
            logger.info(f"Refined HybridBFOv2 initialized with gradient_weight={gradient_weight}")
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> float:
        """
        Hybrid step with safe gradient handling and fallback.
        Gracefully handles cases where gradients are not available.
        """
        if closure is None:
            raise ValueError("HybridBFO requires a closure")
        
        # Check for gradients safely - now checks if ANY parameter has gradients
        params_with_grad = [p for p in self.param_groups[0]["params"] if p.grad is not None]
        has_any_gradients = len(params_with_grad) > 0
        total_params = len(self.param_groups[0]["params"])
        
        if self.verbose:
            logger.info(f"Step {self.current_iter}: Gradients available for {len(params_with_grad)}/{total_params} params")
        
        grad_vector = None
        if has_any_gradients and self.gradient_weight > 0:
            # Collect gradients - handle mixed gradient scenarios
            grad_list = []
            for p in self.param_groups[0]["params"]:
                if p.grad is not None:
                    grad_list.append(p.grad.view(-1))
                else:
                    # For params without gradients, use zeros
                    grad_list.append(torch.zeros_like(p.view(-1)))
            
            if grad_list:
                grad_vector = torch.cat(grad_list)
                
                # Apply momentum if enabled
                if self.use_momentum:
                    self.momentum_buffer = (
                        self.momentum * self.momentum_buffer + 
                        (1 - self.momentum) * grad_vector
                    )
                    grad_vector = self.momentum_buffer
                
                # Gradient step for biasing population
                gradient_step = -self.current_step_size * grad_vector
                
                # Bias population towards gradient direction
                pop_size = self.population.shape[0]
                gradient_bias = self.param_vector + gradient_step
                
                for i in range(pop_size):
                    # Weighted combination of BFO exploration and gradient exploitation
                    self.population[i] = (
                        (1 - self.gradient_weight) * self.population[i] +
                        self.gradient_weight * gradient_bias
                    )
        
        # Perform BFO step (works with or without gradients)
        fitness = super().step(closure)
        
        # Adaptive gradient weight adjustment
        if len(self.fitness_history) > 5:
            recent_improvement = self.fitness_history[-1] - self.fitness_history[-5]
            if abs(recent_improvement) < 1e-6:
                # Reduce gradient influence if stagnating
                self.gradient_weight = max(0.1, self.gradient_weight * 0.9)
                # Also increase elimination probability
                self.defaults["elim_prob"] = min(0.5, self.defaults["elim_prob"] * 1.1)
                if self.verbose:
                    logger.info(f"Adapted gradient_weight to {self.gradient_weight}")
        
        return fitness


# Export improved classes
__all__ = ['BFOv2', 'AdaptiveBFOv2', 'HybridBFOv2']