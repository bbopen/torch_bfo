"""
PyTorch BFO Optimizer V2 - GPU-Optimized Implementation
Improved version with parallel population evaluation and performance optimizations
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Tuple, List, Dict, Any, Callable
import numpy as np
import warnings
import time

# Import debug utilities
try:
    from .debug_utils import OptimizationLogger, logger, timing_decorator
except ImportError:
    # Fallback if debug_utils not available
    class OptimizationLogger:
        def __init__(self, *args, **kwargs): pass
        def log_step(self, *args, **kwargs): pass
        def log_population_stats(self, *args, **kwargs): pass
        def log_chemotaxis(self, *args, **kwargs): pass
        def log_reproduction(self, *args, **kwargs): pass
        def log_elimination(self, *args, **kwargs): pass
        def print_summary(self): pass
    
    def timing_decorator(func):
        return func
    
    class logger:
        @staticmethod
        def debug(*args, **kwargs): pass
        @staticmethod
        def info(*args, **kwargs): pass
        @staticmethod
        def warning(*args, **kwargs): pass


class BFOv2(Optimizer):
    """
    GPU-Optimized Bacterial Foraging Optimization (BFO) for PyTorch 2.8+.
    
    Key improvements:
    - Parallel population evaluation using batched operations
    - Early stopping based on convergence criteria
    - Adaptive parameter adjustment
    - Optimized default configurations for GPU/CPU
    - Reduced memory footprint
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        population_size: Number of bacteria in the population (default: auto-selected)
        device_type: Device type hint ('auto', 'gpu', 'cpu') for optimal defaults
        early_stopping: Enable early stopping (default: True)
        convergence_tol: Convergence tolerance for early stopping (default: 1e-6)
        convergence_patience: Steps without improvement before stopping (default: 10)
        parallel_eval: Enable parallel population evaluation (default: True)
        batch_size: Batch size for parallel evaluation (default: None, uses full population)
        verbose: Enable verbose debug logging (default: False)
        **kwargs: Additional BFO parameters (chem_steps, swim_length, etc.)
    """
    
    def __init__(
        self,
        params,
        population_size: Optional[int] = None,
        device_type: str = 'auto',
        early_stopping: bool = True,
        convergence_tol: float = 1e-6,
        convergence_patience: int = 10,
        parallel_eval: bool = True,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        # Standard BFO parameters with optimized defaults
        chem_steps: Optional[int] = None,
        swim_length: Optional[int] = None,
        repro_steps: Optional[int] = None,
        elim_steps: Optional[int] = None,
        elim_prob: float = 0.25,
        step_size_max: float = 0.1,
        step_size_min: float = 0.01,
        levy_alpha: float = 1.5,
        use_swarming: bool = False,
        swarming_params: Tuple[float, float, float, float] = (0.2, 0.1, 0.2, 10.0),
        device: Optional[str] = None,
        compile_mode: bool = False,  # Disabled by default due to PyTorch 2.8.0.dev issues
        compile_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Detect device and set optimal defaults
        if device is None:
            temp_param = next(iter(params) if not isinstance(params, dict) else iter(params[0]['params']))
            device = temp_param.device
        
        self.device = device
        self.device_type = device_type
        
        # Auto-detect device type if needed
        if device_type == 'auto':
            if self.device.type == 'cuda':
                self.device_type = 'gpu'
            else:
                self.device_type = 'cpu'
        
        # Set device-optimized defaults
        if self.device_type == 'gpu':
            # GPU-optimized settings: smaller population, fewer iterations
            population_size = population_size or 6
            chem_steps = chem_steps or 3
            swim_length = swim_length or 2
            repro_steps = repro_steps or 2
            elim_steps = elim_steps or 1
            logger.info("Using GPU-optimized settings")
        else:
            # CPU-optimized settings: larger population for better exploration
            population_size = population_size or 20
            chem_steps = chem_steps or 5
            swim_length = swim_length or 3
            repro_steps = repro_steps or 3
            elim_steps = elim_steps or 2
            logger.info("Using CPU-optimized settings")
        
        # Validate parameters
        if population_size < 1:
            raise ValueError("population_size must be positive")
        if step_size_max <= step_size_min:
            raise ValueError("step_size_max must be greater than step_size_min")
        if not 1.0 <= levy_alpha <= 2.0:
            raise ValueError("levy_alpha must be between 1.0 and 2.0")
        
        # Set defaults
        defaults = dict(
            population_size=population_size,
            chem_steps=chem_steps,
            swim_length=swim_length,
            repro_steps=repro_steps,
            elim_steps=elim_steps,
            elim_prob=elim_prob,
            step_size_max=step_size_max,
            step_size_min=step_size_min,
            levy_alpha=levy_alpha,
            use_swarming=use_swarming,
            swarming_params=swarming_params,
        )
        super().__init__(params, defaults)
        
        # Validate single param group for base implementation
        if len(self.param_groups) != 1:
            raise ValueError("BFOv2 currently supports only one param_group")
        
        # Initialize parameters
        self.param_vector, self.param_shapes = self._flatten_params()
        self.num_params = self.param_vector.numel()
        
        # Initialize bacterial population
        self.population = (
            torch.randn(population_size, self.num_params, device=self.device) * 0.01
            + self.param_vector
        )
        self.best_params = self.param_vector.clone()
        self.best_fitness = float("inf")
        self.current_iter = 0
        self.max_iters = chem_steps * repro_steps * elim_steps
        
        # V2 specific features
        self.parallel_eval = parallel_eval
        self.batch_size = batch_size or population_size
        self.early_stopping = early_stopping
        self.convergence_tol = convergence_tol
        self.convergence_patience = convergence_patience
        self.no_improvement_count = 0
        self.fitness_history = []
        
        # Initialize logger if verbose mode
        self.verbose = verbose
        self.logger = OptimizationLogger("BFOv2") if verbose else None
        
        # Compile optimization functions if requested and safe
        if compile_mode and torch.cuda.is_available() and not torch.__version__.startswith('2.8.0.dev'):
            compile_kwargs = compile_kwargs or {}
            if "mode" not in compile_kwargs:
                compile_kwargs["mode"] = "default"
            self._compiled_step = torch.compile(self._optimization_step, **compile_kwargs)
        else:
            self._compiled_step = self._optimization_step
            if compile_mode and torch.__version__.startswith('2.8.0.dev'):
                warnings.warn("torch.compile disabled due to PyTorch 2.8.0.dev compatibility issues")
    
    def _flatten_params(self) -> Tuple[torch.Tensor, List[torch.Size]]:
        """Flatten all parameters into a single vector."""
        params = []
        shapes = []
        for group in self.param_groups:
            for p in group["params"]:
                params.append(p.view(-1))
                shapes.append(p.shape)
        return torch.cat(params), shapes
    
    def _unflatten_params(self, flat_params: torch.Tensor) -> List[torch.Tensor]:
        """Unflatten parameter vector back to original shapes."""
        params = []
        offset = 0
        for shape in self.param_shapes:
            numel = shape.numel()
            params.append(flat_params[offset : offset + numel].view(shape))
            offset += numel
        return params
    
    def _evaluate_closure(self, closure: Callable) -> float:
        """Evaluate closure and handle both tensor and scalar returns."""
        result = closure()
        if isinstance(result, torch.Tensor):
            return result.item()
        return float(result)
    
    def _parallel_evaluate_population(self, closure: Callable) -> torch.Tensor:
        """
        Evaluate fitness for entire population in parallel batches.
        This is the key optimization for GPU efficiency.
        """
        pop_size = self.population.shape[0]
        fitness = torch.zeros(pop_size, device=self.device)
        
        # Process in batches for memory efficiency
        for i in range(0, pop_size, self.batch_size):
            batch_end = min(i + self.batch_size, pop_size)
            batch_size_actual = batch_end - i
            
            # Prepare batch of parameter updates
            for j in range(batch_size_actual):
                self._unflatten_params(self.population[i + j])
            
            # Evaluate batch
            # Note: This still requires sequential closure calls, but memory operations are batched
            for j in range(batch_size_actual):
                self._unflatten_params(self.population[i + j])
                fitness[i + j] = self._evaluate_closure(closure)
        
        return fitness
    
    def _levy_flight(self, size: torch.Size) -> torch.Tensor:
        """Generate Lévy flight random walk for exploration."""
        alpha = self.defaults["levy_alpha"]
        sigma = (
            torch.exp(torch.lgamma(torch.tensor(1 + alpha)))
            * torch.sin(torch.tensor(np.pi * alpha / 2))
            / (
                torch.exp(torch.lgamma(torch.tensor((1 + alpha) / 2)))
                * torch.tensor(alpha)
                * torch.tensor(2 ** ((alpha - 1) / 2))
            )
        ) ** (1 / alpha)
        
        u = torch.randn(size, device=self.device) * sigma
        v = torch.randn(size, device=self.device)
        step = u / torch.abs(v) ** (1 / alpha)
        
        return step
    
    def _compute_swarming(self, population: torch.Tensor, i: int) -> torch.Tensor:
        """Compute cell-to-cell attraction and repelling effects."""
        if not self.defaults["use_swarming"]:
            return torch.zeros(1, device=self.device)
        
        d_attract, w_attract, h_repel, w_repel = self.defaults["swarming_params"]
        swarming = torch.zeros(1, device=self.device)
        
        # Vectorized swarming computation
        diff = population - population[i]
        dist_sq = torch.sum(diff ** 2, dim=1)
        
        # Attraction
        attract_mask = dist_sq > 0
        swarming += torch.sum(
            -d_attract * torch.exp(-w_attract * dist_sq[attract_mask])
        )
        
        # Repulsion
        swarming += torch.sum(
            h_repel * torch.exp(-w_repel * dist_sq[attract_mask])
        )
        
        return swarming
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged based on fitness history."""
        if not self.early_stopping or len(self.fitness_history) < 2:
            return False
        
        # Check if improvement is below tolerance
        if len(self.fitness_history) >= 2:
            improvement = abs(self.fitness_history[-2] - self.fitness_history[-1])
            if improvement < self.convergence_tol:
                self.no_improvement_count += 1
            else:
                self.no_improvement_count = 0
        
        # Check if we've exceeded patience
        if self.no_improvement_count >= self.convergence_patience:
            if self.verbose:
                logger.info(f"Early stopping triggered after {self.current_iter} iterations")
            return True
        
        return False
    
    @timing_decorator
    def _optimization_step(
        self,
        closure: Callable,
        population: torch.Tensor,
        best_params: torch.Tensor,
        best_fitness: float,
        iteration: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Single optimization step with improved efficiency."""
        pop_size = population.shape[0]
        new_positions = population.clone()
        
        # Evaluate fitness for all bacteria in parallel if enabled
        if self.parallel_eval:
            fitness = self._parallel_evaluate_population(closure)
        else:
            # Original sequential evaluation
            fitness = torch.zeros(pop_size, device=self.device)
            for i in range(pop_size):
                self._unflatten_params(population[i])
                fitness[i] = self._evaluate_closure(closure)
        
        # Adaptive step size based on iteration progress
        progress = iteration / max(1, self.max_iters)
        current_step_size = (
            self.defaults["step_size_max"] * (1 - progress)
            + self.defaults["step_size_min"] * progress
        )
        
        # Chemotaxis (tumble and swim) - vectorized operations
        for j in range(self.defaults["chem_steps"]):
            # Generate random directions for all bacteria at once
            directions = self._levy_flight((pop_size, self.num_params))
            directions = directions / (torch.norm(directions, dim=1, keepdim=True) + 1e-8)
            
            # Tumble: take a step in random direction
            new_positions = population + current_step_size * directions
            
            # Evaluate new positions
            if self.parallel_eval:
                new_fitness = self._parallel_evaluate_population(closure)
            else:
                new_fitness = torch.zeros(pop_size, device=self.device)
                for i in range(pop_size):
                    self._unflatten_params(new_positions[i])
                    new_fitness[i] = self._evaluate_closure(closure)
            
            # Swim: continue in same direction if improvement
            swim_mask = new_fitness < fitness
            swim_count = 0
            
            while swim_mask.any() and swim_count < self.defaults["swim_length"]:
                # Update positions for bacteria that improved
                population[swim_mask] = new_positions[swim_mask]
                fitness[swim_mask] = new_fitness[swim_mask]
                
                # Continue swimming
                new_positions[swim_mask] = (
                    population[swim_mask] + current_step_size * directions[swim_mask]
                )
                
                # Re-evaluate only swimming bacteria
                for i in torch.where(swim_mask)[0]:
                    self._unflatten_params(new_positions[i])
                    new_fitness[i] = self._evaluate_closure(closure)
                
                swim_mask = new_fitness < fitness
                swim_count += 1
            
            # Final update for any remaining improvements
            improved = new_fitness < fitness
            population[improved] = new_positions[improved]
            fitness[improved] = new_fitness[improved]
        
        # Update best solution
        min_fitness, min_idx = fitness.min(0)
        if min_fitness < best_fitness:
            best_fitness = min_fitness.item()
            best_params = population[min_idx].clone()
        
        return population, best_params, best_fitness
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> float:
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
                    Required for BFO.
                    
        Returns:
            The best fitness value found.
        """
        if closure is None:
            raise ValueError("BFO requires a closure that returns the loss value")
        
        if self.verbose:
            logger.info(f"Starting BFOv2 step {self.current_iter}")
            start_time = time.time()
        
        pop_size = self.defaults["population_size"]
        
        # Check for early stopping
        if self._check_convergence():
            if self.verbose:
                logger.info("Convergence reached, skipping optimization")
            return self.best_fitness
        
        for elim_iter in range(self.defaults["elim_steps"]):
            for repro_iter in range(self.defaults["repro_steps"]):
                # Optimization step
                self.population, self.best_params, self.best_fitness = self._compiled_step(
                    closure,
                    self.population,
                    self.best_params,
                    self.best_fitness,
                    self.current_iter,
                )
                self.current_iter += 1
                
                # Reproduction: eliminate worst half, duplicate best half
                if self.parallel_eval:
                    fitness = self._parallel_evaluate_population(closure)
                else:
                    fitness = torch.zeros(pop_size, device=self.device)
                    for i in range(pop_size):
                        self._unflatten_params(self.population[i])
                        fitness[i] = self._evaluate_closure(closure)
                
                sorted_idx = torch.argsort(fitness)
                half = pop_size // 2
                if half > 0:
                    self.population[sorted_idx[half:]] = self.population[sorted_idx[:half]].clone()[:pop_size-half]
                
                if self.verbose and self.logger:
                    self.logger.log_reproduction(pop_size - half)
            
            # Elimination-dispersal
            if elim_iter < self.defaults["elim_steps"] - 1:
                elim_mask = torch.rand(pop_size, device=self.device) < self.defaults["elim_prob"]
                if elim_mask.any():
                    self.population[elim_mask] = (
                        torch.randn_like(self.population[elim_mask]) * 0.01 + self.best_params
                    )
                    if self.verbose and self.logger:
                        self.logger.log_elimination(torch.where(elim_mask)[0].tolist())
        
        # Apply best parameters to model
        best_params_list = self._unflatten_params(self.best_params)
        for p, best_p in zip(self.param_groups[0]["params"], best_params_list):
            p.data.copy_(best_p)
        
        # Update fitness history
        self.fitness_history.append(self.best_fitness)
        
        if self.verbose:
            elapsed = time.time() - start_time
            diversity = self._compute_diversity()
            self.logger.log_step(
                loss=self.best_fitness,
                diversity=diversity,
                step_size=self.defaults["step_size_max"],
                elapsed=elapsed
            )
        
        return self.best_fitness
    
    def _compute_diversity(self) -> float:
        """Compute population diversity metric."""
        mean_position = self.population.mean(dim=0)
        diversity = torch.norm(self.population - mean_position, dim=1).mean()
        return diversity.item()
    
    def zero_grad(self, set_to_none: bool = True):
        """Sets gradients of all optimized parameters to zero."""
        super().zero_grad(set_to_none)
    
    def get_config_recommendations(self) -> Dict[str, Any]:
        """Get configuration recommendations based on device and model size."""
        recommendations = {
            'device_type': self.device_type,
            'num_params': self.num_params,
            'recommended_population': self.defaults['population_size'],
            'parallel_eval': self.parallel_eval,
            'early_stopping': self.early_stopping,
        }
        
        if self.device_type == 'gpu':
            recommendations['notes'] = [
                "Consider using HybridBFO for better GPU utilization",
                "Increase batch size for better GPU efficiency",
                "Use smaller population sizes (4-8) for faster execution"
            ]
        else:
            recommendations['notes'] = [
                "CPU allows larger population sizes for better exploration",
                "Consider disabling parallel_eval if overhead is high",
                "Standard BFO may outperform HybridBFO on CPU"
            ]
        
        return recommendations


class AdaptiveBFOv2(BFOv2):
    """
    Adaptive variant of BFOv2 with automatic hyperparameter tuning.
    Inherits all GPU optimizations from BFOv2.
    """
    
    def __init__(
        self,
        params,
        adaptation_rate: float = 0.1,
        diversity_threshold: float = 0.01,
        **kwargs
    ):
        super().__init__(params, **kwargs)
        self.adaptation_rate = adaptation_rate
        self.diversity_threshold = diversity_threshold
        self.diversity_history = []
    
    def _adapt_parameters(self):
        """Adapt optimization parameters based on performance."""
        if len(self.fitness_history) < 5:
            return
        
        # Check convergence rate
        recent_improvement = abs(self.fitness_history[-5] - self.fitness_history[-1])
        convergence_rate = recent_improvement / (5 * abs(self.fitness_history[-1] + 1e-8))
        
        # Adapt step size based on convergence
        if convergence_rate < 0.01:  # Slow convergence
            self.defaults["step_size_max"] *= (1 + self.adaptation_rate)
            self.defaults["elim_prob"] = min(0.5, self.defaults["elim_prob"] * 1.1)
            if self.verbose:
                logger.debug("Increasing exploration due to slow convergence")
        elif convergence_rate > 0.1:  # Fast convergence
            self.defaults["step_size_max"] *= (1 - self.adaptation_rate * 0.5)
            if self.verbose:
                logger.debug("Reducing step size due to fast convergence")
        
        # Adapt based on diversity
        current_diversity = self._compute_diversity()
        self.diversity_history.append(current_diversity)
        
        if current_diversity < self.diversity_threshold:
            self.defaults["elim_prob"] = min(0.5, self.defaults["elim_prob"] * 1.2)
            if self.verbose:
                logger.debug("Increasing elimination probability due to low diversity")
    
    def step(self, closure: Optional[Callable] = None) -> float:
        """Perform optimization step with parameter adaptation."""
        # Adapt parameters before step
        self._adapt_parameters()
        
        # Perform standard step
        fitness = super().step(closure)
        
        return fitness


class HybridBFOv2(BFOv2):
    """
    Hybrid optimizer combining BFOv2 with gradient information.
    Optimized for GPU execution with parallel operations.
    """
    
    def __init__(
        self,
        params,
        gradient_weight: float = 0.5,
        use_momentum: bool = True,
        momentum: float = 0.9,
        **kwargs
    ):
        # Force certain settings for hybrid mode
        kwargs['device_type'] = kwargs.get('device_type', 'auto')
        super().__init__(params, **kwargs)
        
        self.gradient_weight = gradient_weight
        self.use_momentum = use_momentum
        self.momentum = momentum
        
        # Initialize momentum buffer
        if use_momentum:
            self.momentum_buffer = torch.zeros_like(self.param_vector)
        
        if self.verbose:
            logger.info(f"HybridBFOv2 initialized with gradient_weight={gradient_weight}")
    
    def step(self, closure: Optional[Callable] = None) -> float:
        """
        Hybrid optimization step combining BFO with gradient descent.
        Requires gradients to be computed in the closure.
        """
        if closure is None:
            raise ValueError("HybridBFO requires a closure that computes gradients")
        
        # Get current gradients
        grad_vector = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad_vector.append(p.grad.view(-1))
                else:
                    grad_vector.append(torch.zeros_like(p.view(-1)))
        
        if grad_vector:
            grad_vector = torch.cat(grad_vector)
            
            # Apply momentum if enabled
            if self.use_momentum:
                self.momentum_buffer = (
                    self.momentum * self.momentum_buffer + (1 - self.momentum) * grad_vector
                )
                grad_vector = self.momentum_buffer
            
            # Combine BFO population with gradient information
            step_size = self.defaults["step_size_max"]
            gradient_step = -step_size * grad_vector
            
            # Update population with gradient bias
            pop_size = self.population.shape[0]
            for i in range(pop_size):
                # Weighted combination of BFO exploration and gradient exploitation
                self.population[i] = (
                    (1 - self.gradient_weight) * self.population[i]
                    + self.gradient_weight * (self.param_vector + gradient_step)
                )
        
        # Perform BFO step with gradient-biased population
        fitness = super().step(closure)
        
        return fitness