"""
PyTorch BFO Optimizer Implementation for PyTorch 2.8+
Supports torch.compile optimization and modern PyTorch patterns
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Tuple, List, Dict, Any, Callable
import numpy as np
import warnings


class BFO(Optimizer):
    """
    Bacterial Foraging Optimization (BFO) for PyTorch 2.8+.
    
    This optimizer implements an adaptive variant of BFO with:
    - Lévy flights for exploration
    - Dynamic step size adaptation
    - Optional swarming behavior
    - torch.compile compatibility for performance
    - GPU acceleration with vectorized operations
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        population_size: Number of bacteria in the population (default: 50)
        chem_steps: Chemotaxis steps per reproduction cycle (default: 10)
        swim_length: Maximum swim steps in one direction (default: 4)
        repro_steps: Reproduction steps per elimination cycle (default: 4)
        elim_steps: Elimination-dispersal steps (default: 2)
        elim_prob: Base elimination probability (default: 0.25)
        step_size_max: Maximum step size (default: 0.1)
        step_size_min: Minimum step size (default: 0.01)
        levy_alpha: Lévy flight parameter (default: 1.5)
        use_swarming: Enable bacterial swarming behavior (default: False)
        swarming_params: Tuple of (d_attract, w_attract, h_repel, w_repel)
        device: Device to run on ('cpu', 'cuda', or specific device)
        compile_mode: Whether to use torch.compile optimization (default: True)
        
    Example:
        >>> model = nn.Linear(10, 1).cuda()
        >>> optimizer = BFO(model.parameters(), population_size=50, compile_mode=True)
        >>> 
        >>> def closure():
        >>>     optimizer.zero_grad()
        >>>     output = model(data)
        >>>     loss = criterion(output, target)
        >>>     return loss.item()
        >>> 
        >>> optimizer.step(closure)
    """
    
    def __init__(
        self,
        params,
        population_size: int = 50,
        chem_steps: int = 10,
        swim_length: int = 4,
        repro_steps: int = 4,
        elim_steps: int = 2,
        elim_prob: float = 0.25,
        step_size_max: float = 0.1,
        step_size_min: float = 0.01,
        levy_alpha: float = 1.5,
        use_swarming: bool = False,
        swarming_params: Tuple[float, float, float, float] = (0.2, 0.1, 0.2, 10.0),
        device: Optional[str] = None,
        compile_mode: bool = True,
    ):
        if population_size < 1:
            raise ValueError("population_size must be positive")
        if step_size_max <= step_size_min:
            raise ValueError("step_size_max must be greater than step_size_min")
        if not 1.0 <= levy_alpha <= 2.0:
            raise ValueError("levy_alpha must be between 1.0 and 2.0")
            
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
            raise ValueError("BFO currently supports only one param_group")
            
        self.param_vector, self.param_shapes = self._flatten_params()
        self.num_params = self.param_vector.numel()
        self.device = device or self.param_vector.device
        
        # Initialize bacterial population
        self.population = (
            torch.randn(population_size, self.num_params, device=self.device) * 0.01
            + self.param_vector
        )
        self.best_params = self.param_vector.clone()
        self.best_fitness = float("inf")
        self.current_iter = 0
        self.max_iters = chem_steps * repro_steps * elim_steps
        
        # Compile optimization functions if requested
        if compile_mode and torch.cuda.is_available():
            self._compiled_step = torch.compile(self._optimization_step, mode="reduce-overhead")
        else:
            self._compiled_step = self._optimization_step
            
    def _flatten_params(self) -> Tuple[torch.Tensor, List[torch.Size]]:
        """Flatten all parameters into a single vector."""
        param_list = []
        param_shapes = []
        
        for p in self.param_groups[0]["params"]:
            param_list.append(p.data.view(-1))
            param_shapes.append(p.shape)
            
        param_vector = torch.cat(param_list)
        return param_vector.to(self.device), param_shapes
        
    def _unflatten_params(self, vector: torch.Tensor) -> List[torch.Tensor]:
        """Unflatten parameter vector back to original shapes."""
        params = []
        idx = 0
        
        for shape in self.param_shapes:
            numel = torch.prod(torch.tensor(shape)).item()
            params.append(vector[idx : idx + numel].view(shape))
            idx += numel
            
        return params
        
    def _levy_flight(self, size: Tuple[int, ...]) -> torch.Tensor:
        """Generate Lévy flight steps for exploration."""
        alpha = self.defaults["levy_alpha"]
        
        # Lévy distribution parameters
        sigma_u = (
            torch.exp(torch.lgamma(torch.tensor(1 + alpha)))
            * torch.sin(torch.tensor(np.pi * alpha / 2))
            / (
                torch.exp(torch.lgamma(torch.tensor((1 + alpha) / 2)))
                * alpha
                * torch.pow(torch.tensor(2.0), (alpha - 1) / 2)
            )
        ).pow(1 / alpha)
        
        u = torch.randn(size, device=self.device) * sigma_u
        v = torch.randn(size, device=self.device)
        
        step = 0.01 * u / torch.abs(v).pow(1 / alpha)
        return step
        
    def _compute_swarming(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute swarming attraction-repulsion forces."""
        if not self.defaults["use_swarming"]:
            return torch.zeros(positions.shape[0], device=self.device)
            
        d_attract, w_attract, h_repel, w_repel = self.defaults["swarming_params"]
        
        # Vectorized distance computation
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        dist_sq = (diff ** 2).sum(-1)
        
        # Attraction and repulsion terms
        attract = -d_attract * torch.exp(-w_attract * dist_sq).sum(1)
        repel = h_repel * torch.exp(-w_repel * dist_sq).sum(1)
        
        return attract + repel
        
    @torch.no_grad()
    def _optimization_step(
        self, 
        closure: Callable,
        population: torch.Tensor,
        best_params: torch.Tensor,
        best_fitness: float,
        iteration: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Core optimization step - compatible with torch.compile."""
        pop_size = self.defaults["population_size"]
        
        # Adaptive step size based on iteration progress
        t = iteration / self.max_iters
        step_size = (
            self.defaults["step_size_max"] * (1 - t) + 
            self.defaults["step_size_min"] * t
        )
        
        # Generate Lévy flight steps
        levy_steps = self._levy_flight((pop_size, self.num_params))
        
        # Evaluate fitness for all bacteria
        fitness = torch.empty(pop_size, device=self.device)
        for i in range(pop_size):
            params_list = self._unflatten_params(population[i])
            for p, new_p in zip(self.param_groups[0]["params"], params_list):
                p.data.copy_(new_p)
            fitness[i] = closure()
            
        # Add swarming contribution
        swarming_term = self._compute_swarming(population)
        fitness += swarming_term
        
        # Tumble: Random direction exploration
        directions = torch.randn_like(population)
        directions = directions / directions.norm(dim=1, keepdim=True).clamp(min=1e-8)
        
        # Update positions with Lévy flights
        new_positions = population + step_size * levy_steps * directions
        
        # Swim: Continue in beneficial directions
        new_fitness = torch.empty_like(fitness)
        for i in range(pop_size):
            params_list = self._unflatten_params(new_positions[i])
            for p, new_p in zip(self.param_groups[0]["params"], params_list):
                p.data.copy_(new_p)
            new_fitness[i] = closure() + swarming_term[i]
            
        # Swimming behavior
        improved = new_fitness < fitness
        swim_count = torch.zeros(pop_size, dtype=torch.int64, device=self.device)
        max_swim = self.defaults["swim_length"]
        
        while improved.any() and (swim_count < max_swim).any():
            mask = improved & (swim_count < max_swim)
            new_positions[mask] += step_size * directions[mask]
            fitness[mask] = new_fitness[mask]
            swim_count[mask] += 1
            
            # Re-evaluate for swimming bacteria
            for i in torch.where(mask)[0]:
                params_list = self._unflatten_params(new_positions[i])
                for p, new_p in zip(self.param_groups[0]["params"], params_list):
                    p.data.copy_(new_p)
                new_fitness[i] = closure() + swarming_term[i]
                
            improved = new_fitness < fitness
            
        # Update best solution
        min_fitness, min_idx = fitness.min(0)
        if min_fitness < best_fitness:
            best_fitness = min_fitness.item()
            best_params = new_positions[min_idx].clone()
            
        return new_positions, best_params, best_fitness
        
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
            
        pop_size = self.defaults["population_size"]
        
        for elim_iter in range(self.defaults["elim_steps"]):
            for repro_iter in range(self.defaults["repro_steps"]):
                for chem_iter in range(self.defaults["chem_steps"]):
                    # Run optimization step (potentially compiled)
                    self.population, self.best_params, self.best_fitness = (
                        self._compiled_step(
                            closure,
                            self.population,
                            self.best_params,
                            self.best_fitness,
                            self.current_iter,
                        )
                    )
                    self.current_iter += 1
                    
                # Reproduction: Clone better half of population
                fitness_values = torch.empty(pop_size, device=self.device)
                for i in range(pop_size):
                    params_list = self._unflatten_params(self.population[i])
                    for p, new_p in zip(self.param_groups[0]["params"], params_list):
                        p.data.copy_(new_p)
                    fitness_values[i] = closure()
                    
                sorted_idx = fitness_values.argsort()
                half = pop_size // 2
                self.population[sorted_idx[half:]] = self.population[sorted_idx[:half]].clone()
                
            # Elimination-Dispersal: Replace worst bacteria probabilistically
            ranks = fitness_values.argsort().argsort().float() / pop_size
            elim_prob_adaptive = self.defaults["elim_prob"] * (1 - self.current_iter / self.max_iters)
            elim_mask = torch.rand_like(ranks) < (elim_prob_adaptive * ranks)
            
            if elim_mask.any():
                self.population[elim_mask] = (
                    torch.randn_like(self.population[elim_mask]) * 0.01 + self.best_params
                )
                
        # Apply best parameters to model
        best_params_list = self._unflatten_params(self.best_params)
        for p, best_p in zip(self.param_groups[0]["params"], best_params_list):
            p.data.copy_(best_p)
            
        return self.best_fitness
        
    def zero_grad(self, set_to_none: bool = True):
        """Sets gradients of all optimized parameters to zero."""
        super().zero_grad(set_to_none)


class AdaptiveBFO(BFO):
    """
    Adaptive variant of BFO with automatic hyperparameter tuning.
    
    This variant automatically adjusts:
    - Step sizes based on convergence rate
    - Population diversity measures
    - Elimination probability based on stagnation
    
    Additional Args:
        adaptation_rate: Rate of hyperparameter adaptation (default: 0.1)
        diversity_threshold: Minimum population diversity (default: 0.01)
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
        self.fitness_history = []
        self.diversity_history = []
        
    def _compute_diversity(self) -> float:
        """Compute population diversity metric."""
        mean_position = self.population.mean(dim=0)
        diversity = torch.norm(self.population - mean_position, dim=1).mean()
        return diversity.item()
        
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> float:
        """Enhanced step with adaptive parameter tuning."""
        fitness = super().step(closure)
        
        # Track fitness history
        self.fitness_history.append(fitness)
        if len(self.fitness_history) > 10:
            self.fitness_history.pop(0)
            
        # Compute and track diversity
        diversity = self._compute_diversity()
        self.diversity_history.append(diversity)
        if len(self.diversity_history) > 10:
            self.diversity_history.pop(0)
            
        # Adaptive parameter adjustment
        if len(self.fitness_history) >= 5:
            # Check for stagnation
            recent_improvement = abs(self.fitness_history[-1] - self.fitness_history[-5])
            if recent_improvement < 1e-6:
                # Increase exploration
                self.defaults["step_size_max"] *= (1 + self.adaptation_rate)
                self.defaults["elim_prob"] = min(0.5, self.defaults["elim_prob"] * 1.1)
                
            # Check diversity
            if diversity < self.diversity_threshold:
                # Increase diversity through elimination
                self.defaults["elim_prob"] = min(0.5, self.defaults["elim_prob"] * 1.2)
                
        return fitness


class HybridBFO(BFO):
    """
    Hybrid BFO that can leverage gradients when available.
    
    Combines bacterial foraging with gradient information for faster convergence
    in differentiable optimization landscapes.
    
    Additional Args:
        gradient_weight: Weight for gradient contribution (default: 0.5)
        use_momentum: Whether to use momentum with gradients (default: True)
        momentum: Momentum coefficient (default: 0.9)
    """
    
    def __init__(
        self,
        params,
        gradient_weight: float = 0.5,
        use_momentum: bool = True,
        momentum: float = 0.9,
        **kwargs
    ):
        super().__init__(params, **kwargs)
        self.gradient_weight = gradient_weight
        self.use_momentum = use_momentum
        self.momentum = momentum
        
        # Initialize momentum buffers
        if self.use_momentum:
            self.momentum_buffers = [
                torch.zeros_like(p.data) for p in self.param_groups[0]["params"]
            ]
            
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> float:
        """
        Hybrid optimization step combining BFO with gradient information.
        
        Args:
            closure: Closure that computes loss and optionally gradients
            
        Returns:
            Best fitness value
        """
        # First perform BFO step
        fitness = super().step(closure)
        
        # Check if gradients are available
        has_gradients = all(
            p.grad is not None for p in self.param_groups[0]["params"]
        )
        
        if has_gradients and self.gradient_weight > 0:
            # Apply gradient-based update
            for i, p in enumerate(self.param_groups[0]["params"]):
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                
                if self.use_momentum:
                    buf = self.momentum_buffers[i]
                    buf.mul_(self.momentum).add_(grad, alpha=1 - self.momentum)
                    grad = buf
                    
                # Blend with BFO update
                step_size = self.defaults["step_size_min"]
                p.data.add_(grad, alpha=-step_size * self.gradient_weight)
                
        return fitness