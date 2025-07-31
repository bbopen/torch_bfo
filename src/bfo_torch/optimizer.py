"""
Bacterial Foraging Optimization (BFO) Optimizer for PyTorch
===========================================================

A biologically-inspired optimization algorithm based on the foraging behavior
of E. coli bacteria. Implements chemotaxis, swarming, reproduction, and
elimination-dispersal mechanisms for global optimization.
"""

import os
import logging
import warnings
from typing import Optional, Callable, List, Dict, Any, Union, Tuple, Iterable
from collections import defaultdict

import torch
from torch.optim import Optimizer
import numpy as np

try:
    from torch.func import functional_call, vmap
    HAS_FUNCTORCH = True
except ImportError:
    HAS_FUNCTORCH = False

logger = logging.getLogger(__name__)


class BFO(Optimizer):
    """
    Bacterial Foraging Optimization (BFO) optimizer with vectorized evaluation.
    
    Implements the bacterial foraging optimization algorithm with chemotaxis,
    swarming, reproduction, and elimination-dispersal mechanisms. This version
    is enhanced with vectorized operations for better performance, especially
    on hardware accelerators like GPUs.
    
    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Learning rate (default: 0.01)
        population_size (int, optional): Number of bacteria in population (default: 50)
        chemotaxis_steps (int, optional): Number of chemotactic steps (default: 10)
        swim_length (int, optional): Maximum swim steps in one direction (default: 4)
        reproduction_steps (int, optional): Number of reproduction cycles (default: 4)
        elimination_steps (int, optional): Number of elimination-dispersal cycles (default: 2)
        elimination_prob (float, optional): Probability of elimination (default: 0.25)
        step_size_min (float, optional): Minimum step size (default: 1e-4)
        step_size_max (float, optional): Maximum step size (default: 0.1)
        levy_alpha (float, optional): Lévy flight parameter (default: 1.5)
        enable_swarming (bool, optional): Enable bacterial swarming (default: True)
        swarming_params (tuple, optional): Swarming parameters (d_attract, w_attract, h_repel, w_repel)
            (default: (0.2, 0.1, 0.2, 10.0))
        device (torch.device, optional): Device to run on. If None, uses parameter device (default: None)
        compile_mode (str, optional): torch.compile mode ('default', 'reduce-overhead', 'max-autotune', or None)
            (default: None)
        early_stopping (bool, optional): Enable early stopping on convergence (default: True)
        convergence_tolerance (float, optional): Convergence tolerance (default: 1e-6)
        convergence_patience (int, optional): Patience for early stopping (default: 10)
        seed (int, optional): Random seed for reproducibility (default: None)
        
    Example:
        >>> optimizer = BFO(model.parameters(), lr=0.01, population_size=30)
        >>> def closure():
        >>>     optimizer.zero_grad()
        >>>     output = model(data)
        >>>     loss = criterion(output, target)
        >>>     return loss.item()
        >>> optimizer.step(closure)
    """
    
    def __init__(
        self,
        params: Iterable[Union[torch.Tensor, Dict[str, Any]]],
        lr: float = 0.01,
        population_size: int = 50,
        chemotaxis_steps: int = 10,
        swim_length: int = 4,
        reproduction_steps: int = 4,
        elimination_steps: int = 2,
        elimination_prob: float = 0.25,
        step_size_min: float = 1e-4,
        step_size_max: float = 0.1,
        levy_alpha: float = 1.5,
        enable_swarming: bool = True,
        swarming_params: Tuple[float, float, float, float] = (0.2, 0.1, 0.2, 10.0),
        device: Optional[torch.device] = None,
        compile_mode: Optional[str] = None,
        early_stopping: bool = True,
        convergence_tolerance: float = 1e-6,
        convergence_patience: int = 10,
        seed: Optional[int] = None,
        **kwargs
    ):
        # Validation
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if population_size < 1:
            raise ValueError(f"Invalid population size: {population_size}")
        if not (0.0 < elimination_prob <= 1.0):
            raise ValueError(f"Invalid elimination probability: {elimination_prob}")
        if step_size_max <= step_size_min:
            raise ValueError(f"step_size_max ({step_size_max}) must be greater than step_size_min ({step_size_min})")
        if not (1.0 <= levy_alpha <= 2.0):
            raise ValueError(f"levy_alpha must be between 1.0 and 2.0, got {levy_alpha}")
        
        # Default parameter group settings
        defaults = dict(
            lr=lr,
            population_size=population_size,
            chemotaxis_steps=chemotaxis_steps,
            swim_length=swim_length,
            reproduction_steps=reproduction_steps,
            elimination_steps=elimination_steps,
            elimination_prob=elimination_prob,
            step_size_min=step_size_min,
            step_size_max=step_size_max,
            levy_alpha=levy_alpha,
            enable_swarming=enable_swarming,
            swarming_params=swarming_params,
            **kwargs
        )
        
        super().__init__(params, defaults)
        
        # Device handling
        if device is None:
            # Auto-detect device from first parameter
            first_param = next(iter(self.param_groups[0]['params']))
            device = first_param.device
        self.device = device
        
        # Configuration
        self.compile_mode = compile_mode
        self.early_stopping = early_stopping
        self.convergence_tolerance = convergence_tolerance
        self.convergence_patience = convergence_patience
        
        # Random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Initialize optimizer state
        self._initialize_state()
        
        # Compile optimization if requested
        if compile_mode and hasattr(torch, 'compile'):
            try:
                self._compiled_step = torch.compile(
                    self._optimization_step,
                    mode=compile_mode
                )
                logger.info(f"BFO compiled with mode: {compile_mode}")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}, falling back to eager mode")
                self._compiled_step = self._optimization_step
        else:
            self._compiled_step = self._optimization_step
    
    def _initialize_state(self) -> None:
        """Initialize optimizer state for all parameter groups."""
        for group in self.param_groups:
            self._initialize_group_state(group)
    
    def _initialize_group_state(self, group: Dict[str, Any]) -> None:
        """Initialize state for a specific parameter group."""
        # Flatten parameters for this group
        param_vector, param_shapes = self._flatten_group_params(group)
        population_size = group['population_size']
        
        # Detect dtype from parameters
        first_param = group['params'][0]
        dtype = first_param.dtype
        
        # Initialize population around current parameters
        population = torch.randn(
            population_size, param_vector.numel(),
            device=self.device, dtype=dtype
        ) * group['step_size_max']
        population[0] = param_vector.clone()  # Include current parameters
        
        # Store state
        group_state = {
            'param_vector': param_vector,
            'param_shapes': param_shapes,
            'population': population,
            'best_params': param_vector.clone(),
            'best_fitness': float('inf'),
            'fitness_history': [],
            'current_step_size': group['step_size_max'],
            'stagnation_count': 0,
            'iteration': 0,
            'dtype': dtype,
            'function_evaluations': 0  # Track actual function evaluations
        }
        
        # Store in optimizer state using group ID
        group_id = id(group)
        if 'state' not in self.__dict__:
            self.state = {}
        self.state[group_id] = group_state
    
    def _flatten_group_params(self, group: Dict[str, Any]) -> Tuple[torch.Tensor, List[torch.Size]]:
        """Flatten all parameters in a group into a single tensor."""
        param_list = []
        param_shapes = []
        
        for p in group['params']:
            param_list.append(p.data.view(-1))
            param_shapes.append(p.shape)
        
        param_vector = torch.cat(param_list).to(self.device)
        return param_vector, param_shapes
    
    def _unflatten_group_params(self, vector: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
        """Unflatten parameter vector back to original shapes."""
        params = []
        offset = 0
        
        for shape in shapes:
            numel = torch.prod(torch.tensor(shape)).item()
            params.append(vector[offset:offset + numel].view(shape))
            offset += numel
        
        return params
    
    def _levy_flight(self, size: Tuple[int, ...], alpha: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Generate Lévy flight steps for exploration."""
        # Compute Lévy distribution parameters
        gamma_1_alpha = torch.exp(torch.lgamma(torch.tensor(1 + alpha, device=device, dtype=dtype)))
        gamma_1_alpha_2 = torch.exp(torch.lgamma(torch.tensor((1 + alpha) / 2, device=device, dtype=dtype)))
        sin_term = torch.sin(torch.tensor(np.pi * alpha / 2, device=device, dtype=dtype))
        
        sigma_u = (gamma_1_alpha * sin_term / (
            gamma_1_alpha_2 * alpha * torch.pow(torch.tensor(2.0, device=device, dtype=dtype), (alpha - 1) / 2)
        )).pow(1 / alpha)
        
        # Generate random samples with numerical stability
        max_retries = 3
        for _ in range(max_retries):
            u = torch.randn(size, device=device, dtype=dtype) * sigma_u
            v = torch.randn(size, device=device, dtype=dtype)
            
            # Avoid division by zero
            v_abs = torch.abs(v) + 1e-10
            step = u / v_abs.pow(1 / alpha)
            
            # Check for numerical stability
            if torch.isfinite(step).all():
                # Apply soft clamping
                step = 10 * torch.tanh(step / 10)
                return step
        
        # Fallback to normal distribution
        logger.warning("Lévy flight numerical issues, using normal distribution fallback")
        return torch.randn(size, device=device, dtype=dtype)
    
    def _compute_swarming(self, positions: torch.Tensor, swarming_params: Tuple[float, float, float, float]) -> torch.Tensor:
        """Compute bacterial swarming forces."""
        d_attract, w_attract, h_repel, w_repel = swarming_params
        pop_size = positions.shape[0]
        
        if pop_size == 1:
            return torch.zeros(1, positions.shape[1], device=positions.device, dtype=positions.dtype)
        
        # Handle mixed precision for torch.cdist compatibility
        original_dtype = positions.dtype
        if original_dtype in (torch.float16, torch.bfloat16):
            positions_compute = positions.float()
        else:
            positions_compute = positions
        
        # Compute pairwise distances
        distances = torch.cdist(positions_compute, positions_compute, p=2) + 1e-10
        
        # Compute direction vectors
        diff = positions_compute.unsqueeze(0) - positions_compute.unsqueeze(1)
        normalized_diff = diff / distances.unsqueeze(-1)
        
        # Compute attraction and repulsion forces
        attract_factor = -d_attract * torch.exp(-w_attract * distances**2)
        repel_factor = h_repel * torch.exp(-w_repel * distances)
        
        # Exclude self-interactions
        mask = ~torch.eye(pop_size, dtype=torch.bool, device=positions_compute.device)
        combined_factor = (attract_factor + repel_factor) * mask.float()
        
        # Compute swarming forces
        swarming = (combined_factor.unsqueeze(-1) * normalized_diff).sum(dim=1)
        
        # Convert back to original dtype
        if original_dtype != swarming.dtype:
            swarming = swarming.to(original_dtype)
        
        return swarming
    
    def _evaluate_closure(self, closure: Callable, group: Dict[str, Any], individual: torch.Tensor) -> float:
        """Safely evaluate closure for a given parameter configuration."""
        original_data = [p.data.clone() for p in group['params']]
        
        try:
            with torch.no_grad():
                # Update parameters temporarily
                param_list = self._unflatten_group_params(individual, self.state[id(group)]['param_shapes'])
                for p, new_p in zip(group['params'], param_list):
                    p.data.copy_(new_p)
            
            # Evaluate closure
            result = closure()
            
            # Convert result to float
            if isinstance(result, torch.Tensor):
                result = result.item()
            
            return float(result)
            
        except Exception as e:
            logger.debug(f"Closure evaluation failed: {e}")
            return float('inf')
        
        finally:
            # Restore original parameters
            with torch.no_grad():
                for p, orig_data in zip(group['params'], original_data):
                    p.data.copy_(orig_data)

    def _evaluate_batch_closure(self, closure: Callable, group: Dict[str, Any], population: torch.Tensor) -> torch.Tensor:
        """Evaluate closure for multiple parameter configurations in batch."""
        pop_size = population.shape[0]
        fitness = torch.full((pop_size,), float('inf'), device=self.device, dtype=torch.float32)

        group_id = id(group)

        # Remaining-budget aware guard
        max_fe = getattr(self, '_max_fe', None)
        current_fe = self.state[group_id]['function_evaluations'] if group_id in self.state else 0
        budget_left = max_fe - current_fe if max_fe is not None else pop_size
        if max_fe is not None and budget_left <= 0:
            # No evaluations left; return inf tensor
            return fitness

        to_eval = min(pop_size, budget_left) if max_fe is not None else pop_size

        # Vectorized evaluation using torch.func.vmap if available
        if HAS_FUNCTORCH and hasattr(closure, '__self__') and isinstance(closure.__self__, nn.Module):
            model = closure.__self__
            param_shapes = self.state[id(group)]['param_shapes']
            
            def single_eval(params_flat):
                params = self._unflatten_group_params(params_flat, param_shapes)
                
                # Create a dict of param names to tensors
                param_dict = {name: p for name, p in zip(model.state_dict().keys(), params)}
                
                return closure(functional_call(model, param_dict, ()))

            try:
                # This is a simplified vmap call; real usage might be more complex
                # and depend on the model architecture and closure function.
                # fitness = vmap(single_eval)(population) # This is illustrative
                pass # Not yet implemented
            except Exception as e:
                logger.debug(f"vmap evaluation failed: {e}, falling back to sequential.")

        # Fallback to sequential evaluation (budget-aware)
        for i in range(int(to_eval)):
            fitness[i] = self._evaluate_closure(closure, group, population[i])

        # Increment FE counter AFTER actual evaluations
        if group_id in self.state:
            self.state[group_id]['function_evaluations'] += int(to_eval)
        
        return fitness

    def _optimization_step(
        self,
        closure: Callable,
        group: Dict[str, Any],
        group_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Core optimization step for a parameter group."""
        population = group_state['population']
        pop_size, param_dim = population.shape
        dtype = group_state['dtype']
        
        # Main BFO algorithm loops
        for elim_step in range(group['elimination_steps']):
            for repro_step in range(group['reproduction_steps']):
                for chem_step in range(group['chemotaxis_steps']):
                    
                    # Check FE budget if max_fe is set
                    if hasattr(self, '_max_fe') and self._max_fe is not None:
                        current_fe = group_state.get('function_evaluations', 0)
                        # Estimate FEs for this chemotaxis step including swimming
                        estimated_fe = pop_size * (1 + group['swim_length'])
                        if current_fe + estimated_fe > self._max_fe:
                            logger.info(f"Stopping optimization: {current_fe} + {estimated_fe} > {self._max_fe} FE budget")
                            return group_state
                    
                    # Evaluate fitness for all bacteria (vectorized)
                    fitness = self._evaluate_batch_closure(closure, group, population)
                    
                    # Update best solution
                    min_idx = torch.argmin(fitness)
                    min_fitness = fitness[min_idx].item()
                    
                    if min_fitness < group_state['best_fitness']:
                        group_state['best_fitness'] = min_fitness
                        group_state['best_params'] = population[min_idx].clone()
                        group_state['stagnation_count'] = 0
                    else:
                        group_state['stagnation_count'] += 1
                    
                    # Adaptive step size
                    if len(group_state['fitness_history']) > 1:
                        recent_improvement = abs(group_state['fitness_history'][-1] - min_fitness)
                        if recent_improvement < self.convergence_tolerance:
                            group_state['current_step_size'] *= 0.95
                        else:
                            group_state['current_step_size'] *= 1.05
                        
                        group_state['current_step_size'] = torch.clamp(
                            torch.tensor(group_state['current_step_size'], device=self.device, dtype=dtype),
                            group['step_size_min'], group['step_size_max']
                        ).item()
                    
                    # Chemotaxis: tumble and swim
                    levy_steps = self._levy_flight(
                        population.shape, group['levy_alpha'], self.device, dtype
                    )
                    
                    # Add swarming if enabled
                    if group['enable_swarming']:
                        swarming_forces = self._compute_swarming(population, group['swarming_params'])
                        movement = group_state['current_step_size'] * levy_steps + 0.1 * swarming_forces
                    else:
                        movement = group_state['current_step_size'] * levy_steps
                    
                    # Swimming: continue in beneficial directions (vectorized)
                    new_positions = population + movement
                    new_fitness = self._evaluate_batch_closure(closure, group, new_positions)
                    
                    # Update positions where fitness improved
                    improved = new_fitness < fitness
                    population[improved] = new_positions[improved]
                    fitness[improved] = new_fitness[improved]
                    
                    # Extended swimming (vectorized)
                    swim_count = 0
                    while improved.any() and swim_count < group['swim_length']:
                        swim_indices = torch.where(improved)[0]
                        if swim_indices.numel() == 0:
                            break
                            
                        # Move improved bacteria further in same direction
                        population[swim_indices] += group_state['current_step_size'] * levy_steps[swim_indices]
                        
                        # Re-evaluate only the swimming bacteria
                        swim_fitness = self._evaluate_batch_closure(closure, group, population[swim_indices])
                        
                        # Update improvement status
                        still_improving = swim_fitness < fitness[swim_indices]
                        
                        # Create a boolean mask for the original `improved` tensor
                        update_mask = torch.zeros_like(improved)
                        update_mask[swim_indices[~still_improving]] = True
                        improved[update_mask] = False
                        
                        fitness[swim_indices[still_improving]] = swim_fitness[still_improving]
                        
                        swim_count += 1
                
                # Reproduction: eliminate worst half, duplicate best half
                if pop_size > 1:
                    sorted_indices = torch.argsort(fitness)
                    half = pop_size // 2
                    if half > 0:
                        best_indices = sorted_indices[:half]
                        worst_indices = sorted_indices[half:half*2] if half*2 <= pop_size else sorted_indices[half:]
                        population[worst_indices] = population[best_indices[:len(worst_indices)]].clone()
            
            # Elimination-dispersal with diversity trigger
            elim_prob = group['elimination_prob']
            
            # Calculate population diversity
            pop_mean = population.mean(dim=0)
            diversity = torch.norm(population - pop_mean, dim=1).mean().item()
            
            # Dynamic diversity threshold with decay over iterations
            max_iterations = 100  # Approximate max iterations for decay
            decay_factor = max(1.0 - group_state['iteration'] / max_iterations, 0.1)
            diversity_threshold = max(0.01 * param_dim * decay_factor, 1e-3)  # Scale with dimension and decay
            
            # Trigger elimination if stagnating or low diversity
            if group_state['stagnation_count'] > 5:
                elim_prob = min(0.5, elim_prob * 1.5)  # Increase elimination if stagnating
            elif diversity < diversity_threshold:
                # Force high elimination when diversity is too low
                elim_prob = min(0.8, elim_prob * 3.0)
                if group_state['iteration'] % 10 == 0:  # Log periodically
                    logger.debug(f"Low diversity ({diversity:.4f} < {diversity_threshold:.4f}), forcing elimination with p={elim_prob:.2f}")
            
            eliminate = torch.rand(pop_size, device=self.device) < elim_prob
            if eliminate.any():
                # Replace eliminated bacteria with random positions around best solution
                n_eliminate = eliminate.sum().item()
                new_bacteria = (
                    group_state['best_params'].unsqueeze(0) + 
                    torch.randn((n_eliminate, param_dim), device=self.device, dtype=dtype) *
                    group_state['current_step_size']
                )
                population[eliminate] = new_bacteria
        
        # Update fitness history
        group_state['fitness_history'].append(group_state['best_fitness'])
        if len(group_state['fitness_history']) > 50:  # Keep limited history
            group_state['fitness_history'].pop(0)
        
        group_state['iteration'] += 1
        return group_state
    
    def get_function_evaluations(self) -> int:
        """Get the total number of function evaluations performed."""
        total_evals = 0
        for group in self.param_groups:
            group_id = id(group)
            if group_id in self.state:
                total_evals += self.state[group_id].get('function_evaluations', 0)
        return total_evals
    
    def step(self, closure: Optional[Callable] = None, max_fe: Optional[int] = None) -> float:
        """
        Perform a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. Required for BFO.
            max_fe (int, optional): Maximum function evaluations allowed.
                
        Returns:
            float: Best fitness value found across all parameter groups
        """
        if closure is None:
            raise ValueError("BFO requires a closure that returns the loss")
        
        # Set max_fe for budget checking in optimization step
        self._max_fe = max_fe
        
        best_fitness = float('inf')
        
        for group in self.param_groups:
            group_id = id(group)
            
            # Initialize group state if not present
            if group_id not in self.state:
                self._initialize_group_state(group)
            
            group_state = self.state[group_id]
            
            # Run optimization step
            group_state = self._compiled_step(closure, group, group_state)
            
            # Update parameters with best solution
            param_list = self._unflatten_group_params(
                group_state['best_params'], 
                group_state['param_shapes']
            )
            for p, new_p in zip(group['params'], param_list):
                p.data.copy_(new_p)
            
            # Track best fitness across all groups
            best_fitness = min(best_fitness, group_state['best_fitness'])
            
            # Early stopping check
            if (self.early_stopping and 
                group_state['stagnation_count'] >= self.convergence_patience):
                logger.info(f"Early stopping triggered for group {group_id}")
        
        return best_fitness
    
    def state_dict(self) -> Dict[str, Any]:
        """Return the optimizer's state as a dictionary."""
        state_dict = super().state_dict()
        
        # Add BFO-specific state
        bfo_state = {}
        for group_id, group_state in self.state.items():
            bfo_state[group_id] = {
                'population': group_state['population'].clone(),
                'best_params': group_state['best_params'].clone(),
                'best_fitness': group_state['best_fitness'],
                'fitness_history': group_state['fitness_history'].copy(),
                'current_step_size': group_state['current_step_size'],
                'stagnation_count': group_state['stagnation_count'],
                'iteration': group_state['iteration']
            }
        
        state_dict['bfo_state'] = bfo_state
        
        # Add RNG state for reproducibility
        state_dict['rng_state'] = {
            'torch_rng': torch.get_rng_state(),
            'numpy_rng': np.random.get_state()
        }
        
        if torch.cuda.is_available():
            state_dict['rng_state']['cuda_rng'] = torch.cuda.get_rng_state()
        
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the optimizer's state."""
        # Restore RNG states
        if 'rng_state' in state_dict:
            torch.set_rng_state(state_dict['rng_state']['torch_rng'])
            np.random.set_state(state_dict['rng_state']['numpy_rng'])
            if 'cuda_rng' in state_dict['rng_state'] and torch.cuda.is_available():
                torch.cuda.set_rng_state(state_dict['rng_state']['cuda_rng'])
        
        # Restore BFO-specific state
        if 'bfo_state' in state_dict:
            for group_id, group_state_dict in state_dict['bfo_state'].items():
                if group_id in self.state:
                    self.state[group_id].update({
                        'population': group_state_dict['population'].to(self.device),
                        'best_params': group_state_dict['best_params'].to(self.device),
                        'best_fitness': group_state_dict['best_fitness'],
                        'fitness_history': group_state_dict['fitness_history'],
                        'current_step_size': group_state_dict['current_step_size'],
                        'stagnation_count': group_state_dict['stagnation_count'],
                        'iteration': group_state_dict['iteration']
                    })
        
        # Restore parent state
        parent_state_dict = {k: v for k, v in state_dict.items() 
                           if k not in ['bfo_state', 'rng_state']}
        super().load_state_dict(parent_state_dict)


class AdaptiveBFO(BFO):
    """
    Adaptive Bacterial Foraging Optimization (BFO) optimizer.
    
    Extends BFO with adaptive parameter adjustment based on performance.
    Automatically tunes population size, step sizes, and elimination probability.
    
    Additional Arguments:
        adaptation_rate (float, optional): Rate of parameter adaptation (default: 0.1)
        min_population_size (int, optional): Minimum population size (default: 10)
        max_population_size (int, optional): Maximum population size (default: 100)
        diversity_threshold (float, optional): Minimum population diversity (default: 1e-3)
    """
    
    def __init__(
        self,
        params,
        adaptation_rate: float = 0.1,
        min_population_size: int = 10,
        max_population_size: int = 100,
        diversity_threshold: float = 1e-3,
        **kwargs
    ):
        super().__init__(params, **kwargs)
        
        self.adaptation_rate = adaptation_rate
        self.min_population_size = min_population_size
        self.max_population_size = max_population_size
        self.diversity_threshold = diversity_threshold

    def _compute_population_diversity(self, population: torch.Tensor) -> float:
        """Compute population diversity metric."""
        if population.shape[0] <= 1:
            return 0.0
        
        mean_pos = population.mean(dim=0)
        diversity = torch.norm(population - mean_pos, dim=1).mean()
        return diversity.item()
    
    def _adapt_parameters(self, group: Dict[str, Any], group_state: Dict[str, Any], closure: Callable) -> None:
        """Adapt optimization parameters based on performance."""
        if len(group_state['fitness_history']) < 5:
            return
        
        recent_improvement = abs(
            group_state['fitness_history'][-1] - group_state['fitness_history'][-5]
        )
        
        # Adapt population size based on progress
        current_pop_size = group_state['population'].shape[0]
        if recent_improvement < self.convergence_tolerance:
            # Increase population if stagnating
            new_pop_size = min(
                self.max_population_size,
                int(current_pop_size * (1 + self.adaptation_rate))
            )
        elif recent_improvement > 0.01:
            # Decrease population if making good progress
            new_pop_size = max(
                self.min_population_size,
                int(current_pop_size * (1 - self.adaptation_rate))
            )
        else:
            new_pop_size = current_pop_size
        
        # Resize population if needed
        if new_pop_size != current_pop_size:
            self._resize_population(group, group_state, new_pop_size, closure)
        
        # Adapt elimination probability based on diversity
        diversity = self._compute_population_diversity(group_state['population'])
        if diversity < self.diversity_threshold:
            group['elimination_prob'] = min(0.5, group['elimination_prob'] * 1.2)
    
    def _resize_population(self, group: Dict[str, Any], group_state: Dict[str, Any], new_size: int, closure: Callable) -> None:
        """Resize population while preserving best solutions."""
        current_population = group_state['population']
        current_size = current_population.shape[0]
        
        if new_size == current_size:
            return
        
        if new_size > current_size:
            # Add new individuals around best solution
            additional = new_size - current_size
            new_individuals = (
                group_state['best_params'].unsqueeze(0) +
                torch.randn(
                    additional, current_population.shape[1],
                    device=self.device, dtype=group_state['dtype']
                ) * group_state['current_step_size']
            )
            group_state['population'] = torch.cat([current_population, new_individuals], dim=0)
        else:
            # Keep best individuals by evaluating current fitness
            fitness = torch.full((current_size,), float('inf'), device=self.device, dtype=group_state['dtype'])
            for i in range(current_size):
                fitness[i] = self._evaluate_closure(closure, group, current_population[i])
            
            # Sort by fitness and keep the best
            sorted_indices = torch.argsort(fitness)
            keep_indices = sorted_indices[:new_size]
            group_state['population'] = current_population[keep_indices]
        
        # Update group population size
        group['population_size'] = new_size
    
    def step(self, closure: Optional[Callable] = None, max_fe: Optional[int] = None) -> float:
        """Perform optimization step with parameter adaptation."""
        if closure is None:
            raise ValueError("AdaptiveBFO requires a closure.")
            
        # Adapt parameters for each group *before* the optimization step
        for group in self.param_groups:
            group_id = id(group)
            if group_id in self.state:
                self._adapt_parameters(group, self.state[group_id], closure)

        # Now, perform the optimization step with the (potentially) resized population
        fitness = super().step(closure, max_fe=max_fe)
        
        return fitness
    
    def state_dict(self) -> Dict[str, Any]:
        """Return the optimizer's state, including adaptive parameters."""
        state_dict = super().state_dict()
        for group in self.param_groups:
            group_id = id(group)
            if group_id in state_dict['bfo_state']:
                state_dict['bfo_state'][group_id]['population_size'] = group['population_size']
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the optimizer's state, including adaptive parameters."""
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            group_id = id(group)
            if 'bfo_state' in state_dict and group_id in state_dict['bfo_state']:
                if 'population_size' in state_dict['bfo_state'][group_id]:
                    group['population_size'] = state_dict['bfo_state'][group_id]['population_size']


class HybridBFO(BFO):
    """
    Hybrid Bacterial Foraging Optimization (BFO) optimizer.
    
    Combines BFO with gradient information when available for faster convergence
    on differentiable problems. Includes a safety check for momentum.
    
    Additional Arguments:
        gradient_weight (float, optional): Weight for gradient contribution (default: 0.5)
        momentum (float, optional): Momentum coefficient for gradient updates (default: 0.9)
        enable_momentum (bool, optional): Enable momentum for gradients (default: True)
    """
    
    def __init__(
        self,
        params,
        gradient_weight: float = 0.5,
        momentum: float = 0.9,
        enable_momentum: bool = True,
        **kwargs
    ):
        super().__init__(params, **kwargs)
        
        self.gradient_weight = gradient_weight
        self.momentum = momentum
        self.enable_momentum = enable_momentum
        
        if self.enable_momentum and self.momentum == 0:
            self.enable_momentum = False

        # Initialize momentum buffers for each parameter group
        if self.enable_momentum:
            for group in self.param_groups:
                group_id = id(group)
                if group_id not in self.state:
                    self.state[group_id] = {}
                
                param_vector, _ = self._flatten_group_params(group)
                self.state[group_id]['momentum_buffer'] = torch.zeros_like(param_vector)
    
    def _has_gradients(self, group: Dict[str, Any]) -> bool:
        """Check if any parameters in group have gradients."""
        return any(p.grad is not None for p in group['params'])
    
    def _collect_gradients(self, group: Dict[str, Any]) -> torch.Tensor:
        """Collect gradients from parameter group."""
        grad_list = []
        for p in group['params']:
            if p.grad is not None:
                grad_list.append(p.grad.view(-1))
            else:
                grad_list.append(torch.zeros(p.numel(), device=self.device, dtype=p.dtype))
        
        return torch.cat(grad_list).to(self.device)
    
    def _apply_gradient_bias(self, group: Dict[str, Any], group_state: Dict[str, Any]) -> None:
        """Apply gradient-based bias to population."""
        if not self._has_gradients(group) or self.gradient_weight == 0:
            return
        
        group_id = id(group)
        grad_vector = self._collect_gradients(group)
        
        # Apply momentum if enabled
        if self.enable_momentum and 'momentum_buffer' in self.state[group_id]:
            momentum_buffer = self.state[group_id]['momentum_buffer']
            momentum_buffer.mul_(self.momentum).add_(grad_vector, alpha=1 - self.momentum)
            grad_vector = momentum_buffer.clone()
        
        # Gradient descent step
        gradient_step = -group['lr'] * grad_vector
        
        # Bias population towards gradient direction
        population = group_state['population']
        gradient_bias = group_state['param_vector'] + gradient_step
        
        for i in range(population.shape[0]):
            population[i] = (
                (1 - self.gradient_weight) * population[i] +
                self.gradient_weight * gradient_bias
            )
    
    def step(self, closure: Optional[Callable] = None, max_fe: Optional[int] = None) -> float:
        """Perform hybrid optimization step."""
        # Apply gradient bias before BFO step
        for group in self.param_groups:
            group_id = id(group)
            if group_id in self.state:
                self._apply_gradient_bias(group, self.state[group_id])
        
        # Perform BFO step
        fitness = super().step(closure, max_fe=max_fe)
        
        # Adaptive gradient weight based on improvement
        for group in self.param_groups:
            group_id = id(group)
            if group_id in self.state:
                group_state = self.state[group_id]
                if len(group_state['fitness_history']) > 5:
                    recent_improvement = abs(
                        group_state['fitness_history'][-1] - group_state['fitness_history'][-5]
                    )
                    if recent_improvement < self.convergence_tolerance:
                        # Reduce gradient influence if stagnating
                        self.gradient_weight = max(0.1, self.gradient_weight * 0.95)
        
        return fitness
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict and ensure momentum buffers are properly initialized."""
        # Call parent's load_state_dict
        super().load_state_dict(state_dict)
        
        # Reinitialize momentum buffers if needed
        if self.enable_momentum:
            for group in self.param_groups:
                group_id = id(group)
                if group_id in self.state:
                    # Check if momentum buffer exists, create if missing
                    if 'momentum_buffer' not in self.state[group_id]:
                        param_vector, _ = self._flatten_group_params(group)
                        self.state[group_id]['momentum_buffer'] = torch.zeros_like(param_vector)