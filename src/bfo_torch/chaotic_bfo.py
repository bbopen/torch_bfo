"""
Chaotic Bacterial Foraging Optimization (ChaoticBFO) with P1 Improvements
========================================================================

Enhanced BFO implementation with:
1. Diversity-triggered elimination with global restart
2. Chaos injection using logistic map
3. GA crossover in reproduction
4. Dynamic diversity threshold
5. Strict FE budget enforcement

Designed specifically for deceptive multimodal functions like Schwefel.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional, Callable, Dict, Any, Tuple
from .optimizer import BFO

logger = logging.getLogger(__name__)


class ChaoticBFO(BFO):
    """
    Chaotic BFO with enhanced exploration capabilities for deceptive landscapes.
    
    Additional Parameters:
        enable_chaos (bool): Enable chaotic Lévy flights (default: True)
        chaos_strength (float): Strength of chaos injection (default: 0.5)
        diversity_trigger_ratio (float): Ratio of population to replace on low diversity (default: 0.5)
        enable_crossover (bool): Enable GA crossover in reproduction (default: True)
        diversity_threshold_decay (float): Decay rate for diversity threshold (default: 0.9)
    """
    
    def __init__(
        self,
        params,
        enable_chaos: bool = True,
        chaos_strength: float = 0.5,
        diversity_trigger_ratio: float = 0.5,
        enable_crossover: bool = True,
        diversity_threshold_decay: float = 0.9,
        restart_fraction: float = 0.3,
        success_tolerance: float = 1e-3,
        domain_bounds: Tuple[float, float] = (-500.0, 500.0),
        elimination_prob: float = 0.25,
        **kwargs
    ):
        super().__init__(params, **kwargs)
        
        # Chaotic BFO specific parameters
        self.enable_chaos = enable_chaos
        self.chaos_strength = chaos_strength
        self.diversity_trigger_ratio = diversity_trigger_ratio
        self.enable_crossover = enable_crossover
        self.diversity_threshold_decay = diversity_threshold_decay
        self.restart_fraction = restart_fraction
        self.success_tolerance = success_tolerance
        self.domain_bounds = domain_bounds

        # Store initial elimination probability for linear decay
        self.initial_elimination_prob = elimination_prob
        
        # Initialize chaos state
        self.chaos_state = torch.rand(1).item()
    
    def step(self, closure: Optional[Callable] = None, max_fe: Optional[int] = None) -> float:
        """Override step to handle max_fe parameter."""
        # Store max_fe for use in _optimization_step
        self._max_fe = max_fe
        # Call parent step method
        return super().step(closure)
    
    def _levy_flight(self, size: torch.Size, alpha: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Enhanced Lévy flight with optional chaos injection."""
        # Try standard Lévy flight first
        try:
            # Standard Lévy flight implementation
            beta = 1.5
            sigma_u = (torch.lgamma(torch.tensor(1 + beta)) * torch.sin(torch.tensor(np.pi * beta / 2)) / 
                      (torch.lgamma(torch.tensor((1 + beta) / 2)) * beta * 2**((beta - 1) / 2)))**(1 / beta)
            
            u = torch.randn(size, device=device, dtype=dtype) * sigma_u
            v = torch.randn(size, device=device, dtype=dtype)
            
            step = u / (torch.abs(v)**(1 / beta))
            
            # Apply chaos if enabled
            if self.enable_chaos:
                chaos = self._generate_chaos(size, device, dtype)
                step = (1 - self.chaos_strength) * step + self.chaos_strength * chaos
            
            return step * (torch.rand(size, device=device, dtype=dtype) < 0.5).float() * 2 - 1
            
        except Exception as e:
            logger.warning(f"Lévy flight calculation failed: {e}, using chaotic fallback")
            # Pure chaotic fallback
            chaos = self._generate_chaos(size, device, dtype)
            return chaos * torch.randn(size, device=device, dtype=dtype)
    
    def _generate_chaos(self, size: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Generate chaos using logistic map."""
        chaos = torch.empty(size, device=device, dtype=dtype)
        
        # Vectorized logistic map iteration
        x = torch.rand(size, device=device, dtype=dtype)
        for _ in range(10):  # Iterate to reach chaotic regime
            x = 4.0 * x * (1.0 - x)
        
        # Scale to [-1, 1] range
        chaos = 2.0 * x - 1.0
        return chaos
    
    def _optimization_step(
        self,
        closure: Callable,
        group: Dict[str, Any],
        group_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced optimization step with diversity-triggered elimination."""
        population = group_state['population']
        pop_size, param_dim = population.shape
        dtype = group_state['dtype']
 
        # Total planned iterations for this group (chemotaxis across all repro & elimination cycles)
        max_iterations = (
            group['chemotaxis_steps'] * group['reproduction_steps'] * group['elimination_steps']
        )

        # Track diversity history for analysis
        if 'diversity_history' not in group_state:
            group_state['diversity_history'] = []
        
        # Main BFO algorithm loops
        for elim_step in range(group['elimination_steps']):
            for repro_step in range(group['reproduction_steps']):
                for chem_step in range(group['chemotaxis_steps']):

                    # Hard FE stop before any evaluation
                    if hasattr(self, '_max_fe') and self._max_fe is not None:
                        remaining_fe = self._max_fe - group_state.get('function_evaluations', 0)
                        if remaining_fe < pop_size:
                            logger.info(f"Stopping: remaining FE {remaining_fe} < pop_size {pop_size}")
                            return group_state
                    
                    # Evaluate fitness for all bacteria (vectorized)
                    fitness = self._evaluate_batch_closure(closure, group, population)
                    
                    # NEW: Strict FE-budget check after actual evaluations
                    if hasattr(self, '_max_fe') and self._max_fe is not None and \
                        group_state.get('function_evaluations', 0) >= self._max_fe:
                        logger.info(
                            f"FE budget {self._max_fe} reached (current: {group_state['function_evaluations']}). Stopping.")
                        return group_state
                    
                    # Update best solution
                    min_idx = torch.argmin(fitness)
                    min_fitness = fitness[min_idx].item()
                    
                    if min_fitness < group_state['best_fitness']:
                        group_state['best_fitness'] = min_fitness
                        group_state['best_params'] = population[min_idx].clone()
                        group_state['stagnation_count'] = 0
                    else:
                        group_state['stagnation_count'] += 1

                    # Early success exit if within tolerance
                    if min_fitness <= self.success_tolerance:
                        logger.info(f"Reached success tolerance {self.success_tolerance}; exiting early.")
                        group_state['best_fitness'] = min_fitness
                        group_state['best_params'] = population[min_idx].clone()
                        return group_state

                    # Micro-restart when stagnant for 3 chemotaxis cycles
                    if group_state['stagnation_count'] >= 3:
                        if hasattr(self, '_max_fe') and self._max_fe is not None:
                            remaining_fe = self._max_fe - group_state.get('function_evaluations', 0)
                        else:
                            remaining_fe = pop_size

                        num_restart = int(pop_size * self.restart_fraction)
                        num_restart = min(num_restart, int(remaining_fe))

                        if num_restart > 0 and remaining_fe > 0:
                            restart_idx = torch.randperm(pop_size, device=self.device)[:num_restart]
                            domain_min, domain_max = self.domain_bounds
                            new_positions = torch.rand((num_restart, param_dim), device=self.device, dtype=dtype)
                            new_positions = new_positions * (domain_max - domain_min) + domain_min
                            population[restart_idx] = new_positions
                            group_state['function_evaluations'] += num_restart
                            logger.info(f"Micro-restart: replaced {num_restart} stagnant bacteria")
                        group_state['stagnation_count'] = 0
                    
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
                    
                    # Chemotaxis with chaotic Lévy flight
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
                
                # Roulette-wheel reproduction with optional crossover
                if pop_size > 1:
                    # Inverse fitness to get higher probability for low loss
                    inv_fit = 1.0 / (fitness + 1e-12)
                    probs = inv_fit / inv_fit.sum()
                    half = pop_size // 2

                    parents_idx = torch.multinomial(probs, half, replacement=True)
                    offspring_idx = torch.randperm(pop_size, device=self.device)[:half]

                    if self.enable_crossover:
                        for dst, p_idx in zip(offspring_idx, parents_idx):
                            parent2_idx = torch.multinomial(probs, 1).item()
                            mask = torch.rand(param_dim, device=self.device) < 0.5
                            child = torch.where(mask, population[p_idx], population[parent2_idx])
                            mutation = torch.randn_like(child) * group_state['current_step_size'] * 0.05
                            population[dst] = child + mutation
                    else:
                        population[offspring_idx] = population[parents_idx].clone()
            
            # Enhanced elimination-dispersal with diversity trigger
            elim_prob = group['elimination_prob']
            
            # Linearly decaying elimination probability
            iter_frac = group_state['iteration'] / max_iterations if max_iterations > 0 else 0.0
            elim_prob = self.initial_elimination_prob * max(0.1, 1.0 - iter_frac)

            # Calculate population diversity
            pop_mean = population.mean(dim=0)
            diversity = torch.norm(population - pop_mean, dim=1).mean().item()
            
            # Dynamic diversity threshold with decay over iterations
            decay_factor = max(1.0 - group_state['iteration'] / max_iterations, 0.1)
            diversity_threshold = max(0.01 * param_dim * decay_factor, 1e-3)
            
            # Track diversity
            group_state['diversity_history'].append(diversity)
            if len(group_state['diversity_history']) > 100:
                group_state['diversity_history'].pop(0)
            
            # Diversity-triggered elimination
            if diversity < diversity_threshold:
                # Force high elimination to inject diversity
                logger.info(f"Low diversity ({diversity:.4f} < {diversity_threshold:.4f}), triggering forced dispersal")
                
                num_replace = int(pop_size * self.diversity_trigger_ratio)

                # FE-aware cap
                if hasattr(self, '_max_fe') and self._max_fe is not None:
                    remaining_fe = self._max_fe - group_state.get('function_evaluations', 0)
                    num_replace = min(num_replace, int(remaining_fe))

                if num_replace > 0:
                    replace_idx = torch.randperm(pop_size, device=self.device)[:num_replace]
                
                # Generate new random positions in full domain
                domain_min, domain_max = self.domain_bounds
                new_positions = torch.rand((num_replace, param_dim), device=self.device, dtype=dtype)
                new_positions = new_positions * (domain_max - domain_min) + domain_min
                
                if num_replace > 0:
                    population[replace_idx] = new_positions
                    group_state['function_evaluations'] += num_replace
                    logger.info(f"Replaced {num_replace} bacteria with random positions")
            else:
                # Standard elimination-dispersal
                if group_state['stagnation_count'] > 5:
                    elim_prob = min(0.5, elim_prob * 1.5)
                
                eliminate = torch.rand(pop_size, device=self.device) < elim_prob
                if eliminate.any():
                    n_eliminate = eliminate.sum().item()
                    new_bacteria = (
                        group_state['best_params'].unsqueeze(0) + 
                        torch.randn(n_eliminate, param_dim, device=self.device, dtype=dtype) * 
                        group['step_size_max'] * 2.0
                    )
                    population[eliminate] = new_bacteria
        
        # Update group state
        group_state['population'] = population
        
        # Update fitness history
        group_state['fitness_history'].append(group_state['best_fitness'])
        if len(group_state['fitness_history']) > 50:
            group_state['fitness_history'].pop(0)
        
        group_state['iteration'] += 1
        return group_state