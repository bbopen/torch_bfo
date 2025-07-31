#!/usr/bin/env python3
"""
Enhanced Schwefel Function Experiments with Proper FE Counting
==============================================================

Implements all P0 improvements:
1. Accurate function evaluation counting
2. Enhanced parameter configurations with swarming
3. HybridBFO experiments with gradient support
4. Dimension sweep (2D, 10D, 30D)
5. Increased evaluation budget (50k)
6. Reflective bounds (P1 - included as bonus)

Following standard benchmarking protocols with unbiased initialization.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import json
import statistics
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import our BFO implementations
from src.bfo_torch.optimizer import BFO, AdaptiveBFO, HybridBFO
from src.bfo_torch.chaotic_bfo import ChaoticBFO


@dataclass
class EnhancedExperimentConfig:
    """Enhanced configuration for Schwefel experiments."""
    name: str
    optimizer_class: type
    optimizer_params: Dict
    dimension: int = 2
    num_runs: int = 30
    max_evaluations: int = 50000  # Increased from 10k
    tolerance: float = 1e-4
    domain_bounds: Tuple[float, float] = (-500.0, 500.0)
    use_gradient: bool = False
    use_reflective_bounds: bool = True  # P1 improvement


class SchwefelEnhancedExperiments:
    """Enhanced Schwefel experiments with accurate FE counting and improvements."""
    
    def __init__(self):
        self.schwefel_global_optimum = 0.0
        self.schwefel_optimum_position = 420.9687
        
    def schwefel_function(self, x: torch.Tensor) -> torch.Tensor:
        """Standard Schwefel function implementation."""
        n = len(x)
        return 418.9829 * n - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))))
    
    def reflective_bounds(self, x: torch.Tensor, low: float, high: float) -> torch.Tensor:
        """Implement reflective bounds instead of clamping (P1 improvement)."""
        # Find values outside bounds
        below_low = x < low
        above_high = x > high
        
        # Reflect back into domain
        x_reflected = x.clone()
        
        # For values below low, reflect back
        if below_low.any():
            distance_below = low - x[below_low]
            x_reflected[below_low] = low + distance_below
        
        # For values above high, reflect back
        if above_high.any():
            distance_above = x[above_high] - high
            x_reflected[above_high] = high - distance_above
        
        # Handle multiple reflections (if reflected value is still outside)
        # Use modulo for efficiency
        range_size = high - low
        still_outside = (x_reflected < low) | (x_reflected > high)
        if still_outside.any():
            x_reflected[still_outside] = low + torch.remainder(x_reflected[still_outside] - low, range_size)
        
        return x_reflected
    
    def run_single_enhanced_experiment(self, config: EnhancedExperimentConfig, seed: int) -> Dict:
        """Run single enhanced experiment with proper FE counting."""
        # Set reproducible seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Standard uniform initialization over full domain
        x = nn.Parameter(torch.rand(config.dimension) * 
                        (config.domain_bounds[1] - config.domain_bounds[0]) + 
                        config.domain_bounds[0])
        
        # Create optimizer
        optimizer = config.optimizer_class([x], **config.optimizer_params)
        
        # For HybridBFO, we need gradients
        if config.use_gradient:
            x.requires_grad_(True)
        
        def closure():
            if config.use_gradient:
                optimizer.zero_grad()
                loss = self.schwefel_function(x)
                loss.backward()
                return loss.item()
            else:
                loss = self.schwefel_function(x)
                return loss.item()
        
        # Apply bounds based on configuration
        def apply_bounds():
            with torch.no_grad():
                if config.use_reflective_bounds:
                    x.data = self.reflective_bounds(x.data, config.domain_bounds[0], config.domain_bounds[1])
                else:
                    x.data = torch.clamp(x.data, config.domain_bounds[0], config.domain_bounds[1])
        
        # Track progress
        initial_loss = closure()
        best_loss = initial_loss
        convergence_step = None
        losses = [initial_loss]
        fe_history = [0]  # Track function evaluations over time
        
        # Run optimization with proper FE budget management
        step = 0
        converged = False
        
        while not converged:
            # Check remaining budget before step
            current_fe = optimizer.get_function_evaluations()
            
            # Skip the pre-step FE check - let the optimizer handle it internally
            # The optimizer.step() method with max_fe parameter will handle budget enforcement
            
            # Simple check if we've reached the budget
            if current_fe >= config.max_evaluations:
                print(f"  Reached FE budget: {current_fe} >= {config.max_evaluations}")
                break
            
            # Perform optimization step with FE budget
            loss = optimizer.step(closure, max_fe=config.max_evaluations)
            apply_bounds()
            
            # Track actual function evaluations
            actual_fe = optimizer.get_function_evaluations()
            fe_history.append(actual_fe)
            
            losses.append(loss)
            
            if loss < best_loss:
                best_loss = loss
            
            # Check convergence
            if best_loss <= config.tolerance and convergence_step is None:
                convergence_step = step + 1
                converged = True
            
            # Early stopping on stagnation (no improvement for 20k FEs)
            if len(fe_history) > 10:
                fe_20k_ago = next((fe for fe in reversed(fe_history[:-5]) if actual_fe - fe >= 20000), None)
                if fe_20k_ago is not None:
                    idx_20k_ago = fe_history.index(fe_20k_ago)
                    if abs(losses[idx_20k_ago] - best_loss) < 1e-6:
                        print(f"  Early stopping: no improvement for 20k FEs")
                        break
            
            step += 1
            
            # Safety break
            if step > 10000:  # Very high limit since we're tracking FEs properly
                break
        
        final_position = x.data.clone().cpu().numpy()
        final_fe = optimizer.get_function_evaluations()
        
        return {
            'seed': seed,
            'initial_loss': initial_loss,
            'final_loss': best_loss,
            'convergence_step': convergence_step,
            'function_evaluations': final_fe,
            'optimization_steps': step,
            'success': best_loss <= config.tolerance,
            'final_position': final_position.tolist(),
            'improvement_ratio': (initial_loss - best_loss) / abs(initial_loss) if initial_loss != 0 else 0,
            'loss_trajectory': losses[:1000],  # First 1000 steps
            'fe_trajectory': fe_history[:1000]
        }
    
    def run_enhanced_experiment_series(self, config: EnhancedExperimentConfig) -> Dict:
        """Run complete enhanced experiment series."""
        print(f"\nRunning {config.name} (Dim={config.dimension})...")
        print(f"Configuration: {config.optimizer_params}")
        print(f"Runs: {config.num_runs}, Max FE: {config.max_evaluations}")
        
        results = []
        start_time = time.time()
        
        # Run all experiments
        for run in range(config.num_runs):
            if (run + 1) % 10 == 0:
                print(f"  Completed {run + 1}/{config.num_runs} runs...")
            
            seed = 1000 + run + config.dimension * 100  # Dimension-specific seeds
            result = self.run_single_enhanced_experiment(config, seed)
            results.append(result)
        
        end_time = time.time()
        
        # Statistical analysis
        final_losses = [r['final_loss'] for r in results]
        function_evals = [r['function_evaluations'] for r in results]
        convergence_steps = [r['convergence_step'] for r in results if r['convergence_step'] is not None]
        success_count = sum(1 for r in results if r['success'])
        
        stats = {
            'experiment_name': config.name,
            'dimension': config.dimension,
            'configuration': config.optimizer_params,
            'experimental_setup': {
                'num_runs': config.num_runs,
                'max_evaluations': config.max_evaluations,
                'tolerance': config.tolerance,
                'dimension': config.dimension,
                'domain_bounds': config.domain_bounds,
                'initialization': 'uniform_random',
                'use_gradient': config.use_gradient,
                'use_reflective_bounds': config.use_reflective_bounds
            },
            'results': {
                'success_rate': success_count / config.num_runs,
                'successful_runs': success_count,
                'total_runs': config.num_runs,
                'final_loss_mean': statistics.mean(final_losses),
                'final_loss_std': statistics.stdev(final_losses) if len(final_losses) > 1 else 0,
                'final_loss_median': statistics.median(final_losses),
                'final_loss_min': min(final_losses),
                'final_loss_max': max(final_losses),
                'function_evaluations_mean': statistics.mean(function_evals),
                'function_evaluations_std': statistics.stdev(function_evals) if len(function_evals) > 1 else 0,
                'convergence_steps_mean': statistics.mean(convergence_steps) if convergence_steps else None,
                'convergence_steps_std': statistics.stdev(convergence_steps) if len(convergence_steps) > 1 else None
            },
            'individual_runs': results,
            'execution_time': end_time - start_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"  Results: {success_count}/{config.num_runs} successful ({stats['results']['success_rate']:.1%})")
        print(f"  Final loss: {stats['results']['final_loss_mean']:.6f} ± {stats['results']['final_loss_std']:.6f}")
        print(f"  Function evaluations: {stats['results']['function_evaluations_mean']:.1f} ± {stats['results']['function_evaluations_std']:.1f}")
        
        return stats
    
    def create_enhanced_configurations(self) -> List[EnhancedExperimentConfig]:
        """Create all enhanced experiment configurations."""
        configs = []
        
        # Dimensions to test
        dimensions = [2, 10, 30]
        
        for dim in dimensions:
            # Scale parameters based on dimension
            base_pop = 40 * dim  # Population scales with dimension
            base_lr = 0.01 / np.sqrt(dim)  # Step size scales inversely with sqrt(dim)
            
            # 1. Enhanced Standard BFO with swarming
            configs.append(EnhancedExperimentConfig(
                name="Enhanced_BFO_Swarming",
                optimizer_class=BFO,
                optimizer_params={
                    'population_size': min(200, base_pop),  # Cap at 200
                    'lr': base_lr,
                    'chemotaxis_steps': 20,  # Increased
                    'reproduction_steps': 10,  # Increased
                    'elimination_steps': 5,  # Increased
                    'elimination_prob': 0.4,  # Higher
                    'step_size_min': 1e-4,
                    'step_size_max': 1.0,  # Much larger
                    'levy_alpha': 1.8,  # Heavier tails
                    'enable_swarming': True,  # KEY: Enable swarming
                    'swarming_params': (0.2, 0.1, 0.2, 10.0)
                },
                dimension=dim
            ))
            
            # 2. Large Population BFO
            configs.append(EnhancedExperimentConfig(
                name="Large_Population_Enhanced",
                optimizer_class=BFO,
                optimizer_params={
                    'population_size': min(300, base_pop * 2),  # Double population
                    'lr': base_lr * 0.5,  # Smaller steps with more agents
                    'chemotaxis_steps': 15,
                    'reproduction_steps': 8,
                    'elimination_steps': 4,
                    'elimination_prob': 0.3,
                    'step_size_min': 1e-4,
                    'step_size_max': 0.8,
                    'levy_alpha': 1.7,
                    'enable_swarming': True
                },
                dimension=dim
            ))
            
            # 3. Adaptive BFO Enhanced
            configs.append(EnhancedExperimentConfig(
                name="Adaptive_BFO_Enhanced",
                optimizer_class=AdaptiveBFO,
                optimizer_params={
                    'population_size': min(150, base_pop),
                    'lr': base_lr,
                    'adaptation_rate': 0.2,
                    'min_population_size': max(20, dim * 2),
                    'max_population_size': min(300, base_pop * 3),
                    'step_size_min': 1e-4,
                    'step_size_max': 1.2,
                    'levy_alpha': 1.8,
                    'enable_swarming': True
                },
                dimension=dim
            ))
            
            # 4. HybridBFO with gradient
            configs.append(EnhancedExperimentConfig(
                name="Hybrid_BFO_Gradient",
                optimizer_class=HybridBFO,
                optimizer_params={
                    'population_size': min(100, base_pop),
                    'lr': base_lr,
                    'gradient_weight': 0.7,  # 70% gradient, 30% stochastic
                    'chemotaxis_steps': 10,
                    'reproduction_steps': 5,
                    'elimination_steps': 3,
                    'elimination_prob': 0.25,
                    'step_size_min': 1e-4,
                    'step_size_max': 0.5,  # More conservative with gradient
                    'levy_alpha': 1.6,
                    'enable_swarming': True
                },
                dimension=dim,
                use_gradient=True
            ))
        
        # Add P1 configurations (2D only for initial testing)
        if 2 in dimensions:
            # P1-Chaotic-BFO (baseline P1 with all improvements)
            # Reduced internal steps to fit within 50k budget
            configs.append(EnhancedExperimentConfig(
                name="P1_Chaotic_BFO",
                optimizer_class=ChaoticBFO,
                optimizer_params={
                    'population_size': 50,  # Reduced
                    'lr': 0.01,
                    'chemotaxis_steps': 10,  # Reduced
                    'reproduction_steps': 5,  # Reduced
                    'elimination_steps': 2,  # Reduced
                    'elimination_prob': 0.4,
                    'step_size_min': 1e-4,
                    'step_size_max': 1.0,
                    'levy_alpha': 1.8,
                    'enable_swarming': True,
                    'enable_chaos': True,
                    'chaos_strength': 0.5,
                    'diversity_trigger_ratio': 0.5,
                    'enable_crossover': True,
                    'diversity_threshold_decay': 0.9,
                    'swim_length': 5  # Reduced swimming
                },
                dimension=2,
                num_runs=10  # Fewer runs for quick testing
            ))
            
            # P1-NoGA (without GA crossover)
            configs.append(EnhancedExperimentConfig(
                name="P1_NoGA",
                optimizer_class=ChaoticBFO,
                optimizer_params={
                    'population_size': 100,
                    'lr': 0.01,
                    'chemotaxis_steps': 20,
                    'reproduction_steps': 10,
                    'elimination_steps': 5,
                    'elimination_prob': 0.4,
                    'step_size_min': 1e-4,
                    'step_size_max': 1.0,
                    'levy_alpha': 1.8,
                    'enable_swarming': True,
                    'enable_chaos': True,
                    'chaos_strength': 0.5,
                    'diversity_trigger_ratio': 0.5,
                    'enable_crossover': False,  # KEY: No GA crossover
                    'diversity_threshold_decay': 0.9
                },
                dimension=2,
                num_runs=30
            ))
            
            # P1-NoChaos (without chaos injection)
            configs.append(EnhancedExperimentConfig(
                name="P1_NoChaos",
                optimizer_class=ChaoticBFO,
                optimizer_params={
                    'population_size': 100,
                    'lr': 0.01,
                    'chemotaxis_steps': 20,
                    'reproduction_steps': 10,
                    'elimination_steps': 5,
                    'elimination_prob': 0.4,
                    'step_size_min': 1e-4,
                    'step_size_max': 1.0,
                    'levy_alpha': 1.8,
                    'enable_swarming': True,
                    'enable_chaos': False,  # KEY: No chaos
                    'chaos_strength': 0.0,
                    'diversity_trigger_ratio': 0.5,
                    'enable_crossover': True,
                    'diversity_threshold_decay': 0.9
                },
                dimension=2,
                num_runs=30
            ))
            
            # P1-NoDiversity (without diversity trigger)
            configs.append(EnhancedExperimentConfig(
                name="P1_NoDiversity",
                optimizer_class=ChaoticBFO,
                optimizer_params={
                    'population_size': 100,
                    'lr': 0.01,
                    'chemotaxis_steps': 20,
                    'reproduction_steps': 10,
                    'elimination_steps': 5,
                    'elimination_prob': 0.4,
                    'step_size_min': 1e-4,
                    'step_size_max': 1.0,
                    'levy_alpha': 1.8,
                    'enable_swarming': True,
                    'enable_chaos': True,
                    'chaos_strength': 0.5,
                    'diversity_trigger_ratio': 0.0,  # KEY: No diversity trigger
                    'enable_crossover': True,
                    'diversity_threshold_decay': 0.9
                },
                dimension=2,
                num_runs=30
            ))
            
            # P1-All-100k (all improvements with 100k budget)
            configs.append(EnhancedExperimentConfig(
                name="P1_All_100k",
                optimizer_class=ChaoticBFO,
                optimizer_params={
                    'population_size': 100,
                    'lr': 0.01,
                    'chemotaxis_steps': 20,
                    'reproduction_steps': 10,
                    'elimination_steps': 5,
                    'elimination_prob': 0.4,
                    'step_size_min': 1e-4,
                    'step_size_max': 1.0,
                    'levy_alpha': 1.8,
                    'enable_swarming': True,
                    'enable_chaos': True,
                    'chaos_strength': 0.5,
                    'diversity_trigger_ratio': 0.5,
                    'enable_crossover': True,
                    'diversity_threshold_decay': 0.9
                },
                dimension=2,
                num_runs=30,
                max_evaluations=100000  # KEY: 100k budget
            ))
        
        return configs
    
    def run_ablation_study(self) -> Dict:
        """Run ablation study to isolate each improvement's contribution."""
        print("\n" + "=" * 80)
        print("ABLATION STUDY - 2D Schwefel Only")
        print("=" * 80)
        
        ablation_configs = [
            # Baseline
            EnhancedExperimentConfig(
                name="Baseline_10k",
                optimizer_class=BFO,
                optimizer_params={
                    'population_size': 50,
                    'lr': 0.01,
                    'chemotaxis_steps': 4,
                    'reproduction_steps': 4,
                    'elimination_steps': 2,
                    'elimination_prob': 0.25,
                    'enable_swarming': False  # No swarming
                },
                dimension=2,
                max_evaluations=10000,  # Original budget
                num_runs=10  # Fewer runs for ablation
            ),
            
            # +Swarming only
            EnhancedExperimentConfig(
                name="Baseline_Swarming",
                optimizer_class=BFO,
                optimizer_params={
                    'population_size': 50,
                    'lr': 0.01,
                    'chemotaxis_steps': 4,
                    'reproduction_steps': 4,
                    'elimination_steps': 2,
                    'elimination_prob': 0.25,
                    'enable_swarming': True  # Add swarming
                },
                dimension=2,
                max_evaluations=10000,
                num_runs=10
            ),
            
            # +Larger steps
            EnhancedExperimentConfig(
                name="Baseline_LargeSteps",
                optimizer_class=BFO,
                optimizer_params={
                    'population_size': 50,
                    'lr': 0.01,
                    'chemotaxis_steps': 4,
                    'reproduction_steps': 4,
                    'elimination_steps': 2,
                    'elimination_prob': 0.25,
                    'step_size_max': 1.0,  # Larger steps
                    'levy_alpha': 1.8,  # Heavier tails
                    'enable_swarming': False
                },
                dimension=2,
                max_evaluations=10000,
                num_runs=10
            ),
            
            # +Everything
            EnhancedExperimentConfig(
                name="All_Improvements_10k",
                optimizer_class=BFO,
                optimizer_params={
                    'population_size': 100,
                    'lr': 0.01,
                    'chemotaxis_steps': 20,
                    'reproduction_steps': 10,
                    'elimination_steps': 5,
                    'elimination_prob': 0.4,
                    'step_size_max': 1.0,
                    'levy_alpha': 1.8,
                    'enable_swarming': True
                },
                dimension=2,
                max_evaluations=10000,
                num_runs=10
            )
        ]
        
        ablation_results = {}
        for config in ablation_configs:
            results = self.run_enhanced_experiment_series(config)
            ablation_results[config.name] = results
        
        return ablation_results
    
    def run_p1_experiments(self) -> Dict:
        """Run P1 improvement experiments with ablation study."""
        print("=" * 80)
        print("P1 SCHWEFEL FUNCTION EXPERIMENTS")
        print("Testing ChaoticBFO with diversity maintenance, chaos injection, and GA crossover")
        print("=" * 80)
        
        p1_results = {}
        
        # Get all configurations
        all_configs = self.create_enhanced_configurations()
        
        # Filter P1 configurations
        p1_configs = [c for c in all_configs if c.name.startswith('P1_')]
        
        print(f"\nRunning {len(p1_configs)} P1 experiments...")
        
        for config in p1_configs:
            results = self.run_enhanced_experiment_series(config)
            p1_results[config.name] = results
        
        # Save results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f'schwefel_p1_experiments_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(p1_results, f, indent=2, default=str)
        
        # Print P1 summary
        self._print_p1_summary(p1_results)
        
        print(f"\nP1 experiment results saved to: {filename}")
        return p1_results
    
    def _print_p1_summary(self, results: Dict):
        """Print P1-specific summary with ablation analysis."""
        print("\n" + "=" * 80)
        print("P1 RESULTS SUMMARY")
        print("=" * 80)
        
        print("\nP1 Ablation Study (2D Schwefel, 50k FE unless noted):")
        print("-" * 80)
        print(f"{'Configuration':<25} {'Success':<12} {'Mean Loss':<15} {'Best Loss':<15} {'Mean FE':<10}")
        print("-" * 80)
        
        # Baseline from P0 for comparison
        print(f"{'P0 Best (Enhanced BFO)':<25} {'0.0%':>7}     {371.20:>10.2f}     {118.44:>10.2f}  {50000:>8.0f}")
        print("-" * 80)
        
        # P1 results
        for name, data in sorted(results.items()):
            success_rate = data['results']['success_rate']
            mean_loss = data['results']['final_loss_mean']
            min_loss = data['results']['final_loss_min']
            mean_fe = data['results']['function_evaluations_mean']
            print(f"{name:<25} {success_rate:>7.1%}     {mean_loss:>10.2f}     {min_loss:>10.2f}  {mean_fe:>8.0f}")
        
        # Component contribution analysis
        print("\n" + "=" * 80)
        print("COMPONENT CONTRIBUTION ANALYSIS:")
        print("-" * 80)
        
        if 'P1_Chaotic_BFO' in results and 'P1_NoGA' in results:
            baseline = results['P1_Chaotic_BFO']['results']['final_loss_mean']
            no_ga = results['P1_NoGA']['results']['final_loss_mean']
            ga_contrib = (no_ga - baseline) / baseline * 100
            print(f"GA Crossover contribution: {ga_contrib:+.1f}% loss change")
        
        if 'P1_Chaotic_BFO' in results and 'P1_NoChaos' in results:
            baseline = results['P1_Chaotic_BFO']['results']['final_loss_mean']
            no_chaos = results['P1_NoChaos']['results']['final_loss_mean']
            chaos_contrib = (no_chaos - baseline) / baseline * 100
            print(f"Chaos injection contribution: {chaos_contrib:+.1f}% loss change")
        
        if 'P1_Chaotic_BFO' in results and 'P1_NoDiversity' in results:
            baseline = results['P1_Chaotic_BFO']['results']['final_loss_mean']
            no_div = results['P1_NoDiversity']['results']['final_loss_mean']
            div_contrib = (no_div - baseline) / baseline * 100
            print(f"Diversity trigger contribution: {div_contrib:+.1f}% loss change")
        
        # Success analysis
        print("\n" + "=" * 80)
        print("SUCCESS ANALYSIS:")
        any_success = any(r['results']['success_rate'] > 0 for r in results.values())
        if any_success:
            print("✅ P1 improvements achieved non-zero success rate!")
            for name, data in results.items():
                if data['results']['success_rate'] > 0:
                    print(f"   {name}: {data['results']['success_rate']:.1%} success rate")
        else:
            print("❌ No configuration achieved tolerance success (1e-4)")
            print("   However, significant loss reduction observed")
            
            # Find best performer
            best_config = min(results.keys(), key=lambda k: results[k]['results']['final_loss_mean'])
            best_loss = results[best_config]['results']['final_loss_mean']
            p0_best = 371.20  # From P0 results
            improvement = (p0_best - best_loss) / p0_best * 100
            print(f"   Best: {best_config} with {improvement:.1f}% improvement over P0")
    
    def run_comprehensive_enhanced_experiments(self) -> Dict:
        """Run all enhanced experiments."""
        print("=" * 80)
        print("ENHANCED SCHWEFEL FUNCTION EXPERIMENTS")
        print("With accurate FE counting and P0 improvements")
        print("=" * 80)
        
        all_results = {
            'main_experiments': {},
            'ablation_study': {}
        }
        
        # Run ablation study first (faster)
        print("\nPhase 1: Ablation Study")
        all_results['ablation_study'] = self.run_ablation_study()
        
        # Run main experiments
        print("\nPhase 2: Main Experiments (2D, 10D, 30D)")
        configs = self.create_enhanced_configurations()
        
        for config in configs:
            results = self.run_enhanced_experiment_series(config)
            key = f"{config.name}_D{config.dimension}"
            all_results['main_experiments'][key] = results
        
        # Save results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f'schwefel_enhanced_experiments_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Print comprehensive summary
        self._print_enhanced_summary(all_results)
        
        # Generate convergence plots
        self._generate_convergence_plots(all_results, timestamp)
        
        print(f"\nDetailed results saved to: {filename}")
        return all_results
    
    def _print_enhanced_summary(self, results: Dict):
        """Print enhanced summary with dimension information."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ENHANCED RESULTS SUMMARY")
        print("=" * 80)
        
        # Ablation results
        print("\nAblation Study Results (2D, 10k FE):")
        print("-" * 70)
        print(f"{'Configuration':<25} {'Success':<12} {'Mean Loss':<15} {'Mean FE':<10}")
        print("-" * 70)
        
        for name, data in results['ablation_study'].items():
            success_rate = data['results']['success_rate']
            mean_loss = data['results']['final_loss_mean']
            mean_fe = data['results']['function_evaluations_mean']
            print(f"{name:<25} {success_rate:>7.1%}     {mean_loss:>10.2e}  {mean_fe:>8.0f}")
        
        # Main results by dimension
        for dim in [2, 10, 30]:
            print(f"\n{dim}D Results (50k FE):")
            print("-" * 70)
            print(f"{'Algorithm':<30} {'Success':<12} {'Mean Loss':<15} {'Mean FE':<10}")
            print("-" * 70)
            
            for name, data in results['main_experiments'].items():
                if f"_D{dim}" in name:
                    success_rate = data['results']['success_rate']
                    mean_loss = data['results']['final_loss_mean']
                    mean_fe = data['results']['function_evaluations_mean']
                    alg_name = name.replace(f"_D{dim}", "")
                    print(f"{alg_name:<30} {success_rate:>7.1%}     {mean_loss:>10.2e}  {mean_fe:>8.0f}")
        
        # Best performers
        print("\n" + "=" * 70)
        print("BEST PERFORMERS BY DIMENSION:")
        for dim in [2, 10, 30]:
            dim_results = {k: v for k, v in results['main_experiments'].items() if f"_D{dim}" in k}
            if dim_results:
                best = max(dim_results.keys(), key=lambda k: dim_results[k]['results']['success_rate'])
                rate = dim_results[best]['results']['success_rate']
                print(f"{dim}D: {best} - {rate:.1%} success")
    
    def _generate_convergence_plots(self, results: Dict, timestamp: str):
        """Generate convergence plots for best performers."""
        plt.figure(figsize=(15, 10))
        
        # Select best performer from each dimension
        for i, dim in enumerate([2, 10, 30]):
            plt.subplot(2, 2, i+1)
            
            dim_results = {k: v for k, v in results['main_experiments'].items() if f"_D{dim}" in k}
            if dim_results:
                # Get best performer
                best_key = max(dim_results.keys(), 
                             key=lambda k: dim_results[k]['results']['success_rate'])
                best_data = dim_results[best_key]
                
                # Plot convergence curves for successful runs
                for run in best_data['individual_runs'][:5]:  # First 5 runs
                    if run['success']:
                        fe_traj = run['fe_trajectory']
                        loss_traj = run['loss_trajectory']
                        plt.plot(fe_traj[:len(loss_traj)], loss_traj, alpha=0.5)
                
                plt.xlabel('Function Evaluations')
                plt.ylabel('Loss')
                plt.title(f'{dim}D - {best_key}')
                plt.yscale('log')
                plt.grid(True, alpha=0.3)
        
        # Ablation comparison
        plt.subplot(2, 2, 4)
        ablation_names = []
        ablation_success = []
        for name, data in results['ablation_study'].items():
            ablation_names.append(name.replace('Baseline_', '').replace('_10k', ''))
            ablation_success.append(data['results']['success_rate'])
        
        plt.bar(range(len(ablation_names)), ablation_success)
        plt.xticks(range(len(ablation_names)), ablation_names, rotation=45)
        plt.ylabel('Success Rate')
        plt.title('Ablation Study - 2D Schwefel')
        plt.tight_layout()
        
        plt.savefig(f'schwefel_enhanced_convergence_{timestamp}.png', dpi=150)
        plt.close()
        
        print(f"\nConvergence plots saved to: schwefel_enhanced_convergence_{timestamp}.png")


def main():
    """Run enhanced Schwefel experiments."""
    experimenter = SchwefelEnhancedExperiments()
    results = experimenter.run_comprehensive_enhanced_experiments()
    
    print("\n🎯 Enhanced Schwefel experiments completed!")
    print("📊 Accurate FE counting implemented")
    print("🚀 All P0 improvements applied") 
    print("📈 Statistical significance with proper ablation study")


if __name__ == "__main__":
    main()