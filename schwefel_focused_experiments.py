#!/usr/bin/env python3
"""
Focused Schwefel Experiments - Demonstrating Key Improvements
=============================================================

A focused set of experiments to demonstrate:
1. Proper FE counting vs previous overcounting
2. Effect of swarming and enhanced parameters
3. 2D performance with 50k budget
4. Quick comparison across key algorithms
"""

import numpy as np
import torch
import torch.nn as nn
import time
import json
import statistics
from typing import Dict, List
from dataclasses import dataclass

from src.bfo_torch.optimizer import BFO, AdaptiveBFO, HybridBFO
from schwefel_enhanced_experiments import SchwefelEnhancedExperiments, EnhancedExperimentConfig


def run_focused_experiments():
    """Run focused experiments demonstrating key improvements."""
    
    experimenter = SchwefelEnhancedExperiments()
    results = {}
    
    print("=" * 80)
    print("FOCUSED SCHWEFEL EXPERIMENTS - KEY IMPROVEMENTS")
    print("=" * 80)
    
    # 1. Demonstrate FE counting difference
    print("\n1. FUNCTION EVALUATION COUNTING COMPARISON")
    print("-" * 50)
    
    # Old-style counting (1 FE per step)
    old_style_fe = 0
    steps = 10
    pop_size = 50
    chem_steps = 4
    
    for _ in range(steps):
        old_style_fe += 1  # Old way: count step as 1 FE
    
    # Actual FEs
    actual_fe = steps * pop_size * chem_steps  # Minimum, could be more with swimming
    
    print(f"Old counting method: {old_style_fe} FEs after {steps} steps")
    print(f"Actual FEs (minimum): {actual_fe} FEs")
    print(f"Undercounting factor: {actual_fe/old_style_fe:.1f}x")
    
    # 2. Key configurations for 2D Schwefel
    configs = [
        # Baseline - original parameters
        EnhancedExperimentConfig(
            name="Original_Baseline",
            optimizer_class=BFO,
            optimizer_params={
                'population_size': 50,
                'lr': 0.01,
                'chemotaxis_steps': 4,
                'reproduction_steps': 4,
                'elimination_steps': 2,
                'elimination_prob': 0.25,
                'step_size_max': 0.1,  # Original small steps
                'levy_alpha': 1.5,  # Original
                'enable_swarming': False  # No swarming
            },
            dimension=2,
            num_runs=10,
            max_evaluations=50000
        ),
        
        # Enhanced with all improvements
        EnhancedExperimentConfig(
            name="Enhanced_All_Improvements",
            optimizer_class=BFO,
            optimizer_params={
                'population_size': 100,
                'lr': 0.01,
                'chemotaxis_steps': 20,
                'reproduction_steps': 10,
                'elimination_steps': 5,
                'elimination_prob': 0.4,
                'step_size_max': 1.0,  # 10x larger
                'levy_alpha': 1.8,  # Heavier tails
                'enable_swarming': True  # With swarming
            },
            dimension=2,
            num_runs=10,
            max_evaluations=50000
        ),
        
        # HybridBFO with gradient
        EnhancedExperimentConfig(
            name="Hybrid_Gradient_70",
            optimizer_class=HybridBFO,
            optimizer_params={
                'population_size': 80,
                'lr': 0.01,
                'gradient_weight': 0.7,
                'chemotaxis_steps': 10,
                'reproduction_steps': 5,
                'elimination_steps': 3,
                'elimination_prob': 0.3,
                'step_size_max': 0.5,
                'levy_alpha': 1.6,
                'enable_swarming': True
            },
            dimension=2,
            num_runs=10,
            max_evaluations=50000,
            use_gradient=True
        )
    ]
    
    # Run experiments
    print("\n2. RUNNING FOCUSED EXPERIMENTS (2D, 50k FE budget)")
    print("-" * 50)
    
    for config in configs:
        result = experimenter.run_enhanced_experiment_series(config)
        results[config.name] = result
    
    # 3. Summary comparison
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY - 2D SCHWEFEL WITH 50K FE BUDGET")
    print("=" * 80)
    print(f"{'Algorithm':<30} {'Success':<10} {'Mean Loss':<15} {'Best Loss':<15} {'Mean FE':<10}")
    print("-" * 80)
    
    for name, data in results.items():
        success_rate = data['results']['success_rate']
        mean_loss = data['results']['final_loss_mean']
        best_loss = data['results']['final_loss_min']
        mean_fe = data['results']['function_evaluations_mean']
        print(f"{name:<30} {success_rate:>6.1%}    {mean_loss:>11.2f}    {best_loss:>11.2f}    {mean_fe:>8.0f}")
    
    # 4. Improvement analysis
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)
    
    baseline = results.get("Original_Baseline", {}).get('results', {})
    enhanced = results.get("Enhanced_All_Improvements", {}).get('results', {})
    
    if baseline and enhanced:
        baseline_success = baseline['success_rate']
        enhanced_success = enhanced['success_rate']
        
        baseline_loss = baseline['final_loss_mean']
        enhanced_loss = enhanced['final_loss_mean']
        
        print(f"Success Rate Improvement: {baseline_success:.1%} → {enhanced_success:.1%}")
        if baseline_success > 0:
            print(f"  Relative improvement: {(enhanced_success/baseline_success - 1)*100:.1f}%")
        else:
            print(f"  Absolute improvement: {enhanced_success:.1%}")
        
        print(f"\nMean Loss Improvement: {baseline_loss:.2f} → {enhanced_loss:.2f}")
        print(f"  Reduction: {baseline_loss - enhanced_loss:.2f} ({(1 - enhanced_loss/baseline_loss)*100:.1f}%)")
        
        # Check if we achieved target
        print(f"\nTarget Achievement (≥20% success on 2D):")
        if enhanced_success >= 0.2:
            print(f"  ✅ SUCCESS: {enhanced_success:.1%} exceeds 20% target!")
        else:
            print(f"  ❌ Not yet: {enhanced_success:.1%} < 20% target")
            print(f"  Recommendation: Proceed to P1 improvements (diversity triggers)")
    
    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f'schwefel_focused_results_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Results saved to: {filename}")
    
    # 5. Best run analysis
    print("\n" + "=" * 80)
    print("BEST RUN ANALYSIS")
    print("=" * 80)
    
    for name, data in results.items():
        runs = data['individual_runs']
        best_run = min(runs, key=lambda r: r['final_loss'])
        
        print(f"\n{name}:")
        print(f"  Best loss: {best_run['final_loss']:.6f}")
        print(f"  FEs used: {best_run['function_evaluations']}")
        print(f"  Success: {'✅' if best_run['success'] else '❌'}")
        print(f"  Final position: [{best_run['final_position'][0]:.2f}, {best_run['final_position'][1]:.2f}]")
        
        # Distance from global optimum
        opt_pos = np.array([420.9687, 420.9687])
        final_pos = np.array(best_run['final_position'])
        distance = np.linalg.norm(final_pos - opt_pos)
        print(f"  Distance from optimum: {distance:.2f}")


if __name__ == "__main__":
    run_focused_experiments()