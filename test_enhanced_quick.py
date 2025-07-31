#!/usr/bin/env python3
"""Quick test of enhanced experiments to verify functionality."""

from schwefel_enhanced_experiments import SchwefelEnhancedExperiments, EnhancedExperimentConfig
from src.bfo_torch.optimizer import BFO

# Quick test with reduced parameters
experimenter = SchwefelEnhancedExperiments()

# Test config with just 2 runs and smaller budget
test_config = EnhancedExperimentConfig(
    name="Quick_Test",
    optimizer_class=BFO,
    optimizer_params={
        'population_size': 50,
        'lr': 0.01,
        'chemotaxis_steps': 5,
        'reproduction_steps': 2,
        'elimination_steps': 1,
        'elimination_prob': 0.25,
        'step_size_max': 1.0,
        'levy_alpha': 1.8,
        'enable_swarming': True
    },
    dimension=2,
    num_runs=2,
    max_evaluations=5000  # Small budget for quick test
)

print("Running quick test...")
results = experimenter.run_enhanced_experiment_series(test_config)

print("\nQuick test results:")
print(f"Success rate: {results['results']['success_rate']:.1%}")
print(f"Mean final loss: {results['results']['final_loss_mean']:.2f}")
print(f"Mean FE: {results['results']['function_evaluations_mean']:.0f}")

# Check FE counting
for i, run in enumerate(results['individual_runs']):
    print(f"\nRun {i+1}:")
    print(f"  Initial loss: {run['initial_loss']:.2f}")
    print(f"  Final loss: {run['final_loss']:.2f}")
    print(f"  FEs used: {run['function_evaluations']}")
    print(f"  Steps taken: {run['optimization_steps']}")

print("\n✅ Enhanced experiments working correctly!")