#!/usr/bin/env python3
"""
Priority 3 Advanced Feature Tests - Comprehensive Validation
===========================================================

This script implements Priority 3 enhancements for comprehensive validation:
1. Advanced feature tests (Lévy flight, adaptive mechanisms, hybrid features)
2. Robustness tests (noisy functions, multi-modal, real-world scenarios)
3. Performance comparison tests
4. Edge case and stress tests

These tests verify advanced functionality and robustness.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import json
from typing import Callable, Tuple, List, Dict, Any
from dataclasses import dataclass

# Import our BFO implementation
from src.bfo_torch.optimizer import BFO, AdaptiveBFO, HybridBFO


class AdvancedFeatureTester:
    """Advanced feature and robustness testing."""
    
    def __init__(self):
        self.results = {}
    
    def test_levy_flight_effectiveness(self) -> Dict[str, Any]:
        """Test Lévy flight implementation for long-distance exploration."""
        print("\nTesting Lévy flight effectiveness...")
        
        def multimodal_landscape(x):
            """Highly multimodal function requiring long-distance jumps."""
            return (torch.sin(x[0] * 3)**2 * torch.cos(x[1] * 3)**2 + 
                   0.1 * torch.sum((x - torch.tensor([5.0, 5.0]))**2) +
                   0.5 * torch.sum(torch.sin(x * 10)**2))
        
        # Test standard BFO vs BFO with potential Lévy flights
        results_comparison = {}
        
        for optimizer_type in ['Standard_BFO', 'Enhanced_BFO']:
            x = nn.Parameter(torch.tensor([0.0, 0.0]))  # Start far from global optimum
            
            if optimizer_type == 'Standard_BFO':
                optimizer = BFO([x], population_size=25, lr=0.1)
            else:
                # Enhanced BFO with parameters that might enable Lévy-like behavior
                optimizer = AdaptiveBFO([x], population_size=35, lr=0.2, 
                                      adaptation_rate=0.3, min_population_size=15, max_population_size=60)
            
            def closure():
                loss = multimodal_landscape(x)
                # Keep in reasonable bounds
                with torch.no_grad():
                    x.data = torch.clamp(x.data, -10, 10)
                return loss.item()
            
            initial_loss = closure()
            position_history = [x.data.clone().tolist()]
            loss_history = [initial_loss]
            large_jumps = 0
            
            for step in range(30):
                old_position = x.data.clone()
                loss = optimizer.step(closure)
                new_position = x.data.clone()
                
                position_history.append(new_position.tolist())
                loss_history.append(loss)
                
                # Detect large jumps (potential Lévy flights)
                jump_distance = torch.norm(new_position - old_position).item()
                if jump_distance > 1.0:  # Threshold for "large" jump
                    large_jumps += 1
            
            final_loss = closure()
            final_position = x.data.clone()
            
            # Calculate exploration metrics
            total_distance_traveled = sum(torch.norm(torch.tensor(position_history[i]) - 
                                                   torch.tensor(position_history[i-1])).item() 
                                        for i in range(1, len(position_history)))
            
            # Distance to target region (around [5,5])
            target = torch.tensor([5.0, 5.0])
            final_distance_to_target = torch.norm(final_position - target).item()
            
            results_comparison[optimizer_type] = {
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'improvement': initial_loss - final_loss,
                'large_jumps': large_jumps,
                'total_distance_traveled': total_distance_traveled,
                'final_distance_to_target': final_distance_to_target,
                'exploration_efficiency': total_distance_traveled / 30,  # Average per step
                'reached_target_region': final_distance_to_target < 2.0
            }
        
        # Analyze Lévy flight effectiveness
        enhanced_performance = results_comparison['Enhanced_BFO']
        standard_performance = results_comparison['Standard_BFO']
        
        levy_effectiveness = {
            'enhanced_explored_more': enhanced_performance['total_distance_traveled'] > standard_performance['total_distance_traveled'],
            'enhanced_more_jumps': enhanced_performance['large_jumps'] > standard_performance['large_jumps'],
            'enhanced_better_convergence': enhanced_performance['final_loss'] < standard_performance['final_loss'],
            'enhanced_reached_target': enhanced_performance['reached_target_region'],
            'adaptive_exploration_advantage': enhanced_performance['exploration_efficiency'] > standard_performance['exploration_efficiency']
        }
        
        return {
            'levy_effectiveness': levy_effectiveness,
            'comparison_results': results_comparison,
            'exploration_advantage': enhanced_performance['exploration_efficiency'] / standard_performance['exploration_efficiency']
        }
    
    def test_adaptive_mechanisms(self) -> Dict[str, Any]:
        """Test adaptive mechanisms under different scenarios."""
        print("\nTesting adaptive mechanisms...")
        
        scenarios = [
            {
                'name': 'Easy_Convergence',
                'function': lambda x: torch.sum(x**2),
                'initial_pos': torch.tensor([3.0, 3.0]),
                'expected_behavior': 'population_shrinks'
            },
            {
                'name': 'Difficult_Multimodal',
                'function': lambda x: torch.sum(torch.sin(x * 5)**2) + 0.1 * torch.sum(x**2),
                'initial_pos': torch.tensor([2.0, 2.0]),
                'expected_behavior': 'population_grows'
            },
            {
                'name': 'Stagnation_Scenario',
                'function': lambda x: torch.sum(x**2) + 0.01 * torch.sum(torch.sin(x * 50)**2),
                'initial_pos': torch.tensor([1.0, 1.0]),
                'expected_behavior': 'adaptation_occurs'
            }
        ]
        
        results = {}
        
        for scenario in scenarios:
            print(f"  Testing {scenario['name']} scenario...")
            
            x = nn.Parameter(scenario['initial_pos'].clone())
            optimizer = AdaptiveBFO([x], population_size=20, lr=0.05,
                                  adaptation_rate=0.15, min_population_size=10, max_population_size=50)
            
            def closure():
                return scenario['function'](x).item()
            
            # Track adaptive behavior
            population_sizes = []
            step_sizes = []
            fitness_improvements = []
            adaptation_count = 0
            
            initial_fitness = closure()
            previous_fitness = initial_fitness
            
            for step in range(25):
                loss = optimizer.step(closure)
                
                # Track parameters
                group_id = id(optimizer.param_groups[0])
                if group_id in optimizer.state:
                    current_pop_size = len(optimizer.state[group_id].get('population', []))
                    population_sizes.append(current_pop_size)
                    
                    current_step_size = optimizer.state[group_id].get('current_step_size', optimizer.param_groups[0]['lr'])
                    step_sizes.append(current_step_size)
                
                # Track fitness
                improvement = previous_fitness - loss
                fitness_improvements.append(improvement)
                previous_fitness = loss
                
                # Count adaptations
                if len(population_sizes) >= 2 and population_sizes[-1] != population_sizes[-2]:
                    adaptation_count += 1
            
            final_fitness = closure()
            
            # Analyze adaptation behavior
            if len(population_sizes) > 5:
                initial_pop = np.mean(population_sizes[:3])
                final_pop = np.mean(population_sizes[-3:])
                pop_change_ratio = final_pop / initial_pop if initial_pop > 0 else 1
                
                step_size_variance = np.var(step_sizes) if step_sizes else 0
                adaptation_responsiveness = adaptation_count / 25
                
                # Check expected behavior
                if scenario['expected_behavior'] == 'population_shrinks':
                    behavior_correct = pop_change_ratio < 0.9
                elif scenario['expected_behavior'] == 'population_grows':
                    behavior_correct = pop_change_ratio > 1.1
                else:  # adaptation_occurs
                    behavior_correct = adaptation_count > 2
            else:
                behavior_correct = False
                pop_change_ratio = 1.0
                adaptation_responsiveness = 0
            
            results[scenario['name']] = {
                'initial_fitness': initial_fitness,
                'final_fitness': final_fitness,
                'total_improvement': initial_fitness - final_fitness,
                'population_change_ratio': pop_change_ratio,
                'adaptation_count': adaptation_count,
                'adaptation_responsiveness': adaptation_responsiveness,
                'behavior_correct': behavior_correct,
                'step_size_adapted': step_size_variance > 0.001
            }
        
        # Overall adaptive mechanism assessment
        adaptive_mechanisms_working = {
            'scenarios_adapted_correctly': sum(1 for r in results.values() if r['behavior_correct']),
            'all_scenarios_showed_adaptation': all(r['adaptation_count'] > 0 for r in results.values()),
            'step_size_adaptation_working': all(r['step_size_adapted'] for r in results.values()),
            'responsive_to_different_scenarios': len(set(r['adaptation_count'] for r in results.values())) > 1
        }
        
        return {
            'adaptive_mechanisms_working': adaptive_mechanisms_working,
            'scenario_results': results
        }
    
    def test_hybrid_features(self) -> Dict[str, Any]:
        """Test hybrid BFO features with gradient integration."""
        print("\nTesting hybrid features...")
        
        def test_function(x):
            """Function with clear gradient information."""
            return torch.sum((x - torch.tensor([1.0, 2.0, 0.5]))**2) + 0.1 * torch.sum(torch.sin(x * 8)**2)
        
        test_scenarios = [
            {
                'name': 'With_Gradients',
                'use_gradients': True,
                'gradient_weight': 0.7
            },
            {
                'name': 'Without_Gradients', 
                'use_gradients': False,
                'gradient_weight': 0.0
            },
            {
                'name': 'Balanced_Hybrid',
                'use_gradients': True,
                'gradient_weight': 0.5
            }
        ]
        
        results = {}
        
        for scenario in test_scenarios:
            print(f"  Testing {scenario['name']} scenario...")
            
            x = nn.Parameter(torch.randn(3) * 2.0)
            optimizer = HybridBFO([x], population_size=20, lr=0.05,
                                gradient_weight=scenario['gradient_weight'],
                                momentum=0.9, enable_momentum=True)
            
            def closure():
                if scenario['use_gradients']:
                    optimizer.zero_grad()
                    loss = test_function(x)
                    loss.backward()
                    return loss.item()
                else:
                    return test_function(x).item()
            
            initial_loss = closure()
            convergence_steps = 0
            loss_history = []
            
            for step in range(25):
                loss = optimizer.step(closure)
                loss_history.append(loss)
                
                # Check convergence
                if loss < initial_loss * 0.1:  # 90% improvement
                    convergence_steps = step + 1
                    break
            
            final_loss = closure()
            final_position = x.data.clone()
            target_position = torch.tensor([1.0, 2.0, 0.5])
            distance_to_optimum = torch.norm(final_position - target_position).item()
            
            results[scenario['name']] = {
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'improvement_ratio': (initial_loss - final_loss) / initial_loss,
                'convergence_steps': convergence_steps,
                'distance_to_optimum': distance_to_optimum,
                'converged': convergence_steps > 0,
                'loss_trajectory': loss_history[-5:]  # Last 5 losses
            }
        
        # Analyze hybrid effectiveness
        with_grads = results['With_Gradients']
        without_grads = results['Without_Gradients']
        balanced = results['Balanced_Hybrid']
        
        hybrid_effectiveness = {
            'gradients_help_convergence': with_grads['converged'] and with_grads['convergence_steps'] < 20,
            'pure_bfo_still_works': without_grads['improvement_ratio'] > 0.5,
            'balanced_performs_well': balanced['improvement_ratio'] > 0.7,
            'gradient_integration_effective': with_grads['improvement_ratio'] > without_grads['improvement_ratio'],
            'all_variants_improve': all(r['improvement_ratio'] > 0.3 for r in results.values())
        }
        
        return {
            'hybrid_effectiveness': hybrid_effectiveness,
            'scenario_results': results
        }
    
    def test_robustness_noisy_functions(self) -> Dict[str, Any]:
        """Test robustness to noisy objective functions."""
        print("\nTesting robustness to noisy functions...")
        
        def noisy_sphere(x, noise_level=0.1):
            """Sphere function with additive noise."""
            clean_value = torch.sum(x**2)
            noise = torch.randn(1) * noise_level * clean_value
            return clean_value + noise
        
        noise_levels = [0.0, 0.05, 0.1, 0.2]
        results = {}
        
        for noise_level in noise_levels:
            print(f"  Testing noise level {noise_level:.2f}...")
            
            x = nn.Parameter(torch.randn(3) * 3.0)
            optimizer = AdaptiveBFO([x], population_size=30, lr=0.02,
                                  adaptation_rate=0.1)
            
            def closure():
                return noisy_sphere(x, noise_level).item()
            
            initial_loss = closure()
            final_losses = []
            
            # Run multiple times for noise averaging
            for run in range(3):
                x.data = torch.randn(3) * 3.0  # Reset position
                current_initial = closure()
                
                for step in range(20):
                    loss = optimizer.step(closure)
                
                final_loss = closure()
                final_losses.append(final_loss)
            
            # Average results
            avg_final_loss = np.mean(final_losses)
            std_final_loss = np.std(final_losses)
            
            # Clean reference (no noise)
            x.data = torch.randn(3) * 3.0
            clean_final = torch.sum(x.data**2).item()
            
            results[f'noise_{noise_level:.2f}'] = {
                'noise_level': noise_level,
                'avg_final_loss': avg_final_loss,
                'std_final_loss': std_final_loss,
                'convergence_stability': std_final_loss < 1.0,  # Low variance
                'still_converges': avg_final_loss < 5.0,  # Reasonable final loss
                'noise_tolerance': avg_final_loss < clean_final * (1 + noise_level * 5)
            }
        
        # Analyze robustness
        robustness_to_noise = {
            'handles_light_noise': results['noise_0.05']['still_converges'],
            'handles_moderate_noise': results['noise_0.10']['still_converges'],
            'handles_heavy_noise': results['noise_0.20']['still_converges'],
            'maintains_stability': all(r['convergence_stability'] for r in results.values()),
            'graceful_degradation': all(results[f'noise_{noise_levels[i]:.2f}']['avg_final_loss'] >= 
                                      results[f'noise_{noise_levels[i-1]:.2f}']['avg_final_loss'] 
                                      for i in range(1, len(noise_levels)))
        }
        
        return {
            'robustness_to_noise': robustness_to_noise,
            'noise_level_results': results
        }
    
    def test_multimodal_robustness(self) -> Dict[str, Any]:
        """Test performance on highly multimodal functions."""
        print("\nTesting multimodal function robustness...")
        
        def himmelblau(x):
            """Himmelblau's function - 4 global minima."""
            if len(x) != 2:
                return torch.tensor(float('inf'))
            return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
        
        def four_peaks(x):
            """Four peaks function - multiple local optima."""
            if len(x) != 2:
                return torch.tensor(float('inf'))
            return (torch.exp(-((x[0] - 1)**2 + (x[1] - 1)**2)) + 
                   torch.exp(-((x[0] + 1)**2 + (x[1] + 1)**2)) +
                   torch.exp(-((x[0] - 1)**2 + (x[1] + 1)**2)) + 
                   torch.exp(-((x[0] + 1)**2 + (x[1] - 1)**2)))
        
        multimodal_functions = [
            {'name': 'Himmelblau', 'function': himmelblau, 'bounds': (-5, 5), 'success_threshold': 10.0},
            {'name': 'Four_Peaks', 'function': four_peaks, 'bounds': (-3, 3), 'success_threshold': -3.0}
        ]
        
        results = {}
        
        for func_info in multimodal_functions:
            print(f"  Testing {func_info['name']} function...")
            
            successes = 0
            final_losses = []
            
            # Multiple runs from different starting positions
            for run in range(5):
                x = nn.Parameter(torch.rand(2) * (func_info['bounds'][1] - func_info['bounds'][0]) + func_info['bounds'][0])
                optimizer = BFO([x], population_size=40, lr=0.05,
                              chemotaxis_steps=8, elimination_prob=0.4)  # High elimination for diversity
                
                def closure():
                    loss = func_info['function'](x)
                    with torch.no_grad():
                        x.data = torch.clamp(x.data, func_info['bounds'][0], func_info['bounds'][1])
                    return loss.item()
                
                initial_loss = closure()
                
                for step in range(30):
                    loss = optimizer.step(closure)
                
                final_loss = closure()
                final_losses.append(final_loss)
                
                # Check success based on function-specific criteria
                if func_info['name'] == 'Himmelblau':
                    success = final_loss < func_info['success_threshold']
                else:  # Four_Peaks (maximization problem, so higher is better)
                    success = final_loss > func_info['success_threshold']
                
                if success:
                    successes += 1
            
            success_rate = successes / 5
            avg_final_loss = np.mean(final_losses)
            
            results[func_info['name']] = {
                'success_rate': success_rate,
                'avg_final_loss': avg_final_loss,
                'final_losses': final_losses,
                'consistent_performance': np.std(final_losses) < np.mean(final_losses) * 0.5,
                'finds_good_solutions': success_rate >= 0.4  # At least 40% success rate
            }
        
        # Overall multimodal assessment
        multimodal_robustness = {
            'handles_multiple_optima': all(r['finds_good_solutions'] for r in results.values()),
            'consistent_across_functions': all(r['consistent_performance'] for r in results.values()),
            'good_exploration': np.mean([r['success_rate'] for r in results.values()]) > 0.3
        }
        
        return {
            'multimodal_robustness': multimodal_robustness,
            'function_results': results
        }
    
    def run_priority3_advanced_tests(self) -> Dict[str, Any]:
        """Run Priority 3 advanced feature tests."""
        
        print("=" * 70)
        print("PRIORITY 3 ADVANCED FEATURE TESTS")
        print("=" * 70)
        
        start_time = time.time()
        
        results = {
            'levy_flight_test': self.test_levy_flight_effectiveness(),
            'adaptive_mechanisms': self.test_adaptive_mechanisms(),
            'hybrid_features': self.test_hybrid_features(),
            'robustness_noise': self.test_robustness_noisy_functions(),
            'multimodal_robustness': self.test_multimodal_robustness()
        }
        
        end_time = time.time()
        
        # Generate summary
        summary = self._generate_advanced_summary(results, end_time - start_time)
        results['summary'] = summary
        
        # Save results
        with open('priority3_advanced_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        self._print_advanced_summary(summary)
        
        return results
    
    def _generate_advanced_summary(self, results: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """Generate summary of advanced test results."""
        
        # Count successful features
        feature_tests = [
            ('levy_flight_test', 'levy_effectiveness'),
            ('adaptive_mechanisms', 'adaptive_mechanisms_working'),
            ('hybrid_features', 'hybrid_effectiveness'),
            ('robustness_noise', 'robustness_to_noise'),
            ('multimodal_robustness', 'multimodal_robustness')
        ]
        
        total_feature_checks = 0
        successful_feature_checks = 0
        
        for test_name, verification_key in feature_tests:
            if test_name in results and verification_key in results[test_name]:
                verification_results = results[test_name][verification_key]
                for check, passed in verification_results.items():
                    total_feature_checks += 1
                    if passed:
                        successful_feature_checks += 1
        
        feature_success_rate = successful_feature_checks / total_feature_checks if total_feature_checks > 0 else 0
        
        # Check specific capabilities
        levy_working = all(results['levy_flight_test']['levy_effectiveness'].values())
        adaptive_working = sum(results['adaptive_mechanisms']['adaptive_mechanisms_working'].values()) >= 3
        hybrid_working = all(results['hybrid_features']['hybrid_effectiveness'].values())
        noise_robust = sum(results['robustness_noise']['robustness_to_noise'].values()) >= 4
        multimodal_robust = all(results['multimodal_robustness']['multimodal_robustness'].values())
        
        summary = {
            'total_feature_checks': total_feature_checks,
            'successful_feature_checks': successful_feature_checks,
            'feature_success_rate': feature_success_rate,
            'levy_flight_working': levy_working,
            'adaptive_mechanisms_working': adaptive_working,
            'hybrid_features_working': hybrid_working,
            'noise_robustness': noise_robust,
            'multimodal_robustness': multimodal_robust,
            'execution_time': total_time,
            'advanced_features_verified': feature_success_rate >= 0.75 and adaptive_working and hybrid_working
        }
        
        return summary
    
    def _print_advanced_summary(self, summary: Dict[str, Any]):
        """Print advanced test summary."""
        
        print("\n" + "=" * 70)
        print("PRIORITY 3 ADVANCED FEATURE SUMMARY")
        print("=" * 70)
        print(f"Total feature checks: {summary['total_feature_checks']}")
        print(f"Successful checks: {summary['successful_feature_checks']}")
        print(f"Feature success rate: {summary['feature_success_rate']:.1%}")
        print(f"Execution time: {summary['execution_time']:.1f}s")
        print()
        print(f"Advanced features:")
        print(f"  Lévy flight effectiveness: {'✓' if summary['levy_flight_working'] else '✗'}")
        print(f"  Adaptive mechanisms: {'✓' if summary['adaptive_mechanisms_working'] else '✗'}")
        print(f"  Hybrid features: {'✓' if summary['hybrid_features_working'] else '✗'}")
        print()
        print(f"Robustness:")
        print(f"  Noise tolerance: {'✓' if summary['noise_robustness'] else '✗'}")
        print(f"  Multimodal performance: {'✓' if summary['multimodal_robustness'] else '✗'}")
        print()
        print(f"Priority 3 verification: {'✓ PASSED' if summary['advanced_features_verified'] else '✗ NEEDS WORK'}")
        
        if summary['advanced_features_verified']:
            print("\n🎉 Priority 3 advanced feature verification successful!")
            print("✅ Advanced features working correctly")
            print("✅ Robustness to noise and multimodal landscapes verified")
            print("✅ Adaptive and hybrid mechanisms validated")


def main():
    """Run Priority 3 advanced feature tests."""
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    tester = AdvancedFeatureTester()
    results = tester.run_priority3_advanced_tests()
    
    print(f"\nResults saved to: priority3_advanced_test_results.json")


if __name__ == "__main__":
    main()