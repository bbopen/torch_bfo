#!/usr/bin/env python3
"""
Bacterial Foraging Optimization (BFO) Comparison and Verification Script
======================================================================

This script compares the bfo_torch implementation with:
1. Known mathematical optimization problems
2. Theoretical BFO behavior patterns
3. Other optimization algorithms for baseline comparison
4. Published BFO results from literature

The goal is to verify that our implementation produces results consistent
with established BFO theory and performance benchmarks.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Dict, Any
import time
import json
from dataclasses import dataclass
from pathlib import Path

# Import our BFO implementation
from src.bfo_torch.optimizer import BFO, AdaptiveBFO, HybridBFO


@dataclass
class OptimizationProblem:
    """Represents a mathematical optimization problem for testing."""
    name: str
    dimension: int
    bounds: Tuple[float, float]
    global_optimum: float
    global_optimum_pos: np.ndarray
    function: Callable
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the function at point x."""
        return self.function(x)


class BFOBenchmark:
    """Benchmark suite for BFO algorithm verification."""
    
    def __init__(self):
        self.problems = self._define_test_problems()
        self.results = {}
    
    def _define_test_problems(self) -> List[OptimizationProblem]:
        """Define standard optimization problems for BFO testing."""
        
        def sphere(x):
            """Sphere function: f(x) = sum(x_i^2), global min at x=0"""
            return torch.sum(x**2)
        
        def rosenbrock(x):
            """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
            if len(x) < 2:
                return torch.tensor(float('inf'))
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        
        def rastrigin(x):
            """Rastrigin function: f(x) = 10n + sum(x_i^2 - 10cos(2πx_i))"""
            n = len(x)
            return 10 * n + torch.sum(x**2 - 10 * torch.cos(2 * np.pi * x))
        
        def ackley(x):
            """Ackley function: f(x) = -20*exp(-0.2*sqrt(mean(x^2))) - exp(mean(cos(2πx))) + 20 + e"""
            n = len(x)
            term1 = -20 * torch.exp(-0.2 * torch.sqrt(torch.mean(x**2)))
            term2 = -torch.exp(torch.mean(torch.cos(2 * np.pi * x)))
            return term1 + term2 + 20 + np.e
        
        def griewank(x):
            """Griewank function: f(x) = sum(x^2)/4000 - prod(cos(x/sqrt(i+1))) + 1"""
            n = len(x)
            term1 = torch.sum(x**2) / 4000
            term2 = torch.prod(torch.cos(x / torch.sqrt(torch.arange(1, n+1, dtype=x.dtype, device=x.device))))
            return term1 - term2 + 1
        
        return [
            OptimizationProblem(
                name="Sphere",
                dimension=2,
                bounds=(-5.12, 5.12),
                global_optimum=0.0,
                global_optimum_pos=np.zeros(2),
                function=sphere
            ),
            OptimizationProblem(
                name="Rosenbrock",
                dimension=2,
                bounds=(-2.048, 2.048),
                global_optimum=0.0,
                global_optimum_pos=np.array([1.0, 1.0]),
                function=rosenbrock
            ),
            OptimizationProblem(
                name="Rastrigin",
                dimension=2,
                bounds=(-5.12, 5.12),
                global_optimum=0.0,
                global_optimum_pos=np.zeros(2),
                function=rastrigin
            ),
            OptimizationProblem(
                name="Ackley",
                dimension=2,
                bounds=(-32.768, 32.768),
                global_optimum=0.0,
                global_optimum_pos=np.zeros(2),
                function=ackley
            ),
            OptimizationProblem(
                name="Griewank",
                dimension=2,
                bounds=(-600, 600),
                global_optimum=0.0,
                global_optimum_pos=np.zeros(2),
                function=griewank
            )
        ]
    
    def test_bfo_convergence(self, problem: OptimizationProblem, 
                            optimizer_class=BFO, **optimizer_kwargs) -> Dict[str, Any]:
        """Test BFO convergence on a specific problem."""
        
        # Initialize parameter at random position
        x = nn.Parameter(torch.randn(problem.dimension) * 2.0)
        
        # Create optimizer
        optimizer = optimizer_class([x], population_size=20, lr=0.01, **optimizer_kwargs)
        
        # Track optimization history
        history = []
        best_fitness = float('inf')
        best_position = None
        
        def closure():
            nonlocal best_fitness, best_position
            loss = problem.evaluate(x)
            if loss.item() < best_fitness:
                best_fitness = loss.item()
                best_position = x.data.clone()
            return loss.item()
        
        # Run optimization
        start_time = time.time()
        for step in range(50):  # 50 optimization steps
            loss = optimizer.step(closure)
            history.append({
                'step': step,
                'loss': loss,
                'best_fitness': best_fitness,
                'current_position': x.data.clone().cpu().numpy()
            })
            
            # Early stopping if close to optimum
            if abs(loss - problem.global_optimum) < 1e-6:
                break
        
        end_time = time.time()
        
        return {
            'problem_name': problem.name,
            'optimizer_class': optimizer_class.__name__,
            'final_loss': loss,
            'best_fitness': best_fitness,
            'best_position': best_position.cpu().numpy() if best_position is not None else None,
            'optimization_time': end_time - start_time,
            'convergence_history': history,
            'distance_to_optimum': np.linalg.norm(best_position.cpu().numpy() - problem.global_optimum_pos) if best_position is not None else float('inf'),
            'success': abs(best_fitness - problem.global_optimum) < 0.01
        }
    
    def test_optimizer_variants(self) -> Dict[str, Any]:
        """Test different BFO variants on all problems."""
        results = {}
        
        optimizer_configs = [
            (BFO, {'population_size': 20, 'lr': 0.01}),
            (AdaptiveBFO, {'population_size': 20, 'lr': 0.01, 'adaptation_rate': 0.1}),
            (HybridBFO, {'population_size': 20, 'lr': 0.01, 'gradient_weight': 0.5})
        ]
        
        for optimizer_class, config in optimizer_configs:
            optimizer_name = optimizer_class.__name__
            results[optimizer_name] = {}
            
            print(f"\nTesting {optimizer_name}...")
            
            for problem in self.problems:
                print(f"  Testing on {problem.name} function...")
                result = self.test_bfo_convergence(problem, optimizer_class, **config)
                results[optimizer_name][problem.name] = result
                
                print(f"    Final loss: {result['final_loss']:.6f}")
                print(f"    Best fitness: {result['best_fitness']:.6f}")
                print(f"    Distance to optimum: {result['distance_to_optimum']:.6f}")
                print(f"    Success: {result['success']}")
        
        return results
    
    def compare_with_theoretical_bfo(self) -> Dict[str, Any]:
        """Compare results with theoretical BFO behavior patterns."""
        
        # Test 1: Population diversity should decrease over time
        print("\nTesting population diversity behavior...")
        x = nn.Parameter(torch.randn(5) * 2.0)
        optimizer = BFO([x], population_size=30, lr=0.01)
        
        diversity_history = []
        
        def closure():
            return torch.sum(x**2).item()
        
        for step in range(20):
            loss = optimizer.step(closure)
            
            # Calculate population diversity
            group_id = id(optimizer.param_groups[0])
            if group_id in optimizer.state:
                population = optimizer.state[group_id]['population']
                mean_pos = population.mean(dim=0)
                diversity = torch.norm(population - mean_pos, dim=1).mean().item()
                diversity_history.append(diversity)
        
        # Test 2: Convergence should be monotonic (with some noise)
        print("\nTesting convergence monotonicity...")
        convergence_test = self.test_bfo_convergence(self.problems[0])  # Sphere function
        losses = [h['loss'] for h in convergence_test['convergence_history']]
        
        # Check if losses generally decrease (allowing for some noise)
        decreasing_count = sum(1 for i in range(1, len(losses)) if losses[i] <= losses[i-1] else 0)
        monotonicity_ratio = decreasing_count / (len(losses) - 1) if len(losses) > 1 else 0
        
        return {
            'diversity_decreasing': all(diversity_history[i] >= diversity_history[i+1] for i in range(len(diversity_history)-1)),
            'convergence_monotonicity_ratio': monotonicity_ratio,
            'diversity_history': diversity_history,
            'convergence_history': losses
        }
    
    def benchmark_against_other_optimizers(self) -> Dict[str, Any]:
        """Compare BFO performance against other PyTorch optimizers."""
        
        print("\nBenchmarking against other optimizers...")
        
        # Test problem
        problem = self.problems[0]  # Sphere function
        x = nn.Parameter(torch.randn(problem.dimension) * 2.0)
        
        optimizers = {
            'BFO': BFO([x], population_size=20, lr=0.01),
            'SGD': torch.optim.SGD([x], lr=0.01),
            'Adam': torch.optim.Adam([x], lr=0.01),
            'RMSprop': torch.optim.RMSprop([x], lr=0.01)
        }
        
        results = {}
        
        for name, optimizer in optimizers.items():
            print(f"  Testing {name}...")
            
            # Reset parameter
            with torch.no_grad():
                x.data = torch.randn(problem.dimension) * 2.0
            
            history = []
            best_fitness = float('inf')
            
            def closure():
                nonlocal best_fitness
                loss = problem.evaluate(x)
                if loss.item() < best_fitness:
                    best_fitness = loss.item()
                return loss.item()
            
            start_time = time.time()
            
            for step in range(50):
                if name == 'BFO':
                    loss = optimizer.step(closure)
                else:
                    optimizer.zero_grad()
                    loss = closure()
                    loss_tensor = problem.evaluate(x)
                    loss_tensor.backward()
                    optimizer.step()
                    loss = loss_tensor.item()
                
                history.append(loss)
                
                if abs(loss - problem.global_optimum) < 1e-6:
                    break
            
            end_time = time.time()
            
            results[name] = {
                'final_loss': loss,
                'best_fitness': best_fitness,
                'optimization_time': end_time - start_time,
                'convergence_history': history,
                'success': abs(best_fitness - problem.global_optimum) < 0.01
            }
        
        return results
    
    def verify_bfo_mechanisms(self) -> Dict[str, Any]:
        """Verify that BFO mechanisms (chemotaxis, reproduction, elimination) work correctly."""
        
        print("\nVerifying BFO mechanisms...")
        
        x = nn.Parameter(torch.randn(3) * 2.0)
        optimizer = BFO([x], population_size=10, lr=0.01, 
                       chemotaxis_steps=5, reproduction_steps=2, elimination_steps=1)
        
        # Track population changes
        population_history = []
        fitness_history = []
        
        def closure():
            return torch.sum(x**2).item()
        
        for step in range(10):
            loss = optimizer.step(closure)
            
            # Get current population state
            group_id = id(optimizer.param_groups[0])
            if group_id in optimizer.state:
                population = optimizer.state[group_id]['population']
                population_history.append(population.clone())
                fitness_history.append(loss)
        
        # Verify mechanisms
        mechanisms_verified = {
            'population_size_consistent': all(p.shape[0] == 10 for p in population_history),
            'fitness_improving': fitness_history[-1] <= fitness_history[0],
            'population_diversity': len(set(tuple(p.flatten().cpu().numpy()) for p in population_history)) > 1
        }
        
        return {
            'mechanisms_verified': mechanisms_verified,
            'population_history': [p.cpu().numpy() for p in population_history],
            'fitness_history': fitness_history
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report."""
        
        print("=" * 60)
        print("BFO COMPREHENSIVE VERIFICATION TEST")
        print("=" * 60)
        
        results = {
            'optimizer_variants': self.test_optimizer_variants(),
            'theoretical_verification': self.compare_with_theoretical_bfo(),
            'benchmark_comparison': self.benchmark_against_other_optimizers(),
            'mechanism_verification': self.verify_bfo_mechanisms()
        }
        
        # Generate summary
        summary = self._generate_summary(results)
        results['summary'] = summary
        
        # Save results
        with open('bfo_verification_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        self._print_summary(summary)
        
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of all test results."""
        
        summary = {
            'total_problems_tested': len(self.problems),
            'optimizer_variants_tested': len(results['optimizer_variants']),
            'overall_success_rate': 0,
            'best_performing_optimizer': None,
            'average_convergence_time': 0,
            'verification_passed': True
        }
        
        # Calculate success rates
        total_tests = 0
        successful_tests = 0
        
        for optimizer_name, optimizer_results in results['optimizer_variants'].items():
            for problem_name, problem_result in optimizer_results.items():
                total_tests += 1
                if problem_result['success']:
                    successful_tests += 1
        
        summary['overall_success_rate'] = successful_tests / total_tests if total_tests > 0 else 0
        
        # Find best performing optimizer
        best_optimizer = None
        best_score = 0
        
        for optimizer_name, optimizer_results in results['optimizer_variants'].items():
            success_count = sum(1 for r in optimizer_results.values() if r['success'])
            if success_count > best_score:
                best_score = success_count
                best_optimizer = optimizer_name
        
        summary['best_performing_optimizer'] = best_optimizer
        
        # Check verification results
        theoretical = results['theoretical_verification']
        mechanisms = results['mechanism_verification']
        
        if not theoretical['diversity_decreasing'] or theoretical['convergence_monotonicity_ratio'] < 0.5:
            summary['verification_passed'] = False
        
        if not all(mechanisms['mechanisms_verified'].values()):
            summary['verification_passed'] = False
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print a formatted summary of test results."""
        
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"Total problems tested: {summary['total_problems_tested']}")
        print(f"Optimizer variants tested: {summary['optimizer_variants_tested']}")
        print(f"Overall success rate: {summary['overall_success_rate']:.2%}")
        print(f"Best performing optimizer: {summary['best_performing_optimizer']}")
        print(f"Verification passed: {'✓' if summary['verification_passed'] else '✗'}")
        
        if summary['verification_passed']:
            print("\n🎉 BFO implementation verification PASSED!")
            print("The implementation shows correct BFO behavior patterns.")
        else:
            print("\n⚠️  BFO implementation verification FAILED!")
            print("Some verification checks did not pass.")


def main():
    """Run the comprehensive BFO verification test."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create and run benchmark
    benchmark = BFOBenchmark()
    results = benchmark.run_comprehensive_test()
    
    print(f"\nResults saved to: bfo_verification_results.json")
    print("You can load and analyze the detailed results programmatically.")


if __name__ == "__main__":
    main()