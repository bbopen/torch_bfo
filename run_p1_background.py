#!/usr/bin/env python3
"""
Background P1 Experiment Runner
==============================

Runs P1 experiments in the background and generates a report when complete.
"""

import os
import sys
import time
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
import statistics

# Import our implementations
from src.bfo_torch.optimizer import BFO
from src.bfo_torch.chaotic_bfo import ChaoticBFO


class P1BackgroundRunner:
    """Run P1 experiments with progress tracking."""
    
    def __init__(self):
        self.schwefel_global_optimum = 0.0
        self.schwefel_optimum_position = 420.9687
        self.results_file = "p1_background_results.json"
        self.progress_file = "p1_background_progress.txt"
        
    def schwefel_function(self, x: torch.Tensor) -> torch.Tensor:
        """Standard Schwefel function implementation."""
        n = len(x)
        return 418.9829 * n - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))))
    
    def log_progress(self, message: str):
        """Log progress to file and console."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.progress_file, 'a') as f:
            f.write(log_message + "\n")
    
    def run_single_experiment(self, config: Dict, seed: int) -> Dict:
        """Run a single experiment."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize
        x = nn.Parameter(torch.rand(2) * 1000 - 500)  # [-500, 500]
        
        # Create optimizer
        optimizer = config['optimizer_class']([x], **config['params'])
        
        def closure():
            loss = self.schwefel_function(x)
            return loss.item()
        
        # Track progress
        initial_loss = closure()
        best_loss = initial_loss
        losses = []
        fe_history = []
        
        # Run optimization
        max_fe = config['max_fe']
        step = 0
        
        while True:
            current_fe = optimizer.get_function_evaluations()
            if current_fe >= max_fe:
                break
                
            loss = optimizer.step(closure, max_fe=max_fe)
            
            # Apply bounds
            with torch.no_grad():
                x.data = torch.clamp(x.data, -500, 500)
            
            # Track
            losses.append(loss)
            fe_history.append(current_fe)
            
            if loss < best_loss:
                best_loss = loss
            
            step += 1
            
            # Safety break
            if step > 1000:
                break
        
        final_fe = optimizer.get_function_evaluations()
        success = best_loss <= 1e-4
        
        return {
            'seed': seed,
            'initial_loss': initial_loss,
            'final_loss': best_loss,
            'success': success,
            'function_evaluations': final_fe,
            'final_position': x.data.clone().cpu().numpy().tolist(),
            'distance_to_optimum': torch.norm(x.data - torch.tensor([420.9687, 420.9687])).item()
        }
    
    def run_configuration(self, name: str, config: Dict) -> Dict:
        """Run all experiments for a configuration."""
        self.log_progress(f"Starting {name}...")
        
        results = []
        num_runs = config['num_runs']
        
        for run in range(num_runs):
            if (run + 1) % 5 == 0:
                self.log_progress(f"  {name}: Completed {run + 1}/{num_runs} runs")
            
            seed = 1000 + run
            result = self.run_single_experiment(config, seed)
            results.append(result)
        
        # Calculate statistics
        final_losses = [r['final_loss'] for r in results]
        success_count = sum(1 for r in results if r['success'])
        
        stats = {
            'name': name,
            'config': config['params'],
            'num_runs': num_runs,
            'success_rate': success_count / num_runs,
            'final_loss_mean': statistics.mean(final_losses),
            'final_loss_std': statistics.stdev(final_losses) if len(final_losses) > 1 else 0,
            'final_loss_min': min(final_losses),
            'individual_runs': results
        }
        
        self.log_progress(f"  {name}: Success rate = {stats['success_rate']:.1%}, Mean loss = {stats['final_loss_mean']:.2f}")
        
        return stats
    
    def create_p1_configurations(self) -> Dict:
        """Create P1 experiment configurations."""
        configs = {
            # Baseline P0 for comparison
            'P0_Enhanced_BFO': {
                'optimizer_class': BFO,
                'params': {
                    'population_size': 50,
                    'lr': 0.01,
                    'chemotaxis_steps': 10,
                    'reproduction_steps': 5,
                    'elimination_steps': 2,
                    'elimination_prob': 0.4,
                    'step_size_max': 1.0,
                    'levy_alpha': 1.8,
                    'enable_swarming': True,
                    'swim_length': 5
                },
                'max_fe': 50000,
                'num_runs': 10
            },
            
            # P1 with all improvements
            'P1_Chaotic_BFO': {
                'optimizer_class': ChaoticBFO,
                'params': {
                    'population_size': 50,
                    'lr': 0.01,
                    'chemotaxis_steps': 10,
                    'reproduction_steps': 5,
                    'elimination_steps': 2,
                    'elimination_prob': 0.4,
                    'step_size_max': 1.0,
                    'levy_alpha': 1.8,
                    'enable_swarming': True,
                    'enable_chaos': True,
                    'chaos_strength': 0.5,
                    'diversity_trigger_ratio': 0.5,
                    'enable_crossover': True,
                    'diversity_threshold_decay': 0.9,
                    'swim_length': 5
                },
                'max_fe': 50000,
                'num_runs': 10
            },
            
            # P1 without GA crossover
            'P1_NoGA': {
                'optimizer_class': ChaoticBFO,
                'params': {
                    'population_size': 50,
                    'lr': 0.01,
                    'chemotaxis_steps': 10,
                    'reproduction_steps': 5,
                    'elimination_steps': 2,
                    'elimination_prob': 0.4,
                    'step_size_max': 1.0,
                    'levy_alpha': 1.8,
                    'enable_swarming': True,
                    'enable_chaos': True,
                    'chaos_strength': 0.5,
                    'diversity_trigger_ratio': 0.5,
                    'enable_crossover': False,  # KEY: No GA
                    'diversity_threshold_decay': 0.9,
                    'swim_length': 5
                },
                'max_fe': 50000,
                'num_runs': 10
            },
            
            # P1 without chaos injection
            'P1_NoChaos': {
                'optimizer_class': ChaoticBFO,
                'params': {
                    'population_size': 50,
                    'lr': 0.01,
                    'chemotaxis_steps': 10,
                    'reproduction_steps': 5,
                    'elimination_steps': 2,
                    'elimination_prob': 0.4,
                    'step_size_max': 1.0,
                    'levy_alpha': 1.8,
                    'enable_swarming': True,
                    'enable_chaos': False,  # KEY: No chaos
                    'chaos_strength': 0.0,
                    'diversity_trigger_ratio': 0.5,
                    'enable_crossover': True,
                    'diversity_threshold_decay': 0.9,
                    'swim_length': 5
                },
                'max_fe': 50000,
                'num_runs': 10
            },
            
            # P1 without diversity trigger
            'P1_NoDiversity': {
                'optimizer_class': ChaoticBFO,
                'params': {
                    'population_size': 50,
                    'lr': 0.01,
                    'chemotaxis_steps': 10,
                    'reproduction_steps': 5,
                    'elimination_steps': 2,
                    'elimination_prob': 0.4,
                    'step_size_max': 1.0,
                    'levy_alpha': 1.8,
                    'enable_swarming': True,
                    'enable_chaos': True,
                    'chaos_strength': 0.5,
                    'diversity_trigger_ratio': 0.0,  # KEY: No diversity trigger
                    'enable_crossover': True,
                    'diversity_threshold_decay': 0.9,
                    'swim_length': 5
                },
                'max_fe': 50000,
                'num_runs': 10
            },
            
            # P1 with 100k budget
            'P1_All_100k': {
                'optimizer_class': ChaoticBFO,
                'params': {
                    'population_size': 50,
                    'lr': 0.01,
                    'chemotaxis_steps': 10,
                    'reproduction_steps': 5,
                    'elimination_steps': 2,
                    'elimination_prob': 0.4,
                    'step_size_max': 1.0,
                    'levy_alpha': 1.8,
                    'enable_swarming': True,
                    'enable_chaos': True,
                    'chaos_strength': 0.5,
                    'diversity_trigger_ratio': 0.5,
                    'enable_crossover': True,
                    'diversity_threshold_decay': 0.9,
                    'swim_length': 5
                },
                'max_fe': 100000,  # KEY: 100k budget
                'num_runs': 10
            }
        }
        
        return configs
    
    def generate_report(self, results: Dict):
        """Generate final report."""
        report = []
        report.append("=" * 80)
        report.append("P1 BACKGROUND EXPERIMENT RESULTS")
        report.append("=" * 80)
        report.append("")
        
        # Summary table
        report.append("Configuration Summary:")
        report.append("-" * 80)
        report.append(f"{'Config':<25} {'Success':<12} {'Mean Loss':<15} {'Min Loss':<15} {'FE Budget':<10}")
        report.append("-" * 80)
        
        for name, data in results.items():
            success_rate = data['success_rate'] * 100
            mean_loss = data['final_loss_mean']
            min_loss = data['final_loss_min']
            fe_budget = data['config'].get('max_fe', 50000)
            report.append(f"{name:<25} {success_rate:>6.1f}%     {mean_loss:>10.2f}     {min_loss:>10.2f}     {fe_budget:>8}")
        
        # Component analysis
        report.append("")
        report.append("=" * 80)
        report.append("COMPONENT CONTRIBUTION ANALYSIS:")
        report.append("-" * 80)
        
        if 'P1_Chaotic_BFO' in results:
            baseline = results['P1_Chaotic_BFO']['final_loss_mean']
            
            if 'P1_NoGA' in results:
                no_ga = results['P1_NoGA']['final_loss_mean']
                ga_contrib = (no_ga - baseline) / baseline * 100
                report.append(f"GA Crossover: {ga_contrib:+.1f}% loss change")
            
            if 'P1_NoChaos' in results:
                no_chaos = results['P1_NoChaos']['final_loss_mean']
                chaos_contrib = (no_chaos - baseline) / baseline * 100
                report.append(f"Chaos Injection: {chaos_contrib:+.1f}% loss change")
            
            if 'P1_NoDiversity' in results:
                no_div = results['P1_NoDiversity']['final_loss_mean']
                div_contrib = (no_div - baseline) / baseline * 100
                report.append(f"Diversity Trigger: {div_contrib:+.1f}% loss change")
        
        # Success analysis
        report.append("")
        report.append("=" * 80)
        report.append("SUCCESS ANALYSIS:")
        
        any_success = any(r['success_rate'] > 0 for r in results.values())
        max_success = max(r['success_rate'] for r in results.values())
        
        if any_success:
            report.append(f"✅ SUCCESS! Achieved {max_success:.1%} success rate")
            for name, data in results.items():
                if data['success_rate'] > 0:
                    report.append(f"   {name}: {data['success_rate']:.1%}")
        else:
            report.append("❌ No configuration achieved tolerance success (1e-4)")
            
            # Find best
            best_config = min(results.keys(), key=lambda k: results[k]['final_loss_mean'])
            best_loss = results[best_config]['final_loss_mean']
            
            if 'P0_Enhanced_BFO' in results:
                p0_loss = results['P0_Enhanced_BFO']['final_loss_mean']
                improvement = (p0_loss - best_loss) / p0_loss * 100
                report.append(f"   Best: {best_config} with {improvement:.1f}% improvement over P0")
        
        # Save report
        report_text = "\n".join(report)
        with open("p1_background_report.txt", 'w') as f:
            f.write(report_text)
        
        self.log_progress("Report saved to: p1_background_report.txt")
        print("\n" + report_text)
    
    def run_all_experiments(self):
        """Run all P1 experiments."""
        self.log_progress("Starting P1 background experiments...")
        
        # Clear progress file
        with open(self.progress_file, 'w') as f:
            f.write("P1 Background Experiment Progress Log\n")
            f.write("=" * 50 + "\n\n")
        
        # Get configurations
        configs = self.create_p1_configurations()
        results = {}
        
        # Run each configuration
        for name, config in configs.items():
            result = self.run_configuration(name, config)
            results[name] = result
            
            # Save intermediate results
            with open(self.results_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Generate final report
        self.log_progress("All experiments completed. Generating report...")
        self.generate_report(results)
        
        self.log_progress("Done!")
        return results


def main():
    """Run P1 experiments in background."""
    runner = P1BackgroundRunner()
    runner.run_all_experiments()


if __name__ == "__main__":
    main()