#!/usr/bin/env python3
"""
Comprehensive benchmarking suite for PyTorch BFO Optimizer
Compares original BFO, BFOv2, and standard PyTorch optimizers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch_bfo_optimizer import BFO, AdaptiveBFO, HybridBFO
from pytorch_bfo_optimizer.optimizer_v2 import BFOv2, AdaptiveBFOv2, HybridBFOv2


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    optimizer_name: str
    task_name: str
    device: str
    final_loss: float
    best_loss: float
    time_per_step: float
    total_time: float
    iterations: int
    converged: bool
    loss_history: List[float]
    memory_peak: Optional[float] = None
    config: Optional[Dict[str, Any]] = None


class BenchmarkTasks:
    """Collection of benchmark tasks"""
    
    @staticmethod
    def rosenbrock(dim: int = 10, device: str = 'cpu') -> Tuple[torch.nn.Parameter, callable]:
        """Rosenbrock function optimization"""
        x = nn.Parameter(torch.randn(dim, device=device))
        
        def closure():
            loss = 0
            for i in range(dim - 1):
                loss += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
            return loss
        
        return [x], closure
    
    @staticmethod
    def rastrigin(dim: int = 10, device: str = 'cpu') -> Tuple[torch.nn.Parameter, callable]:
        """Rastrigin function optimization"""
        x = nn.Parameter(torch.randn(dim, device=device) * 5)
        
        def closure():
            A = 10
            n = len(x)
            return A * n + torch.sum(x**2 - A * torch.cos(2 * np.pi * x))
        
        return [x], closure
    
    @staticmethod
    def neural_network_mnist(device: str = 'cpu') -> Tuple[nn.Module, DataLoader, callable]:
        """Simple neural network for MNIST-like data"""
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        ).to(device)
        
        # Generate synthetic data
        X = torch.randn(1000, 784, device=device)
        y = torch.randint(0, 10, (1000,), device=device)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
        
        def closure(data, target):
            output = model(data)
            loss = F.cross_entropy(output, target)
            return loss
        
        return model, dataloader, closure
    
    @staticmethod
    def conv_network(device: str = 'cpu') -> Tuple[nn.Module, DataLoader, callable]:
        """Small convolutional network"""
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        ).to(device)
        
        # Generate synthetic data
        X = torch.randn(500, 1, 28, 28, device=device)
        y = torch.randint(0, 10, (500,), device=device)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
        
        def closure(data, target):
            output = model(data)
            loss = F.cross_entropy(output, target)
            return loss
        
        return model, dataloader, closure


class OptimizerBenchmark:
    """Main benchmark runner"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results: List[BenchmarkResult] = []
        
    def get_optimizers(self, params, lr: float = 0.01) -> Dict[str, Any]:
        """Get all optimizers to benchmark"""
        optimizers = {
            # Standard PyTorch optimizers
            'SGD': torch.optim.SGD(params, lr=lr),
            'Adam': torch.optim.Adam(params, lr=lr),
            'AdamW': torch.optim.AdamW(params, lr=lr),
            
            # Original BFO variants
            'BFO': BFO(params, population_size=10, compile_mode=False),
            'AdaptiveBFO': AdaptiveBFO(params, population_size=10, compile_mode=False),
            'HybridBFO': HybridBFO(params, population_size=10, gradient_weight=0.5, compile_mode=False),
            
            # Optimized BFOv2 variants
            'BFOv2': BFOv2(params, device_type='auto', parallel_eval=True),
            'AdaptiveBFOv2': AdaptiveBFOv2(params, device_type='auto', parallel_eval=True),
            'HybridBFOv2': HybridBFOv2(params, gradient_weight=0.5, device_type='auto'),
        }
        
        return optimizers
    
    def benchmark_function_optimization(self, task_name: str, task_fn, iterations: int = 100):
        """Benchmark function optimization tasks"""
        print(f"\nBenchmarking {task_name} on {self.device}")
        print("=" * 60)
        
        params, closure = task_fn(device=self.device)
        
        for opt_name, optimizer in self.get_optimizers(params).items():
            print(f"\nTesting {opt_name}...")
            
            # Reset parameters
            for p in params:
                p.data = torch.randn_like(p.data)
            
            # Track metrics
            loss_history = []
            start_time = time.time()
            converged = False
            
            # Optimization loop
            for i in range(iterations):
                if 'BFO' in opt_name:
                    # BFO-style optimizers
                    if 'Hybrid' in opt_name:
                        # Hybrid needs gradients
                        def grad_closure():
                            optimizer.zero_grad()
                            loss = closure()
                            loss.backward()
                            return loss.item()
                        loss = optimizer.step(grad_closure)
                    else:
                        # Pure BFO doesn't need gradients
                        def no_grad_closure():
                            with torch.no_grad():
                                return closure().item()
                        loss = optimizer.step(no_grad_closure)
                else:
                    # Standard optimizers
                    optimizer.zero_grad()
                    loss = closure()
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()
                
                loss_history.append(loss)
                
                # Check convergence
                if len(loss_history) > 10:
                    recent_std = np.std(loss_history[-10:])
                    if recent_std < 1e-6:
                        converged = True
                        break
            
            total_time = time.time() - start_time
            
            # Get memory usage if on GPU
            memory_peak = None
            if self.device == 'cuda':
                memory_peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
                torch.cuda.reset_peak_memory_stats()
            
            # Store results
            result = BenchmarkResult(
                optimizer_name=opt_name,
                task_name=task_name,
                device=self.device,
                final_loss=loss_history[-1],
                best_loss=min(loss_history),
                time_per_step=total_time / len(loss_history),
                total_time=total_time,
                iterations=len(loss_history),
                converged=converged,
                loss_history=loss_history,
                memory_peak=memory_peak,
                config=optimizer.defaults if hasattr(optimizer, 'defaults') else None
            )
            self.results.append(result)
            
            print(f"  Final loss: {result.final_loss:.6f}")
            print(f"  Best loss: {result.best_loss:.6f}")
            print(f"  Time: {result.total_time:.2f}s ({result.time_per_step*1000:.1f}ms/step)")
            print(f"  Converged: {result.converged} ({result.iterations} iterations)")
            if memory_peak:
                print(f"  Peak memory: {memory_peak:.1f} MB")
    
    def benchmark_neural_network(self, task_name: str, task_fn, epochs: int = 5):
        """Benchmark neural network training"""
        print(f"\nBenchmarking {task_name} on {self.device}")
        print("=" * 60)
        
        model, dataloader, closure = task_fn(device=self.device)
        
        for opt_name, optimizer in self.get_optimizers(model.parameters(), lr=0.001).items():
            print(f"\nTesting {opt_name}...")
            
            # Reset model
            for layer in model.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            
            # Track metrics
            loss_history = []
            start_time = time.time()
            
            # Training loop
            for epoch in range(epochs):
                epoch_losses = []
                
                for data, target in dataloader:
                    if 'BFO' in opt_name:
                        # BFO-style optimizers
                        if 'Hybrid' in opt_name:
                            def grad_closure():
                                optimizer.zero_grad()
                                loss = closure(data, target)
                                loss.backward()
                                return loss.item()
                            loss = optimizer.step(grad_closure)
                        else:
                            def no_grad_closure():
                                with torch.no_grad():
                                    return closure(data, target).item()
                            loss = optimizer.step(no_grad_closure)
                    else:
                        # Standard optimizers
                        optimizer.zero_grad()
                        loss = closure(data, target)
                        loss.backward()
                        optimizer.step()
                        loss = loss.item()
                    
                    epoch_losses.append(loss)
                
                avg_loss = np.mean(epoch_losses)
                loss_history.append(avg_loss)
                print(f"    Epoch {epoch+1}: {avg_loss:.4f}")
            
            total_time = time.time() - start_time
            
            # Get memory usage if on GPU
            memory_peak = None
            if self.device == 'cuda':
                memory_peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
                torch.cuda.reset_peak_memory_stats()
            
            # Store results
            result = BenchmarkResult(
                optimizer_name=opt_name,
                task_name=task_name,
                device=self.device,
                final_loss=loss_history[-1],
                best_loss=min(loss_history),
                time_per_step=total_time / (epochs * len(dataloader)),
                total_time=total_time,
                iterations=epochs * len(dataloader),
                converged=False,  # Not checking convergence for NN
                loss_history=loss_history,
                memory_peak=memory_peak,
                config=optimizer.defaults if hasattr(optimizer, 'defaults') else None
            )
            self.results.append(result)
            
            print(f"  Final loss: {result.final_loss:.4f}")
            print(f"  Time: {result.total_time:.2f}s")
            if memory_peak:
                print(f"  Peak memory: {memory_peak:.1f} MB")
    
    def generate_report(self, save_path: str = 'benchmark_results'):
        """Generate comprehensive benchmark report"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save raw results
        results_dict = []
        for r in self.results:
            d = r.__dict__.copy()
            d['loss_history'] = d['loss_history'][:20]  # Truncate for readability
            results_dict.append(d)
        
        with open(os.path.join(save_path, 'results.json'), 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Generate plots
        self._plot_convergence_curves(save_path)
        self._plot_performance_comparison(save_path)
        self._plot_memory_usage(save_path)
        
        # Generate summary table
        self._generate_summary_table(save_path)
    
    def _plot_convergence_curves(self, save_path: str):
        """Plot convergence curves for each task"""
        tasks = set(r.task_name for r in self.results)
        
        for task in tasks:
            plt.figure(figsize=(10, 6))
            
            for r in self.results:
                if r.task_name == task:
                    plt.semilogy(r.loss_history[:100], label=r.optimizer_name)
            
            plt.xlabel('Iteration')
            plt.ylabel('Loss (log scale)')
            plt.title(f'Convergence Curves - {task}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'convergence_{task}.png'))
            plt.close()
    
    def _plot_performance_comparison(self, save_path: str):
        """Plot performance comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time comparison
        tasks = sorted(set(r.task_name for r in self.results))
        optimizers = sorted(set(r.optimizer_name for r in self.results))
        
        time_data = np.zeros((len(tasks), len(optimizers)))
        for i, task in enumerate(tasks):
            for j, opt in enumerate(optimizers):
                for r in self.results:
                    if r.task_name == task and r.optimizer_name == opt:
                        time_data[i, j] = r.time_per_step * 1000  # ms
        
        sns.heatmap(time_data, xticklabels=optimizers, yticklabels=tasks,
                    annot=True, fmt='.1f', cmap='YlOrRd', ax=ax1)
        ax1.set_title('Time per Step (ms)')
        ax1.set_xlabel('Optimizer')
        ax1.set_ylabel('Task')
        
        # Final loss comparison
        loss_data = np.zeros((len(tasks), len(optimizers)))
        for i, task in enumerate(tasks):
            for j, opt in enumerate(optimizers):
                for r in self.results:
                    if r.task_name == task and r.optimizer_name == opt:
                        loss_data[i, j] = r.best_loss
        
        sns.heatmap(loss_data, xticklabels=optimizers, yticklabels=tasks,
                    annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax2)
        ax2.set_title('Best Loss Achieved')
        ax2.set_xlabel('Optimizer')
        ax2.set_ylabel('Task')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'performance_comparison.png'))
        plt.close()
    
    def _plot_memory_usage(self, save_path: str):
        """Plot memory usage comparison (GPU only)"""
        gpu_results = [r for r in self.results if r.memory_peak is not None]
        if not gpu_results:
            return
        
        plt.figure(figsize=(10, 6))
        
        optimizers = sorted(set(r.optimizer_name for r in gpu_results))
        tasks = sorted(set(r.task_name for r in gpu_results))
        
        x = np.arange(len(optimizers))
        width = 0.8 / len(tasks)
        
        for i, task in enumerate(tasks):
            memory_values = []
            for opt in optimizers:
                for r in gpu_results:
                    if r.task_name == task and r.optimizer_name == opt:
                        memory_values.append(r.memory_peak)
                        break
                else:
                    memory_values.append(0)
            
            plt.bar(x + i * width, memory_values, width, label=task)
        
        plt.xlabel('Optimizer')
        plt.ylabel('Peak Memory Usage (MB)')
        plt.title('GPU Memory Usage Comparison')
        plt.xticks(x + width * (len(tasks) - 1) / 2, optimizers, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'memory_usage.png'))
        plt.close()
    
    def _generate_summary_table(self, save_path: str):
        """Generate summary table"""
        with open(os.path.join(save_path, 'summary.txt'), 'w') as f:
            f.write("PyTorch BFO Optimizer Benchmark Summary\n")
            f.write("=" * 80 + "\n\n")
            
            tasks = sorted(set(r.task_name for r in self.results))
            
            for task in tasks:
                f.write(f"\nTask: {task}\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'Optimizer':<15} {'Best Loss':<12} {'Time/Step':<12} {'Converged':<10}\n")
                f.write("-" * 60 + "\n")
                
                task_results = [r for r in self.results if r.task_name == task]
                task_results.sort(key=lambda x: x.best_loss)
                
                for r in task_results:
                    f.write(f"{r.optimizer_name:<15} {r.best_loss:<12.6f} "
                           f"{r.time_per_step*1000:<12.1f}ms {str(r.converged):<10}\n")
            
            # Overall winners
            f.write("\n\nOverall Performance Summary\n")
            f.write("-" * 60 + "\n")
            
            # Best loss per task
            f.write("\nBest Loss Achieved:\n")
            for task in tasks:
                task_results = [r for r in self.results if r.task_name == task]
                best = min(task_results, key=lambda x: x.best_loss)
                f.write(f"  {task}: {best.optimizer_name} ({best.best_loss:.6f})\n")
            
            # Fastest per task
            f.write("\nFastest Optimizer:\n")
            for task in tasks:
                task_results = [r for r in self.results if r.task_name == task]
                fastest = min(task_results, key=lambda x: x.time_per_step)
                f.write(f"  {task}: {fastest.optimizer_name} ({fastest.time_per_step*1000:.1f}ms/step)\n")


def main():
    """Run comprehensive benchmark suite"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmarks on {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create benchmark runner
    benchmark = OptimizerBenchmark(device=device)
    
    # Run function optimization benchmarks
    benchmark.benchmark_function_optimization('Rosenbrock', BenchmarkTasks.rosenbrock, iterations=200)
    benchmark.benchmark_function_optimization('Rastrigin', BenchmarkTasks.rastrigin, iterations=200)
    
    # Run neural network benchmarks
    benchmark.benchmark_neural_network('MNIST_MLP', BenchmarkTasks.neural_network_mnist, epochs=5)
    
    # Only run CNN on GPU due to memory requirements
    if device == 'cuda':
        benchmark.benchmark_neural_network('ConvNet', BenchmarkTasks.conv_network, epochs=3)
    
    # Generate report
    benchmark.generate_report('benchmark_results')
    print("\nBenchmark complete! Results saved to benchmark_results/")


if __name__ == "__main__":
    main()