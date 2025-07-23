#!/usr/bin/env python3
"""
Benchmark PyTorch BFO Optimizer against standard optimizers
Compares performance on various optimization tasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_bfo_optimizer import BFO, AdaptiveBFO, HybridBFO
import time
import numpy as np
import matplotlib.pyplot as plt


def rosenbrock_function(x):
    """Rosenbrock test function (non-convex)."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def ackley_function(x):
    """Ackley test function (many local minima)."""
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    
    sum1 = torch.sum(x**2)
    sum2 = torch.sum(torch.cos(c * x))
    
    term1 = -a * torch.exp(-b * torch.sqrt(sum1 / d))
    term2 = -torch.exp(sum2 / d)
    
    return term1 + term2 + a + torch.exp(torch.tensor(1.0))


def rastrigin_function(x):
    """Rastrigin test function (highly multimodal)."""
    A = 10
    n = len(x)
    return A * n + torch.sum(x**2 - A * torch.cos(2 * np.pi * x))


def benchmark_function_optimization():
    """Benchmark optimizers on test functions."""
    print("=" * 60)
    print("FUNCTION OPTIMIZATION BENCHMARKS")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_functions = [
        ("Rosenbrock", rosenbrock_function, torch.tensor([0.0, 0.0], device=device)),
        ("Ackley", ackley_function, torch.randn(10, device=device)),
        ("Rastrigin", rastrigin_function, torch.randn(10, device=device) * 2),
    ]
    
    optimizers_config = [
        ("BFO", lambda p: BFO(p, population_size=30, compile_mode=False)),
        ("AdaptiveBFO", lambda p: AdaptiveBFO(p, population_size=30, compile_mode=False)),
        ("HybridBFO", lambda p: HybridBFO(p, population_size=30, compile_mode=False)),
        ("Adam", lambda p: optim.Adam(p, lr=0.01)),
        ("SGD", lambda p: optim.SGD(p, lr=0.01, momentum=0.9)),
        ("RMSprop", lambda p: optim.RMSprop(p, lr=0.01)),
    ]
    
    results = {}
    
    for func_name, func, init_point in test_functions:
        print(f"\n{func_name} Function:")
        results[func_name] = {}
        
        for opt_name, opt_factory in optimizers_config:
            # Reset parameters
            x = nn.Parameter(init_point.clone())
            optimizer = opt_factory([x])
            
            losses = []
            start_time = time.time()
            
            for i in range(100):
                if opt_name in ["BFO", "AdaptiveBFO", "HybridBFO"]:
                    def closure():
                        return func(x).item()
                    loss = optimizer.step(closure)
                else:
                    optimizer.zero_grad()
                    loss = func(x)
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()
                
                losses.append(loss)
            
            elapsed = time.time() - start_time
            
            results[func_name][opt_name] = {
                "losses": losses,
                "time": elapsed,
                "final_loss": losses[-1],
                "final_x": x.detach().cpu().numpy()
            }
            
            print(f"  {opt_name}: Loss = {losses[-1]:.6f}, Time = {elapsed:.2f}s")
    
    return results


def benchmark_neural_network_training():
    """Benchmark on neural network training tasks."""
    print("\n" + "=" * 60)
    print("NEURAL NETWORK TRAINING BENCHMARKS")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate synthetic dataset
    torch.manual_seed(42)
    X = torch.randn(1000, 20, device=device)
    y = torch.sum(X * torch.randn(20, device=device), dim=1, keepdim=True)
    y += 0.1 * torch.randn(1000, 1, device=device)
    
    # Split data
    train_X, test_X = X[:800], X[800:]
    train_y, test_y = y[:800], y[800:]
    
    results = {}
    
    optimizers_config = [
        ("BFO", lambda p: BFO(p, population_size=20, compile_mode=torch.cuda.is_available())),
        ("AdaptiveBFO", lambda p: AdaptiveBFO(p, population_size=20, compile_mode=torch.cuda.is_available())),
        ("HybridBFO", lambda p: HybridBFO(p, population_size=20, gradient_weight=0.3, compile_mode=torch.cuda.is_available())),
        ("Adam", lambda p: optim.Adam(p, lr=0.001)),
        ("SGD", lambda p: optim.SGD(p, lr=0.01, momentum=0.9)),
        ("AdamW", lambda p: optim.AdamW(p, lr=0.001, weight_decay=0.01)),
    ]
    
    for opt_name, opt_factory in optimizers_config:
        print(f"\nTraining with {opt_name}...")
        
        # Create model
        model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)
        
        if torch.cuda.is_available() and opt_name not in ["BFO", "AdaptiveBFO", "HybridBFO"]:
            model = torch.compile(model)
        
        optimizer = opt_factory(model.parameters())
        criterion = nn.MSELoss()
        
        train_losses = []
        test_losses = []
        start_time = time.time()
        
        for epoch in range(50):
            # Training
            if opt_name in ["BFO", "AdaptiveBFO"]:
                def closure():
                    output = model(train_X)
                    loss = criterion(output, train_y)
                    return loss.item()
                train_loss = optimizer.step(closure)
            elif opt_name == "HybridBFO":
                def closure():
                    optimizer.zero_grad()
                    output = model(train_X)
                    loss = criterion(output, train_y)
                    loss.backward()
                    return loss.item()
                train_loss = optimizer.step(closure)
            else:
                optimizer.zero_grad()
                output = model(train_X)
                loss = criterion(output, train_y)
                loss.backward()
                optimizer.step()
                train_loss = loss.item()
            
            # Testing
            with torch.no_grad():
                test_output = model(test_X)
                test_loss = criterion(test_output, test_y).item()
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
        
        elapsed = time.time() - start_time
        
        results[opt_name] = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "time": elapsed,
            "final_train_loss": train_losses[-1],
            "final_test_loss": test_losses[-1]
        }
        
        print(f"  Final Train Loss: {train_losses[-1]:.4f}")
        print(f"  Final Test Loss: {test_losses[-1]:.4f}")
        print(f"  Time: {elapsed:.2f}s")
    
    return results


def plot_results(function_results, nn_results):
    """Plot benchmark results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Function optimization results
    for idx, (func_name, func_results) in enumerate(function_results.items()):
        if idx >= 2:
            break
        ax = axes[0, idx]
        
        for opt_name, data in func_results.items():
            losses = data["losses"]
            ax.plot(losses, label=f"{opt_name} (t={data['time']:.1f}s)")
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(f"{func_name} Function Optimization")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Neural network training results
    ax = axes[1, 0]
    for opt_name, data in nn_results.items():
        ax.plot(data["train_losses"], label=opt_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Neural Network Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for opt_name, data in nn_results.items():
        ax.plot(data["test_losses"], label=f"{opt_name} (t={data['time']:.1f}s)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Loss")
    ax.set_title("Neural Network Test Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=150)
    plt.show()


def print_summary(function_results, nn_results):
    """Print summary of results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Function optimization summary
    print("\nFunction Optimization Results:")
    print("-" * 40)
    
    for func_name in function_results:
        print(f"\n{func_name} Function:")
        
        # Sort by final loss
        sorted_results = sorted(
            function_results[func_name].items(),
            key=lambda x: x[1]["final_loss"]
        )
        
        for rank, (opt_name, data) in enumerate(sorted_results, 1):
            print(f"  {rank}. {opt_name}: Loss = {data['final_loss']:.6f}, Time = {data['time']:.2f}s")
    
    # Neural network summary
    print("\nNeural Network Training Results:")
    print("-" * 40)
    
    # Sort by test loss
    sorted_nn = sorted(
        nn_results.items(),
        key=lambda x: x[1]["final_test_loss"]
    )
    
    for rank, (opt_name, data) in enumerate(sorted_nn, 1):
        print(f"  {rank}. {opt_name}:")
        print(f"     Train Loss: {data['final_train_loss']:.4f}")
        print(f"     Test Loss: {data['final_test_loss']:.4f}")
        print(f"     Time: {data['time']:.2f}s")


def main():
    """Run all benchmarks."""
    print("PyTorch BFO Optimizer Benchmark Suite")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Run benchmarks
    function_results = benchmark_function_optimization()
    nn_results = benchmark_neural_network_training()
    
    # Print summary
    print_summary(function_results, nn_results)
    
    # Plot results
    try:
        plot_results(function_results, nn_results)
    except:
        print("\nNote: Install matplotlib to see plots")


if __name__ == "__main__":
    main()