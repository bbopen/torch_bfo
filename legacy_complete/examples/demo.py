"""
PyTorch BFO Optimizer Demo
Demonstrates usage with PyTorch 2.8+ and torch.compile
"""

import torch
import torch.nn as nn
from pytorch_bfo_optimizer import BFO, AdaptiveBFO, HybridBFO
import time
import matplotlib.pyplot as plt


def demo_basic_regression():
    """Basic linear regression example."""
    print("=== Basic Linear Regression with BFO ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate synthetic data
    X = torch.randn(100, 10)
    true_weights = torch.randn(10, 1)
    y = X @ true_weights + 0.1 * torch.randn(100, 1)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)
    
    # Create model
    model = nn.Linear(10, 1).to(device)
    
    # Initialize optimizer with torch.compile support
    optimizer = BFO(
        model.parameters(),
        population_size=20,
        chem_steps=5,
        compile_mode=torch.cuda.is_available()
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    start_time = time.time()
    
    for epoch in range(50):
        def closure():
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            return loss.item()
        
        loss = optimizer.step(closure)
        losses.append(loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds")
    print(f"Final Loss: {losses[-1]:.4f}")
    
    return losses


def demo_adaptive_optimization():
    """Demonstrate AdaptiveBFO on non-convex function."""
    print("\n=== Adaptive BFO on Rosenbrock Function ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize parameters
    x = nn.Parameter(torch.tensor([-1.5, 2.5], device=device))
    
    # AdaptiveBFO with automatic parameter tuning
    optimizer = AdaptiveBFO(
        [x],
        population_size=30,
        adaptation_rate=0.15,
        compile_mode=torch.cuda.is_available()
    )
    
    # Track optimization progress
    positions = []
    losses = []
    
    for i in range(100):
        def rosenbrock():
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        
        loss = optimizer.step(rosenbrock)
        losses.append(loss)
        positions.append(x.detach().cpu().numpy().copy())
        
        if i % 20 == 0:
            print(f"Iteration {i}, Loss: {loss:.6f}, Position: [{x[0].item():.3f}, {x[1].item():.3f}]")
    
    print(f"Final position: [{x[0].item():.6f}, {x[1].item():.6f}]")
    print(f"Final loss: {losses[-1]:.6f}")
    
    return positions, losses


def demo_hybrid_neural_network():
    """Demonstrate HybridBFO on neural network training."""
    print("\n=== Hybrid BFO for Neural Network Training ===")
    
    # Generate spiral dataset
    torch.manual_seed(42)
    n_samples = 1000
    noise = 0.1
    
    t = torch.linspace(0, 4 * torch.pi, n_samples)
    X = torch.stack([
        t * torch.cos(t) + noise * torch.randn(n_samples),
        t * torch.sin(t) + noise * torch.randn(n_samples)
    ], dim=1)
    y = (t > 2 * torch.pi).float().unsqueeze(1)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)
    
    # Create model with torch.compile
    model = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    ).to(device)
    
    if torch.cuda.is_available():
        model = torch.compile(model)
    
    # HybridBFO combines bacterial foraging with gradients
    optimizer = HybridBFO(
        model.parameters(),
        population_size=15,
        gradient_weight=0.3,
        use_momentum=True,
        compile_mode=torch.cuda.is_available()
    )
    
    criterion = nn.BCELoss()
    losses = []
    
    print("Training neural network on spiral dataset...")
    start_time = time.time()
    
    for epoch in range(30):
        def closure():
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()  # Compute gradients for hybrid mode
            return loss.item()
        
        loss = optimizer.step(closure)
        losses.append(loss)
        
        if epoch % 5 == 0:
            accuracy = ((model(X) > 0.5).float() == y).float().mean()
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2%}")
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds")
    
    # Final evaluation
    with torch.no_grad():
        predictions = model(X) > 0.5
        accuracy = (predictions.float() == y).float().mean()
        print(f"Final Accuracy: {accuracy:.2%}")
    
    return losses


def demo_comparison():
    """Compare different BFO variants."""
    print("\n=== Comparison of BFO Variants ===")
    
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test function: Ackley function
    def ackley(x):
        a = 20
        b = 0.2
        c = 2 * torch.pi
        d = x.shape[0]
        
        sum1 = torch.sum(x**2)
        sum2 = torch.sum(torch.cos(c * x))
        
        term1 = -a * torch.exp(-b * torch.sqrt(sum1 / d))
        term2 = -torch.exp(sum2 / d)
        
        return term1 + term2 + a + torch.exp(torch.tensor(1.0))
    
    results = {}
    
    for name, optimizer_class, kwargs in [
        ("Standard BFO", BFO, {}),
        ("Adaptive BFO", AdaptiveBFO, {"adaptation_rate": 0.2}),
        ("Hybrid BFO", HybridBFO, {"gradient_weight": 0.5}),
    ]:
        torch.manual_seed(42)
        x = nn.Parameter(torch.randn(10, device=device) * 2)
        
        optimizer = optimizer_class(
            [x],
            population_size=20,
            compile_mode=torch.cuda.is_available(),
            **kwargs
        )
        
        losses = []
        start_time = time.time()
        
        for _ in range(50):
            if isinstance(optimizer, HybridBFO):
                def closure():
                    optimizer.zero_grad()
                    loss = ackley(x)
                    loss.backward()
                    return loss.item()
            else:
                def closure():
                    return ackley(x).item()
            
            loss = optimizer.step(closure)
            losses.append(loss)
        
        elapsed = time.time() - start_time
        results[name] = {
            "losses": losses,
            "time": elapsed,
            "final_loss": losses[-1]
        }
        
        print(f"{name}: Final loss = {losses[-1]:.6f}, Time = {elapsed:.2f}s")
    
    return results


def plot_results(basic_losses, rosenbrock_data, hybrid_losses, comparison_results):
    """Visualize optimization results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Basic regression
    axes[0, 0].plot(basic_losses)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Linear Regression with BFO")
    axes[0, 0].set_yscale("log")
    
    # Rosenbrock optimization
    positions, losses = rosenbrock_data
    positions = np.array(positions)
    axes[0, 1].plot(positions[:, 0], positions[:, 1], 'o-', markersize=4)
    axes[0, 1].plot(1, 1, 'r*', markersize=15, label='Optimum')
    axes[0, 1].set_xlabel("x1")
    axes[0, 1].set_ylabel("x2")
    axes[0, 1].set_title("Rosenbrock Function Optimization Path")
    axes[0, 1].legend()
    
    # Hybrid neural network
    axes[1, 0].plot(hybrid_losses)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Neural Network Training with HybridBFO")
    
    # Comparison
    for name, data in comparison_results.items():
        axes[1, 1].plot(data["losses"], label=f"{name} (t={data['time']:.1f}s)")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("Comparison of BFO Variants")
    axes[1, 1].legend()
    axes[1, 1].set_yscale("log")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("PyTorch BFO Optimizer Demo")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    # Run demos
    basic_losses = demo_basic_regression()
    rosenbrock_data = demo_adaptive_optimization()
    hybrid_losses = demo_hybrid_neural_network()
    comparison_results = demo_comparison()
    
    # Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        plot_results(basic_losses, rosenbrock_data, hybrid_losses, comparison_results)
    except ImportError:
        print("\nInstall matplotlib to visualize results: pip install matplotlib")