#!/usr/bin/env python3
"""
Demo script showcasing GPU optimization fixes for PyTorch BFO
Shows both the old way (with potential issues) and new way (with fixes)
"""

import torch
import torch.nn as nn
from pytorch_bfo_optimizer import BFO, HybridBFO
import time

# Fix for torch.compile graph breaks (add this at the top of your scripts)
import torch._dynamo as dynamo
dynamo.config.capture_scalar_outputs = True

# Suppress warnings if available
try:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*insufficient SMs.*")
except:
    pass


def demo_old_way():
    """Demo showing potential issues with the old approach."""
    print("\n" + "=" * 50)
    print("OLD WAY (Potential Issues)")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Small model and data (inefficient for GPU)
    model = nn.Linear(10, 1).to(device)
    data = torch.randn(32, 10, device=device)  # Small batch
    target = torch.randn(32, 1, device=device)
    
    # Large population (slow on GPU)
    optimizer = BFO(
        model.parameters(),
        population_size=50,  # Too large for efficient GPU use
        compile_mode=True  # May cause issues without dynamo config
    )
    
    # Closure without no_grad (unnecessary gradient tracking)
    def closure():
        output = model(data)
        loss = nn.functional.mse_loss(output, target)
        return loss.item()  # May cause graph breaks
    
    print("Configuration:")
    print(f"  Device: {device}")
    print(f"  Population size: 50 (large)")
    print(f"  Batch size: 32 (small)")
    print(f"  torch.compile: enabled")
    
    start = time.time()
    for i in range(3):
        loss = optimizer.step(closure)
        print(f"Step {i+1}: Loss = {loss:.6f}")
    
    print(f"Time: {time.time() - start:.2f}s")


def demo_new_way():
    """Demo showing optimized approach with fixes."""
    print("\n" + "=" * 50)
    print("NEW WAY (Optimized)")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Larger model and batch (better GPU utilization)
    model = nn.Sequential(
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)
    
    data = torch.randn(512, 100, device=device)  # Larger batch
    target = torch.randn(512, 1, device=device)
    
    # Smaller population (more efficient)
    optimizer = BFO(
        model.parameters(),
        population_size=10,  # Smaller for efficiency
        compile_mode=True,
        compile_kwargs={"mode": "default"}  # Better for smaller GPUs
    )
    
    # Optimized closure with no_grad
    def closure():
        with torch.no_grad():
            output = model(data)
            loss = nn.functional.mse_loss(output, target)
        return loss.item()  # Now safe with dynamo config
    
    print("Configuration:")
    print(f"  Device: {device}")
    print(f"  Population size: 10 (optimized)")
    print(f"  Batch size: 512 (larger)")
    print(f"  torch.compile: enabled with 'default' mode")
    
    start = time.time()
    for i in range(3):
        loss = optimizer.step(closure)
        print(f"Step {i+1}: Loss = {loss:.6f}")
    
    print(f"Time: {time.time() - start:.2f}s")


def demo_tensor_closure():
    """Demo showing new tensor return support."""
    print("\n" + "=" * 50)
    print("TENSOR CLOSURE (New Feature)")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = nn.Linear(50, 10).to(device)
    data = torch.randn(256, 50, device=device)
    target = torch.randn(256, 10, device=device)
    
    optimizer = BFO(model.parameters(), population_size=5)
    
    # Closure can now return tensor OR scalar
    def closure():
        with torch.no_grad():
            output = model(data)
            loss = nn.functional.mse_loss(output, target)
        return loss  # Returning tensor instead of loss.item()
    
    print("Testing closure that returns tensor (not scalar)...")
    
    for i in range(3):
        loss = optimizer.step(closure)
        print(f"Step {i+1}: Loss = {loss:.6f}")
    
    print("✅ Tensor returns are now supported!")


def demo_hybrid_gpu():
    """Demo showing HybridBFO for best GPU performance."""
    print("\n" + "=" * 50)
    print("HYBRID BFO (Best GPU Performance)")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = nn.Sequential(
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)
    
    data = torch.randn(1024, 100, device=device)
    target = torch.randn(1024, 1, device=device)
    
    # HybridBFO uses gradients for faster convergence
    optimizer = HybridBFO(
        model.parameters(),
        population_size=5,  # Small population
        gradient_weight=0.5,  # Balance BFO and gradients
        compile_mode=False  # Often faster without compile for hybrid
    )
    
    def closure():
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()  # Compute gradients for hybrid mode
        return loss.item()
    
    print("Configuration:")
    print(f"  Device: {device}")
    print(f"  Optimizer: HybridBFO (uses gradients)")
    print(f"  Population size: 5")
    print(f"  Gradient weight: 0.5")
    
    start = time.time()
    for i in range(5):
        loss = optimizer.step(closure)
        print(f"Step {i+1}: Loss = {loss:.6f}")
    
    print(f"Time: {time.time() - start:.2f}s")
    print("💡 HybridBFO often converges faster on differentiable problems!")


def print_recommendations():
    """Print GPU optimization recommendations."""
    print("\n" + "=" * 50)
    print("GPU OPTIMIZATION RECOMMENDATIONS")
    print("=" * 50)
    
    print("""
1. **Fix Graph Breaks**: Add at the top of your script:
   ```python
   import torch._dynamo as dynamo
   dynamo.config.capture_scalar_outputs = True
   ```

2. **Population Size**: Use 5-10 for GPU (not 50+)
   - Smaller populations = less serial overhead
   - Better GPU utilization

3. **Batch Size**: Use larger batches (256-1024+)
   - Amortizes kernel launch overhead
   - Better GPU throughput

4. **Use torch.no_grad()**: In fitness closures
   - Skips unnecessary gradient computation
   - Reduces memory usage

5. **Consider HybridBFO**: For differentiable problems
   - Combines BFO exploration with gradient exploitation
   - Often faster convergence on GPU

6. **Compile Mode**: 
   - Start with compile_mode=False for debugging
   - Use compile_kwargs={"mode": "default"} for smaller GPUs
   - Profile to verify speedup

7. **Memory Tips**:
   - Clear cache between runs: torch.cuda.empty_cache()
   - Monitor usage: nvidia-smi -l 1
   - Reduce population if OOM

8. **When to Use CPU**:
   - Very small models/batches
   - Population size > 20 required
   - Debugging compilation issues
""")


def main():
    print("PyTorch BFO GPU Optimization Demo")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        # Clear any existing allocations
        torch.cuda.empty_cache()
    
    # Run demos
    demo_old_way()
    demo_new_way()
    demo_tensor_closure()
    demo_hybrid_gpu()
    
    # Print recommendations
    print_recommendations()


if __name__ == "__main__":
    main()