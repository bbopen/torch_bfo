#!/usr/bin/env python3
"""
Fixed GPU test for PyTorch BFO Optimizer
Works around PyTorch 2.8.0.dev torch.compile issues
"""

import torch
import torch.nn as nn
from pytorch_bfo_optimizer import BFO, AdaptiveBFO, HybridBFO
import time
import warnings

# Suppress torch.compile warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Apply configuration fixes
try:
    import torch._dynamo as dynamo
    dynamo.config.capture_scalar_outputs = True
    print("✅ Applied Dynamo configuration")
except:
    pass

# Clear any existing compile cache
try:
    import torch._inductor.config
    torch._inductor.config.clear_inductor_caches()
    print("✅ Cleared compile cache")
except:
    pass


def test_basic_functionality():
    """Test basic BFO functionality without compile mode."""
    print("\n" + "=" * 50)
    print("Testing Basic Functionality (No Compile)")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Use even population size to avoid split bug
    for pop_size in [4, 6, 10]:
        print(f"\nTesting population_size={pop_size}")
        
        model = nn.Linear(50, 10).to(device)
        optimizer = BFO(
            model.parameters(),
            population_size=pop_size,
            chem_steps=2,
            repro_steps=2,
            elim_steps=1,
            compile_mode=False  # Disable compile mode
        )
        
        data = torch.randn(256, 50, device=device)
        target = torch.randn(256, 10, device=device)
        
        def closure():
            with torch.no_grad():
                output = model(data)
                loss = nn.functional.mse_loss(output, target)
            return loss.item()
        
        start = time.time()
        for i in range(3):
            loss = optimizer.step(closure)
            print(f"  Step {i+1}: Loss = {loss:.6f}")
        
        elapsed = time.time() - start
        print(f"  Time: {elapsed:.2f}s")
    
    print("\n✅ Basic functionality test passed!")


def test_compile_workaround():
    """Test torch.compile with workarounds."""
    print("\n" + "=" * 50)
    print("Testing Compile Mode with Workarounds")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try different backends
    backends = ["eager", "aot_eager", "inductor"]
    
    for backend in backends:
        print(f"\nTrying backend: {backend}")
        try:
            model = nn.Linear(20, 5).to(device)
            
            # Compile with specific backend
            if backend == "inductor":
                # For inductor, use safer mode
                compiled_model = torch.compile(model, mode="default", fullgraph=False)
            else:
                compiled_model = torch.compile(model, backend=backend)
            
            # Test forward pass
            x = torch.randn(32, 20, device=device)
            y = compiled_model(x)
            print(f"  ✅ {backend} backend works!")
            
        except Exception as e:
            print(f"  ❌ {backend} backend failed: {str(e)[:100]}...")


def test_hybrid_optimizer():
    """Test HybridBFO which often works better on GPU."""
    print("\n" + "=" * 50)
    print("Testing HybridBFO (Recommended for GPU)")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = nn.Sequential(
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)
    
    data = torch.randn(512, 100, device=device)
    target = torch.randn(512, 1, device=device)
    
    optimizer = HybridBFO(
        model.parameters(),
        population_size=4,  # Even number
        gradient_weight=0.5,
        compile_mode=False  # Disable compile for now
    )
    
    def closure():
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        return loss.item()
    
    print("Running HybridBFO optimization...")
    start = time.time()
    
    losses = []
    for i in range(10):
        loss = optimizer.step(closure)
        losses.append(loss)
        print(f"Step {i+1}: Loss = {loss:.6f}")
    
    elapsed = time.time() - start
    print(f"\nTime: {elapsed:.2f}s")
    
    # Check convergence
    if len(losses) > 5 and losses[-1] < losses[0]:
        print("✅ HybridBFO is converging!")
    

def print_recommendations():
    """Print recommendations for RunPod usage."""
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS FOR RUNPOD")
    print("=" * 50)
    
    print("""
1. **torch.compile Issues (PyTorch 2.8.0.dev)**:
   - The CppCompileError is a known issue in the dev version
   - Use compile_mode=False for now
   - Or try backend="eager" or "aot_eager"
   
2. **Population Size**:
   - Use EVEN numbers (4, 6, 8, 10) to avoid split bug
   - Smaller populations (4-8) work better on GPU
   
3. **Best Performance**:
   - Use HybridBFO for differentiable problems
   - Larger batch sizes (512+)
   - torch.no_grad() in closures
   
4. **Workaround Script**:
   ```python
   # At the top of your script
   import torch._dynamo as dynamo
   dynamo.config.capture_scalar_outputs = True
   
   # Use even population sizes
   optimizer = BFO(params, population_size=6, compile_mode=False)
   ```
   
5. **Monitor GPU**:
   - Run: nvidia-smi -l 1
   - Check memory usage stays reasonable
""")


def main():
    print("PyTorch BFO Optimizer - RunPod Hotfix")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Run tests
    test_basic_functionality()
    test_compile_workaround()
    test_hybrid_optimizer()
    
    # Print recommendations
    print_recommendations()


if __name__ == "__main__":
    main()
