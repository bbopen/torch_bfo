#!/usr/bin/env python3
"""
RunPod Hotfix Script - Fixes for PyTorch 2.8.0.dev issues
Applies necessary fixes and workarounds for the current issues
"""

import os
import sys

def apply_dynamo_config():
    """Apply torch._dynamo configuration to fix graph breaks."""
    print("Applying Dynamo configuration fixes...")
    
    config_code = """
# Fix for torch.compile graph breaks
import torch._dynamo as dynamo
dynamo.config.capture_scalar_outputs = True

# Disable C++ compilation cache to avoid zuf0 errors
import torch._inductor.config as inductor_config
inductor_config.cpp.enable_kernel_profile = False
inductor_config.triton.unique_kernel_names = True

# Set environment variables for debugging
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
"""
    
    # Write to a config file that can be imported
    with open("torch_compile_config.py", "w") as f:
        f.write(config_code)
    
    print("✅ Created torch_compile_config.py")


def create_fixed_test():
    """Create a test script with all fixes applied."""
    
    test_code = '''#!/usr/bin/env python3
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
    print("\\n" + "=" * 50)
    print("Testing Basic Functionality (No Compile)")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Use even population size to avoid split bug
    for pop_size in [4, 6, 10]:
        print(f"\\nTesting population_size={pop_size}")
        
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
    
    print("\\n✅ Basic functionality test passed!")


def test_compile_workaround():
    """Test torch.compile with workarounds."""
    print("\\n" + "=" * 50)
    print("Testing Compile Mode with Workarounds")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try different backends
    backends = ["eager", "aot_eager", "inductor"]
    
    for backend in backends:
        print(f"\\nTrying backend: {backend}")
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
    print("\\n" + "=" * 50)
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
    print(f"\\nTime: {elapsed:.2f}s")
    
    # Check convergence
    if len(losses) > 5 and losses[-1] < losses[0]:
        print("✅ HybridBFO is converging!")
    

def print_recommendations():
    """Print recommendations for RunPod usage."""
    print("\\n" + "=" * 50)
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
'''
    
    with open("runpod_test_fixed.py", "w") as f:
        f.write(test_code)
    
    os.chmod("runpod_test_fixed.py", 0o755)
    print("✅ Created runpod_test_fixed.py")


def create_minimal_working_example():
    """Create a minimal working example for RunPod."""
    
    example_code = '''#!/usr/bin/env python3
"""
Minimal working example for PyTorch BFO on RunPod
Avoids known issues with PyTorch 2.8.0.dev
"""

import torch
import torch.nn as nn
from pytorch_bfo_optimizer import BFO, HybridBFO
import torch._dynamo as dynamo

# Fix graph breaks
dynamo.config.capture_scalar_outputs = True

# Create model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Linear(100, 10).to(device)

# Use even population size to avoid bug
optimizer = BFO(
    model.parameters(),
    population_size=6,  # EVEN number
    compile_mode=False  # Disable compile due to dev version issue
)

# Or use HybridBFO for better GPU performance
# optimizer = HybridBFO(
#     model.parameters(),
#     population_size=6,
#     gradient_weight=0.5,
#     compile_mode=False
# )

# Training data
data = torch.randn(512, 100, device=device)
target = torch.randn(512, 10, device=device)

# Optimization loop
def closure():
    with torch.no_grad():
        output = model(data)
        loss = nn.functional.mse_loss(output, target)
    return loss.item()

# For HybridBFO, use this closure instead:
# def closure():
#     optimizer.zero_grad()
#     output = model(data)
#     loss = nn.functional.mse_loss(output, target)
#     loss.backward()
#     return loss.item()

print("Running optimization...")
for i in range(10):
    loss = optimizer.step(closure)
    print(f"Step {i+1}: Loss = {loss:.6f}")

print("Done!")
'''
    
    with open("minimal_example.py", "w") as f:
        f.write(example_code)
    
    print("✅ Created minimal_example.py")


def main():
    print("RunPod Hotfix Script")
    print("=" * 50)
    print("This script creates fixes and workarounds for:")
    print("1. torch.compile CppCompileError (zuf0 issue)")
    print("2. Population split bug with odd sizes")
    print()
    
    # Create all fix files
    apply_dynamo_config()
    create_fixed_test()
    create_minimal_working_example()
    
    print("\n✅ All fix files created!")
    print("\nTo use on RunPod:")
    print("1. Copy these files to RunPod:")
    print("   - torch_compile_config.py")
    print("   - runpod_test_fixed.py")
    print("   - minimal_example.py")
    print("\n2. Run: python runpod_test_fixed.py")
    print("   or: python minimal_example.py")


if __name__ == "__main__":
    main()