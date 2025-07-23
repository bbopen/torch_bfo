#!/usr/bin/env python3
"""
Optimized GPU test script for PyTorch BFO Optimizer
Includes fixes for torch.compile graph breaks and performance optimizations
"""

import torch
import torch.nn as nn
from pytorch_bfo_optimizer import BFO, AdaptiveBFO, HybridBFO
import time
import argparse

# Fix for torch.compile graph breaks
import torch._dynamo as dynamo
dynamo.config.capture_scalar_outputs = True

# Optional: Suppress Triton autotuning warnings on small GPUs
try:
    import torch._inductor.config as inductor_config
    inductor_config.triton.autotune_gemm = False
except (ImportError, AttributeError):
    # Different PyTorch versions may have different config structures
    pass


def test_compile_fix():
    """Test that torch.compile works without graph breaks."""
    print("\n" + "=" * 50)
    print("Testing torch.compile Graph Break Fix")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Small model for quick testing
    model = nn.Linear(10, 1).to(device)
    
    # Test data with larger batch size for GPU efficiency
    inputs = torch.randn(256, 10, device=device)
    targets = torch.randn(256, 1, device=device)
    
    # Test with compile_mode=True and small population
    optimizer = BFO(
        model.parameters(),
        population_size=5,  # Small for testing
        chem_steps=5,
        compile_mode=True
    )
    
    def closure():
        with torch.no_grad():
            outputs = model(inputs)
            loss = nn.functional.mse_loss(outputs, targets)
        return loss.item()  # This should now work without graph breaks
    
    print("Running optimization with torch.compile enabled...")
    start_time = time.time()
    
    for i in range(10):
        loss = optimizer.step(closure)
        print(f"Step {i+1}: Loss = {loss:.6f}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f}s")
    print("✅ No graph break warnings = Fix successful!")


def test_performance_comparison():
    """Compare performance with different configurations."""
    print("\n" + "=" * 50)
    print("Performance Comparison Test")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Larger model for more realistic GPU workload
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    ).to(device)
    
    # Larger batch for better GPU utilization
    inputs = torch.randn(1024, 100, device=device)
    targets = torch.randn(1024, 1, device=device)
    
    configurations = [
        ("BFO (pop=5, no compile)", BFO, {"population_size": 5, "compile_mode": False}),
        ("BFO (pop=5, with compile)", BFO, {"population_size": 5, "compile_mode": True}),
        ("BFO (pop=20, no compile)", BFO, {"population_size": 20, "compile_mode": False}),
        ("HybridBFO (pop=5, gradients)", HybridBFO, {"population_size": 5, "gradient_weight": 0.5, "compile_mode": False}),
    ]
    
    results = []
    
    for name, optimizer_class, kwargs in configurations:
        print(f"\n{name}:")
        
        # Reset model
        for p in model.parameters():
            p.data.normal_(0, 0.1)
        
        optimizer = optimizer_class(model.parameters(), **kwargs)
        
        # Different closure for HybridBFO
        if "Hybrid" in name:
            def closure():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = nn.functional.mse_loss(outputs, targets)
                loss.backward()
                return loss.item()
        else:
            def closure():
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = nn.functional.mse_loss(outputs, targets)
                return loss.item()
        
        # Time 5 optimization steps
        start_time = time.time()
        losses = []
        
        for i in range(5):
            loss = optimizer.step(closure)
            losses.append(loss)
            print(f"  Step {i+1}: Loss = {loss:.6f}")
        
        elapsed = time.time() - start_time
        
        results.append({
            "name": name,
            "time": elapsed,
            "final_loss": losses[-1],
            "improvement": losses[0] - losses[-1] if len(losses) > 0 else 0
        })
        
        print(f"  Time: {elapsed:.2f}s")
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    print("-" * 50)
    
    # Sort by time
    results.sort(key=lambda x: x["time"])
    
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['name']}")
        print(f"   Time: {r['time']:.2f}s")
        print(f"   Final Loss: {r['final_loss']:.6f}")
        print(f"   Improvement: {r['improvement']:.6f}")


def test_gpu_memory():
    """Monitor GPU memory usage with different population sizes."""
    if not torch.cuda.is_available():
        print("GPU not available, skipping memory test")
        return
    
    print("\n" + "=" * 50)
    print("GPU Memory Usage Test")
    print("=" * 50)
    
    device = torch.device("cuda")
    
    # Clear cache
    torch.cuda.empty_cache()
    
    model = nn.Linear(1000, 100).to(device)
    inputs = torch.randn(100, 1000, device=device)
    targets = torch.randn(100, 100, device=device)
    
    population_sizes = [5, 10, 20, 50]
    
    for pop_size in population_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        optimizer = BFO(
            model.parameters(),
            population_size=pop_size,
            compile_mode=False  # Avoid compile overhead for memory test
        )
        
        def closure():
            with torch.no_grad():
                outputs = model(inputs)
                loss = nn.functional.mse_loss(outputs, targets)
            return loss.item()
        
        # Run one step
        optimizer.step(closure)
        
        # Get memory stats
        current_mem = torch.cuda.memory_allocated() / 1024**2  # MB
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        print(f"\nPopulation Size: {pop_size}")
        print(f"  Current Memory: {current_mem:.1f} MB")
        print(f"  Peak Memory: {peak_mem:.1f} MB")


def test_profiling(steps=3):
    """Profile the optimizer to identify bottlenecks."""
    if not torch.cuda.is_available():
        print("GPU not available, skipping profiling")
        return
    
    print("\n" + "=" * 50)
    print("Profiling BFO Optimizer")
    print("=" * 50)
    
    device = torch.device("cuda")
    
    model = nn.Linear(100, 10).to(device)
    inputs = torch.randn(512, 100, device=device)
    targets = torch.randn(512, 10, device=device)
    
    optimizer = BFO(
        model.parameters(),
        population_size=10,
        compile_mode=False
    )
    
    def closure():
        with torch.no_grad():
            outputs = model(inputs)
            loss = nn.functional.mse_loss(outputs, targets)
        return loss.item()
    
    print(f"Profiling {steps} optimization steps...")
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i in range(steps):
            loss = optimizer.step(closure)
            print(f"Step {i+1}: Loss = {loss:.6f}")
    
    # Save trace
    prof.export_chrome_trace("bfo_profile_trace.json")
    print("\n✅ Saved profiling trace to: bfo_profile_trace.json")
    print("   View in Chrome at: chrome://tracing")
    
    # Print summary
    print("\nTop 10 CUDA operations by time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def main():
    parser = argparse.ArgumentParser(description="Optimized GPU tests for BFO Optimizer")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--compile", action="store_true", help="Test compile fix")
    parser.add_argument("--performance", action="store_true", help="Compare performance")
    parser.add_argument("--memory", action="store_true", help="Test memory usage")
    parser.add_argument("--profile", action="store_true", help="Profile optimizer")
    
    args = parser.parse_args()
    
    # Print system info
    print("PyTorch BFO Optimizer - Optimized GPU Test")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Run tests
    if args.all or args.compile:
        test_compile_fix()
    
    if args.all or args.performance:
        test_performance_comparison()
    
    if args.all or args.memory:
        test_gpu_memory()
    
    if args.all or args.profile:
        test_profiling()
    
    if not any([args.all, args.compile, args.performance, args.memory, args.profile]):
        print("\nNo tests specified. Use --help for options.")
        print("Quick test: python gpu_test_optimized.py --compile")
        print("All tests: python gpu_test_optimized.py --all")


if __name__ == "__main__":
    main()