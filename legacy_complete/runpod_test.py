#!/usr/bin/env python3
"""
RunPod GPU Testing Script for PyTorch BFO Optimizer
Tests GPU performance, torch.compile optimization, and all optimizer variants
"""

import torch
import torch.nn as nn
import time
import argparse
from pytorch_bfo_optimizer import BFO, AdaptiveBFO, HybridBFO


def print_system_info():
    """Print system and GPU information."""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    print("=" * 60)
    print()


def benchmark_optimizer(optimizer_class, model, data, target, criterion, iterations=20, **kwargs):
    """Benchmark an optimizer variant."""
    model_copy = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).cuda()
    
    # Copy initial weights
    model_copy.load_state_dict(model.state_dict())
    
    # Create optimizer
    optimizer = optimizer_class(model_copy.parameters(), **kwargs)
    
    # Warmup
    for _ in range(3):
        def closure():
            optimizer.zero_grad()
            output = model_copy(data)
            loss = criterion(output, target)
            if hasattr(optimizer, 'gradient_weight'):  # HybridBFO
                loss.backward()
            return loss.item()
        
        optimizer.step(closure)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    losses = []
    for i in range(iterations):
        def closure():
            optimizer.zero_grad()
            output = model_copy(data)
            loss = criterion(output, target)
            if hasattr(optimizer, 'gradient_weight'):  # HybridBFO
                loss.backward()
            return loss.item()
        
        loss = optimizer.step(closure)
        losses.append(loss)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    return elapsed, losses


def test_torch_compile():
    """Test torch.compile optimization."""
    print("\n" + "=" * 60)
    print("TORCH.COMPILE PERFORMANCE TEST")
    print("=" * 60)
    
    # Create test data
    batch_size = 256
    data = torch.randn(batch_size, 1000).cuda()
    target = torch.randint(0, 10, (batch_size,)).cuda()
    
    # Create model
    model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).cuda()
    
    criterion = nn.CrossEntropyLoss()
    
    # Test configurations
    configs = [
        ("BFO (no compile)", BFO, {"compile_mode": False, "population_size": 20}),
        ("BFO (with compile)", BFO, {"compile_mode": True, "population_size": 20}),
        ("AdaptiveBFO (no compile)", AdaptiveBFO, {"compile_mode": False, "population_size": 20}),
        ("AdaptiveBFO (with compile)", AdaptiveBFO, {"compile_mode": True, "population_size": 20}),
        ("HybridBFO (no compile)", HybridBFO, {"compile_mode": False, "population_size": 20}),
        ("HybridBFO (with compile)", HybridBFO, {"compile_mode": True, "population_size": 20}),
    ]
    
    results = []
    
    for name, optimizer_class, kwargs in configs:
        print(f"\nTesting {name}...")
        try:
            elapsed, losses = benchmark_optimizer(
                optimizer_class, model, data, target, criterion, iterations=10, **kwargs
            )
            
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Initial Loss: {losses[0]:.4f}")
            print(f"  Final Loss: {losses[-1]:.4f}")
            print(f"  Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
            
            results.append({
                "name": name,
                "time": elapsed,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "improvement": (losses[0] - losses[-1]) / losses[0] * 100
            })
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results.append({
                "name": name,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    
    successful = [r for r in results if "error" not in r]
    if successful:
        # Compare compile vs no compile
        for base_name in ["BFO", "AdaptiveBFO", "HybridBFO"]:
            no_compile = next((r for r in successful if r["name"] == f"{base_name} (no compile)"), None)
            with_compile = next((r for r in successful if r["name"] == f"{base_name} (with compile)"), None)
            
            if no_compile and with_compile:
                speedup = (no_compile["time"] - with_compile["time"]) / no_compile["time"] * 100
                print(f"{base_name} torch.compile speedup: {speedup:.1f}%")


def test_large_scale():
    """Test on larger scale problem."""
    print("\n" + "=" * 60)
    print("LARGE SCALE OPTIMIZATION TEST")
    print("=" * 60)
    
    # Larger model
    model = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 100)
    ).cuda()
    
    # Compile model for additional performance
    compiled_model = torch.compile(model)
    
    # Large batch
    batch_size = 512
    data = torch.randn(batch_size, 2048).cuda()
    target = torch.randint(0, 100, (batch_size,)).cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = BFO(
        compiled_model.parameters(),
        population_size=30,
        compile_mode=True,
        use_swarming=True
    )
    
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    print("Testing BFO on large-scale problem...")
    
    # Training loop
    start_time = time.time()
    losses = []
    
    for epoch in range(5):
        def closure():
            optimizer.zero_grad()
            output = compiled_model(data)
            loss = criterion(output, target)
            return loss.item()
        
        loss = optimizer.step(closure)
        losses.append(loss)
        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Time per epoch: {elapsed / 5:.2f}s")


def test_memory_usage():
    """Test GPU memory usage."""
    print("\n" + "=" * 60)
    print("GPU MEMORY USAGE TEST")
    print("=" * 60)
    
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        initial_memory = torch.cuda.memory_allocated() / 1e9
        print(f"Initial GPU memory: {initial_memory:.2f} GB")
        
        # Create large model
        model = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        ).cuda()
        
        model_memory = torch.cuda.memory_allocated() / 1e9 - initial_memory
        print(f"Model memory: {model_memory:.2f} GB")
        
        # Create optimizer with different population sizes
        for pop_size in [10, 30, 50]:
            torch.cuda.empty_cache()
            
            optimizer = BFO(model.parameters(), population_size=pop_size)
            optimizer_memory = torch.cuda.memory_allocated() / 1e9 - initial_memory - model_memory
            print(f"BFO (population_size={pop_size}) memory: {optimizer_memory:.2f} GB")
            
            del optimizer
        
        # Peak memory
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPeak GPU memory usage: {peak_memory:.2f} GB")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="RunPod GPU Testing for PyTorch BFO Optimizer")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests")
    parser.add_argument("--memory", action="store_true", help="Run memory tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Print system info
    print_system_info()
    
    # Run tests
    if args.all or (not args.quick and not args.benchmark and not args.memory):
        # Run all tests
        test_torch_compile()
        test_large_scale()
        test_memory_usage()
    else:
        if args.quick or args.benchmark:
            test_torch_compile()
        if args.memory:
            test_memory_usage()
        if args.benchmark:
            test_large_scale()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()