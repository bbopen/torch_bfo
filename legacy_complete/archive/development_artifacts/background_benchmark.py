#!/usr/bin/env python3
"""
Background benchmark runner for GPU server
Runs the benchmark suite with monitoring and timeout protection
"""

import subprocess
import sys
import time
import os
import signal
import psutil
import argparse

def get_gpu_info():
    """Get GPU utilization info"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        gpu_util, mem_used, mem_total = result.stdout.strip().split(", ")
        return f"GPU: {gpu_util}%, Memory: {mem_used}/{mem_total} MB"
    except:
        return "GPU info unavailable"

def monitor_process(proc, log_file, timeout=3600):
    """Monitor a process with timeout and resource tracking"""
    start_time = time.time()
    
    while proc.poll() is None:
        elapsed = time.time() - start_time
        
        # Check timeout
        if elapsed > timeout:
            print(f"\nTimeout reached ({timeout}s), terminating process...")
            proc.terminate()
            time.sleep(5)
            if proc.poll() is None:
                proc.kill()
            break
        
        # Get system info
        try:
            process = psutil.Process(proc.pid)
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            status = f"[{elapsed:.0f}s] PID: {proc.pid}, CPU: {cpu_percent:.1f}%, Memory: {memory_mb:.1f}MB, {get_gpu_info()}"
            print(f"\r{status}", end="", flush=True)
            
            # Log status
            with open(log_file, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {status}\n")
        except:
            pass
        
        time.sleep(10)  # Update every 10 seconds
    
    return proc.returncode

def main():
    parser = argparse.ArgumentParser(description="Run BFO benchmarks in background")
    parser.add_argument("--log-file", default="benchmark_monitor.log", help="Monitoring log file")
    parser.add_argument("--output-file", default="benchmark_output.log", help="Benchmark output file")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds (default: 3600)")
    parser.add_argument("--reduced", action="store_true", help="Run reduced benchmark for testing")
    args = parser.parse_args()
    
    print(f"Starting BFO benchmark suite...")
    print(f"Monitor log: {args.log_file}")
    print(f"Output log: {args.output_file}")
    print(f"Timeout: {args.timeout}s")
    
    # Set environment for debugging
    env = os.environ.copy()
    env["BFO_DEBUG_LEVEL"] = "INFO"
    env["PYTHONUNBUFFERED"] = "1"
    
    # Prepare benchmark command
    if args.reduced:
        # Create a reduced benchmark script
        reduced_script = """
import sys
sys.path.insert(0, '.')
from benchmarks.benchmark_suite import *

# Reduced benchmark
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running reduced benchmark on {device}")

benchmark = OptimizerBenchmark(device=device)

# Test only a few optimizers on one task
optimizers_to_test = ['Adam', 'BFO', 'BFOv2', 'HybridBFOv2']
benchmark.get_optimizers = lambda params, lr=0.01: {
    k: v for k, v in benchmark.get_optimizers(params, lr).items() 
    if k in optimizers_to_test
}

# Run only Rosenbrock with fewer iterations
benchmark.benchmark_function_optimization('Rosenbrock', BenchmarkTasks.rosenbrock, iterations=50)

# Generate report
benchmark.generate_report('benchmark_results_reduced')
print("\\nReduced benchmark complete!")
"""
        with open("reduced_benchmark.py", "w") as f:
            f.write(reduced_script)
        cmd = [sys.executable, "reduced_benchmark.py"]
    else:
        cmd = [sys.executable, "benchmarks/benchmark_suite.py"]
    
    # Start benchmark process
    with open(args.output_file, "w") as outfile:
        proc = subprocess.Popen(
            cmd,
            stdout=outfile,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )
    
    print(f"Benchmark started with PID: {proc.pid}")
    
    # Monitor the process
    try:
        returncode = monitor_process(proc, args.log_file, args.timeout)
        print(f"\n\nBenchmark completed with return code: {returncode}")
    except KeyboardInterrupt:
        print("\n\nInterrupted, terminating benchmark...")
        proc.terminate()
        time.sleep(5)
        if proc.poll() is None:
            proc.kill()
    
    # Print last few lines of output
    print("\nLast output:")
    try:
        with open(args.output_file, "r") as f:
            lines = f.readlines()
            for line in lines[-20:]:
                print(line.rstrip())
    except:
        pass

if __name__ == "__main__":
    main()