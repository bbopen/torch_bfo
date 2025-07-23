#!/usr/bin/env python3
"""
Background GPU test runner with monitoring for PyTorch BFO Optimizer

Usage:
    python background_gpu_test.py [--log-file FILE] [--check-interval SECONDS]
    
    Run in background:
    nohup python background_gpu_test.py --log-file bfo_test.log > output.log 2>&1 &
    
    Monitor progress:
    tail -f bfo_test.log
    tail -f output.log
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from datetime import datetime
import signal
import psutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pytorch_bfo_optimizer import BFO, AdaptiveBFO, HybridBFO


class TestMonitor:
    """Monitor test execution and system resources"""
    
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.start_time = time.time()
        self.process = psutil.Process()
        
    def log(self, message):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')
                f.flush()
    
    def log_system_stats(self):
        """Log system resource usage"""
        try:
            # CPU and memory
            cpu_percent = self.process.cpu_percent(interval=0.1)
            mem_info = self.process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024
            
            # GPU stats if available
            gpu_stats = ""
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                gpu_util = gpu_mem / gpu_total * 100
                gpu_stats = f", GPU: {gpu_mem:.0f}/{gpu_total:.0f}MB ({gpu_util:.1f}%)"
            
            self.log(f"Resources - CPU: {cpu_percent:.1f}%, Memory: {mem_mb:.0f}MB{gpu_stats}")
        except Exception as e:
            self.log(f"Error getting system stats: {e}")


def test_optimizer(optimizer_class, name, monitor, device='cuda'):
    """Test an optimizer with monitoring"""
    monitor.log(f"\n{'='*60}")
    monitor.log(f"Testing {name}")
    monitor.log(f"{'='*60}")
    
    try:
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        ).to(device)
        
        # Test data
        batch_size = 256
        data = torch.randn(batch_size, 10).to(device)
        target = torch.randn(batch_size, 1).to(device)
        
        # Create optimizer with minimal settings for faster testing
        kwargs = {
            'population_size': 4,
            'chem_steps': 2,
            'swim_length': 2,
            'repro_steps': 1,
            'elim_steps': 1,
            'compile_mode': False,  # Disable due to PyTorch 2.8.0.dev bug
            'verbose': True  # Enable verbose logging
        }
        
        if optimizer_class == HybridBFO:
            kwargs['gradient_weight'] = 0.5
        
        monitor.log(f"Creating {name} optimizer...")
        optimizer = optimizer_class(model.parameters(), **kwargs)
        
        # Closure function
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = nn.MSELoss()(output, target)
            if isinstance(optimizer, HybridBFO):
                loss.backward()
            return loss.item()
        
        # Run optimization steps
        monitor.log("Running optimization steps...")
        losses = []
        
        for step in range(5):
            monitor.log(f"Step {step + 1}/5")
            start = time.time()
            
            loss = optimizer.step(closure)
            elapsed = time.time() - start
            losses.append(loss)
            
            monitor.log(f"  Loss: {loss:.6f}, Time: {elapsed:.3f}s")
            monitor.log_system_stats()
            
            # Check for timeout
            if time.time() - monitor.start_time > 300:  # 5 minute timeout
                monitor.log("WARNING: Test timeout reached!")
                break
        
        # Summary
        monitor.log(f"\n{name} Summary:")
        monitor.log(f"  Initial loss: {losses[0]:.6f}")
        monitor.log(f"  Final loss: {losses[-1]:.6f}")
        monitor.log(f"  Improvement: {losses[0] - losses[-1]:.6f}")
        monitor.log(f"  ✓ {name} completed successfully!")
        
        return True
        
    except Exception as e:
        monitor.log(f"ERROR in {name}: {type(e).__name__}: {e}")
        import traceback
        monitor.log(traceback.format_exc())
        return False


def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print("\nReceived interrupt signal. Exiting...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Background GPU test for BFO Optimizer')
    parser.add_argument('--log-file', default='bfo_gpu_test.log', help='Log file path')
    parser.add_argument('--check-interval', type=int, default=10, help='Status check interval (seconds)')
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create monitor
    monitor = TestMonitor(args.log_file)
    
    # Log environment info
    monitor.log("PyTorch BFO Optimizer - Background GPU Test")
    monitor.log(f"PyTorch version: {torch.__version__}")
    monitor.log(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        monitor.log(f"GPU: {torch.cuda.get_device_name(0)}")
        monitor.log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Set environment variables for debugging
        os.environ['BFO_DEBUG_LEVEL'] = 'INFO'
        os.environ['BFO_LOG_FILE'] = args.log_file.replace('.log', '_debug.log')
        
        # Test optimizers
        optimizers = [
            (BFO, "BFO"),
            (AdaptiveBFO, "AdaptiveBFO"),
            (HybridBFO, "HybridBFO")
        ]
        
        results = []
        for opt_class, name in optimizers:
            success = test_optimizer(opt_class, name, monitor)
            results.append((name, success))
            time.sleep(2)  # Brief pause between tests
        
        # Final summary
        monitor.log("\n" + "="*60)
        monitor.log("TEST SUMMARY")
        monitor.log("="*60)
        
        all_passed = True
        for name, success in results:
            status = "PASSED" if success else "FAILED"
            monitor.log(f"{name}: {status}")
            if not success:
                all_passed = False
        
        total_time = time.time() - monitor.start_time
        monitor.log(f"\nTotal execution time: {total_time:.1f}s")
        
        if all_passed:
            monitor.log("\n✅ All tests passed!")
        else:
            monitor.log("\n❌ Some tests failed!")
            sys.exit(1)
    else:
        monitor.log("ERROR: CUDA not available!")
        sys.exit(1)


if __name__ == "__main__":
    main()