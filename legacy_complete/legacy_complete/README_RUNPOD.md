# RunPod GPU Testing Guide

This guide helps you test the PyTorch BFO Optimizer on RunPod GPU instances.

## Quick Start

1. **Clone the repository on your RunPod instance:**
   ```bash
   git clone -b runpod-gpu-testing https://github.com/yourusername/pytorch-bfo-optimizer.git
   cd pytorch-bfo-optimizer
   ```

2. **Run the setup script:**
   ```bash
   chmod +x runpod_setup.sh
   ./runpod_setup.sh
   ```

3. **Run tests:**
   ```bash
   # Run all tests
   python runpod_test.py --all
   
   # Quick benchmark only
   python runpod_test.py --quick
   
   # Memory usage test
   python runpod_test.py --memory
   ```

## Test Components

### 1. System Information
- PyTorch version verification
- CUDA availability and version
- GPU specifications and memory

### 2. torch.compile Performance Test
- Compares performance with and without torch.compile
- Tests all three optimizer variants (BFO, AdaptiveBFO, HybridBFO)
- Measures speedup from JIT compilation

### 3. Large Scale Test
- Tests on a larger neural network (2048 → 1024 → 512 → 256 → 100)
- Uses compiled model for additional performance
- Demonstrates real-world usage

### 4. Memory Usage Test
- Monitors GPU memory consumption
- Tests different population sizes
- Reports peak memory usage

## Expected Results

### Performance with torch.compile:
- 10-30% speedup for standard operations
- Higher speedup for larger models
- Best performance on newer GPUs (A100, H100)

### Memory Usage:
- Base optimizer: ~100-200MB
- Scales linearly with population size
- Additional memory for model parameters

## Troubleshooting

### CUDA Version Mismatch
If you get CUDA errors, adjust the PyTorch installation in `runpod_setup.sh`:
```bash
# For CUDA 12.1
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory
Reduce population size or batch size:
```python
optimizer = BFO(model.parameters(), population_size=10)  # Smaller population
```

### Compilation Errors
If torch.compile fails, disable it:
```python
optimizer = BFO(model.parameters(), compile_mode=False)
```

## Recommended RunPod Configurations

### Minimum Requirements:
- GPU: RTX 3090 or better
- VRAM: 16GB+
- CUDA: 11.8+
- Python: 3.10+

### Optimal Setup:
- GPU: A100 40GB or H100
- CUDA: 12.1+
- PyTorch: 2.8.0+

## Running Production Workloads

For production use:
```python
import torch
from pytorch_bfo_optimizer import BFO

# Enable all optimizations
model = torch.compile(your_model)
optimizer = BFO(
    model.parameters(),
    population_size=50,
    compile_mode=True,
    use_swarming=True
)
```

## Monitoring Performance

Use the built-in PyTorch profiler:
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    optimizer.step(closure)
    
print(prof.key_averages().table(sort_by="cuda_time_total"))
```