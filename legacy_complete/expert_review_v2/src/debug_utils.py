"""Debug utilities for PyTorch BFO Optimizer"""

import logging
import os
import time
from typing import Optional, Dict, Any
import torch
from functools import wraps

# Configure logging based on environment variables
DEBUG_LEVEL = os.environ.get('BFO_DEBUG_LEVEL', 'WARNING')
LOG_FILE = os.environ.get('BFO_LOG_FILE', None)

# Create logger
logger = logging.getLogger('pytorch_bfo_optimizer')
logger.setLevel(getattr(logging, DEBUG_LEVEL))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging, DEBUG_LEVEL))
console_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler if specified
if LOG_FILE:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(getattr(logging, DEBUG_LEVEL))
    file_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)


def timing_decorator(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.debug(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


class OptimizationLogger:
    """Logger for tracking optimization progress"""
    
    def __init__(self, optimizer_name: str, log_interval: int = 10):
        self.optimizer_name = optimizer_name
        self.log_interval = log_interval
        self.step_count = 0
        self.history: Dict[str, list] = {
            'losses': [],
            'times': [],
            'diversities': [],
            'step_sizes': []
        }
    
    def log_step(self, loss: float, diversity: Optional[float] = None, 
                 step_size: Optional[float] = None, elapsed: Optional[float] = None):
        """Log optimization step details"""
        self.step_count += 1
        self.history['losses'].append(loss)
        
        if elapsed is not None:
            self.history['times'].append(elapsed)
        
        if diversity is not None:
            self.history['diversities'].append(diversity)
        
        if step_size is not None:
            self.history['step_sizes'].append(step_size)
        
        if self.step_count % self.log_interval == 0:
            msg = f"[{self.optimizer_name}] Step {self.step_count}: Loss={loss:.6f}"
            if diversity is not None:
                msg += f", Diversity={diversity:.4f}"
            if step_size is not None:
                msg += f", StepSize={step_size:.4f}"
            if elapsed is not None:
                msg += f", Time={elapsed:.3f}s"
            logger.info(msg)
    
    def log_population_stats(self, population: torch.Tensor, fitness: torch.Tensor):
        """Log population statistics"""
        if logger.level <= logging.DEBUG:
            pop_mean = population.mean().item()
            pop_std = population.std().item()
            fitness_mean = fitness.mean().item()
            fitness_std = fitness.std().item()
            
            logger.debug(f"[{self.optimizer_name}] Population stats: "
                        f"mean={pop_mean:.4f}, std={pop_std:.4f}")
            logger.debug(f"[{self.optimizer_name}] Fitness stats: "
                        f"mean={fitness_mean:.4f}, std={fitness_std:.4f}, "
                        f"best={fitness.min().item():.4f}")
    
    def log_chemotaxis(self, bacterium_idx: int, tumble_count: int, swim_count: int):
        """Log chemotaxis details"""
        if logger.level <= logging.DEBUG:
            logger.debug(f"[{self.optimizer_name}] Bacterium {bacterium_idx}: "
                        f"tumbles={tumble_count}, swims={swim_count}")
    
    def log_reproduction(self, eliminated_count: int):
        """Log reproduction event"""
        logger.debug(f"[{self.optimizer_name}] Reproduction: "
                    f"eliminated {eliminated_count} bacteria")
    
    def log_elimination(self, eliminated_indices: list):
        """Log elimination event"""
        if eliminated_indices and logger.level <= logging.DEBUG:
            logger.debug(f"[{self.optimizer_name}] Elimination: "
                        f"dispersed bacteria {eliminated_indices}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get optimization summary"""
        if not self.history['losses']:
            return {}
        
        summary = {
            'total_steps': self.step_count,
            'final_loss': self.history['losses'][-1],
            'best_loss': min(self.history['losses']),
            'loss_reduction': self.history['losses'][0] - self.history['losses'][-1]
        }
        
        if self.history['times']:
            summary['total_time'] = sum(self.history['times'])
            summary['avg_time_per_step'] = summary['total_time'] / len(self.history['times'])
        
        if self.history['diversities']:
            summary['final_diversity'] = self.history['diversities'][-1]
            summary['avg_diversity'] = sum(self.history['diversities']) / len(self.history['diversities'])
        
        return summary
    
    def print_summary(self):
        """Print optimization summary"""
        summary = self.get_summary()
        if summary:
            logger.info(f"\n[{self.optimizer_name}] Optimization Summary:")
            for key, value in summary.items():
                logger.info(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")


def enable_debug_mode():
    """Enable debug mode for PyTorch BFO Optimizer"""
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)
    logger.info("Debug mode enabled for PyTorch BFO Optimizer")


def disable_debug_mode():
    """Disable debug mode"""
    logger.setLevel(logging.WARNING)
    for handler in logger.handlers:
        handler.setLevel(logging.WARNING)


# Environment variable helpers
def get_debug_settings():
    """Get current debug settings"""
    return {
        'debug_level': DEBUG_LEVEL,
        'log_file': LOG_FILE,
        'torch_logs': os.environ.get('TORCH_LOGS', ''),
        'torch_compile_debug': os.environ.get('TORCH_COMPILE_DEBUG', ''),
        'torchdynamo_verbose': os.environ.get('TORCHDYNAMO_VERBOSE', '')
    }


def print_debug_instructions():
    """Print instructions for enabling debug mode"""
    print("""
PyTorch BFO Optimizer Debug Instructions:

1. Enable debug logging:
   export BFO_DEBUG_LEVEL=DEBUG
   export BFO_LOG_FILE=/path/to/bfo_debug.log

2. Enable PyTorch compile debugging (if using compile_mode=True):
   export TORCH_LOGS="+dynamo,guards,recompiles"
   export TORCH_COMPILE_DEBUG=1
   export TORCHDYNAMO_VERBOSE=1

3. Run with verbose mode:
   optimizer = BFO(params, verbose=True)

4. Monitor in real-time:
   tail -f /path/to/bfo_debug.log

5. Background execution:
   nohup python your_script.py > output.log 2>&1 &
   # Monitor with: tail -f output.log
""")


if __name__ == "__main__":
    print_debug_instructions()