"""PyTorch BFO Optimizer - Bacterial Foraging Optimization for PyTorch 2.8+"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"

# Legacy API (V1) - Original implementation
from .optimizer import BFO, AdaptiveBFO, HybridBFO

# Production API (V3) - Production-ready implementation with all fixes
from .optimizer_v3_fixed import BFOv2, AdaptiveBFOv2, HybridBFOv2

# Debug utilities
from .debug_utils import enable_debug_mode, disable_debug_mode, print_debug_instructions

# Main exports - Legacy classes for backward compatibility
__all__ = [
    # Legacy V1 API (for backward compatibility)
    "BFO", "AdaptiveBFO", "HybridBFO",
    
    # Production V3 API (recommended for new code)
    "BFOv2", "AdaptiveBFOv2", "HybridBFOv2",
    
    # Debug utilities
    "enable_debug_mode", "disable_debug_mode", "print_debug_instructions"
]