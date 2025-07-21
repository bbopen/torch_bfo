"""PyTorch BFO Optimizer - Bacterial Foraging Optimization for PyTorch 2.8+"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"

from .optimizer import BFO, AdaptiveBFO, HybridBFO

__all__ = ["BFO", "AdaptiveBFO", "HybridBFO"]