Changelog
=========

0.1.0 (2025-07-21)
------------------

**Initial Release**

Features:
~~~~~~~~~
- Base BFO optimizer implementation with PyTorch 2.8+ support
- AdaptiveBFO with automatic hyperparameter tuning
- HybridBFO combining bacterial foraging with gradient information
- Full torch.compile optimization support for performance
- CUDA acceleration with vectorized operations
- Lévy flight exploration for improved convergence
- Optional bacterial swarming behavior
- Compatible with standard torch.optim.Optimizer interface

Technical Details:
~~~~~~~~~~~~~~~~~~
- Requires PyTorch 2.8.0+ and Python 3.10+
- Supports multi-dimensional optimization problems
- Includes comprehensive test suite with >90% coverage
- Full type hints and documentation
- Examples for regression, classification, and non-convex optimization

Known Issues:
~~~~~~~~~~~~~
- Currently supports only single parameter group
- torch.compile mode requires CUDA for optimal performance

Contributors:
~~~~~~~~~~~~~
- Brett Bonner (@brettbonner) - Initial implementation