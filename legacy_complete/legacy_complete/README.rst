pytorch_bfo_optimizer
=====================

A PyTorch 2.8+ extension providing Bacterial Foraging Optimization (BFO) as a gradient-free/hybrid optimizer for neural network training. Features torch.compile support for optimal performance and modern PyTorch patterns.

Features
--------

- **PyTorch 2.8+ Native**: Built with latest PyTorch features and torch.compile optimization
- **Adaptive Step Sizes**: Dynamic adaptation with Lévy flights for efficient exploration
- **Vectorized Operations**: CUDA-accelerated operations for GPU performance
- **Hybrid Mode**: Optional gradient integration for faster convergence
- **Drop-in Replacement**: Compatible with ``torch.optim.Optimizer`` interface
- **torch.compile Support**: Optimized performance with JIT compilation
- **Multiple Variants**: Base BFO, AdaptiveBFO, and HybridBFO

Installation
------------

.. code-block:: bash

   pip install pytorch_bfo_optimizer

Requirements
~~~~~~~~~~~~

- Python 3.10+
- PyTorch 2.8.0+
- NumPy 1.24.0+

Quick Start
-----------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from pytorch_bfo_optimizer import BFO

   # Define model and data
   model = nn.Linear(10, 1).cuda()
   optimizer = BFO(model.parameters(), population_size=50)

   # Define closure for BFO
   def closure():
       optimizer.zero_grad()
       output = model(data)
       loss = criterion(output, target)
       return loss.item()

   # Optimization step
   optimizer.step(closure)

With torch.compile
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Enable torch.compile optimization (default)
   optimizer = BFO(model.parameters(), compile_mode=True)

   # Or compile your model for additional performance
   model = torch.compile(model)
   optimizer = BFO(model.parameters())

Adaptive Variant
~~~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_bfo_optimizer import AdaptiveBFO

   # Automatically adjusts parameters during optimization
   optimizer = AdaptiveBFO(
       model.parameters(),
       adaptation_rate=0.1,
       diversity_threshold=0.01
   )

Hybrid Variant
~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_bfo_optimizer import HybridBFO

   # Combines BFO with gradient information
   optimizer = HybridBFO(
       model.parameters(),
       gradient_weight=0.5,
       use_momentum=True
   )

   def closure():
       optimizer.zero_grad()
       output = model(data)
       loss = criterion(output, target)
       loss.backward()  # Compute gradients for hybrid mode
       return loss.item()

Advanced Examples
-----------------

Training Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from pytorch_bfo_optimizer import BFO

   # Create model
   model = nn.Sequential(
       nn.Linear(784, 128),
       nn.ReLU(),
       nn.Linear(128, 10)
   ).cuda()

   # Setup optimizer
   optimizer = BFO(
       model.parameters(),
       population_size=30,
       step_size_max=0.1,
       use_swarming=True,
       compile_mode=True
   )

   criterion = nn.CrossEntropyLoss()

   # Training loop
   for epoch in range(num_epochs):
       for batch_data, batch_target in dataloader:
           batch_data = batch_data.cuda()
           batch_target = batch_target.cuda()
           
           def closure():
               output = model(batch_data)
               loss = criterion(output, batch_target)
               return loss.item()
           
           loss = optimizer.step(closure)
           print(f"Epoch {epoch}, Loss: {loss:.4f}")

Non-Convex Optimization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from pytorch_bfo_optimizer import AdaptiveBFO

   # Optimize Rosenbrock function
   x = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
   optimizer = AdaptiveBFO([x], population_size=50)

   def rosenbrock():
       return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

   for i in range(100):
       loss = optimizer.step(rosenbrock)
       if i % 10 == 0:
           print(f"Iteration {i}, Loss: {loss:.6f}, x: {x.data}")

API Reference
-------------

BFO Parameters
~~~~~~~~~~~~~~

- **population_size** (int): Number of bacteria in population (default: 50)
- **chem_steps** (int): Chemotaxis steps per reproduction (default: 10)
- **swim_length** (int): Maximum swim steps (default: 4)
- **repro_steps** (int): Reproduction steps per elimination (default: 4)
- **elim_steps** (int): Elimination-dispersal steps (default: 2)
- **elim_prob** (float): Base elimination probability (default: 0.25)
- **step_size_max** (float): Maximum step size (default: 0.1)
- **step_size_min** (float): Minimum step size (default: 0.01)
- **levy_alpha** (float): Lévy flight parameter, 1.0-2.0 (default: 1.5)
- **use_swarming** (bool): Enable swarming behavior (default: False)
- **swarming_params** (tuple): (d_attract, w_attract, h_repel, w_repel)
- **device** (str): Device to run on (default: auto-detect)
- **compile_mode** (bool): Use torch.compile optimization (default: True)

Performance Tips
----------------

1. **GPU Acceleration**: Always use CUDA when available for best performance
2. **torch.compile**: Keep ``compile_mode=True`` for optimized execution
3. **Population Size**: Larger populations explore better but compute slower
4. **Batch Processing**: Process multiple samples in closure for efficiency
5. **Hyperparameters**: Start with defaults, then tune based on problem

Algorithm Details
-----------------

BFO simulates the foraging behavior of E. coli bacteria through:

1. **Chemotaxis**: Bacteria move toward nutrients via tumble and swim
2. **Swarming**: Cell-to-cell attraction and repulsion
3. **Reproduction**: Healthier bacteria split, weaker ones die
4. **Elimination-Dispersal**: Random elimination with dispersal to new locations

The implementation features:

- Lévy flights for improved exploration
- Adaptive step sizes based on iteration progress
- Vectorized operations for GPU efficiency
- Optional gradient hybridization

Citation
--------

If you use this library, please cite:

.. code-block:: bibtex

   @software{pytorch_bfo_optimizer,
     author = {Brett Bonner},
     title = {pytorch_bfo_optimizer: Bacterial Foraging Optimization for PyTorch 2.8+},
     version = {0.1.0},
     year = {2025},
     url = {https://github.com/brettbonner/pytorch-bfo-optimizer},
   }

References
----------

- Passino, K. M. (2002). Biomimicry of bacterial foraging for distributed optimization and control. IEEE Control Systems Magazine.
- PyTorch Documentation: https://pytorch.org/docs/stable/

License
-------

MIT License - see LICENSE file for details.