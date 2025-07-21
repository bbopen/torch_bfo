pytorch_bfo_optimizer Documentation
===================================

Welcome to the documentation for pytorch_bfo_optimizer, a PyTorch 2.8+ implementation of 
Bacterial Foraging Optimization with torch.compile support.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   quickstart
   api_reference
   examples
   contributing

Overview
--------

pytorch_bfo_optimizer provides gradient-free and hybrid optimization algorithms based on 
bacterial foraging behavior. Key features include:

* **PyTorch 2.8+ Native**: Built with latest PyTorch features
* **torch.compile Support**: Optimized performance with JIT compilation
* **Multiple Variants**: BFO, AdaptiveBFO, and HybridBFO
* **GPU Acceleration**: Fully vectorized operations
* **Easy Integration**: Drop-in replacement for torch.optim optimizers

Quick Example
-------------

.. code-block:: python

   import torch
   import torch.nn as nn
   from pytorch_bfo_optimizer import BFO

   model = nn.Linear(10, 1).cuda()
   optimizer = BFO(model.parameters(), population_size=50)

   def closure():
       output = model(data)
       loss = criterion(output, target)
       return loss.item()

   optimizer.step(closure)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`