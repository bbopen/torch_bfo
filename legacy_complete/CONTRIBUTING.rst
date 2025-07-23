Contributing to pytorch_bfo_optimizer
=====================================

We welcome contributions to pytorch_bfo_optimizer! This document provides guidelines
for contributing to the project.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/your-username/pytorch-bfo-optimizer.git
      cd pytorch-bfo-optimizer

3. Create a new branch for your feature or bugfix:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

4. Install the package in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

Development Workflow
--------------------

Code Style
~~~~~~~~~~

We use Black for code formatting and follow PEP 8 guidelines:

.. code-block:: bash

   make format  # Format code
   make lint    # Check code style

Type Checking
~~~~~~~~~~~~~

We use mypy for static type checking:

.. code-block:: bash

   make type-check

Testing
~~~~~~~

All new features should include tests. Run the test suite with:

.. code-block:: bash

   make test

For specific tests:

.. code-block:: bash

   pytest tests/test_optimizer.py::TestBFO::test_convergence -v

Documentation
~~~~~~~~~~~~~

Update documentation for any new features:

.. code-block:: bash

   make docs

Pull Request Process
--------------------

1. Ensure all tests pass and code is properly formatted
2. Update the README.rst if needed
3. Add an entry to CHANGES.rst
4. Ensure your commits have clear, descriptive messages
5. Push to your fork and submit a pull request
6. Wait for review and address any feedback

Guidelines
----------

Code Quality
~~~~~~~~~~~~

- Write clear, self-documenting code
- Add docstrings to all public functions and classes
- Include type hints for all function arguments and returns
- Keep functions focused and modular

Testing
~~~~~~~

- Aim for high test coverage (>90%)
- Test edge cases and error conditions
- Use pytest fixtures for common test setups
- Mock external dependencies when appropriate

Performance
~~~~~~~~~~~

- Profile code changes for performance impact
- Ensure torch.compile compatibility
- Optimize for GPU execution where possible
- Document any performance considerations

Areas for Contribution
----------------------

We especially welcome contributions in these areas:

- Additional optimizer variants
- Performance optimizations
- Documentation improvements
- Example notebooks
- Benchmark comparisons
- Bug fixes

Questions?
----------

Feel free to open an issue for any questions about contributing!