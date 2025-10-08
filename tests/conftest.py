"""
Pytest configuration for BFO tests.

Sets default device to CUDA when available for faster testing.
"""

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Default device for tests - use CUDA if available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def set_default_device(device):
    """Automatically set default tensor type to use GPU if available."""
    if device.type == "cuda":
        torch.set_default_device(device)
        yield
        torch.set_default_device("cpu")  # Reset after test
    else:
        yield

