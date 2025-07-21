"""
Test suite for PyTorch BFO Optimizer
Tests compatibility with PyTorch 2.8+ and torch.compile
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from pytorch_bfo_optimizer import BFO, AdaptiveBFO, HybridBFO


@pytest.fixture
def simple_model():
    """Create a simple linear model for testing."""
    torch.manual_seed(42)
    model = nn.Linear(2, 1)
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = torch.tensor([[5.0], [11.0], [17.0]])
    criterion = nn.MSELoss()
    return model, X, y, criterion


@pytest.fixture
def device():
    """Get appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestBFO:
    """Test basic BFO optimizer functionality."""
    
    def test_initialization(self, simple_model):
        """Test optimizer initialization with various parameters."""
        model, _, _, _ = simple_model
        
        # Default initialization
        optimizer = BFO(model.parameters())
        assert optimizer.defaults["population_size"] == 50
        assert optimizer.defaults["step_size_max"] == 0.1
        
        # Custom initialization
        optimizer = BFO(
            model.parameters(),
            population_size=20,
            step_size_max=0.05,
            compile_mode=False
        )
        assert optimizer.defaults["population_size"] == 20
        assert optimizer.defaults["step_size_max"] == 0.05
        
    def test_invalid_params(self):
        """Test invalid parameter handling."""
        with pytest.raises(ValueError):
            BFO([], population_size=0)
            
        with pytest.raises(ValueError):
            BFO([], step_size_max=0.01, step_size_min=0.02)
            
        with pytest.raises(ValueError):
            BFO([], levy_alpha=3.0)
            
    def test_convergence(self, simple_model):
        """Test that BFO reduces loss over iterations."""
        model, X, y, criterion = simple_model
        
        optimizer = BFO(
            model.parameters(),
            population_size=10,
            chem_steps=2,
            repro_steps=2,
            elim_steps=1,
            step_size_max=0.05,
            compile_mode=False  # Disable for testing
        )
        
        initial_loss = criterion(model(X), y).item()
        losses = [initial_loss]
        
        def closure():
            outputs = model(X)
            loss = criterion(outputs, y)
            return loss.item()
            
        for _ in range(10):
            fitness = optimizer.step(closure)
            losses.append(fitness)
            
        # Check that loss decreased
        assert losses[-1] < initial_loss
        assert min(losses) == losses[-1]  # Best fitness should be returned
        
    def test_device_handling(self, simple_model, device):
        """Test optimizer works correctly on different devices."""
        model, X, y, criterion = simple_model
        model = model.to(device)
        X = X.to(device)
        y = y.to(device)
        
        optimizer = BFO(model.parameters(), population_size=5, compile_mode=False)
        
        def closure():
            outputs = model(X)
            loss = criterion(outputs, y)
            return loss.item()
            
        # Should not raise any errors
        optimizer.step(closure)
        
    def test_closure_requirement(self, simple_model):
        """Test that optimizer requires closure."""
        model, _, _, _ = simple_model
        optimizer = BFO(model.parameters())
        
        with pytest.raises(ValueError, match="requires a closure"):
            optimizer.step()
            
    def test_zero_grad(self, simple_model):
        """Test zero_grad functionality."""
        model, X, y, criterion = simple_model
        optimizer = BFO(model.parameters(), compile_mode=False)
        
        # Create some gradients
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # Check gradients exist
        for p in model.parameters():
            assert p.grad is not None
            
        # Zero gradients
        optimizer.zero_grad()
        
        # Check gradients are None (with set_to_none=True)
        for p in model.parameters():
            assert p.grad is None
            
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compile_mode(self, simple_model):
        """Test torch.compile integration."""
        model, X, y, criterion = simple_model
        model = model.cuda()
        X = X.cuda()
        y = y.cuda()
        
        optimizer = BFO(
            model.parameters(),
            population_size=5,
            compile_mode=True
        )
        
        def closure():
            outputs = model(X)
            loss = criterion(outputs, y)
            return loss.item()
            
        # Should use compiled version
        optimizer.step(closure)
        

class TestAdaptiveBFO:
    """Test AdaptiveBFO functionality."""
    
    def test_adaptation(self, simple_model):
        """Test adaptive parameter adjustment."""
        model, X, y, criterion = simple_model
        
        optimizer = AdaptiveBFO(
            model.parameters(),
            population_size=10,
            adaptation_rate=0.2,
            compile_mode=False
        )
        
        initial_step_max = optimizer.defaults["step_size_max"]
        initial_elim_prob = optimizer.defaults["elim_prob"]
        
        def closure():
            outputs = model(X)
            loss = criterion(outputs, y)
            return loss.item()
            
        # Run enough iterations to trigger adaptation
        for _ in range(10):
            optimizer.step(closure)
            
        # Parameters should have adapted
        assert len(optimizer.fitness_history) > 0
        assert len(optimizer.diversity_history) > 0
        
    def test_diversity_computation(self, simple_model):
        """Test diversity metric computation."""
        model, _, _, _ = simple_model
        
        optimizer = AdaptiveBFO(
            model.parameters(),
            population_size=5,
            compile_mode=False
        )
        
        diversity = optimizer._compute_diversity()
        assert isinstance(diversity, float)
        assert diversity >= 0
        

class TestHybridBFO:
    """Test HybridBFO functionality."""
    
    def test_gradient_integration(self, simple_model):
        """Test hybrid optimizer with gradient information."""
        model, X, y, criterion = simple_model
        
        optimizer = HybridBFO(
            model.parameters(),
            population_size=5,
            gradient_weight=0.3,
            compile_mode=False
        )
        
        def closure():
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()  # Compute gradients
            return loss.item()
            
        initial_params = [p.clone() for p in model.parameters()]
        optimizer.step(closure)
        
        # Parameters should have changed
        for p_init, p_new in zip(initial_params, model.parameters()):
            assert not torch.equal(p_init, p_new)
            
    def test_momentum_buffers(self, simple_model):
        """Test momentum buffer initialization and usage."""
        model, _, _, _ = simple_model
        
        optimizer = HybridBFO(
            model.parameters(),
            use_momentum=True,
            momentum=0.9,
            compile_mode=False
        )
        
        # Check momentum buffers initialized
        assert len(optimizer.momentum_buffers) == len(list(model.parameters()))
        for buf in optimizer.momentum_buffers:
            assert torch.all(buf == 0)
            

class TestIntegration:
    """Integration tests with real optimization scenarios."""
    
    def test_nonconvex_optimization(self):
        """Test on a non-convex optimization problem."""
        torch.manual_seed(42)
        
        # Rosenbrock function
        def rosenbrock(x):
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
            
        x = nn.Parameter(torch.tensor([0.0, 0.0]))
        optimizer = BFO([x], population_size=20, compile_mode=False)
        
        def closure():
            return rosenbrock(x).item()
            
        initial_loss = closure()
        for _ in range(50):
            optimizer.step(closure)
            
        final_loss = closure()
        assert final_loss < initial_loss
        
    def test_neural_network_training(self):
        """Test training a small neural network."""
        torch.manual_seed(42)
        
        # Generate synthetic data
        X = torch.randn(100, 10)
        y = torch.sum(X * torch.randn(10), dim=1, keepdim=True) + 0.1 * torch.randn(100, 1)
        
        # Define model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        
        criterion = nn.MSELoss()
        optimizer = BFO(model.parameters(), population_size=10, compile_mode=False)
        
        def closure():
            outputs = model(X)
            loss = criterion(outputs, y)
            return loss.item()
            
        initial_loss = closure()
        for _ in range(20):
            optimizer.step(closure)
            
        final_loss = closure()
        assert final_loss < initial_loss * 0.5  # Expect significant improvement
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_performance(self):
        """Test performance on GPU with larger model."""
        torch.manual_seed(42)
        device = torch.device("cuda")
        
        # Larger model and data
        model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        ).to(device)
        
        X = torch.randn(1000, 100, device=device)
        y = torch.randn(1000, 1, device=device)
        
        criterion = nn.MSELoss()
        optimizer = BFO(
            model.parameters(),
            population_size=20,
            compile_mode=True  # Enable compilation
        )
        
        def closure():
            outputs = model(X)
            loss = criterion(outputs, y)
            return loss.item()
            
        # Should complete without errors
        optimizer.step(closure)