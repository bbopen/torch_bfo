"""
Test suite for BFO optimizers.

Tests basic functionality, device handling, state management,
and optimizer behavior across different scenarios.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from bfo_torch import BFO, AdaptiveBFO, HybridBFO


class SimpleModel(nn.Module):
    """Simple test model."""
    def __init__(self, input_size=10, hidden_size=5, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def model():
    """Create test model."""
    torch.manual_seed(42)
    return SimpleModel()


@pytest.fixture
def data():
    """Create test data."""
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    return X, y


class TestBFO:
    """Test standard BFO optimizer."""
    
    def test_initialization(self, model):
        """Test optimizer initialization."""
        optimizer = BFO(model.parameters(), lr=0.01)
        
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]['lr'] == 0.01
        assert optimizer.param_groups[0]['population_size'] == 50
        assert isinstance(optimizer.device, torch.device)
    
    def test_parameter_validation(self, model):
        """Test parameter validation."""
        # Test invalid learning rate
        with pytest.raises(ValueError, match="Invalid learning rate"):
            BFO(model.parameters(), lr=-0.01)
        
        # Test invalid population size
        with pytest.raises(ValueError, match="Invalid population size"):
            BFO(model.parameters(), population_size=0)
        
        # Test invalid elimination probability
        with pytest.raises(ValueError, match="Invalid elimination probability"):
            BFO(model.parameters(), elimination_prob=1.5)
    
    def test_step_requires_closure(self, model):
        """Test that step requires closure."""
        optimizer = BFO(model.parameters())
        
        with pytest.raises(ValueError, match="BFO requires a closure"):
            optimizer.step()
    
    def test_basic_optimization(self, model, data):
        """Test basic optimization step."""
        X, y = data
        optimizer = BFO(model.parameters(), lr=0.01, population_size=10)
        
        def closure():
            optimizer.zero_grad()
            output = model(X)
            loss = F.mse_loss(output, y)
            return loss.item()
        
        initial_loss = closure()
        final_loss = optimizer.step(closure)
        
        assert isinstance(final_loss, float)
        assert final_loss >= 0
    
    def test_multiple_steps(self, model, data):
        """Test multiple optimization steps."""
        X, y = data
        optimizer = BFO(model.parameters(), lr=0.01, population_size=5)
        
        def closure():
            optimizer.zero_grad()
            output = model(X)
            loss = F.mse_loss(output, y)
            return loss.item()
        
        losses = []
        for _ in range(3):
            loss = optimizer.step(closure)
            losses.append(loss)
        
        assert len(losses) == 3
        assert all(isinstance(loss, float) for loss in losses)
    
    def test_state_dict(self, model):
        """Test state dictionary save/load."""
        optimizer = BFO(model.parameters(), lr=0.01, population_size=5)
        
        # Initialize state by taking a step
        def closure():
            return torch.tensor(1.0).item()
        
        optimizer.step(closure)
        
        # Save state
        state_dict = optimizer.state_dict()
        assert 'bfo_state' in state_dict
        assert 'rng_state' in state_dict
        
        # Create new optimizer and load state
        new_optimizer = BFO(model.parameters(), lr=0.02, population_size=10)
        new_optimizer.load_state_dict(state_dict)
        
        # Check that state was loaded correctly
        assert new_optimizer.param_groups[0]['lr'] == 0.01  # Should be restored
    
    def test_device_handling(self, model):
        """Test device parameter handling."""
        # Test CPU
        optimizer = BFO(model.parameters(), device=torch.device('cpu'))
        assert optimizer.device == torch.device('cpu')
        
        # Test CUDA if available
        if torch.cuda.is_available():
            cuda_model = model.cuda()
            optimizer = BFO(cuda_model.parameters())
            assert optimizer.device.type == 'cuda'
    
    def test_mixed_precision(self):
        """Test mixed precision support."""
        model = SimpleModel().half()  # FP16
        optimizer = BFO(model.parameters(), lr=0.01, population_size=5)
        
        def closure():
            X = torch.randn(10, 10, dtype=torch.float16)
            y = torch.randn(10, 1, dtype=torch.float16)
            output = model(X)
            loss = F.mse_loss(output, y)
            return loss.item()
        
        loss = optimizer.step(closure)
        assert isinstance(loss, float)
    
    def test_multiple_parameter_groups(self):
        """Test multiple parameter groups."""
        model = SimpleModel()
        
        optimizer = BFO([
            {'params': model.fc1.parameters(), 'lr': 0.01, 'population_size': 5},
            {'params': model.fc2.parameters(), 'lr': 0.001, 'population_size': 3}
        ])
        
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]['lr'] == 0.01
        assert optimizer.param_groups[1]['lr'] == 0.001


class TestAdaptiveBFO:
    """Test Adaptive BFO optimizer."""
    
    def test_initialization(self, model):
        """Test AdaptiveBFO initialization."""
        optimizer = AdaptiveBFO(
            model.parameters(),
            lr=0.01,
            adaptation_rate=0.1,
            min_population_size=5,
            max_population_size=50
        )
        
        assert optimizer.adaptation_rate == 0.1
        assert optimizer.min_population_size == 5
        assert optimizer.max_population_size == 50
    
    def test_basic_optimization(self, model, data):
        """Test basic AdaptiveBFO optimization."""
        X, y = data
        optimizer = AdaptiveBFO(model.parameters(), lr=0.01, population_size=5)
        
        def closure():
            optimizer.zero_grad()
            output = model(X)
            loss = F.mse_loss(output, y)
            return loss.item()
        
        loss = optimizer.step(closure)
        assert isinstance(loss, float)


class TestHybridBFO:
    """Test Hybrid BFO optimizer."""
    
    def test_initialization(self, model):
        """Test HybridBFO initialization."""
        optimizer = HybridBFO(
            model.parameters(),
            lr=0.01,
            gradient_weight=0.5,
            momentum=0.9
        )
        
        assert optimizer.gradient_weight == 0.5
        assert optimizer.momentum == 0.9
        assert optimizer.enable_momentum == True
    
    def test_gradient_handling(self, model, data):
        """Test gradient handling in HybridBFO."""
        X, y = data
        optimizer = HybridBFO(model.parameters(), lr=0.01, population_size=5)
        
        def closure():
            optimizer.zero_grad()
            output = model(X)
            loss = F.mse_loss(output, y)
            loss.backward()  # Compute gradients
            return loss.item()
        
        loss = optimizer.step(closure)
        assert isinstance(loss, float)
    
    def test_no_gradients(self, model, data):
        """Test HybridBFO without gradients."""
        X, y = data
        optimizer = HybridBFO(model.parameters(), lr=0.01, population_size=5)
        
        def closure():
            # No gradients computed
            optimizer.zero_grad()
            output = model(X)
            loss = F.mse_loss(output, y)
            return loss.item()
        
        loss = optimizer.step(closure)
        assert isinstance(loss, float)


class TestOptimizationFunctions:
    """Test mathematical optimization functions."""
    
    def test_rosenbrock_function(self):
        """Test optimization on Rosenbrock function."""
        # Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
        x = nn.Parameter(torch.tensor([-1.2, 1.0]))
        optimizer = BFO([x], lr=0.01, population_size=20)
        
        def closure():
            loss = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
            return loss.item()
        
        initial_loss = closure()
        
        # Run optimization
        for _ in range(5):
            loss = optimizer.step(closure)
        
        # Should improve
        assert loss < initial_loss
    
    def test_sphere_function(self):
        """Test optimization on sphere function."""
        # Sphere function: f(x) = sum(x_i^2)
        x = nn.Parameter(torch.randn(5) * 5)  # Start far from optimum
        optimizer = AdaptiveBFO([x], lr=0.01, population_size=15)
        
        def closure():
            loss = (x**2).sum()
            return loss.item()
        
        initial_loss = closure()
        
        # Run optimization
        for _ in range(3):
            loss = optimizer.step(closure)
        
        # Should improve significantly
        assert loss < initial_loss


if __name__ == "__main__":
    pytest.main([__file__])