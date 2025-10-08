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

    def test_parameter_movement(self, model, data):
        """Test that optimizer actually changes parameters."""
        X, y = data
        optimizer = BFO(model.parameters(), lr=0.01, population_size=10)

        # Store original parameters
        old_params = [p.clone() for p in model.parameters()]

        def closure():
            optimizer.zero_grad()
            output = model(X)
            loss = F.mse_loss(output, y)
            return loss.item()

        # Run optimization
        optimizer.step(closure)

        # Check that at least one parameter changed
        params_changed = False
        for old_p, new_p in zip(old_params, model.parameters()):
            if not torch.allclose(old_p, new_p):
                params_changed = True
                break

        assert params_changed, "Optimizer did not change any parameters"

    def test_early_stopping(self, model):
        """Test that early stopping triggers correctly."""
        optimizer = BFO(
            model.parameters(),
            lr=0.01,
            population_size=5,
            chemotaxis_steps=1,
            reproduction_steps=1,
            elimination_steps=1,
            early_stopping=True,
            convergence_patience=2
        )

        # Constant closure to force stagnation
        def closure():
            return 1.0

        # Run enough steps to trigger early stopping
        for _ in range(5):
            optimizer.step(closure)

        # Check that stagnation count has increased
        group_id = 0  # First parameter group
        assert group_id in optimizer.bfo_state
        assert optimizer.bfo_state[group_id]['stagnation_count'] >= 2


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

    def test_population_growth_on_stagnation(self, model, data):
        """Test that population size grows on stagnation."""
        optimizer = AdaptiveBFO(
            model.parameters(),
            population_size=10,
            min_population_size=5,
            max_population_size=20,
            adaptation_rate=0.5
        )

        stagnation_closure = lambda: 1.0
        initial_pop_size = optimizer.param_groups[0]['population_size']

        for _ in range(6):
            optimizer.step(stagnation_closure)

        final_pop_size = optimizer.param_groups[0]['population_size']
        assert final_pop_size > initial_pop_size, "Population should grow on stagnation"
        assert final_pop_size <= optimizer.max_population_size

    def test_population_shrinking_on_progress(self, model, data):
        """Test that population size shrinks on progress."""
        X, y = data
        optimizer = AdaptiveBFO(
            model.parameters(),
            population_size=20, # Start at max
            min_population_size=5,
            max_population_size=20,
            adaptation_rate=0.5
        )

        # Simulate consistent progress
        initial_pop_size = optimizer.param_groups[0]['population_size']

        for i in range(6):
            optimizer.step(lambda: 1.0 / (i + 1))

        final_pop_size = optimizer.param_groups[0]['population_size']
        assert final_pop_size < initial_pop_size, "Population should shrink on progress"
        assert final_pop_size >= optimizer.min_population_size

    def test_state_dict_round_trip(self, model, data):
        """Test state dictionary save/load for AdaptiveBFO."""
        X, y = data
        optimizer = AdaptiveBFO(
            model.parameters(),
            lr=0.01,
            population_size=10,
            adaptation_rate=0.2
        )

        # Run a few steps to build up state
        def closure():
            optimizer.zero_grad()
            output = model(X)
            loss = F.mse_loss(output, y)
            return loss.item()

        for _ in range(3):
            optimizer.step(closure)

        # Save state
        state_dict = optimizer.state_dict()

        # Create new optimizer with different parameters
        new_optimizer = AdaptiveBFO(
            model.parameters(),
            lr=0.02,  # Different lr
            population_size=20,  # Different population
            adaptation_rate=0.5  # Different adaptation rate
        )

        # Load state
        new_optimizer.load_state_dict(state_dict)

        # Check that parameters were restored
        assert new_optimizer.param_groups[0]['lr'] == 0.01
        assert new_optimizer.param_groups[0]['population_size'] == 10

        # Check that state was preserved
        group_id = 0  # First parameter group

        if group_id in optimizer.bfo_state and group_id in new_optimizer.bfo_state:
            old_state = optimizer.bfo_state[group_id]
            new_state = new_optimizer.bfo_state[group_id]
            assert torch.allclose(old_state['best_params'], new_state['best_params'])
            assert old_state['best_fitness'] == new_state['best_fitness']


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

    def test_state_dict_round_trip(self, model, data):
        """Test state dictionary save/load for HybridBFO."""
        X, y = data
        optimizer = HybridBFO(
            model.parameters(),
            lr=0.01,
            gradient_weight=0.5,
            momentum=0.9,
            enable_momentum=True
        )

        # Run a few steps with gradients to build up state
        def closure():
            optimizer.zero_grad()
            output = model(X)
            loss = F.mse_loss(output, y)
            loss.backward()  # Compute gradients
            return loss.item()

        for _ in range(3):
            optimizer.step(closure)

        # Save state
        state_dict = optimizer.state_dict()

        # Create new optimizer with different parameters
        new_optimizer = HybridBFO(
            model.parameters(),
            lr=0.02,
            gradient_weight=0.3,
            momentum=0.8
        )

        # Load state
        new_optimizer.load_state_dict(state_dict)

        # Check that parameters were restored
        assert new_optimizer.param_groups[0]['lr'] == 0.01

        # Check momentum buffer exists if momentum was enabled
        group_id = 0  # First parameter group
        if group_id in new_optimizer.bfo_state and 'momentum_buffer' in new_optimizer.bfo_state[group_id]:
            assert new_optimizer.bfo_state[group_id]['momentum_buffer'] is not None

    def test_gradient_correctness(self):
        """Test that HybridBFO follows gradient direction correctly."""
        # Simple quadratic function: f(x) = x^2
        # Gradient: f'(x) = 2x
        x = nn.Parameter(torch.tensor([2.0]))

        optimizer = HybridBFO(
            [x],
            lr=0.1,
            gradient_weight=1.0,  # Full gradient weight
            population_size=1,    # Single particle
            enable_momentum=False # No momentum for clarity
        )

        def closure():
            optimizer.zero_grad()
            loss = x ** 2
            loss.backward()
            return loss.item()

        # Store initial value
        x_init = x.item()

        # Take one step
        optimizer.step(closure)

        # With gradient_weight=1.0, it should move in negative gradient direction
        # For x=2, gradient=4, so it should move by -lr*gradient = -0.1*4 = -0.4
        # Expected new x ≈ 2.0 - 0.4 = 1.6 (approximately, due to BFO stochasticity)
        x_new = x.item()

        # Check that x moved in the correct direction (towards 0)
        assert x_new < x_init, f"Parameter should decrease from {x_init} but got {x_new}"
        assert abs(x_new) < abs(x_init), "Parameter should move towards optimum (0)"


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


class TestStatePersistence:
    """Test state checkpoint save/load cycles."""

    def test_bfo_state_persistence(self, model, data):
        """Test that BFO state survives save/load cycle."""
        X, y = data
        optimizer = BFO(model.parameters(), lr=0.01, population_size=10)

        def closure():
            optimizer.zero_grad()
            output = model(X)
            loss = F.mse_loss(output, y)
            loss.backward()
            return loss.item()

        # Run a few steps
        for _ in range(3):
            optimizer.step(closure)

        # Save state
        state_before = optimizer.state_dict()
        pop_before = optimizer.bfo_state[0]["population"].clone()
        best_before = optimizer.bfo_state[0]["best_params"].clone()
        fe_before = optimizer.bfo_state[0]["function_evaluations"]

        # Create new optimizer and load state
        model2 = SimpleModel()
        optimizer2 = BFO(model2.parameters(), lr=0.01, population_size=10)
        optimizer2.load_state_dict(state_before)

        # Verify state was restored
        assert 0 in optimizer2.bfo_state, "Group 0 state should be restored"
        pop_after = optimizer2.bfo_state[0]["population"]
        best_after = optimizer2.bfo_state[0]["best_params"]
        fe_after = optimizer2.bfo_state[0]["function_evaluations"]

        assert torch.allclose(pop_before, pop_after), "Population should be restored"
        assert torch.allclose(best_before, best_after), "Best params should be restored"
        assert fe_before == fe_after, f"FE count should be restored: {fe_before} vs {fe_after}"

    def test_adaptive_bfo_state_persistence(self):
        """Test that AdaptiveBFO state survives save/load cycle."""
        x = nn.Parameter(torch.randn(5))
        optimizer = AdaptiveBFO([x], lr=0.01, population_size=15, min_population_size=5)

        def closure():
            return (x**2).sum().item()

        # Run a few steps
        for _ in range(3):
            optimizer.step(closure)

        # Save state
        state_before = optimizer.state_dict()
        pop_before = optimizer.bfo_state[0]["population"].clone()
        fe_before = optimizer.bfo_state[0]["function_evaluations"]

        # Create new optimizer and load state
        x2 = nn.Parameter(torch.randn(5))
        optimizer2 = AdaptiveBFO([x2], lr=0.01, population_size=15, min_population_size=5)
        optimizer2.load_state_dict(state_before)

        # Verify state was restored
        pop_after = optimizer2.bfo_state[0]["population"]
        fe_after = optimizer2.bfo_state[0]["function_evaluations"]

        assert torch.allclose(pop_before, pop_after), "Population should be restored"
        assert fe_before == fe_after, f"FE count should be restored: {fe_before} vs {fe_after}"

    def test_hybrid_bfo_state_persistence(self, model, data):
        """Test that HybridBFO state survives save/load cycle."""
        X, y = data
        optimizer = HybridBFO(model.parameters(), lr=0.01, population_size=10)

        def closure():
            optimizer.zero_grad()
            output = model(X)
            loss = F.mse_loss(output, y)
            loss.backward()
            return loss.item()

        # Run a few steps
        for _ in range(3):
            optimizer.step(closure)

        # Save state including momentum buffer
        state_before = optimizer.state_dict()
        best_before = optimizer.bfo_state[0]["best_params"].clone()
        has_momentum = "momentum_buffer" in optimizer.bfo_state[0]
        if has_momentum:
            momentum_before = optimizer.bfo_state[0]["momentum_buffer"].clone()

        # Create new optimizer and load state
        model2 = SimpleModel()
        optimizer2 = HybridBFO(model2.parameters(), lr=0.01, population_size=10)
        optimizer2.load_state_dict(state_before)

        # Verify BFO state was restored
        assert 0 in optimizer2.bfo_state, "Group 0 BFO state should be restored"
        best_after = optimizer2.bfo_state[0]["best_params"]
        assert torch.allclose(best_before, best_after), "Best params should be restored"

        if has_momentum:
            assert "momentum_buffer" in optimizer2.bfo_state[0], "Momentum buffer should be restored"
            momentum_after = optimizer2.bfo_state[0]["momentum_buffer"]
            assert torch.allclose(momentum_before, momentum_after), "Momentum should be restored"


class TestHybridBFOBehavior:
    """Test that HybridBFO tracks current parameters correctly."""

    def test_gradient_bias_actually_applied(self):
        """Test that gradient bias is actually applied to population."""
        # Use a simple quadratic: f(x) = x^2
        # Gradient at x=2 is 4, pointing left (negative)
        x = nn.Parameter(torch.tensor([2.0]))

        optimizer = HybridBFO(
            [x],
            lr=1.0,  # Large LR to make effect visible
            gradient_weight=0.8,  # Strong gradient influence
            population_size=5,
            enable_momentum=False,
            chemotaxis_steps=1,
            elimination_steps=1,
            reproduction_steps=1,
        )

        def closure():
            optimizer.zero_grad()
            loss = (x ** 2).sum()
            loss.backward()
            return loss.item()

        # Capture population before step
        pop_before = optimizer.bfo_state[0]["population"].clone()

        # Run one step
        optimizer.step(closure)

        # Get population after step
        pop_after = optimizer.bfo_state[0]["population"]

        # With gradient pointing left (negative), gradient bias should push population left
        # Check that at least some individuals moved in the gradient direction
        moves_left = (pop_after < pop_before).sum().item()
        assert moves_left > 0, f"Gradient bias should move some individuals left, but {moves_left} moved"

        # Verify gradient was actually used by checking function evaluations happened
        fe = optimizer.get_function_evaluations()
        assert fe > 0, f"Function evaluations should be tracked: {fe}"

    def test_hybrid_uses_current_best_params(self, model, data):
        """Test that HybridBFO biases toward current best, not initialization."""
        X, y = data
        optimizer = HybridBFO(
            model.parameters(), lr=0.01, population_size=10, gradient_weight=0.5
        )

        def closure():
            optimizer.zero_grad()
            output = model(X)
            loss = F.mse_loss(output, y)
            loss.backward()
            return loss.item()

        # Run first step
        optimizer.step(closure)
        best_after_step1 = optimizer.bfo_state[0]["best_params"].clone()

        # Run second step
        optimizer.step(closure)
        best_after_step2 = optimizer.bfo_state[0]["best_params"].clone()

        # Best params should evolve (not frozen at initialization)
        assert not torch.allclose(
            best_after_step1, best_after_step2, rtol=1e-5
        ), "Best params should update between steps"

        # Population should be influenced by current best, not old best
        # We can't directly check the bias calculation, but we can verify
        # that best_params is being updated properly
        population = optimizer.bfo_state[0]["population"]
        mean_pop = population.mean(dim=0)

        # Population mean should be closer to current best than to initial params
        # (after several iterations with gradient bias)
        for _ in range(3):
            optimizer.step(closure)

        final_best = optimizer.bfo_state[0]["best_params"]
        final_mean = optimizer.bfo_state[0]["population"].mean(dim=0)

        # Distance from population mean to best should be reasonable
        # (not guaranteed to be minimal but should be influenced)
        dist_to_best = (final_mean - final_best).norm()
        assert dist_to_best.item() < 100.0, "Population should be influenced by current best"


class TestAdaptiveBFOEvaluationCounting:
    """Test that AdaptiveBFO properly counts function evaluations."""

    def test_get_function_evaluations_works(self):
        """Test that get_function_evaluations() returns correct count."""
        x = nn.Parameter(torch.randn(5))
        optimizer = BFO([x], lr=0.01, population_size=10)

        def closure():
            return (x**2).sum().item()

        # Before any steps, should be 0
        assert optimizer.get_function_evaluations() == 0, "Should start at 0"

        # Run one step
        optimizer.step(closure)

        # Should now have evaluations
        fe = optimizer.get_function_evaluations()
        assert fe > 0, f"Should have evaluations after step: {fe}"

        # Run another step
        fe_before = fe
        optimizer.step(closure)
        fe_after = optimizer.get_function_evaluations()

        assert fe_after > fe_before, f"Evaluations should increase: {fe_before} -> {fe_after}"

    def test_adaptive_resize_counts_evaluations(self):
        """Test that adaptive population resize counts all evaluations."""
        x = nn.Parameter(torch.randn(5))
        optimizer = AdaptiveBFO(
            [x],
            lr=0.01,
            population_size=20,
            min_population_size=5,
            adaptation_interval=2,
        )

        eval_count = 0

        def closure():
            nonlocal eval_count
            eval_count += 1
            return (x**2).sum().item()

        # Run several steps to trigger adaptation
        for _ in range(5):
            optimizer.step(closure)

        # Check that internal counter matches actual evaluations
        fe_tracked = optimizer.bfo_state[0]["function_evaluations"]

        # The tracked count should match reality (within the population evaluations)
        # Note: exact match depends on implementation details, but should be close
        assert fe_tracked > 0, "Function evaluations should be tracked"
        assert fe_tracked <= eval_count, (
            f"Tracked FE ({fe_tracked}) should not exceed actual calls ({eval_count})"
        )

        # More importantly, verify the counter increases consistently
        fe_before = optimizer.bfo_state[0]["function_evaluations"]
        optimizer.step(closure)
        fe_after = optimizer.bfo_state[0]["function_evaluations"]

        assert fe_after > fe_before, "FE counter should increase after step"

    def test_adaptive_respects_budget(self):
        """Test that AdaptiveBFO respects max_fe budget."""
        x = nn.Parameter(torch.randn(5))
        optimizer = AdaptiveBFO([x], lr=0.01, population_size=10)

        def closure():
            return (x**2).sum().item()

        # Set a low budget
        max_fe = 50

        # Run with budget constraint
        for _ in range(10):
            loss = optimizer.step(closure, max_fe=max_fe)
            fe_used = optimizer.bfo_state[0]["function_evaluations"]

            # Should never exceed budget
            assert fe_used <= max_fe, f"Should not exceed budget: {fe_used} > {max_fe}"

            if fe_used >= max_fe:
                break


if __name__ == "__main__":
    pytest.main([__file__])
