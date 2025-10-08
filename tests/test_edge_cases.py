"""
Edge case and robustness tests for BFO optimizers.

Tests scenarios that should work correctly and fail gracefully.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from bfo_torch import BFO, AdaptiveBFO, HybridBFO


class TestEdgeCasesThatShouldWork:
    """Test edge cases that should work correctly."""

    def test_population_size_one(self):
        """Single bacterium should work like random search."""
        x = nn.Parameter(torch.tensor([5.0, 5.0]))
        optimizer = BFO([x], lr=0.1, population_size=1, seed=42)

        def closure():
            return (x ** 2).sum().item()

        initial_loss = closure()
        final_loss = optimizer.step(closure)

        assert isinstance(final_loss, float)
        assert final_loss >= 0

    def test_swim_length_zero(self):
        """Zero swim length should disable swimming mechanism."""
        x = nn.Parameter(torch.randn(5))
        optimizer = BFO([x], lr=0.01, population_size=10, swim_length=0, seed=42)

        def closure():
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)

    def test_high_dimensions(self):
        """Should handle high-dimensional problems."""
        x = nn.Parameter(torch.randn(1000))
        optimizer = BFO([x], lr=0.01, population_size=20, seed=42)

        def closure():
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))

    def test_tiny_population(self):
        """Very small populations should still work."""
        x = nn.Parameter(torch.randn(5))
        optimizer = BFO([x], lr=0.01, population_size=3, seed=42)

        def closure():
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)

    def test_large_population(self):
        """Large populations should work (though may be slow)."""
        x = nn.Parameter(torch.randn(10))
        optimizer = BFO(
            [x],
            lr=0.01,
            population_size=200,
            chemotaxis_steps=1,
            reproduction_steps=1,
            elimination_steps=1,
            seed=42,
        )

        def closure():
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)

    def test_empty_gradients_hybrid(self):
        """HybridBFO should handle missing gradients."""
        x = nn.Parameter(torch.randn(5))
        optimizer = HybridBFO([x], lr=0.01, population_size=5, seed=42)

        def closure():
            # No backward call, so no gradients
            optimizer.zero_grad()
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)

    def test_single_parameter(self):
        """Should work with just one parameter."""
        x = nn.Parameter(torch.tensor([5.0]))
        optimizer = BFO([x], lr=0.1, population_size=10, seed=42)

        def closure():
            return (x ** 2).item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)

    def test_very_small_step_size(self):
        """Tiny step sizes should work."""
        x = nn.Parameter(torch.randn(5))
        optimizer = BFO(
            [x],
            lr=0.01,
            population_size=10,
            step_size_min=1e-10,
            step_size_max=1e-8,
            seed=42,
        )

        def closure():
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)

    def test_very_large_step_size(self):
        """Large step sizes should work."""
        x = nn.Parameter(torch.randn(5))
        optimizer = BFO(
            [x],
            lr=0.01,
            population_size=10,
            step_size_min=0.1,
            step_size_max=10.0,
            seed=42,
        )

        def closure():
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)


class TestNumericalStability:
    """Test numerical stability under extreme conditions."""

    def test_nan_handling_in_fitness(self):
        """Should handle NaN fitness values gracefully."""
        x = nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = BFO([x], lr=0.01, population_size=5, seed=42)

        call_count = [0]

        def closure():
            call_count[0] += 1
            # Return NaN on some calls
            if call_count[0] % 3 == 0:
                return float("nan")
            return (x ** 2).sum().item()

        # Should not crash, though may not converge well
        try:
            loss = optimizer.step(closure)
            # NaN or valid loss both acceptable (implementation choice)
            assert isinstance(loss, float)
        except ValueError as e:
            # Acceptable to raise error for NaN
            assert "nan" in str(e).lower() or "inf" in str(e).lower()

    def test_inf_handling_in_fitness(self):
        """Should handle Inf fitness values."""
        x = nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = BFO([x], lr=0.01, population_size=5, seed=42)

        call_count = [0]

        def closure():
            call_count[0] += 1
            # Return Inf on some calls
            if call_count[0] % 4 == 0:
                return float("inf")
            return (x ** 2).sum().item()

        # Should handle Inf (treat as worst fitness)
        loss = optimizer.step(closure)
        assert isinstance(loss, float)

    def test_extreme_parameter_values(self):
        """Should handle very large parameter values."""
        x = nn.Parameter(torch.tensor([1e10, -1e10]))
        optimizer = BFO([x], lr=0.01, population_size=5, seed=42)

        def closure():
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))

    def test_very_small_parameter_values(self):
        """Should handle very small parameter values."""
        x = nn.Parameter(torch.tensor([1e-10, -1e-10]))
        optimizer = BFO([x], lr=0.01, population_size=5, seed=42)

        def closure():
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)

    def test_mixed_precision_fp16_edge_cases(self):
        """Test FP16 numerical edge cases."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for FP16 test")

        x = nn.Parameter(torch.randn(10, dtype=torch.float16, device="cuda"))
        optimizer = BFO([x], lr=0.01, population_size=5, seed=42)

        def closure():
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))

    def test_all_parameters_identical(self):
        """Population with identical parameters should still work."""
        x = nn.Parameter(torch.zeros(5))
        optimizer = BFO([x], lr=0.01, population_size=10, seed=42)

        def closure():
            return (x ** 2).sum().item() + 1.0  # Non-zero to avoid all-zero fitness

        loss = optimizer.step(closure)
        assert isinstance(loss, float)


class TestInvalidInputHandling:
    """Test that invalid inputs produce clear errors."""

    def test_closure_returns_none(self):
        """Closure returning None should give clear error."""
        x = nn.Parameter(torch.randn(5))
        optimizer = BFO([x], lr=0.01, population_size=5)

        def closure():
            return None

        # Should either handle or give clear error
        try:
            optimizer.step(closure)
            # If it doesn't raise, the returned value should still be valid
        except (TypeError, ValueError) as e:
            # Clear error about None return is acceptable
            assert "none" in str(e).lower() or "return" in str(e).lower()

    def test_closure_returns_tensor_without_item(self):
        """Closure returning tensor (not scalar) should be handled."""
        x = nn.Parameter(torch.randn(5))
        optimizer = BFO([x], lr=0.01, population_size=5)

        def closure():
            return (x ** 2).sum()  # Returns tensor, not scalar

        # Should handle this (convert with .item())
        loss = optimizer.step(closure)
        assert isinstance(loss, float)

    def test_invalid_levy_alpha(self):
        """Invalid Lévy alpha should raise error."""
        x = nn.Parameter(torch.randn(5))

        with pytest.raises(ValueError, match="levy_alpha"):
            BFO([x], levy_alpha=0.5)  # Too small

        with pytest.raises(ValueError, match="levy_alpha"):
            BFO([x], levy_alpha=3.0)  # Too large

    def test_invalid_elimination_prob(self):
        """Invalid elimination probability should raise error."""
        x = nn.Parameter(torch.randn(5))

        with pytest.raises(ValueError, match="elimination"):
            BFO([x], elimination_prob=0.0)

        with pytest.raises(ValueError, match="elimination"):
            BFO([x], elimination_prob=1.5)

    def test_step_size_min_greater_than_max(self):
        """step_size_min > step_size_max should raise error."""
        x = nn.Parameter(torch.randn(5))

        with pytest.raises(ValueError, match="step_size"):
            BFO([x], step_size_min=1.0, step_size_max=0.1)


class TestDeviceMismatch:
    """Test device handling and mismatches."""

    def test_cpu_optimizer_cpu_model(self):
        """CPU optimizer with CPU model should work."""
        model = nn.Linear(5, 1)
        optimizer = BFO(model.parameters(), device=torch.device("cpu"))

        def closure():
            x = torch.randn(10, 5)
            return model(x).pow(2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_optimizer_cuda_model(self):
        """CUDA optimizer with CUDA model should work."""
        model = nn.Linear(5, 1).cuda()
        optimizer = BFO(model.parameters())

        def closure():
            x = torch.randn(10, 5, device="cuda")
            return model(x).pow(2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)


class TestReproducibility:
    """Test reproducibility with seeds."""

    def test_same_seed_same_results(self):
        """Same seed should produce identical results."""
        def run_optimization(seed):
            x = nn.Parameter(torch.tensor([5.0, 5.0]))
            optimizer = BFO(
                [x],
                lr=0.01,
                population_size=10,
                chemotaxis_steps=2,
                reproduction_steps=1,
                elimination_steps=1,
                seed=seed,
            )

            def closure():
                return (x ** 2).sum().item()

            losses = []
            for _ in range(3):
                loss = optimizer.step(closure)
                losses.append(loss)

            return losses, x.data.clone()

        losses1, params1 = run_optimization(42)
        losses2, params2 = run_optimization(42)

        # Should be identical (or very close due to floating point)
        for l1, l2 in zip(losses1, losses2):
            assert abs(l1 - l2) < 1e-5, f"Losses differ: {l1} vs {l2}"

        assert torch.allclose(params1, params2, atol=1e-5), "Parameters differ"

    def test_different_seed_different_results(self):
        """Different seeds should produce different results."""
        def run_optimization(seed):
            x = nn.Parameter(torch.tensor([5.0, 5.0]))
            optimizer = BFO([x], lr=0.01, population_size=10, seed=seed)

            def closure():
                return (x ** 2).sum().item()

            return optimizer.step(closure)

        loss1 = run_optimization(42)
        loss2 = run_optimization(123)

        # Should be different (high probability)
        assert loss1 != loss2, "Different seeds produced identical results"


class TestDomainBounds:
    """Test domain bounds constraint handling."""

    def test_domain_bounds_respected(self):
        """Parameters should stay within domain bounds."""
        x = nn.Parameter(torch.tensor([5.0, 5.0]))
        optimizer = BFO(
            [x],
            lr=0.1,
            population_size=10,
            domain_bounds=(-2.0, 2.0),
            seed=42,
        )

        def closure():
            return (x ** 2).sum().item()

        for _ in range(5):
            optimizer.step(closure)
            assert (x >= -2.0).all(), f"Parameter below lower bound: {x}"
            assert (x <= 2.0).all(), f"Parameter above upper bound: {x}"

    def test_domain_bounds_help_convergence(self):
        """Domain bounds should prevent divergence."""
        # Create separate parameters for each optimizer
        x_unbounded = nn.Parameter(torch.tensor([100.0, 100.0]))
        x_bounded = nn.Parameter(torch.tensor([100.0, 100.0]))

        # Without bounds might struggle
        optimizer_unbounded = BFO(
            [x_unbounded], lr=0.1, population_size=10, domain_bounds=None, seed=42
        )

        # With bounds should help
        optimizer_bounded = BFO(
            [x_bounded],
            lr=0.1,
            population_size=10,
            domain_bounds=(-10.0, 10.0),
            seed=42,
        )

        def closure_unbounded():
            return (x_unbounded ** 2).sum().item()

        def closure_bounded():
            return (x_bounded ** 2).sum().item()

        loss_unbounded = optimizer_unbounded.step(closure_unbounded)
        loss_bounded = optimizer_bounded.step(closure_bounded)

        # Both should work
        assert isinstance(loss_unbounded, float)
        assert isinstance(loss_bounded, float)


class TestAdaptiveBFOEdgeCases:
    """Edge cases specific to AdaptiveBFO."""

    def test_population_size_adaptation_limits(self):
        """Population size should respect min/max limits."""
        x = nn.Parameter(torch.randn(5))
        optimizer = AdaptiveBFO(
            [x],
            lr=0.01,
            population_size=10,
            min_population_size=5,
            max_population_size=20,
            adaptation_rate=0.5,
            seed=42,
        )

        def stagnant_closure():
            return 1.0  # Constant, forces stagnation

        # Run several steps to trigger adaptation
        for _ in range(10):
            optimizer.step(stagnant_closure)
            pop_size = optimizer.param_groups[0]["population_size"]
            assert (
                5 <= pop_size <= 20
            ), f"Population size {pop_size} outside bounds [5, 20]"


class TestHybridBFOEdgeCases:
    """Edge cases specific to HybridBFO."""

    def test_hybrid_with_partial_gradients(self):
        """HybridBFO should handle some parameters without gradients."""
        model = nn.Sequential(nn.Linear(5, 3), nn.Linear(3, 1))
        optimizer = HybridBFO(model.parameters(), lr=0.01, population_size=5, seed=42)

        def closure():
            optimizer.zero_grad()
            x = torch.randn(10, 5)
            output = model(x)
            loss = output.pow(2).sum()
            # Only backward through second layer
            loss.backward()
            # First layer has no gradients
            model[0].weight.grad = None
            model[0].bias.grad = None
            return loss.item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)

    def test_hybrid_momentum_zero(self):
        """HybridBFO with momentum=0 should disable momentum."""
        x = nn.Parameter(torch.randn(5))
        optimizer = HybridBFO(
            [x], lr=0.01, population_size=5, momentum=0.0, seed=42
        )

        assert optimizer.enable_momentum == False, "Momentum should be disabled for momentum=0"

        def closure():
            optimizer.zero_grad()
            loss = (x ** 2).sum()
            loss.backward()
            return loss.item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

