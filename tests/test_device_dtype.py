"""
Device and dtype robustness tests for BFO optimizers.

Tests CPU/CUDA/MPS devices and FP16/FP32/FP64 dtypes.
"""

import pytest
import torch
import torch.nn as nn

from bfo_torch import BFO, AdaptiveBFO, HybridBFO


class TestCPUDevice:
    """Test CPU device handling."""

    def test_bfo_cpu(self):
        """BFO should work on CPU."""
        model = nn.Linear(5, 1)
        optimizer = BFO(model.parameters(), device=torch.device("cpu"))

        def closure():
            x = torch.randn(10, 5)
            return model(x).pow(2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)
        assert optimizer.device == torch.device("cpu")

    def test_adaptive_bfo_cpu(self):
        """AdaptiveBFO should work on CPU."""
        x = nn.Parameter(torch.randn(10))
        optimizer = AdaptiveBFO([x], device=torch.device("cpu"))

        def closure():
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)

    def test_hybrid_bfo_cpu(self):
        """HybridBFO should work on CPU."""
        x = nn.Parameter(torch.randn(10))
        optimizer = HybridBFO([x], device=torch.device("cpu"))

        def closure():
            optimizer.zero_grad()
            loss = (x ** 2).sum()
            loss.backward()
            return loss.item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDADevice:
    """Test CUDA device handling."""

    def test_bfo_cuda(self):
        """BFO should work on CUDA."""
        model = nn.Linear(5, 1).cuda()
        optimizer = BFO(model.parameters())

        def closure():
            x = torch.randn(10, 5, device="cuda")
            return model(x).pow(2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)
        assert optimizer.device.type == "cuda"

    def test_adaptive_bfo_cuda(self):
        """AdaptiveBFO should work on CUDA."""
        x = nn.Parameter(torch.randn(10, device="cuda"))
        optimizer = AdaptiveBFO([x])

        def closure():
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)
        assert optimizer.device.type == "cuda"

    def test_hybrid_bfo_cuda(self):
        """HybridBFO should work on CUDA."""
        x = nn.Parameter(torch.randn(10, device="cuda"))
        optimizer = HybridBFO([x])

        def closure():
            optimizer.zero_grad()
            loss = (x ** 2).sum()
            loss.backward()
            return loss.item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)


class TestFP32Dtype:
    """Test FP32 (default) dtype."""

    def test_bfo_fp32(self):
        """BFO should work with FP32."""
        x = nn.Parameter(torch.randn(10, dtype=torch.float32))
        optimizer = BFO([x])

        def closure():
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)
        assert x.dtype == torch.float32

    def test_all_variants_fp32(self):
        """All variants should work with FP32."""
        for optimizer_class in [BFO, AdaptiveBFO, HybridBFO]:
            x = nn.Parameter(torch.randn(5, dtype=torch.float32))
            optimizer = optimizer_class([x], population_size=5)

            if optimizer_class == HybridBFO:

                def closure():
                    optimizer.zero_grad()
                    loss = (x ** 2).sum()
                    loss.backward()
                    return loss.item()
            else:

                def closure():
                    return (x ** 2).sum().item()

            loss = optimizer.step(closure)
            assert isinstance(loss, float)
            assert x.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for FP16")
class TestFP16Dtype:
    """Test FP16 (half precision) dtype."""

    def test_bfo_fp16(self):
        """BFO should work with FP16."""
        x = nn.Parameter(torch.randn(10, dtype=torch.float16, device="cuda"))
        optimizer = BFO([x], population_size=5)

        def closure():
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))
        assert x.dtype == torch.float16

    def test_adaptive_bfo_fp16(self):
        """AdaptiveBFO should work with FP16."""
        x = nn.Parameter(torch.randn(10, dtype=torch.float16, device="cuda"))
        optimizer = AdaptiveBFO([x], population_size=5)

        def closure():
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))

    def test_hybrid_bfo_fp16(self):
        """HybridBFO should work with FP16."""
        x = nn.Parameter(torch.randn(10, dtype=torch.float16, device="cuda"))
        optimizer = HybridBFO([x], population_size=5)

        def closure():
            optimizer.zero_grad()
            loss = (x ** 2).sum()
            loss.backward()
            return loss.item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))


class TestFP64Dtype:
    """Test FP64 (double precision) dtype for scientific computing."""

    def test_bfo_fp64(self):
        """BFO should work with FP64."""
        x = nn.Parameter(torch.randn(10, dtype=torch.float64))
        optimizer = BFO([x])

        def closure():
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)
        assert x.dtype == torch.float64

    def test_fp64_higher_precision(self):
        """FP64 should maintain higher precision."""
        x = nn.Parameter(torch.tensor([1e-10, 2e-10], dtype=torch.float64))
        optimizer = BFO([x], population_size=5, seed=42)

        def closure():
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)
        # Should handle very small values
        assert torch.isfinite(torch.tensor(loss))

    def test_all_variants_fp64(self):
        """All variants should work with FP64."""
        for optimizer_class in [BFO, AdaptiveBFO, HybridBFO]:
            x = nn.Parameter(torch.randn(5, dtype=torch.float64))
            optimizer = optimizer_class([x], population_size=5)

            if optimizer_class == HybridBFO:

                def closure():
                    optimizer.zero_grad()
                    loss = (x ** 2).sum()
                    loss.backward()
                    return loss.item()
            else:

                def closure():
                    return (x ** 2).sum().item()

            loss = optimizer.step(closure)
            assert isinstance(loss, float)
            assert x.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for BF16")
class TestBF16Dtype:
    """Test BF16 (bfloat16) dtype for modern training."""

    def test_bfo_bf16(self):
        """BFO should work with BF16."""
        if not hasattr(torch, "bfloat16"):
            pytest.skip("BFloat16 not available")

        x = nn.Parameter(torch.randn(10, dtype=torch.bfloat16, device="cuda"))
        optimizer = BFO([x], population_size=5)

        def closure():
            return (x ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))
        assert x.dtype == torch.bfloat16


class TestMixedDtype:
    """Test mixed dtypes within same model."""

    def test_mixed_dtype_parameters(self):
        """Should handle mixed dtypes in parameter groups."""
        x1 = nn.Parameter(torch.randn(5, dtype=torch.float32))
        x2 = nn.Parameter(torch.randn(5, dtype=torch.float64))

        optimizer = BFO(
            [
                {"params": [x1], "population_size": 5},
                {"params": [x2], "population_size": 5},
            ]
        )

        call_count = [0]

        def closure():
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return (x1 ** 2).sum().item()
            else:
                return (x2 ** 2).sum().item()

        loss = optimizer.step(closure)
        assert isinstance(loss, float)
        assert x1.dtype == torch.float32
        assert x2.dtype == torch.float64


class TestCrossDeviceCheckpoints:
    """Test saving/loading checkpoints across devices."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_checkpoint_cpu_to_cuda(self):
        """Should be able to load CPU checkpoint on CUDA."""
        # Train on CPU
        x_cpu = nn.Parameter(torch.randn(5))
        optimizer_cpu = BFO([x_cpu], population_size=5, seed=42)

        def closure_cpu():
            return (x_cpu ** 2).sum().item()

        optimizer_cpu.step(closure_cpu)
        state_dict = optimizer_cpu.state_dict()

        # Load on CUDA
        x_cuda = nn.Parameter(torch.randn(5, device="cuda"))
        optimizer_cuda = BFO([x_cuda], population_size=5)
        optimizer_cuda.load_state_dict(state_dict)

        # Should work
        def closure_cuda():
            return (x_cuda ** 2).sum().item()

        loss = optimizer_cuda.step(closure_cuda)
        assert isinstance(loss, float)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_checkpoint_cuda_to_cpu(self):
        """Should be able to load CUDA checkpoint on CPU."""
        # Train on CUDA
        x_cuda = nn.Parameter(torch.randn(5, device="cuda"))
        optimizer_cuda = BFO([x_cuda], population_size=5, seed=42)

        def closure_cuda():
            return (x_cuda ** 2).sum().item()

        optimizer_cuda.step(closure_cuda)
        state_dict = optimizer_cuda.state_dict()

        # Load on CPU
        x_cpu = nn.Parameter(torch.randn(5))
        optimizer_cpu = BFO([x_cpu], population_size=5, device=torch.device("cpu"))
        optimizer_cpu.load_state_dict(state_dict)

        # Should work
        def closure_cpu():
            return (x_cpu ** 2).sum().item()

        loss = optimizer_cpu.step(closure_cpu)
        assert isinstance(loss, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

