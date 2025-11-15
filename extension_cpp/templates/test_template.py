"""
Template for testing custom CUDA kernels

Instructions:
1. Copy this file to tests/test_my_kernel.py
2. Replace "my_kernel" with your kernel name
3. Implement the test cases
4. Run: pytest tests/test_my_kernel.py -v
"""

import pytest
import torch
from extension_cpp.ops import reference_my_kernel

from extension_cpp import my_kernel


class TestMyKernel:
    """Test suite for my_kernel CUDA implementation"""

    @pytest.fixture
    def device(self):
        """Ensure CUDA is available"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")

    def test_correctness_simple(self, device):
        """Test that CUDA kernel matches reference implementation on simple input."""
        x = torch.randn(2, 10, 64, device=device)

        cuda_output = my_kernel(x)
        ref_output = reference_my_kernel(x)

        assert torch.allclose(cuda_output, ref_output, rtol=1e-5, atol=1e-6), (
            "CUDA output doesn't match reference implementation"
        )

    def test_correctness_large(self, device):
        """Test on larger tensors."""
        x = torch.randn(32, 128, 256, device=device)

        cuda_output = my_kernel(x)
        ref_output = reference_my_kernel(x)

        assert torch.allclose(cuda_output, ref_output, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1, 1),  # Minimal
            (2, 10, 64),  # Small
            (16, 100, 128),  # Medium
            (32, 256, 512),  # Large
        ],
    )
    def test_various_shapes(self, device, shape):
        """Test that kernel works with various input shapes."""
        x = torch.randn(*shape, device=device)

        cuda_output = my_kernel(x)
        ref_output = reference_my_kernel(x)

        assert cuda_output.shape == ref_output.shape
        assert torch.allclose(cuda_output, ref_output, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            # torch.bfloat16,  # Uncomment if supported
        ],
    )
    def test_different_dtypes(self, device, dtype):
        """Test that kernel works with different data types."""
        x = torch.randn(2, 10, 64, device=device, dtype=dtype)

        cuda_output = my_kernel(x)

        assert cuda_output.dtype == dtype
        # For fp16, use looser tolerances
        if dtype == torch.float16:
            ref_output = reference_my_kernel(x)
            assert torch.allclose(cuda_output, ref_output, rtol=1e-2, atol=1e-3)
        else:
            ref_output = reference_my_kernel(x)
            assert torch.allclose(cuda_output, ref_output, rtol=1e-5, atol=1e-6)

    def test_output_shape(self, device):
        """Test that output has correct shape."""
        batch, seq_len, features = 4, 20, 128
        x = torch.randn(batch, seq_len, features, device=device)

        output = my_kernel(x)

        # Adjust expected shape based on your kernel
        expected_shape = (batch, seq_len, features)
        assert output.shape == expected_shape

    def test_output_device(self, device):
        """Test that output is on the same device as input."""
        x = torch.randn(2, 10, 64, device=device)
        output = my_kernel(x)

        assert output.device == x.device

    def test_contiguous_input(self, device):
        """Test with non-contiguous input."""
        x = torch.randn(2, 10, 64, device=device)
        x_transposed = x.transpose(0, 1).contiguous().transpose(0, 1)

        cuda_output = my_kernel(x_transposed)
        ref_output = reference_my_kernel(x_transposed)

        assert torch.allclose(cuda_output, ref_output, rtol=1e-5, atol=1e-6)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_torch_compile(self, device):
        """Test that kernel works with torch.compile."""
        x = torch.randn(2, 10, 64, device=device)

        # This should work if fake implementation is registered
        compiled_fn = torch.compile(my_kernel)
        output = compiled_fn(x)

        assert output.shape == x.shape

    def test_zero_input(self, device):
        """Test with zero input."""
        x = torch.zeros(2, 10, 64, device=device)
        output = my_kernel(x)

        # Verify output shape at minimum
        assert output.shape == x.shape

    def test_ones_input(self, device):
        """Test with ones input."""
        x = torch.ones(2, 10, 64, device=device)
        output = my_kernel(x)

        assert output.shape == x.shape


class TestMyKernelGradients:
    """Test gradient computation (if implemented)"""

    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")

    @pytest.mark.skip(reason="Implement when backward pass is ready")
    def test_backward_pass(self, device):
        """Test that gradients are computed correctly."""
        x = torch.randn(2, 10, 64, device=device, requires_grad=True)

        output = my_kernel(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    @pytest.mark.skip(reason="Implement when backward pass is ready")
    def test_gradcheck(self, device):
        """Test gradients using torch.autograd.gradcheck."""
        from torch.autograd import gradcheck

        x = torch.randn(2, 5, 8, device=device, dtype=torch.float64, requires_grad=True)

        # gradcheck requires double precision
        test = gradcheck(my_kernel, (x,), eps=1e-6, atol=1e-4)
        assert test, "Gradient check failed"


@pytest.mark.benchmark
class TestMyKernelPerformance:
    """Performance benchmarks (run with: pytest -v -m benchmark)"""

    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")

    def test_benchmark_small(self, device, benchmark):
        """Benchmark on small input."""
        x = torch.randn(16, 64, 128, device=device)

        def run():
            output = my_kernel(x)
            torch.cuda.synchronize()
            return output

        benchmark(run)

    def test_benchmark_medium(self, device, benchmark):
        """Benchmark on medium input."""
        x = torch.randn(32, 128, 256, device=device)

        def run():
            output = my_kernel(x)
            torch.cuda.synchronize()
            return output

        benchmark(run)

    def test_benchmark_large(self, device, benchmark):
        """Benchmark on large input."""
        x = torch.randn(64, 256, 512, device=device)

        def run():
            output = my_kernel(x)
            torch.cuda.synchronize()
            return output

        benchmark(run)

    def test_compare_cuda_vs_reference(self, device):
        """Compare CUDA kernel speed vs reference implementation."""
        import time

        x = torch.randn(128, 512, 1024, device=device)
        num_runs = 100

        # Warmup
        for _ in range(10):
            _ = my_kernel(x)
        torch.cuda.synchronize()

        # CUDA kernel
        start = time.time()
        for _ in range(num_runs):
            _ = my_kernel(x)
        torch.cuda.synchronize()
        cuda_time = time.time() - start

        # Reference implementation
        start = time.time()
        for _ in range(num_runs):
            _ = reference_my_kernel(x)
        torch.cuda.synchronize()
        ref_time = time.time() - start

        speedup = ref_time / cuda_time
        print(f"\nCUDA time: {cuda_time * 1000 / num_runs:.2f} ms")
        print(f"Reference time: {ref_time * 1000 / num_runs:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")

        # CUDA should be faster (or at least not much slower)
        # Comment this out if your kernel isn't optimized yet
        # assert speedup > 0.5, f"CUDA kernel is too slow (speedup: {speedup:.2f}x)"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
