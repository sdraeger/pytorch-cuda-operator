# Quick Start Guide

Get your first custom CUDA kernel running in 5 minutes!

## Step 1: Verify Prerequisites

```bash
# Check CUDA
nvcc --version

# Check PyTorch with CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

Both should return `True` or show version numbers.

## Step 2: Install the Extension

From the repository root (where `setup.py` is located):

```bash
pip install -e .
```

You should see compilation output. If it succeeds, you're ready!

## Step 3: Test the Installation

```bash
python -c "from extension_cpp import convrnn_forward; print('Success!')"
```

## Step 4: Try the Example Kernel

```python
import torch
from extension_cpp import convrnn_forward

# Create sample inputs
x = torch.randn(2, 10, 64, device='cuda')
kernel = torch.randn(1, 1, 3, device='cuda')
hidden = torch.randn(2, 64, device='cuda')

# Run the kernel
output = convrnn_forward(x, kernel, hidden)
print(f"Output shape: {output[0].shape}")
```

**Note**: The example `convrnn` kernel is currently a placeholder that returns zeros. You'll implement the actual logic!

## Step 5: Create Your First Custom Kernel

### 5.1 Copy the Templates

```bash
# From the extension_cpp directory
cp templates/kernel_template.cu csrc/cuda/my_add.cu
```

### 5.2 Implement a Simple Addition Kernel

Edit `csrc/cuda/my_add.cu`:

```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

namespace extension_cpp {

// CUDA kernel
template <typename scalar_t>
__global__ void add_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t value,
    int64_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = input[idx] + value;
    }
}

// Host function
torch::Tensor my_add_forward(torch::Tensor input, double value) {
    const auto size = input.numel();
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "my_add", ([&] {
        add_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<scalar_t>(value),
            size
        );
    }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// Register operator
TORCH_LIBRARY(extension_cpp, m) {
    m.def("my_add_forward(Tensor input, float value) -> Tensor");
}

TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
    m.impl("my_add_forward", &my_add_forward);
}

} // namespace extension_cpp
```

### 5.3 Add Python Wrapper

Add to `ops.py`:

```python
def my_add(input: Tensor, value: float) -> Tensor:
    """Add a scalar value to all elements of the input tensor."""
    return torch.ops.extension_cpp.my_add_forward(input, value)

# Add to __all__
__all__.append("my_add")
```

### 5.4 Rebuild and Test

```bash
# Rebuild
pip install -e . --force-reinstall --no-deps

# Test
python -c "
import torch
from extension_cpp import my_add

x = torch.randn(10, device='cuda')
y = my_add(x, 5.0)
print('Input:', x[:5])
print('Output:', y[:5])
print('Difference:', (y - x)[:5])  # Should be all 5.0
"
```

## Step 6: Validate Against Reference

Add reference implementation to `ops.py`:

```python
def reference_my_add(input: Tensor, value: float) -> Tensor:
    """Pure PyTorch reference implementation."""
    return input + value
```

Test correctness:

```python
import torch
from extension_cpp import my_add
from extension_cpp.ops import reference_my_add

x = torch.randn(1000, device='cuda')
cuda_out = my_add(x, 5.0)
ref_out = reference_my_add(x, 5.0)

assert torch.allclose(cuda_out, ref_out)
print("✓ CUDA kernel matches reference!")
```

## Next Steps

1. **Read the full documentation**: See [README.md](README.md) for complete API documentation
2. **Learn best practices**: Check [DEVELOPMENT.md](DEVELOPMENT.md) for optimization tips
3. **Write tests**: Use the test template in `templates/test_template.py`
4. **Optimize**: Profile with `nsight-compute` to find bottlenecks

## Common Issues

### "No CUDA runtime found"
- Ensure CUDA toolkit is installed
- Set `CUDA_HOME` environment variable: `export CUDA_HOME=/usr/local/cuda`

### "ImportError: cannot import name 'my_add'"
- Make sure you rebuilt after changes: `pip install -e . --force-reinstall --no-deps`
- Check that you added the function to `__all__` in `ops.py`

### "Operator not found"
- Verify the operator name matches exactly in:
  - `TORCH_LIBRARY` (C++)
  - `torch.ops.extension_cpp.xxx` (Python)
- Check that you imported the module properly

### Compilation errors
- Check CUDA compute capability matches your GPU
- Enable debug mode: `DEBUG=1 pip install -e . -v`
- Check GCC version compatibility with CUDA toolkit

## Getting Help

- Check existing kernels in `csrc/cuda/` for examples
- Read PyTorch extension docs: https://pytorch.org/tutorials/advanced/cpp_extension.html
- Look at CUDA programming guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

Happy kernel hacking!
