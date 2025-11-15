# CUDA Kernel Extension Template

This template provides a complete structure for creating custom PyTorch CUDA kernels using the `torch.library` API. It includes all the boilerplate code needed to build, register, and use CUDA kernels in PyTorch.

## Repository Structure

```
extension_cpp/
├── README.md                      # This file
├── __init__.py                    # Package initialization (loads compiled module)
├── ops.py                         # Python API and operator registration
├── csrc/                          # C++/CUDA source code
│   └── cuda/
│       └── convrnn.cu             # Example CUDA kernel implementation
└── templates/                     # Templates for new kernels (see below)
    ├── kernel_template.cu         # CUDA kernel template
    └── ops_template.py            # Python ops template
```

## Quick Start

### 1. Installation

From the repository root:

```bash
pip install -e .
```

This will compile the CUDA extension and install the package in development mode.

### 2. Using Existing Kernels

```python
import torch
from extension_cpp import convrnn_forward

# Example usage
x = torch.randn(2, 10, 64, device='cuda')
kernel = torch.randn(1, 1, 3, device='cuda')
hidden = torch.randn(2, 64, device='cuda')

output = convrnn_forward(x, kernel, hidden)
```

## Adding a New CUDA Kernel

Follow these steps to add a new custom kernel:

### Step 1: Create the CUDA Kernel File

Create a new `.cu` file in `csrc/cuda/`:

```bash
touch csrc/cuda/my_kernel.cu
```

Use the structure:

```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace extension_cpp {

// Your CUDA kernel implementation
__global__ void my_kernel_cuda(/* params */) {
    // CUDA kernel code
}

// Host function that launches the kernel
torch::Tensor my_kernel_forward(torch::Tensor input) {
    // Setup and launch kernel
    // Return results
}

// Register operators in TORCH_LIBRARY section (see below)
}
```

### Step 2: Register the Operator

In the **same `.cu` file**, add the operator registration at the bottom:

```cpp
// Defines the operators
TORCH_LIBRARY(extension_cpp, m) {
    m.def("my_kernel_forward(Tensor input) -> Tensor");
}

// Register implementation for CUDA
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
    m.impl("my_kernel_forward", &my_kernel_forward);
}
```

**Important**: Only **one file** should contain the `PYBIND11_MODULE` macro. This is already in `convrnn.cu`. All other `.cu` files should only have `TORCH_LIBRARY` and `TORCH_LIBRARY_IMPL`.

### Step 3: Add Python API

Add your Python wrapper to `ops.py`:

```python
def my_kernel(input: Tensor) -> Tensor:
    """Description of your kernel"""
    return torch.ops.extension_cpp.my_kernel_forward(input)

# Optional: Register fake implementation for meta device
@torch.library.register_fake("extension_cpp::my_kernel_forward")
def _(input):
    return torch.empty_like(input)

# Export the function
__all__.append("my_kernel")
```

### Step 4: Rebuild and Test

```bash
pip install -e . --force-reinstall --no-deps
python -c "from extension_cpp import my_kernel; print('Success!')"
```

## Build Configuration

The build is configured in the root `setup.py`:

- **Library name**: `extension_cpp` (change in `setup.py` line 14)
- **CUDA compute capabilities**: Edit `GPU_TARGETS` in `setup.py`
- **Debug mode**: `DEBUG=1 pip install -e .`
- **CPU only**: `USE_CUDA=0 pip install -e .`

## Key Concepts

### Operator Registration

PyTorch's `torch.library` API requires three components:

1. **TORCH_LIBRARY**: Define the operator schema
2. **TORCH_LIBRARY_IMPL**: Register the implementation for specific backend (CUDA/CPU)
3. **Python wrapper**: Call via `torch.ops.namespace.operator_name`

### Fake Implementations

Register fake implementations for torch.compile and meta device support:

```python
@torch.library.register_fake("extension_cpp::my_op")
def _(inputs):
    # Return tensors with correct shape but no computation
    return torch.empty_like(inputs)
```

### Autograd Support

To add backward pass support:

```python
def my_kernel_backward(ctx, grad_output):
    inputs = ctx.saved_tensors
    # Compute gradients
    return grad_input

def setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)

torch.library.register_autograd(
    "extension_cpp::my_kernel_forward",
    my_kernel_backward,
    setup_context=setup_context
)
```

## Reference Implementation Pattern

For testing, always provide a pure PyTorch reference implementation:

```python
def reference_my_kernel(input: Tensor) -> Tensor:
    """Pure PyTorch implementation for testing/validation"""
    # Implement using standard PyTorch operations
    return result
```

This allows you to validate your CUDA kernel against a known-correct implementation.

## Testing Your Kernel

Create a test file to compare CUDA vs reference:

```python
import torch
from extension_cpp import my_kernel
from extension_cpp.ops import reference_my_kernel

x = torch.randn(10, 20, device='cuda')

cuda_output = my_kernel(x)
ref_output = reference_my_kernel(x)

assert torch.allclose(cuda_output, ref_output, rtol=1e-5)
print("Test passed!")
```

## Debugging

Enable debug mode for verbose compilation:

```bash
DEBUG=1 pip install -e . -v
```

Check CUDA compilation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import extension_cpp._C; print('Extension loaded')"
```

## Common Issues

1. **"No CUDA runtime found"**: Ensure CUDA toolkit is installed and `CUDA_HOME` is set
2. **Import errors**: Make sure to `pip install -e .` after changes to C++ code
3. **Operator not found**: Check that operator name matches between TORCH_LIBRARY and Python code
4. **Compilation errors**: Check CUDA compute capability matches your GPU

## Example: The ConvRNN Kernel

See [`csrc/cuda/convrnn.cu`](csrc/cuda/convrnn.cu) and the corresponding Python API in [`ops.py`](ops.py) for a complete working example.

## Additional Resources

- [PyTorch C++ Extension Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [torch.library API Documentation](https://pytorch.org/docs/stable/library.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## License

[Specify your license here]
