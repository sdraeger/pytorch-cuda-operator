"""
PyTorch CUDA Extension Template

This package provides a template structure for creating custom CUDA kernels
for PyTorch using the torch.library API.

The extension is built using PyTorch's C++/CUDA extension mechanism and provides
a clean separation between:
- C++/CUDA kernel implementations (csrc/)
- Python API wrappers (ops.py)
- Template files for new kernels (templates/)

To use this template:
1. Copy template files from templates/ directory
2. Implement your CUDA kernel in csrc/cuda/
3. Add Python wrappers in ops.py
4. Register your kernel using torch.library API

See README.md and DEVELOPMENT.md for detailed instructions.
"""

import torch

__version__ = "0.1.0"

# ============================================================================
# Template Package Initialization
# ============================================================================
#
# When you add your custom CUDA kernels, uncomment and modify the sections below:
#
# 1. Load the compiled C++ extension:
#    try:
#        import extension_cpp._C as _C
#    except ImportError:
#        from . import _C
#
# 2. Import your ops module:
#    from . import ops
#
# 3. Export your public API:
#    from .ops import your_kernel_function
#    __all__ = ["your_kernel_function"]
#
# ============================================================================

__all__ = []

# For now, this is a template package with no compiled kernels.
# Once you add CUDA kernels following the templates, update this file accordingly.
