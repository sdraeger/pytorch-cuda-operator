# PyTorch CUDA Kernel Extension Template

A production-ready template for creating custom PyTorch CUDA kernels using modern PyTorch APIs (`torch.library`). This template provides all the boilerplate code and best practices needed to quickly develop, test, and deploy high-performance CUDA operations.

## Features

- `torch.library` API for operator registration
- Complete build system with PyTorch C++/CUDA extensions
- Template files for rapid kernel development
- Reference implementation pattern for testing
- Support for multiple GPU architectures
- Debug and release build configurations
- Ready for torch.compile with fake implementations
- Comprehensive documentation and examples

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- CUDA Toolkit (matching your PyTorch CUDA version)
- C++ compiler (GCC 9.4+ recommended)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Install in development mode
pip install -e .

# Or with CUDA disabled (CPU only)
USE_CUDA=0 pip install -e .
```

### Verify Installation

```python
import torch
import extension_cpp

print("Extension loaded successfully!")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Project Structure

```
.
├── extension_cpp/              # Main package directory
│   ├── __init__.py            # Package initialization
│   ├── csrc/                  # C++/CUDA source code
│   │   └── cuda/              # CUDA kernel implementations
│   ├── templates/             # Template files for new kernels
│   │   ├── kernel_template.cu     # CUDA kernel template
│   │   ├── ops_template.py        # Python API template
│   │   └── test_template.py       # Test template
│   ├── tests/                 # Unit tests
│   ├── README.md              # Detailed development guide
│   ├── DEVELOPMENT.md         # Development workflow
│   ├── QUICKSTART.md          # Quick reference guide
│   └── TEMPLATE_USAGE.md      # How to use templates
├── setup.py                   # Build configuration
├── pyproject.toml            # Python project metadata
└── requirements.txt          # Python dependencies
```

## Creating Your First CUDA Kernel

See [extension_cpp/TEMPLATE_USAGE.md](extension_cpp/TEMPLATE_USAGE.md) for detailed instructions on using the template files.

Quick overview:

1. **Copy the kernel template**: Start with `templates/kernel_template.cu`
2. **Implement your CUDA kernel**: Add your GPU computation logic
3. **Add Python API**: Use `templates/ops_template.py` for the Python wrapper
4. **Create tests**: Use `templates/test_template.py` as a starting point
5. **Build and test**: `pip install -e . && pytest`

## Documentation

- [**README.md**](extension_cpp/README.md) - Complete development guide
- [**QUICKSTART.md**](extension_cpp/QUICKSTART.md) - Quick reference for common tasks
- [**DEVELOPMENT.md**](extension_cpp/DEVELOPMENT.md) - Development workflow and best practices
- [**TEMPLATE_USAGE.md**](extension_cpp/TEMPLATE_USAGE.md) - How to use the template files
- [**SUMMARY.md**](extension_cpp/SUMMARY.md) - Project overview and architecture

## Build Configuration

### Environment Variables

- `USE_CUDA=1/0` - Enable/disable CUDA compilation (default: 1)
- `DEBUG=1/0` - Enable debug mode with `-g` flags (default: 0)
- `TORCH_CUDA_ARCH_LIST` - Semicolon-separated list of CUDA architectures

### Example: Multi-architecture Build

```bash
export TORCH_CUDA_ARCH_LIST="Turing;Ampere;Ada;Hopper"
pip install -e .
```

### Debug Build

```bash
DEBUG=1 pip install -e . -v
```

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest extension_cpp/tests/

# Run specific test file
pytest extension_cpp/tests/test_your_kernel.py

# Run with verbose output
pytest -v -s extension_cpp/tests/
```

## Key Concepts

### torch.library API

This template uses PyTorch's modern `torch.library` API for operator registration:

1. **TORCH_LIBRARY** - Define operator schema in C++
2. **TORCH_LIBRARY_IMPL** - Register backend implementation (CUDA/CPU)
3. **Python wrapper** - Call via `torch.ops.namespace.operator_name`

### Reference Implementation Pattern

Always provide a pure PyTorch reference implementation alongside your CUDA kernel:

- Use for correctness testing
- Helps catch numerical errors
- Serves as documentation
- Enables CPU fallback if needed

### Fake Implementations

Register fake implementations for torch.compile support:

```python
@torch.library.register_fake("extension_cpp::my_op")
def _(inputs):
    return torch.empty_like(inputs)
```

## Performance Tips

1. **Choose the right thread/block configuration** - See CUDA programming guides
2. **Minimize memory transfers** - Keep data on GPU when possible
3. **Use shared memory** - For frequently accessed data
4. **Coalesce memory accesses** - Align memory access patterns
5. **Profile your kernel** - Use `torch.cuda.nvprof` or Nsight

## Common Issues

**Import Error**: Module not found
- Solution: Run `pip install -e .` after any C++ code changes

**CUDA Runtime Error**: No CUDA-capable device
- Solution: Verify `torch.cuda.is_available()` returns True

**Compilation Error**: Unsupported GPU architecture
- Solution: Check `GPU_TARGETS` in `setup.py` matches your GPU

**Operator Not Found**: RuntimeError
- Solution: Verify operator name matches between TORCH_LIBRARY and Python code

## Contributing

When contributing:

1. Follow the existing code structure
2. Add tests for new kernels
3. Include reference implementations
4. Update documentation
5. Test on multiple GPU architectures if possible

## Acknowledgments

This template follows PyTorch best practices and is designed for production use.

## Support

For questions and issues:
- Check the documentation in `extension_cpp/`
- Review the template files in `extension_cpp/templates/`
- Open an issue on GitHub
