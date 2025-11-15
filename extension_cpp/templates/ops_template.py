"""
Template for adding Python API for a new CUDA kernel

Instructions:
1. Copy the sections below into ops.py
2. Replace "my_kernel" with your kernel name
3. Update the type hints and docstrings
4. Add your function name to __all__ at the top of ops.py
5. Implement the reference implementation for testing
"""

import torch
from torch import Tensor

# ============================================================================
# Main API Function
# ============================================================================


def my_kernel(input: Tensor) -> Tensor:
    """
    Brief description of what your kernel does.

    Parameters
    ----------
    input : Tensor
        Description of input tensor and its expected shape (e.g., (batch, length, features))

    Returns
    -------
    output : Tensor
        Description of output tensor and its shape

    Examples
    --------
    >>> x = torch.randn(2, 10, 64, device='cuda')
    >>> output = my_kernel(x)
    >>> output.shape
    torch.Size([2, 10, 64])
    """
    output = torch.ops.extension_cpp.my_kernel_forward(input)
    return output


# ============================================================================
# Fake Implementation (for torch.compile and meta device)
# ============================================================================

try:

    @torch.library.register_fake("extension_cpp::my_kernel_forward")
    def _(input):
        """
        Fake implementation that returns correctly shaped tensor without computation.
        This is used by torch.compile for shape inference.
        """
        # Return a tensor with the correct output shape
        # For operations that preserve shape:
        fake_output = torch.empty_like(input)

        # For operations that change shape:
        # batch, length, features = input.shape
        # fake_output = torch.empty(batch, new_length, new_features,
        #                          dtype=input.dtype, device=input.device)

        return fake_output

except RuntimeError as e:
    print(f"Warning: Could not register fake for my_kernel_forward: {e}")


# ============================================================================
# Autograd Registration (if you need gradients)
# ============================================================================

# def my_kernel_backward(ctx, grad_output):
#     """
#     Backward pass for the kernel.
#
#     Parameters
#     ----------
#     ctx : context object containing saved tensors
#     grad_output : Tensor
#         Gradient with respect to the output
#
#     Returns
#     -------
#     grad_input : Tensor
#         Gradient with respect to the input
#     """
#     input, = ctx.saved_tensors
#
#     # Compute gradient
#     # grad_input = torch.ops.extension_cpp.my_kernel_backward(grad_output, input)
#
#     return grad_input


# def setup_context(ctx, inputs, output):
#     """Setup context for backward pass by saving necessary tensors."""
#     # Save inputs for backward
#     ctx.save_for_backward(*inputs)


# # Register autograd
# torch.library.register_autograd(
#     "extension_cpp::my_kernel_forward",
#     my_kernel_backward,
#     setup_context=setup_context
# )


# ============================================================================
# Reference Implementation (for testing and validation)
# ============================================================================


def reference_my_kernel(input: Tensor) -> Tensor:
    """
    Pure PyTorch reference implementation.

    Use this to validate your CUDA kernel against a known-correct implementation.
    Should produce the same results as the CUDA version (within numerical precision).

    Parameters
    ----------
    input : Tensor
        Same as main function

    Returns
    -------
    output : Tensor
        Same as main function

    Examples
    --------
    >>> # Compare CUDA vs reference
    >>> x = torch.randn(2, 10, 64, device='cuda')
    >>> cuda_out = my_kernel(x)
    >>> ref_out = reference_my_kernel(x)
    >>> torch.allclose(cuda_out, ref_out, rtol=1e-5)
    True
    """
    # TODO: Implement using standard PyTorch operations
    # This should match the behavior of your CUDA kernel

    # Example: element-wise operation
    output = input.clone()

    # Example: reduction operation
    # output = input.sum(dim=-1)

    # Example: sequential operation
    # batch, length, features = input.shape
    # output = torch.zeros_like(input)
    # for i in range(length):
    #     output[:, i, :] = some_operation(input[:, i, :])

    return output


# ============================================================================
# Don't forget to add to __all__ in ops.py!
# ============================================================================
# __all__.extend(["my_kernel", "reference_my_kernel"])
