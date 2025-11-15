// Template for creating a new CUDA kernel
//
// Instructions:
// 1. Copy this file to csrc/cuda/YOUR_KERNEL_NAME.cu
// 2. Replace all instances of "MY_KERNEL" with your kernel name
// 3. Replace all instances of "my_kernel" with your kernel name (lowercase)
// 4. Implement the CUDA kernel and host function
// 5. Update the operator signature to match your inputs/outputs
// 6. Rebuild: pip install -e . --force-reinstall --no-deps

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

namespace extension_cpp {

// ============================================================================
// CUDA Kernel Implementation
// ============================================================================

template <typename scalar_t>
__global__ void my_kernel_cuda_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t size) {

    // Get thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (idx < size) {
        // TODO: Implement your kernel logic here
        // Example: output[idx] = input[idx] * 2.0;
        output[idx] = input[idx];
    }
}

// ============================================================================
// Host Function (launches CUDA kernel)
// ============================================================================

torch::Tensor my_kernel_forward(
    torch::Tensor input) {

    // Get tensor properties
    const auto batch_size = input.size(0);
    const auto feature_dim = input.size(1);
    const auto total_size = input.numel();

    // Create output tensor with same shape and options as input
    auto output = torch::empty_like(input);

    // Configure kernel launch parameters
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;

    // Launch kernel with appropriate scalar type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "my_kernel_cuda", ([&] {
        my_kernel_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total_size
        );
    }));

    // Check for CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

// ============================================================================
// Backward Pass (optional - implement if you need gradients)
// ============================================================================

// torch::Tensor my_kernel_backward(
//     torch::Tensor grad_output,
//     torch::Tensor input) {
//
//     // TODO: Implement gradient computation
//     auto grad_input = torch::empty_like(input);
//
//     return grad_input;
// }

// ============================================================================
// PyBind11 Module Registration (ONLY in ONE file - already in convrnn.cu)
// ============================================================================
//
// DO NOT uncomment this if you already have it in another .cu file!
// The PYBIND11_MODULE should only appear ONCE in your entire extension.
//
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// ============================================================================
// Operator Definition
// ============================================================================

TORCH_LIBRARY(extension_cpp, m) {
    // Define the operator schema
    // Syntax: "operator_name(input types) -> output types"
    m.def("my_kernel_forward(Tensor input) -> Tensor");

    // If you have multiple inputs/outputs, specify them all:
    // m.def("my_kernel_forward(Tensor input1, Tensor input2, int param) -> (Tensor, Tensor)");
}

// ============================================================================
// CUDA Implementation Registration
// ============================================================================

TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
    // Register the CUDA implementation
    m.impl("my_kernel_forward", &my_kernel_forward);
}

// ============================================================================
// CPU Implementation Registration (optional)
// ============================================================================

// If you want to support CPU as well, implement a CPU version:
//
// torch::Tensor my_kernel_forward_cpu(torch::Tensor input) {
//     // CPU implementation
//     return output;
// }
//
// TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
//     m.impl("my_kernel_forward", &my_kernel_forward_cpu);
// }

} // namespace extension_cpp
