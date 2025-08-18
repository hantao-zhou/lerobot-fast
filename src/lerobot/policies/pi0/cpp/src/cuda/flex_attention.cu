#include <torch/extension.h>
#include <cmath>

// Simple CUDA-enabled attention implementation using ATen operations.
torch::Tensor flex_attention_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor mask,
    double scale) {
    // Compute scaled dot-product scores
    torch::Tensor scores = torch::matmul(query, key.transpose(-2, -1));
    scores = scores * scale;

    // Apply additive mask if provided
    if (mask.defined() && mask.numel() > 0) {
        scores = scores + mask;
    }

    // Softmax and compute output
    torch::Tensor attn = torch::softmax(scores, -1);
    return torch::matmul(attn, value);
}
