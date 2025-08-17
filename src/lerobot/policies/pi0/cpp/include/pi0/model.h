#pragma once

#include <torch/torch.h>

namespace lerobot {
namespace pi0 {

// Simple configuration structure for Pi0Model components
struct ModelConfig {
    int64_t vocab_size{0};
    int64_t embed_dim{0};
    int64_t num_heads{0};
    int64_t hidden_dim{0};
};

// A minimal C++ implementation mirroring Pi0Model architecture
class Pi0Model : public torch::nn::Module {
public:
    explicit Pi0Model(const ModelConfig& cfg);

    // Forward pass expecting token indices and state tensor
    torch::Tensor forward(const torch::Tensor& tokens,
                          const torch::Tensor& state);

private:
    ModelConfig cfg_;
    torch::nn::Embedding token_embedding_{nullptr};
    torch::nn::Linear state_proj_{nullptr};
    torch::nn::MultiheadAttention attention_{nullptr};
    torch::nn::Linear output_{nullptr};
};

} // namespace pi0
} // namespace lerobot

