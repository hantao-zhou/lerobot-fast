#include "pi0/model.h"

namespace lerobot {
namespace pi0 {

Pi0Model::Pi0Model(const ModelConfig& cfg) : cfg_(cfg) {
    token_embedding_ = register_module("token_embedding", torch::nn::Embedding(cfg.vocab_size, cfg.embed_dim));
    state_proj_ = register_module("state_proj", torch::nn::Linear(cfg.hidden_dim, cfg.embed_dim));
    attention_ = register_module("attention", torch::nn::MultiheadAttention(
        torch::nn::MultiheadAttentionOptions(cfg.embed_dim, cfg.num_heads)));
    output_ = register_module("output", torch::nn::Linear(cfg.embed_dim, cfg.hidden_dim));
}

// Forward pass combining token embeddings, attention and state projection
// tokens: [batch, seq], state: [batch, hidden_dim]
torch::Tensor Pi0Model::forward(const torch::Tensor& tokens,
                                 const torch::Tensor& state) {
    auto token_emb = token_embedding_->forward(tokens); // [B, S, E]
    auto state_proj = state_proj_->forward(state);      // [B, E]
    state_proj = state_proj.unsqueeze(1).expand({token_emb.size(0), token_emb.size(1), cfg_.embed_dim});

    auto query = token_emb.permute({1, 0, 2}); // [S, B, E]
    auto key = query;
    auto value = query;
    auto attn_output = std::get<0>(attention_->forward(query, key, value));
    attn_output = attn_output.permute({1, 0, 2}); // [B, S, E]

    auto combined = attn_output + state_proj;
    return output_->forward(combined);
}

} // namespace pi0
} // namespace lerobot

// Register with TorchScript
TORCH_LIBRARY(pi0, m) {
    namespace F = lerobot::pi0;
    m.class_<F::Pi0Model>("Pi0Model")
        .def(torch::init<F::ModelConfig>())
        .def("forward", &F::Pi0Model::forward);
}

