#include "pi0/pi0.h"

namespace lerobot {
namespace pi0 {

Pi0::Pi0(const Config& cfg) : cfg_(cfg) {}

torch::Tensor Pi0::forward(const torch::Tensor& input) {
    // Placeholder implementation matching Python API.
    return torch::zeros_like(input);
}

std::shared_ptr<Pi0> create_pi0(const Config& cfg) {
    return std::make_shared<Pi0>(cfg);
}

} // namespace pi0
} // namespace lerobot

