#pragma once

#include <memory>
#include <torch/torch.h>
#include "pi0/config.h"

namespace lerobot {
namespace pi0 {

class Pi0 : public torch::nn::Module {
public:
    explicit Pi0(const Config& cfg);

    torch::Tensor forward(const torch::Tensor& input);

private:
    Config cfg_;
};

std::shared_ptr<Pi0> create_pi0(const Config& cfg);

} // namespace pi0
} // namespace lerobot

