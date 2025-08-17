#pragma once

#include <cstdint>

namespace lerobot {
namespace pi0 {

struct Config {
    int64_t n_obs_steps = 1;
    int64_t chunk_size = 50;
    int64_t n_action_steps = 50;
};

} // namespace pi0
} // namespace lerobot

