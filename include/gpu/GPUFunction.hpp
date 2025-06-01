#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace harmonics {

using GPUDataVariant =
    std::variant<float, std::vector<float>, std::vector<int32_t>, std::vector<uint8_t>>;

struct GPUFunction {
    std::string name;
    std::string shader;
    std::function<GPUDataVariant(const std::vector<GPUDataVariant>&)> cpuFallback;
};

} // namespace harmonics
