#pragma once
// Reader for NumPy .npy files (format 1.0 to 3.0), little-endian float32 only.
// It is how the golden files the Python writes reach the native tests and the
// model-package self-test.

#include <cstdint>
#include <filesystem>
#include <vector>

#include "rudra/platform/result.hpp"

namespace rudra {

struct NpyArray {
    std::vector<std::int64_t> shape;
    std::vector<float> data;   // C order

    std::int64_t size() const noexcept {
        std::int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

Result<NpyArray> read_npy(const std::filesystem::path& path);

}  // namespace rudra
