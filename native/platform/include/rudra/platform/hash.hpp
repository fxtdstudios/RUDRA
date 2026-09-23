#pragma once
// Hashes. XXH64 keys the content-addressed caches (a decoded frame's pixels);
// SHA-256 verifies files against the manifests and SHA256SUMS the Python writes.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

#include "rudra/platform/result.hpp"

namespace rudra {

std::uint64_t xxh64(std::span<const std::byte> data, std::uint64_t seed = 0) noexcept;

class Sha256 {
public:
    Sha256() noexcept;
    void update(std::span<const std::byte> data) noexcept;
    std::array<std::uint8_t, 32> finish() noexcept;

private:
    void block(const std::uint8_t* p) noexcept;
    std::array<std::uint32_t, 8> h_{};
    std::array<std::uint8_t, 64> buf_{};
    std::size_t buf_len_ = 0;
    std::uint64_t total_ = 0;
};

std::string to_hex(std::span<const std::uint8_t> bytes);
std::string sha256_hex(std::span<const std::byte> data);
Result<std::string> sha256_file(const std::filesystem::path& path);

}  // namespace rudra
