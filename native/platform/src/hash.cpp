#include "rudra/platform/hash.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#include <vector>

namespace rudra {

// ---- XXH64 (reference algorithm, little-endian reads) ----------------------

namespace {
constexpr std::uint64_t P1 = 11400714785074694791ULL;
constexpr std::uint64_t P2 = 14029467366897019727ULL;
constexpr std::uint64_t P3 = 1609587929392839161ULL;
constexpr std::uint64_t P4 = 9650029242287828579ULL;
constexpr std::uint64_t P5 = 2870177450012600261ULL;

// Little-endian loads, assembled byte by byte: portable (no compiler builtins)
// and compiled to a single load on every little-endian target.
inline std::uint64_t read64(const std::byte* p) noexcept {
    std::uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | std::to_integer<std::uint64_t>(p[i]);
    return v;
}
inline std::uint32_t read32(const std::byte* p) noexcept {
    std::uint32_t v = 0;
    for (int i = 3; i >= 0; --i) v = (v << 8) | std::to_integer<std::uint32_t>(p[i]);
    return v;
}
inline std::uint64_t round64(std::uint64_t acc, std::uint64_t input) noexcept {
    acc += input * P2;
    acc = std::rotl(acc, 31);
    return acc * P1;
}
inline std::uint64_t merge(std::uint64_t acc, std::uint64_t val) noexcept {
    acc ^= round64(0, val);
    return acc * P1 + P4;
}
}  // namespace

std::uint64_t xxh64(std::span<const std::byte> data, std::uint64_t seed) noexcept {
    const std::byte* p = data.data();
    const std::byte* const end = p + data.size();
    std::uint64_t h;
    if (data.size() >= 32) {
        std::uint64_t v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
        const std::byte* const limit = end - 32;
        do {
            v1 = round64(v1, read64(p));      p += 8;
            v2 = round64(v2, read64(p));      p += 8;
            v3 = round64(v3, read64(p));      p += 8;
            v4 = round64(v4, read64(p));      p += 8;
        } while (p <= limit);
        h = std::rotl(v1, 1) + std::rotl(v2, 7) + std::rotl(v3, 12) + std::rotl(v4, 18);
        h = merge(h, v1); h = merge(h, v2); h = merge(h, v3); h = merge(h, v4);
    } else {
        h = seed + P5;
    }
    h += static_cast<std::uint64_t>(data.size());
    while (p + 8 <= end) {
        h ^= round64(0, read64(p));
        h = std::rotl(h, 27) * P1 + P4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= static_cast<std::uint64_t>(read32(p)) * P1;
        h = std::rotl(h, 23) * P2 + P3;
        p += 4;
    }
    while (p < end) {
        h ^= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(*p)) * P5;
        h = std::rotl(h, 11) * P1;
        ++p;
    }
    h ^= h >> 33; h *= P2;
    h ^= h >> 29; h *= P3;
    h ^= h >> 32;
    return h;
}

// ---- SHA-256 (FIPS 180-4) ------------------------------------------------------

namespace {
constexpr std::array<std::uint32_t, 64> K = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
}  // namespace

Sha256::Sha256() noexcept
    : h_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19} {}

void Sha256::block(const std::uint8_t* p) noexcept {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = (std::uint32_t(p[4 * i]) << 24) | (std::uint32_t(p[4 * i + 1]) << 16) |
               (std::uint32_t(p[4 * i + 2]) << 8) | std::uint32_t(p[4 * i + 3]);
    for (int i = 16; i < 64; ++i) {
        const std::uint32_t s0 = std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 = std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
        const std::uint32_t S1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
        const std::uint32_t ch = (e & f) ^ (~e & g);
        const std::uint32_t t1 = h + S1 + ch + K[i] + w[i];
        const std::uint32_t S0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
        const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d; h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
}

void Sha256::update(std::span<const std::byte> data) noexcept {
    const auto* p = reinterpret_cast<const std::uint8_t*>(data.data());
    std::size_t n = data.size();
    total_ += n;
    if (buf_len_) {
        const std::size_t take = std::min(n, 64 - buf_len_);
        std::memcpy(buf_.data() + buf_len_, p, take);
        buf_len_ += take; p += take; n -= take;
        if (buf_len_ == 64) { block(buf_.data()); buf_len_ = 0; }
    }
    while (n >= 64) { block(p); p += 64; n -= 64; }
    if (n) { std::memcpy(buf_.data(), p, n); buf_len_ = n; }
}

std::array<std::uint8_t, 32> Sha256::finish() noexcept {
    const std::uint64_t bits = total_ * 8;
    const std::uint8_t pad = 0x80;
    update(std::as_bytes(std::span(&pad, 1)));
    const std::uint8_t zero = 0;
    while (buf_len_ != 56) update(std::as_bytes(std::span(&zero, 1)));
    std::uint8_t len[8];
    for (int i = 0; i < 8; ++i) len[i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
    update(std::as_bytes(std::span(len, 8)));
    std::array<std::uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j) out[4 * i + j] = static_cast<std::uint8_t>(h_[i] >> (24 - 8 * j));
    return out;
}

std::string to_hex(std::span<const std::uint8_t> bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string s;
    s.reserve(bytes.size() * 2);
    for (auto b : bytes) { s.push_back(digits[b >> 4]); s.push_back(digits[b & 15]); }
    return s;
}

std::string sha256_hex(std::span<const std::byte> data) {
    Sha256 h;
    h.update(data);
    const auto d = h.finish();
    return to_hex(d);
}

Result<std::string> sha256_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return make_error(ErrorCode::NotFound, "The file could not be opened.", path.string());
    Sha256 h;
    std::vector<char> buf(1 << 20);
    while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const auto got = static_cast<std::size_t>(in.gcount());
        if (got) h.update(std::as_bytes(std::span(buf.data(), got)));
    }
    if (in.bad()) return make_error(ErrorCode::IoError, "The file could not be read.", path.string());
    const auto d = h.finish();
    return to_hex(d);
}

const char* to_string(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::InvalidArgument: return "invalid-argument";
        case ErrorCode::NotFound: return "not-found";
        case ErrorCode::IoError: return "io-error";
        case ErrorCode::ParseError: return "parse-error";
        case ErrorCode::Unsupported: return "unsupported";
        case ErrorCode::ContractMismatch: return "contract-mismatch";
        case ErrorCode::IntegrityError: return "integrity-error";
        case ErrorCode::BackendError: return "backend-error";
        case ErrorCode::ParityError: return "parity-error";
        case ErrorCode::Busy: return "busy";
    }
    return "unknown";
}

}  // namespace rudra
