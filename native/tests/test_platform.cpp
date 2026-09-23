#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "rudra/platform/hash.hpp"
#include "rudra/platform/npy.hpp"

using namespace rudra;

namespace {
std::span<const std::byte> bytes(const std::string& s) { return std::as_bytes(std::span(s.data(), s.size())); }
}  // namespace

TEST(Xxh64, ReferenceVectors) {
    // From the reference implementation (python-xxhash).
    EXPECT_EQ(xxh64(bytes("")), 0xef46db3751d8e999ULL);
    EXPECT_EQ(xxh64(bytes("a")), 0xd24ec4f1a98c6e5bULL);
    EXPECT_EQ(xxh64(bytes("abc")), 0x44bc2cf5ad770999ULL);
    std::string long_input;
    for (int r = 0; r < 3; ++r)
        for (int i = 0; i < 256; ++i) long_input.push_back(static_cast<char>(i));
    EXPECT_EQ(xxh64(bytes(long_input), 7), 0xb1e10f6c5294cd6bULL);   // exercises the 32-byte stripes
}

TEST(Sha256, Fips180Vectors) {
    EXPECT_EQ(sha256_hex(bytes("")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(sha256_hex(bytes("abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(sha256_hex(bytes("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    Sha256 h;   // one million 'a', fed in uneven chunks
    const std::string chunk(997, 'a');
    std::size_t fed = 0;
    while (fed < 1000000) {
        const std::size_t n = std::min<std::size_t>(chunk.size(), 1000000 - fed);
        h.update(std::as_bytes(std::span(chunk.data(), n)));
        fed += n;
    }
    const auto d = h.finish();
    EXPECT_EQ(to_hex(d), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Result, CarriesValueOrError) {
    Result<int> ok = 3;
    ASSERT_TRUE(ok);
    EXPECT_EQ(*ok, 3);
    Result<int> bad = make_error(ErrorCode::Unsupported, "no", "why");
    ASSERT_FALSE(bad);
    EXPECT_EQ(bad.error().code, ErrorCode::Unsupported);
    Result<void> v;
    EXPECT_TRUE(v);
}

TEST(Npy, ReadsPythonWrittenFloat32) {
    auto a = read_npy(std::filesystem::path(RUDRA_GOLDEN_DIR) / "core" / "in_image.npy");
    ASSERT_TRUE(a) << a.error().detail;
    ASSERT_EQ(a->shape, (std::vector<std::int64_t>{1, 3, 24, 32}));
    EXPECT_EQ(a->data.size(), 1u * 3 * 24 * 32);
    for (float v : a->data) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LT(v, 1.0f);
    }
}

TEST(Npy, RefusesWhatItCannotRead) {
    auto missing = read_npy("does/not/exist.npy");
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error().code, ErrorCode::NotFound);
}
