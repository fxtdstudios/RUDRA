#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "rudra/platform/hash.hpp"
#include "rudra/platform/npy.hpp"
#include "rudra/platform/process.hpp"

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

TEST(Sha1, Fips180Vectors) {
    EXPECT_EQ(sha1_hex(bytes("")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    EXPECT_EQ(sha1_hex(bytes("abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
    EXPECT_EQ(sha1_hex(bytes("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
              "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
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

// Processes: cmake -E is the one program every build machine has.
namespace {
namespace fs = std::filesystem;
fs::path scratch(const std::string& name) {
    const fs::path d = fs::temp_directory_path() / ("rudra-process-test-" + name);
    fs::create_directories(d);
    return d;
}
}  // namespace

TEST(Process, FindExecutable) {
    EXPECT_TRUE(find_executable(RUDRA_CMAKE_COMMAND).has_value());   // a path is taken as it is
    EXPECT_FALSE(find_executable("rudra-no-such-program-xyz").has_value());
    EXPECT_FALSE(find_executable((scratch("find") / "missing").string()).has_value());
}

TEST(Process, RunCapturesOutputAndExit) {
    auto r = run_process({RUDRA_CMAKE_COMMAND, "-E", "echo", "hello world", "a\"b", ""});
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r->exit_code, 0);
    EXPECT_EQ(r->out.substr(0, r->out.find_last_not_of("\r\n") + 1), "hello world a\"b ");   // quoting survives
    auto bad = run_process({RUDRA_CMAKE_COMMAND, "-E", "cat", (scratch("run") / "missing.txt").string()});
    ASSERT_TRUE(bad.ok());
    EXPECT_NE(bad->exit_code, 0);
    EXPECT_FALSE(bad->err.empty());
    EXPECT_FALSE(run_process({"rudra-no-such-program-xyz"}).ok());
    EXPECT_FALSE(run_process({}).ok());
}

TEST(Process, StreamsLargeStdoutAndLogsStderr) {
    const fs::path dir = scratch("stream");
    std::string data(3 * 1024 * 1024 + 17, '\0');   // larger than any pipe buffer
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<char>((i * 131 + 7) & 0xff);
    { std::ofstream(dir / "big.bin", std::ios::binary).write(data.data(), std::streamsize(data.size())); }
    auto p = Process::start({RUDRA_CMAKE_COMMAND, "-E", "cat", (dir / "big.bin").string()}, dir / "err.log");
    ASSERT_TRUE(p.ok());
    std::vector<std::uint8_t> got;
    std::vector<std::uint8_t> buf(1 << 16);
    for (;;) {
        const std::size_t n = (*p)->read(buf);
        got.insert(got.end(), buf.begin(), buf.begin() + std::ptrdiff_t(n));
        if (n < buf.size()) break;
    }
    EXPECT_EQ((*p)->wait(), 0);
    EXPECT_FALSE((*p)->running());
    ASSERT_EQ(got.size(), data.size());
    EXPECT_TRUE(std::equal(got.begin(), got.end(), reinterpret_cast<const std::uint8_t*>(data.data())));

    auto e = Process::start({RUDRA_CMAKE_COMMAND, "-E", "cat", (dir / "nope.bin").string()}, dir / "err2.log");
    ASSERT_TRUE(e.ok());
    std::uint8_t one[1];
    EXPECT_EQ((*e)->read(one), 0u);
    EXPECT_NE((*e)->wait(), 0);
    EXPECT_GT(fs::file_size(dir / "err2.log"), 0u);   // stderr went to the log
}

TEST(Process, KillEndsARunningProgram) {
    auto p = Process::start({RUDRA_CMAKE_COMMAND, "-E", "sleep", "30"});
    ASSERT_TRUE(p.ok());
    EXPECT_TRUE((*p)->running());
    (*p)->kill();
    (*p)->wait();
    EXPECT_FALSE((*p)->running());
}
