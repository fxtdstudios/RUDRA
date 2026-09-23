// The still decoder against rudra/decode.py (goldens from tools/emit_decode_golden.py).

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>

#include <nlohmann/json.hpp>

#include "rudra/media/still.hpp"
#include "rudra/platform/npy.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

const fs::path kDir = fs::path(RUDRA_GOLDEN_DIR) / "decode";

nlohmann::json index_json() {
    std::ifstream in(kDir / "index.json");
    return nlohmann::json::parse(in);
}

}  // namespace

TEST(Decode, EveryFixtureMatchesDecodeSdr) {
    const auto idx = index_json();
    ASSERT_GE(idx.at("cases").size(), 15u);
    for (const auto& c : idx.at("cases")) {
        const std::string file = c.at("file");
        SCOPED_TRACE(file);
        auto r = decode_sdr_file(kDir / file);
        if (c.contains("refused")) {
            ASSERT_FALSE(r);
            EXPECT_EQ(r.error().code, ErrorCode::Unsupported);
            EXPECT_EQ(r.error().message, c.at("refused").get<std::string>());
            continue;
        }
        ASSERT_TRUE(r) << r.error().message;
        EXPECT_EQ(r->bits, c.at("bits").get<int>());
        EXPECT_EQ(r->distinct_codes, c.at("distinct_codes").get<int>());
        EXPECT_EQ(r->source, c.at("source").get<std::string>());
        auto want = read_npy(kDir / c.at("expected").get<std::string>());
        ASSERT_TRUE(want);
        const auto got = r->rgb.buffer().span();
        ASSERT_EQ(got.size(), want->data.size());
        // Lossless formats decode to identical integers, so identical floats.
        // JPEG is a decoder's arithmetic: libjpeg-turbo's ISLOW IDCT and fancy
        // upsampling are specified to the bit, so it is held exact too.
        std::size_t diff = 0;
        for (std::size_t i = 0; i < got.size(); ++i) diff += got[i] != want->data[i];
        EXPECT_EQ(diff, 0u) << diff << " of " << got.size() << " values differ";
    }
}

TEST(Decode, RejectsGarbageAndEmpty) {
    EXPECT_FALSE(decode_sdr(std::span<const std::uint8_t>{}));
    const std::uint8_t junk[] = {1, 2, 3, 4, 5, 6, 7, 8};
    auto r = decode_sdr(junk);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, ErrorCode::Unsupported);
    EXPECT_FALSE(decode_sdr_file(kDir / "does_not_exist.png"));
}
