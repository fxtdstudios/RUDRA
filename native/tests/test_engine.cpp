#include <gtest/gtest.h>

#include "rudra/engine/scheduling.hpp"
#include "rudra/render/viewer_backend.hpp"

using namespace rudra;

TEST(Generation, ASeekMakesOlderResultsStale) {
    Generation g;
    const auto requested = g.current();
    EXPECT_TRUE(g.is_current(requested));
    g.bump();   // the user seeks while the request is in flight
    EXPECT_FALSE(g.is_current(requested));
    EXPECT_TRUE(g.is_current(g.current()));
}

TEST(OutputScale, EachSwapchainGetsItsOwnUnit) {
    const Nits thousand{1000.0f}, sdr_white{203.0f};
    EXPECT_FLOAT_EQ(output_scale(OutputPath::ScRgb, thousand, sdr_white), 12.5f);        // 1.0 = 80 nits
    EXPECT_FLOAT_EQ(output_scale(OutputPath::Edr, thousand, sdr_white), 1000.0f / 203.0f);
    EXPECT_FLOAT_EQ(output_scale(OutputPath::Hdr10, thousand, sdr_white), 0.1f);          // then PQ
}
