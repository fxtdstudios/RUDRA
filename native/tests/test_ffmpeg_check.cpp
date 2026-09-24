// Phase 4 step 10: ffmpeg's capabilities asked before any work, and the
// 16-frame HDR10 self-test through the real pipeline, cached per build.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "rudra/platform/process.hpp"
#include "rudra/video/ffmpeg_check.hpp"

using namespace rudra;
namespace fs = std::filesystem;

TEST(FfmpegCheck, CapabilitiesOfThePathBuild) {
    if (!find_executable("ffmpeg") || !find_executable("ffprobe")) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    auto c = probe_ffmpeg();
    ASSERT_TRUE(c) << c.error().message;
    EXPECT_EQ(c->ffmpeg_sha256.size(), 64u);
    EXPECT_TRUE(c->version.rfind("ffmpeg version", 0) == 0) << c->version;
    EXPECT_TRUE(c->zscale && c->libx265 && c->prores_ks && c->alphaextract) << pyjson::dumps(c->to_json(), 2);
    EXPECT_TRUE(c->missing().empty());
}

#ifndef _WIN32
TEST(FfmpegCheck, AMissingEncoderIsReportedBeforeAnyWork) {
    // A stand-in ffmpeg whose tables lack libx265 and zscale.
    const fs::path dir = fs::temp_directory_path() / "rudra-fake-ffmpeg";
    fs::create_directories(dir);
    const fs::path fake = dir / "ffmpeg";
    {
        std::ofstream f(fake);
        f << "#!/bin/sh\n"
             "case \"$*\" in\n"
             "  *-version*) echo 'ffmpeg version 0.0-fake' ;;\n"
             "  *-encoders*) echo ' V....D prores_ks            Apple ProRes (iCodec Pro)' ;;\n"
             "  *-filters*) echo ' ... alphaextract      V->N       Extract an alpha channel' ;;\n"
             "esac\n";
    }
    fs::permissions(fake, fs::perms::owner_all);
    auto c = probe_ffmpeg(fake, fake);
    ASSERT_TRUE(c) << c.error().message;
    EXPECT_FALSE(c->libx265);
    EXPECT_FALSE(c->zscale);
    EXPECT_TRUE(c->prores_ks);
    ASSERT_EQ(c->missing().size(), 2u);
    EXPECT_EQ(c->missing()[1], "the libx265 encoder, for HDR10 and HLG");
#ifdef RUDRA_HAVE_STILL_DECODE
    auto t = ffmpeg_self_test(*c, dir / "cache");
    ASSERT_TRUE(t);
    EXPECT_FALSE(t->passed);
    EXPECT_EQ(t->detail, "This FFmpeg build needs libx265 and zscale support");
    EXPECT_FALSE(fs::exists(dir / "cache"));
#endif
    fs::remove_all(dir);
}
#endif

#ifdef RUDRA_HAVE_STILL_DECODE
TEST(FfmpegCheck, SelfTestPassesAndIsCachedPerBuild) {
    if (!find_executable("ffmpeg") || !find_executable("ffprobe")) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    auto c = probe_ffmpeg();
    ASSERT_TRUE(c);
    const fs::path cache = fs::temp_directory_path() / "rudra-selftest-cache";
    fs::remove_all(cache);
    auto first = ffmpeg_self_test(*c, cache);
    ASSERT_TRUE(first) << first.error().message;
    EXPECT_TRUE(first->passed) << first->detail;
    EXPECT_FALSE(first->cached);
    auto second = ffmpeg_self_test(*c, cache);
    ASSERT_TRUE(second);
    EXPECT_TRUE(second->passed && second->cached);
    FfmpegCapabilities other = *c;
    other.version += " (rebuilt)";   // same binary, a different build under it
    auto third = ffmpeg_self_test(other, cache);
    ASSERT_TRUE(third);
    EXPECT_TRUE(third->passed);
    EXPECT_FALSE(third->cached);
    fs::remove_all(cache);
}
#endif
