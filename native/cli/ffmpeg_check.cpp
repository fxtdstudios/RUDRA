#include "ffmpeg_check.hpp"

#include <cstdio>

#include "rudra/video/ffmpeg_check.hpp"

namespace rudra {
namespace fs = std::filesystem;

// What this machine's ffmpeg can deliver, and whether a 16-frame HDR10 export
// runs through it and passes QC. Exit 0 when every delivery is possible and
// the self-test passes, 1 when something is missing or failed.
int cmd_ffmpeg_check(const std::vector<std::string>& args) {
    std::optional<fs::path> ffmpeg, ffprobe;
    bool force = false, self_test = true;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--force") force = true;
        else if (args[i] == "--no-self-test") self_test = false;
        else if (args[i] == "--ffmpeg" && i + 1 < args.size()) ffmpeg = fs::path(args[++i]);
        else if (args[i] == "--ffprobe" && i + 1 < args.size()) ffprobe = fs::path(args[++i]);
        else {
            std::fprintf(stderr, "usage: rudra-native ffmpeg-check [--ffmpeg PATH] [--ffprobe PATH] [--force] [--no-self-test]\n");
            return 64;
        }
    }
    auto caps = probe_ffmpeg(ffmpeg, ffprobe);
    if (!caps) {
        std::fprintf(stderr, "error: %s\n", caps.error().message.c_str());
        return 2;
    }
    pyjson::Value report = caps->to_json();
    bool ok = caps->missing().empty();
#ifdef RUDRA_HAVE_STILL_DECODE
    if (self_test) {
        auto t = ffmpeg_self_test(*caps, default_cache_dir(), force);
        if (!t) {
            std::fprintf(stderr, "error: %s\n", t.error().message.c_str());
            return 2;
        }
        pyjson::set(*std::get<std::shared_ptr<pyjson::Dict>>(report.v), "self_test",
                    pyjson::Dict{{"passed", t->passed}, {"cached", t->cached}, {"seconds", t->seconds},
                                 {"detail", t->detail}});
        ok = ok && t->passed;
    }
#else
    (void)force;
    (void)self_test;
#endif
    std::printf("%s\n", pyjson::dumps(report, 2).c_str());
    return ok ? 0 : 1;
}

}  // namespace rudra
