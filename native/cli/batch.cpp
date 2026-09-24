#include "batch.hpp"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <map>

#include "rudra/video/queue_runner.hpp"

namespace rudra {
namespace fs = std::filesystem;

namespace {
std::atomic<bool> g_stop{false};
extern "C" void on_interrupt(int) { g_stop = true; }

std::vector<fs::path> default_roots(const fs::path& queue) {
    std::vector<fs::path> roots;
    if (const char* env = std::getenv("RUDRA_PACKAGE_ROOTS")) {
#ifdef _WIN32
        const char sep = ';';
#else
        const char sep = ':';
#endif
        std::string s(env);
        std::size_t at = 0;
        while (at <= s.size()) {
            const std::size_t end = std::min(s.find(sep, at), s.size());
            if (end > at) roots.emplace_back(s.substr(at, end - at));
            at = end + 1;
        }
    }
    const fs::path qdir = fs::absolute(queue).parent_path();
    roots.push_back(qdir / "dist" / "models");
    roots.push_back(fs::current_path() / "dist" / "models");
    roots.push_back(qdir);
    return roots;
}
}  // namespace

int cmd_batch(const std::vector<std::string>& args) {
    auto usage = [] {
        std::fprintf(stderr, "usage: rudra-native batch run <queue.json> [--retry-failed] [--package DIR] [--runtime ...] [--device ...]\n"
                             "       rudra-native batch status <queue.json>\n");
        return 64;
    };
    if (args.size() < 2) return usage();
    const fs::path queue = args[1];
    if (args[0] == "status") {
        std::printf("%s\n", queue_status(queue).c_str());
        return 0;
    }
    if (args[0] != "run") return usage();
    bool retry = false;
    VideoQueueOptions o;
    for (std::size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "--retry-failed") { retry = true; continue; }
        if (i + 1 >= args.size()) return usage();
        const std::string& k = args[i];
        const std::string v = args[++i];
        if (k == "--package") o.package = fs::path(v);
        else if (k == "--runtime") o.runtime = v;
        else if (k == "--device") {
            static const std::map<std::string, Device> m{{"cpu", Device::Cpu}, {"cuda", Device::Cuda}, {"mps", Device::Mps},
                                                         {"directml", Device::DirectML}, {"coreml", Device::CoreML},
                                                         {"rocm", Device::Rocm}, {"openvino", Device::OpenVino}};
            const auto it = m.find(v);
            if (it == m.end()) return usage();
            o.device = it->second;
        } else return usage();
    }
    o.package_roots = default_roots(queue);
    o.cancel = &g_stop;
    o.print = [](const std::string& line) {
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
    };
    std::signal(SIGINT, on_interrupt);
#ifdef RUDRA_HAVE_STILL_DECODE
    auto r = run_video_queue(queue, retry, o);
    if (!r) {
        std::fprintf(stderr, "error [%s]: %s\n", to_string(r.error().code), r.error().message.c_str());
        return r.error().code == ErrorCode::Cancelled ? 130 : 2;
    }
    return *r;
#else
    std::fprintf(stderr, "this build has no video pipeline (OpenCV is off)\n");
    return 2;
#endif
}

}  // namespace rudra
