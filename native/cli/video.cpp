#include "video.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>

#include "rudra/core/model_manifest.hpp"
#include "rudra/infer/backend.hpp"
#include "rudra/video/convert.hpp"

namespace rudra {
namespace fs = std::filesystem;

namespace {

int usage_error(const std::string& message) {
    std::fprintf(stderr, "rudra-native video: error: %s\n", message.c_str());
    return 64;
}

bool one_of(const std::string& v, const std::set<std::string>& choices) { return choices.count(v) > 0; }

}  // namespace

int cmd_video(const std::vector<std::string>& args) {
    // video.py add_arguments, with the package in the checkpoint's place.
    if (args.size() < 2) return usage_error("usage: rudra-native video <package> <input> --output <file> [options]");
    VideoConvertArgs a;
    a.package = args[0];
    a.input = args[1];
    std::string runtime = "auto", device = "cpu";
    bool have_output = false;
    for (std::size_t i = 2; i < args.size(); ++i) {
        const std::string& k = args[i];
        if (i + 1 >= args.size()) return usage_error("argument " + k + ": expected one argument");
        const std::string v = args[++i];
        auto number = [&](double& out) {
            char* end = nullptr;
            out = std::strtod(v.c_str(), &end);
            return end && *end == '\0' && !v.empty();
        };
        auto integer = [&](int& out) {
            char* end = nullptr;
            const long l = std::strtol(v.c_str(), &end, 10);
            out = int(l);
            return end && *end == '\0' && !v.empty();
        };
        double d = 0;
        if (k == "--output") a.output = v, have_output = true;
        else if (k == "--format") {
            if (!one_of(v, {"hdr10", "hlg", "prores422", "prores422hq", "prores4444"})) return usage_error("invalid --format: " + v);
            a.format = v;
        } else if (k == "--alpha-mode") {
            if (v != "straight") return usage_error("invalid --alpha-mode: " + v);
            a.alpha_mode = v;
        } else if (k == "--input-transfer") {
            if (!one_of(v, {"auto", "srgb", "rec709", "gamma22", "gamma24"})) return usage_error("invalid --input-transfer: " + v);
            a.input_transfer = v;
        } else if (k == "--input-primaries") {
            if (!one_of(v, {"auto", "rec709", "rec2020"})) return usage_error("invalid --input-primaries: " + v);
            a.input_primaries = v;
        } else if (k == "--input-matrix") {
            if (!one_of(v, {"auto", "bt709", "bt2020nc", "gbr"})) return usage_error("invalid --input-matrix: " + v);
            a.input_matrix = v;
        } else if (k == "--input-range") {
            if (!one_of(v, {"auto", "full", "limited"})) return usage_error("invalid --input-range: " + v);
            a.input_range = v;
        } else if (k == "--audio") {
            if (!one_of(v, {"copy", "aac", "none"})) return usage_error("invalid --audio: " + v);
            a.audio = v;
        } else if (k == "--peak-nits") { if (!number(a.peak_nits)) return usage_error("invalid --peak-nits: " + v); }
        else if (k == "--min-nits") { if (!number(a.min_nits)) return usage_error("invalid --min-nits: " + v); }
        else if (k == "--knee-nits") { if (!number(d)) return usage_error("invalid --knee-nits: " + v); a.knee_nits = d; }
        else if (k == "--crf") { if (!integer(a.crf)) return usage_error("invalid --crf: " + v); }
        else if (k == "--preset") {
            if (!one_of(v, {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}))
                return usage_error("invalid --preset: " + v);
            a.preset = v;
        } else if (k == "--tile-size") { if (!integer(a.tile_size)) return usage_error("invalid --tile-size: " + v); }
        else if (k == "--tile-overlap") { if (!integer(a.tile_overlap)) return usage_error("invalid --tile-overlap: " + v); }
        else if (k == "--shadow-smoothing") { if (!number(a.shadow_smoothing)) return usage_error("invalid --shadow-smoothing: " + v); }
        else if (k == "--cut-threshold") { if (!number(a.cut_threshold)) return usage_error("invalid --cut-threshold: " + v); }
        else if (k == "--work-dir") a.work_dir = fs::path(v);
        else if (k == "--runtime") runtime = v;
        else if (k == "--device") device = v;
        else return usage_error("unrecognized argument: " + k);
    }
    if (!have_output) return usage_error("the following arguments are required: --output");

    auto fail = [](const Error& e) {
        std::fprintf(stderr, "error [%s]: %s\n", to_string(e.code), e.message.c_str());
        if (!e.detail.empty()) std::fprintf(stderr, "  %s\n", e.detail.c_str());
        return 2;
    };
    auto m = read_manifest(a.package);
    if (!m) return fail(m.error());
    static const std::map<std::string, Device> devices{{"cpu", Device::Cpu},       {"cuda", Device::Cuda},
                                                       {"mps", Device::Mps},       {"directml", Device::DirectML},
                                                       {"coreml", Device::CoreML}, {"rocm", Device::Rocm},
                                                       {"openvino", Device::OpenVino}};
    const auto dev = devices.find(device);
    if (dev == devices.end()) return usage_error("invalid --device: " + device);
    Result<std::unique_ptr<InferenceBackend>> b = make_error(ErrorCode::Unsupported, "No runtime");
    if (runtime == "libtorch") b = make_libtorch_backend(*m, dev->second);
    else if (runtime == "onnxruntime") b = make_onnxruntime_backend(*m, dev->second);
    else {
        for (auto r : compiled_runtimes())
            if (r == Runtime::LibTorch) b = make_libtorch_backend(*m, dev->second);
        if (!b) b = make_onnxruntime_backend(*m, dev->second);
    }
    if (!b) return fail(b.error());
    VideoConvertHooks hooks;
    hooks.print = [](const std::string& line) {
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
    };
    auto r = convert_video(a, *m, **b, hooks);
    if (!r) return fail(r.error());
    return 0;
}

}  // namespace rudra
