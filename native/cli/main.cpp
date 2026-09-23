// rudra-native: the headless tool over librudra.
//
//   rudra-native version
//   rudra-native info <package>
//   rudra-native diff <package> [--runtime libtorch|onnxruntime|all] [--device cpu|cuda|mps|directml|coreml|rocm|openvino]
//   rudra-native bench <package> [--runtime ...] [--device ...] [--size 1920x1080] [--iters 5]
//
// `diff` is the native half of Gate A (NATIVE_ARCHITECTURE.md 12): it runs the
// package's golden frames through each compiled runtime, untiled and tiled, and
// compares against what eager PyTorch produced when the package was exported.
// Exit code 0 only if every runtime asked for passes.
//
// `bench` times inference for the budget table (NATIVE_ARCHITECTURE.md 6.6):
// one warm-up, then the median of --iters runs, untiled and tiled 512/64, on
// a synthetic frame. Wall time, fields back in host memory, so a GPU run is
// timed to completion. Each result is also printed as a BENCH line for scripts.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rudra/core/model_manifest.hpp"
#include "rudra/infer/tiler.hpp"
#include "rudra/platform/npy.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

int fail(const Error& e) {
    std::fprintf(stderr, "error [%s]: %s\n  %s\n", to_string(e.code), e.message.c_str(), e.detail.c_str());
    return 2;
}

Result<SdrImage> load_sdr(const fs::path& file) {
    auto a = read_npy(file);
    if (!a) return a.error();
    if (a->shape.size() != 3 || a->shape[2] != 3)
        return make_error(ErrorCode::ParseError, "A golden input is not an H x W x 3 image.", file.string());
    const int h = static_cast<int>(a->shape[0]), w = static_cast<int>(a->shape[1]);
    PlanarBuffer b(3, h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c)
                b.at(c, y, x) = a->data[(static_cast<std::size_t>(y) * w + x) * 3 + c];
    return SdrImage(std::move(b));
}

struct Stat {
    double max_abs = 0.0;
    double excess = -1e300;   // max(|d| - (atol + rtol |ref|)); <= 0 passes
    void add(double ref, double got, const Tolerance& t) {
        const double d = std::abs(ref - got);
        max_abs = std::max(max_abs, d);
        excess = std::max(excess, d - (t.atol + t.rtol * std::abs(ref)));
    }
};

Result<void> compare(const fs::path& file, std::span<const float> got, const Tolerance& tol, Stat& s) {
    auto a = read_npy(file);
    if (!a) return a.error();
    if (static_cast<std::size_t>(a->size()) != got.size())
        return make_error(ErrorCode::ParityError, "An output has the wrong size.",
                          file.string() + ": expected " + std::to_string(a->size()) + " values, got " +
                              std::to_string(got.size()));
    for (std::size_t i = 0; i < got.size(); ++i) s.add(a->data[i], got[i], tol);
    return {};
}

Result<Device> parse_device(const std::string& s) {
    static const std::map<std::string, Device> m{{"cpu", Device::Cpu},       {"cuda", Device::Cuda},
                                                 {"mps", Device::Mps},       {"directml", Device::DirectML},
                                                 {"coreml", Device::CoreML}, {"rocm", Device::Rocm},
                                                 {"openvino", Device::OpenVino}};
    auto it = m.find(s);
    if (it == m.end()) return make_error(ErrorCode::InvalidArgument, "Unknown device.", s);
    return it->second;
}

int cmd_info(const fs::path& pkg) {
    auto m = read_manifest(pkg);
    if (!m) return fail(m.error());
    std::printf("%s  (contract %s)\n", m->name.c_str(), m->contract.c_str());
    std::printf("  source     %s  sha256 %s\n", m->source_file.c_str(), m->source_sha256.c_str());
    std::printf("  heads      residual_gate=%d shadow_gate=%d curve=%d (curve params %d)\n", m->has_residual_gate,
                m->has_shadow_gate, m->has_curve, m->curve_params);
    std::printf("  baseline   corpus_ev %+.1f  log_scale %.1f  max_hdr %.1f\n", m->corpus_ev, m->log_scale, m->max_hdr);
    std::printf("  tiling     %d px, overlap %d\n", m->tile_size, m->overlap);
    auto v = verify_package_files(*m);
    std::printf("  files      %s\n", v ? "sha256 verified" : v.error().detail.c_str());
    std::printf("  runtimes   ");
    for (auto r : compiled_runtimes()) std::printf("%s ", to_string(r));
    std::printf("\n");
    return v ? 0 : 3;
}

int cmd_diff(const fs::path& pkg, const std::string& which, Device device) {
    auto m = read_manifest(pkg);
    if (!m) return fail(m.error());
    if (auto v = verify_package_files(*m); !v) return fail(v.error());

    std::ifstream in(m->root / m->golden);
    nlohmann::json g;
    try {
        in >> g;
    } catch (const std::exception& e) {
        return fail(make_error(ErrorCode::ParseError, "The golden index could not be read.", e.what()));
    }
    const fs::path gdir = (m->root / m->golden).parent_path();

    std::vector<Runtime> runtimes;
    for (auto r : compiled_runtimes())
        if (which == "all" || which == to_string(r)) runtimes.push_back(r);
    if (runtimes.empty())
        return fail(make_error(ErrorCode::Unsupported, "No requested runtime is compiled into this build.", which));

    bool all_pass = true;
    for (auto rt : runtimes) {
        auto backend = rt == Runtime::LibTorch ? make_libtorch_backend(*m, device) : make_onnxruntime_backend(*m, device);
        if (!backend) return fail(backend.error());
        const auto info = (*backend)->info();
        // CPU LibTorch runs eager's own kernels: held to bit-exact-grade 1e-5.
        // LibTorch on a GPU is true fp32 with the vendor's summation order;
        // ONNX Runtime is a different graph compiler on any device.
        const char* tol_key = rt == Runtime::OnnxRuntime ? "onnx" : device == Device::Cpu ? "torchscript" : "gpu_fp32";
        const Tolerance tol = m->tolerance.at(tol_key);
        std::map<std::string, Stat> stats;
        int frames = 0;

        for (const auto& f : g.at("frames")) {
            auto sdr = load_sdr(gdir / f.at("sdr").at("file").get<std::string>());
            if (!sdr) return fail(sdr.error());
            auto r = infer_frame(**backend, *sdr, TileConfig{0, 0});
            if (!r) return fail(r.error());
            const float scale = r->scalars.residual_scale, weight = r->scalars.shadow_weight;
            const std::vector<std::pair<std::string, std::span<const float>>> outs{
                {"residual_scale", std::span<const float>(&scale, 1)},
                {"shadow_weight", std::span<const float>(&weight, 1)},
                {"curve_params", r->scalars.curve_params},
                {"residual", r->fields.residual.span()},
                {"highlight", r->fields.highlight.span()},
                {"shadow", r->fields.shadow.span()}};
            for (const auto& [key, span] : outs)
                if (auto c = compare(gdir / f.at(key).at("file").get<std::string>(), span, tol, stats[key]); !c)
                    return fail(c.error());
            ++frames;
        }

        if (g.contains("stitch") && !g.at("stitch").is_null()) {
            const auto& s = g.at("stitch");
            auto sdr = load_sdr(gdir / s.at("sdr").at("file").get<std::string>());
            if (!sdr) return fail(sdr.error());
            auto r = infer_frame(**backend, *sdr, TileConfig{s.at("tile_size").get<int>(), s.at("overlap").get<int>()});
            if (!r) return fail(r.error());
            for (const auto& [key, buf] : std::vector<std::pair<std::string, const PlanarBuffer*>>{
                     {"stitched residual", &r->fields.residual},
                     {"stitched highlight", &r->fields.highlight},
                     {"stitched shadow", &r->fields.shadow}}) {
                const std::string field = key.substr(key.find(' ') + 1);
                if (auto c = compare(gdir / s.at(field).at("file").get<std::string>(), buf->span(), tol, stats[key]); !c)
                    return fail(c.error());
            }
        }

        bool pass = true;
        std::printf("%s %s on %s (%s): %d golden frames + stitch, atol %.0e rtol %.0e (%s)\n", to_string(info.runtime),
                    info.version.c_str(), to_string(info.device), info.detail.c_str(), frames, tol.atol, tol.rtol, tol_key);
        for (const auto& [key, st] : stats) {
            const bool ok = st.excess <= 0.0;
            pass = pass && ok;
            std::printf("  %-20s max |d| %.3e  %s\n", key.c_str(), st.max_abs, ok ? "pass" : "FAIL");
        }
        std::printf("  => %s\n\n", pass ? "PASS" : "FAIL");
        all_pass = all_pass && pass;
    }
    return all_pass ? 0 : 1;
}

int cmd_bench(const fs::path& pkg, const std::string& which, Device device, int w, int h, int iters) {
    auto m = read_manifest(pkg);
    if (!m) return fail(m.error());
    // Deterministic synthetic frame: a lit gradient with a clipped patch and
    // fine texture, so every head has something to do.
    PlanarBuffer b(3, h, w);
    std::uint32_t seed = 20260923u;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            seed = seed * 1664525u + 1013904223u;
            const float n = float(seed >> 8) / float(1u << 24) * 0.02f;
            const float g = float(x) / float(w) * 0.8f + float(y) / float(h) * 0.2f;
            const bool clip = std::abs(x - w * 3 / 4) < w / 10 && std::abs(y - h / 3) < h / 8;
            for (int c = 0; c < 3; ++c) b.at(c, y, x) = clip ? 1.0f : std::clamp(g * (1.0f - 0.15f * float(c)) + n, 0.0f, 1.0f);
        }
    const SdrImage frame(std::move(b));

    std::vector<Runtime> runtimes;
    for (auto r : compiled_runtimes())
        if (which == "all" || which == to_string(r)) runtimes.push_back(r);
    if (runtimes.empty())
        return fail(make_error(ErrorCode::Unsupported, "No requested runtime is compiled into this build.", which));

    using clock = std::chrono::steady_clock;
    for (auto rt : runtimes) {
        auto backend = rt == Runtime::LibTorch ? make_libtorch_backend(*m, device) : make_onnxruntime_backend(*m, device);
        if (!backend) return fail(backend.error());
        const auto info = (*backend)->info();
        std::printf("%s %s on %s (%s), %dx%d, fp32, median of %d after one warm-up\n", to_string(info.runtime),
                    info.version.c_str(), to_string(info.device), info.detail.c_str(), w, h, iters);
        const std::pair<const char*, TileConfig> modes[] = {{"untiled", TileConfig{0, 0}},
                                                            {"tiled", TileConfig{m->tile_size, m->overlap}}};
        for (const auto& [name, cfg] : modes) {
            std::vector<double> ms;
            for (int i = 0; i <= iters; ++i) {
                const auto t0 = clock::now();
                auto r = infer_frame(**backend, frame, cfg);
                const auto t1 = clock::now();
                if (!r) {
                    std::printf("  %-8s %s\n", name, r.error().message.c_str());
                    ms.clear();
                    break;
                }
                if (i > 0) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            if (ms.empty()) continue;
            std::sort(ms.begin(), ms.end());
            const double med = ms[ms.size() / 2];
            std::printf("  %-8s median %8.1f ms  min %8.1f ms\n", name, med, ms.front());
            std::printf("BENCH %s %s %dx%d %s %.2f\n", to_string(info.runtime), to_string(info.device), w, h, name, med);
        }
    }
    return 0;
}

void usage() {
    std::fprintf(stderr,
                 "usage: rudra-native version\n"
                 "       rudra-native info <package>\n"
                 "       rudra-native diff <package> [--runtime libtorch|onnxruntime|all] [--device cpu|cuda|mps|"
                 "directml|coreml|rocm|openvino]\n"
                 "       rudra-native bench <package> [--runtime ...] [--device ...] [--size WxH] [--iters N]\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) { usage(); return 64; }
    if (args[0] == "version") {
        std::printf("rudra-native 0.1.0 (model contract %d.x)\n", kSupportedContractMajor);
        return 0;
    }
    if (args.size() < 2) { usage(); return 64; }
    const fs::path pkg = args[1];
    if (args[0] == "info") return cmd_info(pkg);
    if (args[0] == "diff") {
        std::string runtime = "all", device = "cpu";
        for (std::size_t i = 2; i + 1 < args.size(); i += 2) {
            if (args[i] == "--runtime") runtime = args[i + 1];
            else if (args[i] == "--device") device = args[i + 1];
            else { usage(); return 64; }
        }
        auto d = parse_device(device);
        if (!d) return fail(d.error());
        return cmd_diff(pkg, runtime, *d);
    }
    if (args[0] == "bench") {
        std::string runtime = "all", device = "cpu";
        int w = 1920, h = 1080, iters = 5;
        for (std::size_t i = 2; i + 1 < args.size(); i += 2) {
            if (args[i] == "--runtime") runtime = args[i + 1];
            else if (args[i] == "--device") device = args[i + 1];
            else if (args[i] == "--size") {
                if (std::sscanf(args[i + 1].c_str(), "%dx%d", &w, &h) != 2 || w <= 0 || h <= 0) { usage(); return 64; }
            } else if (args[i] == "--iters") iters = std::max(1, std::atoi(args[i + 1].c_str()));
            else { usage(); return 64; }
        }
        auto d = parse_device(device);
        if (!d) return fail(d.error());
        return cmd_bench(pkg, runtime, *d, w, h, iters);
    }
    usage();
    return 64;
}
