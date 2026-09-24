// rudra-native: the headless tool over librudra.
//
//   rudra-native version
//   rudra-native info <package>
//   rudra-native diff <package> [--runtime libtorch|onnxruntime|all] [--device cpu|cuda|mps|directml|coreml|rocm|openvino]
//   rudra-native bench <package> [--runtime ...] [--device ...] [--size 1920x1080] [--iters 5]
//   rudra-native master <package> <image> --out <file.exr> [--runtime ...] [--device ...] [--params JSON]
//   rudra-native master-check <package> <golden-dir> [--runtime ...] [--device ...]
//   rudra-native bench-scopes [--iters 7]
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
//
// `bench-scopes` times what the viewer does on the CPU after a slider move
// (core/scopes.cpp: the 768-side sample, computeStats, buildScopes and the
// vectorscope) at 1080p and 4K, median of --iters, one thread.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rudra/core/model_manifest.hpp"
#include "rudra/core/scopes.hpp"
#include "rudra/core/view.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/infer/self_test.hpp"
#include "rudra/infer/tiler.hpp"
#include "rudra/platform/npy.hpp"

#ifdef RUDRA_HAVE_STILL_DECODE
#include "master.hpp"
#endif

using namespace rudra;
namespace fs = std::filesystem;

namespace {

int fail(const Error& e) {
    std::fprintf(stderr, "error [%s]: %s\n  %s\n", to_string(e.code), e.message.c_str(), e.detail.c_str());
    return 2;
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

    std::vector<Runtime> runtimes;
    for (auto r : compiled_runtimes())
        if (which == "all" || which == to_string(r)) runtimes.push_back(r);
    if (runtimes.empty())
        return fail(make_error(ErrorCode::Unsupported, "No requested runtime is compiled into this build.", which));

    bool all_pass = true;
    for (auto rt : runtimes) {
        auto backend = rt == Runtime::LibTorch ? make_libtorch_backend(*m, device) : make_onnxruntime_backend(*m, device);
        if (!backend) return fail(backend.error());
        auto r = self_test(*m, **backend);   // infer/self_test, shared with the app's first load
        if (!r) return fail(r.error());
        std::printf("%s %s on %s (%s): %d golden frames + stitch, atol %.0e rtol %.0e (%s)\n",
                    to_string(r->backend.runtime), r->backend.version.c_str(), to_string(r->backend.device),
                    r->backend.detail.c_str(), r->frames, r->tolerance.atol, r->tolerance.rtol, r->tolerance_key.c_str());
        for (const auto& [key, st] : r->outputs)
            std::printf("  %-20s max |d| %.3e  %s\n", key.c_str(), st.max_abs, st.pass() ? "pass" : "FAIL");
        std::printf("  => %s\n\n", r->pass() ? "PASS" : "FAIL");
        all_pass = all_pass && r->pass();
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

#ifdef RUDRA_HAVE_STILL_DECODE
Result<std::unique_ptr<InferenceBackend>> backend_for(const ModelManifest& m, const std::string& runtime, Device device) {
    if (runtime == "onnxruntime") return make_onnxruntime_backend(m, device);
    if (runtime == "libtorch") return make_libtorch_backend(m, device);
    // Default: the reference runtime when it is compiled in.
    for (auto r : compiled_runtimes())
        if (r == Runtime::LibTorch) return make_libtorch_backend(m, device);
    return make_onnxruntime_backend(m, device);
}

int cmd_master(const fs::path& pkg, const fs::path& image, const fs::path& out, const std::string& runtime,
               Device device, const std::string& params) {
    auto m = read_manifest(pkg);
    if (!m) return fail(m.error());
    auto b = backend_for(*m, runtime, device);
    if (!b) return fail(b.error());
    auto q = master_request_from_json(params.empty() ? "{}" : params);
    if (!q) return fail(q.error());
    if (q->checkpoint.empty()) q->checkpoint = m->source_file;
    auto r = render_master(*m, **b, image, *q, out);
    if (!r) return fail(r.error());
    std::printf("%s  %dx%d, %d-bit source, MaxCLL %d, MaxFALL %d, peak %.1f nits\n  sidecar %s\n",
                r->exr.string().c_str(), r->width, r->height, r->source_bits, r->maxcll, r->maxfall, r->peak_nits,
                r->sidecar.string().c_str());
    return 0;
}

int half_ulp(std::uint16_t a, std::uint16_t b) {
    auto key = [](std::uint16_t h) { return (h & 0x8000) ? -int(h & 0x7fff) : int(h); };
    return std::abs(key(a) - key(b));
}

std::string read_text(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}

// Every golden master from tools/emit_master_golden.py, rendered here and
// compared: EXR pixels within 1 half-float ulp, every EXR header attribute
// equal, every sidecar field equal (peak within its 0.1-nit rounding).
int cmd_master_check(const fs::path& pkg, const fs::path& dir, const std::string& runtime, Device device) {
    auto m = read_manifest(pkg);
    if (!m) return fail(m.error());
    auto b = backend_for(*m, runtime, device);
    if (!b) return fail(b.error());
    std::ifstream in(dir / "index.json");
    if (!in) return fail(make_error(ErrorCode::NotFound, "No master goldens there.", dir.string()));
    const auto idx = nlohmann::json::parse(in);
    const fs::path tmp = fs::temp_directory_path() / "rudra_master_check";
    fs::create_directories(tmp);
    const auto info = (*b)->info();
    std::printf("master parity: %s %s on %s, %zu cases (oracle %s)\n", to_string(info.runtime), info.version.c_str(),
                to_string(info.device), idx.at("cases").size(), idx.value("oracle", "").c_str());
    bool all = true;
    for (const auto& c : idx.at("cases")) {
        const std::string name = c.at("name");
        auto q = master_request_from_json(c.at("params").dump());
        if (!q) return fail(q.error());
        const fs::path out = tmp / c.at("exr").get<std::string>();
        auto r = render_master(*m, **b, dir / c.at("image").get<std::string>(), *q, out);
        if (!r) return fail(r.error());
        auto got = read_exr(out), want = read_exr(dir / c.at("exr").get<std::string>());
        if (!got || !want) return fail(!got ? got.error() : want.error());

        int worst = 0;
        std::size_t off = 0;
        for (std::size_t i = 0; i < want->half_bits.size() && i < got->half_bits.size(); ++i) {
            const int d = half_ulp(got->half_bits[i], want->half_bits[i]);
            worst = std::max(worst, d);
            off += d != 0;
        }
        const bool same_shape = got->half_bits.size() == want->half_bits.size() && !want->half_bits.empty();
        std::string header_diff;
        if (got->attributes.size() != want->attributes.size()) header_diff = "attribute count";
        for (std::size_t i = 0; header_diff.empty() && i < want->attributes.size(); ++i)
            if (got->attributes[i] != want->attributes[i] || got->attribute_types[i] != want->attribute_types[i])
                header_diff = want->attributes[i].first;

        const auto gs = nlohmann::json::parse(read_text(r->sidecar));
        const auto ws = nlohmann::json::parse(read_text(dir / c.at("sidecar").get<std::string>()));
        std::string side_diff;
        for (const auto& [k, v] : ws.items()) {
            if (!gs.contains(k)) { side_diff = k; break; }
            if (k == "peak_nits") {
                if (std::abs(gs[k].get<double>() - v.get<double>()) > 0.1 + 1e-9) side_diff = k;
            } else if (gs[k] != v) {
                side_diff = k;
            }
            if (!side_diff.empty()) break;
        }
        const bool bytes_equal = read_text(r->sidecar) == read_text(dir / c.at("sidecar").get<std::string>());
        const bool ok = same_shape && worst <= 1 && header_diff.empty() && side_diff.empty();
        all = all && ok;
        std::printf("  %-18s %dx%d  pixels %s (worst %d half ulp, %zu of %zu off)  header %s  sidecar %s%s  => %s\n",
                    name.c_str(), r->width, r->height, worst <= 1 ? "ok" : "FAIL", worst, off, want->half_bits.size(),
                    header_diff.empty() ? "equal" : ("differs at " + header_diff).c_str(),
                    side_diff.empty() ? "equal" : ("differs at " + side_diff).c_str(),
                    bytes_equal ? ", byte-identical" : "", ok ? "PASS" : "FAIL");
    }
    std::printf("  => %s\n", all ? "PASS" : "FAIL");
    return all ? 0 : 1;
}
#endif

int cmd_bench_scopes(int iters) {
    std::printf("Viewer measurements and scopes on the CPU (sample, computeStats, buildScopes, vectorscope), median of %d\n",
                iters);
    for (auto [w, h] : {std::pair{1920, 1080}, std::pair{3840, 2160}}) {
        PlanarBuffer m(3, h, w), b(3, h, w);
        std::uint32_t x = 2463534242u;   // xorshift: a busy, noise-like frame
        for (auto* buf : {&m, &b})
            for (float& v : buf->span()) {
                x ^= x << 13; x ^= x >> 17; x ^= x << 5;
                v = float(x % 100000u) / 100000.0f * 0.3f;
            }
        const std::vector<float> hi(std::size_t(w) * h, 0.7f), sh(std::size_t(w) * h, 0.2f);
        const Reductions rm = reduce_ladder(m), rb = reduce_ladder(b);
        std::vector<double> t_measure, t_vector;
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            const SampleGrid g = sample_grid(w, h);
            const PlanarBuffer ms = take_sample(m, g), bs = take_sample(b, g);
            const Measured r = measure_view(ms, bs, g, hi, sh, MaskCoverage{}, rm, rb, std::size_t(w) * h);
            const auto t1 = std::chrono::steady_clock::now();
            const auto img = vectorscope(ms);
            const auto t2 = std::chrono::steady_clock::now();
            if (r.scopes.mid.empty() || img.empty()) return 1;
            t_measure.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            t_vector.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
        std::sort(t_measure.begin(), t_measure.end());
        std::sort(t_vector.begin(), t_vector.end());
        const SampleGrid g = sample_grid(w, h);
        const double tm = t_measure[t_measure.size() / 2], tv = t_vector[t_vector.size() / 2];
        std::printf("  %dx%d (sample %dx%d): measurements and scopes %.2f ms, vectorscope %.2f ms\n", w, h, g.width,
                    g.height, tm, tv);
        std::printf("BENCH scopes %dx%d %.3f %.3f\n", w, h, tm, tv);
    }
    return 0;
}

void usage() {
    std::fprintf(stderr,
                 "usage: rudra-native version\n"
                 "       rudra-native info <package>\n"
                 "       rudra-native diff <package> [--runtime libtorch|onnxruntime|all] [--device cpu|cuda|mps|"
                 "directml|coreml|rocm|openvino]\n"
                 "       rudra-native bench <package> [--runtime ...] [--device ...] [--size WxH] [--iters N]\n"
                 "       rudra-native master <package> <image> --out <file.exr> [--runtime ...] [--device ...] [--params JSON]\n"
                 "       rudra-native master-check <package> <golden-dir> [--runtime ...] [--device ...]\n"
                 "       rudra-native bench-scopes [--iters N]\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) { usage(); return 64; }
    if (args[0] == "version") {
        std::printf("rudra-native 0.1.0 (model contract %d.x)\n", kSupportedContractMajor);
        return 0;
    }
    if (args[0] == "bench-scopes") {
        int iters = 7;
        if (args.size() == 3 && args[1] == "--iters") iters = std::max(1, std::atoi(args[2].c_str()));
        return cmd_bench_scopes(iters);
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
#ifdef RUDRA_HAVE_STILL_DECODE
    if (args[0] == "master" || args[0] == "master-check") {
        if (args.size() < 3) { usage(); return 64; }
        std::string runtime = "auto", device = "cpu", out, params;
        for (std::size_t i = 3; i + 1 < args.size(); i += 2) {
            if (args[i] == "--runtime") runtime = args[i + 1];
            else if (args[i] == "--device") device = args[i + 1];
            else if (args[i] == "--out") out = args[i + 1];
            else if (args[i] == "--params") params = args[i + 1];
            else { usage(); return 64; }
        }
        auto d = parse_device(device);
        if (!d) return fail(d.error());
        if (args[0] == "master-check") return cmd_master_check(pkg, args[2], runtime, *d);
        if (out.empty()) { usage(); return 64; }
        return cmd_master(pkg, args[2], out, runtime, *d, params);
    }
#endif
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
