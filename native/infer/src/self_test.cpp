#include "rudra/infer/self_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <span>
#include <vector>

#include <nlohmann/json.hpp>

#include "rudra/infer/tiler.hpp"
#include "rudra/platform/npy.hpp"

namespace rudra {
namespace fs = std::filesystem;

namespace {

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
                b.at(c, y, x) = a->data[(static_cast<std::size_t>(y) * std::size_t(w) + std::size_t(x)) * 3 + std::size_t(c)];
    return SdrImage(std::move(b));
}

Result<void> compare(const fs::path& file, std::span<const float> got, const Tolerance& t, SelfTestStat& s) {
    auto a = read_npy(file);
    if (!a) return a.error();
    if (static_cast<std::size_t>(a->size()) != got.size())
        return make_error(ErrorCode::ParityError, "An output has the wrong size.",
                          file.string() + ": expected " + std::to_string(a->size()) + " values, got " +
                              std::to_string(got.size()));
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double ref = a->data[i], d = std::abs(ref - double(got[i]));
        s.max_abs = std::max(s.max_abs, d);
        s.excess = std::max(s.excess, d - (t.atol + t.rtol * std::abs(ref)));
    }
    return {};
}

}  // namespace

bool SelfTestReport::pass() const {
    if (frames == 0) return false;
    return std::all_of(outputs.begin(), outputs.end(), [](const auto& o) { return o.second.pass(); });
}

std::string SelfTestReport::summary() const {
    double worst = 0.0;
    std::string failed;
    for (const auto& [k, s] : outputs) {
        worst = std::max(worst, s.max_abs);
        if (!s.pass()) failed += (failed.empty() ? "" : ", ") + k;
    }
    char buf[96];
    std::snprintf(buf, sizeof buf, "%d golden frames%s, worst |d| %.1e", frames, stitched ? " + stitch" : "", worst);
    return (pass() ? "passed, " : "FAILED (" + failed + "), ") + std::string(buf);
}

std::string tolerance_key(const BackendInfo& info) {
    // CPU LibTorch runs eager's own kernels; LibTorch on a GPU is true fp32
    // with the vendor's summation order; ONNX Runtime is another graph
    // compiler on any device.
    if (info.runtime == Runtime::OnnxRuntime) return "onnx";
    return info.device == Device::Cpu ? "torchscript" : "gpu_fp32";
}

Result<SelfTestReport> self_test(const ModelManifest& m, InferenceBackend& backend,
                                 const std::function<void(int, int)>& progress) {
    std::ifstream in(m.root / m.golden);
    nlohmann::json g;
    try {
        in >> g;
    } catch (const std::exception& e) {
        return make_error(ErrorCode::ParseError, "The golden index could not be read.", e.what());
    }
    const fs::path gdir = (m.root / m.golden).parent_path();
    SelfTestReport r;
    r.backend = backend.info();
    r.tolerance_key = tolerance_key(r.backend);
    const auto tol = m.tolerance.find(r.tolerance_key);
    if (tol == m.tolerance.end())
        return make_error(ErrorCode::Unsupported, "The package has no tolerance for this backend.", r.tolerance_key);
    r.tolerance = tol->second;
    const bool has_stitch = g.contains("stitch") && !g.at("stitch").is_null();
    const int total = int(g.at("frames").size()) + (has_stitch ? 1 : 0);
    try {
        for (const auto& f : g.at("frames")) {
            auto sdr = load_sdr(gdir / f.at("sdr").at("file").get<std::string>());
            if (!sdr) return sdr.error();
            auto out = infer_frame(backend, *sdr, TileConfig{0, 0});
            if (!out) return out.error();
            const float scale = out->scalars.residual_scale, weight = out->scalars.shadow_weight;
            const std::vector<std::pair<std::string, std::span<const float>>> outs{
                {"residual_scale", std::span<const float>(&scale, 1)},
                {"shadow_weight", std::span<const float>(&weight, 1)},
                {"curve_params", out->scalars.curve_params},
                {"residual", out->fields.residual.span()},
                {"highlight", out->fields.highlight.span()},
                {"shadow", out->fields.shadow.span()}};
            for (const auto& [key, span] : outs)
                if (auto c = compare(gdir / f.at(key).at("file").get<std::string>(), span, r.tolerance, r.outputs[key]); !c)
                    return c.error();
            ++r.frames;
            if (progress) progress(r.frames, total);
        }
        if (has_stitch) {
            const auto& s = g.at("stitch");
            auto sdr = load_sdr(gdir / s.at("sdr").at("file").get<std::string>());
            if (!sdr) return sdr.error();
            auto out = infer_frame(backend, *sdr, TileConfig{s.at("tile_size").get<int>(), s.at("overlap").get<int>()});
            if (!out) return out.error();
            for (const auto& [field, buf] : std::vector<std::pair<std::string, const PlanarBuffer*>>{
                     {"residual", &out->fields.residual}, {"highlight", &out->fields.highlight}, {"shadow", &out->fields.shadow}})
                if (auto c = compare(gdir / s.at(field).at("file").get<std::string>(), buf->span(), r.tolerance,
                                     r.outputs["stitched " + field]);
                    !c)
                    return c.error();
            r.stitched = true;
            if (progress) progress(total, total);
        }
    } catch (const std::exception& e) {
        return make_error(ErrorCode::ParseError, "The golden index is malformed.", e.what());
    }
    return r;
}

}  // namespace rudra
