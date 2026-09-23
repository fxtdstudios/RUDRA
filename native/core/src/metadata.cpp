#include "rudra/core/metadata.hpp"

#include <algorithm>
#include <cmath>

#include "rudra/core/hdr10.hpp"
#include "rudra/core/numeric.hpp"

namespace rudra {
using pyjson::Dict;
using pyjson::List;
using pyjson::Value;

namespace {

template <class F>
std::vector<double> gather(std::span<const FrameStats> w, F f) {
    std::vector<double> v;
    v.reserve(w.size());
    for (const auto& s : w) v.push_back(f(s));
    return v;
}

double vmin(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }
double vmax(const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); }

// Python round() to an int, half to even.
std::int64_t py_round(double x) { return static_cast<std::int64_t>(std::nearbyint(x)); }

}  // namespace

std::vector<Shot> detect_shots(std::span<const FrameStats> stats, double threshold) {
    if (stats.empty()) return {};
    std::vector<int> cuts{0};
    for (std::size_t i = 1; i < stats.size(); ++i) {
        const auto& a = stats[i - 1].log_hist;
        const auto& b = stats[i].log_hist;
        std::vector<double> terms(a.size());
        for (std::size_t k = 0; k < a.size(); ++k) {
            const double d = a[k] - b[k];
            terms[k] = (d * d) / ((a[k] + b[k]) + 1e-9);
        }
        const double dist = 0.5 * np_pairwise_sum(terms);
        if (dist > threshold) cuts.push_back(stats[i].index - stats[0].index);
    }
    cuts.push_back(static_cast<int>(stats.size()));
    std::vector<Shot> shots;
    for (std::size_t i = 0; i + 1 < cuts.size(); ++i) shots.emplace_back(cuts[i], cuts[i + 1] - cuts[i]);
    return shots;
}

Value l1_per_shot(std::span<const FrameStats> stats, std::span<const Shot> shots) {
    List out;
    for (const auto& [start, length] : shots) {
        const auto w = stats.subspan(std::size_t(start), std::size_t(length));
        const auto mins = gather(w, [](const FrameStats& s) { return s.min_nits; });
        const auto avgs = gather(w, [](const FrameStats& s) { return s.avg_nits; });
        const auto maxs = gather(w, [](const FrameStats& s) { return s.max_nits; });
        const double avg = np_mean(avgs);
        out.push_back(Dict{{"start", start},
                           {"duration", length},
                           {"min_pq", pq12(vmin(mins))},
                           {"avg_pq", pq12(avg)},
                           {"max_pq", pq12(vmax(maxs))},
                           {"min_nits", vmin(mins)},
                           {"avg_nits", avg},
                           {"max_nits", vmax(maxs)}});
    }
    return out;
}

Value to_dovi_generate_json(std::span<const FrameStats> stats, std::span<const Shot> shots, double peak, double min_nits) {
    const StaticMetadata md = maxcll_maxfall(stats);
    const Value l1 = l1_per_shot(stats, shots);
    List blocks;
    for (const auto& shot : *std::get<std::shared_ptr<List>>(l1.v)) {
        const auto& d = *std::get<std::shared_ptr<Dict>>(shot.v);
        auto get = [&](const char* k) {
            for (const auto& [key, v] : d)
                if (key == k) return v;
            return Value{};
        };
        blocks.push_back(Dict{{"start", get("start")},
                              {"duration", get("duration")},
                              {"metadata_blocks", List{Dict{{"Level", 1},
                                                            {"min_pq", get("min_pq")},
                                                            {"max_pq", get("max_pq")},
                                                            {"avg_pq", get("avg_pq")}}}}});
    }
    return Dict{{"cm_version", "V40"},
                {"profile", "8.1"},
                {"length", static_cast<std::int64_t>(stats.size())},
                {"level6", Dict{{"max_display_mastering_luminance", py_round(peak)},
                                {"min_display_mastering_luminance", py_round(min_nits * 10000)},
                                {"max_content_light_level", md.maxcll},
                                {"max_frame_average_light_level", md.maxfall}}},
                {"shots", std::move(blocks)}};
}

Value to_hdr10plus_json(std::span<const FrameStats> stats, std::span<const Shot> shots, int target) {
    List scenes, firsts, counts;
    int scene_id = 0;
    for (const auto& [start, length] : shots) {
        const auto w = stats.subspan(std::size_t(start), std::size_t(length));
        List maxscl, index, values;
        for (int c = 0; c < 3; ++c)
            maxscl.push_back(vmax(gather(w, [c](const FrameStats& s) { return s.maxscl_nits[std::size_t(c)]; })));
        for (std::size_t i = 0; i < kStatPercentiles.size(); ++i) {
            index.push_back(kStatPercentiles[i]);
            values.push_back(vmax(gather(w, [i](const FrameStats& s) { return s.percentiles_nits[i]; })));
        }
        scenes.push_back(Dict{
            {"SceneId", scene_id++},
            {"SceneFirstFrameIndex", start},
            {"SceneFrameNumbers", length},
            {"TargetedSystemDisplayMaximumLuminance", target},
            {"LuminanceParameters",
             Dict{{"Units", "cd/m2"},
                  {"AverageMaxRGB", np_mean(gather(w, [](const FrameStats& s) { return s.avg_nits; }))},
                  {"MaxSCL", std::move(maxscl)},
                  {"LuminanceDistributions",
                   Dict{{"DistributionIndex", std::move(index)}, {"DistributionValues", std::move(values)}}}}}});
        firsts.push_back(start);
        counts.push_back(length);
    }
    return Dict{{"JSONInfo", Dict{{"Generator", "RUDRA delivery"}, {"Version", "1.0"}, {"Units", "cd/m2"}}},
                {"SceneInfoSummary", Dict{{"SceneFirstFrameIndex", std::move(firsts)},
                                          {"SceneFrameNumbers", std::move(counts)}}},
                {"SceneInfo", std::move(scenes)}};
}

Value to_rudra_sidecar(std::span<const FrameStats> stats, std::span<const Shot> shots) {
    const StaticMetadata md = maxcll_maxfall(stats);
    List frames;
    for (const auto& s : stats) {
        Dict pct;
        for (std::size_t i = 0; i < kStatPercentiles.size(); ++i)
            pct.emplace_back(pyjson::repr(kStatPercentiles[i]), s.percentiles_nits[i]);
        frames.push_back(Dict{{"index", s.index},
                              {"min_nits", s.min_nits},
                              {"avg_nits", s.avg_nits},
                              {"max_nits", s.max_nits},
                              {"min_pq", pq12(s.min_nits)},
                              {"avg_pq", pq12(s.avg_nits)},
                              {"max_pq", pq12(s.max_nits)},
                              {"maxscl_nits", List{s.maxscl_nits[0], s.maxscl_nits[1], s.maxscl_nits[2]}},
                              {"percentiles", std::move(pct)}});
    }
    return Dict{{"generator", "rudra.delivery.metadata"},
                {"convention", "MaxRGB per pixel; nits absolute; pq codes 12-bit (0-4095)"},
                {"frame_count", static_cast<std::int64_t>(stats.size())},
                {"max_cll_nits", md.maxcll},
                {"max_fall_nits", md.maxfall},
                {"shots", l1_per_shot(stats, shots)},
                {"frames", std::move(frames)}};
}

}  // namespace rudra
