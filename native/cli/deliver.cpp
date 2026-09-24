#include "deliver.hpp"

#include <cstdio>
#include <cstdlib>

#include "rudra/core/measure.hpp"
#include "rudra/deliver/sequence_encode.hpp"
#include "rudra/platform/pyjson.hpp"

namespace rudra {
namespace fs = std::filesystem;

// rudra/delivery/cli.py _cmd_deliver: MaxCLL and MaxFALL measured from the
// frames after the roll-off into the peak, then the frames encoded as they are.
int cmd_deliver(const std::vector<std::string>& args) {
    auto usage = [](const std::string& m) {
        std::fprintf(stderr, "rudra-native deliver: error: %s\n", m.c_str());
        return 64;
    };
    if (args.empty()) return usage("the following arguments are required: input");
    fs::path input = args[0], output;
    SequenceEncodeOptions o;
    o.source = Primaries::Rec709;   // `deliver`'s default --source-space
    double nits_scale = 203.0;
    std::string space = "rec709";
    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& k = args[i];
        if (k == "--no-verify-tags") { o.verify_tags = false; continue; }
        if (i + 1 >= args.size()) return usage("argument " + k + ": expected one argument");
        const std::string v = args[++i];
        char* end = nullptr;
        const double d = std::strtod(v.c_str(), &end);
        const bool num = end && *end == '\0' && !v.empty();
        if (k == "--output") output = v;
        else if (k == "--target") {
            if (!find_sequence_target(v)) return usage("invalid --target: " + v);
            o.target = v;
        } else if (k == "--fps" && num) o.fps = d;
        else if (k == "--peak-nits" && num) o.peak_nits = d;
        else if (k == "--min-nits" && num) o.min_nits = d;
        else if (k == "--nits-scale" && num) nits_scale = d;
        else if (k == "--source-space") {
            if (v == "rec2020") o.source = Primaries::Rec2020;
            else if (v == "rec709") o.source = Primaries::Rec709;
            else if (v == "p3d65") o.source = Primaries::P3D65;
            else return usage("invalid --source-space: " + v);
            space = v;
        } else return usage("unrecognized or invalid argument: " + k + " " + v);
    }
    if (output.empty()) return usage("the following arguments are required: --output");
    auto fail = [](const Error& e) {
        std::fprintf(stderr, "error: %s\n", e.message.c_str());
        return 2;
    };
    auto paths = list_linear_frames(input);
    if (!paths) return fail(paths.error());

    std::vector<FrameStats> stats;
    for (std::size_t i = 0; i < paths->size(); ++i) {
        auto f = load_linear_frame((*paths)[i], nits_scale);
        if (!f) return fail(f.error());
        auto m = shoulder_to_peak(*f, o.peak_nits);
        if (!m) return fail(m.error());
        stats.push_back(analyze_frame(*m, int(i)));
    }
    const StaticMetadata sm = maxcll_maxfall(stats);
    o.maxcll = sm.maxcll;
    o.maxfall = sm.maxfall;
    o.shoulder = false;
    o.note = [](const std::string& n) { std::fprintf(stderr, "%s\n", n.c_str()); };
    std::size_t next_index = 0;
    Error load_error{};
    bool load_failed = false;
    auto next = [&]() -> std::optional<NitsFrame> {
        if (next_index >= paths->size()) return std::nullopt;
        auto f = load_linear_frame((*paths)[next_index++], nits_scale);
        if (f) f = shoulder_to_peak(*f, o.peak_nits);
        if (!f) {
            load_error = f.error();
            load_failed = true;
            return std::nullopt;
        }
        return std::move(*f);
    };
    auto out = encode_sequence(next, output, o);
    if (load_failed) return fail(load_error);
    if (!out) return fail(out.error());

    auto tags_value = [](const std::optional<ColourTags>& t) -> pyjson::Value {
        if (!t) return nullptr;
        pyjson::Dict d;
        for (const auto& [k, v] : *t) d.emplace_back(k, v);
        return d;
    };
    auto tags = colour_tags(*out);
    const SequenceTarget* t = find_sequence_target(o.target);
    const pyjson::Value report = pyjson::Dict{
        {"file", out->string()},
        {"frames", int(paths->size())},
        {"target", o.target},
        {"fps", o.fps},
        {"peak_nits", o.peak_nits},
        {"maxcll", sm.maxcll},
        {"maxfall", sm.maxfall},
        {"colour_tags", tags ? tags_value(*tags) : pyjson::Value()},
        {"prores_frame_tags", tags_value(prores_frame_tags(*out))},
        {"tags_verified", o.verify_tags},
        {"note", t->note},
    };
    std::printf("%s\n", pyjson::dumps(report, 2).c_str());
    return 0;
}

}  // namespace rudra
