#include "rudra/video/convert.hpp"

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>

#include "rudra/core/hdr10.hpp"
#include "rudra/deliver/video_encode.hpp"
#include "rudra/deliver/video_master.hpp"
#include "rudra/media/png16.hpp"
#include "rudra/media/video_decode.hpp"
#include "rudra/media/video_probe.hpp"
#include "rudra/video/predictor.hpp"
#include "rudra/video/qc.hpp"

namespace rudra {
namespace fs = std::filesystem;

fs::path video_sidecar_path(const fs::path& output) {
    fs::path s = output;
    s += ".json";
    return s;
}

#ifdef RUDRA_HAVE_STILL_DECODE

namespace {

Error refuse(std::string message) { return make_error(ErrorCode::InvalidArgument, std::move(message)); }

std::string lower(std::string s) {
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

fs::path resolved(const fs::path& p) {
    std::error_code ec;
    fs::path r = fs::weakly_canonical(fs::absolute(p, ec), ec);
    return ec ? fs::absolute(p) : r;
}

}  // namespace

Result<VideoConvertResult> convert_video(const VideoConvertArgs& a, const ModelManifest& package,
                                         InferenceBackend& backend, const VideoConvertHooks& hooks) {
    const fs::path source = resolved(a.input), output = resolved(a.output);
    const fs::path sidecar = video_sidecar_path(output);
    const std::string ext = lower(output.extension().string());
    if (ext != ".mp4" && ext != ".mov" && ext != ".mkv") return refuse("Output must be MP4, MOV or MKV");
    const DeliveryProfile* profile = nullptr;
    for (const auto& p : delivery_profiles())
        if (p.name == a.format) profile = &p;
    if (!profile) return refuse("Unknown delivery format: " + a.format);
    if (profile->codec == "prores" && ext != ".mov") return refuse("ProRes delivery requires a .mov output");
    std::error_code ec;
    if (!fs::is_regular_file(source, ec)) return make_error(ErrorCode::NotFound, source.string());
    if (source == output || fs::exists(output, ec) || fs::exists(sidecar, ec))
        return refuse("Refusing to overwrite source/output/sidecar");
    auto ffmpeg = require_executable("ffmpeg");
    if (!ffmpeg) return ffmpeg.error();
    auto encoders = run_tool({ffmpeg->string(), "-hide_banner", "-encoders"});
    if (!encoders) return encoders.error();
    auto filters = run_tool({ffmpeg->string(), "-hide_banner", "-filters"});
    if (!filters) return filters.error();
    if (encoders->find(profile->encoder) == std::string::npos || filters->find("zscale") == std::string::npos)
        return make_error(ErrorCode::Unsupported, "This FFmpeg build needs " + profile->encoder + " and zscale support");
    if (!(0 <= a.shadow_smoothing && a.shadow_smoothing < 1) || !(0 < a.cut_threshold && a.cut_threshold <= 1))
        return refuse("Invalid smoothing/cut threshold");
    if (a.tile_size < 0 || a.tile_overlap < 0 || (a.tile_size && a.tile_overlap >= a.tile_size))
        return refuse("Invalid tile size/overlap");
    if (!(0 <= a.crf && a.crf <= 51) || !(0 <= a.min_nits && a.min_nits < a.peak_nits))
        return refuse("Invalid mastering/encoding settings");
    if (auto m = master_to_pq(PlanarBuffer(3, 1, 1, 0.0f), a.peak_nits, a.knee_nits); !m) return m.error();

    VideoArgs va;
    va.format = a.format;
    va.alpha_mode = a.alpha_mode;
    va.input_transfer = a.input_transfer, va.input_primaries = a.input_primaries;
    va.input_matrix = a.input_matrix, va.input_range = a.input_range;
    auto probe = probe_video(source);
    if (!probe) return probe.error();
    auto video = open_video_source(*probe, va);
    if (!video) return video.error();
    const bool alpha = video->alpha;
    const int width = video->stream.width, height = video->stream.height, total = video->clock.frames;

    fs::create_directories(output.parent_path(), ec);
    const fs::path work_parent = resolved(a.work_dir ? *a.work_dir : output.parent_path());
    fs::create_directories(work_parent, ec);
    const int channels = alpha ? 4 : 3;
    const std::uint64_t frame_bytes = std::uint64_t(width) * height * channels * 2;
    if (spool_space_low(work_parent, frame_bytes)) return make_error(ErrorCode::IoError, "Insufficient working disk space");
    VideoPredictor predictor(backend, ModelConstants{package.log_scale, package.max_hdr, package.corpus_ev}, a.tile_size,
                             a.tile_overlap);
    ShadowSmoother smoother(a.shadow_smoothing, a.cut_threshold);
    const auto started = std::chrono::steady_clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); };

    const auto& c = video->contract;
    const auto& clk = video->clock;
    pyjson::Dict report = {
        {"source", source.string()},
        {"output", output.string()},
        {"input_contract", pyjson::Dict{{"transfer", c.transfer}, {"primaries", c.primaries}, {"matrix", c.matrix},
                                        {"range", c.range}}},
        {"timing", pyjson::Dict{{"fps", clk.fps}, {"frames", clk.frames}, {"start", clk.start},
                                {"duration", clk.duration}, {"tolerance", clk.tolerance}}},
        {"delivery_format", a.format},
        {"alpha_mode", alpha && a.alpha_mode ? pyjson::Value(*a.alpha_mode) : pyjson::Value()},
        {"encoder_input_precision_bits", 10},
        {"alpha_preserved", alpha},
        {"checkpoint", resolved(package.root).string()},
        {"checkpoint_sha256", package.source_sha256},
        {"mastering_peak_nits", a.peak_nits},
        {"mastering_min_nits", a.format == "hdr10" ? pyjson::Value(a.min_nits) : pyjson::Value()},
        {"audio_mode", a.audio},
        {"shadow_smoothing", a.shadow_smoothing},
        {"cut_threshold", a.cut_threshold},
        {"preserve_outside", true},
        {"recovery_mode", "all"},
        {"tile_size", a.tile_size},
        {"tile_overlap", a.tile_overlap},
        {"timestamp_policy", "Video begins at zero; audio retains relative offset, clipped to video interval"},
        {"scene_cuts", pyjson::List{}},
        {"frames", pyjson::List{}},
        {"warnings", a.shadow_smoothing != 0.0 ? pyjson::List{"Scalar shadow smoothing is not validated temporal reconstruction"}
                                        : pyjson::List{}},
    };
    pyjson::List cuts, frames;
    std::vector<VideoFrameLight> lights;

    auto spool_dir = StagingDir::create(work_parent, "rudra-video-");
    if (!spool_dir) return spool_dir.error();
    const fs::path spool = spool_dir->path();
    {
        auto decoder = VideoDecoder::open(source, *video, spool / "decode.log");
        if (!decoder) return decoder.error();
        for (int index = 0; index < total; ++index) {
            if (hooks.cancel && hooks.cancel->load()) return make_error(ErrorCode::Cancelled, "Cancelled");
            auto frame = (*decoder)->next();
            if (!frame) return frame.error();
            auto pred = predictor.predict(rgb_from_frame(*frame), c.transfer, c.primaries, smoother);
            if (!pred) return pred.error();
            auto sf = master_video_frame(pred->hdr, index, a.format, a.peak_nits, a.knee_nits,
                                         alpha ? std::span<const std::uint16_t>(frame->data) : std::span<const std::uint16_t>());
            if (!sf) return sf.error();
            frames.push_back(pyjson::Dict{{"index", index}, {"shadow_weight", pred->shadow_weight},
                                          {"max_cll", sf->light.max_cll}, {"frame_average", sf->light.frame_average}});
            lights.push_back(sf->light);
            if (pred->cut) cuts.push_back(index);
            if (spool_space_low(spool, frame_bytes)) return make_error(ErrorCode::IoError, "Working disk is full");
            if (auto w = write_png16(spool / spool_frame_name(index), sf->packed, width, height, sf->channels); !w)
                return w.error();
            if (index % 10 == 0 || index + 1 == total) {
                if (hooks.progress) hooks.progress({"inference", index + 1, total});
                if (hooks.print) {
                    char line[96];
                    std::snprintf(line, sizeof line, "Frame %d/%d, %.2f fps", index + 1, total,
                                  (index + 1) / std::max(elapsed(), 0.001));
                    hooks.print(line);
                }
            }
        }
        if (auto done = (*decoder)->finish(); !done) return done.error();
    }
    pyjson::set(report, "scene_cuts", cuts);
    pyjson::set(report, "frames", frames);
    const LightCeilings ceil = light_ceilings(lights);

    auto staging = StagingDir::create(output.parent_path(), "rudra-master-");
    if (!staging) return staging.error();
    const fs::path staged = staging->path() / output.filename();
    VideoEncodeRequest er;
    er.format = a.format, er.preset = a.preset, er.crf = a.crf, er.audio = a.audio;
    er.peak_nits = a.peak_nits, er.min_nits = a.min_nits, er.alpha = alpha;
    auto command = encode_command(ffmpeg->string(), er, clk, spool, source, staged, ceil.max_cll, ceil.max_fall);
    if (!command) return command.error();
    pyjson::List cmd;
    for (const auto& s : *command) cmd.push_back(s);
    pyjson::set(report, "encode_command", cmd);
    if (hooks.progress) hooks.progress({"encoding", total, total});
    if (auto ran = run_tool(*command); !ran) return ran.error();
    if (hooks.progress) hooks.progress({"quality_check", total, total});
    VideoQcRequest qr;
    qr.audio_mode = a.audio, qr.format = a.format, qr.alpha = alpha;
    qr.expected = ExpectedLight{double(ceil.max_cll), double(ceil.max_fall), a.peak_nits, a.min_nits};
    auto qc = run_quality_check(video->streams_json, clk, staged, qr);
    if (!qc) return qc.error();
    if (alpha) {
        auto al = run_alpha_check(staged, spool, total, width, height);
        if (!al) return al.error();
        pyjson::set(*std::get<std::shared_ptr<pyjson::Dict>>(qc->v), "alpha", *al);
    }
    pyjson::set(report, "qc", *qc);
    if (a.format == "hlg")
        pyjson::set(report, "hlg_reference",
                    pyjson::Dict{{"display_peak_nits", a.peak_nits}, {"black_nits", 0},
                                 {"system_gamma", 1.2 + 0.42 * std::log10(a.peak_nits / 1000)}});
    pyjson::set(report, "max_cll", ceil.max_cll);
    pyjson::set(report, "max_fall", ceil.max_fall);
    pyjson::set(report, "elapsed_seconds", elapsed());
    const pyjson::Value rep(report);
    const fs::path staged_json = staging->path() / sidecar.filename();
    {
        std::ofstream out(staged_json, std::ios::binary);
        out << pyjson::dumps(rep, 2);
        if (!out) return make_error(ErrorCode::IoError, "Could not write the report", staged_json.string());
    }
    if (auto p = publish_video(staged, staged_json, output, sidecar); !p) return p.error();
    if (hooks.print) hooks.print("QC passed: " + output.string() + "\nReport: " + sidecar.string());
    return VideoConvertResult{output, sidecar, rep};
}

#endif

}  // namespace rudra
