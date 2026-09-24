#include "rudra/media/video_probe.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <numeric>

#include <nlohmann/json.hpp>

#include "rudra/platform/process.hpp"

namespace rudra {
namespace fs = std::filesystem;
using nlohmann::json;

namespace {

Error refuse(std::string message) { return make_error(ErrorCode::Unsupported, std::move(message)); }

std::optional<std::string> string_key(const json& j, const char* key) {
    const auto it = j.find(key);
    if (it == j.end() || !it->is_string()) return std::nullopt;
    return it->get<std::string>();
}

// Python float() of a JSON value: a number, or a string holding one.
std::optional<double> to_float(const json& v) {
    if (v.is_number()) return v.get<double>();
    if (!v.is_string()) return std::nullopt;
    const std::string s = v.get<std::string>();
    const char* begin = s.c_str();
    while (*begin == ' ' || *begin == '\t' || *begin == '\n') ++begin;
    char* end = nullptr;
    const double d = std::strtod(begin, &end);   // correctly rounded, as Python's float()
    if (end == begin) return std::nullopt;
    while (*end == ' ' || *end == '\t' || *end == '\n') ++end;
    if (*end) return std::nullopt;
    return d;
}

}  // namespace

Result<VideoProbe> parse_video_probe(std::string_view streams_json, std::string_view frames_json) {
    VideoProbe out;
    out.streams_json = std::string(streams_json);
    out.frames_json = std::string(frames_json);
    json info, frames;
    try {
        info = json::parse(streams_json);
        if (!frames_json.empty()) frames = json::parse(frames_json);
    } catch (const json::exception& e) {
        return make_error(ErrorCode::ParseError, "ffprobe output is not JSON", e.what());
    }
    const auto streams = info.find("streams");
    if (streams == info.end() || !streams->is_array())
        return make_error(ErrorCode::ParseError, "ffprobe output has no streams");
    for (const auto& s : *streams) {
        if (s.value("codec_type", "") != "video") continue;
        VideoStreamInfo v;
        v.index = s.value("index", 0);
        if (!s.contains("width") || !s.contains("height") || !s["width"].is_number_integer() ||
            !s["height"].is_number_integer())
            return make_error(ErrorCode::ParseError, "Video stream without a size");
        v.width = s["width"].get<int>();
        v.height = s["height"].get<int>();
        v.codec_name = s.value("codec_name", "");
        v.pix_fmt = s.value("pix_fmt", "");
        v.color_transfer = string_key(s, "color_transfer");
        v.color_primaries = string_key(s, "color_primaries");
        v.color_space = string_key(s, "color_space");
        v.color_range = string_key(s, "color_range");
        if (auto f = string_key(s, "field_order")) v.field_order = *f;
        if (auto f = string_key(s, "sample_aspect_ratio")) v.sample_aspect_ratio = *f;
        if (auto f = string_key(s, "avg_frame_rate")) v.avg_frame_rate = *f;
        if (auto f = string_key(s, "time_base")) v.time_base = *f;
        // rotation = float(tags.get('rotate', 0)); then each side data's rotation wins.
        if (const auto tags = s.find("tags"); tags != s.end() && tags->contains("rotate")) {
            const auto r = to_float((*tags)["rotate"]);
            if (!r) return make_error(ErrorCode::ParseError, "Unreadable rotate tag");
            v.rotation = *r;
        }
        if (const auto side = s.find("side_data_list"); side != s.end() && side->is_array()) {
            for (const auto& d : *side) {
                if (!d.contains("rotation")) continue;
                const auto r = to_float(d["rotation"]);
                if (!r) return make_error(ErrorCode::ParseError, "Unreadable display rotation");
                v.rotation = *r;
            }
        }
        out.video_streams.push_back(std::move(v));
    }
    if (!frames_json.empty()) {
        const auto list = frames.find("frames");
        if (list != frames.end() && list->is_array()) {
            for (const auto& f : *list) {
                VideoFrameInfo fi;
                if (f.contains("best_effort_timestamp_time")) fi.pts = to_float(f["best_effort_timestamp_time"]);
                fi.width = f.value("width", 0);
                fi.height = f.value("height", 0);
                out.frames.push_back(fi);
            }
        }
    }
    return out;
}

Result<VideoProbe> probe_video(const fs::path& file) {
    auto ffprobe = require_executable("ffprobe");
    if (!ffprobe) return ffprobe.error();
    const std::string exe = ffprobe->string();
    const std::string path = file.string();
    auto streams = run_tool({exe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path});
    if (!streams) return streams.error();
    auto frames = run_tool({exe, "-v", "error", "-show_streams", "-show_format", "-select_streams", "v:0",
                            "-show_frames", "-show_entries",
                            "frame=best_effort_timestamp_time,duration_time,pkt_duration_time,width,height:frame_side_data",
                            "-of", "json", path});
    if (!frames) return frames.error();
    return parse_video_probe(*streams, *frames);
}

bool has_alpha(std::string_view pix_fmt) {
    for (const char* s : {"rgba", "bgra", "argb", "abgr", "yuva", "gbrap", "ya"})
        if (pix_fmt.find(s) != std::string_view::npos) return true;
    return false;
}

Result<InputContract> input_contract(const VideoStreamInfo& stream, const VideoArgs& args) {
    if (stream.color_transfer == "smpte2084" || stream.color_transfer == "arib-std-b67")
        return refuse("Input is already HDR; this command accepts SDR only");
    const std::string& pix = stream.pix_fmt;
    if (has_alpha(pix)) {
        if (args.format != "prores4444") return refuse("Alpha input requires --format prores4444");
        if (args.alpha_mode != "straight")
            return refuse("Declare --alpha-mode straight; premultiplied input must be unpremultiplied first");
    }
    if (stream.field_order != "progressive" && stream.field_order != "unknown")
        return refuse("Interlaced input must be deinterlaced explicitly");
    if (std::fmod(stream.rotation, 360.0) != 0.0) return refuse("Bake the input display rotation before conversion");
    const std::string& sar = stream.sample_aspect_ratio;
    if (sar != "1:1" && sar != "0:1" && sar != "N/A")
        return refuse("Anamorphic input must be converted to square pixels explicitly");
    if (stream.width % 2 || stream.height % 2)
        return refuse("HDR10 4:2:0 requires even dimensions; crop or pad explicitly");

    auto pick = [](const std::optional<std::string>& tag,
                   std::initializer_list<std::pair<const char*, const char*>> map) -> std::string {
        if (!tag) return {};
        for (const auto& [from, to] : map)
            if (*tag == from) return to;
        return {};
    };
    std::string transfer = args.input_transfer;
    if (transfer == "auto")
        transfer = pick(stream.color_transfer, {{"bt709", "rec709"}, {"iec61966-2-1", "srgb"}, {"gamma22", "gamma22"}});
    std::string primaries = args.input_primaries;
    if (primaries == "auto") primaries = pick(stream.color_primaries, {{"bt709", "rec709"}, {"bt2020", "rec2020"}});
    const bool rgb = pix.starts_with("rgb") || pix.starts_with("bgr") || pix.starts_with("gbr");
    std::string matrix = args.input_matrix;
    if (matrix == "auto")
        matrix = rgb ? "gbr" : pick(stream.color_space, {{"bt709", "bt709"}, {"bt2020nc", "bt2020nc"}});
    std::string range = args.input_range;
    if (range == "auto") range = rgb ? "full" : pick(stream.color_range, {{"tv", "limited"}, {"pc", "full"}});
    for (const auto& [name, value] : {std::pair<const char*, const std::string*>{"transfer", &transfer},
                                      {"primaries", &primaries}, {"matrix", &matrix}, {"range", &range}}) {
        if (value->empty())
            return refuse(std::string("Missing/unsupported colour ") + name + "; specify --input-" + name);
    }
    if (rgb && (range != "full" || matrix != "gbr")) return refuse("RGB input requires full range and GBR matrix");
    return InputContract{transfer, primaries, matrix, range};
}

Result<VideoClock> video_timing(const VideoStreamInfo& stream, const std::vector<VideoFrameInfo>& frames) {
    // Fraction('0/0') raises in Python; here it is the same refusal as a zero rate.
    const auto fps = parse_fraction(stream.avg_frame_rate);
    if (!fps || fps->num <= 0 || frames.empty()) return refuse("No valid video frame rate/timestamps");
    std::vector<double> pts;
    pts.reserve(frames.size());
    for (const auto& f : frames) {
        if (!f.pts || !std::isfinite(*f.pts)) return refuse("Invalid video timestamps");
        pts.push_back(*f.pts);
    }
    const auto tb = parse_fraction(stream.time_base);
    if (!tb) return make_error(ErrorCode::ParseError, "Invalid video time base");
    const double tick = tb->value();
    const double tolerance = std::max(tick * 1.5, 0.000002);
    const double period = Fraction{fps->den, fps->num}.value();   // float(1 / fps)
    for (std::size_t i = 0; i < pts.size(); ++i) {
        if (std::fabs((pts[i] - pts[0]) - static_cast<double>(i) * period) > tolerance)
            return refuse("Variable frame rate or discontinuous timestamps: normalize explicitly before conversion");
    }
    return VideoClock{fps->str(), static_cast<int>(pts.size()), pts[0], static_cast<double>(pts.size()) * period,
                      tolerance};
}

std::string decoder_filter(const InputContract& contract, bool alpha) {
    const std::string matrix = contract.matrix == "bt709"      ? "709"
                               : contract.matrix == "bt2020nc" ? "2020_ncl"
                                                               : "gbr";
    return "zscale=matrixin=" + matrix + ":rangein=" + contract.range + ":matrix=gbr:range=full," +
           (alpha ? "format=gbrap16le,format=rgba64le" : "format=gbrp16le,format=rgb48le");
}

Result<VideoSource> open_video_source(const VideoProbe& probe, const VideoArgs& args) {
    if (probe.video_streams.size() != 1) return refuse("Select a source containing exactly one video stream");
    const VideoStreamInfo& stream = probe.video_streams.front();
    auto contract = input_contract(stream, args);
    if (!contract) return contract.error();
    auto clock = video_timing(stream, probe.frames);
    if (!clock) return clock.error();
    for (const auto& f : probe.frames)
        if (f.width != stream.width || f.height != stream.height)
            return refuse("Changing frame dimensions are unsupported");
    return VideoSource{stream, *contract, *clock, has_alpha(stream.pix_fmt), probe.streams_json};
}

Result<VideoSource> open_video_source(const fs::path& file, const VideoArgs& args) {
    auto probe = probe_video(file);
    if (!probe) return probe.error();
    return open_video_source(*probe, args);
}

}  // namespace rudra
