#include "rudra/deliver/video_qc.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include <nlohmann/json.hpp>

#include "rudra/core/hdr10.hpp"

namespace rudra {
namespace fs = std::filesystem;
using ojson = nlohmann::ordered_json;

namespace {

pyjson::Value to_py(const ojson& j) {
    switch (j.type()) {
        case ojson::value_t::null: return nullptr;
        case ojson::value_t::boolean: return j.get<bool>();
        case ojson::value_t::number_integer:
        case ojson::value_t::number_unsigned: return j.get<std::int64_t>();
        case ojson::value_t::number_float: return j.get<double>();
        case ojson::value_t::string: return j.get<std::string>();
        case ojson::value_t::array: {
            pyjson::List l;
            for (const auto& v : j) l.push_back(to_py(v));
            return l;
        }
        case ojson::value_t::object: {
            pyjson::Dict d;
            for (auto it = j.begin(); it != j.end(); ++it) d.emplace_back(it.key(), to_py(it.value()));
            return d;
        }
        default: return nullptr;
    }
}

// str(value) as the Python's f-string prints what .get returned.
std::string py_str(const ojson* v) {
    if (!v || v->is_null()) return "None";
    if (v->is_string()) return v->get<std::string>();
    if (v->is_boolean()) return v->get<bool>() ? "True" : "False";
    if (v->is_number_integer() || v->is_number_unsigned()) return std::to_string(v->get<std::int64_t>());
    if (v->is_number_float()) return pyjson::repr(v->get<double>());
    return v->dump();
}

const ojson* key(const ojson& o, const char* k) {
    const auto it = o.find(k);
    return it == o.end() ? nullptr : &*it;
}

// Python float() of a JSON value (a number or a numeric string); nullopt otherwise.
std::optional<double> num(const ojson* v) {
    if (!v) return std::nullopt;
    if (v->is_number()) return v->get<double>();
    if (!v->is_string()) return std::nullopt;
    const std::string s = v->get<std::string>();
    char* end = nullptr;
    const double d = std::strtod(s.c_str(), &end);
    if (end == s.c_str()) return std::nullopt;
    return d;
}

// Two JSON values equal as Python compares a str or an int with ==.
bool py_equal(const ojson* v, const ojson& want) {
    if (!v) return want.is_null();
    return *v == want;
}

const ojson* first_of_type(const ojson& streams, const char* type) {
    for (const auto& s : streams)
        if (s.value("codec_type", "") == type) return &s;
    return nullptr;
}

}  // namespace

Result<VideoQcFindings> evaluate_video_qc(std::string_view source_streams_json, const VideoClock& clock,
                                          std::string_view output_streams_json, std::string_view output_frames_json,
                                          const VideoQcRequest& r) {
    ojson src, info, frames_doc;
    try {
        src = ojson::parse(source_streams_json);
        info = ojson::parse(output_streams_json);
        frames_doc = ojson::parse(output_frames_json);
    } catch (const ojson::exception& e) {
        return make_error(ErrorCode::ParseError, "ffprobe output is not JSON", e.what());
    }
    const DeliveryProfile* profile = nullptr;
    for (const auto& p : delivery_profiles())
        if (p.name == r.format) profile = &p;
    if (!profile) return make_error(ErrorCode::InvalidArgument, "Unknown delivery format: " + r.format);
    const ojson empty = ojson::array();
    const ojson& src_streams = src.contains("streams") ? src["streams"] : empty;
    const ojson& out_streams = info.contains("streams") ? info["streams"] : empty;
    const ojson& frames = frames_doc.contains("frames") ? frames_doc["frames"] : empty;
    const ojson* src_video = first_of_type(src_streams, "video");
    const ojson* video = first_of_type(out_streams, "video");
    if (!src_video || !video) return make_error(ErrorCode::InvalidArgument, "Export QC failed: no video stream");

    VideoQcFindings f;
    auto& errors = f.errors;
    const bool prores = profile->codec == "prores";
    const std::string pix = r.format == "prores4444" ? (r.alpha ? "yuva444p12le" : "yuv444p12le") : profile->pixel_format;
    const std::vector<std::pair<const char*, ojson>> expected_tags = {
        {"codec_name", profile->codec},   {"pix_fmt", pix},
        {"color_primaries", "bt2020"},    {"color_transfer", profile->transfer},
        {"color_space", "bt2020nc"},      {"color_range", "tv"},
        {"width", src_video->value("width", ojson())}, {"height", src_video->value("height", ojson())}};
    for (const auto& [k, want] : expected_tags) {
        const ojson* got = key(*video, k);
        // MOV's nclc atom carries primaries, transfer and matrix, not a range bit.
        if (std::string(k) == "color_range" && prores && (!got || got->is_null())) continue;
        if (!py_equal(got, want)) errors.push_back(std::string(k) + ": expected " + py_str(&want) + ", got " + py_str(got));
    }
    if (static_cast<int>(frames.size()) != clock.frames) errors.push_back("Video frame count changed");
    auto fps = parse_fraction(clock.fps);
    if (!fps) return fps.error();
    const double period = Fraction{fps->den, fps->num}.value();
    auto tb = parse_fraction(video->value("time_base", "1/1"));
    if (!tb) return tb.error();
    const double tick = tb->value();
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const auto t = num(key(frames[i], "best_effort_timestamp_time"));
        if (!t || std::fabs(*t - static_cast<double>(i) * period) > std::max(2 * tick, 0.00001)) {
            errors.push_back("Output presentation timestamps changed");
            break;
        }
    }
    ojson side = ojson::array();
    if (!frames.empty() && frames[0].contains("side_data_list")) side = frames[0]["side_data_list"];
    auto has_side = [&](const char* type) {
        return std::any_of(side.begin(), side.end(), [&](const ojson& d) { return d.value("side_data_type", "") == type; });
    };
    if (r.format == "hdr10") {
        if (!has_side("Mastering display metadata")) errors.push_back("Missing mastering display SEI");
        if (!has_side("Content light level metadata")) errors.push_back("Missing MaxCLL/MaxFALL SEI");
    }
    if (r.format == "hlg" && (has_side("Mastering display metadata") || has_side("Content light level metadata")))
        errors.push_back("HLG incorrectly contains HDR10 static metadata");
    if (prores && !py_equal(key(*video, "codec_tag_string"), ojson(profile->tag)))
        errors.push_back("Incorrect ProRes profile tag");
    if (r.expected && r.format == "hdr10") {
        for (const auto& d : side) {
            const std::string type = d.value("side_data_type", "");
            if (type == "Content light level metadata") {
                const auto mc = num(key(d, "max_content")), ma = num(key(d, "max_average"));
                if (!mc || !ma || *mc != r.expected->max_cll || *ma != r.expected->max_fall)
                    errors.push_back("Encoded content-light metadata differs from measured values");
            }
            if (type == "Mastering display metadata") {
                for (const auto& [k, want] : {std::pair<const char*, double>{"max_luminance", r.expected->peak},
                                              {"min_luminance", r.expected->minimum}}) {
                    const ojson* v = key(d, k);
                    auto fr = v && v->is_string() ? parse_fraction(v->get<std::string>()) : Result<Fraction>(
                                  make_error(ErrorCode::ParseError, "missing"));
                    if (!fr || std::fabs(fr->value() - want) > 0.0001) errors.push_back(std::string("Wrong mastering ") + k);
                }
            }
        }
    }
    if (const auto d = num(key(*video, "duration")); d && std::fabs(*d - clock.duration) > std::max(2 * tick, 0.00001))
        errors.push_back("Video duration changed");

    std::vector<const ojson*> source_audio, target_audio;
    if (r.audio_mode != "none")
        for (const auto& s : src_streams)
            if (s.value("codec_type", "") == "audio") source_audio.push_back(&s);
    for (const auto& s : out_streams)
        if (s.value("codec_type", "") == "audio") target_audio.push_back(&s);
    if (source_audio.size() != target_audio.size()) errors.push_back("Audio stream count changed");
    for (std::size_t i = 0; i < std::min(source_audio.size(), target_audio.size()); ++i) {
        const ojson& s = *source_audio[i];
        const ojson& d = *target_audio[i];
        for (const char* k : {"channels", "sample_rate"}) {
            const ojson* a = key(s, k);
            const ojson* b = key(d, k);
            if (!(a ? py_equal(b, *a) : (!b || b->is_null()))) errors.push_back(std::string("Audio ") + k + " changed");
        }
        if (r.audio_mode == "copy") {
            const ojson* a = key(s, "codec_name");
            const ojson* b = key(d, "codec_name");
            if (!(a ? py_equal(b, *a) : (!b || b->is_null()))) errors.push_back("Audio codec changed in copy mode");
        }
        const double s_start = num(key(s, "start_time")).value_or(0.0);
        const double d_start = num(key(d, "start_time")).value_or(0.0);
        const double expected_start = std::max(0.0, s_start - clock.start);
        if (std::fabs(d_start - expected_start) > 0.06) errors.push_back("Audio start offset changed");
        if (s.contains("duration") && d.contains("duration")) {
            const double expected_end = std::min(clock.duration, s_start + num(key(s, "duration")).value_or(0.0) - clock.start);
            const double actual_end = d_start + num(key(d, "duration")).value_or(0.0);
            if (std::fabs(actual_end - expected_end) > 0.06) errors.push_back("Audio end offset changed");
        }
    }
    f.video_frames = static_cast<int>(frames.size());
    f.audio_streams = static_cast<int>(target_audio.size());
    f.video = to_py(*video);
    f.side_data = to_py(side);
    f.range_contract = prores ? "limited; ProRes MOV nclc may omit a separate range flag" : "limited, signalled";
    return f;
}

Result<pyjson::Value> qc_verdict(const VideoQcFindings& f) {
    if (!f.errors.empty()) {
        std::string msg = "Export QC failed: ";
        for (std::size_t i = 0; i < f.errors.size(); ++i) msg += (i ? "; " : "") + f.errors[i];
        return make_error(ErrorCode::IntegrityError, msg);
    }
    return pyjson::Value(pyjson::Dict{{"passed", true},
                                      {"video_frames", f.video_frames},
                                      {"audio_streams", f.audio_streams},
                                      {"video", f.video},
                                      {"hdr_side_data", f.side_data},
                                      {"decoded_without_errors", true},
                                      {"range_contract", f.range_contract}});
}

std::vector<std::string> qc_decode_command(const std::string& ffmpeg, const fs::path& output) {
    return {ffmpeg, "-v", "error", "-xerror", "-nostdin", "-i", output.string(), "-map", "0:v:0", "-map", "0:a?",
            "-f", "null", "-"};
}

Result<pyjson::Value> alpha_verdict(int max_error) {
    if (max_error > kAlphaToleranceCodes)
        return make_error(ErrorCode::IntegrityError, "Alpha QC failed: maximum 16-bit code error " +
                                                         std::to_string(max_error) + " exceeds 128");
    return pyjson::Value(pyjson::Dict{{"passed", true},
                                      {"max_error_16bit_codes", max_error},
                                      {"tolerance_16bit_codes", kAlphaToleranceCodes},
                                      {"note", "Source alpha bypasses grading; encoder input quantizes to 10 bits"}});
}

}  // namespace rudra
