#include "rudra/deliver/sequence_encode.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <sstream>

#include <nlohmann/json.hpp>

#include "rudra/core/gamut.hpp"
#include "rudra/core/hdr10.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/platform/npy.hpp"
#include "rudra/platform/process.hpp"
#include "rudra/platform/pyjson.hpp"

namespace rudra {
namespace fs = std::filesystem;

namespace {

constexpr double kWhiteNits = 203.0;   // DIFFUSE_WHITE_NITS
const char* kPrimaries = "bt2020";
const char* kMatrix = "bt2020nc";
const char* kSwsMatrix = "bt2020";

std::string transfer_name(const std::string& t) {
    if (t == "pq") return "smpte2084";
    if (t == "hlg") return "arib-std-b67";
    return "unknown";
}

Error encode_error(std::string message) { return make_error(ErrorCode::IoError, std::move(message)); }

std::string py_repr_str(const std::string& s) { return "'" + s + "'"; }

long long py_round(double v) { return static_cast<long long>(std::nearbyint(v)); }

double hlg_oetf64(double x) {
    const double a = 0.17883277, b = 0.28466892, c = 0.55991073;
    x = std::max(x, 0.0);
    return x <= 1.0 / 12.0 ? std::sqrt(std::max(3.0 * x, 0.0)) : a * std::log(std::max(12.0 * x - b, 1e-12)) + c;
}

std::string first_line(const std::string& s) {
    const auto e = s.find_first_of("\r\n");
    std::string l = s.substr(0, e);
    while (!l.empty() && (l.back() == ' ' || l.back() == '\t')) l.pop_back();
    return l;
}

std::string ffmpeg_version() {
    auto r = run_process({"ffmpeg", "-version"});
    if (!r || r->out.empty()) return "unknown build";
    std::string l = first_line(r->out);
    while (!l.empty() && l.front() == ' ') l.erase(l.begin());
    return l;
}

std::vector<std::string> lines_of(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream in(text);
    while (std::getline(in, cur)) {
        if (!cur.empty() && cur.back() == '\r') cur.pop_back();
        out.push_back(cur);
    }
    return out;
}

std::string strip(std::string s) {
    const char* ws = " \t\r\n";
    s.erase(0, s.find_first_not_of(ws));
    const auto e = s.find_last_not_of(ws);
    s.erase(e == std::string::npos ? 0 : e + 1);
    return s;
}

std::string mismatch(const ColourTags& want, const ColourTags& got) {
    std::vector<std::pair<std::string, std::pair<std::string, std::string>>> wrong;
    for (const auto& [k, v] : want) {
        // Captured by name, not as the binding: Apple clang 15 cannot capture a
        // structured binding in a lambda (C++20 allows it, from clang 16).
        const std::string& key = k;
        const bool has = std::any_of(got.begin(), got.end(), [&key](const auto& p) { return p.first == key; });
        const std::string g = has ? tag_of(got, k) : "absent";
        if (!has || g != v) wrong.push_back({k, {v, g}});
    }
    std::sort(wrong.begin(), wrong.end());
    std::string out;
    for (std::size_t i = 0; i < wrong.size(); ++i)
        out += (i ? ", " : "") + wrong[i].first + ": wanted " + wrong[i].second.first + ", file says " + wrong[i].second.second;
    return out;
}

std::vector<std::uint8_t> file_bytes(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::size_t find4(const std::vector<std::uint8_t>& d, const char* tag) {
    const auto it = std::search(d.begin(), d.end(), tag, tag + 4);
    return it == d.end() ? std::string::npos : std::size_t(it - d.begin());
}

unsigned be16(const std::vector<std::uint8_t>& d, std::size_t at) { return (unsigned(d[at]) << 8) | d[at + 1]; }

std::string name_of(const std::map<int, const char*>& m, int v) {
    const auto it = m.find(v);
    return it == m.end() ? std::to_string(v) : it->second;
}

const std::map<int, const char*> kProPrimaries{{0, "unspecified"}, {1, "bt709"},     {2, "unspecified"},
                                               {5, "bt470bg"},     {6, "smpte170m"}, {9, "bt2020"},
                                               {11, "smpte431"},   {12, "smpte432"}};
const std::map<int, const char*> kProTransfer{{0, "unspecified"}, {1, "bt709"}, {2, "unspecified"},
                                              {16, "smpte2084"},  {18, "arib-std-b67"}};
const std::map<int, const char*> kProMatrix{{0, "unspecified"}, {1, "bt709"}, {2, "unspecified"},
                                            {6, "smpte170m"},   {9, "bt2020nc"}};

}  // namespace

std::string tag_of(const ColourTags& tags, const std::string& key) {
    for (const auto& [k, v] : tags)
        if (k == key) return v;
    return {};
}

const std::vector<SequenceTarget>& sequence_targets() {
    static const std::vector<SequenceTarget> t = {
        {"prores4444", {"-c:v", "prores_ks", "-profile:v", "4444", "-qscale:v", "5"}, "yuv444p12le", ".mov", "pq",
         "12-bit ProRes 4444, Rec.2020 PQ. The finishing format."},
        {"prores422hq", {"-c:v", "prores_ks", "-profile:v", "3", "-qscale:v", "9"}, "yuv422p10le", ".mov", "pq",
         "10-bit ProRes 422 HQ, Rec.2020 PQ. The review format."},
        {"hdr10", {"-c:v", "libx265", "-preset", "slow", "-crf", "12"}, "yuv420p10le", ".mp4", "pq",
         "HEVC Main10 PQ with static HDR10 metadata. The delivery format."},
        {"hlg", {"-c:v", "libx265", "-preset", "slow", "-crf", "12"}, "yuv420p10le", ".mp4", "hlg",
         "HEVC Main10 hybrid log-gamma, for broadcast paths that refuse PQ."},
    };
    return t;
}

const SequenceTarget* find_sequence_target(const std::string& name) {
    for (const auto& t : sequence_targets())
        if (t.name == name) return &t;
    return nullptr;
}

Result<ColourTags> expected_tags(const std::string& target) {
    const SequenceTarget* t = find_sequence_target(target);
    if (!t) return encode_error("unknown target " + py_repr_str(target));
    return ColourTags{{"color_primaries", kPrimaries}, {"color_space", kMatrix}, {"color_transfer", transfer_name(t->transfer)}};
}

Result<NitsFrame> shoulder_to_peak(const NitsFrame& rgb, double peak) {
    PlanarBuffer f(3, rgb.height(), rgb.width());
    for (int c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < rgb.plane_size(); ++i) f.plane(c)[i] = static_cast<float>(rgb.plane(c)[i] / 10000.0);
    auto m = master_to_peak(f, peak);
    if (!m) return m.error();
    NitsFrame out(rgb.height(), rgb.width());
    for (int c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < rgb.plane_size(); ++i) out.plane(c)[i] = m->plane(c)[i];
    return out;
}

Result<std::vector<std::uint16_t>> encode_sequence_frame(const NitsFrame& in, const SequenceTarget& target, double peak,
                                                         Primaries source, bool shoulder) {
    NitsFrame rgb = in;
    const std::size_t n = rgb.plane_size();
    if (source != Primaries::Rec2020) {
        PlanarBuffer f(3, rgb.height(), rgb.width());
        for (int c = 0; c < 3; ++c)
            for (std::size_t i = 0; i < n; ++i) f.plane(c)[i] = static_cast<float>(rgb.plane(c)[i] / kWhiteNits);
        const PlanarBuffer conv = convert_primaries(f, source, Primaries::Rec2020);
        for (int c = 0; c < 3; ++c)
            for (std::size_t i = 0; i < n; ++i) rgb.plane(c)[i] = double(conv.plane(c)[i]) * kWhiteNits;
    }
    if (shoulder) {
        auto s = shoulder_to_peak(rgb, peak);
        if (!s) return s.error();
        rgb = std::move(*s);
    }
    std::vector<std::uint16_t> out(n * 3);
    if (target.transfer == "pq") {
        for (int c = 0; c < 3; ++c)
            for (std::size_t i = 0; i < n; ++i) {
                const float coded = pq_oetf(static_cast<float>(std::clamp(rgb.plane(c)[i], 0.0, 10000.0)));
                out[i * 3 + std::size_t(c)] = static_cast<std::uint16_t>(std::clamp(coded, 0.0f, 1.0f) * 65535.0f + 0.5f);
            }
    } else if (target.transfer == "hlg") {
        // hlg_inverse_ootf in float64 on luminance, then the OETF.
        const double gamma = 1.2 + 0.42 * std::log2(std::max(peak, 1e-6) / 1000.0);
        const double pk = std::max(peak, 1e-6);
        for (std::size_t i = 0; i < n; ++i) {
            double d[3];
            for (int c = 0; c < 3; ++c) d[c] = std::max(rgb.plane(c)[i] / pk, 0.0);
            const double yd = d[0] * 0.2627 + d[1] * 0.6780 + d[2] * 0.0593;
            const double ys = std::pow(std::max(yd, 1e-12), 1.0 / gamma);
            const double ratio = ys / std::max(yd, 1e-12);
            for (int c = 0; c < 3; ++c) {
                const double coded = hlg_oetf64(d[c] * ratio);
                out[i * 3 + std::size_t(c)] = static_cast<std::uint16_t>(std::clamp(coded, 0.0, 1.0) * 65535.0 + 0.5);
            }
        }
    } else {
        for (int c = 0; c < 3; ++c)
            for (std::size_t i = 0; i < n; ++i) {
                const double coded = std::clamp(rgb.plane(c)[i] / std::max(peak, 1e-6), 0.0, 1.0);
                out[i * 3 + std::size_t(c)] = static_cast<std::uint16_t>(std::clamp(coded, 0.0, 1.0) * 65535.0 + 0.5);
            }
    }
    return out;
}

std::vector<std::string> sequence_encode_command(const SequenceTarget& t, int width, int height, double fps, double peak,
                                                 std::optional<int> maxcll, std::optional<int> maxfall, double min_nits,
                                                 const fs::path& output, bool write_colr, bool prores_metadata) {
    std::vector<std::string> a = {"ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
                                  "-s", std::to_string(width) + "x" + std::to_string(height), "-r", pyjson::repr(fps),
                                  "-i", "-", "-vf", std::string("scale=out_color_matrix=") + kSwsMatrix + ":out_range=tv"};
    a.insert(a.end(), t.codec.begin(), t.codec.end());
    a.insert(a.end(), {"-pix_fmt", t.pix_fmt, "-color_primaries", kPrimaries, "-colorspace", kMatrix, "-color_trc",
                       transfer_name(t.transfer), "-color_range", "tv"});
    const bool prores = std::find(t.codec.begin(), t.codec.end(), "prores_ks") != t.codec.end();
    const bool x265 = std::find(t.codec.begin(), t.codec.end(), "libx265") != t.codec.end();
    if (t.suffix == ".mov" && write_colr) a.insert(a.end(), {"-movflags", "+write_colr"});
    if (prores && prores_metadata)
        a.insert(a.end(), {"-bsf:v", std::string("prores_metadata=color_primaries=") + kPrimaries +
                                         ":color_trc=" + transfer_name(t.transfer) + ":colorspace=" + kMatrix});
    if (x265) {
        std::string params = std::string("repeat-headers=1:colorprim=") + kPrimaries + ":transfer=" +
                             transfer_name(t.transfer) + ":colormatrix=" + kMatrix + ":range=limited";
        if (t.transfer == "pq") {
            params += ":hdr10-opt=1:master-display=G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(" +
                      std::to_string(py_round(peak * 10000)) + "," + std::to_string(py_round(min_nits * 10000)) + ")";
            if (maxcll && maxfall) params += ":max-cll=" + std::to_string(*maxcll) + "," + std::to_string(*maxfall);
        }
        a.insert(a.end(), {"-x265-params", params});
    }
    a.push_back(output.string());
    return a;
}

bool ffmpeg_supports(const std::string& kind, const std::string& name, const std::string& flag) {
    // Never destroyed, for the same reason as media's video registry: a
    // worker can still ask while the process exits.
    static std::mutex& mu = *new std::mutex;
    static std::map<std::string, bool>& cache = *new std::map<std::string, bool>;
    const std::string key = kind + "=" + name + ":" + flag;
    {
        std::lock_guard lk(mu);
        if (auto it = cache.find(key); it != cache.end()) return it->second;
    }
    bool ok = false;
    if (find_executable("ffmpeg")) {
        auto r = run_process({"ffmpeg", "-hide_banner", "-h", kind + "=" + name});
        ok = r && (r->out + r->err).find(flag) != std::string::npos;
    }
    std::lock_guard lk(mu);
    cache[key] = ok;
    return ok;
}

std::optional<ColourTags> prores_frame_tags(const fs::path& path) {
    const auto d = file_bytes(path);
    const std::size_t marker = find4(d, "icpf");
    if (marker == std::string::npos) return std::nullopt;
    const std::size_t head = marker + 4;
    if (d.size() < head + 20) return std::nullopt;
    const unsigned size = be16(d, head), width = be16(d, head + 8), height = be16(d, head + 10);
    if (size < 20 || width == 0 || height == 0) return std::nullopt;
    return ColourTags{{"color_primaries", name_of(kProPrimaries, d[head + 14])},
                      {"color_transfer", name_of(kProTransfer, d[head + 15])},
                      {"color_space", name_of(kProMatrix, d[head + 16])}};
}

std::optional<ColourTags> container_colr(const fs::path& path) {
    const auto d = file_bytes(path);
    const std::size_t at = find4(d, "colr");
    if (at == std::string::npos || d.size() < at + 14) return std::nullopt;
    const std::string subtype(d.begin() + std::ptrdiff_t(at + 4), d.begin() + std::ptrdiff_t(at + 8));
    if (subtype != "nclc" && subtype != "nclx" && subtype != "prof" && subtype != "rICC") return std::nullopt;
    if (subtype == "prof" || subtype == "rICC") return ColourTags{{"subtype", subtype}};
    return ColourTags{{"subtype", subtype},
                      {"color_primaries", name_of(kProPrimaries, int(be16(d, at + 8)))},
                      {"color_transfer", name_of(kProTransfer, int(be16(d, at + 10)))},
                      {"color_space", name_of(kProMatrix, int(be16(d, at + 12)))}};
}

Result<ColourTags> colour_tags(const fs::path& path) {
    const std::vector<std::string> keys = {"color_primaries", "color_transfer", "color_space"};
    if (!find_executable("ffprobe"))
        return encode_error("ffprobe is not on PATH, so the colour tags of " + path.filename().string() +
                            " cannot be checked. Install ffmpeg (it ships ffprobe) or pass verify_tags=False.");
    const std::vector<std::string> base = {"ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                           "stream=color_primaries,color_transfer,color_space", "-of", "json",
                                           path.string()};
    std::vector<std::string> with_optional = base;
    with_optional.insert(with_optional.begin() + 1, {"-show_optional_fields", "always"});
    std::string last_err;
    for (const std::vector<std::string>* args : {static_cast<const std::vector<std::string>*>(&with_optional), &base}) {
        auto done = run_process(*args);
        if (!done) return done.error();
        last_err = done->err;
        if (done->exit_code != 0) continue;
        try {
            const auto j = nlohmann::json::parse(done->out);
            nlohmann::json stream = nlohmann::json::object();
            if (j.contains("streams") && j["streams"].is_array() && !j["streams"].empty()) stream = j["streams"][0];
            ColourTags out;
            for (const auto& k : keys) {
                std::string v = "unknown";
                if (stream.contains(k)) v = stream[k].is_string() ? stream[k].get<std::string>() : stream[k].dump();
                out.emplace_back(k, v);
            }
            return out;
        } catch (const nlohmann::json::exception&) {
            continue;
        }
    }
    const auto ls = lines_of(strip(last_err));
    return encode_error("ffprobe could not read " + path.filename().string() + ": " + (ls.empty() ? "no output" : ls.back()));
}

Result<void> verify_sequence_tags(const fs::path& path, const std::string& target, std::string* note) {
    auto want = expected_tags(target);
    if (!want) return want.error();
    const SequenceTarget* t = find_sequence_target(target);
    const std::string name = path.filename().string();
    if (std::find(t->codec.begin(), t->codec.end(), "prores_ks") != t->codec.end()) {
        const auto frame = prores_frame_tags(path);
        if (!frame)
            return encode_error(name + " has no readable ProRes frame header, so there is no way to confirm what colour it "
                                       "claims to be. Not delivered. Build: " + ffmpeg_version() + ".");
        const std::string bad = mismatch(*want, *frame);
        if (!bad.empty())
            return encode_error(name + " encoded, but its ProRes frame headers do not say what they should (" + bad +
                                "). Resolve and FCP read those headers, so this file would come up as the wrong colour "
                                "with nothing reporting it. The file is still at " + path.string() + ". Build: " +
                                ffmpeg_version() + ". Pass --no-verify-tags to accept it.");
        auto shown = colour_tags(path);
        if (!shown) return shown.error();
        if (!mismatch(*want, *shown).empty() && note) {
            std::string dict = "{";
            for (std::size_t i = 0; i < shown->size(); ++i)
                dict += (i ? ", '" : "'") + (*shown)[i].first + "': '" + (*shown)[i].second + "'";
            dict += "}";
            *note = "note: " + name + " is correctly tagged in its ProRes frame headers, but this ffmpeg writes an "
                    "incomplete colr atom, so ffprobe reports " + dict + ". Tools that read the frame headers (Resolve, "
                    "FCP) are unaffected; anything reading only the container may assume Rec.709.";
        }
        return {};
    }
    auto got = colour_tags(path);
    if (!got) return got.error();
    const std::string bad = mismatch(*want, *got);
    if (!bad.empty())
        return encode_error(name + " encoded, but its colour tags did not land (" + bad +
                            "). An untagged Rec.2020 file plays as Rec.709 SDR everywhere, with no error shown, so it "
                            "is not delivered. The file is still at " + path.string() +
                            " if you want to look at it. This is an ffmpeg build difference, not something in the "
                            "pixels: " + ffmpeg_version() + ". Pass --no-verify-tags (or verify_tags=False) to accept the "
                            "file as it is.");
    return {};
}

Result<fs::path> encode_sequence(const std::function<std::optional<NitsFrame>()>& next, const fs::path& output_in,
                                 const SequenceEncodeOptions& o) {
    const SequenceTarget* spec = find_sequence_target(o.target);
    if (!spec) return encode_error("unknown target " + py_repr_str(o.target) + ". Choose from: hdr10, hlg, prores422hq, prores4444");
    std::optional<NitsFrame> frame = next();
    if (!frame) return encode_error("no frames to encode");
    const int height = frame->height(), width = frame->width();
    if (!find_executable("ffmpeg"))
        return encode_error("ffmpeg is not on PATH, so nothing can be encoded. Install it (winget install Gyan.FFmpeg) and try again.");
    fs::path output = output_in;
    output.replace_extension(spec->suffix);
    std::error_code ec;
    if (output.has_parent_path()) fs::create_directories(output.parent_path(), ec);
    const bool prores = spec->suffix == ".mov";
    const auto args = sequence_encode_command(*spec, width, height, o.fps, o.peak_nits, o.maxcll, o.maxfall, o.min_nits,
                                              output, prores && ffmpeg_supports("muxer", "mov", "write_colr"),
                                              prores && ffmpeg_supports("bsf", "prores_metadata", "color_primaries"));
    const fs::path log = fs::temp_directory_path() / ("rudra-encode-" + std::to_string(std::hash<std::string>{}(output.string())) + ".log");
    auto process = Process::start_writer(args, log);
    if (!process) return process.error();
    int written = 0;
    for (;;) {
        auto codes = encode_sequence_frame(*frame, *spec, o.peak_nits, o.source, o.shoulder);
        if (!codes) {
            (*process)->kill();
            (*process)->wait();
            return codes.error();
        }
        if (!(*process)->write(std::span(reinterpret_cast<const std::uint8_t*>(codes->data()), codes->size() * 2))) break;
        ++written;
        frame = next();
        if (!frame) break;
        if (frame->height() != height || frame->width() != width) {
            (*process)->kill();
            (*process)->wait();
            return encode_error("frame " + std::to_string(written) + " is " + std::to_string(frame->width()) + "x" +
                                std::to_string(frame->height()) + ", the first was " + std::to_string(width) + "x" +
                                std::to_string(height) + ". A sequence must not change size.");
        }
    }
    (*process)->close_input();
    const int code = (*process)->wait();
    std::string stderr_text;
    {
        std::ifstream in(log, std::ios::binary);
        stderr_text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    fs::remove(log, ec);
    if (code != 0) {
        auto ls = lines_of(strip(stderr_text));
        if (ls.empty()) ls = {"unknown error"};
        std::string tail;
        for (std::size_t i = ls.size() > 6 ? ls.size() - 6 : 0; i < ls.size(); ++i) tail += (tail.empty() ? "" : " | ") + ls[i];
        std::string cmd;
        for (const auto& s : args) cmd += (cmd.empty() ? "" : " ") + s;
        return encode_error("ffmpeg failed after " + std::to_string(written) + " frame(s): " + tail + "\n  command: " + cmd);
    }
    if (o.verify_tags) {
        std::string note;
        if (auto v = verify_sequence_tags(output, o.target, &note); !v) return v.error();
        if (!note.empty() && o.note) o.note(note);
    }
    return output;
}

Result<std::vector<fs::path>> list_linear_frames(const fs::path& path) {
    std::error_code ec;
    if (fs::is_directory(path, ec)) {
        std::vector<fs::path> found;
        for (const auto& e : fs::directory_iterator(path, ec)) {
            std::string ext = e.path().extension().string();
            for (char& c : ext) c = char(std::tolower(static_cast<unsigned char>(c)));
            if (ext == ".exr" || ext == ".npy") found.push_back(e.path());
        }
        if (found.empty()) return make_error(ErrorCode::NotFound, "no .exr/.npy frames in " + path.string());
        std::sort(found.begin(), found.end());
        return found;
    }
    if (!fs::exists(path, ec)) return make_error(ErrorCode::NotFound, path.string());
    return std::vector<fs::path>{path};
}

Result<NitsFrame> load_linear_frame(const fs::path& path, double nits_scale) {
    std::string ext = path.extension().string();
    for (char& c : ext) c = char(std::tolower(static_cast<unsigned char>(c)));
    NitsFrame out;
    auto fill = [&](int h, int w, auto&& at) {
        out = NitsFrame(h, w);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    out.plane(c)[std::size_t(y) * w + x] = std::max(at(c, y, x) * nits_scale, 0.0);
    };
    if (ext == ".npy") {
        auto a = read_npy_f64(path);
        if (!a) return a.error();
        if (a->shape.size() != 3 || a->shape[2] < 3) return make_error(ErrorCode::InvalidArgument, "expected (H, W, 3) frames", path.string());
        const int h = int(a->shape[0]), w = int(a->shape[1]), ch = int(a->shape[2]);
        fill(h, w, [&](int c, int y, int x) { return a->data[(std::size_t(y) * w + x) * ch + c]; });
        return out;
    }
    if (ext == ".exr") {
        auto e = read_exr(path);
        if (!e) return e.error();
        const PlanarBuffer& p = e->pixels;
        if (p.channels() < 3) return make_error(ErrorCode::InvalidArgument, "expected (H, W, 3) frames", path.string());
        fill(p.height(), p.width(), [&](int c, int y, int x) { return double(p.at(c, y, x)); });
        return out;
    }
    return make_error(ErrorCode::InvalidArgument, "unsupported frame format: " + path.string());
}

}  // namespace rudra
