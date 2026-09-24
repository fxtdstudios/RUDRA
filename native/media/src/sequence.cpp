#include "rudra/media/sequence.hpp"

#include <cstdio>
#include <map>
#include <mutex>

#include "rudra/platform/hash.hpp"
#include "rudra/platform/process.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace rudra {
namespace fs = std::filesystem;
namespace {

// natural_key: re.split(r"(\d+)") gives text at even positions, digits at odd.
std::vector<std::string> split_digits(const std::string& s) {
    std::vector<std::string> parts{""};
    bool digits = false;
    for (char c : s) {
        const bool d = std::isdigit(static_cast<unsigned char>(c)) != 0;
        if (d != digits) {
            parts.emplace_back();
            digits = d;
        }
        parts.back() += c;
    }
    if (digits) parts.emplace_back();   // the split always ends on text
    return parts;
}

int compare_number(const std::string& a, const std::string& b) {
    const auto sa = a.find_first_not_of('0'), sb = b.find_first_not_of('0');
    const std::string na = sa == std::string::npos ? "" : a.substr(sa), nb = sb == std::string::npos ? "" : b.substr(sb);
    if (na.size() != nb.size()) return na.size() < nb.size() ? -1 : 1;
    return na.compare(nb) < 0 ? -1 : (na == nb ? 0 : 1);
}

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Path.suffix: the last dot-part, but a name that is only a leading dot has none.
std::string suffix_of(const fs::path& p) {
    const std::string n = p.filename().string();
    const auto dot = n.rfind('.');
    if (dot == std::string::npos || dot == 0 || dot == n.size() - 1) return {};
    return n.substr(dot);
}

std::string strip(std::string s, const char* chars) {
    const auto b = s.find_first_not_of(chars);
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(chars);
    return s.substr(b, e - b + 1);
}

// PurePath's lexical clean-up, so messages and names read as str(Path) does:
// repeated separators collapse, "." parts and a trailing separator go, and
// an empty result is ".". ".." is kept, as pathlib keeps it.
fs::path pathlib_clean(const std::string& s) {
#if defined(_WIN32)
    const auto is_sep = [](char c) { return c == '/' || c == '\\'; };
    const char sep = '\\';
#else
    const auto is_sep = [](char c) { return c == '/'; };
    const char sep = '/';
#endif
    std::string out, part;
    const bool absolute = !s.empty() && is_sep(s[0]);
    std::vector<std::string> parts;
    for (std::size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || is_sep(s[i])) {
            if (!part.empty() && part != ".") parts.push_back(part);
            part.clear();
        } else {
            part += s[i];
        }
    }
    if (absolute) out += sep;
    for (std::size_t i = 0; i < parts.size(); ++i) out += (i ? std::string(1, sep) : std::string()) + parts[i];
    if (out.empty()) out = ".";
    return fs::path(out);
}

}  // namespace

const std::vector<std::string>& frame_suffixes() {
    static const std::vector<std::string> s{".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"};
    return s;
}
const std::vector<std::string>& video_suffixes() {
    static const std::vector<std::string> s{".avi", ".m2ts", ".m4v", ".mkv", ".mov", ".mp4", ".mxf", ".ts", ".webm"};
    return s;
}

bool natural_less(const std::string& a, const std::string& b) {
    const auto pa = split_digits(a), pb = split_digits(b);
    for (std::size_t i = 0; i < std::min(pa.size(), pb.size()); ++i) {
        int c;
        if (i % 2) c = compare_number(pa[i], pb[i]);
        else {
            const std::string la = lower(pa[i]), lb = lower(pb[i]);
            c = la < lb ? -1 : (la == lb ? 0 : 1);
        }
        if (c) return c < 0;
    }
    return pa.size() < pb.size();
}

namespace {
// Cache folder -> the video it holds frames of. Never destroyed: an engine's
// decode worker can still ask for a movie frame while the process exits, and
// locking a destroyed mutex aborts on macOS ("mutex lock failed").
struct Videos {
    std::mutex mu;
    std::map<fs::path, FrameSequence> by_cache;
};
Videos& videos() {
    static Videos* v = new Videos;
    return *v;
}

void register_video(const FrameSequence& seq) {
    Videos& v = videos();
    std::lock_guard lk(v.mu);
    v.by_cache[seq.cache] = seq;
}

std::optional<FrameSequence> video_of(const fs::path& frame) {
    Videos& v = videos();
    std::lock_guard lk(v.mu);
    const auto it = v.by_cache.find(frame.parent_path());
    if (it == v.by_cache.end()) return std::nullopt;
    return it->second;
}
}  // namespace

Result<FrameSequence> open_sequence(const std::string& raw) {
    std::string s = strip(raw, " \t\r\n\v\f");
    if (s.empty()) return make_error(ErrorCode::InvalidArgument, "No path given.");
    s = strip(strip(s, "\""), "'");
    if (!s.empty() && s[0] == '~') {
        const char* home = std::getenv("HOME");
#if defined(_WIN32)
        if (!home) home = std::getenv("USERPROFILE");
#endif
        if (home && (s.size() == 1 || s[1] == '/' || s[1] == '\\')) s = std::string(home) + s.substr(1);
    }
    const fs::path path = pathlib_clean(s);
    std::error_code ec;
    if (!fs::exists(path, ec)) return make_error(ErrorCode::NotFound, "Nothing at " + path.string());
    FrameSequence seq;
    seq.path = path;
    if (fs::is_directory(path, ec)) {
        for (const auto& e : fs::directory_iterator(path, ec)) {
            if (!e.is_regular_file(ec)) continue;
            const std::string suf = lower(suffix_of(e.path()));
            if (std::find(frame_suffixes().begin(), frame_suffixes().end(), suf) != frame_suffixes().end())
                seq.frames.push_back(e.path());
        }
        std::stable_sort(seq.frames.begin(), seq.frames.end(), [](const fs::path& x, const fs::path& y) {
            return natural_less(x.filename().string(), y.filename().string());
        });
        if (seq.frames.empty()) {
            std::string looked;
            for (const auto& f : frame_suffixes()) looked += (looked.empty() ? "" : " ") + f;
            return make_error(ErrorCode::NotFound, "No frames in " + path.filename().string() + ". Looked for " + looked);
        }
        return seq;
    }
    const std::string suf = lower(suffix_of(path));
    if (std::find(video_suffixes().begin(), video_suffixes().end(), suf) == video_suffixes().end()) {
        std::string list;
        for (const auto& v : video_suffixes()) list += (list.empty() ? "" : "/") + v.substr(1);
        return make_error(ErrorCode::Unsupported, path.filename().string() +
                                                      " is not a folder or a video RUDRA can read. Point at a folder of "
                                                      "frames, or a " + list + " file.");
    }
    if (!find_executable("ffmpeg") || !find_executable("ffprobe"))
        return make_error(ErrorCode::Unsupported,
                          "ffmpeg is not on PATH, so video cannot be read. Install it (winget install Gyan.FFmpeg) and "
                          "restart the server, or point at a folder of frames instead.");
    seq.kind = "video";
    auto first_line = [&](std::vector<std::string> extra) -> std::string {
        std::vector<std::string> argv = {"ffprobe", "-v", "error", "-select_streams", "v:0"};
        argv.insert(argv.end(), extra.begin(), extra.end());
        argv.insert(argv.end(), {"-of", "default=nokey=1:noprint_wrappers=1", path.string()});
        auto r = run_process(argv);
        if (!r) return {};
        const std::string out = strip(r->out, " \t\r\n");
        return out.substr(0, out.find_first_of("\r\n"));
    };
    // _probe: the rate as a float (24 when it cannot be read), the count by packets.
    double fps = 24.0;
    const std::string rate = first_line({"-show_entries", "stream=avg_frame_rate"});
    if (const auto slash = rate.find('/'); slash != std::string::npos) {
        char* e1 = nullptr;
        char* e2 = nullptr;
        const std::string a = rate.substr(0, slash), b = rate.substr(slash + 1);
        const double num = std::strtod(a.c_str(), &e1), den = std::strtod(b.c_str(), &e2);
        if (e1 && *e1 == '\0' && e2 && *e2 == '\0' && !a.empty() && !b.empty()) fps = den != 0 ? num / den : 24.0;
    }
    int count = 0;
    {
        const std::string counted = first_line({"-count_packets", "-show_entries", "stream=nb_read_packets"});
        char* e = nullptr;
        const long n = std::strtol(counted.c_str(), &e, 10);
        if (!counted.empty() && e && *e == '\0') count = int(n);
    }
    if (!count) return make_error(ErrorCode::NotFound, "ffprobe found no video frames in " + path.filename().string());
    seq.fps = fps;
    const std::string resolved = fs::weakly_canonical(fs::absolute(path, ec), ec).string();
    const std::string token = sha1_hex(std::as_bytes(std::span(resolved.data(), resolved.size()))).substr(0, 12);
    seq.cache = fs::temp_directory_path() / "rudra_seq" / token;
    fs::create_directories(seq.cache, ec);
    for (int i = 0; i < count; ++i) {
        char name[32];
        std::snprintf(name, sizeof name, "%06d.png", i);
        seq.frames.push_back(seq.cache / name);
    }
    register_video(seq);
    return seq;
}

std::string FrameSequence::name_of(int index) const {
    if (kind == "frames") return frames.at(std::size_t(index)).filename().string();
    char n[16];
    std::snprintf(n, sizeof n, "_%06d", index + 1);
    return path.stem().string() + n;
}

Result<fs::path> sequence_frame_file(const FrameSequence& seq, int index) {
    if (index < 0 || index >= seq.count())
        return make_error(ErrorCode::InvalidArgument,
                          "frame " + std::to_string(index) + " is outside 0.." + std::to_string(seq.count() - 1));
    const fs::path cached = seq.frames[std::size_t(index)];
    if (seq.kind == "frames") return cached;
    std::error_code ec;
    if (fs::is_regular_file(cached, ec)) return cached;
    // _extract: one frame, by seeking rather than decoding everything before it.
    const double when = index / std::max(seq.fps.value_or(24.0), 1e-6);
    char ss[64];
    std::snprintf(ss, sizeof ss, "%.6f", when);
    fs::path partial = cached;
    partial.replace_extension(".tmp.png");
    auto r = run_process({"ffmpeg", "-v", "error", "-y", "-accurate_seek", "-ss", ss, "-i", seq.path.string(),
                          "-frames:v", "1", partial.string()});
    if (!r || r->exit_code != 0 || !fs::is_regular_file(partial, ec)) {
        fs::remove(partial, ec);
        std::string last = "unknown error";
        if (r) {
            const std::string e = strip(r->err, " \t\r\n");
            if (!e.empty()) last = strip(e.substr(e.find_last_of('\n') == std::string::npos ? 0 : e.find_last_of('\n') + 1), "\r");
        }
        return make_error(ErrorCode::IoError, "ffmpeg could not read frame " + std::to_string(index) + ": " + last);
    }
    // Renamed last, so a cache entry never exists half-written.
    fs::rename(partial, cached, ec);
    if (ec) return make_error(ErrorCode::IoError, "ffmpeg could not read frame " + std::to_string(index) + ": " + ec.message());
    return cached;
}

Result<fs::path> ensure_frame_file(const fs::path& frame) {
    std::error_code ec;
    if (fs::is_regular_file(frame, ec)) return frame;
    if (auto v = video_of(frame)) {
        int index = 0;
        std::sscanf(frame.stem().string().c_str(), "%d", &index);
        return sequence_frame_file(*v, index);
    }
    return frame;   // the decoder reports what is wrong with it
}

}  // namespace rudra
