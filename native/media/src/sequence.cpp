#include "rudra/media/sequence.hpp"

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
    return make_error(ErrorCode::Unsupported,
                      "Video opens natively once libav is in (Phase 4); point at a folder of frames for now.",
                      path.string());
}

}  // namespace rudra
