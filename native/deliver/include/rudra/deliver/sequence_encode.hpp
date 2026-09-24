#pragma once
// A finished HDR sequence to one file a timeline takes (Phase 4, step 8):
// rudra/delivery/video.py encode_sequence. Frames in absolute nits are shaped
// to 16-bit RGB code values (primaries to Rec.2020, the shoulder into the
// mastering peak, PQ or HLG) and written down ffmpeg's stdin with the Python's
// command; the finished file's colour tags are read back where each format
// keeps them (the HEVC VUI through ffprobe, the ProRes frame header) and a
// file whose tags did not land is not handed over.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <utility>
#include <optional>
#include <string>
#include <vector>

#include "rudra/core/color.hpp"
#include "rudra/core/master.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct SequenceTarget {
    std::string name;
    std::vector<std::string> codec;
    std::string pix_fmt, suffix, transfer, note;   // transfer: pq, hlg or linear
};
const std::vector<SequenceTarget>& sequence_targets();   // prores4444, prores422hq, hdr10, hlg
const SequenceTarget* find_sequence_target(const std::string& name);

// Tags in the order the Python's dicts keep them (subtype, color_primaries,
// color_transfer, color_space).
using ColourTags = std::vector<std::pair<std::string, std::string>>;
std::string tag_of(const ColourTags& tags, const std::string& key);   // "" when absent
Result<ColourTags> expected_tags(const std::string& target);

// shoulder_to_peak: master_to_peak on nits (float32 inside, as the Python's).
Result<NitsFrame> shoulder_to_peak(const NitsFrame& rgb_nits, double peak_nits);

// _encode_frame: one frame in nits to height x width x 3 RGB 16-bit codes.
Result<std::vector<std::uint16_t>> encode_sequence_frame(const NitsFrame& rgb_nits, const SequenceTarget& target,
                                                         double peak_nits, Primaries source, bool shoulder = true);

// The ffmpeg command encode_sequence builds. `write_colr` and
// `prores_metadata` are what _ffmpeg_supports answered on this machine.
std::vector<std::string> sequence_encode_command(const SequenceTarget& target, int width, int height, double fps,
                                                 double peak_nits, std::optional<int> maxcll, std::optional<int> maxfall,
                                                 double min_nits, const std::filesystem::path& output, bool write_colr,
                                                 bool prores_metadata);

// _ffmpeg_supports: `ffmpeg -hide_banner -h kind=name` mentions `flag`; cached.
bool ffmpeg_supports(const std::string& kind, const std::string& name, const std::string& flag);

// The colour description in the first ProRes frame header, or nullopt.
std::optional<ColourTags> prores_frame_tags(const std::filesystem::path& path);
// The MOV/MP4 colr atom ("subtype" plus the three tags), or nullopt.
std::optional<ColourTags> container_colr(const std::filesystem::path& path);
// ffprobe's view of the tags, "unknown" for any the file does not carry.
Result<ColourTags> colour_tags(const std::filesystem::path& path);

// _verify_tags: an error with the Python's words, or ok; a note (ProRes whose
// colr atom is incomplete) goes to `note`.
Result<void> verify_sequence_tags(const std::filesystem::path& path, const std::string& target,
                                  std::string* note = nullptr);

struct SequenceEncodeOptions {
    std::string target = "hdr10";
    double fps = 24.0, peak_nits = 1000.0, min_nits = 0.005;
    std::optional<int> maxcll, maxfall;
    Primaries source = Primaries::Rec2020;
    bool verify_tags = true, shoulder = true;
    std::function<void(const std::string&)> note;   // what the Python prints to stderr
};

// Frames come one at a time from `next` (nullopt at the end), so a long 4K
// sequence is never held in memory. Returns the file written (output with
// the target's suffix).
Result<std::filesystem::path> encode_sequence(const std::function<std::optional<NitsFrame>()>& next,
                                              const std::filesystem::path& output, const SequenceEncodeOptions& options);

// `rudra deliver`'s inputs (delivery/cli.py _frames, bench.load_frame): a
// folder's .exr and .npy files in name order, or the one file; a frame as
// float64 nits, max(value * nits_scale, 0), its first three channels.
Result<std::vector<std::filesystem::path>> list_linear_frames(const std::filesystem::path& path);
Result<NitsFrame> load_linear_frame(const std::filesystem::path& path, double nits_scale);

}  // namespace rudra
