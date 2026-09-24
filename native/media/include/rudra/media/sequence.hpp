#pragma once
// Open a shot by path: a folder of frames, or a video file. Port of
// ui/sequence.py (Sequence.open, natural_key, describe, _probe, _extract): the
// same frame suffixes, the same natural order (frame_2 before frame_10), the
// same messages. A video is counted by its packets and read a frame at a time
// by seeking (ffmpeg -accurate_seek -ss), each frame kept as a PNG in the
// Studio's cache folder, so a movie scrubs like a folder.

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "rudra/platform/result.hpp"

namespace rudra {

struct FrameSequence {
    std::filesystem::path path;
    std::string kind = "frames";                  // "frames" or "video"
    std::vector<std::filesystem::path> frames;    // a folder's frames; for a video, each frame's cache file
    std::optional<double> fps;                    // a video's rate (describe()'s "fps"); none for frames
    std::filesystem::path cache;                  // a video's frame cache folder
    int count() const noexcept { return static_cast<int>(frames.size()); }
    std::string name_of(int index) const;         // the file's name, or "<stem>_000001" for a video
};

// frame_bytes' file: a folder's frame as it is, a video's frame extracted into
// the cache on first use ("ffmpeg could not read frame N: ..." as the Python).
Result<std::filesystem::path> sequence_frame_file(const FrameSequence& seq, int index);
// The same for a path that may be a video frame's cache file not yet written:
// frames that come from open_sequence know their video; a plain file is itself.
Result<std::filesystem::path> ensure_frame_file(const std::filesystem::path& frame);

// Sequence.open(raw): strips blanks and quotes, expands ~.
Result<FrameSequence> open_sequence(const std::string& raw);

// natural_key ordering: digit runs compare as numbers, the rest case-folded.
bool natural_less(const std::string& a, const std::string& b);

const std::vector<std::string>& frame_suffixes();   // sorted, as the Python prints them
const std::vector<std::string>& video_suffixes();

}  // namespace rudra
