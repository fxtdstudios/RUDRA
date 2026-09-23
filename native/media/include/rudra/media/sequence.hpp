#pragma once
// Open a shot by path: a folder of frames. Port of ui/sequence.py's folder
// half (Sequence.open, natural_key, describe): the same frame suffixes, the
// same natural order (frame_2 before frame_10), the same messages. Video files
// are recognised and refused until libav arrives (Phase 4).

#include <filesystem>
#include <string>
#include <vector>

#include "rudra/platform/result.hpp"

namespace rudra {

struct FrameSequence {
    std::filesystem::path path;
    std::string kind = "frames";
    std::vector<std::filesystem::path> frames;
    int count() const noexcept { return static_cast<int>(frames.size()); }
    std::string name_of(int index) const { return frames.at(std::size_t(index)).filename().string(); }
};

// Sequence.open(raw): strips blanks and quotes, expands ~.
Result<FrameSequence> open_sequence(const std::string& raw);

// natural_key ordering: digit runs compare as numbers, the rest case-folded.
bool natural_less(const std::string& a, const std::string& b);

const std::vector<std::string>& frame_suffixes();   // sorted, as the Python prints them
const std::vector<std::string>& video_suffixes();

}  // namespace rudra
