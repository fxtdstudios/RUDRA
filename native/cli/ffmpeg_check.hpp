#pragma once
#include <string>
#include <vector>

namespace rudra {
// rudra-native ffmpeg-check [--ffmpeg PATH] [--ffprobe PATH] [--force] [--no-self-test]
int cmd_ffmpeg_check(const std::vector<std::string>& args);
}  // namespace rudra
