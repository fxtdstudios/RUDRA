#pragma once
// rudra-native video: rudra/video.py's command line, on a model package.
#include <string>
#include <vector>

namespace rudra {
// args after "video": <package> <input> --output <file> [video.py's options] [--runtime ...] [--device ...]
int cmd_video(const std::vector<std::string>& args);
}  // namespace rudra
