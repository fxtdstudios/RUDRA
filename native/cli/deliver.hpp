#pragma once
#include <string>
#include <vector>

namespace rudra {
// rudra-native deliver <frames> --output <stem> [--target ...] [--fps N] [--peak-nits N] [--min-nits N]
//                      [--source-space rec2020|rec709|p3d65] [--nits-scale N] [--no-verify-tags]
int cmd_deliver(const std::vector<std::string>& args);
}  // namespace rudra
