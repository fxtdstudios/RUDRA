#pragma once
#include <string>
#include <vector>

namespace rudra {
// rudra-native batch run <queue.json> [--retry-failed] [--package DIR] [--runtime ...] [--device ...]
// rudra-native batch status <queue.json>
int cmd_batch(const std::vector<std::string>& args);
}  // namespace rudra
