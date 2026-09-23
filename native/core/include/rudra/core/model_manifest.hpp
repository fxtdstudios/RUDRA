#pragma once
// The model package's manifest.json (written by tools/export_model.py).
// Capability negotiation (NATIVE_ARCHITECTURE.md 5.9): the engine enables only
// what a model declares, and refuses a contract major it does not know.

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "rudra/platform/result.hpp"

namespace rudra {

inline constexpr int kSupportedContractMajor = 1;

struct Tolerance {
    double atol = 0.0;
    double rtol = 0.0;
};

struct ModelManifest {
    std::filesystem::path root;          // the package folder
    std::string contract;                // "1.0"
    std::string name;
    std::string source_file, source_sha256;
    float corpus_ev = -1.0f;
    float log_scale = 16.0f;
    float max_hdr = 4.0f;
    bool has_residual_gate = false, has_shadow_gate = false, has_curve = false;
    int curve_params = 1;
    int tile_size = 512, overlap = 64;
    std::filesystem::path torchscript, onnx_frame, onnx_tile, golden;
    std::map<std::string, std::string> file_sha256;   // relative file -> sha256
    std::vector<std::string> onnx_frame_inputs, onnx_tile_inputs;
    std::map<std::string, Tolerance> tolerance;       // "torchscript", "onnx"
};

// Reads and validates root/manifest.json. Refuses an unknown contract major.
Result<ModelManifest> read_manifest(const std::filesystem::path& package_root);

// Recomputes the sha256 of every file the manifest lists and compares.
Result<void> verify_package_files(const ModelManifest& m);

}  // namespace rudra
