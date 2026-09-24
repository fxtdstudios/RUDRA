#pragma once
// The checkpoint manager's model (Phase 3 step 10): which model packages this
// machine has, which one a bare start opens, and which runtimes and devices
// can run it.
//
// Discovery is ui/server.py's, for packages instead of .pt files
// (tests/golden/catalog, tools/emit_catalog_golden.py). Roots are searched in
// order and the first that yields a model wins:
//   - a root with a models.json is described, not guessed: its default if a
//     package of that source file is there, else the first registered
//     SDR2HDR package, in registry order; entries of another kind (the
//     temporal refiner) are never listed;
//   - any other root is a training output: a package per run folder, whose
//     source is shipped_*.pt or best.pt, the shipped ones first, the newest
//     export from there.
// Folders named _invalid_* or starting with a dot are never listed. The
// manager lists every readable package besides (the server can only load
// what it would pick), and `pick()` falls back to the first of them when no
// rule picks one: a package, unlike a bare .pt, says what it is.

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rudra/core/model_manifest.hpp"
#include "rudra/infer/backend.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct CatalogEntry {
    std::filesystem::path package;           // the package folder
    std::size_t root = 0;                    // index into the roots searched
    std::optional<ModelManifest> manifest;   // nullopt: unreadable (problem says why)
    std::string problem;
    std::string title, note;                 // from the root's models.json
    bool registered = false;                 // listed by the root's models.json (/api/checkpoints)
    bool is_default = false;                 // the registry's default
    std::string label() const;               // title, else the manifest's name, else the folder's
};

struct Catalog {
    std::vector<std::filesystem::path> roots;
    std::vector<CatalogEntry> entries;       // root order; registry order, then by folder name
    std::optional<std::size_t> chosen;       // what the server's rules pick
    std::optional<std::size_t> pick() const; // chosen, else the first readable entry
    std::optional<std::size_t> find(const std::filesystem::path& package) const;
};

Catalog scan_packages(const std::vector<std::filesystem::path>& roots);

// The roots a bare start searches: RUDRA_PACKAGE_ROOTS (the platform's path
// separator between them), then `extra` (folders added in the manager), then
// the app's own models folder and the per-user one.
std::vector<std::filesystem::path> package_roots(const std::vector<std::filesystem::path>& extra,
                                                 const std::filesystem::path& app_models,
                                                 const std::filesystem::path& user_models);

// A runtime and a device this build could run a package on.
struct BackendChoice {
    Runtime runtime = Runtime::OnnxRuntime;
    Device device = Device::Cpu;
    bool operator==(const BackendChoice&) const = default;
    std::string label() const;               // "LibTorch on CUDA"
    std::string key() const;                 // "libtorch/cuda", for settings
    static std::optional<BackendChoice> from_key(const std::string& k);
};

// Every compiled runtime with the devices this platform can have, fastest
// first (the GPU paths, then ONNX Runtime and LibTorch on the CPU): the order
// a first load tries them in.
std::vector<BackendChoice> backend_choices();

// Opens the package on `preferred`, or on the first choice that opens. The
// choice taken comes back with the backend.
struct OpenedBackend {
    std::unique_ptr<InferenceBackend> backend;
    BackendChoice choice;
};
Result<OpenedBackend> open_backend(const ModelManifest& m, std::optional<BackendChoice> preferred = std::nullopt);
Result<std::unique_ptr<InferenceBackend>> open_backend_on(const ModelManifest& m, const BackendChoice& c);

// The page's #device pill: the GPU's name, or "CPU" (info.gpu).
std::string device_pill(const BackendInfo& info);

}  // namespace rudra
