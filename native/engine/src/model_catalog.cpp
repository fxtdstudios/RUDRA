#include "rudra/engine/model_catalog.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <system_error>

#include <nlohmann/json.hpp>

namespace rudra {
namespace fs = std::filesystem;

namespace {

struct RegistryModel {
    std::string file, kind, title, note;
};
struct Registry {
    std::optional<std::string> default_file;
    std::vector<RegistryModel> models;
};

// models.json, as registry() reads it: a bad file is an empty registry.
std::optional<Registry> read_registry(const fs::path& root) {
    const fs::path p = root / "models.json";
    std::error_code ec;
    if (!fs::is_regular_file(p, ec)) return std::nullopt;
    Registry r;
    try {
        std::ifstream in(p);
        const auto j = nlohmann::json::parse(in);
        if (j.contains("default") && j.at("default").is_string()) r.default_file = j.at("default").get<std::string>();
        if (j.contains("models") && j.at("models").is_array())
            for (const auto& m : j.at("models")) {
                if (!m.is_object()) continue;
                auto str = [&](const char* k) {
                    return m.contains(k) && m.at(k).is_string() ? m.at(k).get<std::string>() : std::string();
                };
                r.models.push_back({str("file"), str("kind"), str("title"), str("note")});
            }
    } catch (const std::exception&) {
        return Registry{};
    }
    return r;
}

bool hidden(const fs::path& p) {
    const std::string n = p.filename().string();
    return n.empty() || n[0] == '.' || n.rfind("_invalid_", 0) == 0;
}

// The package folders in a root: the root itself if it is one, else each
// folder in it with a manifest.json, by name.
std::vector<fs::path> package_dirs(const fs::path& root) {
    std::error_code ec;
    if (fs::is_regular_file(root / "manifest.json", ec)) return {root};
    std::vector<fs::path> out;
    if (!fs::is_directory(root, ec)) return out;
    for (const auto& e : fs::directory_iterator(root, fs::directory_options::skip_permission_denied, ec)) {
        std::error_code e2;
        if (!e.is_directory(e2) || hidden(e.path())) continue;
        if (fs::is_regular_file(e.path() / "manifest.json", e2)) out.push_back(e.path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

CatalogEntry read_entry(const fs::path& dir, std::size_t root) {
    CatalogEntry e;
    e.package = dir;
    e.root = root;
    auto m = read_manifest(dir);
    if (m) e.manifest = std::move(*m);
    else e.problem = m.error().message + (m.error().detail.empty() ? "" : " " + m.error().detail);
    return e;
}

std::string source_of(const CatalogEntry& e) { return e.manifest ? e.manifest->source_file : std::string(); }

}  // namespace

std::string CatalogEntry::label() const {
    if (!title.empty()) return title;
    if (manifest && !manifest->name.empty()) return manifest->name;
    return package.filename().string();
}

std::optional<std::size_t> Catalog::pick() const {
    if (chosen) return chosen;
    for (std::size_t i = 0; i < entries.size(); ++i)
        if (entries[i].manifest) return i;
    return std::nullopt;
}

std::optional<std::size_t> Catalog::find(const fs::path& package) const {
    std::error_code ec;
    const fs::path want = fs::weakly_canonical(package, ec);
    for (std::size_t i = 0; i < entries.size(); ++i) {
        std::error_code e2;
        if (fs::weakly_canonical(entries[i].package, e2) == want) return i;
    }
    return std::nullopt;
}

Catalog scan_packages(const std::vector<fs::path>& roots) {
    Catalog c;
    c.roots = roots;
    for (std::size_t ri = 0; ri < roots.size(); ++ri) {
        const fs::path& root = roots[ri];
        std::vector<CatalogEntry> here;
        for (const auto& d : package_dirs(root)) here.push_back(read_entry(d, ri));
        std::optional<std::size_t> pick;   // index into `here`

        if (auto reg = read_registry(root)) {
            // Described, not guessed: registry order for what it lists.
            std::vector<CatalogEntry> ordered;
            std::vector<bool> used(here.size(), false);
            std::map<std::string, std::string> kind_of;
            for (const auto& m : reg->models) kind_of.emplace(m.file, m.kind);
            for (const auto& m : reg->models) {
                if (m.kind != "sdr2hdr") continue;
                for (std::size_t i = 0; i < here.size(); ++i) {
                    if (used[i] || source_of(here[i]) != m.file) continue;
                    used[i] = true;
                    CatalogEntry e = here[i];
                    e.title = m.title;
                    e.note = m.note;
                    e.registered = true;
                    e.is_default = reg->default_file && *reg->default_file == m.file;
                    ordered.push_back(std::move(e));
                    break;
                }
            }
            // Then what it does not list, unless it names it as another kind.
            for (std::size_t i = 0; i < here.size(); ++i) {
                if (used[i]) continue;
                const auto k = kind_of.find(source_of(here[i]));
                if (k != kind_of.end() && k->second != "sdr2hdr") continue;
                ordered.push_back(here[i]);
            }
            here = std::move(ordered);
            for (std::size_t i = 0; i < here.size() && !pick; ++i)
                if (here[i].is_default) pick = i;
            for (std::size_t i = 0; i < here.size() && !pick; ++i)
                if (here[i].registered) pick = i;
        } else {
            // A training output: <run>/shipped_*.pt or <run>/best.pt, shipped
            // first, the newest from there.
            auto shipped = [&](const CatalogEntry& e) { return source_of(e).rfind("shipped_", 0) == 0; };
            auto candidate = [&](const CatalogEntry& e) {
                return e.manifest && e.package != root && (shipped(e) || source_of(e) == "best.pt");
            };
            bool any_shipped = false;
            for (const auto& e : here) any_shipped = any_shipped || (candidate(e) && shipped(e));
            for (std::size_t i = 0; i < here.size(); ++i) {
                if (!candidate(here[i]) || (any_shipped && !shipped(here[i]))) continue;
                if (!pick || here[i].manifest->exported > here[*pick].manifest->exported) pick = i;
            }
        }
        if (pick && !c.chosen) c.chosen = c.entries.size() + *pick;
        for (auto& e : here) c.entries.push_back(std::move(e));
    }
    return c;
}

std::vector<fs::path> package_roots(const std::vector<fs::path>& extra, const fs::path& app_models,
                                    const fs::path& user_models) {
    std::vector<fs::path> out;
    auto add = [&](const fs::path& p) {
        if (p.empty()) return;
        if (std::find(out.begin(), out.end(), p) == out.end()) out.push_back(p);
    };
#ifdef _WIN32
    constexpr char sep = ';';
#else
    constexpr char sep = ':';
#endif
    if (const char* env = std::getenv("RUDRA_PACKAGE_ROOTS")) {
        std::string s(env);
        std::size_t at = 0;
        while (at <= s.size()) {
            const std::size_t end = std::min(s.find(sep, at), s.size());
            std::string part = s.substr(at, end - at);
            const auto first = part.find_first_not_of(" \t"), last = part.find_last_not_of(" \t");
            if (first != std::string::npos) add(part.substr(first, last - first + 1));
            at = end + 1;
        }
    }
    for (const auto& p : extra) add(p);
    add(app_models);
    add(user_models);
    return out;
}

std::string BackendChoice::label() const {
    const char* r = runtime == Runtime::LibTorch ? "LibTorch" : "ONNX Runtime";
    const char* d = "CPU";
    switch (device) {
        case Device::Cpu: d = "CPU"; break;
        case Device::Cuda: d = "CUDA"; break;
        case Device::Mps: d = "MPS"; break;
        case Device::DirectML: d = "DirectML"; break;
        case Device::CoreML: d = "Core ML"; break;
        case Device::Rocm: d = "ROCm"; break;
        case Device::OpenVino: d = "OpenVINO"; break;
    }
    return std::string(r) + " on " + d;
}

std::string BackendChoice::key() const { return std::string(to_string(runtime)) + "/" + to_string(device); }

std::optional<BackendChoice> BackendChoice::from_key(const std::string& k) {
    // Any pair, compiled here or not: a setting from another build still reads.
    for (Runtime r : {Runtime::LibTorch, Runtime::OnnxRuntime})
        for (Device d : {Device::Cpu, Device::Cuda, Device::Mps, Device::DirectML, Device::CoreML, Device::Rocm,
                         Device::OpenVino})
            if (BackendChoice{r, d}.key() == k) return BackendChoice{r, d};
    return std::nullopt;
}

std::vector<BackendChoice> backend_choices() {
    // The order the app has always tried: the platform's GPU paths, then
    // ONNX Runtime on the CPU (the faster of the two there), then LibTorch.
    const std::vector<BackendChoice> order = {
#if defined(_WIN32)
        {Runtime::LibTorch, Device::Cuda}, {Runtime::OnnxRuntime, Device::DirectML},
#elif defined(__APPLE__)
        {Runtime::OnnxRuntime, Device::CoreML}, {Runtime::LibTorch, Device::Mps},
#else
        {Runtime::LibTorch, Device::Cuda},
#endif
        {Runtime::OnnxRuntime, Device::Cpu}, {Runtime::LibTorch, Device::Cpu},
    };
    const auto compiled = compiled_runtimes();
    std::vector<BackendChoice> out;
    for (const auto& c : order)
        if (std::find(compiled.begin(), compiled.end(), c.runtime) != compiled.end()) out.push_back(c);
    return out;
}

Result<std::unique_ptr<InferenceBackend>> open_backend_on(const ModelManifest& m, const BackendChoice& c) {
    return c.runtime == Runtime::LibTorch ? make_libtorch_backend(m, c.device) : make_onnxruntime_backend(m, c.device);
}

Result<OpenedBackend> open_backend(const ModelManifest& m, std::optional<BackendChoice> preferred) {
    if (preferred) {
        auto b = open_backend_on(m, *preferred);
        if (!b) return b.error();
        return OpenedBackend{std::move(*b), *preferred};
    }
    std::string tried;
    for (const auto& c : backend_choices()) {
        auto b = open_backend_on(m, c);
        if (b) return OpenedBackend{std::move(*b), c};
        tried += (tried.empty() ? "" : "; ") + c.label() + ": " + b.error().message;
    }
    return make_error(ErrorCode::Unsupported, "No runtime in this build can open the package.", tried);
}

std::string device_pill(const BackendInfo& info) {
    if (info.device == Device::Cpu) return "CPU";
    return BackendChoice{info.runtime, info.device}.label().substr(BackendChoice{info.runtime, info.device}.label().find(" on ") + 4);
}

}  // namespace rudra
