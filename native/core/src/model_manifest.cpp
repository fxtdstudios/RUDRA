#include "rudra/core/model_manifest.hpp"

#include <fstream>

#include <nlohmann/json.hpp>

#include "rudra/platform/hash.hpp"

namespace rudra {

Result<ModelManifest> read_manifest(const std::filesystem::path& package_root) {
    const auto path = package_root / "manifest.json";
    std::ifstream in(path);
    if (!in) return make_error(ErrorCode::NotFound, "This folder is not a RUDRA model package.", path.string());

    nlohmann::json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        return make_error(ErrorCode::ParseError, "The model manifest is not valid JSON.", e.what());
    }

    ModelManifest m;
    m.root = package_root;
    try {
        m.contract = j.at("contract").get<std::string>();
        const int major = std::stoi(m.contract.substr(0, m.contract.find('.')));
        if (major != kSupportedContractMajor)
            return make_error(ErrorCode::ContractMismatch,
                              "This model package was written for a different version of RUDRA.",
                              "contract " + m.contract + ", this build reads " +
                                  std::to_string(kSupportedContractMajor) + ".x");
        m.name = j.at("name").get<std::string>();
        m.source_file = j.at("source").at("file").get<std::string>();
        m.source_sha256 = j.at("source").at("sha256").get<std::string>();
        const auto& net = j.at("network");
        m.corpus_ev = net.at("corpus_ev").get<float>();
        m.log_scale = net.at("log_scale").get<float>();
        m.max_hdr = net.at("max_hdr").get<float>();
        m.has_residual_gate = net.at("heads").at("residual_gate").get<bool>();
        m.has_shadow_gate = net.at("heads").at("shadow_gate").get<bool>();
        m.has_curve = net.at("heads").at("curve").get<bool>();
        m.curve_params = net.at("curve_params").get<int>();
        m.tile_size = j.at("tiling").at("tile_size").get<int>();
        m.overlap = j.at("tiling").at("overlap").get<int>();
        const auto& files = j.at("files");
        m.torchscript = files.at("torchscript").get<std::string>();
        m.onnx_frame = files.at("onnx_frame").get<std::string>();
        m.onnx_tile = files.at("onnx_tile").get<std::string>();
        m.golden = files.at("golden").get<std::string>();
        for (const char* key : {"torchscript", "onnx_frame", "onnx_tile"})
            m.file_sha256[files.at(key).get<std::string>()] = files.at(std::string(key) + "_sha256").get<std::string>();
        m.onnx_frame_inputs = j.at("onnx_inputs").at("frame").get<std::vector<std::string>>();
        m.onnx_tile_inputs = j.at("onnx_inputs").at("tile").get<std::vector<std::string>>();
        for (const char* key : {"torchscript", "onnx"}) {
            const auto& t = j.at("tolerance").at(key);
            m.tolerance[key] = Tolerance{t.at("atol").get<double>(), t.at("rtol").get<double>()};
        }
    } catch (const std::exception& e) {
        return make_error(ErrorCode::ParseError, "The model manifest is missing a required field.", e.what());
    }
    if (m.has_curve != (m.curve_params > 1))
        return make_error(ErrorCode::ParseError, "The model manifest contradicts itself.",
                          "curve head flag and curve_params disagree");
    return m;
}

Result<void> verify_package_files(const ModelManifest& m) {
    for (const auto& [file, expected] : m.file_sha256) {
        auto got = sha256_file(m.root / file);
        if (!got) return got.error();
        if (*got != expected)
            return make_error(ErrorCode::IntegrityError, "A model file does not match its recorded hash.",
                              file + ": expected " + expected + ", got " + *got);
    }
    return {};
}

}  // namespace rudra
