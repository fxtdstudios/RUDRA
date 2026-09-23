#include "rudra/platform/npy.hpp"

#include <bit>
#include <cctype>
#include <cstring>
#include <fstream>
#include <string>

namespace rudra {
namespace {

Error parse_error(const std::filesystem::path& p, const std::string& why) {
    return make_error(ErrorCode::ParseError, "The array file is not a float32 .npy file.",
                      p.string() + ": " + why);
}

// The header is a Python dict literal, e.g.
//   {'descr': '<f4', 'fortran_order': False, 'shape': (1, 3, 24, 32), }
std::string value_after(const std::string& header, const std::string& key) {
    const auto k = header.find("'" + key + "'");
    if (k == std::string::npos) return {};
    const auto colon = header.find(':', k);
    if (colon == std::string::npos) return {};
    auto start = header.find_first_not_of(' ', colon + 1);
    if (start == std::string::npos) return {};
    if (header[start] == '(') {
        const auto end = header.find(')', start);
        return end == std::string::npos ? std::string{} : header.substr(start, end - start + 1);
    }
    const auto end = header.find_first_of(",}", start);
    return header.substr(start, end - start);
}

}  // namespace

Result<NpyArray> read_npy(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return make_error(ErrorCode::NotFound, "The array file could not be opened.", path.string());

    char magic[6];
    in.read(magic, 6);
    if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0) return parse_error(path, "bad magic");
    unsigned char ver[2];
    in.read(reinterpret_cast<char*>(ver), 2);
    std::uint32_t header_len = 0;
    if (ver[0] == 1) {
        unsigned char b[2];
        in.read(reinterpret_cast<char*>(b), 2);
        header_len = b[0] | (b[1] << 8);
    } else if (ver[0] == 2 || ver[0] == 3) {
        unsigned char b[4];
        in.read(reinterpret_cast<char*>(b), 4);
        header_len = b[0] | (b[1] << 8) | (b[2] << 16) | (std::uint32_t(b[3]) << 24);
    } else {
        return parse_error(path, "unsupported format version " + std::to_string(ver[0]));
    }
    std::string header(header_len, '\0');
    in.read(header.data(), header_len);
    if (!in) return parse_error(path, "truncated header");

    const std::string descr = value_after(header, "descr");
    if (descr != "'<f4'" && descr != "'=f4'" && !(descr == "'|f4'"))
        return parse_error(path, "dtype " + descr + " (need '<f4')");
    if (value_after(header, "fortran_order") != "False") return parse_error(path, "Fortran order");

    NpyArray arr;
    const std::string shape = value_after(header, "shape");
    if (shape.size() < 2) return parse_error(path, "no shape");
    std::size_t i = 1;
    while (i < shape.size() - 1) {
        while (i < shape.size() - 1 && (shape[i] == ' ' || shape[i] == ',')) ++i;
        if (i >= shape.size() - 1) break;
        std::size_t j = i;
        while (j < shape.size() - 1 && std::isdigit(static_cast<unsigned char>(shape[j]))) ++j;
        if (j == i) return parse_error(path, "bad shape " + shape);
        arr.shape.push_back(std::stoll(shape.substr(i, j - i)));
        i = j;
    }

    const auto n = static_cast<std::size_t>(arr.size());
    arr.data.resize(n);
    in.read(reinterpret_cast<char*>(arr.data.data()), static_cast<std::streamsize>(n * sizeof(float)));
    if (static_cast<std::size_t>(in.gcount()) != n * sizeof(float)) return parse_error(path, "truncated data");
    static_assert(std::endian::native == std::endian::little,
                  "the .npy reader assumes a little-endian host; every RUDRA target is one");
    return arr;
}

}  // namespace rudra
