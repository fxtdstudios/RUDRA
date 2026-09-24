#include "rudra/platform/npy.hpp"

#include <bit>
#include <cctype>
#include <cstring>
#include <fstream>
#include <string>

namespace rudra {
namespace {

Error parse_error(const std::filesystem::path& p, const std::string& why) {
    return make_error(ErrorCode::ParseError, "The array file is not a float .npy file.",
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

namespace {

struct Header {
    std::vector<std::int64_t> shape;
    bool f64 = false;
    bool u8 = false;
    bool u16 = false;
};

Result<Header> read_header(std::ifstream& in, const std::filesystem::path& path, bool allow_f64) {
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

    Header h;
    const std::string descr = value_after(header, "descr");
    const bool f4 = descr == "'<f4'" || descr == "'=f4'" || descr == "'|f4'";
    h.f64 = descr == "'<f8'" || descr == "'=f8'";
    h.u8 = descr == "'|u1'";
    h.u16 = descr == "'<u2'" || descr == "'=u2'";
    if (!f4 && !h.u8 && !h.u16 && !(allow_f64 && h.f64))
        return parse_error(path, "dtype " + descr + (allow_f64 ? " (need '<f4', '|u1', '<u2' or '<f8')" : " (need '<f4', '|u1' or '<u2')"));
    if (value_after(header, "fortran_order") != "False") return parse_error(path, "Fortran order");

    const std::string shape = value_after(header, "shape");
    if (shape.size() < 2) return parse_error(path, "no shape");
    std::size_t i = 1;
    while (i < shape.size() - 1) {
        while (i < shape.size() - 1 && (shape[i] == ' ' || shape[i] == ',')) ++i;
        if (i >= shape.size() - 1) break;
        std::size_t j = i;
        while (j < shape.size() - 1 && std::isdigit(static_cast<unsigned char>(shape[j]))) ++j;
        if (j == i) return parse_error(path, "bad shape " + shape);
        h.shape.push_back(std::stoll(shape.substr(i, j - i)));
        i = j;
    }
    return h;
}

template <class T>
bool read_payload(std::ifstream& in, std::vector<T>& out, std::size_t n) {
    out.resize(n);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n * sizeof(T)));
    return static_cast<std::size_t>(in.gcount()) == n * sizeof(T);
}

std::size_t count(const std::vector<std::int64_t>& shape) {
    std::size_t n = 1;
    for (auto d : shape) n *= static_cast<std::size_t>(d);
    return n;
}

static_assert(std::endian::native == std::endian::little,
              "the .npy reader assumes a little-endian host; every RUDRA target is one");

}  // namespace

Result<NpyArray> read_npy(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return make_error(ErrorCode::NotFound, "The array file could not be opened.", path.string());
    auto h = read_header(in, path, false);
    if (!h) return h.error();
    NpyArray arr;
    arr.shape = h->shape;
    if (h->u8) {
        std::vector<std::uint8_t> b;
        if (!read_payload(in, b, count(arr.shape))) return parse_error(path, "truncated data");
        arr.data.assign(b.begin(), b.end());
        return arr;
    }
    if (h->u16) {
        std::vector<std::uint16_t> b;
        if (!read_payload(in, b, count(arr.shape))) return parse_error(path, "truncated data");
        arr.data.assign(b.begin(), b.end());   // exact: every uint16 is a float
        return arr;
    }
    if (!read_payload(in, arr.data, count(arr.shape))) return parse_error(path, "truncated data");
    return arr;
}

Result<NpyArrayF64> read_npy_f64(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return make_error(ErrorCode::NotFound, "The array file could not be opened.", path.string());
    auto h = read_header(in, path, true);
    if (!h) return h.error();
    NpyArrayF64 arr;
    arr.shape = h->shape;
    const std::size_t n = count(arr.shape);
    if (h->f64) {
        if (!read_payload(in, arr.data, n)) return parse_error(path, "truncated data");
    } else if (h->u8) {
        std::vector<std::uint8_t> b;
        if (!read_payload(in, b, n)) return parse_error(path, "truncated data");
        arr.data.assign(b.begin(), b.end());
    } else if (h->u16) {
        std::vector<std::uint16_t> b;
        if (!read_payload(in, b, n)) return parse_error(path, "truncated data");
        arr.data.assign(b.begin(), b.end());
    } else {
        std::vector<float> f;
        if (!read_payload(in, f, n)) return parse_error(path, "truncated data");
        arr.data.assign(f.begin(), f.end());
    }
    return arr;
}

}  // namespace rudra
