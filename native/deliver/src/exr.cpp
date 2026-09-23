#include "rudra/deliver/exr.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>

#include "rudra/core/gamut.hpp"
#include "rudra/platform/pyjson.hpp"

namespace rudra {

const Chromaticities kAp0Chromaticities{0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767};
const Chromaticities kAp1Chromaticities{0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767};
const Chromaticities kRec2020Chromaticities{0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290};

namespace {

constexpr std::int32_t kMagic = 20000630;

void put_i32(std::vector<std::uint8_t>& o, std::int32_t v) {
    const auto u = static_cast<std::uint32_t>(v);
    for (int k = 0; k < 4; ++k) o.push_back(std::uint8_t(u >> (8 * k)));
}
void put_u64(std::vector<std::uint8_t>& o, std::uint64_t v) {
    for (int k = 0; k < 8; ++k) o.push_back(std::uint8_t(v >> (8 * k)));
}
void put_f32(std::vector<std::uint8_t>& o, float f) { put_i32(o, std::bit_cast<std::int32_t>(f)); }
void put_str(std::vector<std::uint8_t>& o, const std::string& s) {
    o.insert(o.end(), s.begin(), s.end());
    o.push_back(0);
}
void attr(std::vector<std::uint8_t>& o, const std::string& name, const std::string& type,
          const std::vector<std::uint8_t>& payload) {
    put_str(o, name);
    put_str(o, type);
    put_i32(o, static_cast<std::int32_t>(payload.size()));
    o.insert(o.end(), payload.begin(), payload.end());
}

}  // namespace

std::uint16_t float_to_half(float f) noexcept {
    const std::uint32_t x = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const std::uint32_t abs = x & 0x7fffffffu;
    if (abs >= 0x7f800000u) {   // inf or nan
        return std::uint16_t(sign | (abs > 0x7f800000u ? 0x7e00u : 0x7c00u));
    }
    if (abs >= 0x477ff000u) return std::uint16_t(sign | 0x7c00u);   // rounds to inf
    if (abs < 0x38800000u) {   // half subnormal or zero
        if (abs < 0x33000000u) return std::uint16_t(sign);   // below half the smallest subnormal
        const int e = int(abs >> 23);
        const std::uint32_t mant = (abs & 0x7fffffu) | 0x800000u;
        const int shift = 126 - e;   // the value in units of 2^-24 is mant >> shift
        const std::uint32_t q = mant >> shift;
        const std::uint32_t rem = mant & ((1u << (shift)) - 1u);
        const std::uint32_t half = 1u << (shift - 1);
        std::uint32_t r = q;
        if (rem > half || (rem == half && (q & 1u))) ++r;
        return std::uint16_t(sign | r);
    }
    // Normal: rebias exponent, round the 13 dropped mantissa bits to nearest even.
    std::uint32_t h = ((abs - 0x38000000u) >> 13);
    const std::uint32_t rem = abs & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return std::uint16_t(sign | h);
}

Result<std::vector<std::uint8_t>> exr_bytes(const PlanarBuffer& rgb, bool half,
                                            const std::optional<Chromaticities>& chroma, const ExrAttributes& attributes) {
    const int n_ch = rgb.channels();
    if (n_ch != 3 && n_ch != 4) return make_error(ErrorCode::InvalidArgument, "expected a 3 or 4 channel image");
    const int w = rgb.width(), h = rgb.height();
    static const char* names[] = {"R", "G", "B", "A"};
    std::vector<int> order(static_cast<std::size_t>(n_ch));
    for (int i = 0; i < n_ch; ++i) order[std::size_t(i)] = i;
    std::sort(order.begin(), order.end(), [](int a, int b) { return std::strcmp(names[a], names[b]) < 0; });

    std::vector<std::uint8_t> head;
    {
        std::vector<std::uint8_t> ch;
        for (int i : order) {
            put_str(ch, names[i]);
            put_i32(ch, half ? 1 : 2);
            ch.insert(ch.end(), {0, 0, 0, 0});
            put_i32(ch, 1);
            put_i32(ch, 1);
        }
        ch.push_back(0);
        attr(head, "channels", "chlist", ch);
    }
    attr(head, "compression", "compression", {0});
    std::vector<std::uint8_t> box;
    for (int v : {0, 0, w - 1, h - 1}) put_i32(box, v);
    attr(head, "dataWindow", "box2i", box);
    attr(head, "displayWindow", "box2i", box);
    attr(head, "lineOrder", "lineOrder", {0});
    std::vector<std::uint8_t> f1, v2;
    put_f32(f1, 1.0f);
    put_f32(v2, 0.0f);
    put_f32(v2, 0.0f);
    attr(head, "pixelAspectRatio", "float", f1);
    attr(head, "screenWindowCenter", "v2f", v2);
    attr(head, "screenWindowWidth", "float", f1);
    if (chroma) {
        std::vector<std::uint8_t> c;
        for (double v : *chroma) put_f32(c, static_cast<float>(v));
        attr(head, "chromaticities", "chromaticities", c);
    }
    for (const auto& [k, v] : attributes) attr(head, k, "string", std::vector<std::uint8_t>(v.begin(), v.end()));
    head.push_back(0);

    const int bpp = half ? 2 : 4;
    const std::int64_t row_size = std::int64_t(w) * bpp * n_ch;
    const std::int64_t chunk = 8 + row_size;
    const std::int64_t data_start = 8 + std::int64_t(head.size()) + 8 * std::int64_t(h);

    std::vector<std::uint8_t> out;
    out.reserve(std::size_t(data_start + chunk * h));
    put_i32(out, kMagic);
    put_i32(out, 2);
    out.insert(out.end(), head.begin(), head.end());
    for (int y = 0; y < h; ++y) put_u64(out, std::uint64_t(data_start + std::int64_t(y) * chunk));

    auto clean = [half](float v) {
        if (std::isnan(v)) v = 0.0f;
        else if (v == std::numeric_limits<float>::infinity()) v = 3.4e38f;
        else if (v == -std::numeric_limits<float>::infinity()) v = 0.0f;
        if (half) v = std::clamp(v, -65504.0f, 65504.0f);
        return v;
    };
    for (int y = 0; y < h; ++y) {
        put_i32(out, y);
        put_i32(out, static_cast<std::int32_t>(row_size));
        for (int c : order) {
            const float* row = rgb.plane(c) + std::size_t(y) * std::size_t(w);
            for (int x = 0; x < w; ++x) {
                const float v = clean(row[x]);
                if (half) {
                    const std::uint16_t hv = float_to_half(v);
                    out.push_back(std::uint8_t(hv));
                    out.push_back(std::uint8_t(hv >> 8));
                } else {
                    put_f32(out, v);
                }
            }
        }
    }
    return out;
}

Result<void> write_exr(const std::filesystem::path& path, const PlanarBuffer& rgb, bool half,
                       const std::optional<Chromaticities>& chroma, const ExrAttributes& attributes) {
    auto bytes = exr_bytes(rgb, half, chroma, attributes);
    if (!bytes) return bytes.error();
    std::error_code ec;
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return make_error(ErrorCode::IoError, "The EXR file could not be created.", path.string());
    f.write(reinterpret_cast<const char*>(bytes->data()), static_cast<std::streamsize>(bytes->size()));
    if (!f) return make_error(ErrorCode::IoError, "The EXR file could not be written.", path.string());
    return {};
}

const char* primaries_name(Primaries p) noexcept {
    switch (p) {
        case Primaries::Rec709: return "rec709";
        case Primaries::Rec2020: return "rec2020";
        case Primaries::P3D65: return "p3d65";
        case Primaries::Ap0: return "ap0";
        case Primaries::Ap1: return "ap1";
    }
    return "?";
}

namespace {

Result<void> write_aces_like(const std::filesystem::path& path, const PlanarBuffer& rgb, Primaries source,
                             Primaries dst, double exposure, const ExrAttributes& provenance, bool ap0) {
    PlanarBuffer scaled = rgb;
    const float k = static_cast<float>(exposure);
    for (float& v : scaled.span()) v = v * k;
    const PlanarBuffer out = convert_primaries(scaled, source, dst);
    ExrAttributes attrs;
    if (ap0) attrs.emplace_back("acesImageContainerFlag", "1");
    else attrs.emplace_back("rudra:container", "ACEScg (AP1)");
    attrs.emplace_back("rudra:sourceSpace", primaries_name(source));
    attrs.emplace_back("rudra:exposureConvention", "0.18 = 18% grey (scene), diffuse white = 1.0");
    attrs.emplace_back("rudra:exposureScale", pyjson::repr(exposure));
    if (!provenance.empty()) {
        pyjson::Dict d;
        for (const auto& [kk, v] : provenance) d.emplace_back(kk, v);
        attrs.emplace_back("rudra:provenance", pyjson::dumps(d, -1, true));
    }
    return write_exr(path, out, true, ap0 ? kAp0Chromaticities : kAp1Chromaticities, attrs);
}

}  // namespace

Result<void> write_aces_exr(const std::filesystem::path& path, const PlanarBuffer& rgb, Primaries source,
                            double exposure, const ExrAttributes& provenance) {
    return write_aces_like(path, rgb, source, Primaries::Ap0, exposure, provenance, true);
}

Result<void> write_acescg_exr(const std::filesystem::path& path, const PlanarBuffer& rgb, Primaries source,
                              double exposure, const ExrAttributes& provenance) {
    return write_aces_like(path, rgb, source, Primaries::Ap1, exposure, provenance, false);
}

namespace {
float half_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu;
    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            int e = -1;
            do { ++e; mant <<= 1; } while (!(mant & 0x400u));
            bits = sign | std::uint32_t(127 - 15 - e) << 23 | (mant & 0x3ffu) << 13;
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | mant << 13;
    } else {
        bits = sign | (exp + 112) << 23 | mant << 13;
    }
    return std::bit_cast<float>(bits);
}
}  // namespace

Result<ExrImage> read_exr(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return make_error(ErrorCode::NotFound, "The EXR file could not be opened.", path.string());
    const std::vector<std::uint8_t> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto bad = [&](const char* why) { return make_error(ErrorCode::ParseError, "The EXR file is not readable.", path.string() + ": " + why); };
    std::size_t pos = 0;
    auto i32 = [&](std::size_t at) {
        std::int32_t v;
        std::memcpy(&v, raw.data() + at, 4);
        return v;
    };
    if (raw.size() < 8 || i32(0) != kMagic) return bad("not an EXR file");
    if (i32(4) & 0x1a00) return bad("tiled, deep or multi-part");
    pos = 8;
    auto cstr = [&](std::size_t& at) {
        const std::size_t start = at;
        while (at < raw.size() && raw[at] != 0) ++at;
        std::string s(raw.begin() + std::ptrdiff_t(start), raw.begin() + std::ptrdiff_t(at));
        ++at;
        return s;
    };
    ExrImage img;
    std::vector<std::string> channels;
    int x0 = 0, y0 = 0, x1 = -1, y1 = -1, compression = 0;
    while (pos < raw.size() && raw[pos] != 0) {
        const std::string name = cstr(pos), type = cstr(pos);
        const auto size = static_cast<std::size_t>(i32(pos));
        pos += 4;
        if (pos + size > raw.size()) return bad("truncated header");
        std::vector<std::uint8_t> payload(raw.begin() + std::ptrdiff_t(pos), raw.begin() + std::ptrdiff_t(pos + size));
        if (type == "chlist") {
            std::size_t c = pos;
            while (raw[c] != 0) {
                channels.push_back(cstr(c));
                img.pixel_types.push_back(i32(c));
                c += 16;
            }
        } else if (type == "compression") {
            compression = payload[0];
        } else if (name == "dataWindow") {
            x0 = i32(pos); y0 = i32(pos + 4); x1 = i32(pos + 8); y1 = i32(pos + 12);
        }
        img.attribute_types.emplace_back(name, type);
        img.attributes.emplace_back(name, std::move(payload));
        pos += size;
    }
    ++pos;
    if (compression != 0) return bad("compressed");
    const int w = x1 - x0 + 1, h = y1 - y0 + 1;
    if (w <= 0 || h <= 0 || channels.empty()) return bad("empty data window");
    pos += 8 * std::size_t(h);
    static const char* order[] = {"R", "G", "B", "A"};
    const int nout = int(std::count(channels.begin(), channels.end(), std::string("A"))) ? 4 : 3;
    img.pixels = PlanarBuffer(nout, h, w);
    const bool all_half = std::all_of(img.pixel_types.begin(), img.pixel_types.end(), [](int t) { return t == 1; });
    if (all_half) img.half_bits.assign(std::size_t(nout) * std::size_t(w) * std::size_t(h), 0);
    for (int r = 0; r < h; ++r) {
        if (pos + 8 > raw.size()) return bad("truncated data");
        const int y = i32(pos) - y0;
        pos += 8;
        for (std::size_t c = 0; c < channels.size(); ++c) {
            int plane = -1;
            for (int k = 0; k < nout; ++k)
                if (channels[c] == order[k]) plane = k;
            const bool half = img.pixel_types[c] == 1;
            const std::size_t bytes = std::size_t(w) * (half ? 2 : 4);
            if (pos + bytes > raw.size()) return bad("truncated data");
            if (plane >= 0) {
                for (int x = 0; x < w; ++x) {
                    float v;
                    if (half) {
                        std::uint16_t hv;
                        std::memcpy(&hv, raw.data() + pos + std::size_t(x) * 2, 2);
                        v = half_to_float(hv);
                        if (all_half) img.half_bits[(std::size_t(plane) * std::size_t(h) + std::size_t(y)) * std::size_t(w) + std::size_t(x)] = hv;
                    } else {
                        std::memcpy(&v, raw.data() + pos + std::size_t(x) * 4, 4);
                    }
                    img.pixels.at(plane, y, x) = v;
                }
            }
            pos += bytes;
        }
    }
    return img;
}

std::string ocio_config_text() {
    auto matrix_yaml = [](const Mat3& m) {
        std::string s = std::string(8, ' ') + "- !<MatrixTransform> {matrix: [";
        const double m4[16] = {m[0][0], m[0][1], m[0][2], 0, m[1][0], m[1][1], m[1][2], 0,
                               m[2][0], m[2][1], m[2][2], 0, 0, 0, 0, 1};
        for (int i = 0; i < 16; ++i) {
            char b[64];
            std::snprintf(b, sizeof b, "%.10f", m4[i]);
            s += b;
            if (i < 15) s += ", ";
        }
        return s + "]}";
    };
    struct Space { const char* name; Primaries src; const char* label; };
    const Space spaces[] = {{"RUDRA Scene Linear (Rec.2020)", Primaries::Rec2020, "Rec.2020"},
                            {"Linear Rec.709", Primaries::Rec709, "Linear Rec.709"},
                            {"ACEScg", Primaries::Ap1, "ACEScg"}};
    std::string blocks;
    for (const auto& sp : spaces) {
        blocks += "  - !<ColorSpace>\n    name: " + std::string(sp.name) +
                  "\n    family: Scene\n    encoding: scene-linear\n    bitdepth: 32f\n    description: |\n      Linear " +
                  sp.label +
                  " to ACES2065-1 via Bradford CAT.\n      Generated by rudra.delivery.aces; matrices derived from "
                  "primaries, not copied.\n    to_scene_reference: !<GroupTransform>\n      children:\n" +
                  matrix_yaml(rgb_to_rgb_matrix(sp.src, Primaries::Ap0)) + "\n";
    }
    return std::string(
           "ocio_profile_version: 2\n"
           "\n"
           "description: RUDRA delivery working spaces (pair with the official ACES OCIO config for display transforms)\n"
           "name: rudra-delivery\n"
           "\n"
           "roles:\n"
           "  scene_linear: RUDRA Scene Linear (Rec.2020)\n"
           "  aces_interchange: ACES2065-1\n"
           "  default: RUDRA Scene Linear (Rec.2020)\n"
           "  data: Raw\n"
           "\n"
           "file_rules:\n"
           "  - !<Rule> {name: Default, colorspace: default}\n"
           "\n"
           "colorspaces:\n"
           "  - !<ColorSpace>\n"
           "    name: ACES2065-1\n"
           "    family: Scene\n"
           "    encoding: scene-linear\n"
           "    bitdepth: 32f\n"
           "    description: SMPTE ST 2065-1 AP0 primaries, ACES white. Reference space.\n"
           "\n"
           "  - !<ColorSpace>\n"
           "    name: Raw\n"
           "    family: Utility\n"
           "    encoding: data\n"
           "    isdata: true\n"
           "\n") +
           blocks;
}

}  // namespace rudra
