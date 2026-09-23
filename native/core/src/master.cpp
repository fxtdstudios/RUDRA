#include "rudra/core/master.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "rudra/core/gamut.hpp"

namespace rudra {
namespace {

// rudra/anchor.py and rudra/chroma.py use float64 Rec.2020 weights.
constexpr double kLuma2020[3] = {0.2627, 0.6780, 0.0593};
constexpr double kDiffuseWhiteNits = 203.0;

double srgb_to_linear_d(double v) noexcept {
    return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
}

double smoothstep_ramp(double code, double knee, double softness) noexcept {
    const double r = std::clamp((code - (knee - softness)) / (2.0 * softness), 0.0, 1.0);
    return r * r * (3.0 - 2.0 * r);
}

double finite_or(double v, double fallback) noexcept { return std::isfinite(v) ? v : fallback; }

// np.median: the mean of the two middle values for an even count.
double median(std::vector<double> v) {
    assert(!v.empty());
    const std::size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    const double hi = v[mid];
    if (v.size() % 2 == 1) return hi;
    const double lo = *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid));
    return (lo + hi) / 2.0;
}

}  // namespace

NitsFrame nits_from_network(const NetworkLinearImage& network) {
    NitsFrame out(network.height(), network.width());
    const auto in = network.buffer().span();
    auto o = out.span();
    for (std::size_t i = 0; i < in.size(); ++i) o[i] = static_cast<double>(in[i]) * 10000.0;
    return out;
}

void apply_region_ev(NitsFrame& nits, std::span<const RegionBand> bands, double softness_stops, double ceiling_nits) {
    if (!any_graded(bands)) return;
    const std::size_t n = nits.plane_size();
    for (std::size_t i = 0; i < n; ++i) {
        double px[3] = {nits.plane(0)[i], nits.plane(1)[i], nits.plane(2)[i]};
        const double g = region_ev_gain(px, bands, softness_stops);
        for (int c = 0; c < 3; ++c) nits.plane(c)[i] = std::clamp(px[c] * g, 0.0, ceiling_nits);
    }
}

void anchor_to_sdr(NitsFrame& nits, const SdrImage& sdr, double knee, double softness) {
    assert(nits.height() == sdr.height() && nits.width() == sdr.width());
    assert(knee > 0.0 && knee < 1.0);
    const std::size_t n = nits.plane_size();
    const PlanarBuffer& s = sdr.buffer();
    constexpr double eps = 1e-4;   // nits
    std::vector<double> want(n), code(n);
    std::vector<double> band;
    for (std::size_t i = 0; i < n; ++i) {
        double target = 0.0, actual = 0.0, mx = -1.0;
        for (int c = 0; c < 3; ++c) {
            const double v = s.plane(c)[i];
            target += srgb_to_linear_d(v) * kLuma2020[c];
            actual += nits.plane(c)[i] * kLuma2020[c];
            mx = std::max(mx, v);
        }
        target *= kDiffuseWhiteNits;
        want[i] = (target + eps) / (actual + eps);
        code[i] = mx;
        if (mx > knee - softness && mx < knee + softness && actual > 1e-9) band.push_back(want[i]);
    }
    const double hold = band.size() >= 64 ? median(std::move(band)) : median(want);
    for (std::size_t i = 0; i < n; ++i) {
        double gain = want[i];
        if (code[i] > knee - softness) {
            const double r = smoothstep_ramp(code[i], knee, softness);
            gain = want[i] * (1.0 - r) + hold * r;
        }
        gain = finite_or(gain, 1.0);
        for (int c = 0; c < 3; ++c) nits.plane(c)[i] *= gain;
    }
}

std::vector<float> gaussian_blur_replicate(std::span<const float> plane, int height, int width, double sigma) {
    // cv::getGaussianKernel(ksize, sigma, CV_32F): taps in float, summed in
    // double, normalised, stored as float.
    const int ksize = static_cast<int>(std::lround(sigma * 4.0 * 2.0 + 1.0)) | 1;
    const int radius = ksize / 2;
    std::vector<float> k(static_cast<std::size_t>(ksize));
    const double scale2x = -0.5 / (sigma * sigma);
    double sum = 0.0;
    for (int i = 0; i < ksize; ++i) {
        const double x = i - (ksize - 1) * 0.5;
        k[static_cast<std::size_t>(i)] = static_cast<float>(std::exp(scale2x * x * x));
        sum += k[static_cast<std::size_t>(i)];
    }
    sum = 1.0 / sum;
    for (float& t : k) t = static_cast<float>(t * sum);

    auto clampi = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };
    std::vector<float> rows(plane.size()), out(plane.size());
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            double acc = 0.0;
            for (int t = -radius; t <= radius; ++t)
                acc += double(k[static_cast<std::size_t>(t + radius)]) *
                       plane[static_cast<std::size_t>(y) * width + clampi(x + t, 0, width - 1)];
            rows[static_cast<std::size_t>(y) * width + x] = static_cast<float>(acc);
        }
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            double acc = 0.0;
            for (int t = -radius; t <= radius; ++t)
                acc += double(k[static_cast<std::size_t>(t + radius)]) *
                       rows[static_cast<std::size_t>(clampi(y + t, 0, height - 1)) * width + x];
            out[static_cast<std::size_t>(y) * width + x] = static_cast<float>(acc);
        }
    return out;
}

void carry_source_chroma(NitsFrame& nits, const SdrImage& sdr, double knee, double softness, double mask_sigma) {
    assert(nits.height() == sdr.height() && nits.width() == sdr.width());
    assert(knee > 0.0 && knee < 1.0);
    const std::size_t n = nits.plane_size();
    const PlanarBuffer& s = sdr.buffer();
    std::vector<float> ramp(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double code = std::max({double(s.plane(0)[i]), double(s.plane(1)[i]), double(s.plane(2)[i])});
        ramp[i] = static_cast<float>(smoothstep_ramp(code, knee, softness));
    }
    const std::vector<float> blurred = gaussian_blur_replicate(ramp, nits.height(), nits.width(), mask_sigma);
    constexpr double eps = 1e-6;
    for (std::size_t i = 0; i < n; ++i) {
        double source[3], source_luma = 0.0, hdr_luma = 0.0;
        for (int c = 0; c < 3; ++c) {
            source[c] = srgb_to_linear_d(s.plane(c)[i]) * kDiffuseWhiteNits;
            source_luma += source[c] * kLuma2020[c];
            hdr_luma += nits.plane(c)[i] * kLuma2020[c];
        }
        const double k = (hdr_luma + eps) / (source_luma + eps);
        const double r = blurred[i];
        // 1.0 - ramp stays float32 in numpy (the ramp is a float32 array).
        const double keep = static_cast<double>(1.0f - blurred[i]);
        for (int c = 0; c < 3; ++c) {
            const double v = source[c] * k * keep + nits.plane(c)[i] * r;
            nits.plane(c)[i] = finite_or(v, 0.0);
        }
    }
}

PlanarBuffer scene_linear(const NitsFrame& nits) {
    PlanarBuffer out(3, nits.height(), nits.width());
    const auto in = nits.span();
    auto o = out.span();
    for (std::size_t i = 0; i < in.size(); ++i) o[i] = static_cast<float>(in[i] / kDiffuseWhiteNits);
    return out;
}

MasterPixels render_master_pixels(const NetworkLinearImage& network, const SdrImage& sdr,
                                  const ModelConstants& model, const MasterParams& params) {
    MasterPixels m;
    m.nits = nits_from_network(network);
    apply_region_ev(m.nits, params.regions, params.region_softness_stops, double(model.max_hdr) * 10000.0);
    if (params.anchor) anchor_to_sdr(m.nits, sdr, params.anchor_knee);
    if (params.carry_chroma) carry_source_chroma(m.nits, sdr, params.chroma_knee);
    PlanarBuffer linear = scene_linear(m.nits);
    if (params.container == MasterContainer::Aces2065) {
        m.pixels = convert_primaries(linear, params.source_primaries, Primaries::Ap0);
        m.primaries = Primaries::Ap0;
    } else {
        m.pixels = std::move(linear);
        m.primaries = params.source_primaries;
    }
    return m;
}

}  // namespace rudra
