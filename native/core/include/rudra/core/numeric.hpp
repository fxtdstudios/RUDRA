#pragma once
// numpy's summation, reproduced: np.sum and np.mean of a contiguous float64
// array use pairwise summation (numpy/_core/src/umath/loops_utils.h), so a
// mean computed any other way differs in the last digits, and a JSON sidecar
// that prints 17 significant digits would differ in bytes. Checked against
// numpy 2.x on 5 to 100 003 elements.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace rudra {

inline double np_pairwise_sum(std::span<const double> a) noexcept {
    const std::size_t n = a.size();
    if (n < 8) {
        double r = 0.0;
        for (double v : a) r += v;
        return r;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[std::size_t(j)];
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + std::size_t(j)];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return np_pairwise_sum(a.first(n2)) + np_pairwise_sum(a.subspan(n2));
}

inline double np_mean(std::span<const double> a) noexcept {
    if (a.empty()) return std::numeric_limits<double>::quiet_NaN();
    return np_pairwise_sum(a) / static_cast<double>(a.size());
}

// np.std (ddof 0): mean, then the mean of squared deviations, both pairwise.
inline double np_std(std::span<const double> a) {
    if (a.empty()) return std::numeric_limits<double>::quiet_NaN();
    const double m = np_mean(a);
    std::vector<double> d(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) d[i] = (a[i] - m) * (a[i] - m);
    return std::sqrt(np_mean(d));
}

inline bool any_nan(std::span<const double> a) noexcept {
    return std::any_of(a.begin(), a.end(), [](double v) { return std::isnan(v); });
}

// np.median: NaN if any NaN; the mean of the two middle values for an even count.
inline double np_median(std::vector<double> v) {
    if (v.empty() || any_nan(v)) return std::numeric_limits<double>::quiet_NaN();
    const std::size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(mid), v.end());
    const double hi = v[mid];
    if (v.size() % 2) return hi;
    const double lo = *std::max_element(v.begin(), v.begin() + std::ptrdiff_t(mid));
    return (lo + hi) / 2.0;
}

// np.percentile(v, q), linear method; NaN if any NaN.
inline double np_percentile(std::vector<double> v, double q) {
    if (v.empty() || any_nan(v)) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    const double idx = (q / 100.0) * static_cast<double>(v.size() - 1);
    const double lo_f = std::floor(idx);
    const std::size_t lo = static_cast<std::size_t>(lo_f), hi = std::min(lo + 1, v.size() - 1);
    const double t = idx - lo_f, d = v[hi] - v[lo];
    return t >= 0.5 ? v[hi] - d * (1.0 - t) : v[lo] + d * t;
}

// np.linspace(start, stop, num): i * step + start, the last one exactly stop.
inline std::vector<double> np_linspace(double start, double stop, std::size_t num) {
    std::vector<double> y(num);
    const double step = (stop - start) / static_cast<double>(num - 1);
    for (std::size_t i = 0; i < num; ++i) y[i] = static_cast<double>(i) * step + start;
    if (num > 1) y[num - 1] = stop;
    return y;
}

// np.gradient(f, x) in 1-D for increasing, non-uniform x, edge_order 1.
inline std::vector<double> np_gradient(std::span<const double> f, std::span<const double> x) {
    const std::size_t n = f.size();
    std::vector<double> out(n);
    for (std::size_t i = 1; i + 1 < n; ++i) {
        const double dx1 = x[i] - x[i - 1], dx2 = x[i + 1] - x[i];
        const double a = -(dx2) / (dx1 * (dx1 + dx2));
        const double b = (dx2 - dx1) / (dx1 * dx2);
        const double c = dx1 / (dx2 * (dx1 + dx2));
        out[i] = a * f[i - 1] + b * f[i] + c * f[i + 1];
    }
    out[0] = (f[1] - f[0]) / (x[1] - x[0]);
    out[n - 1] = (f[n - 1] - f[n - 2]) / (x[n - 1] - x[n - 2]);
    return out;
}

// np.interp(x, xp, fp): linear, clamped to the end values.
inline double np_interp(double x, std::span<const double> xp, std::span<const double> fp) {
    const std::size_t n = xp.size();
    if (x < xp[0]) return fp[0];
    if (x > xp[n - 1]) return fp[n - 1];
    if (x == xp[n - 1]) return fp[n - 1];
    const std::size_t j = static_cast<std::size_t>(std::upper_bound(xp.begin(), xp.end(), x) - xp.begin()) - 1;
    if (x == xp[j]) return fp[j];
    const double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    double r = slope * (x - xp[j]) + fp[j];
    if (std::isnan(r)) {
        r = slope * (x - xp[j + 1]) + fp[j + 1];
        if (std::isnan(r) && fp[j] == fp[j + 1]) r = fp[j];
    }
    return r;
}

}  // namespace rudra
