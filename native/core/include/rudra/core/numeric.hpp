#pragma once
// numpy's summation, reproduced: np.sum and np.mean of a contiguous float64
// array use pairwise summation (numpy/_core/src/umath/loops_utils.h), so a
// mean computed any other way differs in the last digits, and a JSON sidecar
// that prints 17 significant digits would differ in bytes. Checked against
// numpy 2.x on 5 to 100 003 elements.

#include <cstddef>
#include <span>

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
    return a.empty() ? 0.0 : np_pairwise_sum(a) / static_cast<double>(a.size());
}

}  // namespace rudra
