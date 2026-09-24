#include "rudra/core/js_format.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <string>

namespace rudra {

std::string js_number(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
    if (v == 0.0) return "0";   // -0 prints as 0 too
    // The shortest round-trip digits, as d.ddde±x.
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
    std::string sci(buf, r.ptr);
    const bool neg = sci[0] == '-';
    if (neg) sci.erase(0, 1);
    const auto e = sci.find('e');
    const int exp = std::atoi(sci.c_str() + e + 1);
    std::string digits;
    for (std::size_t i = 0; i < e; ++i)
        if (sci[i] != '.') digits += sci[i];
    const int k = int(digits.size());
    const int n = exp + 1;   // the decimal point sits after n digits
    std::string out;
    if (k <= n && n <= 21) {
        out = digits + std::string(std::size_t(n - k), '0');
    } else if (0 < n && n <= 21) {
        out = digits.substr(0, std::size_t(n)) + "." + digits.substr(std::size_t(n));
    } else if (-6 < n && n <= 0) {
        out = "0." + std::string(std::size_t(-n), '0') + digits;
    } else {
        out = digits.substr(0, 1);
        if (k > 1) out += "." + digits.substr(1);
        out += (n - 1 >= 0 ? "e+" : "e-") + std::to_string(std::abs(n - 1));
    }
    return neg ? "-" + out : out;
}

std::string js_to_fixed(double v, int digits) {
    if (std::isnan(v)) return "NaN";
    if (std::abs(v) >= 1e21 || std::isinf(v)) return js_number(v);
    const bool neg = v < 0.0;   // -0 is not below 0: (-0).toFixed(1) is "0.0"
    const double a = std::abs(v);   // and -0 prints unsigned
    // The exact decimal expansion (a double's is finite), far enough to decide.
    char buf[1600];
    const auto r = std::to_chars(buf, buf + sizeof buf, a, std::chars_format::fixed, digits + 1100);
    std::string s(buf, r.ptr);
    const auto dot = s.find('.');
    std::string whole = s.substr(0, dot);
    std::string frac = s.substr(dot + 1);
    std::string keep = whole + frac.substr(0, std::size_t(digits));
    // The nearest n, the larger of two: halves go up.
    if (frac[std::size_t(digits)] >= '5') {
        int i = int(keep.size()) - 1;
        while (i >= 0 && keep[std::size_t(i)] == '9') keep[std::size_t(i--)] = '0';
        if (i < 0) keep.insert(keep.begin(), '1');
        else ++keep[std::size_t(i)];
    }
    const std::size_t int_len = keep.size() - std::size_t(digits);
    std::string out = keep.substr(0, int_len);
    if (digits > 0) out += "." + keep.substr(int_len);
    return neg ? "-" + out : out;
}

double js_round(double v) { return std::floor(v + 0.5); }

std::string js_fmt(double v, int digits) {
    if (!std::isfinite(v)) return "\u2014";
    // Intl.NumberFormat rounds the number's shortest decimal form (the digits
    // String(x) prints), halves away from zero: 99.9995 to three places is
    // 100.000, where toFixed, on the exact binary value 99.99949..., says 99.999.
    const bool neg = std::signbit(v);
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof buf, std::abs(v), std::chars_format::scientific);
    std::string sci(buf, r.ptr);
    const auto e = sci.find('e');
    const int exp = std::atoi(sci.c_str() + e + 1);
    std::string d;
    for (std::size_t i = 0; i < e; ++i)
        if (sci[i] != '.') d += sci[i];
    // d is the digits of v = 0.d * 10^(exp + 1); line them up as whole.frac.
    std::string whole, frac;
    const int point = exp + 1;
    if (point <= 0) {
        whole = "0";
        frac = std::string(std::size_t(-point), '0') + d;
    } else if (point >= int(d.size())) {
        whole = d + std::string(std::size_t(point - int(d.size())), '0');
    } else {
        whole = d.substr(0, std::size_t(point));
        frac = d.substr(std::size_t(point));
    }
    if (int(frac.size()) < digits + 1) frac += std::string(std::size_t(digits + 1 - int(frac.size())), '0');
    std::string keep = whole + frac.substr(0, std::size_t(digits));
    if (frac[std::size_t(digits)] >= '5') {
        int i = int(keep.size()) - 1;
        while (i >= 0 && keep[std::size_t(i)] == '9') keep[std::size_t(i--)] = '0';
        if (i < 0) keep.insert(keep.begin(), '1');
        else ++keep[std::size_t(i)];
    }
    const std::size_t int_len = keep.size() - std::size_t(digits);
    std::string w = keep.substr(0, int_len);
    while (w.size() > 1 && w[0] == '0') w.erase(0, 1);
    for (int i = int(w.size()) - 3; i > 0; i -= 3) w.insert(std::size_t(i), ",");
    std::string out = w;
    if (digits > 0) out += "." + keep.substr(int_len);
    return (neg ? "-" : "") + out;
}

std::string js_signed(double v, int digits) {
    if (!std::isfinite(v)) return "\u2014";
    return std::string(v >= 0 ? "+" : "\u2212") + js_fmt(std::abs(v), digits);
}

std::string nits_label(double v) { return v >= 1000 ? js_number(v / 1000) + "k" : js_number(v); }

}  // namespace rudra
