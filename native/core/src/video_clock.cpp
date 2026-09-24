#include "rudra/core/video_clock.hpp"

#include <charconv>
#include <numeric>

namespace rudra {

namespace {
bool parse_int(std::string_view s, std::int64_t& out) {
    if (!s.empty() && s.front() == '+') s.remove_prefix(1);
    const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
    return ec == std::errc() && p == s.data() + s.size() && !s.empty();
}

}  // namespace

std::string Fraction::str() const {
    return den == 1 ? std::to_string(num) : std::to_string(num) + "/" + std::to_string(den);
}

double Fraction::value() const {
    // Both parts are exact doubles for any rate or time base ffprobe prints, and
    // IEEE division is correctly rounded, as float(Fraction) is.
    return static_cast<double>(num) / static_cast<double>(den);
}

Result<Fraction> parse_fraction(std::string_view text) {
    while (!text.empty() && text.front() == ' ') text.remove_prefix(1);
    while (!text.empty() && text.back() == ' ') text.remove_suffix(1);
    std::int64_t num = 0, den = 1;
    if (const auto slash = text.find('/'); slash != std::string_view::npos) {
        if (!parse_int(text.substr(0, slash), num) || !parse_int(text.substr(slash + 1), den))
            return make_error(ErrorCode::ParseError, "Invalid literal for Fraction: '" + std::string(text) + "'");
    } else if (const auto dot = text.find('.'); dot != std::string_view::npos) {
        std::string digits(text.substr(0, dot));
        const std::string_view frac = text.substr(dot + 1);
        digits += frac;
        if (frac.size() > 18 || !parse_int(digits, num))
            return make_error(ErrorCode::ParseError, "Invalid literal for Fraction: '" + std::string(text) + "'");
        den = 1;
        for (std::size_t i = 0; i < frac.size(); ++i) den *= 10;
    } else if (!parse_int(text, num)) {
        return make_error(ErrorCode::ParseError, "Invalid literal for Fraction: '" + std::string(text) + "'");
    }
    if (den == 0) return make_error(ErrorCode::InvalidArgument, "Fraction(" + std::to_string(num) + ", 0)");
    if (den < 0) num = -num, den = -den;
    const std::int64_t g = std::gcd(num < 0 ? -num : num, den);
    if (g > 1) num /= g, den /= g;
    return Fraction{num, den};
}

}  // namespace rudra
