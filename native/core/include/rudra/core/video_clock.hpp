#pragma once
// A clip's clock as rudra/video.py timing() returns it: the rational rate as
// str(Fraction), the frame count, the first timestamp, the duration and the
// timestamp tolerance. Read by media (the probe) and deliver (the encoder, QC).
// And Python's Fraction, which both read rates and time bases with.

#include <cstdint>
#include <string>
#include <string_view>

#include "rudra/platform/result.hpp"

namespace rudra {

struct VideoClock {
    std::string fps;                           // str(Fraction): "24000/1001" or "25"
    int frames = 0;
    double start = 0, duration = 0, tolerance = 0;
};


// A reduced fraction as Python's Fraction reads "n/d", "n" or a decimal.
struct Fraction {
    std::int64_t num = 0, den = 1;
    std::string str() const;                   // str(Fraction)
    double value() const;                      // float(Fraction), correctly rounded
};
Result<Fraction> parse_fraction(std::string_view text);

}  // namespace rudra
