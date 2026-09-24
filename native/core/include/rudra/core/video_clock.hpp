#pragma once
// A clip's clock as rudra/video.py timing() returns it: the rational rate as
// str(Fraction), the frame count, the first timestamp, the duration and the
// timestamp tolerance. Read by media (the probe) and deliver (the encoder, QC).

#include <string>

namespace rudra {

struct VideoClock {
    std::string fps;                           // str(Fraction): "24000/1001" or "25"
    int frames = 0;
    double start = 0, duration = 0, tolerance = 0;
};

}  // namespace rudra
