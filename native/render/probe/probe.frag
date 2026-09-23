#version 440
// Gate B test card (docs/NATIVE_ARCHITECTURE.md section 12, day 5).
// Top: six flat patches at known absolute luminance. Bottom: a log ramp from
// 0.05 to 10 000 nits with black ticks at 203 nits (SDR white), 1 000 nits and
// the peak the display reports. Nothing is tone-mapped (ADR-005): above the
// display's peak the patches clip, which is the point.
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag;

layout(std140, binding = 0) uniform Probe {
    mat4 clip_corr;
    vec4 encode;
};

const float kPatch[6] = float[6](100.0, 203.0, 400.0, 600.0, 1000.0, 2000.0);
const float kSurround = 1.0;           // nits: the card's dark grey
const float kRampLo = -4.321928;       // log2(0.05)
const float kRampHi = 13.287712;       // log2(10000)

float rampX(float nits) { return (log2(nits) - kRampLo) / (kRampHi - kRampLo); }

float nitsAt(vec2 uv) {
    if (uv.y < 0.62) {
        float cell = uv.x * 6.0;
        int i = clamp(int(floor(cell)), 0, 5);
        float fx = fract(cell);
        float fy = uv.y / 0.62;
        return (fx > 0.1 && fx < 0.9 && fy > 0.15 && fy < 0.9) ? kPatch[i] : kSurround;
    }
    if (uv.y < 0.68) return kSurround;
    float tick = 0.0015;
    if (abs(uv.x - rampX(203.0)) < tick || abs(uv.x - rampX(1000.0)) < tick) return 0.0;
    if (encode.z > 0.0 && abs(uv.x - rampX(encode.z)) < tick && uv.y < 0.74) return 0.0;
    return exp2(mix(kRampLo, kRampHi, uv.x));
}

// SMPTE ST 2084 inverse EOTF, input in [0,1] of 10 000 nits.
float pq(float y) {
    const float m1 = 0.1593017578125, m2 = 78.84375;
    const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    float p = pow(clamp(y, 0.0, 1.0), m1);
    return pow((c1 + c2 * p) / (1.0 + c3 * p), m2);
}

float srgbOetf(float v) {
    return v <= 0.0031308 ? v * 12.92 : 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

void main() {
    float nits = nitsAt(v_uv);
    float v;
    if (encode.y < 0.5) v = nits * encode.x;
    else if (encode.y < 1.5) v = pq(nits / 10000.0);
    else v = srgbOetf(clamp(nits / 203.0, 0.0, 1.0));
    frag = vec4(v, v, v, 1.0);
}
