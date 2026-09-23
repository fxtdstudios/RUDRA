#version 440
// The display pass (docs/view.spec.md section 2). A line-for-line port of
// native/core/src/view.cpp, the reference it is held to (rudra-gpu-parity),
// and of the DISPLAY shader in ui/compositor.js, which it replaces.
//
// Texels are fetched by pixel, so output row r is image row r on every API,
// as in composite.frag. Presenting to a swapchain puts the flip, if the API
// needs one, in the vertex stage (render/, Phase 2 step 9), never here.

layout(location = 0) out vec4 frag;

layout(std140, binding = 0) uniform View {
    vec4 view;    // x mode (0 image, 1 false colour, 2 difference), y exposure (10000 / view peak), z wipe (< 0 off), w wipe half width
    vec4 extra;   // x log2(1 + diff gain), y frame width, z 1 to show the baseline, w unused
    vec4 target;  // x path (0 SDR, 1 scRGB, 2 HDR10, 3 EDR), y ceiling nits, z nits per 1.0 (linear paths)
    vec4 pic[3];  // rows of source primaries -> swapchain primaries (xyz)
    vec4 gfx[3];  // rows of Rec.709 -> swapchain primaries, for the overlays
};

layout(binding = 1) uniform sampler2D model;      // rgb reconstruction, network units
layout(binding = 2) uniform sampler2D baseline;   // rgb analytic baseline

const float kPeak = 10000.0;

float lum2020(vec3 c) { return dot(c, vec3(0.2627, 0.6780, 0.0593)); }

float linearToSrgb(float x) {
    x = clamp(x, 0.0, 1.0);
    return x > 0.0031308 ? 1.055 * pow(x, 1.0 / 2.4) - 0.055 : 12.92 * x;
}

float srgbToLinear(float c) {
    c = clamp(c, 0.0, 1.0);
    return c > 0.04045 ? pow((c + 0.055) / 1.055, 2.4) : c / 12.92;
}

// SMPTE ST 2084 inverse EOTF of absolute nits, as core/hdr10.cpp pq_oetf.
float pq(float nits) {
    const float m1 = 0.1593017578125, m2 = 78.84375;
    const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    float p = pow(clamp(nits / 10000.0, 0.0, 1.0), m1);
    return pow((c1 + c2 * p) / (1.0 + c3 * p), m2);
}

float encodeNits(float n) { return int(target.x + 0.5) == 2 ? pq(n) : n / target.z; }

vec3 falseColour(float n) {
    if (n <     0.1) return vec3(0.169, 0.122, 0.239);
    if (n <     1.0) return vec3(0.184, 0.294, 0.561);
    if (n <    10.0) return vec3(0.184, 0.561, 0.722);
    if (n <   100.0) return vec3(0.200, 0.627, 0.416);
    if (n <   160.0) return vec3(0.604, 0.655, 0.698);
    if (n <   250.0) return vec3(0.914, 0.929, 0.945);
    if (n <  1000.0) return vec3(0.910, 0.765, 0.290);
    if (n <  4000.0) return vec3(0.910, 0.529, 0.227);
    if (n < 10000.0) return vec3(0.816, 0.263, 0.184);
    return vec3(0.780, 0.290, 0.780);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float u = (float(p.x) + 0.5) / extra.y;
    bool wiping = view.z >= 0.0;
    vec3 b = texelFetch(baseline, p, 0).rgb;
    vec3 src = (!wiping && extra.z > 0.5) ? b : texelFetch(model, p, 0).rgb;
    vec3 h = (wiping && u < view.z) ? b : src;
    int mode = int(view.x + 0.5);
    vec3 c;
    if (mode == 1) {
        c = falseColour(lum2020(h) * kPeak);
    } else if (mode == 2) {
        float d = lum2020(abs(src - b));
        float v = clamp(log2(1.0 + d * kPeak) / extra.x, 0.0, 1.0);
        c = v * vec3(0.95, 0.62, 0.28);
    } else if (target.x < 0.5) {
        c = vec3(linearToSrgb(h.r * view.y), linearToSrgb(h.g * view.y), linearToSrgb(h.b * view.y));
    } else {
        c = clamp(h * kPeak, 0.0, target.y);
    }
    bool handle = wiping && abs(u - view.z) < view.w;
    if (target.x < 0.5) {
        if (handle) c = vec3(1.0) - c;
        frag = vec4(c, 1.0);
        return;
    }
    // HDR: the picture in nits (source primaries), or an overlay at the SDR white.
    vec3 n;
    vec4 r0, r1, r2;
    if (mode == 0) {
        n = handle ? vec3(target.y) - c : c;
        r0 = pic[0]; r1 = pic[1]; r2 = pic[2];
    } else {
        vec3 g = handle ? vec3(1.0) - c : c;
        n = 203.0 * vec3(srgbToLinear(g.r), srgbToLinear(g.g), srgbToLinear(g.b));
        r0 = gfx[0]; r1 = gfx[1]; r2 = gfx[2];
    }
    // Row by row in the order core/view.cpp sums them.
    vec3 o = vec3(r0.x * n.x + r0.y * n.y + r0.z * n.z,
                  r1.x * n.x + r1.y * n.y + r1.z * n.z,
                  r2.x * n.x + r2.y * n.y + r2.z * n.z);
    frag = vec4(encodeNits(o.r), encodeNits(o.g), encodeNits(o.b), 1.0);
}
