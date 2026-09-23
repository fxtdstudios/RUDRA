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
};

layout(binding = 1) uniform sampler2D model;      // rgb reconstruction, network units
layout(binding = 2) uniform sampler2D baseline;   // rgb analytic baseline

const float kPeak = 10000.0;

float lum2020(vec3 c) { return dot(c, vec3(0.2627, 0.6780, 0.0593)); }

float linearToSrgb(float x) {
    x = clamp(x, 0.0, 1.0);
    return x > 0.0031308 ? 1.055 * pow(x, 1.0 / 2.4) - 0.055 : 12.92 * x;
}

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
    } else {
        c = vec3(linearToSrgb(h.r * view.y), linearToSrgb(h.g * view.y), linearToSrgb(h.b * view.y));
    }
    if (wiping && abs(u - view.z) < view.w) c = vec3(1.0) - c;
    frag = vec4(c, 1.0);
}
