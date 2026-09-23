#version 440
// The composite on the GPU (docs/composite.spec.md section 2). A line-for-line
// port of native/core/src/composite.cpp, which is the reference it is held to
// (rudra-gpu-parity), and of ui/compositor.js, which it replaces.
//
// Inputs are fetched by pixel, never filtered: texture row r, framebuffer row r
// and readback row r are the same pixel on every API.

layout(location = 0) out vec4 frag;

layout(std140, binding = 0) uniform Composite {
    vec4 model;     // x log_scale, y max_hdr, z baseline scale (2^-ev * 203/10000), w strength
    vec4 control;   // x mode (0 all, 1 highlights, 2 shadows, 3 off), y preserve, z shadow weight, w region softness (stops)
    vec4 counts;    // x curve knots (0: no curve), y region bands
    vec4 curve[9];  // curve params, packed four to a vec4: [0] exposure, [1..] knots
    vec4 bands[8];  // x log2(low nits), y log2(high nits), z ev
};

layout(binding = 1) uniform sampler2D sdrShadow;   // rgb SDR code values, a shadow mask
layout(binding = 2) uniform sampler2D fields;      // rgb log residual (scale folded in), a highlight mask

float curveParam(int i) { return curve[i >> 2][i & 3]; }

float srgbToLinear(float x) {
    x = clamp(x, 0.0, 1.0);
    return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}

// inverse_aces_approx: the larger root, input clamped just below one.
float inverseAces(float y) {
    y = clamp(y, 0.0, 0.995);
    float qa = y * 2.43 - 2.51;
    float qb = y * 0.59 - 0.03;
    float qc = y * 0.14;
    float disc = max(qb * qb - 4.0 * qa * qc, 0.0);
    float den = min(2.0 * qa, -1e-7);
    float s = sqrt(disc);
    return max(max((-qb - s) / den, (-qb + s) / den), 0.0);
}

float curveLog2(float code) {
    int knots = int(counts.x);
    float pos = clamp(code, 0.0, 1.0) * float(knots - 1);
    int lo = min(int(floor(pos)), knots - 2);
    float frac = pos - float(lo);
    float v0 = curveParam(1 + lo);
    float v1 = curveParam(2 + lo);
    return curveParam(0) + v0 + (v1 - v0) * frac;
}

float baselineOf(float code) {
    float b = inverseAces(srgbToLinear(code)) * model.z;
    return counts.x >= 2.0 ? b * exp2(curveLog2(code)) : b;
}

// log1p / expm1 without the cancellation of log(1 + x) near zero (Kahan).
float log1pAccurate(float x) {
    float u = 1.0 + x;
    return u == 1.0 ? x : log(u) * (x / (u - 1.0));
}
float expm1Accurate(float x) {
    float u = exp(x);
    if (u == 1.0) return x;
    float um1 = u - 1.0;
    return um1 == -1.0 ? -1.0 : um1 * (x / log(u));
}

float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

// qualifier_mask and region_ev_gain (rudra/delivery/controls.py), in fp32.
float regionGain(vec3 nits) {
    float y = max(dot(max(nits, vec3(0.0)), vec3(0.2627, 0.6780, 0.0593)), 1e-6);
    float logY = log2(y);
    float soft = max(control.w, 1e-3);
    float total = 0.0;
    int n = int(counts.y);
    for (int i = 0; i < n; ++i) {
        float rise = clamp((logY - (bands[i].x - soft)) / soft, 0.0, 1.0);
        float fall = clamp(((bands[i].y + soft) - logY) / soft, 0.0, 1.0);
        float m = min(rise, fall);
        total += bands[i].z * (m * m * (3.0 - 2.0 * m));
    }
    return exp2(total);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 a = texelFetch(sdrShadow, p, 0);
    vec4 f = texelFetch(fields, p, 0);
    vec3 s = clamp(a.rgb, 0.0, 1.0);
    float shadow = a.a;
    float highlight = f.a;

    float logScale = model.x;
    float y = 0.2126 * s.r + 0.7152 * s.g + 0.0722 * s.b;
    float hp = sigmoid((y - 0.82) * 24.0);
    float sp = sigmoid((0.10 - y) * 24.0) * control.z;
    int mode = int(control.x);
    float gate = mode == 0 ? max(hp, sp) : mode == 1 ? hp : mode == 2 ? sp : 0.0;
    gate *= model.w;
    float recovery = max(highlight, shadow);
    float logCeiling = log1pAccurate(model.y * logScale);

    vec3 pred;
    for (int c = 0; c < 3; ++c) {
        float base = baselineOf(s[c]);
        float pl = clamp(log1pAccurate(base * logScale) + f[c] * gate, 0.0, logCeiling);
        float v = expm1Accurate(pl) / logScale;
        if (control.y > 0.5) v = base + recovery * (v - base);
        pred[c] = v;
    }
    if (counts.y > 0.0) {
        vec3 nits = pred * 10000.0;
        pred = clamp(nits * regionGain(nits), vec3(0.0), vec3(model.y * 10000.0)) / 10000.0;
    }
    frag = vec4(pred, 1.0);
}
