#version 440
// One step of the exact reduction ladder (docs/view.spec.md section 4), the
// REDUCE shader of ui/compositor.js: a 2x2 box of the source, combined by max
// or sum. The first step reads RGB and takes max(R, G, B); later steps read r.
// Run to one texel it gives the exact peak (MaxCLL) and the sum (MaxFALL).

layout(location = 0) out vec4 frag;

layout(std140, binding = 0) uniform Reduce {
    vec4 sizes;   // x, y source size; z op (0 max, 1 sum); w 1 on the first step
};

layout(binding = 1) uniform sampler2D src;

float take(ivec2 p) {
    vec4 t = texelFetch(src, p, 0);
    return sizes.w > 0.5 ? max(max(t.r, t.g), t.b) : t.r;
}

void main() {
    ivec2 s = ivec2(gl_FragCoord.xy) * 2;
    ivec2 size = ivec2(sizes.xy);
    bool isMax = sizes.z < 0.5;
    float acc = isMax ? -1e30 : 0.0;
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
            ivec2 p = s + ivec2(i, j);
            if (p.x >= size.x || p.y >= size.y) continue;
            float v = take(p);
            acc = isMax ? max(acc, v) : acc + v;
        }
    }
    frag = vec4(acc, 0.0, 0.0, 1.0);
}
