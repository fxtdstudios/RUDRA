#version 440
// A copy of the display pass's picture: its values are already encoded for
// the swapchain (docs/view.spec.md sections 2 and 9), so nothing is changed
// here but the sampling (nearest at and above 1:1, trilinear below) and the
// guides (section 11), drawn in device pixels: a port of core/guides.cpp.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag;

layout(std140, binding = 0) uniform Blit {
    mat4 clip_corr;
    vec4 rect;      // left, top, right, bottom in window pixels
    vec4 window;    // width, height, 1 when the framebuffer's y points up, the SDR white encoded
    vec4 guides;    // x action safe, y title safe, z centre (0 or 1), w aspect (0 none)
};

layout(binding = 1) uniform sampler2D picture;

const float kLine = 46.0 / 255.0;
const float kMask = 0.6;
const int kDash = 3;

bool dashOn(int k) { return ((k / kDash) % 2) == 0; }

bool onBox(float l, float t, float w, float h, float inset, int px, int py) {
    float x0 = l + inset * w, x1 = l + (1.0 - inset) * w;
    float y0 = t + inset * h, y1 = t + (1.0 - inset) * h;
    int c0 = int(floor(x0)), c1 = int(ceil(x1)) - 1;
    int r0 = int(floor(y0)), r1 = int(ceil(y1)) - 1;
    if ((px == c0 || px == c1) && py >= r0 && py <= r1) return dashOn(py - r0);
    if ((py == r0 || py == r1) && px >= c0 && px <= c1) return dashOn(px - c0);
    return false;
}

void main() {
    vec3 c = texture(picture, v_uv).rgb;
    float l = rect.x, t = rect.y, w = rect.z - rect.x, h = rect.w - rect.y;
    int px = int(floor(gl_FragCoord.x));
    int py = int(floor(window.z > 0.5 ? window.y - gl_FragCoord.y : gl_FragCoord.y));
    float mask = 0.0, line = 0.0;
    if (guides.w > 0.0) {
        float cx = float(px) + 0.5, cy = float(py) + 0.5;
        if (guides.w > w / h) {
            float mh = w / guides.w, top = t + (h - mh) / 2.0;
            if (cy < top || cy >= top + mh) mask = kMask;
        } else {
            float mw = h * guides.w, left = l + (w - mw) / 2.0;
            if (cx < left || cx >= left + mw) mask = kMask;
        }
    }
    bool on = (guides.x > 0.5 && onBox(l, t, w, h, 0.05, px, py)) || (guides.y > 0.5 && onBox(l, t, w, h, 0.10, px, py));
    if (guides.z > 0.5) {
        int cc = int(floor(l + w / 2.0)), cr = int(floor(t + h / 2.0));
        int arm = max(8, int(floor(0.02 * min(w, h))));
        if ((px == cc && abs(py - cr) <= arm) || (py == cr && abs(px - cc) <= arm)) on = true;
    }
    if (on) line = kLine;
    c = c * (1.0 - mask);
    c = c * (1.0 - line) + vec3(window.w) * line;
    frag = vec4(c, 1.0);
}
