#version 440
// The picture placed in the viewer window: a quad over the rectangle the
// viewport maths chose (core/viewport.hpp), in window pixels, top-left origin.
// uv (0, 0) is the picture's top-left texel on every API.

layout(location = 0) out vec2 v_uv;

layout(std140, binding = 0) uniform Blit {
    mat4 clip_corr;   // QRhi::clipSpaceCorrMatrix()
    vec4 rect;        // left, top, right, bottom in window pixels
    vec4 window;      // width, height in pixels
};

void main() {
    vec2 corner = vec2(gl_VertexIndex & 1, (gl_VertexIndex >> 1) & 1);   // triangle strip 0..3
    vec2 px = mix(rect.xy, rect.zw, corner);
    vec2 ndc = vec2(px.x / window.x * 2.0 - 1.0, 1.0 - px.y / window.y * 2.0);
    v_uv = corner;
    gl_Position = clip_corr * vec4(ndc, 0.0, 1.0);
}
