#version 440
// Full-screen triangle from the vertex index: no vertex buffer.
layout(location = 0) out vec2 v_uv;   // (0,0) top-left, (1,1) bottom-right

layout(std140, binding = 0) uniform Probe {
    mat4 clip_corr;   // QRhi::clipSpaceCorrMatrix(): one NDC convention on every API
    vec4 encode;      // x: linear scale, y: 0 linear / 1 PQ / 2 SDR sRGB, z: display peak nits (0 unknown)
};

void main() {
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0;
    v_uv = vec2(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    gl_Position = clip_corr * vec4(pos, 0.0, 1.0);
}
