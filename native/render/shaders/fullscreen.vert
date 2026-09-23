#version 440
// Full-screen triangle from the vertex index. The composite fetches texels by
// gl_FragCoord, so no texture coordinates and no NDC convention are involved.
void main() {
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0;
    gl_Position = vec4(pos, 0.0, 1.0);
}
