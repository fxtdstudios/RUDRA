#version 440
// A copy of the display pass's picture: its values are already encoded for
// the swapchain (docs/view.spec.md sections 2 and 9), so nothing is changed
// here but the sampling (nearest at and above 1:1, trilinear below).

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag;

layout(binding = 1) uniform sampler2D picture;

void main() { frag = vec4(texture(picture, v_uv).rgb, 1.0); }
