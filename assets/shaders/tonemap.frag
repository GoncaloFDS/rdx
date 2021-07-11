#version 460

layout(binding = 0, set = 0) uniform sampler2D initial_image;

layout(location = 0) out vec4 output_color;

void main() {
    vec4 color = texture(initial_image, gl_FragCoord.xy);
    output_color  = color;
}
