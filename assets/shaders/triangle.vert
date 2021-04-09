#version 450

layout (location = 0) in vec3 v_positon;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec3 v_color;

layout (location = 0) out vec3 out_color;

layout (push_constant) uniform constants {
    vec3 data;
    mat4 mvp;
} push_consts;

void main() {
    gl_Position = push_consts.mvp * vec4(v_positon, 1.0f);
    out_color = v_color;
}
