#version 330

uniform sampler2DArray u_textures;

in vec2 v_uv;
in float v_alpha;
flat in float v_layer;
in vec4 v_tint;

out vec4 f_color;

void main() {
    vec4 color = texture(u_textures, vec3(v_uv, v_layer));
    color.rgb = mix(color.rgb, v_tint.rgb, v_tint.a);
    f_color = vec4(color.rgb, color.a * v_alpha);
}
