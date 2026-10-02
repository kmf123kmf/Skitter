#version 330

uniform sampler2DArray u_textures;
uniform float u_zoom;

// Outline mode (u_outline_px > 0) draws each sprite as a region marker: a
// line in the tint color, a dark band inside it, and an interior fill.
uniform float u_outline_px;   // line width, screen pixels
uniform float u_edge_px;      // dark band width inside the line
uniform float u_line_alpha;   // opacity of the line and its dark band
uniform float u_fill_alpha;   // interior opacity when not projecting
uniform int u_project;        // 1: fill with texture layer 0 at this world position
uniform vec2 u_texture_size;  // world size the projected texture spans

in vec2 v_uv;
in vec2 v_local;
in vec2 v_half;
in vec2 v_world;
in float v_alpha;
flat in float v_layer;
in vec4 v_tint;
in vec3 v_offset;

out vec4 f_color;

float band(float dist, float width) {
    return 1.0 - smoothstep(width - 0.5, width + 0.5, dist);
}

void main() {
    // Distance inside the rectangle edge, in screen pixels.
    vec2 q = abs(v_local) - v_half;
    float inside_px = -max(q.x, q.y) * u_zoom;

    vec4 color;
    if (u_outline_px > 0.0) {
        vec2 uv = v_world / u_texture_size;
        bool beyond = any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)));
        vec4 fill = vec4(v_tint.rgb, u_fill_alpha);
        if (u_project == 1) {
            // Parts of regions hanging past the texture show a neutral gray.
            fill = beyond ? vec4(0.24, 0.24, 0.26, 0.92)
                          : vec4(texture(u_textures, vec3(uv, 0.0)).rgb, 1.0);
        }
        color = mix(fill, vec4(0.0, 0.0, 0.0, max(fill.a, 0.55)),
                    band(inside_px, u_outline_px + u_edge_px) * u_line_alpha);
        color = mix(color, vec4(v_tint.rgb, 1.0), band(inside_px, u_outline_px) * u_line_alpha);
    } else {
        color = texture(u_textures, vec3(v_uv, v_layer));
        color.rgb = mix(color.rgb, v_tint.rgb, v_tint.a);
        color.rgb = clamp(color.rgb + v_offset, 0.0, 1.0);
    }
    f_color = vec4(color.rgb, color.a * v_alpha);
}
