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
uniform int u_linear;         // 1: output linear light (for blending in a float buffer)

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

vec3 srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

vec3 linear_to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

// Shift an sRGB color in OKLab (matching's tint model: every texel of a
// tile moves by the same OKLab amount). Out-of-gamut results are clipped.
vec3 oklab_shift(vec3 srgb, vec3 shift) {
    if (shift == vec3(0.0)) return srgb;
    vec3 rgb = srgb_to_linear(srgb);
    vec3 lms = vec3(
        0.4122214708 * rgb.r + 0.5363325363 * rgb.g + 0.0514459929 * rgb.b,
        0.2119034982 * rgb.r + 0.6806995451 * rgb.g + 0.1073969566 * rgb.b,
        0.0883024619 * rgb.r + 0.2817188376 * rgb.g + 0.6299787005 * rgb.b);
    lms = sign(lms) * pow(abs(lms), vec3(1.0 / 3.0));
    vec3 lab = vec3(
        0.2104542553 * lms.x + 0.7936177850 * lms.y - 0.0040720468 * lms.z,
        1.9779984951 * lms.x - 2.4285922050 * lms.y + 0.4505937099 * lms.z,
        0.0259040371 * lms.x + 0.7827717662 * lms.y - 0.8086757660 * lms.z) + shift;
    lms = vec3(
        lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z,
        lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z,
        lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z);
    lms = lms * lms * lms;
    rgb = vec3(
        4.0767416621 * lms.x - 3.3077115913 * lms.y + 0.2309699292 * lms.z,
        -1.2684380046 * lms.x + 2.6097574011 * lms.y - 0.3413193965 * lms.z,
        -0.0041960863 * lms.x - 0.7034186147 * lms.y + 1.7076147010 * lms.z);
    return linear_to_srgb(rgb);
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
        color.rgb = oklab_shift(color.rgb, v_offset);
        color.rgb = mix(color.rgb, v_tint.rgb, v_tint.a);
    }
    if (u_linear == 1) color.rgb = srgb_to_linear(color.rgb);
    f_color = vec4(color.rgb, color.a * v_alpha);
}
