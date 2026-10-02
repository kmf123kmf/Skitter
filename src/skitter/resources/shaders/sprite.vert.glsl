#version 330

// Unit quad corner in [-0.5, 0.5], shared by every instance.
in vec2 in_corner;

// Per-instance attributes (see INSTANCE_DTYPE in sprites.py).
in vec2 in_pos;       // world-space center
in vec2 in_size;      // world-space width, height (negative mirrors the texture)
in float in_rotation; // radians, clockwise on screen
in float in_alpha;
in float in_layer;    // texture array layer index
in vec4 in_tint;      // rgb tint color, a = tint strength

uniform vec2 u_center;     // camera center, world units
uniform float u_zoom;      // screen pixels per world unit
uniform vec2 u_viewport;   // viewport size, logical pixels
uniform float u_shadow_px; // grow each quad by this many screen pixels for a shadow

out vec2 v_uv;
out vec2 v_local;          // offset from the sprite center along its axes, world units
out vec2 v_half;           // half the sprite size, world units
out vec2 v_world;
out float v_alpha;
flat out float v_layer;
out vec4 v_tint;

void main() {
    float margin = u_shadow_px / u_zoom;
    vec2 extent = in_size + 2.0 * margin * sign(in_size);
    vec2 local = in_corner * extent;

    float c = cos(in_rotation);
    float s = sin(in_rotation);
    vec2 world = in_pos + vec2(c * local.x - s * local.y, s * local.x + c * local.y);

    // World y grows downward (image convention); clip space y grows upward.
    vec2 ndc = (world - u_center) * u_zoom / (u_viewport * 0.5);
    gl_Position = vec4(ndc.x, -ndc.y, 0.0, 1.0);

    v_uv = in_corner * (extent / in_size) + 0.5;
    v_local = abs(in_corner * extent) * sign(in_corner);
    v_half = abs(in_size) * 0.5;
    v_world = world;
    v_alpha = in_alpha;
    v_layer = in_layer;
    v_tint = in_tint;
}
