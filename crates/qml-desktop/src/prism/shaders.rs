//! Original WGSL Shaders for PROJECT PRISM — Procedural Singularity Backdrop,
//! Depth-Aware Iridescent Crystalline Geometry, Luminous Neural Filaments,
//! GPU-Driven Orbital Particle Systems, and 2-Pass Separable HDR Selective Bloom.

/// Pass 1: Procedural cosmic obsidian backdrop with subtle mathematical interference,
/// volumetric ultraviolet depth, and (in SingularityCore mode) a gravitational-lensing
/// accretion horizon that breathes with real laboratory compute energy.
pub const BACKDROP_WGSL: &str = r#"
struct FrameUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    params: vec4<f32>, // x=time, y=energy, z=mode, w=aspect
    bloom_params: vec4<f32>, // x=bloom_intensity, y=viewport_w, z=viewport_h, w=reduced_motion
};

@group(0) @binding(0) var<uniform> u: FrameUniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    let p = positions[vid];
    var out: VsOut;
    out.pos = vec4<f32>(p, 0.9999, 1.0);
    out.uv = p * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}

fn hash21(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(123.34, 456.21));
    let d = dot(q, q + 45.32);
    return fract(d * (q.x + q.y));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let aspect = max(u.params.w, 0.1);
    let t = u.params.x;
    let energy = u.params.y;
    let mode = i32(u.params.z + 0.5);

    var p = (in.uv - vec2<f32>(0.5, 0.5)) * vec2<f32>(aspect, 1.0);
    let r = length(p);
    let angle = atan2(p.y, p.x);

    // Near-black obsidian atmospheric base with subtle deep ultraviolet depth
    var col = vec3<f32>(0.014, 0.017, 0.036);
    let radial_nebula = exp(-r * 2.1);
    col += vec3<f32>(0.045, 0.022, 0.105) * radial_nebula * (0.75 + 0.4 * energy);
    col += vec3<f32>(0.008, 0.042, 0.072) * exp(-abs(p.y) * 5.0) * exp(-r * 1.4) * 0.45;

    // Subtle scientific polar + Cartesian interference reticle
    let ring_wave = smoothstep(0.985, 1.0, cos(r * 28.0 - t * 0.25));
    let grid_x = smoothstep(0.988, 1.0, cos(p.x * 32.0));
    let grid_y = smoothstep(0.988, 1.0, cos(p.y * 32.0));
    let reticle_mask = exp(-r * 1.9);
    col += vec3<f32>(0.16, 0.12, 0.34) * (grid_x + grid_y) * 0.06 * reticle_mask;
    col += vec3<f32>(0.09, 0.32, 0.44) * ring_wave * 0.08 * reticle_mask;

    // SingularityCore mode (Command Center): procedural gravitational lensing halo & accretion field
    if (mode == 0) {
        let horizon_r = 0.155;
        let dist_h = abs(r - horizon_r);
        let photon_ring = exp(-dist_h * 52.0) * (1.25 + 0.85 * energy);
        let swirl = 0.5 + 0.5 * sin(angle * 4.0 - t * 0.85 + 6.0 / (r + 0.18));
        let spiral2 = 0.5 + 0.5 * cos(angle * 7.0 + t * 0.55 - 10.0 * r);
        let accretion_band = exp(-abs(p.y * 1.35 + sin(p.x * 5.0 + t * 0.4) * 0.03) * 14.0)
            * smoothstep(0.11, 0.22, r) * exp(-r * 2.4);

        // Dark event horizon depression in the exact center
        let core_shadow = smoothstep(horizon_r * 0.68, horizon_r * 1.02, r);
        col *= mix(0.18, 1.0, core_shadow);

        // Luminous amethyst / ultraviolet / electric cyan spectral accretion
        let spectral = mix(
            vec3<f32>(0.54, 0.25, 0.98),
            vec3<f32>(0.12, 0.86, 0.96),
            swirl * 0.65 + spiral2 * 0.35
        );
        col += spectral * photon_ring * 1.15;
        col += mix(vec3<f32>(0.42, 0.18, 0.86), vec3<f32>(0.14, 0.78, 0.94), swirl)
            * accretion_band * (0.62 + 0.65 * energy);

        // Subtle gravitational lens distortion arcs
        let lens_arc = exp(-abs(r - 0.32 - 0.015 * sin(angle * 3.0 + t * 0.35)) * 42.0);
        col += vec3<f32>(0.36, 0.24, 0.78) * lens_arc * 0.38;
    } else if (mode == 3) {
        // Morphogenesis split chamber glow: Amethyst left -> Electric Cyan right
        let left_glow = exp(-length(p - vec2<f32>(-0.36, 0.0)) * 3.0);
        let right_glow = exp(-length(p - vec2<f32>(0.36, 0.0)) * 3.0);
        col += vec3<f32>(0.40, 0.16, 0.76) * left_glow * 0.28;
        col += vec3<f32>(0.08, 0.56, 0.74) * right_glow * 0.28;
    } else if (mode == 5) {
        // Consciousness Dual Observatory: twin vertical coherence columns
        let c1 = exp(-abs(p.x + 0.28) * 9.0) * 0.18;
        let c2 = exp(-abs(p.x - 0.28) * 9.0) * 0.18;
        col += vec3<f32>(0.10, 0.46, 0.66) * c1 + vec3<f32>(0.46, 0.18, 0.78) * c2;
    }

    // Distant sub-pixel pinpoint micro-starfield shimmer (circular Gaussian, never blocky)
    let star_grid = (in.uv * vec2<f32>(aspect, 1.0)) * 58.0;
    let cell = floor(star_grid);
    let h = hash21(cell);
    if (h > 0.965) {
        let jx = hash21(cell + vec2<f32>(17.1, 3.7)) * 0.6 - 0.3;
        let jy = hash21(cell + vec2<f32>(9.3, 29.4)) * 0.6 - 0.3;
        let f = fract(star_grid) - vec2<f32>(0.5 + jx, 0.5 + jy);
        let pinpoint = exp(-dot(f, f) * 78.0);
        let twinkle = 0.55 + 0.45 * sin(t * 1.4 + h * 120.0);
        col += vec3<f32>(0.62, 0.72, 1.0) * pinpoint * twinkle * 0.36;
    }

    return vec4<f32>(col, 1.0);
}
"#;

/// Pass 2: Depth-aware GPU-instanced 3D crystalline geometry with Fresnel iridescence,
/// dual key/rim lighting, and HDR emission for selective bloom.
pub const CRYSTAL_WGSL: &str = r#"
struct FrameUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    params: vec4<f32>,
    bloom_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: FrameUniforms;

struct VertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct InstanceIn {
    @location(2) i_pos: vec3<f32>,
    @location(3) i_scale: vec3<f32>,
    @location(4) i_rot: vec3<f32>,
    @location(5) i_color: vec4<f32>,
    @location(6) i_meta: vec4<f32>, // x=emission, y=selected, z=unused, w=unused
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) fx_data: vec2<f32>,
};

fn rot_euler(r: vec3<f32>) -> mat3x3<f32> {
    let cx = cos(r.x); let sx = sin(r.x);
    let cy = cos(r.y); let sy = sin(r.y);
    let cz = cos(r.z); let sz = sin(r.z);
    let rx = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, cx, sx),
        vec3<f32>(0.0, -sx, cx)
    );
    let ry = mat3x3<f32>(
        vec3<f32>(cy, 0.0, -sy),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(sy, 0.0, cy)
    );
    let rz = mat3x3<f32>(
        vec3<f32>(cz, sz, 0.0),
        vec3<f32>(-sz, cz, 0.0),
        vec3<f32>(0.0, 0.0, 1.0)
    );
    return rz * ry * rx;
}

@vertex
fn vs_main(v: VertexIn, inst: InstanceIn) -> VsOut {
    let rmat = rot_euler(inst.i_rot);
    let local = v.pos * inst.i_scale;
    let world = rmat * local + inst.i_pos;
    let inv_scale = vec3<f32>(
        1.0 / max(abs(inst.i_scale.x), 0.001),
        1.0 / max(abs(inst.i_scale.y), 0.001),
        1.0 / max(abs(inst.i_scale.z), 0.001)
    );
    let wnorm = normalize(rmat * (v.normal * inv_scale));

    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(world, 1.0);
    out.world_pos = world;
    out.world_normal = wnorm;
    out.color = inst.i_color;
    out.fx_data = inst.i_meta.xy;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let v = normalize(u.camera_pos.xyz - in.world_pos);
    let ndotv = clamp(abs(dot(n, v)), 0.0, 1.0);

    // Dual scientific studio lights: cool cyan key + warm amethyst rim
    let l1 = normalize(vec3<f32>(0.65, 0.85, 0.55));
    let l2 = normalize(vec3<f32>(-0.75, -0.35, -0.60));
    let diff1 = max(dot(n, l1), 0.0);
    let diff2 = max(dot(n, l2), 0.0);

    let h1 = normalize(l1 + v);
    let spec1 = pow(max(dot(n, h1), 0.0), 36.0);

    // Iridescent Fresnel crystal rim (Ultraviolet -> Amethyst -> Electric Cyan)
    let fresnel = pow(1.0 - ndotv, 2.6);
    let irid_t = clamp(0.5 + 0.5 * sin(ndotv * 5.0 + u.params.x * 0.6), 0.0, 1.0);
    let irid_col = mix(
        vec3<f32>(0.65, 0.42, 1.0),
        vec3<f32>(0.16, 0.90, 0.98),
        irid_t
    );

    let base = in.color.rgb;
    let emission = in.fx_data.x;
    let selected = in.fx_data.y;

    var lit = base * (0.24 + 0.68 * diff1 + 0.25 * diff2);
    lit += vec3<f32>(0.72, 0.88, 1.0) * spec1 * 0.55;
    lit += irid_col * fresnel * (0.65 + 0.55 * emission);
    lit += base * emission * 0.48;

    // Selection pulse highlight
    if (selected > 0.01) {
        let pulse = 0.7 + 0.3 * sin(u.params.x * 4.5);
        lit += vec3<f32>(0.18, 0.94, 1.0) * (0.55 + 0.75 * fresnel) * selected * pulse;
    }

    // Subtle depth atmospheric perspective
    let dist = length(u.camera_pos.xyz - in.world_pos);
    let fog = exp(-dist * 0.018);
    let final_rgb = mix(vec3<f32>(0.018, 0.020, 0.045), lit, fog);
    return vec4<f32>(final_rgb, in.color.a);
}
"#;

/// Pass 3: Luminous 3D neural pathway filaments rendered as camera-aligned ribbons
/// with traveling spectral energy pulses and smooth edge antialiasing.
pub const FILAMENT_WGSL: &str = r#"
struct FrameUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    params: vec4<f32>,
    bloom_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: FrameUniforms;

struct EdgeIn {
    @location(0) p0: vec3<f32>,
    @location(1) p1: vec3<f32>,
    @location(2) color0: vec4<f32>,
    @location(3) color1: vec4<f32>,
    @location(4) fx_data: vec4<f32>, // x=thickness, y=pulse_speed, z=pulse_phase, w=intensity
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>, // x=along edge [0..1], y=across [-1..1]
    @location(1) color: vec4<f32>,
    @location(2) fx_data: vec3<f32>, // x=pulse_speed, y=pulse_phase, z=intensity
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, edge: EdgeIn) -> VsOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(0.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0,  1.0),
        vec2<f32>(0.0, -1.0),
        vec2<f32>(1.0,  1.0),
        vec2<f32>(0.0,  1.0),
    );
    let q = quad[vid];
    let world_p = mix(edge.p0, edge.p1, q.x);
    let tangent = normalize(edge.p1 - edge.p0 + vec3<f32>(1e-5, 0.0, 0.0));
    let to_cam = normalize(u.camera_pos.xyz - world_p);
    let side = normalize(cross(tangent, to_cam) + vec3<f32>(0.0, 1e-5, 0.0));
    let displaced = world_p + side * (q.y * edge.fx_data.x);

    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(displaced, 1.0);
    out.uv = q;
    out.color = mix(edge.color0, edge.color1, q.x);
    out.fx_data = edge.fx_data.yzw;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let across = abs(in.uv.y);
    let core = exp(-across * across * 6.5);
    let halo = exp(-across * across * 1.8) * 0.45;
    let end_fade = smoothstep(0.0, 0.06, in.uv.x) * smoothstep(1.0, 0.94, in.uv.x);

    let wave = 0.5 + 0.5 * sin(in.uv.x * 12.566 - u.params.x * in.fx_data.x + in.fx_data.y);
    let pulse = 0.65 + 0.85 * pow(wave, 3.0);

    let alpha = (core + halo) * end_fade * in.color.a;
    let rgb = in.color.rgb * (0.85 + 0.8 * core) * pulse * in.fx_data.z;
    return vec4<f32>(rgb * alpha, alpha);
}
"#;

/// Pass 4: GPU-driven orbital particle system — orbits evaluated completely on GPU.
pub const PARTICLE_WGSL: &str = r#"
struct FrameUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    params: vec4<f32>,
    bloom_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: FrameUniforms;

struct ParticleIn {
    @location(0) center: vec3<f32>,
    @location(1) orbit: vec4<f32>, // x=radius, y=inclination, z=az_offset, w=angular_speed
    @location(2) shape: vec4<f32>, // x=vertical_amp, y=size, z=emission, w=unused
    @location(3) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) emission: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, p: ParticleIn) -> VsOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    let q = quad[vid];
    let t = u.params.x;
    let angle = p.orbit.z + t * p.orbit.w;
    let r = p.orbit.x;
    let ci = cos(p.orbit.y);
    let si = sin(p.orbit.y);

    let ox = r * cos(angle);
    let oz = r * sin(angle);
    let oy = p.shape.x * sin(angle * 2.0 + t * 0.5);
    // Tilt orbital plane by inclination around X axis
    let world_center = p.center + vec3<f32>(
        ox,
        oy * ci - oz * si,
        oy * si + oz * ci
    );

    let billboard = world_center
        + u.camera_right.xyz * (q.x * p.shape.y)
        + u.camera_up.xyz * (q.y * p.shape.y);

    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(billboard, 1.0);
    out.local_uv = q;
    out.color = p.color;
    out.emission = p.shape.z;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d2 = dot(in.local_uv, in.local_uv);
    if (d2 > 1.0) {
        discard;
    }
    let glow = exp(-d2 * 4.2);
    let core = exp(-d2 * 14.0);
    let rgb = in.color.rgb * (glow * in.emission + core * 1.4);
    let a = glow * in.color.a;
    return vec4<f32>(rgb * a, a);
}
"#;

/// Pass 5a: Selective Bloom bright-pass threshold extraction.
pub const BLOOM_THRESHOLD_WGSL: &str = r#"
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    let p = positions[vid];
    var out: VsOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let col = textureSample(src_tex, src_samp, in.uv).rgb;
    let luma = dot(col, vec3<f32>(0.2126, 0.7152, 0.0722));
    let knee = smoothstep(0.55, 1.25, luma);
    return vec4<f32>(col * knee, 1.0);
}
"#;

/// Pass 5b: Separable 9-tap Gaussian bloom blur (horizontal or vertical via `dir`).
pub const BLOOM_BLUR_WGSL: &str = r#"
struct BlurUniforms {
    texel_dir: vec4<f32>, // xy = step in UV space
};

@group(0) @binding(0) var<uniform> b: BlurUniforms;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var src_samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    let p = positions[vid];
    var out: VsOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let step_uv = b.texel_dir.xy;
    var acc = textureSample(src_tex, src_samp, in.uv).rgb * 0.227027;
    acc += textureSample(src_tex, src_samp, in.uv + step_uv * 1.384615).rgb * 0.316216;
    acc += textureSample(src_tex, src_samp, in.uv - step_uv * 1.384615).rgb * 0.316216;
    acc += textureSample(src_tex, src_samp, in.uv + step_uv * 3.230769).rgb * 0.070270;
    acc += textureSample(src_tex, src_samp, in.uv - step_uv * 3.230769).rgb * 0.070270;
    return vec4<f32>(acc, 1.0);
}
"#;

/// Pass 6: Final HDR + Selective Bloom ACES filmic tone-mapped composite into egui's render pass.
pub const COMPOSITE_WGSL: &str = r#"
struct CompositeUniforms {
    params: vec4<f32>, // x=bloom_intensity, y=vignette, z=0, w=0
};

@group(0) @binding(0) var<uniform> c: CompositeUniforms;
@group(0) @binding(1) var scene_tex: texture_2d<f32>;
@group(0) @binding(2) var bloom_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    let p = positions[vid];
    var out: VsOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c1 = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c1 * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let scene = textureSample(scene_tex, samp, in.uv).rgb;
    let bloom = textureSample(bloom_tex, samp, in.uv).rgb;
    var hdr = scene + bloom * c.params.x;

    // Subtle observatory frame vignette
    let centered = in.uv - vec2<f32>(0.5, 0.5);
    let vig = 1.0 - dot(centered, centered) * c.params.y;
    hdr *= clamp(vig, 0.62, 1.0);

    let mapped = aces_tonemap(hdr);
    return vec4<f32>(mapped, 1.0);
}
"#;
