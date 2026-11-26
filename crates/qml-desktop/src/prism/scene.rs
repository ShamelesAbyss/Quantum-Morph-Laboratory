//! Unified 3D Scientific Scene Intermediate Representation & Adaptive Graphics Presets.

use eframe::egui::Color32;
use serde::{Deserialize, Serialize};

/// User-selectable graphics quality tiers optimizing between battery efficiency and 2560x1440 144Hz fidelity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GraphicsPreset {
    Minimal,
    Balanced,
    #[default]
    High,
    Ultra,
    Cinematic,
}

impl GraphicsPreset {
    pub const ALL: [Self; 5] = [
        Self::Minimal,
        Self::Balanced,
        Self::High,
        Self::Ultra,
        Self::Cinematic,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Minimal => "Minimal (2.5D Low-Power)",
            Self::Balanced => "Balanced (75% Res, Bloom)",
            Self::High => "High (100% Res, HDR Bloom)",
            Self::Ultra => "Ultra (Full Particles & Bloom)",
            Self::Cinematic => "Cinematic (Maximum Fidelity)",
        }
    }
}

/// Configurable GPU visual rendering parameters persisted with user preferences.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphicsSettings {
    pub preset: GraphicsPreset,
    /// Offscreen render target scale factor (`0.5`..=`1.0`).
    pub resolution_scale: f32,
    /// Particle density multiplier (`0.0`..=`1.0`).
    pub particle_density: f32,
    /// Enable 2-pass separable Gaussian selective bloom.
    pub bloom_enabled: bool,
    /// Selective bloom intensity (`0.0`..=`1.6`).
    pub bloom_intensity: f32,
    /// Geometry complexity multiplier (`0.5`..=`1.0`).
    pub geometry_complexity: f32,
    /// Global animation speed multiplier (`0.0`..=`1.5`).
    pub animation_intensity: f32,
    /// Accessibility reduced-motion mode (freezes auto-orbit and pulses).
    pub reduced_motion: bool,
    /// Target frame cap (`30`, `60`, `120`, `144`).
    pub fps_limit: u32,
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        Self::from_preset(GraphicsPreset::High)
    }
}

impl GraphicsSettings {
    pub fn from_preset(preset: GraphicsPreset) -> Self {
        match preset {
            GraphicsPreset::Minimal => Self {
                preset,
                resolution_scale: 0.65,
                particle_density: 0.15,
                bloom_enabled: false,
                bloom_intensity: 0.0,
                geometry_complexity: 0.5,
                animation_intensity: 0.5,
                reduced_motion: false,
                fps_limit: 60,
            },
            GraphicsPreset::Balanced => Self {
                preset,
                resolution_scale: 0.80,
                particle_density: 0.50,
                bloom_enabled: true,
                bloom_intensity: 0.65,
                geometry_complexity: 0.75,
                animation_intensity: 0.85,
                reduced_motion: false,
                fps_limit: 60,
            },
            GraphicsPreset::High => Self {
                preset,
                resolution_scale: 1.0,
                particle_density: 0.75,
                bloom_enabled: true,
                bloom_intensity: 0.90,
                geometry_complexity: 1.0,
                animation_intensity: 1.0,
                reduced_motion: false,
                fps_limit: 120,
            },
            GraphicsPreset::Ultra => Self {
                preset,
                resolution_scale: 1.0,
                particle_density: 1.0,
                bloom_enabled: true,
                bloom_intensity: 1.15,
                geometry_complexity: 1.0,
                animation_intensity: 1.0,
                reduced_motion: false,
                fps_limit: 144,
            },
            GraphicsPreset::Cinematic => Self {
                preset,
                resolution_scale: 1.0,
                particle_density: 1.0,
                bloom_enabled: true,
                bloom_intensity: 1.35,
                geometry_complexity: 1.0,
                animation_intensity: 1.15,
                reduced_motion: false,
                fps_limit: 144,
            },
        }
    }

    pub fn effective_anim_scale(&self) -> f32 {
        if self.reduced_motion {
            0.0
        } else {
            self.animation_intensity.clamp(0.0, 2.0)
        }
    }
}

/// Procedural backdrop shader mode tailored to each scientific workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundMode {
    /// Computational Singularity centerpiece: gravitational-lensing event horizon + accretion glow.
    SingularityCore = 0,
    /// Deep observatory grid for Q-Scope neural anatomy.
    DeepObservatory = 1,
    /// Deep void with faint starlight interference for Q-Entangle Latent Constellation.
    ConstellationVoid = 2,
    /// Split-chamber energy field for Q-Forge Morphogenesis.
    MorphogenesisField = 3,
    /// 3D objective-space coordinate horizon for Q-Genesis Crucible.
    CrucibleGrid = 4,
    /// Dual-column coherence backdrop for Q-Behavior Observatory.
    ConsciousnessDual = 5,
    /// Precision spacecraft engineering schematic for Hardware Observatory.
    HardwareSchematic = 6,
}

/// Procedural 3D mesh archetypes instanced on the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrystalShape {
    /// Faceted octahedral amethyst crystal (attention heads, singularity facets, RMQ organisms).
    Octahedron = 0,
    /// Beveled tensor prism slab (embeddings, MLP blocks, layer planes, hardware units).
    PrismSlab = 1,
    /// Smooth icosahedral sphere (latent constellation nodes, shared cores, candidate nuclei).
    IcoSphere = 2,
    /// Precision torus ring (mathematical orbital rings, shared basis halos).
    TorusRing = 3,
}

/// One GPU-instanced 3D crystalline object.
#[derive(Debug, Clone)]
pub struct CrystalInstance {
    pub position: [f32; 3],
    pub scale: [f32; 3],
    pub rotation: [f32; 3],
    pub color: [f32; 4],
    /// HDR emission intensity (values > 0.7 trigger selective Gaussian bloom).
    pub emission: f32,
    pub shape: CrystalShape,
    /// Selection / hover highlight strength (`0.0`..=`1.0`).
    pub selected: f32,
}

/// One luminous 3D neural pathway or affinity filament connecting two 3D points.
#[derive(Debug, Clone)]
pub struct FilamentEdge {
    pub p0: [f32; 3],
    pub p1: [f32; 3],
    pub color0: [f32; 4],
    pub color1: [f32; 4],
    pub thickness: f32,
    pub pulse_speed: f32,
    pub pulse_phase: f32,
    pub intensity: f32,
}

/// One GPU-driven orbital particle (orbit animated directly in WGSL vertex shader).
#[derive(Debug, Clone)]
pub struct OrbitalParticle {
    pub center: [f32; 3],
    pub orbit_radius: f32,
    pub inclination: f32,
    pub azimuth_offset: f32,
    pub angular_speed: f32,
    pub vertical_amp: f32,
    pub size: f32,
    pub color: [f32; 4],
    pub emission: f32,
}

/// Interactive pickable 3D region linked to a scientific entity ID.
#[derive(Debug, Clone)]
pub struct PickableNode {
    pub id: String,
    pub label: String,
    pub detail: String,
    pub position: [f32; 3],
    pub radius: f32,
    pub accent: Color32,
}

/// Optional screen-projected 3D scientific callout badge.
#[derive(Debug, Clone)]
pub struct OverlayLabel {
    pub world_pos: [f32; 3],
    pub text: String,
    pub color: Color32,
    pub prominent: bool,
}

/// Explanatory legend item drawn cleanly in the viewport corner.
#[derive(Debug, Clone)]
pub struct LegendItem {
    pub label: String,
    pub color: Color32,
}

/// Complete scene specification passed to the WGPU 3D Prism Renderer for one frame.
#[derive(Debug, Clone)]
pub struct PrismScene {
    pub background: BackgroundMode,
    /// Real laboratory energy / activity level (`0.15` idle .. `1.0` active compute).
    pub energy: f32,
    /// Explicit scientific provenance badge (e.g. "ACTUAL MEASURED DATA", "ARTISTIC IDENTITY", "ILLUSTRATIVE DEMO").
    pub provenance_badge: String,
    pub provenance_color: Color32,
    pub title: String,
    pub subtitle: String,
    pub instances: Vec<CrystalInstance>,
    pub filaments: Vec<FilamentEdge>,
    pub particles: Vec<OrbitalParticle>,
    pub pickables: Vec<PickableNode>,
    pub labels: Vec<OverlayLabel>,
    pub legend: Vec<LegendItem>,
}

impl PrismScene {
    pub fn new(
        background: BackgroundMode,
        title: impl Into<String>,
        subtitle: impl Into<String>,
        provenance_badge: impl Into<String>,
        provenance_color: Color32,
    ) -> Self {
        Self {
            background,
            energy: 0.28,
            provenance_badge: provenance_badge.into(),
            provenance_color,
            title: title.into(),
            subtitle: subtitle.into(),
            instances: Vec::new(),
            filaments: Vec::new(),
            particles: Vec::new(),
            pickables: Vec::new(),
            labels: Vec::new(),
            legend: Vec::new(),
        }
    }

    /// Helper to add a deterministic ring of GPU-animated orbital particles around `center`.
    pub fn add_orbital_halo(
        &mut self,
        center: [f32; 3],
        count: usize,
        min_r: f32,
        max_r: f32,
        tilt_spread: f32,
        seed: u64,
    ) {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next_f = || -> f32 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = ((s >> 40) as u32) as f32 / (u32::MAX >> 8) as f32;
            bits.clamp(0.0, 1.0)
        };
        for i in 0..count {
            let t = next_f();
            let radius = min_r + (max_r - min_r) * t;
            let incl = (next_f() - 0.5) * 2.0 * tilt_spread;
            let az = (i as f32 / count.max(1) as f32) * std::f32::consts::TAU + next_f() * 0.4;
            let speed = (0.14 + 0.34 * (1.0 - t)) * if i % 3 == 0 { -1.0 } else { 1.0 };
            let v_amp = (next_f() - 0.5) * 0.45;
            let size = 0.025 + next_f() * 0.048;
            let c_pick = next_f();
            let color = if c_pick < 0.45 {
                [0.545, 0.361, 0.965, 0.85] // Ultraviolet
            } else if c_pick < 0.78 {
                [0.133, 0.827, 0.933, 0.90] // Electric Cyan
            } else if c_pick < 0.93 {
                [0.655, 0.545, 0.980, 0.85] // Amethyst
            } else {
                [0.94, 0.96, 1.0, 0.95] // Silver starlight highlight
            };
            self.particles.push(OrbitalParticle {
                center,
                orbit_radius: radius,
                inclination: incl,
                azimuth_offset: az,
                angular_speed: speed,
                vertical_amp: v_amp,
                size,
                color,
                emission: 1.1 + next_f() * 0.9,
            });
        }
    }
}

/// Convert egui `Color32` to linear-friendly normalized `[f32; 4]`.
pub fn color_to_rgba(c: Color32, alpha: f32) -> [f32; 4] {
    [
        c.r() as f32 / 255.0,
        c.g() as f32 / 255.0,
        c.b() as f32 / 255.0,
        alpha,
    ]
}
