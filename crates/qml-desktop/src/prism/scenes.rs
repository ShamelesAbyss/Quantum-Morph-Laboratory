//! Scientific 3D Scene Builders for all Quantum Morph Laboratory Workspaces.
//!
//! Every builder strictly separates ACTUAL MEASURED DATA from ARTISTIC IDENTITY
//! and ILLUSTRATIVE PREVIEW states with explicit provenance badges and legends.

use crate::prism::scene::{
    BackgroundMode, CrystalInstance, CrystalShape, FilamentEdge, LegendItem, OverlayLabel,
    PickableNode, PrismScene, color_to_rgba,
};
use crate::theme::{AMBER, AMETHYST, CRIMSON, CYAN, EMERALD, TEXT_MUTED, ULTRAVIOLET};
use eframe::egui::Color32;
use qml_behavior::{BehaviorComparisonReport, InterventionReport};
use qml_genesis::{GenesisArchive, GenesisProfile, LayerGene};
use qml_hardware::HardwareReport;
use qml_scope::{LayerRole, ModelScopeReport};
use std::collections::HashMap;
use std::f32::consts::TAU;

/// 6 Scientific Visualization Modes for Q-Scope 3D Neural Anatomy Observatory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScopeVisualMode {
    #[default]
    Anatomy,
    SpectralRank,
    QuantSensitivity,
    ParamDistribution,
    Outliers,
    CompressionEligibility,
}

impl ScopeVisualMode {
    pub const ALL: [Self; 6] = [
        Self::Anatomy,
        Self::SpectralRank,
        Self::QuantSensitivity,
        Self::ParamDistribution,
        Self::Outliers,
        Self::CompressionEligibility,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Anatomy => "Architectural Anatomy",
            Self::SpectralRank => "Spectral Structure",
            Self::QuantSensitivity => "Quantization Sensitivity",
            Self::ParamDistribution => "Parameter Distribution",
            Self::Outliers => "Numerical Outliers (6σ)",
            Self::CompressionEligibility => "Compression Eligibility",
        }
    }
}

/// 1. Singularity Command Center — Artistic Computational Singularity Centerpiece.
///    Real active jobs & telemetry drive orbital energy and resonance without claiming
///    physical quantum computation.
pub fn build_singularity_scene(
    time_sec: f32,
    is_busy: bool,
    active_task_label: Option<&str>,
    model_count: usize,
    gpu_name: Option<&str>,
) -> PrismScene {
    let energy = if is_busy { 0.95 } else { 0.38 };
    let subtitle = match active_task_label {
        Some(lbl) => format!("ACTIVE COMPUTE RESONANCE — {lbl}"),
        None => format!(
            "Artistic Identity Centerpiece  •  {} Model(s) Registered  •  {}",
            model_count,
            gpu_name.unwrap_or("Rayon CPU Backend")
        ),
    };
    let mut scene = PrismScene::new(
        BackgroundMode::SingularityCore,
        "THE COMPUTATIONAL SINGULARITY",
        subtitle,
        if is_busy {
            "ARTISTIC IDENTITY • LIVE COMPUTE RESONANCE"
        } else {
            "ARTISTIC LABORATORY IDENTITY"
        },
        if is_busy { CYAN } else { AMETHYST },
    );
    scene.energy = energy;

    // Central faceted amethyst/cyan singularity core (counter-rotating dual octahedra)
    scene.instances.push(CrystalInstance {
        position: [0.0, 0.0, 0.0],
        scale: [0.72, 1.05, 0.72],
        rotation: [0.22 * time_sec, 0.45 * time_sec, 0.14 * time_sec],
        color: color_to_rgba(ULTRAVIOLET, 0.92),
        emission: 0.95 + 0.45 * energy,
        shape: CrystalShape::Octahedron,
        selected: if is_busy { 0.6 } else { 0.0 },
    });
    scene.instances.push(CrystalInstance {
        position: [0.0, 0.0, 0.0],
        scale: [0.46, 0.46, 0.46],
        rotation: [-0.35 * time_sec, -0.60 * time_sec, 0.25 * time_sec],
        color: color_to_rgba(CYAN, 0.96),
        emission: 1.55 + 0.55 * energy,
        shape: CrystalShape::IcoSphere,
        selected: 0.0,
    });

    // 4 layered precision mathematical orbital rings
    let rings: [(f32, [f32; 3], Color32, f32); 4] = [
        (1.65, [0.35, time_sec * 0.28, 0.18], CYAN, 1.25),
        (2.25, [-0.52, -time_sec * 0.21, 0.45], ULTRAVIOLET, 1.05),
        (2.90, [0.88, time_sec * 0.14, -0.32], AMETHYST, 0.85),
        (
            3.55,
            [-0.22, -time_sec * 0.09, 0.78],
            Color32::from_rgb(120, 145, 255),
            0.65,
        ),
    ];
    for (r, rot, col, em) in rings {
        scene.instances.push(CrystalInstance {
            position: [0.0, 0.0, 0.0],
            scale: [r, r, r],
            rotation: rot,
            color: color_to_rgba(col, 0.86),
            emission: em * (0.85 + 0.35 * energy),
            shape: CrystalShape::TorusRing,
            selected: 0.0,
        });
    }

    // 10 orbiting crystalline laboratory subsystem nodes + luminous neural pathways
    let subsystems: [(&str, &str, Color32); 10] = [
        ("qscope", "Q-Scope", CYAN),
        ("qentangle", "Q-Entangle", AMETHYST),
        ("qforge", "Q-Forge", ULTRAVIOLET),
        ("qgenesis", "Q-Genesis", EMERALD),
        ("qcore", "Q-Core", CYAN),
        ("qbehavior", "Q-Behavior", AMBER),
        ("qmf", "QMF v1", AMETHYST),
        ("library", "Library", Color32::from_rgb(160, 190, 255)),
        ("rmq", "RMQ Lab", ULTRAVIOLET),
        ("hardware", "Hardware", EMERALD),
    ];
    let mut anchors = Vec::with_capacity(subsystems.len());
    for (i, (id, label, col)) in subsystems.iter().enumerate() {
        let a = (i as f32 / subsystems.len() as f32) * TAU + time_sec * 0.09;
        let elev = ((i as f32 * 1.7 + time_sec * 0.25).sin()) * 0.95;
        let rad = 2.65 + (i % 2) as f32 * 0.65;
        let pos = [rad * a.cos(), elev, rad * a.sin()];
        anchors.push((pos, *col));
        scene.instances.push(CrystalInstance {
            position: pos,
            scale: [0.19, 0.28, 0.19],
            rotation: [a, time_sec * 0.5 + i as f32, 0.3],
            color: color_to_rgba(*col, 0.92),
            emission: 1.05,
            shape: if i % 2 == 0 {
                CrystalShape::Octahedron
            } else {
                CrystalShape::IcoSphere
            },
            selected: 0.0,
        });
        scene.pickables.push(PickableNode {
            id: (*id).to_owned(),
            label: (*label).to_owned(),
            detail: "Click to enter subsystem workspace".to_owned(),
            position: pos,
            radius: 0.36,
            accent: *col,
        });
        scene.labels.push(OverlayLabel {
            world_pos: [pos[0], pos[1] + 0.32, pos[2]],
            text: (*label).to_owned(),
            color: *col,
            prominent: false,
        });
        // Luminous radial pathway from Singularity Core -> Subsystem Crystal
        scene.filaments.push(FilamentEdge {
            p0: [0.0, 0.0, 0.0],
            p1: pos,
            color0: color_to_rgba(CYAN, 0.72),
            color1: color_to_rgba(*col, 0.65),
            thickness: 0.022 + 0.014 * energy,
            pulse_speed: 2.2 + 1.8 * energy,
            pulse_phase: i as f32 * 0.85,
            intensity: 0.9 + 0.6 * energy,
        });
    }

    // Ring interconnections between neighboring subsystem crystals
    for i in 0..anchors.len() {
        let j = (i + 1) % anchors.len();
        let k = (i + 3) % anchors.len();
        scene.filaments.push(FilamentEdge {
            p0: anchors[i].0,
            p1: anchors[j].0,
            color0: color_to_rgba(anchors[i].1, 0.48),
            color1: color_to_rgba(anchors[j].1, 0.48),
            thickness: 0.015,
            pulse_speed: 1.5,
            pulse_phase: i as f32 * 0.6,
            intensity: 0.75,
        });
        scene.filaments.push(FilamentEdge {
            p0: anchors[i].0,
            p1: anchors[k].0,
            color0: color_to_rgba(ULTRAVIOLET, 0.25),
            color1: color_to_rgba(CYAN, 0.25),
            thickness: 0.010,
            pulse_speed: 1.0,
            pulse_phase: i as f32 * 1.1,
            intensity: 0.45,
        });
    }

    // 1,400 GPU-animated orbital particles (zero CPU per-frame vertex update!)
    scene.add_orbital_halo([0.0, 0.0, 0.0], 1400, 1.15, 4.8, 0.72, 2026);
    scene
}

/// 2. Q-Scope — 3D Neural Anatomy Observatory.
///    Visualizes actual inspected tensor architecture from `ModelScopeReport` across 6
///    color-mapped scientific modes, or an explicitly labeled illustrative preview when idle.
pub fn build_qscope_anatomy_scene(
    report: Option<&ModelScopeReport>,
    mode: ScopeVisualMode,
    selected_tensor_idx: usize,
    time_sec: f32,
) -> PrismScene {
    let Some(rep) = report else {
        // Explicitly labeled illustrative preview so the observatory feels alive on first open
        let mut scene = PrismScene::new(
            BackgroundMode::DeepObservatory,
            "NEURAL ANATOMY OBSERVATORY (DEMO PREVIEW)",
            "Select a .safetensors model and click 'Run Q-Scope' to inspect actual weights.",
            "ILLUSTRATIVE ARCHITECTURE PREVIEW — NOT REAL MODEL DATA",
            AMBER,
        );
        let layers = 8;
        for l in 0..layers {
            let y = (l as f32 - (layers as f32 - 1.0) * 0.5) * 0.68;
            let roles = [
                (-1.2, CYAN, CrystalShape::Octahedron),
                (-0.4, AMETHYST, CrystalShape::Octahedron),
                (0.4, ULTRAVIOLET, CrystalShape::PrismSlab),
                (1.2, EMERALD, CrystalShape::PrismSlab),
            ];
            for (x, col, shape) in roles {
                scene.instances.push(CrystalInstance {
                    position: [x, y, 0.0],
                    scale: [0.24, 0.16, 0.24],
                    rotation: [0.0, time_sec * 0.25 + l as f32 * 0.2, 0.0],
                    color: color_to_rgba(col, 0.78),
                    emission: 0.7,
                    shape,
                    selected: 0.0,
                });
            }
            if l + 1 < layers {
                let y2 = ((l + 1) as f32 - (layers as f32 - 1.0) * 0.5) * 0.68;
                scene.filaments.push(FilamentEdge {
                    p0: [0.0, y, 0.0],
                    p1: [0.0, y2, 0.0],
                    color0: color_to_rgba(ULTRAVIOLET, 0.55),
                    color1: color_to_rgba(CYAN, 0.55),
                    thickness: 0.018,
                    pulse_speed: 2.0,
                    pulse_phase: l as f32 * 0.5,
                    intensity: 0.85,
                });
            }
        }
        scene.add_orbital_halo([0.0, 0.0, 0.0], 320, 2.2, 4.4, 0.45, 77);
        return scene;
    };

    let mut scene = PrismScene::new(
        BackgroundMode::DeepObservatory,
        format!("3D NEURAL ANATOMY — {}", mode.label().to_uppercase()),
        format!(
            "{} tensors  •  {} params  •  Click any crystal to inspect its histogram & SVD spectrum",
            rep.tensors.len(),
            rep.summary.total_parameters
        ),
        "ACTUAL MEASURED Q-SCOPE TELEMETRY",
        EMERALD,
    );

    // Organize tensors by layer index
    let max_layer = rep
        .tensors
        .iter()
        .filter_map(|t| t.layer_index)
        .max()
        .unwrap_or(0);
    let max_params = rep
        .tensors
        .iter()
        .map(|t| t.element_count)
        .max()
        .unwrap_or(1) as f32;
    let max_susc = rep
        .tensors
        .iter()
        .filter_map(|t| {
            t.stats
                .as_ref()
                .map(|s| s.quantization_susceptibility_score as f32)
        })
        .fold(1.0_f32, f32::max);
    let max_crest = rep
        .tensors
        .iter()
        .filter_map(|t| t.stats.as_ref().map(|s| s.crest_factor as f32))
        .fold(1.0_f32, f32::max);

    let layer_span = (max_layer + 2).max(2) as f32;
    let y_spacing = (6.8 / layer_span).clamp(0.16, 0.72);
    let y_center = (max_layer as f32) * 0.5 * y_spacing;

    // Track per-layer spine anchors for residual stream filaments
    let mut spine_pts: HashMap<usize, [f32; 3]> = HashMap::new();

    for (idx, t) in rep.tensors.iter().enumerate() {
        let (x, z, base_y, shape) = match t.role {
            LayerRole::Embedding => (0.0, 0.0, -y_center - 0.85, CrystalShape::PrismSlab),
            LayerRole::LmHead => (0.0, 0.0, y_center + 0.85, CrystalShape::PrismSlab),
            _ => {
                let l = t.layer_index.unwrap_or(0);
                let y = l as f32 * y_spacing - y_center;
                spine_pts.entry(l).or_insert([0.0, y, 0.0]);
                // Arrange attention on left wing, MLP on right wing, norms on central spine
                match t.role {
                    LayerRole::QProj | LayerRole::QkvProj => {
                        (-1.75, 0.45, y, CrystalShape::Octahedron)
                    }
                    LayerRole::KProj => (-1.25, -0.45, y, CrystalShape::Octahedron),
                    LayerRole::VProj => (-0.75, 0.45, y, CrystalShape::Octahedron),
                    LayerRole::OProj => (-0.35, -0.35, y, CrystalShape::IcoSphere),
                    LayerRole::GateProj => (0.65, 0.42, y, CrystalShape::PrismSlab),
                    LayerRole::UpProj => (1.25, -0.42, y, CrystalShape::PrismSlab),
                    LayerRole::DownProj => (1.85, 0.0, y, CrystalShape::PrismSlab),
                    LayerRole::Norm => (0.0, 0.0, y, CrystalShape::TorusRing),
                    _ => (0.0, 0.85, y, CrystalShape::IcoSphere),
                }
            }
        };

        let size_norm =
            ((t.element_count as f32).ln().max(1.0) / max_params.ln().max(1.0)).clamp(0.25, 1.0);
        let susc = t
            .stats
            .as_ref()
            .map_or(0.0, |s| s.quantization_susceptibility_score);
        let crest = t.stats.as_ref().map_or(0.0, |s| s.crest_factor);
        let out6 = t.stats.as_ref().map_or(0, |s| s.outlier_6sigma_count);

        let (col, emission, detail) = match mode {
            ScopeVisualMode::Anatomy => {
                let c = match t.role {
                    LayerRole::Embedding | LayerRole::LmHead => CYAN,
                    LayerRole::QProj
                    | LayerRole::KProj
                    | LayerRole::VProj
                    | LayerRole::OProj
                    | LayerRole::QkvProj => ULTRAVIOLET,
                    LayerRole::GateProj | LayerRole::UpProj | LayerRole::DownProj => AMETHYST,
                    LayerRole::Norm => EMERALD,
                    _ => Color32::from_rgb(160, 175, 220),
                };
                (c, 0.65, format!("{}  shape={:?}", t.role.as_str(), t.shape))
            }
            ScopeVisualMode::SpectralRank => {
                if let Some(svd) = &t.matrix_structure {
                    let ratio =
                        (svd.effective_rank / svd.computed_rank.max(1) as f64).clamp(0.0, 1.0);
                    let c = if ratio < 0.55 {
                        CYAN
                    } else if ratio < 0.80 {
                        ULTRAVIOLET
                    } else {
                        AMBER
                    };
                    (
                        c,
                        0.6 + (1.0 - ratio as f32) * 0.9,
                        format!(
                            "eff_rank={:.2} / {} (95%E={})",
                            svd.effective_rank, svd.computed_rank, svd.rank_95_energy
                        ),
                    )
                } else {
                    (
                        Color32::from_rgb(65, 74, 110),
                        0.2,
                        "SVD not sampled".to_owned(),
                    )
                }
            }
            ScopeVisualMode::QuantSensitivity => {
                let n = (susc as f32 / max_susc).clamp(0.0, 1.0);
                let c = heat_color(n);
                (c, 0.45 + n * 1.15, format!("susceptibility={susc:.2}"))
            }
            ScopeVisualMode::ParamDistribution => {
                let c = crate::theme::spectral_color(f64::from(size_norm));
                (
                    c,
                    0.45 + size_norm * 0.95,
                    format!(
                        "{} params ({:.2} MiB)",
                        t.element_count,
                        t.data_bytes as f64 / 1_048_576.0
                    ),
                )
            }
            ScopeVisualMode::Outliers => {
                let n = (crest as f32 / max_crest).clamp(0.0, 1.0);
                let c = heat_color(n);
                (
                    c,
                    0.45 + n * 1.25,
                    format!("crest={crest:.2}  out6σ={out6}"),
                )
            }
            ScopeVisualMode::CompressionEligibility => {
                let (c, tag) = if matches!(t.role, LayerRole::Embedding | LayerRole::LmHead) {
                    (AMBER, "Protect (INT8/FP16)")
                } else if susc > 20.0 {
                    (CRIMSON, "High-Crest (INT8/Q4_1)")
                } else {
                    (EMERALD, "Aggressive (Q4_0/RMQ)")
                };
                (c, 0.85, tag.to_owned())
            }
        };

        let is_sel = idx == selected_tensor_idx;
        let base_s = match shape {
            CrystalShape::TorusRing => [0.22, 0.22, 0.22],
            CrystalShape::PrismSlab
                if matches!(t.role, LayerRole::Embedding | LayerRole::LmHead) =>
            {
                [1.8, 0.14, 0.85]
            }
            CrystalShape::PrismSlab => [
                0.20 + 0.22 * size_norm,
                0.09 + 0.06 * size_norm,
                0.18 + 0.16 * size_norm,
            ],
            _ => {
                let r = 0.09 + 0.12 * size_norm;
                [r, r * 1.15, r]
            }
        };
        let rot = if shape == CrystalShape::TorusRing {
            [0.0, time_sec * 0.3, 0.0]
        } else {
            [0.0, if is_sel { time_sec * 0.9 } else { 0.0 }, 0.0]
        };

        scene.instances.push(CrystalInstance {
            position: [x, base_y, z],
            scale: base_s,
            rotation: rot,
            color: color_to_rgba(col, if is_sel { 0.98 } else { 0.84 }),
            emission: if is_sel { emission + 0.85 } else { emission },
            shape,
            selected: if is_sel { 1.0 } else { 0.0 },
        });
        scene.pickables.push(PickableNode {
            id: idx.to_string(),
            label: t.name.clone(),
            detail,
            position: [x, base_y, z],
            radius: (base_s[0].max(base_s[1]) * 1.25).clamp(0.12, 0.42),
            accent: col,
        });
    }

    // Connect residual stream spine across layers
    for l in 0..max_layer {
        if let (Some(&p0), Some(&p1)) = (spine_pts.get(&l), spine_pts.get(&(l + 1))) {
            scene.filaments.push(FilamentEdge {
                p0,
                p1,
                color0: color_to_rgba(ULTRAVIOLET, 0.55),
                color1: color_to_rgba(CYAN, 0.55),
                thickness: 0.016,
                pulse_speed: 2.0,
                pulse_phase: l as f32 * 0.4,
                intensity: 0.8,
            });
        }
    }

    scene.labels.push(OverlayLabel {
        world_pos: [-1.25, y_center + 0.45, 0.0],
        text: "SELF-ATTENTION WING (Q/K/V/O)".to_owned(),
        color: ULTRAVIOLET,
        prominent: true,
    });
    scene.labels.push(OverlayLabel {
        world_pos: [1.35, y_center + 0.45, 0.0],
        text: "SwiGLU MLP WING (GATE/UP/DOWN)".to_owned(),
        color: AMETHYST,
        prominent: true,
    });

    scene.legend = match mode {
        ScopeVisualMode::Anatomy => vec![
            LegendItem {
                label: "Embeddings/Head".into(),
                color: CYAN,
            },
            LegendItem {
                label: "Attention".into(),
                color: ULTRAVIOLET,
            },
            LegendItem {
                label: "MLP".into(),
                color: AMETHYST,
            },
            LegendItem {
                label: "RMSNorm".into(),
                color: EMERALD,
            },
        ],
        ScopeVisualMode::QuantSensitivity | ScopeVisualMode::Outliers => vec![
            LegendItem {
                label: "Low".into(),
                color: CYAN,
            },
            LegendItem {
                label: "Moderate".into(),
                color: ULTRAVIOLET,
            },
            LegendItem {
                label: "Elevated".into(),
                color: AMBER,
            },
            LegendItem {
                label: "Peak Outlier/Susc".into(),
                color: CRIMSON,
            },
        ],
        ScopeVisualMode::CompressionEligibility => vec![
            LegendItem {
                label: "Aggressive Q4_0/RMQ".into(),
                color: EMERALD,
            },
            LegendItem {
                label: "Protect Embed/Head".into(),
                color: AMBER,
            },
            LegendItem {
                label: "High-Crest Sensitive".into(),
                color: CRIMSON,
            },
        ],
        _ => vec![
            LegendItem {
                label: "Low / Compact".into(),
                color: CYAN,
            },
            LegendItem {
                label: "Mid".into(),
                color: ULTRAVIOLET,
            },
            LegendItem {
                label: "High / Dense".into(),
                color: AMBER,
            },
        ],
    };
    scene.add_orbital_halo([0.0, 0.0, 0.0], 360, 2.8, 5.2, 0.35, 135);
    scene
}

/// 3. Q-Entangle — 3D Latent Constellation (Classical Grassmannian Subspace Affinity).
pub fn build_qentangle_constellation_scene(
    report: Option<&ModelScopeReport>,
    selected_pair_idx: Option<usize>,
    affinity_threshold: f64,
    time_sec: f32,
) -> PrismScene {
    let Some(rep) = report else {
        let mut scene = PrismScene::new(
            BackgroundMode::ConstellationVoid,
            "THE LATENT CONSTELLATION (AWAITING Q-SCOPE DATA)",
            "Run Q-Scope on a model to measure cross-layer Grassmannian subspace affinity.",
            "NO MODEL ANALYZED YET",
            TEXT_MUTED,
        );
        scene.add_orbital_halo([0.0, 0.0, 0.0], 400, 1.5, 4.5, 0.8, 999);
        return scene;
    };

    let mut scene = PrismScene::new(
        BackgroundMode::ConstellationVoid,
        "THE LATENT CONSTELLATION — GRASSMANNIAN SUBSPACE AFFINITY",
        format!(
            "{} measured layer pairs  •  Classical principal-angle subspace geometry (not physical quantum entanglement)",
            rep.cross_layer_relations.len()
        ),
        "ACTUAL MEASURED GRASSMANNIAN AFFINITY",
        CYAN,
    );

    // Collect unique tensors involved in measured cross-layer affinities
    let mut names: Vec<String> = Vec::new();
    for p in &rep.cross_layer_relations {
        if !names.contains(&p.tensor_a) {
            names.push(p.tensor_a.clone());
        }
        if !names.contains(&p.tensor_b) {
            names.push(p.tensor_b.clone());
        }
    }
    names.sort();

    // Group spatially by role sector & layer height in 3D cylinder/constellation
    let mut positions: HashMap<String, [f32; 3]> = HashMap::new();
    let n = names.len().max(1);
    for (i, name) in names.iter().enumerate() {
        let (layer_opt, role) = qml_scope::classify_tensor_name(name);
        let role_sector = match role {
            LayerRole::QProj | LayerRole::QkvProj => 0.0,
            LayerRole::KProj => 1.0,
            LayerRole::VProj => 2.0,
            LayerRole::OProj => 3.0,
            LayerRole::GateProj => 4.0,
            LayerRole::UpProj => 5.0,
            LayerRole::DownProj => 6.0,
            _ => 7.0,
        };
        let layer_f = layer_opt.unwrap_or(i) as f32;
        let angle = (role_sector / 7.0) * TAU + layer_f * 0.09;
        let radius = 2.0 + (role_sector % 2.0) * 0.55;
        let y = ((i as f32 / n as f32) - 0.5) * 4.2;
        let pos = [radius * angle.cos(), y, radius * angle.sin()];
        positions.insert(name.clone(), pos);

        let col = match role {
            LayerRole::KProj => CYAN,
            LayerRole::QProj | LayerRole::VProj | LayerRole::OProj | LayerRole::QkvProj => {
                ULTRAVIOLET
            }
            _ => AMETHYST,
        };
        scene.instances.push(CrystalInstance {
            position: pos,
            scale: [0.13, 0.13, 0.13],
            rotation: [0.0, time_sec * 0.4 + i as f32 * 0.2, 0.0],
            color: color_to_rgba(col, 0.90),
            emission: 0.95,
            shape: CrystalShape::IcoSphere,
            selected: 0.0,
        });
        scene.pickables.push(PickableNode {
            id: format!("node:{name}"),
            label: name.clone(),
            detail: format!("Role: {}", role.as_str()),
            position: pos,
            radius: 0.22,
            accent: col,
        });
    }

    // Draw glowing affinity filaments for pairs meeting display threshold (or top 40 pairs)
    for (p_idx, pair) in rep.cross_layer_relations.iter().enumerate() {
        let is_sel = selected_pair_idx == Some(p_idx);
        if !is_sel && pair.shared_structure_score < affinity_threshold && p_idx >= 40 {
            continue;
        }
        if let (Some(&p0), Some(&p1)) =
            (positions.get(&pair.tensor_a), positions.get(&pair.tensor_b))
        {
            let aff = pair.shared_structure_score.clamp(0.0, 1.0) as f32;
            let col = if is_sel {
                EMERALD
            } else if aff >= 0.30 {
                CYAN
            } else if aff >= 0.15 {
                AMETHYST
            } else {
                ULTRAVIOLET
            };
            scene.filaments.push(FilamentEdge {
                p0,
                p1,
                color0: color_to_rgba(col, 0.35 + aff * 0.65),
                color1: color_to_rgba(CYAN, 0.35 + aff * 0.65),
                thickness: if is_sel { 0.045 } else { 0.008 + aff * 0.042 },
                pulse_speed: 1.4 + aff * 3.2,
                pulse_phase: p_idx as f32 * 0.5,
                intensity: if is_sel { 1.8 } else { 0.55 + aff * 1.25 },
            });
            // Place a subtle pickable midpoint node for top/selected pairs so clicking the filament inspects it!
            if p_idx < 24 || is_sel {
                let mid = [
                    (p0[0] + p1[0]) * 0.5,
                    (p0[1] + p1[1]) * 0.5,
                    (p0[2] + p1[2]) * 0.5,
                ];
                scene.pickables.push(PickableNode {
                    id: format!("pair:{p_idx}"),
                    label: format!(
                        "{} <-> {}",
                        short_name(&pair.tensor_a),
                        short_name(&pair.tensor_b)
                    ),
                    detail: format!(
                        "shared={:.4}  left={:.4}  right={:.4}  cos={:.4}",
                        pair.shared_structure_score,
                        pair.left_subspace_affinity,
                        pair.right_subspace_affinity,
                        pair.weight_cosine_similarity
                    ),
                    position: mid,
                    radius: 0.16,
                    accent: col,
                });
            }
        }
    }

    scene.legend = vec![
        LegendItem {
            label: "High Affinity (>=0.30, RMQ Eligible)".into(),
            color: CYAN,
        },
        LegendItem {
            label: "Moderate (0.15–0.30)".into(),
            color: AMETHYST,
        },
        LegendItem {
            label: "Low (<0.15)".into(),
            color: ULTRAVIOLET,
        },
    ];
    scene.add_orbital_halo([0.0, 0.0, 0.0], 550, 1.8, 4.8, 0.9, 404);
    scene
}

/// 4. Q-Forge — The Morphogenesis Chamber.
///    Left: dense source architecture monoliths; Right: transformed representation geometry
///    (shared latent halos for RMQ, tiered quantized prisms, and cyan sparse residual pathways).
pub fn build_qforge_morphogenesis_scene(
    strategy_label: &str,
    is_running: bool,
    progress_frac: f32,
    compression_ratio: Option<f64>,
    shared_latents: usize,
    time_sec: f32,
) -> PrismScene {
    let mut scene = PrismScene::new(
        BackgroundMode::MorphogenesisField,
        format!("MORPHOGENESIS CHAMBER — {}", strategy_label.to_uppercase()),
        if let Some(cr) = compression_ratio {
            format!(
                "Measured Compression: {cr:.2}× vs source  •  {shared_latents} Shared Latent Structures"
            )
        } else if is_running {
            format!(
                "Live Transformation Active — {:.0}% complete",
                progress_frac * 100.0
            )
        } else {
            "Schematic Representation Flow: Original Dense Weights (Left) -> Transformed QMF v1 (Right)".to_owned()
        },
        if compression_ratio.is_some() {
            "VERIFIED Q-FORGE RESULT • SCHEMATIC FLOW"
        } else if is_running {
            "LIVE TRANSFORMATION PROGRESS"
        } else {
            "SCHEMATIC REPRESENTATION PREVIEW"
        },
        if is_running { CYAN } else { ULTRAVIOLET },
    );
    scene.energy = if is_running { 0.95 } else { 0.42 };

    let layers = 6;
    let comp_scale = compression_ratio
        .map(|cr| (1.0 / (cr as f32).sqrt()).clamp(0.32, 0.92))
        .unwrap_or(0.55);

    // Left stack: Original Dense Weight Monoliths
    for i in 0..layers {
        let y = (i as f32 - 2.5) * 0.56;
        let p_left = [-2.2, y, 0.0];
        scene.instances.push(CrystalInstance {
            position: p_left,
            scale: [0.85, 0.24, 0.65],
            rotation: [0.0, 0.18 * (time_sec * 0.3 + i as f32 * 0.2).sin(), 0.0],
            color: color_to_rgba(AMETHYST, 0.82),
            emission: 0.65,
            shape: CrystalShape::PrismSlab,
            selected: 0.0,
        });

        // Right stack: Transformed Representation (compact blocks + optional shared latent core)
        let p_right = [2.2, y, 0.0];
        let active_layer = !is_running || (i as f32 / layers as f32) <= progress_frac;
        scene.instances.push(CrystalInstance {
            position: p_right,
            scale: [0.85 * comp_scale, 0.20, 0.65 * comp_scale],
            rotation: [0.0, -0.25 * time_sec + i as f32 * 0.15, 0.0],
            color: color_to_rgba(if active_layer { CYAN } else { ULTRAVIOLET }, 0.90),
            emission: if active_layer { 1.15 } else { 0.35 },
            shape: if strategy_label.contains("RMQ") || strategy_label.contains("SVD") {
                CrystalShape::Octahedron
            } else {
                CrystalShape::PrismSlab
            },
            selected: if is_running && active_layer { 0.6 } else { 0.0 },
        });

        // Morphogenesis energy conduit from source -> central manifold -> target
        let mid = [0.0, y * 0.55, (i as f32 * 0.8).sin() * 0.35];
        scene.filaments.push(FilamentEdge {
            p0: p_left,
            p1: mid,
            color0: color_to_rgba(AMETHYST, 0.65),
            color1: color_to_rgba(ULTRAVIOLET, 0.85),
            thickness: 0.022,
            pulse_speed: if is_running { 4.5 } else { 1.8 },
            pulse_phase: i as f32 * 0.7,
            intensity: if active_layer { 1.15 } else { 0.45 },
        });
        scene.filaments.push(FilamentEdge {
            p0: mid,
            p1: p_right,
            color0: color_to_rgba(ULTRAVIOLET, 0.85),
            color1: color_to_rgba(CYAN, 0.95),
            thickness: 0.022,
            pulse_speed: if is_running { 4.5 } else { 1.8 },
            pulse_phase: i as f32 * 0.7 + 1.2,
            intensity: if active_layer { 1.25 } else { 0.45 },
        });
    }

    // Central transformation manifold / shared latent ring
    scene.instances.push(CrystalInstance {
        position: [0.0, 0.0, 0.0],
        scale: [0.88, 0.88, 0.88],
        rotation: [0.6, time_sec * 0.55, 0.35],
        color: color_to_rgba(CYAN, 0.88),
        emission: 1.35,
        shape: CrystalShape::TorusRing,
        selected: 0.0,
    });
    scene.instances.push(CrystalInstance {
        position: [0.0, 0.0, 0.0],
        scale: [0.36, 0.52, 0.36],
        rotation: [0.2, -time_sec * 0.7, 0.0],
        color: color_to_rgba(ULTRAVIOLET, 0.94),
        emission: 1.25,
        shape: CrystalShape::Octahedron,
        selected: 0.0,
    });

    scene.labels.push(OverlayLabel {
        world_pos: [-2.2, 2.0, 0.0],
        text: "IMMUTABLE SOURCE (READ-ONLY)".to_owned(),
        color: AMETHYST,
        prominent: true,
    });
    scene.labels.push(OverlayLabel {
        world_pos: [2.2, 2.0, 0.0],
        text: "TRANSFORMED QMF v1".to_owned(),
        color: CYAN,
        prominent: true,
    });
    scene.add_orbital_halo([0.0, 0.0, 0.0], 500, 0.9, 3.6, 0.65, 808);
    scene
}

/// 5. Q-Genesis — The Evolutionary Crucible (3D Objective Space + Recorded Lineage Tree).
pub fn build_qgenesis_crucible_scene(
    archive: Option<&GenesisArchive>,
    profile: GenesisProfile,
    selected_candidate_id: Option<usize>,
    generation_filter: Option<usize>,
    time_sec: f32,
) -> PrismScene {
    let Some(arch) = archive else {
        let mut scene = PrismScene::new(
            BackgroundMode::CrucibleGrid,
            "THE EVOLUTIONARY CRUCIBLE (READY)",
            "Click 'Run Q-Genesis Evolution' to evolve multi-objective representation genomes in 3D space.",
            "AWAITING EVOLUTIONARY RUN",
            TEXT_MUTED,
        );
        scene.add_orbital_halo([0.0, 0.0, 0.0], 450, 1.2, 4.0, 0.5, 515);
        return scene;
    };

    let winner_id = arch.select_best(profile).map(|c| c.id);
    let mut scene = PrismScene::new(
        BackgroundMode::CrucibleGrid,
        "THE EVOLUTIONARY CRUCIBLE — 3D OBJECTIVE SPACE & RECORDED LINEAGE",
        format!(
            "{} candidates  •  {} Pareto-optimal  •  X=BPP, Y=Matvec Error, Z=FLOPs  •  Lines = actual parent_ids",
            arch.total_candidates_evaluated,
            arch.pareto_frontier.len()
        ),
        "ACTUAL MEASURED EVOLUTIONARY ARCHIVE",
        EMERALD,
    );

    let min_bpp = arch
        .all_candidates
        .iter()
        .map(|c| c.fitness.effective_bpp)
        .fold(f64::INFINITY, f64::min);
    let max_bpp = arch
        .all_candidates
        .iter()
        .map(|c| c.fitness.effective_bpp)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_err = arch
        .all_candidates
        .iter()
        .map(|c| c.fitness.mean_matvec_rel_error)
        .fold(f64::INFINITY, f64::min);
    let max_err = arch
        .all_candidates
        .iter()
        .map(|c| c.fitness.mean_matvec_rel_error)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_fl = arch
        .all_candidates
        .iter()
        .map(|c| c.fitness.estimated_matvec_flops as f64)
        .fold(f64::INFINITY, f64::min);
    let max_fl = arch
        .all_candidates
        .iter()
        .map(|c| c.fitness.estimated_matvec_flops as f64)
        .fold(f64::NEG_INFINITY, f64::max);

    let bpp_span = (max_bpp - min_bpp).max(1e-6);
    let err_span = (max_err - min_err).max(1e-6);
    let fl_span = (max_fl - min_fl).max(1.0);

    // 3D Objective Space coordinate axes
    let axis_origin = [-2.4, -1.6, -2.0];
    let axes = [
        ([2.4, -1.6, -2.0], CYAN, "EFFECTIVE BPP ->"),
        ([-2.4, 2.0, -2.0], AMETHYST, "+ MATVEC REL ERR"),
        ([-2.4, -1.6, 2.0], ULTRAVIOLET, "ESTIMATED FLOPS ->"),
    ];
    for (end, col, lbl) in axes {
        scene.filaments.push(FilamentEdge {
            p0: axis_origin,
            p1: end,
            color0: color_to_rgba(col, 0.55),
            color1: color_to_rgba(col, 0.85),
            thickness: 0.015,
            pulse_speed: 0.0,
            pulse_phase: 0.0,
            intensity: 0.7,
        });
        scene.labels.push(OverlayLabel {
            world_pos: end,
            text: lbl.to_owned(),
            color: col,
            prominent: false,
        });
    }

    let mut id_pos: HashMap<usize, [f32; 3]> = HashMap::new();
    for cand in &arch.all_candidates {
        if let Some(max_gen) = generation_filter
            && cand.generation > max_gen
        {
            continue;
        }
        let nx = ((cand.fitness.effective_bpp - min_bpp) / bpp_span) as f32;
        let ny = ((cand.fitness.mean_matvec_rel_error - min_err) / err_span) as f32;
        let nz = ((cand.fitness.estimated_matvec_flops as f64 - min_fl) / fl_span) as f32;
        let pos = [-2.1 + nx * 4.2, -1.3 + ny * 3.0, -1.7 + nz * 3.4];
        id_pos.insert(cand.id, pos);

        // Encode dominant representation family in organism geometry!
        let has_rmq = cand
            .genes
            .iter()
            .any(|g| matches!(g, LayerGene::RmqSharedBasis { .. }));
        let has_svd = cand
            .genes
            .iter()
            .any(|g| matches!(g, LayerGene::LowRankSvd { .. }));
        let shape = if has_rmq {
            CrystalShape::Octahedron
        } else if has_svd {
            CrystalShape::IcoSphere
        } else {
            CrystalShape::PrismSlab
        };

        let is_winner = winner_id == Some(cand.id);
        let is_sel = selected_candidate_id == Some(cand.id);
        let col = if is_winner {
            EMERALD
        } else if cand.on_pareto_frontier {
            CYAN
        } else {
            ULTRAVIOLET
        };
        let sz = if is_winner {
            0.22
        } else if cand.on_pareto_frontier {
            0.16
        } else {
            0.095
        };

        scene.instances.push(CrystalInstance {
            position: pos,
            scale: [sz, sz * 1.15, sz],
            rotation: [
                0.3,
                time_sec * (if cand.on_pareto_frontier { 0.8 } else { 0.2 }) + cand.id as f32,
                0.2,
            ],
            color: color_to_rgba(col, if cand.on_pareto_frontier { 0.95 } else { 0.58 }),
            emission: if is_winner {
                1.7
            } else if cand.on_pareto_frontier {
                1.2
            } else {
                0.35
            },
            shape,
            selected: if is_sel || is_winner { 1.0 } else { 0.0 },
        });
        scene.pickables.push(PickableNode {
            id: cand.id.to_string(),
            label: format!(
                "Genome #{} (Gen {}){}",
                cand.id,
                cand.generation,
                if is_winner {
                    " [WINNER]"
                } else if cand.on_pareto_frontier {
                    " • PARETO"
                } else {
                    ""
                }
            ),
            detail: format!(
                "{:.3} bpp  •  MV err {:.5}  •  {} FLOPs",
                cand.fitness.effective_bpp,
                cand.fitness.mean_matvec_rel_error,
                cand.fitness.estimated_matvec_flops
            ),
            position: pos,
            radius: sz * 1.5,
            accent: col,
        });

        // Draw ONLY recorded parent-child lineage filaments from actual `parent_ids`!
        for &pid in &cand.parent_ids {
            if let Some(&p_pos) = id_pos.get(&pid) {
                scene.filaments.push(FilamentEdge {
                    p0: p_pos,
                    p1: pos,
                    color0: color_to_rgba(AMETHYST, 0.40),
                    color1: color_to_rgba(col, 0.65),
                    thickness: if cand.on_pareto_frontier {
                        0.018
                    } else {
                        0.008
                    },
                    pulse_speed: 1.8,
                    pulse_phase: cand.id as f32 * 0.4,
                    intensity: if cand.on_pareto_frontier { 1.0 } else { 0.45 },
                });
            }
        }
    }

    // Connect Pareto frontier curve in BPP order
    let mut pareto_sorted: Vec<&qml_genesis::CandidateRecord> =
        arch.pareto_frontier.iter().collect();
    pareto_sorted.sort_by(|a, b| a.fitness.effective_bpp.total_cmp(&b.fitness.effective_bpp));
    for w in pareto_sorted.windows(2) {
        if let (Some(&p0), Some(&p1)) = (id_pos.get(&w[0].id), id_pos.get(&w[1].id)) {
            scene.filaments.push(FilamentEdge {
                p0,
                p1,
                color0: color_to_rgba(CYAN, 0.90),
                color1: color_to_rgba(EMERALD, 0.90),
                thickness: 0.026,
                pulse_speed: 2.6,
                pulse_phase: w[0].id as f32,
                intensity: 1.35,
            });
        }
    }

    scene.legend = vec![
        LegendItem {
            label: "Profile Winner".into(),
            color: EMERALD,
        },
        LegendItem {
            label: "Pareto Frontier".into(),
            color: CYAN,
        },
        LegendItem {
            label: "Dominated Trial".into(),
            color: ULTRAVIOLET,
        },
        LegendItem {
            label: "Octa=RMQ • Sphere=SVD • Slab=Quant".into(),
            color: AMETHYST,
        },
    ];
    scene.add_orbital_halo([0.0, 0.0, 0.0], 400, 2.5, 5.0, 0.4, 717);
    scene
}

/// 6. Q-Behavior — The Consciousness Observatory (Artistic Name; Classical Activation Science).
///    Dual Parallel 3D Transformer Columns with measured divergence & RSA bridges.
pub fn build_qbehavior_dual_scene(
    behavior: Option<&BehaviorComparisonReport>,
    intervention: Option<&InterventionReport>,
    selected_layer: usize,
    time_sec: f32,
) -> PrismScene {
    let layers_slice = behavior
        .map(|b| b.layers.as_slice())
        .or_else(|| intervention.map(|i| i.layers.as_slice()));

    let Some(layers) = layers_slice else {
        let mut scene = PrismScene::new(
            BackgroundMode::ConsciousnessDual,
            "Q-BEHAVIOR ACTIVATION OBSERVATORY (AWAITING PROBE)",
            "Compare a Reference QMF vs Candidate QMF or run a reversible MLP/Attention ablation.",
            "CLASSICAL ACTIVATION SCIENCE • NO PROBE LOADED",
            TEXT_MUTED,
        );
        scene.add_orbital_halo([0.0, 0.0, 0.0], 350, 1.8, 4.4, 0.5, 303);
        return scene;
    };

    let (title, subtitle) = if let Some(b) = behavior {
        (
            "DUAL NEURAL ACTIVATION OBSERVATORY — REFERENCE vs TRANSFORMED".to_owned(),
            format!(
                "{} layers traced  •  Peak Drift: L{} (Rel L2={:.4})  •  Final Cos={:.4}  •  Mean KL={:.5}",
                b.num_layers,
                b.peak_divergence_layer,
                b.peak_rel_l2_divergence,
                b.final_layer_cosine,
                b.mean_logit_kl_divergence
            ),
        )
    } else if let Some(iv) = intervention {
        (
            format!("REVERSIBLE CAUSAL INTERVENTION — {:?}", iv.intervention),
            format!(
                "{} layers traced  •  Mean Logit KL={:.5}  •  Top-1 Flip Rate={:.1}%",
                iv.layers.len(),
                iv.mean_logit_kl_divergence,
                iv.top1_flip_rate * 100.0
            ),
        )
    } else {
        unreachable!()
    };

    let mut scene = PrismScene::new(
        BackgroundMode::ConsciousnessDual,
        title,
        subtitle,
        "ACTUAL CAPTURED ACTIVATION TRAJECTORIES (CLASSICAL)",
        CYAN,
    );

    let n = layers.len().max(1);
    let y_step = (6.2 / n as f32).clamp(0.16, 0.65);
    let y_mid = (n as f32 - 1.0) * 0.5 * y_step;

    for m in layers {
        let y = m.layer_index as f32 * y_step - y_mid;
        let drift = (m.mean_rel_l2_divergence as f32 / 0.65).clamp(0.0, 1.0);
        let bridge_col = if drift < 0.25 {
            CYAN
        } else if drift < 0.45 {
            ULTRAVIOLET
        } else if drift < 0.70 {
            AMBER
        } else {
            CRIMSON
        };
        let is_sel = m.layer_index == selected_layer;

        let p_ref = [-1.45, y, 0.0];
        let p_cand = [1.45, y, 0.0];

        // Left column: Reference model layer
        scene.instances.push(CrystalInstance {
            position: p_ref,
            scale: [0.58, (y_step * 0.48).clamp(0.06, 0.20), 0.42],
            rotation: [0.0, 0.0, 0.0],
            color: color_to_rgba(CYAN, 0.85),
            emission: 0.7,
            shape: CrystalShape::PrismSlab,
            selected: if is_sel { 0.7 } else { 0.0 },
        });
        // Right column: Transformed / Intervened model layer (glows crimson/amber on real drift!)
        scene.instances.push(CrystalInstance {
            position: p_cand,
            scale: [0.58, (y_step * 0.48).clamp(0.06, 0.20), 0.42],
            rotation: [
                0.0,
                if drift > 0.55 {
                    (time_sec * 1.5).sin() * 0.12
                } else {
                    0.0
                },
                0.0,
            ],
            color: color_to_rgba(bridge_col, 0.90),
            emission: 0.65 + drift * 1.35,
            shape: CrystalShape::PrismSlab,
            selected: if is_sel { 1.0 } else { 0.0 },
        });

        // Coherence / Divergence bridge filament between Reference and Candidate layer
        scene.filaments.push(FilamentEdge {
            p0: p_ref,
            p1: p_cand,
            color0: color_to_rgba(CYAN, 0.65),
            color1: color_to_rgba(bridge_col, 0.92),
            thickness: 0.014 + drift * 0.042,
            pulse_speed: 1.5 + drift * 4.5,
            pulse_phase: m.layer_index as f32 * 0.35,
            intensity: 0.75 + drift * 1.45,
        });

        scene.pickables.push(PickableNode {
            id: m.layer_index.to_string(),
            label: format!("Layer {:02}", m.layer_index),
            detail: format!(
                "Cos={:.4}  •  Rel L2={:.4}  •  RSA={:.4}",
                m.mean_cosine_similarity, m.mean_rel_l2_divergence, m.rsa_correlation
            ),
            position: [0.0, y, 0.0],
            radius: 0.28,
            accent: bridge_col,
        });
    }

    scene.labels.push(OverlayLabel {
        world_pos: [-1.45, y_mid + 0.5, 0.0],
        text: "REFERENCE MODEL".to_owned(),
        color: CYAN,
        prominent: true,
    });
    scene.labels.push(OverlayLabel {
        world_pos: [1.45, y_mid + 0.5, 0.0],
        text: "CANDIDATE / INTERVENED".to_owned(),
        color: AMETHYST,
        prominent: true,
    });
    scene.legend = vec![
        LegendItem {
            label: "Stable Pathway (Low L2)".into(),
            color: CYAN,
        },
        LegendItem {
            label: "Moderate Drift".into(),
            color: ULTRAVIOLET,
        },
        LegendItem {
            label: "Elevated Drift".into(),
            color: AMBER,
        },
        LegendItem {
            label: "Severe Cliff (High L2 / RSA Drop)".into(),
            color: CRIMSON,
        },
    ];
    scene.add_orbital_halo([0.0, 0.0, 0.0], 300, 2.2, 4.6, 0.35, 909);
    scene
}

/// 7. Hardware Observatory — Computational Anatomy (Procedural CPU + WGPU Core Schematic).
///    Strictly reflects discovered host CPU cores & WGPU adapters; never fabricates temps/power.
pub fn build_hardware_anatomy_scene(hw: &HardwareReport, time_sec: f32) -> PrismScene {
    let has_gpu = !hw.gpus.is_empty();
    let gpu_label = hw
        .gpus
        .first()
        .map(|g| format!("{} ({})", g.name, g.backend))
        .unwrap_or_else(|| "No WGPU Adapter Detected".to_owned());

    let mut scene = PrismScene::new(
        BackgroundMode::HardwareSchematic,
        "COMPUTATIONAL ANATOMY — HOST SILICON SCHEMATIC",
        format!(
            "{} ({} logical threads)  •  {}",
            hw.cpu_brand, hw.logical_cores, gpu_label
        ),
        "DISCOVERED HARDWARE TOPOLOGY (SCHEMATIC)",
        EMERALD,
    );

    // Left: CPU Core Array (scaled to actual logical_cores, capped visually to 32 tiles)
    let cores = hw.logical_cores.clamp(1, 32);
    let cols = (cores as f32).sqrt().ceil() as usize;
    for i in 0..cores {
        let r = i / cols;
        let c = i % cols;
        let x = -2.2 + (c as f32 - (cols as f32 - 1.0) * 0.5) * 0.34;
        let z = (r as f32 - 2.0) * 0.34;
        let pulse = 0.5 + 0.5 * (time_sec * 1.6 + i as f32 * 0.4).sin();
        scene.instances.push(CrystalInstance {
            position: [x, 0.0, z],
            scale: [0.24, 0.14 + pulse * 0.06, 0.24],
            rotation: [0.0, 0.0, 0.0],
            color: color_to_rgba(ULTRAVIOLET, 0.88),
            emission: 0.65 + pulse * 0.4,
            shape: CrystalShape::PrismSlab,
            selected: 0.0,
        });
    }
    scene.pickables.push(PickableNode {
        id: "cpu".to_owned(),
        label: hw.cpu_brand.clone(),
        detail: format!(
            "{} logical threads  •  SIMD: {}",
            hw.logical_cores,
            hw.simd_features.join(", ")
        ),
        position: [-2.2, 0.2, 0.0],
        radius: 0.9,
        accent: ULTRAVIOLET,
    });
    scene.labels.push(OverlayLabel {
        world_pos: [-2.2, 1.05, 0.0],
        text: format!("CPU ARRAY ({} THREADS)", hw.logical_cores),
        color: AMETHYST,
        prominent: true,
    });

    // Right: WGPU Discrete/Compute Core Matrix (when adapter discovered)
    let gpu_tiles = if has_gpu { 24 } else { 4 };
    for i in 0..gpu_tiles {
        let r = i / 6;
        let c = i % 6;
        let x = 2.0 + (c as f32 - 2.5) * 0.32;
        let z = (r as f32 - 1.5) * 0.36;
        let col = if has_gpu { CYAN } else { TEXT_MUTED };
        let pulse = 0.5 + 0.5 * (time_sec * 2.2 + i as f32 * 0.3).sin();
        scene.instances.push(CrystalInstance {
            position: [x, 0.0, z],
            scale: [0.22, 0.18 + pulse * 0.08, 0.22],
            rotation: [0.0, 0.0, 0.0],
            color: color_to_rgba(col, 0.90),
            emission: if has_gpu { 0.85 + pulse * 0.55 } else { 0.2 },
            shape: CrystalShape::Octahedron,
            selected: 0.0,
        });
    }
    scene.pickables.push(PickableNode {
        id: "gpu".to_owned(),
        label: gpu_label.clone(),
        detail: if let Some(g) = hw.gpus.first() {
            format!(
                "Max Buffer: {:.0} MiB  •  Workgroup X: {}",
                g.max_buffer_size as f64 / 1_048_576.0,
                g.max_compute_workgroup_size_x
            )
        } else {
            "CPU Rayon fallback active".to_owned()
        },
        position: [2.0, 0.2, 0.0],
        radius: 1.0,
        accent: CYAN,
    });
    scene.labels.push(OverlayLabel {
        world_pos: [2.0, 1.05, 0.0],
        text: if has_gpu {
            "WGPU COMPUTE ENGINE".to_owned()
        } else {
            "NO GPU ADAPTER".to_owned()
        },
        color: CYAN,
        prominent: true,
    });

    // PCIe / Memory Interconnect bus filaments between CPU & GPU
    for i in 0..6 {
        let z = (i as f32 - 2.5) * 0.26;
        scene.filaments.push(FilamentEdge {
            p0: [-1.15, 0.05, z],
            p1: [1.0, 0.05, z],
            color0: color_to_rgba(ULTRAVIOLET, 0.75),
            color1: color_to_rgba(CYAN, 0.90),
            thickness: 0.018,
            pulse_speed: 3.0,
            pulse_phase: i as f32 * 0.55,
            intensity: 1.05,
        });
    }

    scene.add_orbital_halo([0.0, 0.0, 0.0], 320, 2.5, 4.6, 0.25, 4070);
    scene
}

fn heat_color(t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.33 {
        let u = t / 0.33;
        lerp_c(CYAN, ULTRAVIOLET, u)
    } else if t < 0.66 {
        let u = (t - 0.33) / 0.33;
        lerp_c(ULTRAVIOLET, AMBER, u)
    } else {
        let u = (t - 0.66) / 0.34;
        lerp_c(AMBER, CRIMSON, u)
    }
}

fn lerp_c(a: Color32, b: Color32, t: f32) -> Color32 {
    let l = |x: u8, y: u8| {
        ((x as f32) + (y as f32 - x as f32) * t)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Color32::from_rgb(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()))
}

fn short_name(s: &str) -> &str {
    s.strip_prefix("model.layers.").unwrap_or(s)
}
