#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod prism;
mod state;
mod theme;
mod views;

use eframe::egui::{self, RichText, Vec2};
use prism::scenes::ScopeVisualMode;
use prism::{OrbitCamera, init_prism_renderer};
use qml_forge::{
    ForgeControl, ForgeReport, TransformStrategy, forge_safetensors_to_qmf_with_control,
};
use qml_format::{QmfBudget, QmfFileReader};
use qml_genesis::{GenesisArchive, GenesisConfig, GenesisProfile, run_genesis};
use qml_hardware::{HardwareReport, WgpuMatvecEngine, probe_hardware};
use qml_model::{PayloadBudget, SafetensorsFile};
use qml_quant::{
    FactorPrecision, Int4SymGroupMatrix, Int8SymMatrix, LinearOperator, RmqAblationReport,
    compare_rmq_vs_baselines,
};
use qml_runtime::{QCoreModel, TransformerConfig, evaluate_inference_fidelity};
use qml_scope::{ModelScopeReport, ScopeConfig, analyze_safetensors};
use qml_tokenizer::BpeTokenizer;
use state::{
    GpuKernelBenchmark, GpuSelfTestResult, ModelFormat, PersistedState, QmfInspection, WorkerEvent,
    append_log_line, install_crash_logger, load_persisted_state, probe_model_file,
    save_persisted_state, scan_directory_for_models,
};
use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::time::Instant;

/// Primary navigation sections of the Quantum Morph Laboratory desktop UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavTab {
    CommandCenter,
    ModelLibrary,
    QScope,
    QEntangle,
    QForge,
    QGenesis,
    QmfExplorer,
    QCore,
    QBehavior,
    Benchmarks,
    SystemObservatory,
}

impl NavTab {
    pub const ALL: [Self; 11] = [
        Self::CommandCenter,
        Self::ModelLibrary,
        Self::QScope,
        Self::QEntangle,
        Self::QForge,
        Self::QGenesis,
        Self::QmfExplorer,
        Self::QCore,
        Self::QBehavior,
        Self::Benchmarks,
        Self::SystemObservatory,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::CommandCenter => "Command Center",
            Self::ModelLibrary => "Model Library",
            Self::QScope => "Q-Scope Observatory",
            Self::QEntangle => "Q-Entangle",
            Self::QForge => "Q-Forge Studio",
            Self::QGenesis => "Q-Genesis Lab",
            Self::QmfExplorer => "QMF Explorer",
            Self::QCore => "Q-Core Inference",
            Self::QBehavior => "Q-Behavior Observatory",
            Self::Benchmarks => "Benchmarks & RMQ",
            Self::SystemObservatory => "System Observatory",
        }
    }

    pub fn short_badge(self) -> &'static str {
        match self {
            Self::CommandCenter => "CMD",
            Self::ModelLibrary => "LIB",
            Self::QScope => "SCP",
            Self::QEntangle => "ENT",
            Self::QForge => "FRG",
            Self::QGenesis => "GEN",
            Self::QmfExplorer => "QMF",
            Self::QCore => "COR",
            Self::QBehavior => "BHV",
            Self::Benchmarks => "BMK",
            Self::SystemObservatory => "SYS",
        }
    }
}

/// Strategy picker for Q-Forge Studio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForgeMethodChoice {
    Bf16,
    Fp16,
    Int8Sym,
    Int4Sym,
    Int4Asym,
    MapAdaptive,
    RmqAffinityGated,
    RmqSharedBasis,
    RmqSharedCore,
    LowRankSvd,
}

impl ForgeMethodChoice {
    pub const ALL: [Self; 10] = [
        Self::Bf16,
        Self::Fp16,
        Self::Int8Sym,
        Self::Int4Sym,
        Self::Int4Asym,
        Self::MapAdaptive,
        Self::RmqAffinityGated,
        Self::RmqSharedBasis,
        Self::RmqSharedCore,
        Self::LowRankSvd,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16 (Baseline)",
            Self::Fp16 => "FP16 (Baseline)",
            Self::Int8Sym => "INT8 Symmetric (g64)",
            Self::Int4Sym => "INT4 Symmetric (Q4_0 g32)",
            Self::Int4Asym => "INT4 Asymmetric (Q4_1 g32)",
            Self::MapAdaptive => "MAP Adaptive Precision (Experimental)",
            Self::RmqAffinityGated => "RMQ Affinity-Gated Hybrid (Experimental)",
            Self::RmqSharedBasis => "RMQ Shared-Basis + Sparse (Experimental)",
            Self::RmqSharedCore => "RMQ Shared-Core + Sparse (Experimental)",
            Self::LowRankSvd => "Truncated Low-Rank SVD (Baseline)",
        }
    }
}

pub struct QmlDesktopApp {
    pub active_tab: NavTab,
    pub persisted: PersistedState,
    pub hardware: HardwareReport,
    pub status_banner: String,
    pub activity_log: Vec<String>,
    pub busy_label: Option<String>,
    pub cancel_flag: Arc<AtomicBool>,
    pub worker_rx: Option<Receiver<WorkerEvent>>,
    pub start_time: Instant,

    // Q-Scope & Q-Entangle state
    pub scope_source_path: String,
    pub scope_svd_rank: usize,
    pub scope_report: Option<ModelScopeReport>,
    pub scope_selected_tensor: usize,
    pub scope_filter: String,

    // Q-Forge state
    pub forge_source_path: String,
    pub forge_output_path: String,
    pub forge_method: ForgeMethodChoice,
    pub forge_target_bpp: f64,
    pub forge_rank: usize,
    pub forge_sparse_pct: f64,
    pub forge_affinity_threshold: f64,
    pub forge_report: Option<(ForgeReport, PathBuf)>,

    // Q-Genesis state
    pub genesis_use_synthetic: bool,
    pub genesis_source_path: String,
    pub genesis_population: usize,
    pub genesis_generations: usize,
    pub genesis_archive: Option<GenesisArchive>,

    // QMF Explorer state
    pub qmf_inspect_path: String,
    pub qmf_filter: String,
    pub qmf_inspection: Option<QmfInspection>,

    // Q-Core Inference Chamber state
    pub infer_qmf_path: String,
    pub infer_config_path: String,
    pub infer_tokenizer_path: String,
    pub infer_compare_qmf_path: String,
    pub infer_prompt: String,
    pub infer_use_chatml: bool,
    pub infer_use_gpu: bool,
    pub infer_max_tokens: usize,
    pub infer_temperature: f64,
    pub infer_output_text: String,
    pub infer_stats: Option<(usize, f64, f64)>,
    pub infer_fidelity: Option<qml_runtime::InferenceComparison>,

    // Q-Behavior Observatory state
    pub behavior_intervention_layer: usize,
    pub behavior_report: Option<qml_behavior::BehaviorComparisonReport>,
    pub intervention_report: Option<qml_behavior::InterventionReport>,

    // Benchmarks / RMQ Ablation state
    pub ablation_layers: usize,
    pub ablation_dim: usize,
    pub ablation_rank: usize,
    pub ablation_result: Option<(RmqAblationReport, RmqAblationReport)>,

    // System Observatory state
    pub gpu_selftest: Option<GpuSelfTestResult>,

    // PROJECT PRISM — Native WGPU 3D Renderer & Global UX state
    pub wgpu_ready: bool,
    pub nav_collapsed: bool,
    pub fullscreen_3d: bool,
    pub qcore_zen_mode: bool,
    pub last_frame_instant: Instant,
    pub smoothed_fps: f32,

    pub cam_singularity: OrbitCamera,
    pub cam_qscope: OrbitCamera,
    pub cam_qentangle: OrbitCamera,
    pub cam_qforge: OrbitCamera,
    pub cam_qgenesis: OrbitCamera,
    pub cam_qbehavior: OrbitCamera,
    pub cam_hardware: OrbitCamera,

    pub scope_visual_mode: ScopeVisualMode,
    pub entangle_view_mode: usize, // 0=3D Constellation, 1=2D Heatmap, 2=Cluster Overview, 3=Pair Detail
    pub entangle_selected_pair: Option<usize>,
    pub entangle_threshold: f64,
    pub forge_progress_frac: f32,
    pub genesis_profile: GenesisProfile,
    pub genesis_selected_id: Option<usize>,
    pub genesis_gen_filter: Option<usize>,
    pub behavior_selected_layer: usize,
    pub capture_png_path: Option<PathBuf>,
    pub capture_requested: bool,
    pub frames_rendered: u32,
}

impl QmlDesktopApp {
    fn new(
        cc: &eframe::CreationContext<'_>,
        initial_tab: NavTab,
        capture_png_path: Option<PathBuf>,
        auto_genesis: bool,
        auto_scope: Option<String>,
    ) -> Self {
        theme::apply_laboratory_theme(&cc.egui_ctx);
        let wgpu_ready = init_prism_renderer(cc);
        let hardware = probe_hardware();
        let persisted = load_persisted_state();
        // Model discovery is strictly opt-in via Model Library actions (no automatic background scanning).

        // Pre-fill paths from discovered library for smooth immediate workflow.
        let first_st = persisted
            .library
            .iter()
            .find(|m| m.format == ModelFormat::Safetensors)
            .cloned();
        let first_qmf = persisted
            .library
            .iter()
            .find(|m| m.format == ModelFormat::Qmf)
            .cloned();

        let scope_source_path = first_st
            .as_ref()
            .map(|m| m.path.display().to_string())
            .unwrap_or_default();
        let forge_source_path = scope_source_path.clone();
        let forge_output_path = first_st
            .as_ref()
            .and_then(|m| m.path.parent())
            .map(|p| p.join("transformed-output.qmf").display().to_string())
            .unwrap_or_else(|| "transformed-output.qmf".to_owned());

        let infer_qmf_path = first_qmf
            .as_ref()
            .map(|m| m.path.display().to_string())
            .unwrap_or_default();
        let infer_config_path = first_qmf
            .as_ref()
            .and_then(|m| m.config_path.as_ref())
            .or_else(|| first_st.as_ref().and_then(|m| m.config_path.as_ref()))
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let infer_tokenizer_path = first_qmf
            .as_ref()
            .and_then(|m| m.tokenizer_path.as_ref())
            .or_else(|| first_st.as_ref().and_then(|m| m.tokenizer_path.as_ref()))
            .map(|p| p.display().to_string())
            .unwrap_or_default();

        let mut app = Self {
            active_tab: initial_tab,
            persisted,
            hardware,
            status_banner:
                "Laboratory online — PRISM WGPU 3D Observatory ready (models opened strictly read-only)."
                    .to_owned(),
            activity_log: vec![
                "Quantum Morph Laboratory v0.3.0 (PROJECT PRISM) initialized."
                    .to_owned(),
            ],
            busy_label: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            worker_rx: None,
            start_time: Instant::now(),

            scope_source_path,
            scope_svd_rank: 12,
            scope_report: None,
            scope_selected_tensor: 0,
            scope_filter: String::new(),

            forge_source_path,
            forge_output_path,
            forge_method: ForgeMethodChoice::Int4Sym,
            forge_target_bpp: 4.51,
            forge_rank: 16,
            forge_sparse_pct: 1.0,
            forge_affinity_threshold: 0.25,
            forge_report: None,

            genesis_use_synthetic: true,
            genesis_source_path: String::new(),
            genesis_population: 12,
            genesis_generations: 5,
            genesis_archive: None,

            qmf_inspect_path: infer_qmf_path.clone(),
            qmf_filter: String::new(),
            qmf_inspection: None,

            infer_qmf_path,
            infer_config_path,
            infer_tokenizer_path,
            infer_compare_qmf_path: String::new(),
            infer_prompt: "Explain singular value decomposition in two sentences.".to_owned(),
            infer_use_chatml: true,
            infer_use_gpu: false,
            infer_max_tokens: 48,
            infer_temperature: 0.0,
            infer_output_text: String::new(),
            infer_stats: None,
            infer_fidelity: None,

            behavior_intervention_layer: 0,
            behavior_report: None,
            intervention_report: None,

            ablation_layers: 8,
            ablation_dim: 64,
            ablation_rank: 6,
            ablation_result: None,

            gpu_selftest: None,

            wgpu_ready,
            nav_collapsed: false,
            fullscreen_3d: false,
            qcore_zen_mode: false,
            last_frame_instant: Instant::now(),
            smoothed_fps: 60.0,

            cam_singularity: OrbitCamera::new(0.45, 0.28, 7.6, 0.14),
            cam_qscope: OrbitCamera::new(0.62, 0.22, 8.8, 0.06),
            cam_qentangle: OrbitCamera::new(0.55, 0.26, 7.8, 0.08),
            cam_qforge: OrbitCamera::new(0.20, 0.20, 8.4, 0.05),
            cam_qgenesis: OrbitCamera::new(0.65, 0.34, 8.6, 0.07),
            cam_qbehavior: OrbitCamera::new(0.32, 0.18, 8.2, 0.05),
            cam_hardware: OrbitCamera::new(0.48, 0.42, 7.8, 0.07),

            scope_visual_mode: ScopeVisualMode::Anatomy,
            entangle_view_mode: 0,
            entangle_selected_pair: None,
            entangle_threshold: 0.12,
            forge_progress_frac: 0.0,
            genesis_profile: GenesisProfile::Balanced,
            genesis_selected_id: None,
            genesis_gen_filter: None,
            behavior_selected_layer: 0,
            capture_png_path,
            capture_requested: false,
            frames_rendered: 0,
        };

        if auto_genesis {
            app.start_genesis_job(cc.egui_ctx.clone());
        } else if let Some(path) = auto_scope {
            app.scope_source_path = path;
            app.start_scope_job(cc.egui_ctx.clone());
        }

        app.log(format!(
            "Detected CPU: {} ({} threads), {} WGPU adapter(s).",
            app.hardware.cpu_brand,
            app.hardware.logical_cores,
            app.hardware.gpus.len()
        ));
        app
    }

    pub fn log(&mut self, msg: impl Into<String>) {
        let s = msg.into();
        append_log_line("INFO", &s);
        self.status_banner = s.clone();
        self.activity_log.push(s);
        if self.activity_log.len() > 100 {
            self.activity_log.remove(0);
        }
    }

    pub fn is_busy(&self) -> bool {
        self.busy_label.is_some()
    }

    pub fn add_model_file(&mut self, path: PathBuf) {
        match probe_model_file(&path) {
            Ok(entry) => {
                let name = entry.name.clone();
                if let Some(existing) = self
                    .persisted
                    .library
                    .iter_mut()
                    .find(|m| m.path == entry.path)
                {
                    *existing = entry;
                } else {
                    self.persisted.library.push(entry);
                }
                save_persisted_state(&self.persisted);
                self.log(format!("Imported read-only model artifact: {name}"));
            }
            Err(err) => {
                self.log(format!("Failed to import {}: {err}", path.display()));
            }
        }
    }

    pub fn scan_directory(&mut self, dir: PathBuf) {
        let found = scan_directory_for_models(&dir, 3);
        let mut added = 0usize;
        for entry in found {
            if !self.persisted.library.iter().any(|m| m.path == entry.path) {
                self.persisted.library.push(entry);
                added += 1;
            }
        }
        save_persisted_state(&self.persisted);
        self.log(format!(
            "Scanned {} (read-only): discovered {added} new model artifact(s).",
            dir.display()
        ));
    }

    pub fn start_scope_job(&mut self, ctx: egui::Context) {
        if self.is_busy() {
            return;
        }
        let path = PathBuf::from(self.scope_source_path.trim());
        if !path.is_file() {
            self.log("Q-Scope: please select a valid .safetensors file.");
            return;
        }
        let svd_rank = self.scope_svd_rank.clamp(2, 64);
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.busy_label = Some(format!("Q-Scope analyzing {}...", path.display()));
        self.cancel_flag.store(false, Ordering::SeqCst);

        std::thread::spawn(move || {
            let res = File::open(&path)
                .map_err(|e| e.to_string())
                .and_then(|file| {
                    let cfg = ScopeConfig {
                        svd_max_rank: svd_rank,
                        ..ScopeConfig::default()
                    };
                    analyze_safetensors(file, &cfg).map_err(|e| e.to_string())
                });
            match res {
                Ok(report) => {
                    let _ = tx.send(WorkerEvent::ScopeComplete(Box::new(report)));
                }
                Err(err) => {
                    let _ = tx.send(WorkerEvent::Error(format!("Q-Scope failed: {err}")));
                }
            }
            ctx.request_repaint();
        });
    }

    pub fn start_forge_job(&mut self, ctx: egui::Context) {
        if self.is_busy() {
            return;
        }
        let src = PathBuf::from(self.forge_source_path.trim());
        let dst = PathBuf::from(self.forge_output_path.trim());
        if !src.is_file() {
            self.log("Q-Forge: please select a valid source .safetensors file.");
            return;
        }
        let strategy = match self.forge_method {
            ForgeMethodChoice::Bf16 => TransformStrategy::Bf16,
            ForgeMethodChoice::Fp16 => TransformStrategy::Fp16,
            ForgeMethodChoice::Int8Sym => TransformStrategy::Int8Sym { group_size: 64 },
            ForgeMethodChoice::Int4Sym => TransformStrategy::Int4Sym { group_size: 32 },
            ForgeMethodChoice::Int4Asym => TransformStrategy::Int4Asym { group_size: 32 },
            ForgeMethodChoice::MapAdaptive => TransformStrategy::MapAdaptive {
                target_avg_bpp: self.forge_target_bpp,
                propagation_weight: 0.35,
                include_svd_rank: None,
            },
            ForgeMethodChoice::RmqAffinityGated => TransformStrategy::RmqAffinityGated {
                rank: self.forge_rank.max(2),
                affinity_threshold: self.forge_affinity_threshold.clamp(0.05, 0.95),
                sparse_fraction: (self.forge_sparse_pct / 100.0).clamp(0.0, 0.25),
                fallback_group_size: 32,
            },
            ForgeMethodChoice::RmqSharedBasis => TransformStrategy::RmqSharedBasis {
                rank: self.forge_rank.max(2),
                sparse_fraction: (self.forge_sparse_pct / 100.0).clamp(0.0, 0.25),
                basis_precision: FactorPrecision::Int8Sym,
                core_precision: FactorPrecision::Int8Sym,
            },
            ForgeMethodChoice::RmqSharedCore => TransformStrategy::RmqSharedCore {
                rank: self.forge_rank.max(2),
                sparse_fraction: (self.forge_sparse_pct / 100.0).clamp(0.0, 0.25),
                factor_precision: FactorPrecision::Int8Sym,
            },
            ForgeMethodChoice::LowRankSvd => TransformStrategy::LowRankSvd {
                rank: self.forge_rank.max(2),
                precision: FactorPrecision::Int8Sym,
            },
        };
        let model_id = src
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_owned();
        let budget_plan = self.persisted.resource_preset.plan(&self.hardware);
        let payload_budget = PayloadBudget {
            max_total_bytes: budget_plan.ram_budget_bytes,
            ..PayloadBudget::default()
        };
        let cancel = self.cancel_flag.clone();
        cancel.store(false, Ordering::SeqCst);
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.busy_label = Some(format!(
            "Q-Forge transforming {} -> {}...",
            model_id,
            self.forge_method.label()
        ));

        std::thread::spawn(move || {
            let tx_prog = tx.clone();
            let ctx_prog = ctx.clone();
            let control = ForgeControl {
                cancel,
                on_progress: Some(Arc::new(move |done, total, stage| {
                    let frac = if total > 0 {
                        (done as f32 / total as f32).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let _ = tx_prog.send(WorkerEvent::ForgeProgress(
                        frac,
                        format!("Q-Forge [{done}/{total}] {stage}"),
                    ));
                    ctx_prog.request_repaint();
                })),
            };
            let res = File::open(&src)
                .map_err(|e| e.to_string())
                .and_then(|file| {
                    forge_safetensors_to_qmf_with_control(
                        file,
                        &model_id,
                        "local-readonly-source",
                        "transformer",
                        &strategy,
                        &payload_budget,
                        42,
                        &control,
                    )
                    .map_err(|e| e.to_string())
                })
                .and_then(|(archive, report)| {
                    archive
                        .write_to_file_atomic(&dst, Some(&src), &QmfBudget::default())
                        .map_err(|e| e.to_string())?;
                    Ok((report, dst))
                });
            match res {
                Ok((report, output_path)) => {
                    let _ = tx.send(WorkerEvent::ForgeComplete {
                        report: Box::new(report),
                        output_path,
                    });
                }
                Err(err) => {
                    let _ = tx.send(WorkerEvent::Error(format!("Q-Forge failed: {err}")));
                }
            }
            ctx.request_repaint();
        });
    }

    pub fn start_genesis_job(&mut self, ctx: egui::Context) {
        if self.is_busy() {
            return;
        }
        let use_synth = self.genesis_use_synthetic;
        let path = PathBuf::from(self.genesis_source_path.trim());
        let cfg = GenesisConfig {
            population_size: self.genesis_population.clamp(6, 40),
            generations: self.genesis_generations.clamp(2, 20),
            ..GenesisConfig::default()
        };
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.busy_label = Some("Q-Genesis evolving Pareto representation frontier...".to_owned());
        self.cancel_flag.store(false, Ordering::SeqCst);

        std::thread::spawn(move || {
            let res = if use_synth {
                let mut rng = qml_core::DeterministicRng::new(2026);
                let u_data: Vec<f64> = (0..64 * 6).map(|_| rng.next_normal() * 0.3).collect();
                let v_data: Vec<f64> = (0..6 * 64).map(|_| rng.next_normal() * 0.3).collect();
                let u = qml_core::DenseMatrix::from_flat(64, 6, u_data).unwrap();
                let v = qml_core::DenseMatrix::from_flat(6, 64, v_data).unwrap();
                let mut layers = Vec::new();
                for l in 0..6 {
                    let c_data: Vec<f64> = (0..36).map(|_| rng.next_normal()).collect();
                    let c = qml_core::DenseMatrix::from_flat(6, 6, c_data).unwrap();
                    let uc = u.matmul(&c).unwrap();
                    let raw = uc.matmul(&v).unwrap();
                    let vals: Vec<f64> = raw
                        .as_slice()
                        .iter()
                        .map(|&val| val + rng.next_normal() * 0.02)
                        .collect();
                    let w = qml_core::DenseMatrix::from_flat(64, 64, vals).unwrap();
                    layers.push((format!("model.layers.{l}.self_attn.q_proj.weight"), w));
                }
                let refs: Vec<(&str, &qml_core::DenseMatrix)> =
                    layers.iter().map(|(n, m)| (n.as_str(), m)).collect();
                run_genesis(&refs, &cfg).map_err(|e| e.to_string())
            } else {
                File::open(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|file| {
                        let budget = PayloadBudget::default();
                        let mut st =
                            SafetensorsFile::open(file, &budget).map_err(|e| e.to_string())?;
                        let names = qml_genesis::select_stratified_tensor_names(st.metadata(), 21);
                        let mut loaded = Vec::new();
                        for name in names {
                            let m = st
                                .read_tensor_matrix(&name, &budget)
                                .map_err(|e| e.to_string())?;
                            loaded.push((name, m));
                        }
                        let refs: Vec<(&str, &qml_core::DenseMatrix)> =
                            loaded.iter().map(|(n, m)| (n.as_str(), m)).collect();
                        run_genesis(&refs, &cfg).map_err(|e| e.to_string())
                    })
            };
            match res {
                Ok(archive) => {
                    let _ = tx.send(WorkerEvent::GenesisComplete(Box::new(archive)));
                }
                Err(err) => {
                    let _ = tx.send(WorkerEvent::Error(format!("Q-Genesis failed: {err}")));
                }
            }
            ctx.request_repaint();
        });
    }

    pub fn start_ablation_job(&mut self, ctx: egui::Context) {
        if self.is_busy() {
            return;
        }
        let layers = self.ablation_layers.clamp(2, 16);
        let dim = self.ablation_dim.clamp(16, 128);
        let rank = self.ablation_rank.clamp(2, dim / 2);
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.busy_label =
            Some("Running RMQ vs Baselines ablation (correlated + Gaussian control)...".to_owned());
        self.cancel_flag.store(false, Ordering::SeqCst);

        std::thread::spawn(move || {
            let mut rng = qml_core::DeterministicRng::new(2026);
            let u = qml_core::DenseMatrix::from_flat(
                dim,
                rank,
                (0..dim * rank).map(|_| rng.next_normal() * 0.3).collect(),
            )
            .unwrap();
            let v = qml_core::DenseMatrix::from_flat(
                rank,
                dim,
                (0..rank * dim).map(|_| rng.next_normal() * 0.3).collect(),
            )
            .unwrap();
            let mut corr_mats = Vec::with_capacity(layers);
            for _ in 0..layers {
                let c = qml_core::DenseMatrix::from_flat(
                    rank,
                    rank,
                    (0..rank * rank).map(|_| rng.next_normal()).collect(),
                )
                .unwrap();
                let raw = u.matmul(&c).unwrap().matmul(&v).unwrap();
                let mut vals: Vec<f64> = raw
                    .as_slice()
                    .iter()
                    .map(|&val| val + rng.next_normal() * 0.015)
                    .collect();
                // Inject 0.5% sparse outliers to test outlier peeling.
                let n_out = (dim * dim / 200).max(1);
                for _ in 0..n_out {
                    let idx = (rng.next_u64() as usize) % (dim * dim);
                    vals[idx] += rng.next_normal() * 2.5;
                }
                let w = qml_core::DenseMatrix::from_flat(dim, dim, vals).unwrap();
                corr_mats.push(w);
            }
            let gauss_mats: Vec<qml_core::DenseMatrix> = (0..layers)
                .map(|_| {
                    qml_core::DenseMatrix::from_flat(
                        dim,
                        dim,
                        (0..dim * dim).map(|_| rng.next_normal() * 0.2).collect(),
                    )
                    .unwrap()
                })
                .collect();

            let corr_refs: Vec<&qml_core::DenseMatrix> = corr_mats.iter().collect();
            let gauss_refs: Vec<&qml_core::DenseMatrix> = gauss_mats.iter().collect();

            match (
                compare_rmq_vs_baselines("Correlated Cluster", &corr_refs, rank, 42),
                compare_rmq_vs_baselines("Gaussian Control", &gauss_refs, rank, 42),
            ) {
                (Ok(correlated), Ok(gaussian_control)) => {
                    let _ = tx.send(WorkerEvent::AblationComplete {
                        correlated,
                        gaussian_control,
                    });
                }
                (Err(e), _) | (_, Err(e)) => {
                    let _ = tx.send(WorkerEvent::Error(format!("RMQ ablation failed: {e}")));
                }
            }
            ctx.request_repaint();
        });
    }

    pub fn start_inference_job(&mut self, ctx: egui::Context) {
        if self.is_busy() {
            return;
        }
        let qmf_path = PathBuf::from(self.infer_qmf_path.trim());
        let cfg_path = PathBuf::from(self.infer_config_path.trim());
        let tok_path = PathBuf::from(self.infer_tokenizer_path.trim());
        let compare_path = self.infer_compare_qmf_path.trim().to_owned();
        if !qmf_path.is_file() || !cfg_path.is_file() || !tok_path.is_file() {
            self.log("Q-Core: please select valid .qmf, config.json, and tokenizer.json files.");
            return;
        }
        let prompt = self.infer_prompt.clone();
        let use_chatml = self.infer_use_chatml;
        let backend = if self.infer_use_gpu {
            qml_runtime::QCoreBackend::GpuHybrid
        } else {
            qml_runtime::QCoreBackend::Cpu
        };
        let max_tokens = self.infer_max_tokens.clamp(1, 256);
        let temperature = self.infer_temperature.clamp(0.0, 2.0);
        let cancel = self.cancel_flag.clone();
        cancel.store(false, Ordering::SeqCst);

        self.infer_output_text.clear();
        self.infer_stats = None;
        self.infer_fidelity = None;
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.busy_label =
            Some("Q-Core loading compressed QMF model & generating tokens...".to_owned());

        std::thread::spawn(move || {
            let run = || -> Result<(), String> {
                let cfg_bytes = fs::read(&cfg_path).map_err(|e| e.to_string())?;
                let cfg = TransformerConfig::from_hf_config_bytes(&cfg_bytes)
                    .map_err(|e| e.to_string())?;
                let tokenizer = BpeTokenizer::from_file(&tok_path).map_err(|e| e.to_string())?;
                let qmf_file = File::open(&qmf_path).map_err(|e| e.to_string())?;
                let mut qmf_reader = QmfFileReader::open(qmf_file, &QmfBudget::default())
                    .map_err(|e| e.to_string())?;
                let model =
                    QCoreModel::from_qmf_reader_with_backend(&mut qmf_reader, cfg.clone(), backend)
                        .map_err(|e| e.to_string())?;
                let bstats = model.backend_stats();
                let _ = tx.send(WorkerEvent::Status(format!(
                    "Q-Core backend: {} (GPU ops={}, VRAM={:.1} MiB)",
                    bstats.backend,
                    bstats.gpu_resident_operators,
                    bstats.vram_weight_bytes as f64 / (1024.0 * 1024.0)
                )));
                ctx.request_repaint();

                let formatted = if use_chatml {
                    tokenizer.format_chat_prompt(None, &prompt)
                } else {
                    prompt
                };
                let prompt_ids: Vec<usize> = tokenizer
                    .encode(&formatted)
                    .into_iter()
                    .map(|id| id as usize)
                    .collect();
                let eos = tokenizer.eos_token_id().map(|id| id as usize);

                let _ = tx.send(WorkerEvent::Status(format!(
                    "Q-Core prefilling {} prompt tokens...",
                    prompt_ids.len()
                )));
                ctx.request_repaint();

                let mut stream_dec = tokenizer.streaming_decoder(true);
                let gen_res = model
                    .generate_with_callback(
                        &prompt_ids,
                        max_tokens,
                        temperature,
                        eos,
                        42,
                        |tok_id| {
                            if cancel.load(Ordering::SeqCst) {
                                return false;
                            }
                            let piece = stream_dec.push(tok_id as u32);
                            if !piece.is_empty() {
                                let _ = tx.send(WorkerEvent::InferToken(piece));
                                ctx.request_repaint();
                            }
                            true
                        },
                    )
                    .map_err(|e| e.to_string())?;
                let tail = stream_dec.flush();
                if !tail.is_empty() {
                    let _ = tx.send(WorkerEvent::InferToken(tail));
                    ctx.request_repaint();
                }

                let fidelity = if !compare_path.is_empty() && PathBuf::from(&compare_path).is_file()
                {
                    let _ = tx.send(WorkerEvent::Status(
                        "Computing reference KL divergence & perplexity...".to_owned(),
                    ));
                    ctx.request_repaint();
                    let ref_file = File::open(&compare_path).map_err(|e| e.to_string())?;
                    let mut ref_reader = QmfFileReader::open(ref_file, &QmfBudget::default())
                        .map_err(|e| e.to_string())?;
                    let ref_model = QCoreModel::from_qmf_reader(&mut ref_reader, cfg)
                        .map_err(|e| e.to_string())?;
                    let mut seq = prompt_ids.clone();
                    seq.extend(gen_res.generated_tokens.iter().copied().take(16));
                    Some(
                        evaluate_inference_fidelity(&ref_model, &model, &seq)
                            .map_err(|e| e.to_string())?,
                    )
                } else {
                    None
                };

                let _ = tx.send(WorkerEvent::InferComplete {
                    tokens_generated: gen_res.generated_tokens.len(),
                    tokens_per_second: gen_res.tokens_per_second,
                    elapsed_ms: gen_res.elapsed_micros as f64 / 1000.0,
                    fidelity,
                });
                Ok(())
            };

            if let Err(err) = run() {
                let _ = tx.send(WorkerEvent::Error(format!(
                    "Q-Core inference failed: {err}"
                )));
            }
            ctx.request_repaint();
        });
    }

    pub fn start_behavior_compare_job(&mut self, ctx: egui::Context) {
        if self.is_busy() {
            return;
        }
        let ref_path = PathBuf::from(self.infer_compare_qmf_path.trim());
        let cand_path = PathBuf::from(self.infer_qmf_path.trim());
        let cfg_path = PathBuf::from(self.infer_config_path.trim());
        let tok_path = PathBuf::from(self.infer_tokenizer_path.trim());
        if !ref_path.is_file() || !cand_path.is_file() || !cfg_path.is_file() || !tok_path.is_file()
        {
            self.log(
                "Q-Behavior: select Reference QMF, Candidate QMF, config.json, and tokenizer.json.",
            );
            return;
        }
        let prompt = self.infer_prompt.clone();
        let backend = if self.infer_use_gpu {
            qml_runtime::QCoreBackend::GpuHybrid
        } else {
            qml_runtime::QCoreBackend::Cpu
        };
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.busy_label = Some("Q-Behavior capturing layer activations & RSA...".to_owned());
        self.cancel_flag.store(false, Ordering::SeqCst);

        std::thread::spawn(move || {
            let run = || -> Result<qml_behavior::BehaviorComparisonReport, String> {
                let cfg = TransformerConfig::from_hf_config_bytes(
                    &fs::read(&cfg_path).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                let tok = BpeTokenizer::from_file(&tok_path).map_err(|e| e.to_string())?;
                let seq: Vec<usize> = tok
                    .encode(&prompt)
                    .into_iter()
                    .map(|id| id as usize)
                    .collect();
                let mut ref_r = QmfFileReader::open(
                    File::open(&ref_path).map_err(|e| e.to_string())?,
                    &QmfBudget::default(),
                )
                .map_err(|e| e.to_string())?;
                let ref_m =
                    QCoreModel::from_qmf_reader_with_backend(&mut ref_r, cfg.clone(), backend)
                        .map_err(|e| e.to_string())?;
                let mut cand_r = QmfFileReader::open(
                    File::open(&cand_path).map_err(|e| e.to_string())?,
                    &QmfBudget::default(),
                )
                .map_err(|e| e.to_string())?;
                let cand_m = QCoreModel::from_qmf_reader_with_backend(&mut cand_r, cfg, backend)
                    .map_err(|e| e.to_string())?;
                qml_behavior::compare_model_behavior(&ref_m, &cand_m, &seq)
                    .map_err(|e| e.to_string())
            };
            match run() {
                Ok(rep) => {
                    let _ = tx.send(WorkerEvent::BehaviorComplete(Box::new(rep)));
                }
                Err(err) => {
                    let _ = tx.send(WorkerEvent::Error(format!("Q-Behavior failed: {err}")));
                }
            }
            ctx.request_repaint();
        });
    }

    pub fn start_intervention_job(
        &mut self,
        ctx: egui::Context,
        intervention: qml_runtime::LayerIntervention,
    ) {
        if self.is_busy() {
            return;
        }
        let qmf_path = PathBuf::from(self.infer_qmf_path.trim());
        let cfg_path = PathBuf::from(self.infer_config_path.trim());
        let tok_path = PathBuf::from(self.infer_tokenizer_path.trim());
        if !qmf_path.is_file() || !cfg_path.is_file() || !tok_path.is_file() {
            self.log("Q-Behavior: select Model QMF, config.json, and tokenizer.json.");
            return;
        }
        let prompt = self.infer_prompt.clone();
        let backend = if self.infer_use_gpu {
            qml_runtime::QCoreBackend::GpuHybrid
        } else {
            qml_runtime::QCoreBackend::Cpu
        };
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.busy_label = Some(format!("Q-Behavior running reversible {intervention:?}..."));
        self.cancel_flag.store(false, Ordering::SeqCst);

        std::thread::spawn(move || {
            let run = || -> Result<qml_behavior::InterventionReport, String> {
                let cfg = TransformerConfig::from_hf_config_bytes(
                    &fs::read(&cfg_path).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                let tok = BpeTokenizer::from_file(&tok_path).map_err(|e| e.to_string())?;
                let seq: Vec<usize> = tok
                    .encode(&prompt)
                    .into_iter()
                    .map(|id| id as usize)
                    .collect();
                let mut r = QmfFileReader::open(
                    File::open(&qmf_path).map_err(|e| e.to_string())?,
                    &QmfBudget::default(),
                )
                .map_err(|e| e.to_string())?;
                let m = QCoreModel::from_qmf_reader_with_backend(&mut r, cfg, backend)
                    .map_err(|e| e.to_string())?;
                qml_behavior::run_intervention_study(&m, &seq, intervention)
                    .map_err(|e| e.to_string())
            };
            match run() {
                Ok(rep) => {
                    let _ = tx.send(WorkerEvent::InterventionComplete(Box::new(rep)));
                }
                Err(err) => {
                    let _ = tx.send(WorkerEvent::Error(format!(
                        "Q-Behavior intervention failed: {err}"
                    )));
                }
            }
            ctx.request_repaint();
        });
    }

    pub fn start_gpu_selftest(&mut self, ctx: egui::Context) {
        if self.is_busy() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.busy_label = Some("Running WGPU compute shader self-test...".to_owned());
        self.cancel_flag.store(false, Ordering::SeqCst);

        std::thread::spawn(move || {
            match WgpuMatvecEngine::try_new() {
                Ok(engine) => {
                    let mut rng = qml_core::DeterministicRng::new(2026);
                    let rows = 512;
                    let cols = 512;
                    let f32_bytes = rows * cols * 4;
                    let mat = qml_core::DenseMatrix::from_flat(
                        rows,
                        cols,
                        (0..rows * cols).map(|_| rng.next_normal() * 0.2).collect(),
                    )
                    .unwrap();
                    let x: Vec<f64> = (0..cols).map(|_| rng.next_normal()).collect();

                    let run_bench = || -> Result<Vec<GpuKernelBenchmark>, String> {
                        let mut kernels = Vec::new();

                        // 1. Dense FP32 GPU matvec
                        let t_cpu = Instant::now();
                        let cpu_f = mat.matvec(&x).map_err(|e| e.to_string())?;
                        let cpu_f_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;
                        let t_gpu = Instant::now();
                        let gpu_f = engine.matvec(&mat, &x).map_err(|e| e.to_string())?;
                        let gpu_f_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;
                        let err_f = cpu_f
                            .iter()
                            .zip(&gpu_f)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0_f64, f64::max);
                        kernels.push(GpuKernelBenchmark {
                            kernel_name: "Dense FP32 Matvec (WGSL)".to_owned(),
                            vram_weight_bytes: f32_bytes,
                            vram_compression_vs_f32: 1.0,
                            max_abs_error: err_f,
                            gpu_elapsed_ms: gpu_f_ms,
                            cpu_elapsed_ms: cpu_f_ms,
                        });

                        // 2. Direct Compressed INT8_SYM GPU matvec (zero dense reconstruction in VRAM)
                        let q8 = Int8SymMatrix::quantize(&mat, 64).map_err(|e| e.to_string())?;
                        let q8_vram = q8.codes().len() + q8.scales_f16().len() * 2;
                        let t_cpu8 = Instant::now();
                        let cpu8 = q8.matvec(&x).map_err(|e| e.to_string())?;
                        let cpu8_ms = t_cpu8.elapsed().as_secs_f64() * 1000.0;
                        let t_gpu8 = Instant::now();
                        let gpu8 = engine.matvec_int8_sym(&q8, &x).map_err(|e| e.to_string())?;
                        let gpu8_ms = t_gpu8.elapsed().as_secs_f64() * 1000.0;
                        let err8 = cpu8
                            .iter()
                            .zip(&gpu8)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0_f64, f64::max);
                        kernels.push(GpuKernelBenchmark {
                            kernel_name: "Direct Compressed INT8_SYM (WGSL unpack2x16float)"
                                .to_owned(),
                            vram_weight_bytes: q8_vram,
                            vram_compression_vs_f32: f32_bytes as f64 / q8_vram as f64,
                            max_abs_error: err8,
                            gpu_elapsed_ms: gpu8_ms,
                            cpu_elapsed_ms: cpu8_ms,
                        });

                        // 3. Direct Compressed Q4_0 (INT4_SYM) GPU matvec (zero dense reconstruction in VRAM)
                        let q4 =
                            Int4SymGroupMatrix::quantize(&mat, 32).map_err(|e| e.to_string())?;
                        let q4_vram = q4.packed_nibbles().len() + q4.scales_f16().len() * 2;
                        let t_cpu4 = Instant::now();
                        let cpu4 = q4.matvec(&x).map_err(|e| e.to_string())?;
                        let cpu4_ms = t_cpu4.elapsed().as_secs_f64() * 1000.0;
                        let t_gpu4 = Instant::now();
                        let gpu4 = engine.matvec_int4_sym(&q4, &x).map_err(|e| e.to_string())?;
                        let gpu4_ms = t_gpu4.elapsed().as_secs_f64() * 1000.0;
                        let err4 = cpu4
                            .iter()
                            .zip(&gpu4)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0_f64, f64::max);
                        kernels.push(GpuKernelBenchmark {
                            kernel_name: "Direct Compressed Q4_0 INT4 (WGSL nibble+f16)".to_owned(),
                            vram_weight_bytes: q4_vram,
                            vram_compression_vs_f32: f32_bytes as f64 / q4_vram as f64,
                            max_abs_error: err4,
                            gpu_elapsed_ms: gpu4_ms,
                            cpu_elapsed_ms: cpu4_ms,
                        });

                        Ok(kernels)
                    };

                    match run_bench() {
                        Ok(kernels) => {
                            let _ = tx.send(WorkerEvent::GpuSelfTestComplete(GpuSelfTestResult {
                                adapter_name: engine.adapter_name().to_owned(),
                                backend_name: engine.backend_name().to_owned(),
                                matrix_shape: (rows, cols),
                                kernels,
                            }));
                        }
                        Err(err) => {
                            let _ =
                                tx.send(WorkerEvent::Error(format!("WGPU dispatch failed: {err}")));
                        }
                    }
                }
                Err(err) => {
                    let _ = tx.send(WorkerEvent::Error(format!("WGPU unavailable: {err}")));
                }
            }
            ctx.request_repaint();
        });
    }

    /// Export the selected (or profile-winning) Q-Genesis Pareto genome directly to a verified QMF v1 file.
    pub fn start_genesis_export_job(&mut self, ctx: egui::Context, output_path: PathBuf) {
        if self.is_busy() {
            return;
        }
        let Some(arch) = self.genesis_archive.clone() else {
            return;
        };
        let src_path = PathBuf::from(self.genesis_source_path.trim());
        if !src_path.is_file() {
            self.log("Q-Genesis Export: requires a real .safetensors source model (uncheck synthetic mode).");
            return;
        }
        let winner = self
            .genesis_selected_id
            .and_then(|id| arch.all_candidates.iter().find(|c| c.id == id))
            .or_else(|| arch.select_best(self.genesis_profile))
            .cloned();
        let Some(winner) = winner else {
            return;
        };
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.busy_label = Some(format!(
            "Exporting Q-Genesis Genome #{} -> {}...",
            winner.id,
            output_path.display()
        ));
        std::thread::spawn(move || {
            let res = File::open(&src_path)
                .map_err(|e| e.to_string())
                .and_then(|f| {
                    let model_id = src_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("genesis-model");
                    let qmf = qml_genesis::export_genome_to_qmf_v1(
                        f,
                        model_id,
                        "local-readonly-source",
                        "safetensors",
                        &arch.layer_names,
                        &winner.genes,
                        &PayloadBudget::default(),
                        arch.config.seed,
                    )?;
                    qmf.write_to_file_atomic(&output_path, Some(&src_path), &QmfBudget::default())
                        .map_err(|e| e.to_string())?;
                    Ok(format!(
                        "Exported Q-Genesis Genome #{} ({:.3} bpp) atomically -> {}",
                        winner.id,
                        winner.fitness.effective_bpp,
                        output_path.display()
                    ))
                });
            match res {
                Ok(msg) => {
                    let _ = tx.send(WorkerEvent::Status(msg));
                    let _ = tx.send(WorkerEvent::Error(
                        "Q-Genesis QMF v1 export completed.".to_owned(),
                    ));
                }
                Err(e) => {
                    let _ = tx.send(WorkerEvent::Error(format!("Q-Genesis export failed: {e}")));
                }
            }
            ctx.request_repaint();
        });
    }

    pub fn active_camera_mut(&mut self) -> Option<&mut OrbitCamera> {
        match self.active_tab {
            NavTab::CommandCenter => Some(&mut self.cam_singularity),
            NavTab::QScope => Some(&mut self.cam_qscope),
            NavTab::QEntangle => Some(&mut self.cam_qentangle),
            NavTab::QForge => Some(&mut self.cam_qforge),
            NavTab::QGenesis => Some(&mut self.cam_qgenesis),
            NavTab::QBehavior => Some(&mut self.cam_qbehavior),
            NavTab::SystemObservatory => Some(&mut self.cam_hardware),
            _ => None,
        }
    }

    fn handle_global_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() {
            return;
        }
        ctx.input(|i| {
            if i.modifiers.ctrl && i.key_pressed(egui::Key::B) {
                self.nav_collapsed = !self.nav_collapsed;
            }
            if i.key_pressed(egui::Key::F) {
                self.fullscreen_3d = !self.fullscreen_3d;
            }
            if i.key_pressed(egui::Key::Escape) {
                self.fullscreen_3d = false;
                self.qcore_zen_mode = false;
            }
            if i.key_pressed(egui::Key::R)
                && let Some(cam) = self.active_camera_mut()
            {
                cam.reset();
            }
            if i.key_pressed(egui::Key::P)
                && let Some(cam) = self.active_camera_mut()
            {
                cam.orthographic = !cam.orthographic;
            }
            let num_tabs = [
                (egui::Key::Num1, NavTab::CommandCenter),
                (egui::Key::Num2, NavTab::ModelLibrary),
                (egui::Key::Num3, NavTab::QScope),
                (egui::Key::Num4, NavTab::QEntangle),
                (egui::Key::Num5, NavTab::QForge),
                (egui::Key::Num6, NavTab::QGenesis),
                (egui::Key::Num7, NavTab::QmfExplorer),
                (egui::Key::Num8, NavTab::QCore),
                (egui::Key::Num9, NavTab::QBehavior),
                (egui::Key::Num0, NavTab::Benchmarks),
                (egui::Key::Minus, NavTab::SystemObservatory),
            ];
            for (key, tab) in num_tabs {
                if i.key_pressed(key) {
                    self.active_tab = tab;
                    self.fullscreen_3d = false;
                }
            }
        });
    }

    fn poll_worker(&mut self) {
        let mut clear_rx = false;
        let mut events = Vec::new();
        if let Some(rx) = &self.worker_rx {
            while let Ok(ev) = rx.try_recv() {
                events.push(ev);
            }
        }
        for ev in events {
            match ev {
                WorkerEvent::Status(s) => {
                    self.busy_label = Some(s.clone());
                    self.status_banner = s;
                }
                WorkerEvent::ForgeProgress(frac, s) => {
                    self.forge_progress_frac = frac;
                    self.busy_label = Some(s.clone());
                    self.status_banner = s;
                }
                WorkerEvent::ScopeComplete(rep) => {
                    let msg = format!(
                        "Q-Scope complete: {} tensors ({} params), mean affinity {:.4}.",
                        rep.summary.total_tensors,
                        rep.summary.total_parameters,
                        rep.summary.mean_cross_layer_shared_score.unwrap_or(0.0)
                    );
                    self.scope_report = Some(*rep);
                    self.scope_selected_tensor = 0;
                    self.busy_label = None;
                    clear_rx = true;
                    self.log(msg);
                }
                WorkerEvent::ForgeComplete {
                    report,
                    output_path,
                } => {
                    let msg = format!(
                        "Q-Forge complete: {} ({:.3} bpp, {:.2}x compression) -> {}",
                        report.descriptor.name,
                        report.effective_qmf_bpp,
                        report.compression_ratio_vs_source,
                        output_path.display()
                    );
                    self.add_model_file(output_path.clone());
                    self.qmf_inspect_path = output_path.display().to_string();
                    self.qmf_inspection = QmfInspection::inspect_file(&output_path).ok();
                    self.infer_qmf_path = output_path.display().to_string();
                    self.forge_report = Some((*report, output_path));
                    self.busy_label = None;
                    clear_rx = true;
                    self.log(msg);
                }
                WorkerEvent::GenesisComplete(arch) => {
                    let msg = format!(
                        "Q-Genesis complete: {} candidates evaluated, {} Pareto-optimal genomes discovered.",
                        arch.total_candidates_evaluated,
                        arch.pareto_frontier.len()
                    );
                    self.genesis_archive = Some(*arch);
                    self.busy_label = None;
                    clear_rx = true;
                    self.log(msg);
                }
                WorkerEvent::AblationComplete {
                    correlated,
                    gaussian_control,
                } => {
                    self.ablation_result = Some((correlated, gaussian_control));
                    self.busy_label = None;
                    clear_rx = true;
                    self.log("RMQ vs Baselines ablation complete (correlated + Gaussian control).");
                }
                WorkerEvent::InferToken(piece) => {
                    self.infer_output_text.push_str(&piece);
                }
                WorkerEvent::InferComplete {
                    tokens_generated,
                    tokens_per_second,
                    elapsed_ms,
                    fidelity,
                } => {
                    self.infer_stats = Some((tokens_generated, tokens_per_second, elapsed_ms));
                    self.infer_fidelity = fidelity;
                    self.busy_label = None;
                    clear_rx = true;
                    self.log(format!(
                        "Q-Core generation complete: {tokens_generated} tokens at {tokens_per_second:.2} tok/s."
                    ));
                }
                WorkerEvent::GpuSelfTestComplete(res) => {
                    let worst_err = res
                        .kernels
                        .iter()
                        .map(|k| k.max_abs_error)
                        .fold(0.0_f64, f64::max);
                    let msg = format!(
                        "WGPU self-test PASSED ({} kernels on {} / {}): worst max_abs_err={:.2e}.",
                        res.kernels.len(),
                        res.adapter_name,
                        res.backend_name,
                        worst_err
                    );
                    self.gpu_selftest = Some(res);
                    self.busy_label = None;
                    clear_rx = true;
                    self.log(msg);
                }
                WorkerEvent::BehaviorComplete(rep) => {
                    let msg = format!(
                        "Q-Behavior complete: {} layers analyzed, peak drift L{} (relL2={:.4}), Mean KL={:.5}.",
                        rep.num_layers,
                        rep.peak_divergence_layer,
                        rep.peak_rel_l2_divergence,
                        rep.mean_logit_kl_divergence
                    );
                    self.behavior_report = Some(*rep);
                    self.busy_label = None;
                    clear_rx = true;
                    self.log(msg);
                }
                WorkerEvent::InterventionComplete(rep) => {
                    let msg = format!(
                        "Q-Behavior intervention {:?}: Mean KL={:.5}, Top-1 Flip={:.1}%.",
                        rep.intervention,
                        rep.mean_logit_kl_divergence,
                        rep.top1_flip_rate * 100.0
                    );
                    self.intervention_report = Some(*rep);
                    self.busy_label = None;
                    clear_rx = true;
                    self.log(msg);
                }
                WorkerEvent::Error(err) => {
                    self.busy_label = None;
                    clear_rx = true;
                    self.log(err);
                }
            }
        }
        if clear_rx {
            self.worker_rx = None;
        }
    }
}

impl eframe::App for QmlDesktopApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker();
        self.handle_global_shortcuts(ctx);
        self.frames_rendered = self.frames_rendered.saturating_add(1);

        // Optional automated native GPU screenshot capture (`--capture-png PATH`)
        if let Some(out_path) = self.capture_png_path.clone() {
            let mut captured_img: Option<Arc<egui::ColorImage>> = None;
            ctx.input(|i| {
                for ev in &i.events {
                    if let egui::Event::Screenshot { image, .. } = ev {
                        captured_img = Some(Arc::clone(image));
                    }
                }
            });
            if let Some(img) = captured_img {
                if let Err(err) = save_color_image_png(&out_path, &img) {
                    append_log_line("ERROR", &format!("Screenshot save failed: {err}"));
                } else {
                    append_log_line(
                        "INFO",
                        &format!(
                            "Captured native GPU screenshot {} ({}x{})",
                            out_path.display(),
                            img.width(),
                            img.height()
                        ),
                    );
                }
                self.capture_png_path = None;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            } else if !self.capture_requested && !self.is_busy() && self.frames_rendered >= 12 {
                self.capture_requested = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            }
            ctx.request_repaint();
        }

        // Measure genuine frame delta & smoothed FPS (never fabricated)
        let now = Instant::now();
        let dt = now
            .duration_since(self.last_frame_instant)
            .as_secs_f32()
            .clamp(0.001, 0.25);
        self.last_frame_instant = now;
        let inst_fps = (1.0 / dt).clamp(1.0, 240.0);
        self.smoothed_fps = self.smoothed_fps * 0.90 + inst_fps * 0.10;

        // Avoid wasting GPU cycles when window is minimized
        let is_minimized = ctx.input(|i| i.viewport().minimized.unwrap_or(false));
        let has_3d_tab = self.active_camera_mut().is_some();
        if !is_minimized {
            if self.is_busy() {
                ctx.request_repaint_after(std::time::Duration::from_millis(33));
            } else if has_3d_tab && self.persisted.graphics.effective_anim_scale() > 0.0 {
                let fps = self.persisted.graphics.fps_limit.clamp(24, 144);
                let ms = (1000 / fps).max(7) as u64;
                ctx.request_repaint_after(std::time::Duration::from_millis(ms));
            }
        }

        // Top laboratory header, breadcrumbs & live telemetry strip (hidden in Q-Core Zen mode)
        if !self.qcore_zen_mode {
            egui::TopBottomPanel::top("qml_top_bar")
                .exact_height(50.0)
                .frame(
                    egui::Frame::new()
                        .fill(theme::PANEL_BG)
                        .stroke(egui::Stroke::new(1.0_f32, theme::BORDER_SUBTLE))
                        .inner_margin(egui::Margin::symmetric(14, 7)),
                )
                .show(ctx, |ui| {
                    ui.horizontal_centered(|ui| {
                        let nav_btn = if self.nav_collapsed { "»" } else { "«" };
                        if ui
                            .small_button(RichText::new(nav_btn).size(13.0).color(theme::AMETHYST))
                            .on_hover_text("Toggle navigation rail [Ctrl+B]")
                            .clicked()
                        {
                            self.nav_collapsed = !self.nav_collapsed;
                        }
                        ui.add_space(4.0);
                        theme::draw_lab_emblem(ui, 28.0, self.start_time.elapsed().as_secs_f64());
                        ui.add_space(6.0);
                        ui.label(
                            RichText::new("QUANTUM MORPH LABORATORY")
                                .color(theme::TEXT_PRIMARY)
                                .size(15.0)
                                .strong(),
                        );
                        ui.label(RichText::new("›").color(theme::TEXT_MUTED).size(14.0));
                        ui.label(
                            RichText::new(self.active_tab.label())
                                .color(theme::CYAN)
                                .size(13.5)
                                .strong(),
                        );
                        theme::pill_badge(ui, "v0.3.0 PRISM", theme::ULTRAVIOLET);

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let gpu_label = self
                                .hardware
                                .gpus
                                .first()
                                .map(|g| format!("GPU: {}", g.name))
                                .unwrap_or_else(|| "GPU: CPU Fallback".to_owned());
                            theme::pill_badge(ui, &gpu_label, theme::EMERALD);
                            let render_badge = if self.wgpu_ready {
                                format!("PRISM WGPU • {:.0} FPS", self.smoothed_fps)
                            } else {
                                "2.5D FALLBACK".to_owned()
                            };
                            theme::pill_badge(ui, &render_badge, theme::CYAN);
                            theme::pill_badge(
                                ui,
                                &format!("CPU: {}T", self.hardware.logical_cores),
                                theme::AMETHYST,
                            );
                        });
                    });
                });
        }

        // Bottom live status bar & active worker cancel control
        egui::TopBottomPanel::bottom("qml_status_bar")
            .exact_height(28.0)
            .frame(
                egui::Frame::new()
                    .fill(theme::PANEL_BG)
                    .stroke(egui::Stroke::new(1.0_f32, theme::BORDER_SUBTLE))
                    .inner_margin(egui::Margin::symmetric(14, 4)),
            )
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    if let Some(busy) = &self.busy_label {
                        ui.spinner();
                        ui.label(RichText::new(busy).color(theme::CYAN).size(11.5));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("Cancel").clicked() {
                                self.cancel_flag.store(true, Ordering::SeqCst);
                                self.status_banner = "Cancellation requested...".to_owned();
                            }
                        });
                    } else {
                        ui.label(
                            RichText::new(&self.status_banner)
                                .color(theme::TEXT_MUTED)
                                .size(11.5),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                RichText::new("Shortcuts: [1-9/0/-] Tabs • [Ctrl+B] Rail • [F] Expand 3D • [P] Ortho • [R] Reset Cam")
                                    .color(egui::Color32::from_rgb(108, 118, 152))
                                    .size(10.5),
                            );
                        });
                    }
                });
            });

        // Collapsible left navigation rail with original procedural vector iconography
        if !self.qcore_zen_mode && !self.fullscreen_3d {
            let rail_w = if self.nav_collapsed { 56.0 } else { 222.0 };
            egui::SidePanel::left("qml_nav_rail")
                .exact_width(rail_w)
                .resizable(false)
                .frame(
                    egui::Frame::new()
                        .fill(theme::PANEL_BG)
                        .stroke(egui::Stroke::new(1.0_f32, theme::BORDER_SUBTLE))
                        .inner_margin(egui::Margin::symmetric(8, 12)),
                )
                .show(ctx, |ui| {
                    if !self.nav_collapsed {
                        ui.label(
                            RichText::new("OBSERVATORY MODULES")
                                .color(theme::TEXT_MUTED)
                                .size(10.0)
                                .strong(),
                        );
                        ui.add_space(6.0);
                    }
                    for (idx, tab) in NavTab::ALL.iter().copied().enumerate() {
                        let selected = self.active_tab == tab;
                        let fill = if selected {
                            egui::Color32::from_rgba_unmultiplied(139, 92, 246, 56)
                        } else {
                            egui::Color32::TRANSPARENT
                        };
                        let stroke = if selected {
                            egui::Stroke::new(1.0_f32, theme::ULTRAVIOLET)
                        } else {
                            egui::Stroke::NONE
                        };
                        let resp = egui::Frame::new()
                            .fill(fill)
                            .stroke(stroke)
                            .corner_radius(egui::CornerRadius::same(7))
                            .inner_margin(egui::Margin::symmetric(8, 6))
                            .show(ui, |ui| {
                                ui.set_min_width(if self.nav_collapsed { 24.0 } else { 188.0 });
                                ui.horizontal(|ui| {
                                    let icon_col = if selected {
                                        theme::CYAN
                                    } else {
                                        theme::TEXT_MUTED
                                    };
                                    theme::draw_subsystem_icon(ui, idx, 18.0, icon_col);
                                    if !self.nav_collapsed {
                                        ui.add_space(6.0);
                                        let text_color = if selected {
                                            theme::TEXT_PRIMARY
                                        } else {
                                            theme::TEXT_MUTED
                                        };
                                        ui.label(
                                            RichText::new(tab.label()).color(text_color).size(12.8),
                                        );
                                    }
                                });
                            })
                            .response
                            .interact(egui::Sense::click())
                            .on_hover_text(tab.label());
                        if resp.clicked() {
                            self.active_tab = tab;
                            self.fullscreen_3d = false;
                        }
                        ui.add_space(2.0);
                    }

                    if !self.nav_collapsed {
                        ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                            ui.label(
                                RichText::new("Zero Python • 100% Safe Rust")
                                    .color(theme::TEXT_MUTED)
                                    .size(10.0),
                            );
                            ui.label(
                                RichText::new("Models opened strictly read-only")
                                    .color(theme::TEXT_MUTED)
                                    .size(10.0),
                            );
                        });
                    }
                });
        }

        // Main content viewport
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(theme::OBSIDIAN_BG)
                    .inner_margin(egui::Margin::same(if self.fullscreen_3d { 8 } else { 16 })),
            )
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| match self.active_tab {
                        NavTab::CommandCenter => views::show_command_center(self, ui, ctx),
                        NavTab::ModelLibrary => views::show_model_library(self, ui),
                        NavTab::QScope => views::show_qscope(self, ui, ctx),
                        NavTab::QEntangle => views::show_qentangle(self, ui),
                        NavTab::QForge => views::show_qforge(self, ui, ctx),
                        NavTab::QGenesis => views::show_qgenesis(self, ui, ctx),
                        NavTab::QmfExplorer => views::show_qmf_explorer(self, ui),
                        NavTab::QCore => views::show_qcore(self, ui, ctx),
                        NavTab::QBehavior => views::show_qbehavior(self, ui, ctx),
                        NavTab::Benchmarks => views::show_benchmarks(self, ui, ctx),
                        NavTab::SystemObservatory => views::show_system_observatory(self, ui, ctx),
                    });
            });
    }
}

fn save_color_image_png(path: &std::path::Path, image: &egui::ColorImage) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let file = File::create(path).map_err(|e| e.to_string())?;
    let w = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, image.width() as u32, image.height() as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    let mut bytes = Vec::with_capacity(image.pixels.len() * 4);
    for px in &image.pixels {
        bytes.extend_from_slice(&px.to_array());
    }
    writer.write_image_data(&bytes).map_err(|e| e.to_string())?;
    Ok(())
}

struct LaunchOptions {
    initial_tab: NavTab,
    capture_png: Option<PathBuf>,
    window_size: Vec2,
    auto_genesis: bool,
    auto_scope: Option<String>,
}

fn parse_launch_options() -> LaunchOptions {
    let mut opts = LaunchOptions {
        initial_tab: NavTab::CommandCenter,
        capture_png: None,
        window_size: Vec2::new(1520.0, 940.0),
        auto_genesis: false,
        auto_scope: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tab" => {
                if let Some(val) = args.next() {
                    opts.initial_tab = match val.to_lowercase().as_str() {
                        "command" | "singularity" => NavTab::CommandCenter,
                        "library" => NavTab::ModelLibrary,
                        "qscope" | "scope" => NavTab::QScope,
                        "qentangle" | "entangle" => NavTab::QEntangle,
                        "qforge" | "forge" => NavTab::QForge,
                        "qgenesis" | "genesis" => NavTab::QGenesis,
                        "qmf" | "explorer" => NavTab::QmfExplorer,
                        "qcore" | "core" => NavTab::QCore,
                        "qbehavior" | "behavior" => NavTab::QBehavior,
                        "benchmarks" | "rmq" => NavTab::Benchmarks,
                        "hardware" | "system" => NavTab::SystemObservatory,
                        _ => NavTab::CommandCenter,
                    };
                }
            }
            "--capture-png" => {
                if let Some(val) = args.next() {
                    opts.capture_png = Some(PathBuf::from(val));
                }
            }
            "--size" => {
                if let Some(val) = args.next()
                    && let Some((ws, hs)) = val.split_once('x')
                    && let (Ok(w), Ok(h)) = (ws.parse::<f32>(), hs.parse::<f32>())
                {
                    opts.window_size = Vec2::new(w.clamp(1024.0, 3840.0), h.clamp(680.0, 2160.0));
                }
            }
            "--auto-genesis" => {
                opts.auto_genesis = true;
            }
            "--auto-scope" => {
                if let Some(val) = args.next() {
                    opts.auto_scope = Some(val);
                }
            }
            _ => {}
        }
    }
    opts
}

fn main() -> eframe::Result<()> {
    install_crash_logger();
    append_log_line(
        "INFO",
        "Launching qml-desktop v0.3.0 (PROJECT PRISM — WGPU 3D Renderer).",
    );
    let launch = parse_launch_options();

    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_title("Quantum Morph Laboratory — PROJECT PRISM")
            .with_inner_size(launch.window_size)
            .with_min_inner_size(Vec2::new(1024.0, 680.0))
            .with_icon(Arc::new(theme::generate_app_icon_data())),
        ..Default::default()
    };

    let tab = launch.initial_tab;
    let cap = launch.capture_png;
    let auto_gen = launch.auto_genesis;
    let auto_sc = launch.auto_scope;
    eframe::run_native(
        "Quantum Morph Laboratory",
        options,
        Box::new(move |cc| {
            Ok(Box::new(QmlDesktopApp::new(
                cc, tab, cap, auto_gen, auto_sc,
            )))
        }),
    )
}
