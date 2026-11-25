#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod state;
mod theme;
mod views;

use eframe::egui::{self, RichText, Vec2};
use qml_forge::{ForgeReport, TransformStrategy, forge_safetensors_to_qmf};
use qml_format::QmfArchiveV1;
use qml_genesis::{GenesisArchive, GenesisConfig, run_genesis};
use qml_hardware::{HardwareReport, WgpuMatvecEngine, probe_hardware};
use qml_model::{PayloadBudget, SafetensorsFile};
use qml_quant::{FactorPrecision, RmqAblationReport, compare_rmq_vs_baselines};
use qml_runtime::{QCoreModel, TransformerConfig, evaluate_inference_fidelity};
use qml_scope::{ModelScopeReport, ScopeConfig, analyze_safetensors};
use qml_tokenizer::BpeTokenizer;
use state::{
    GpuSelfTestResult, ModelFormat, PersistedState, QmfInspection, WorkerEvent, append_log_line,
    default_discovery_directories, install_crash_logger, load_persisted_state, probe_model_file,
    save_persisted_state, scan_directory_for_models,
};
use std::fs::{self, File};
use std::io::BufWriter;
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
    Benchmarks,
    SystemObservatory,
}

impl NavTab {
    pub const ALL: [Self; 10] = [
        Self::CommandCenter,
        Self::ModelLibrary,
        Self::QScope,
        Self::QEntangle,
        Self::QForge,
        Self::QGenesis,
        Self::QmfExplorer,
        Self::QCore,
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
    RmqSharedBasis,
    RmqSharedCore,
    LowRankSvd,
}

impl ForgeMethodChoice {
    pub const ALL: [Self; 9] = [
        Self::Bf16,
        Self::Fp16,
        Self::Int8Sym,
        Self::Int4Sym,
        Self::Int4Asym,
        Self::MapAdaptive,
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
    pub forge_report: Option<(ForgeReport, PathBuf)>,

    // Q-Genesis state
    pub genesis_use_synthetic: bool,
    pub genesis_source_path: String,
    pub genesis_population: usize,
    pub genesis_generations: usize,
    pub genesis_archive: Option<GenesisArchive>,

    // QMF Explorer state
    pub qmf_inspect_path: String,
    pub qmf_inspection: Option<QmfInspection>,

    // Q-Core Inference Chamber state
    pub infer_qmf_path: String,
    pub infer_config_path: String,
    pub infer_tokenizer_path: String,
    pub infer_compare_qmf_path: String,
    pub infer_prompt: String,
    pub infer_use_chatml: bool,
    pub infer_max_tokens: usize,
    pub infer_temperature: f64,
    pub infer_output_text: String,
    pub infer_stats: Option<(usize, f64, f64)>,
    pub infer_fidelity: Option<qml_runtime::InferenceComparison>,

    // Benchmarks / RMQ Ablation state
    pub ablation_layers: usize,
    pub ablation_dim: usize,
    pub ablation_rank: usize,
    pub ablation_result: Option<(RmqAblationReport, RmqAblationReport)>,

    // System Observatory state
    pub gpu_selftest: Option<GpuSelfTestResult>,
}

impl QmlDesktopApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        theme::apply_laboratory_theme(&cc.egui_ctx);
        let hardware = probe_hardware();
        let mut persisted = load_persisted_state();

        // Auto-discover local cache models on first launch (read-only) if library is empty.
        if persisted.library.is_empty() {
            for dir in default_discovery_directories() {
                let found = scan_directory_for_models(&dir, 3);
                for entry in found {
                    if !persisted.library.iter().any(|e| e.path == entry.path) {
                        persisted.library.push(entry);
                    }
                }
            }
            if !persisted.library.is_empty() {
                save_persisted_state(&persisted);
            }
        }

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
            active_tab: NavTab::CommandCenter,
            persisted,
            hardware,
            status_banner:
                "Laboratory online — all engines ready (imported models are strictly read-only)."
                    .to_owned(),
            activity_log: vec![
                "Quantum Morph Laboratory v0.1.0-alpha.1 (PROJECT SINGULARITY) initialized."
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
            forge_report: None,

            genesis_use_synthetic: true,
            genesis_source_path: String::new(),
            genesis_population: 12,
            genesis_generations: 5,
            genesis_archive: None,

            qmf_inspect_path: infer_qmf_path.clone(),
            qmf_inspection: None,

            infer_qmf_path,
            infer_config_path,
            infer_tokenizer_path,
            infer_compare_qmf_path: String::new(),
            infer_prompt: "Explain singular value decomposition in two sentences.".to_owned(),
            infer_use_chatml: true,
            infer_max_tokens: 48,
            infer_temperature: 0.0,
            infer_output_text: String::new(),
            infer_stats: None,
            infer_fidelity: None,

            ablation_layers: 8,
            ablation_dim: 64,
            ablation_rank: 6,
            ablation_result: None,

            gpu_selftest: None,
        };

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
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.busy_label = Some(format!(
            "Q-Forge transforming {} -> {}...",
            model_id,
            self.forge_method.label()
        ));
        self.cancel_flag.store(false, Ordering::SeqCst);

        std::thread::spawn(move || {
            let res = File::open(&src)
                .map_err(|e| e.to_string())
                .and_then(|file| {
                    forge_safetensors_to_qmf(
                        file,
                        &model_id,
                        "local-readonly-source",
                        "transformer",
                        &strategy,
                        &PayloadBudget::default(),
                        42,
                    )
                    .map_err(|e| e.to_string())
                })
                .and_then(|(archive, report)| {
                    if let Some(parent) = dst.parent().filter(|p| !p.as_os_str().is_empty()) {
                        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                    }
                    let out_file = File::create(&dst).map_err(|e| e.to_string())?;
                    let mut writer = BufWriter::new(out_file);
                    archive.write_to(&mut writer).map_err(|e| e.to_string())?;
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
                        let names: Vec<String> = st
                            .metadata()
                            .tensors
                            .iter()
                            .filter(|t| {
                                t.shape.len() == 2
                                    && t.shape[0] > 0
                                    && t.shape[1] > 0
                                    && t.shape[0] <= 2048
                                    && t.shape[1] <= 2048
                            })
                            .take(6)
                            .map(|t| t.name.clone())
                            .collect();
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
                let archive = QmfArchiveV1::read_from(qmf_file).map_err(|e| e.to_string())?;
                let model =
                    QCoreModel::from_qmf(&archive, cfg.clone()).map_err(|e| e.to_string())?;

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
                            let piece = tokenizer.decode_token(tok_id as u32, true);
                            let _ = tx.send(WorkerEvent::InferToken(piece));
                            ctx.request_repaint();
                            true
                        },
                    )
                    .map_err(|e| e.to_string())?;

                let fidelity = if !compare_path.is_empty() && PathBuf::from(&compare_path).is_file()
                {
                    let _ = tx.send(WorkerEvent::Status(
                        "Computing reference KL divergence & perplexity...".to_owned(),
                    ));
                    ctx.request_repaint();
                    let ref_file = File::open(&compare_path).map_err(|e| e.to_string())?;
                    let ref_archive =
                        QmfArchiveV1::read_from(ref_file).map_err(|e| e.to_string())?;
                    let ref_model =
                        QCoreModel::from_qmf(&ref_archive, cfg).map_err(|e| e.to_string())?;
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
                    let mat = qml_core::DenseMatrix::from_flat(
                        rows,
                        cols,
                        (0..rows * cols).map(|_| rng.next_normal() * 0.2).collect(),
                    )
                    .unwrap();
                    let x: Vec<f64> = (0..cols).map(|_| rng.next_normal()).collect();

                    let t_cpu = Instant::now();
                    let cpu = mat.matvec(&x).unwrap();
                    let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;

                    let t_gpu = Instant::now();
                    match engine.matvec(&mat, &x) {
                        Ok(gpu) => {
                            let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;
                            let max_abs_error = cpu
                                .iter()
                                .zip(&gpu)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0_f64, f64::max);
                            let _ = tx.send(WorkerEvent::GpuSelfTestComplete(GpuSelfTestResult {
                                adapter_name: engine.adapter_name().to_owned(),
                                backend_name: engine.backend_name().to_owned(),
                                matrix_shape: (rows, cols),
                                max_abs_error,
                                gpu_elapsed_ms: gpu_ms,
                                cpu_elapsed_ms: cpu_ms,
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
                    let msg = format!(
                        "WGPU self-test PASSED on {} ({}): max_abs_err={:.2e} ({:.2} ms).",
                        res.adapter_name, res.backend_name, res.max_abs_error, res.gpu_elapsed_ms
                    );
                    self.gpu_selftest = Some(res);
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
        if self.is_busy() {
            ctx.request_repaint_after(std::time::Duration::from_millis(80));
        }

        // Top laboratory header & telemetry strip
        egui::TopBottomPanel::top("qml_top_bar")
            .exact_height(52.0)
            .frame(
                egui::Frame::new()
                    .fill(theme::PANEL_BG)
                    .stroke(egui::Stroke::new(1.0_f32, theme::BORDER_SUBTLE))
                    .inner_margin(egui::Margin::symmetric(16, 8)),
            )
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    theme::draw_lab_emblem(ui, 30.0, self.start_time.elapsed().as_secs_f64());
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new("QUANTUM MORPH LABORATORY")
                            .color(theme::TEXT_PRIMARY)
                            .size(16.0)
                            .strong(),
                    );
                    theme::pill_badge(ui, "v0.1.0-alpha.1", theme::ULTRAVIOLET);
                    theme::pill_badge(ui, "RESEARCH PROTOTYPE", theme::CYAN);

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let gpu_label = self
                            .hardware
                            .gpus
                            .first()
                            .map(|g| format!("GPU: {}", g.name))
                            .unwrap_or_else(|| "GPU: CPU Fallback".to_owned());
                        theme::pill_badge(ui, &gpu_label, theme::EMERALD);
                        theme::pill_badge(
                            ui,
                            &format!("CPU: {}T", self.hardware.logical_cores),
                            theme::AMETHYST,
                        );
                    });
                });
            });

        // Bottom live status bar & active worker cancel control
        egui::TopBottomPanel::bottom("qml_status_bar")
            .exact_height(30.0)
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
                        ui.label(RichText::new(busy).color(theme::CYAN).size(12.0));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Cancel").clicked() {
                                self.cancel_flag.store(true, Ordering::SeqCst);
                                self.status_banner = "Cancellation requested...".to_owned();
                            }
                        });
                    } else {
                        ui.label(
                            RichText::new(&self.status_banner)
                                .color(theme::TEXT_MUTED)
                                .size(12.0),
                        );
                    }
                });
            });

        // Left navigation rail
        egui::SidePanel::left("qml_nav_rail")
            .exact_width(218.0)
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(theme::PANEL_BG)
                    .stroke(egui::Stroke::new(1.0_f32, theme::BORDER_SUBTLE))
                    .inner_margin(egui::Margin::symmetric(10, 14)),
            )
            .show(ctx, |ui| {
                ui.label(
                    RichText::new("LABORATORY SUBSYSTEMS")
                        .color(theme::TEXT_MUTED)
                        .size(10.5)
                        .strong(),
                );
                ui.add_space(6.0);
                for tab in NavTab::ALL {
                    let selected = self.active_tab == tab;
                    let fill = if selected {
                        egui::Color32::from_rgba_unmultiplied(139, 92, 246, 52)
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
                        .corner_radius(egui::CornerRadius::same(6))
                        .inner_margin(egui::Margin::symmetric(8, 7))
                        .show(ui, |ui| {
                            ui.set_min_width(184.0);
                            ui.horizontal(|ui| {
                                let code_color = if selected {
                                    theme::CYAN
                                } else {
                                    theme::TEXT_MUTED
                                };
                                ui.label(
                                    RichText::new(tab.short_badge())
                                        .color(code_color)
                                        .monospace()
                                        .size(11.0)
                                        .strong(),
                                );
                                ui.add_space(4.0);
                                let text_color = if selected {
                                    theme::TEXT_PRIMARY
                                } else {
                                    theme::TEXT_MUTED
                                };
                                ui.label(RichText::new(tab.label()).color(text_color).size(13.0));
                            });
                        })
                        .response
                        .interact(egui::Sense::click());
                    if resp.clicked() {
                        self.active_tab = tab;
                    }
                    ui.add_space(2.0);
                }

                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    ui.label(
                        RichText::new("Zero Python • 100% Safe Rust")
                            .color(theme::TEXT_MUTED)
                            .size(10.5),
                    );
                    ui.label(
                        RichText::new("Models opened strictly read-only")
                            .color(theme::TEXT_MUTED)
                            .size(10.5),
                    );
                });
            });

        // Main content viewport
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(theme::OBSIDIAN_BG)
                    .inner_margin(egui::Margin::same(18)),
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
                        NavTab::Benchmarks => views::show_benchmarks(self, ui, ctx),
                        NavTab::SystemObservatory => views::show_system_observatory(self, ui, ctx),
                    });
            });
    }
}

fn main() -> eframe::Result<()> {
    install_crash_logger();
    append_log_line("INFO", "Launching qml-desktop native application.");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Quantum Morph Laboratory — Neural Representation Research Workbench")
            .with_inner_size(Vec2::new(1440.0, 900.0))
            .with_min_inner_size(Vec2::new(1024.0, 680.0)),
        ..Default::default()
    };

    eframe::run_native(
        "Quantum Morph Laboratory",
        options,
        Box::new(|cc| Ok(Box::new(QmlDesktopApp::new(cc)))),
    )
}
