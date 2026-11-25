//! Persistent laboratory state, model library discovery, crash logging, and background job messages.

use qml_forge::ForgeReport;
use qml_format::{QmfBudget, QmfManifest, QmfRankOne};
use qml_genesis::GenesisArchive;
use qml_hardware::ResourcePreset;
use qml_model::{inspect_gguf, inspect_safetensors};
use qml_quant::RmqAblationReport;
use qml_runtime::InferenceComparison;
use qml_scope::ModelScopeReport;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Supported local model artifact formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    Safetensors,
    Gguf,
    Qmf,
}

impl ModelFormat {
    pub fn label(self) -> &'static str {
        match self {
            Self::Safetensors => "Safetensors",
            Self::Gguf => "GGUF",
            Self::Qmf => "QMF",
        }
    }
}

/// One imported local model artifact tracked in the laboratory library.
/// Imported model files are always opened read-only and never modified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub path: PathBuf,
    pub format: ModelFormat,
    pub file_bytes: u64,
    pub tensor_count: usize,
    pub detail: String,
    pub config_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
}

/// Persistent user preferences and library entries stored under `%LOCALAPPDATA%`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedState {
    pub library: Vec<ModelEntry>,
    pub resource_preset: ResourcePreset,
    pub scan_dirs: Vec<PathBuf>,
}

impl Default for PersistedState {
    fn default() -> Self {
        Self {
            library: Vec::new(),
            resource_preset: ResourcePreset::Balanced,
            scan_dirs: Vec::new(),
        }
    }
}

/// Resolve the local application data directory for state and crash logs.
pub fn app_data_dir() -> PathBuf {
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        PathBuf::from(local).join("QuantumMorphLaboratory")
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".quantum_morph_laboratory")
    } else {
        PathBuf::from(".qml-desktop-data")
    }
}

/// Append a timestamped diagnostic line to the runtime log file.
pub fn append_log_line(level: &str, message: &str) {
    let dir = app_data_dir().join("logs");
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = dir.join("qml-desktop.log");
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "[{level}] {message}");
    }
}

/// Install a crash hook that records panics cleanly to `qml-desktop.log`.
pub fn install_crash_logger() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        append_log_line("PANIC", &format!("{info}"));
        prev(info);
    }));
}

/// Load persisted state, returning defaults if none exists or parsing fails.
pub fn load_persisted_state() -> PersistedState {
    let path = app_data_dir().join("state.json");
    match fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
        Err(_) => PersistedState::default(),
    }
}

/// Save persisted state atomically to `%LOCALAPPDATA%\QuantumMorphLaboratory\state.json`.
pub fn save_persisted_state(state: &PersistedState) {
    let dir = app_data_dir();
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = dir.join("state.json");
    if let Ok(json) = serde_json::to_string_pretty(state) {
        let _ = fs::write(path, json);
    }
}

/// Inspect a single local file read-only and construct a [`ModelEntry`] if supported.
pub fn probe_model_file(path: &Path) -> Result<ModelEntry, String> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    let meta = fs::metadata(path).map_err(|e| format!("cannot stat {}: {e}", path.display()))?;
    let file_bytes = meta.len();
    let parent = path.parent();
    let config_path = parent
        .map(|p| p.join("config.json"))
        .filter(|p| p.is_file());
    let tokenizer_path = parent
        .map(|p| p.join("tokenizer.json"))
        .filter(|p| p.is_file());

    let display_name = parent
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .zip(path.file_name().and_then(|n| n.to_str()))
        .map(|(dir, file)| format!("{dir}/{file}"))
        .unwrap_or_else(|| path.display().to_string());

    match ext.as_str() {
        "safetensors" => {
            let file = File::open(path).map_err(|e| e.to_string())?;
            let inspected = inspect_safetensors(file).map_err(|e| e.to_string())?;
            let first_dtype = inspected
                .tensors
                .first()
                .map(|t| format!("{:?}", t.dtype))
                .unwrap_or_else(|| "unknown".to_owned());
            Ok(ModelEntry {
                name: display_name,
                path: path.to_path_buf(),
                format: ModelFormat::Safetensors,
                file_bytes,
                tensor_count: inspected.tensors.len(),
                detail: format!(
                    "Safetensors ({first_dtype}, header {} B)",
                    inspected.header_bytes
                ),
                config_path,
                tokenizer_path,
            })
        }
        "gguf" => {
            let file = File::open(path).map_err(|e| e.to_string())?;
            let inspected = inspect_gguf(file).map_err(|e| e.to_string())?;
            Ok(ModelEntry {
                name: display_name,
                path: path.to_path_buf(),
                format: ModelFormat::Gguf,
                file_bytes,
                tensor_count: inspected.tensor_count as usize,
                detail: format!(
                    "GGUF v{} ({} metadata KV pairs)",
                    inspected.version, inspected.kv_count
                ),
                config_path,
                tokenizer_path,
            })
        }
        "qmf" => {
            let file = File::open(path).map_err(|e| e.to_string())?;
            let summary = qml_format::inspect_qmf_v1_header(file, &QmfBudget::default())
                .map(|(m, _, _)| {
                    (
                        m.tensors.len(),
                        format!(
                            "QMF v1 ({}, {}, {} shared latents)",
                            m.model_id,
                            m.transformation,
                            m.shared_latents.len()
                        ),
                    )
                })
                .unwrap_or_else(|_| (1, "QMF v0 (single rank-1 matrix)".to_owned()));
            Ok(ModelEntry {
                name: display_name,
                path: path.to_path_buf(),
                format: ModelFormat::Qmf,
                file_bytes,
                tensor_count: summary.0,
                detail: summary.1,
                config_path,
                tokenizer_path,
            })
        }
        _ => Err(format!(
            "unsupported file extension '.{ext}' (expected .safetensors, .gguf, or .qmf)"
        )),
    }
}

/// Scan a directory (up to depth 3) read-only for `.safetensors`, `.gguf`, and `.qmf` files.
pub fn scan_directory_for_models(root: &Path, max_depth: usize) -> Vec<ModelEntry> {
    let mut found = Vec::new();
    scan_dir_rec(root, max_depth, &mut found);
    found
}

fn scan_dir_rec(dir: &Path, depth: usize, out: &mut Vec<ModelEntry>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if depth > 0 {
                scan_dir_rec(&path, depth - 1, out);
            }
        } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            let lower = ext.to_ascii_lowercase();
            if matches!(lower.as_str(), "safetensors" | "gguf" | "qmf")
                && let Ok(model) = probe_model_file(&path)
            {
                out.push(model);
            }
        }
    }
}

/// Candidate standard local model directories (read-only opt-in scan).
pub fn default_discovery_directories() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(profile) = std::env::var("USERPROFILE") {
        let p = PathBuf::from(&profile);
        dirs.push(
            p.join(".gemini")
                .join("antigravity")
                .join("scratch")
                .join("qml-models-cache"),
        );
        dirs.push(p.join(".cache").join("lm-studio").join("models"));
        dirs.push(p.join(".cache").join("huggingface").join("hub"));
        dirs.push(p.join(".ollama").join("models"));
    }
    dirs.into_iter().filter(|d| d.is_dir()).collect()
}

/// Result of inspecting a QMF v0 or v1 file for the QMF Explorer view.
#[derive(Debug, Clone)]
pub enum QmfInspection {
    V1 {
        manifest: QmfManifest,
        manifest_bytes: u64,
        payload_bytes: u64,
    },
    V0 {
        rows: usize,
        cols: usize,
        path: PathBuf,
    },
}

impl QmfInspection {
    pub fn inspect_file(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        if let Ok((manifest, manifest_bytes, payload_bytes)) =
            qml_format::inspect_qmf_v1_header(file, &QmfBudget::default())
        {
            return Ok(Self::V1 {
                manifest,
                manifest_bytes,
                payload_bytes,
            });
        }
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        let v0 = QmfRankOne::from_bytes(&bytes).map_err(|e| e.to_string())?;
        Ok(Self::V0 {
            rows: v0.factors().rows(),
            cols: v0.factors().cols(),
            path: path.to_path_buf(),
        })
    }
}

/// Result of running the live WGPU compute self-test in System Observatory.
#[derive(Debug, Clone)]
pub struct GpuSelfTestResult {
    pub adapter_name: String,
    pub backend_name: String,
    pub matrix_shape: (usize, usize),
    pub max_abs_error: f64,
    pub gpu_elapsed_ms: f64,
    pub cpu_elapsed_ms: f64,
}

/// Events streamed back from background worker threads to the GUI thread.
pub enum WorkerEvent {
    Status(String),
    ScopeComplete(Box<ModelScopeReport>),
    ForgeComplete {
        report: Box<ForgeReport>,
        output_path: PathBuf,
    },
    GenesisComplete(Box<GenesisArchive>),
    AblationComplete {
        correlated: RmqAblationReport,
        gaussian_control: RmqAblationReport,
    },
    InferToken(String),
    InferComplete {
        tokens_generated: usize,
        tokens_per_second: f64,
        elapsed_ms: f64,
        fidelity: Option<InferenceComparison>,
    },
    GpuSelfTestComplete(GpuSelfTestResult),
    Error(String),
}
