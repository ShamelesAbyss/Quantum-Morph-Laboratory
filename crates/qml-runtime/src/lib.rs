//! Q-Core — Native CPU Reference Transformer Inference Runtime.
//!
//! Executes decoder-only Llama-style transformer architectures **directly over
//! QMF v1 compressed linear operators** (`DenseF32`, `FP16`, `BF16`, `INT8`,
//! `INT4`, `LowRankSvd`, and `RmqSharedBasis + SparseResidual`) without ever
//! reconstructing full dense weight matrices in memory.
//!
//! Includes RMSNorm, Rotary Position Embeddings (RoPE), Grouped-Query Attention
//! with incremental KV-cache, SwiGLU MLP, autoregressive token generation, and
//! rigorous reference-vs-transformed logit KL divergence & perplexity evaluation.

mod admission;
mod bias;
#[cfg(test)]
mod sentinel_tests;
pub mod streaming;
pub use streaming::{StreamingCalibAndRefResult, stream_calibrate_and_reference};

use qml_core::{DeterministicRng, MatrixError};
use qml_format::{QmfArchiveV1, QmfError, QmfFileReader, QmfLatentCache, QmfOperator};
use qml_hardware::{
    GpuFullConfig, GpuFullLayerSpec, GpuFullModel, GpuFullOperator, GpuKvCache, GpuKvMode,
    PersistentGpuMatrix, WgpuMatvecEngine,
};
use qml_model::{DType, encode_safetensors, f16_to_f32, f32_to_f16};
use qml_quant::{
    DEFAULT_HESSIAN_BLOCK_SIZE, EchoBaseOperator, LayerCalibrationAccumulator, LinearOperator,
    ModelCalibrationProfile, QuantizedFactor, SparseResidual,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

/// Architecture hyperparameters for a Llama-style decoder-only transformer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_seq_len: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl TransformerConfig {
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if self.vocab_size == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_layers == 0
            || self.num_heads == 0
            || self.num_kv_heads == 0
            || self.max_seq_len == 0
        {
            return Err(RuntimeError::InvalidConfig(
                "all transformer dimensions must be non-zero".to_owned(),
            ));
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(RuntimeError::InvalidConfig(format!(
                "hidden_size {} not divisible by num_heads {}",
                self.hidden_size, self.num_heads
            )));
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(RuntimeError::InvalidConfig(format!(
                "num_heads {} not divisible by num_kv_heads {}",
                self.num_heads, self.num_kv_heads
            )));
        }
        let head_dim = self.hidden_size / self.num_heads;
        if !head_dim.is_multiple_of(2) {
            return Err(RuntimeError::InvalidConfig(format!(
                "head_dim {head_dim} must be even for RoPE"
            )));
        }
        for (name, value) in [
            ("rms_norm_eps", self.rms_norm_eps),
            ("rope_theta", self.rope_theta),
        ] {
            if !value.is_finite()
                || value <= 0.0
                || !(value as f32).is_finite()
                || value as f32 <= 0.0
            {
                return Err(RuntimeError::InvalidConfig(format!(
                    "{name} must be positive and finite in both CPU and GPU precision"
                )));
            }
        }
        // Shader arguments use u32. Check before any narrowing or dimension arithmetic.
        for dimension in [
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_layers,
            self.num_heads,
            self.num_kv_heads,
            self.max_seq_len,
        ] {
            u32::try_from(dimension).map_err(|_| {
                RuntimeError::InvalidConfig(
                    "transformer dimension exceeds the supported u32 range".into(),
                )
            })?;
        }
        let span = |name: &str, factors: &[usize]| -> Result<(), RuntimeError> {
            factors
                .iter()
                .try_fold(1_usize, |total, &factor| total.checked_mul(factor))
                .filter(|&bytes| bytes <= isize::MAX as usize)
                .ok_or_else(|| {
                    RuntimeError::InvalidConfig(format!(
                        "{name} dimension/byte arithmetic exceeds addressable storage"
                    ))
                })?;
            Ok(())
        };
        let kv_dim = self
            .num_kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| RuntimeError::InvalidConfig("KV dimension overflow".into()))?;
        span(
            "embedding/output projection",
            &[self.vocab_size, self.hidden_size, 8],
        )?;
        span(
            "attention projection",
            &[self.hidden_size, self.hidden_size, 8],
        )?;
        span(
            "MLP projection",
            &[self.hidden_size, self.intermediate_size, 8],
        )?;
        span("KV projection", &[kv_dim, self.hidden_size, 8])?;
        span(
            "sequence activations",
            &[self.max_seq_len, self.hidden_size, 8],
        )?;
        span("layer traces", &[self.num_layers, self.hidden_size, 8])?;
        span(
            "KV cache",
            &[self.num_layers, self.max_seq_len, kv_dim, 2, 8],
        )?;
        span(
            "layer directory",
            &[self.num_layers, std::mem::size_of::<TransformerBlock>()],
        )?;
        self.num_layers
            .checked_mul(7)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| RuntimeError::InvalidConfig("operator count overflow".into()))?;
        let largest_angle = (self.max_seq_len - 1).max(1) as f64
            / f64::from(self.rope_theta as f32)
                .min(1.0)
                .powf((head_dim - 2) as f64 / head_dim as f64);
        if !(largest_angle as f32).is_finite() {
            return Err(RuntimeError::InvalidConfig(
                "RoPE angle exceeds finite GPU precision".into(),
            ));
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Admit the supported unscaled Llama/Qwen2 subset of Hugging Face configurations.
    pub fn from_hf_config_bytes(bytes: &[u8]) -> Result<Self, RuntimeError> {
        admission::parse(bytes)
    }

    /// Bind cached calibration/reference data to effective settings and runtime semantics.
    /// Advance this domain whenever execution/admission changes invalidate earlier evidence.
    pub fn execution_identity(&self) -> Result<String, RuntimeError> {
        self.validate()?;
        let settings = serde_json::to_vec(self)
            .map_err(|e| RuntimeError::InvalidConfig(format!("execution identity: {e}")))?;
        let mut sha = qml_core::Sha256::new();
        sha.update(b"Q-Core/SENTINEL-execution-v1\0");
        sha.update(&settings);
        Ok(sha.finalize_hex())
    }
}

#[derive(Debug)]
pub enum RuntimeError {
    InvalidConfig(String),
    Qmf(QmfError),
    Matrix(MatrixError),
    TokenOutOfRange { token: usize, vocab_size: usize },
    ContextLengthExceeded { position: usize, max_seq_len: usize },
    EmptyPrompt,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(m) => write!(f, "invalid transformer config: {m}"),
            Self::Qmf(e) => write!(f, "QMF load error: {e}"),
            Self::Matrix(e) => write!(f, "operator execution error: {e}"),
            Self::TokenOutOfRange { token, vocab_size } => {
                write!(f, "token id {token} out of vocab range 0..{vocab_size}")
            }
            Self::ContextLengthExceeded {
                position,
                max_seq_len,
            } => write!(
                f,
                "sequence position {position} exceeds max_seq_len {max_seq_len}"
            ),
            Self::EmptyPrompt => write!(f, "prompt token sequence must not be empty"),
        }
    }
}

impl std::error::Error for RuntimeError {}
impl From<QmfError> for RuntimeError {
    fn from(v: QmfError) -> Self {
        Self::Qmf(v)
    }
}
impl From<MatrixError> for RuntimeError {
    fn from(v: MatrixError) -> Self {
        Self::Matrix(v)
    }
}

/// Storage precision mode for CPU incremental KV-cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum KvCacheMode {
    /// Full-precision f64 KV-cache reference.
    #[default]
    Fp64,
    /// Per-token symmetric INT8 KV-cache with f16 scales (~7.5x smaller than f64).
    Int8,
}

#[derive(Debug, Clone)]
struct Int8KvEntry {
    scale_f16: u16,
    codes: Vec<i8>,
}

impl Int8KvEntry {
    fn encode(vec: &[f64]) -> Self {
        let absmax = vec.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let scale = if absmax > 0.0 { absmax / 127.0 } else { 0.0 };
        let scale_f16 = f32_to_f16(scale as f32);
        let eff = f64::from(f16_to_f32(scale_f16));
        let codes = if eff > 0.0 {
            vec.iter()
                .map(|&v| (v / eff).round().clamp(-127.0, 127.0) as i8)
                .collect()
        } else {
            vec![0i8; vec.len()]
        };
        Self { scale_f16, codes }
    }

    #[inline]
    fn scale(&self) -> f64 {
        f64::from(f16_to_f32(self.scale_f16))
    }
}

/// Incremental key-value cache for autoregressive decoding (supports FP64, INT8, and GPU-resident KV).
pub struct KvCache {
    mode: KvCacheMode,
    keys: Vec<Vec<Vec<f64>>>,   // [layer][pos][kv_dim] (Fp64 mode)
    values: Vec<Vec<Vec<f64>>>, // [layer][pos][kv_dim] (Fp64 mode)
    keys_i8: Vec<Vec<Int8KvEntry>>,
    values_i8: Vec<Vec<Int8KvEntry>>,
    gpu_kv: Option<GpuKvCache>,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        Self::with_mode(num_layers, KvCacheMode::Fp64)
    }

    pub fn new_int8(num_layers: usize) -> Self {
        Self::with_mode(num_layers, KvCacheMode::Int8)
    }

    pub fn with_mode(num_layers: usize, mode: KvCacheMode) -> Self {
        Self {
            mode,
            keys: (0..num_layers).map(|_| Vec::new()).collect(),
            values: (0..num_layers).map(|_| Vec::new()).collect(),
            keys_i8: (0..num_layers).map(|_| Vec::new()).collect(),
            values_i8: (0..num_layers).map(|_| Vec::new()).collect(),
            gpu_kv: None,
        }
    }

    pub fn mode(&self) -> KvCacheMode {
        self.mode
    }

    pub fn seq_len(&self) -> usize {
        if let Some(g) = &self.gpu_kv {
            return g.seq_len();
        }
        match self.mode {
            KvCacheMode::Fp64 => self.keys.first().map_or(0, Vec::len),
            KvCacheMode::Int8 => self.keys_i8.first().map_or(0, Vec::len),
        }
    }
}

/// Execution backend selection for Q-Core inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum QCoreBackend {
    /// Multi-threaded Rayon CPU execution over compressed QMF operators.
    #[default]
    Cpu,
    /// Persistent GPU-resident compressed linear operators via WGPU compute shaders,
    /// with RMSNorm/RoPE/Attention on CPU.
    GpuHybrid,
    /// Full GPU-resident single-submission transformer execution (RMSNorm, RoPE,
    /// Qwen2 biases, VRAM KV-cache, GQA attention, SwiGLU, residuals, GPU Hadamard,
    /// and direct compressed matvecs in ONE submission per token), falling back to
    /// `GpuHybrid` if any operator requires CPU fallback.
    GpuFull,
}

/// Runtime backend telemetry for an initialized [`QCoreModel`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QCoreBackendStats {
    pub backend: String,
    pub gpu_adapter: Option<String>,
    pub gpu_resident_operators: usize,
    pub cpu_fallback_operators: usize,
    pub vram_weight_bytes: u64,
    pub shared_latents_deduped_gpu: usize,
}

enum GpuOperator {
    Single(Arc<PersistentGpuMatrix>),
    LowRankSvd {
        left: Arc<PersistentGpuMatrix>,
        right: Arc<PersistentGpuMatrix>,
    },
    RmqSharedBasis {
        u_shared: Arc<PersistentGpuMatrix>,
        core: Arc<PersistentGpuMatrix>,
        vt_shared: Arc<PersistentGpuMatrix>,
        sparse_residual: Option<SparseResidual>,
    },
    RmqSharedCore {
        left: Arc<PersistentGpuMatrix>,
        shared_core: Arc<PersistentGpuMatrix>,
        right: Arc<PersistentGpuMatrix>,
        sparse_residual: Option<SparseResidual>,
    },
    EchoCorrected {
        base: Arc<PersistentGpuMatrix>,
        low_rank: Option<(Arc<PersistentGpuMatrix>, Arc<PersistentGpuMatrix>)>,
        sparse_residual: Option<SparseResidual>,
    },
}

impl GpuOperator {
    fn to_full_op(&self) -> Option<GpuFullOperator> {
        match self {
            Self::Single(m) => Some(GpuFullOperator::Single(Arc::clone(m))),
            Self::EchoCorrected {
                base,
                low_rank,
                sparse_residual,
            } if sparse_residual.is_none() => match low_rank {
                None => Some(GpuFullOperator::Single(Arc::clone(base))),
                Some((left, right)) => Some(GpuFullOperator::EchoLowRank {
                    base: Arc::clone(base),
                    left: Arc::clone(left),
                    right: Arc::clone(right),
                }),
            },
            _ => None,
        }
    }
    fn matvec(&self, x: &[f64]) -> Result<Vec<f64>, MatrixError> {
        let map_err = |_| MatrixError::Empty;
        match self {
            Self::Single(m) => m.matvec(x).map_err(map_err),
            Self::LowRankSvd { left, right } => {
                let t = right.matvec(x).map_err(map_err)?;
                left.matvec(&t).map_err(map_err)
            }
            Self::RmqSharedBasis {
                u_shared,
                core,
                vt_shared,
                sparse_residual,
            } => {
                let t1 = vt_shared.matvec(x).map_err(map_err)?;
                let t2 = core.matvec(&t1).map_err(map_err)?;
                let mut y = u_shared.matvec(&t2).map_err(map_err)?;
                if let Some(sp) = sparse_residual {
                    sp.add_matvec_into(x, &mut y)?;
                }
                Ok(y)
            }
            Self::RmqSharedCore {
                left,
                shared_core,
                right,
                sparse_residual,
            } => {
                let t1 = right.matvec(x).map_err(map_err)?;
                let t2 = shared_core.matvec(&t1).map_err(map_err)?;
                let mut y = left.matvec(&t2).map_err(map_err)?;
                if let Some(sp) = sparse_residual {
                    sp.add_matvec_into(x, &mut y)?;
                }
                Ok(y)
            }
            Self::EchoCorrected {
                base,
                low_rank,
                sparse_residual,
            } => {
                let mut y = base.matvec(x).map_err(map_err)?;
                if let Some((left, right)) = low_rank {
                    let t = right.matvec(x).map_err(map_err)?;
                    let dy = left.matvec(&t).map_err(map_err)?;
                    for (a, &b) in y.iter_mut().zip(&dy) {
                        *a += b;
                    }
                }
                if let Some(sp) = sparse_residual {
                    sp.add_matvec_into(x, &mut y)?;
                }
                Ok(y)
            }
        }
    }
}

struct RuntimeOperator {
    cpu: QmfOperator,
    gpu: Option<GpuOperator>,
}

impl RuntimeOperator {
    fn cpu_only(cpu: QmfOperator) -> Self {
        Self { cpu, gpu: None }
    }

    fn matvec(&self, x: &[f64]) -> Result<Vec<f64>, MatrixError> {
        if let Some(gpu) = &self.gpu {
            gpu.matvec(x)
        } else {
            self.cpu.matvec(x)
        }
    }

    fn decode_row(&self, row: usize) -> Result<Vec<f64>, MatrixError> {
        self.cpu.decode_row(row)
    }
}

struct GpuUploadContext {
    engine: Arc<WgpuMatvecEngine>,
    latent_gpu_cache: HashMap<usize, Arc<PersistentGpuMatrix>>,
    vram_weight_bytes: u64,
    gpu_resident_operators: usize,
    cpu_fallback_operators: usize,
}

impl GpuUploadContext {
    fn new(engine: Arc<WgpuMatvecEngine>) -> Self {
        Self {
            engine,
            latent_gpu_cache: HashMap::new(),
            vram_weight_bytes: 0,
            gpu_resident_operators: 0,
            cpu_fallback_operators: 0,
        }
    }

    fn upload_or_cache_latent(
        &mut self,
        factor: &Arc<QuantizedFactor>,
    ) -> Option<Arc<PersistentGpuMatrix>> {
        let key = Arc::as_ptr(factor) as usize;
        if let Some(existing) = self.latent_gpu_cache.get(&key) {
            return Some(Arc::clone(existing));
        }
        let uploaded = Arc::new(self.engine.upload_quantized_factor(factor).ok()?);
        self.vram_weight_bytes += uploaded.vram_bytes();
        self.latent_gpu_cache.insert(key, Arc::clone(&uploaded));
        Some(uploaded)
    }

    fn wrap(&mut self, cpu: QmfOperator) -> RuntimeOperator {
        let gpu_op = match &cpu {
            QmfOperator::DenseF32(m) => self.engine.upload_f32(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(Arc::new(p))
            }),
            QmfOperator::DenseF16(m) => self.engine.upload_fp16(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(Arc::new(p))
            }),
            QmfOperator::DenseBf16(m) => self.engine.upload_bf16(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(Arc::new(p))
            }),
            QmfOperator::Int8Sym(m) => self.engine.upload_int8_sym(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(Arc::new(p))
            }),
            QmfOperator::Int4Sym(m) => self.engine.upload_int4_sym(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(Arc::new(p))
            }),
            QmfOperator::Int4Asym(m) => self.engine.upload_int4_asym(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(Arc::new(p))
            }),
            QmfOperator::LowRankSvd(svd) => {
                let left = self.engine.upload_quantized_factor(svd.left()).ok();
                let right = self.engine.upload_quantized_factor(svd.right()).ok();
                match (left, right) {
                    (Some(l), Some(r)) => {
                        self.vram_weight_bytes += l.vram_bytes() + r.vram_bytes();
                        Some(GpuOperator::LowRankSvd {
                            left: Arc::new(l),
                            right: Arc::new(r),
                        })
                    }
                    _ => None,
                }
            }
            QmfOperator::RmqSharedBasis {
                u_shared,
                core,
                vt_shared,
                sparse_residual,
                ..
            } => {
                let u_gpu = self.upload_or_cache_latent(u_shared);
                let vt_gpu = self.upload_or_cache_latent(vt_shared);
                let c_gpu = self.engine.upload_quantized_factor(core).ok();
                match (u_gpu, c_gpu, vt_gpu) {
                    (Some(u), Some(c), Some(vt)) => {
                        self.vram_weight_bytes += c.vram_bytes();
                        Some(GpuOperator::RmqSharedBasis {
                            u_shared: u,
                            core: Arc::new(c),
                            vt_shared: vt,
                            sparse_residual: sparse_residual.clone(),
                        })
                    }
                    _ => None,
                }
            }
            QmfOperator::RmqSharedCore {
                left,
                shared_core,
                right,
                sparse_residual,
                ..
            } => {
                let s_gpu = self.upload_or_cache_latent(shared_core);
                let l_gpu = self.engine.upload_quantized_factor(left).ok();
                let r_gpu = self.engine.upload_quantized_factor(right).ok();
                match (l_gpu, s_gpu, r_gpu) {
                    (Some(l), Some(s), Some(r)) => {
                        self.vram_weight_bytes += l.vram_bytes() + r.vram_bytes();
                        Some(GpuOperator::RmqSharedCore {
                            left: Arc::new(l),
                            shared_core: s,
                            right: Arc::new(r),
                            sparse_residual: sparse_residual.clone(),
                        })
                    }
                    _ => None,
                }
            }
            QmfOperator::Int3Asym(m) => self.engine.upload_int3_asym(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(Arc::new(p))
            }),
            QmfOperator::Int2Asym(m) => self.engine.upload_int2_asym(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(Arc::new(p))
            }),
            QmfOperator::TernaryPacked(m) => self.engine.upload_ternary_packed(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(Arc::new(p))
            }),
            QmfOperator::VectorQuant(m) => self.engine.upload_vector_quant(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(Arc::new(p))
            }),
            QmfOperator::EchoCorrected(echo) => {
                let base_gpu = match echo.base() {
                    EchoBaseOperator::Int3Asym(m) => self.engine.upload_int3_asym(m).ok(),
                    EchoBaseOperator::Int2Asym(m) => self.engine.upload_int2_asym(m).ok(),
                    EchoBaseOperator::Ternary(m) => self.engine.upload_ternary_packed(m).ok(),
                    EchoBaseOperator::VectorQuant(m) => self.engine.upload_vector_quant(m).ok(),
                };
                if let Some(base) = base_gpu {
                    self.vram_weight_bytes += base.vram_bytes();
                    let low_rank = if let Some(lr) = echo.low_rank_adapter() {
                        match (
                            self.engine.upload_quantized_factor(lr.left()).ok(),
                            self.engine.upload_quantized_factor(lr.right()).ok(),
                        ) {
                            (Some(l), Some(r)) => {
                                self.vram_weight_bytes += l.vram_bytes() + r.vram_bytes();
                                Some((Arc::new(l), Arc::new(r)))
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    Some(GpuOperator::EchoCorrected {
                        base: Arc::new(base),
                        low_rank,
                        sparse_residual: echo.sparse_residual().cloned(),
                    })
                } else {
                    None
                }
            }
        };
        if gpu_op.is_some() {
            self.gpu_resident_operators += 1;
        } else {
            self.cpu_fallback_operators += 1;
        }
        RuntimeOperator { cpu, gpu: gpu_op }
    }
}

struct TransformerBlock {
    input_norm: Vec<f64>,
    q_proj: RuntimeOperator,
    q_bias: Option<Vec<f64>>,
    k_proj: RuntimeOperator,
    k_bias: Option<Vec<f64>>,
    v_proj: RuntimeOperator,
    v_bias: Option<Vec<f64>>,
    o_proj: RuntimeOperator,
    post_attn_norm: Vec<f64>,
    gate_proj: RuntimeOperator,
    up_proj: RuntimeOperator,
    down_proj: RuntimeOperator,
}

/// Executable decoder-only transformer loaded directly from a QMF v1 container.
///
/// Retains embeddings and linear projections in their native compressed `QmfOperator`
/// form (zero dense `f64` reconstruction of embeddings or weights), deduplicates
/// shared RMQ latent structures across layers via `Arc`, supports Qwen2 attention
/// biases, INT8 KV-cache, and optionally executes the full transformer inside a single
/// GPU submission (`QCoreBackend::GpuFull`) or hybrid GPU matvecs (`QCoreBackend::GpuHybrid`).
pub struct QCoreModel {
    config: TransformerConfig,
    embed_tokens: Arc<RuntimeOperator>,
    layers: Vec<TransformerBlock>,
    final_norm: Vec<f64>,
    lm_head: Arc<RuntimeOperator>,
    gpu_full: Option<Arc<GpuFullModel>>,
    shared_latents_deduped: usize,
    backend_stats: QCoreBackendStats,
}

fn try_build_gpu_full(
    engine: Arc<WgpuMatvecEngine>,
    config: &TransformerConfig,
    layers: &[TransformerBlock],
    final_norm: &[f64],
    lm_head: &RuntimeOperator,
) -> Option<Arc<GpuFullModel>> {
    let lm_full = lm_head.gpu.as_ref()?.to_full_op()?;
    let mut specs = Vec::with_capacity(layers.len());
    for blk in layers {
        specs.push(GpuFullLayerSpec {
            input_norm: blk.input_norm.clone(),
            q_proj: blk.q_proj.gpu.as_ref()?.to_full_op()?,
            q_bias: blk.q_bias.clone(),
            k_proj: blk.k_proj.gpu.as_ref()?.to_full_op()?,
            k_bias: blk.k_bias.clone(),
            v_proj: blk.v_proj.gpu.as_ref()?.to_full_op()?,
            v_bias: blk.v_bias.clone(),
            o_proj: blk.o_proj.gpu.as_ref()?.to_full_op()?,
            post_attn_norm: blk.post_attn_norm.clone(),
            gate_proj: blk.gate_proj.gpu.as_ref()?.to_full_op()?,
            up_proj: blk.up_proj.gpu.as_ref()?.to_full_op()?,
            down_proj: blk.down_proj.gpu.as_ref()?.to_full_op()?,
        });
    }
    let full_cfg = GpuFullConfig {
        vocab_size: config.vocab_size,
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        num_layers: config.num_layers,
        num_heads: config.num_heads,
        num_kv_heads: config.num_kv_heads,
        rms_norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta,
        max_kv_len: config.max_seq_len.min(4096),
    };
    GpuFullModel::new(engine, full_cfg, specs, final_norm.to_vec(), lm_full)
        .ok()
        .map(Arc::new)
}

/// Output of autoregressive generation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GenerationResult {
    pub prompt_tokens: Vec<usize>,
    pub generated_tokens: Vec<usize>,
    pub elapsed_micros: u64,
    /// Emitted tokens per second over the entire generation call, including prefill.
    /// This is not a decode-only rate; prompt tokens are excluded from the numerator.
    pub tokens_per_second: f64,
}

fn generation_rate(generated_tokens: usize, elapsed_micros: u64) -> f64 {
    if elapsed_micros == 0 {
        0.0
    } else {
        generated_tokens as f64 * 1_000_000.0 / elapsed_micros as f64
    }
}

/// Strictly runtime-only, reversible causal intervention applied during a traced forward pass.
/// Never mutates stored QMF weights.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LayerIntervention {
    /// Scale the residual stream output of `layer` by `scale`.
    ScaleResidual { layer: usize, scale: f64 },
    /// Zero out the self-attention residual contribution of `layer`.
    AblateAttention { layer: usize },
    /// Zero out the SwiGLU MLP residual contribution of `layer`.
    AblateMlp { layer: usize },
}

/// Output logits and per-layer post-block hidden states captured during one forward step.
#[derive(Debug, Clone)]
pub struct ForwardTrace {
    pub logits: Vec<f64>,
    pub layer_hidden: Vec<Vec<f64>>,
}

/// Quantitative comparison of a transformed model against a dense reference model on identical tokens.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InferenceComparison {
    pub sequence_len: usize,
    pub reference_perplexity: f64,
    pub transformed_perplexity: f64,
    pub perplexity_delta: f64,
    pub mean_kl_divergence: f64,
    pub max_kl_divergence: f64,
    pub mean_logit_rmse: f64,
    pub top1_token_agreement: f64,
}

impl QCoreModel {
    /// Load an executable model directly from a validated QMF v1 archive (CPU backend).
    pub fn from_qmf(
        archive: &QmfArchiveV1,
        config: TransformerConfig,
    ) -> Result<Self, RuntimeError> {
        Self::from_qmf_with_backend(archive, config, QCoreBackend::Cpu)
    }

    /// Load an executable model from a validated QMF v1 archive with an explicit backend.
    pub fn from_qmf_with_backend(
        archive: &QmfArchiveV1,
        config: TransformerConfig,
        backend: QCoreBackend,
    ) -> Result<Self, RuntimeError> {
        Self::build_model(
            config,
            backend,
            |name, cache| archive.load_operator_cached(name, cache),
            |name| archive.load_vector_f64(name),
            |name, expected| {
                let entry = archive
                    .manifest()
                    .tensors
                    .iter()
                    .find(|tensor| tensor.name == name);
                bias::validate_qmf(entry, name, expected)?;
                let shape = entry.map(|tensor| tensor.shape.as_slice());
                bias::load(name, shape, expected, || Ok(archive.load_vector_f64(name)?))
            },
        )
    }

    /// Load an executable model lazily from a file-backed [`QmfFileReader`] (CPU backend).
    pub fn from_qmf_reader<R: std::io::Read + std::io::Seek>(
        reader: &mut QmfFileReader<R>,
        config: TransformerConfig,
    ) -> Result<Self, RuntimeError> {
        Self::from_qmf_reader_with_backend(reader, config, QCoreBackend::Cpu)
    }

    /// Load an executable model lazily from a file-backed [`QmfFileReader`] with an explicit
    /// execution backend (`Cpu` or `GpuHybrid` with persistent VRAM-resident compressed weights).
    pub fn from_qmf_reader_with_backend<R: std::io::Read + std::io::Seek>(
        reader: &mut QmfFileReader<R>,
        config: TransformerConfig,
        backend: QCoreBackend,
    ) -> Result<Self, RuntimeError> {
        let mut latent_cache = QmfLatentCache::new();
        config.validate()?;
        let mut gpu_ctx = match backend {
            QCoreBackend::Cpu => None,
            QCoreBackend::GpuHybrid | QCoreBackend::GpuFull => WgpuMatvecEngine::try_new()
                .ok()
                .map(|e| GpuUploadContext::new(Arc::new(e))),
        };
        let mut wrap_op = |op: QmfOperator| -> RuntimeOperator {
            if let Some(ctx) = gpu_ctx.as_mut() {
                ctx.wrap(op)
            } else {
                RuntimeOperator::cpu_only(op)
            }
        };

        let embed_raw =
            reader.load_operator_cached("model.embed_tokens.weight", &mut latent_cache)?;
        if embed_raw.rows() != config.vocab_size || embed_raw.cols() != config.hidden_size {
            return Err(RuntimeError::InvalidConfig(format!(
                "embed_tokens shape {}x{} does not match vocab_size={} hidden_size={}",
                embed_raw.rows(),
                embed_raw.cols(),
                config.vocab_size,
                config.hidden_size
            )));
        }
        let embed_op = if config.tie_word_embeddings {
            Arc::new(wrap_op(embed_raw))
        } else {
            Arc::new(RuntimeOperator::cpu_only(embed_raw))
        };

        // Grow only after real layers load; a declared count alone must not reserve enormous RAM.
        let mut layers = Vec::new();
        for l in 0..config.num_layers {
            let p = format!("model.layers.{l}");
            layers.push(TransformerBlock {
                input_norm: reader.load_vector_f64(&format!("{p}.input_layernorm.weight"))?,
                q_proj: wrap_op(reader.load_operator_cached(
                    &format!("{p}.self_attn.q_proj.weight"),
                    &mut latent_cache,
                )?),
                q_bias: bias::from_reader(
                    reader,
                    &format!("{p}.self_attn.q_proj.bias"),
                    config.hidden_size,
                )?,
                k_proj: wrap_op(reader.load_operator_cached(
                    &format!("{p}.self_attn.k_proj.weight"),
                    &mut latent_cache,
                )?),
                k_bias: bias::from_reader(
                    reader,
                    &format!("{p}.self_attn.k_proj.bias"),
                    config.num_kv_heads * config.head_dim(),
                )?,
                v_proj: wrap_op(reader.load_operator_cached(
                    &format!("{p}.self_attn.v_proj.weight"),
                    &mut latent_cache,
                )?),
                v_bias: bias::from_reader(
                    reader,
                    &format!("{p}.self_attn.v_proj.bias"),
                    config.num_kv_heads * config.head_dim(),
                )?,
                o_proj: wrap_op(reader.load_operator_cached(
                    &format!("{p}.self_attn.o_proj.weight"),
                    &mut latent_cache,
                )?),
                post_attn_norm: reader
                    .load_vector_f64(&format!("{p}.post_attention_layernorm.weight"))?,
                gate_proj: wrap_op(reader.load_operator_cached(
                    &format!("{p}.mlp.gate_proj.weight"),
                    &mut latent_cache,
                )?),
                up_proj: wrap_op(
                    reader.load_operator_cached(
                        &format!("{p}.mlp.up_proj.weight"),
                        &mut latent_cache,
                    )?,
                ),
                down_proj: wrap_op(reader.load_operator_cached(
                    &format!("{p}.mlp.down_proj.weight"),
                    &mut latent_cache,
                )?),
            });
        }

        let final_norm = reader.load_vector_f64("model.norm.weight")?;
        let lm_head = if config.tie_word_embeddings {
            Arc::clone(&embed_op)
        } else {
            Arc::new(wrap_op(
                reader.load_operator_cached("lm_head.weight", &mut latent_cache)?,
            ))
        };

        let gpu_full = if backend == QCoreBackend::GpuFull {
            gpu_ctx.as_ref().and_then(|ctx| {
                try_build_gpu_full(
                    Arc::clone(&ctx.engine),
                    &config,
                    &layers,
                    &final_norm,
                    &lm_head,
                )
            })
        } else {
            None
        };

        let backend_stats = match gpu_ctx {
            Some(ctx) => {
                let mode_tag = if gpu_full.is_some() {
                    "WGPU Full-Resident"
                } else {
                    "WGPU Hybrid"
                };
                QCoreBackendStats {
                    backend: format!(
                        "{mode_tag} ({} / {})",
                        ctx.engine.adapter_name(),
                        ctx.engine.backend_name()
                    ),
                    gpu_adapter: Some(ctx.engine.adapter_name().to_owned()),
                    gpu_resident_operators: ctx.gpu_resident_operators,
                    cpu_fallback_operators: ctx.cpu_fallback_operators,
                    vram_weight_bytes: ctx.vram_weight_bytes
                        + gpu_full.as_ref().map_or(0, |f| f.vram_runtime_bytes()),
                    shared_latents_deduped_gpu: ctx.latent_gpu_cache.len(),
                }
            }
            None => QCoreBackendStats {
                backend: "Rayon Multi-Threaded CPU".to_owned(),
                gpu_adapter: None,
                gpu_resident_operators: 0,
                cpu_fallback_operators: config.num_layers * 7 + 1,
                vram_weight_bytes: 0,
                shared_latents_deduped_gpu: 0,
            },
        };

        Ok(Self {
            config,
            embed_tokens: embed_op,
            layers,
            final_norm,
            lm_head,
            gpu_full,
            shared_latents_deduped: latent_cache.len(),
            backend_stats,
        })
    }

    fn build_model<FOp, FVec>(
        config: TransformerConfig,
        backend: QCoreBackend,
        load_op: FOp,
        load_vec: FVec,
        load_bias: impl Fn(&str, usize) -> Result<Option<Vec<f64>>, RuntimeError>,
    ) -> Result<Self, RuntimeError>
    where
        FOp: Fn(&str, &mut QmfLatentCache) -> Result<QmfOperator, QmfError>,
        FVec: Fn(&str) -> Result<Vec<f64>, QmfError>,
    {
        config.validate()?;
        let mut latent_cache = QmfLatentCache::new();
        let mut gpu_ctx = match backend {
            QCoreBackend::Cpu => None,
            QCoreBackend::GpuHybrid | QCoreBackend::GpuFull => WgpuMatvecEngine::try_new()
                .ok()
                .map(|e| GpuUploadContext::new(Arc::new(e))),
        };
        let mut wrap_op = |op: QmfOperator| -> RuntimeOperator {
            if let Some(ctx) = gpu_ctx.as_mut() {
                ctx.wrap(op)
            } else {
                RuntimeOperator::cpu_only(op)
            }
        };

        let embed_raw = load_op("model.embed_tokens.weight", &mut latent_cache)?;
        if embed_raw.rows() != config.vocab_size || embed_raw.cols() != config.hidden_size {
            return Err(RuntimeError::InvalidConfig(format!(
                "embed_tokens shape {}x{} does not match vocab_size={} hidden_size={}",
                embed_raw.rows(),
                embed_raw.cols(),
                config.vocab_size,
                config.hidden_size
            )));
        }
        let embed_op = if config.tie_word_embeddings {
            Arc::new(wrap_op(embed_raw))
        } else {
            Arc::new(RuntimeOperator::cpu_only(embed_raw))
        };

        let mut layers = Vec::new();
        for l in 0..config.num_layers {
            layers.push(TransformerBlock {
                input_norm: load_vec(&format!("model.layers.{l}.input_layernorm.weight"))?,
                q_proj: wrap_op(load_op(
                    &format!("model.layers.{l}.self_attn.q_proj.weight"),
                    &mut latent_cache,
                )?),
                q_bias: load_bias(
                    &format!("model.layers.{l}.self_attn.q_proj.bias"),
                    config.hidden_size,
                )?,
                k_proj: wrap_op(load_op(
                    &format!("model.layers.{l}.self_attn.k_proj.weight"),
                    &mut latent_cache,
                )?),
                k_bias: load_bias(
                    &format!("model.layers.{l}.self_attn.k_proj.bias"),
                    config.num_kv_heads * config.head_dim(),
                )?,
                v_proj: wrap_op(load_op(
                    &format!("model.layers.{l}.self_attn.v_proj.weight"),
                    &mut latent_cache,
                )?),
                v_bias: load_bias(
                    &format!("model.layers.{l}.self_attn.v_proj.bias"),
                    config.num_kv_heads * config.head_dim(),
                )?,
                o_proj: wrap_op(load_op(
                    &format!("model.layers.{l}.self_attn.o_proj.weight"),
                    &mut latent_cache,
                )?),
                post_attn_norm: load_vec(&format!(
                    "model.layers.{l}.post_attention_layernorm.weight"
                ))?,
                gate_proj: wrap_op(load_op(
                    &format!("model.layers.{l}.mlp.gate_proj.weight"),
                    &mut latent_cache,
                )?),
                up_proj: wrap_op(load_op(
                    &format!("model.layers.{l}.mlp.up_proj.weight"),
                    &mut latent_cache,
                )?),
                down_proj: wrap_op(load_op(
                    &format!("model.layers.{l}.mlp.down_proj.weight"),
                    &mut latent_cache,
                )?),
            });
        }

        let final_norm = load_vec("model.norm.weight")?;
        let lm_head = if config.tie_word_embeddings {
            Arc::clone(&embed_op)
        } else {
            Arc::new(wrap_op(load_op("lm_head.weight", &mut latent_cache)?))
        };

        let gpu_full = if backend == QCoreBackend::GpuFull {
            gpu_ctx.as_ref().and_then(|ctx| {
                try_build_gpu_full(
                    Arc::clone(&ctx.engine),
                    &config,
                    &layers,
                    &final_norm,
                    &lm_head,
                )
            })
        } else {
            None
        };

        let backend_stats = match gpu_ctx {
            Some(ctx) => {
                let mode_tag = if gpu_full.is_some() {
                    "WGPU Full-Resident"
                } else {
                    "WGPU Hybrid"
                };
                QCoreBackendStats {
                    backend: format!(
                        "{mode_tag} ({} / {})",
                        ctx.engine.adapter_name(),
                        ctx.engine.backend_name()
                    ),
                    gpu_adapter: Some(ctx.engine.adapter_name().to_owned()),
                    gpu_resident_operators: ctx.gpu_resident_operators,
                    cpu_fallback_operators: ctx.cpu_fallback_operators,
                    vram_weight_bytes: ctx.vram_weight_bytes
                        + gpu_full.as_ref().map_or(0, |f| f.vram_runtime_bytes()),
                    shared_latents_deduped_gpu: ctx.latent_gpu_cache.len(),
                }
            }
            None => QCoreBackendStats {
                backend: "Rayon Multi-Threaded CPU".to_owned(),
                gpu_adapter: None,
                gpu_resident_operators: 0,
                cpu_fallback_operators: config.num_layers * 7 + 1,
                vram_weight_bytes: 0,
                shared_latents_deduped_gpu: 0,
            },
        };

        Ok(Self {
            config,
            embed_tokens: embed_op,
            layers,
            final_norm,
            lm_head,
            gpu_full,
            shared_latents_deduped: latent_cache.len(),
            backend_stats,
        })
    }

    pub fn backend_stats(&self) -> &QCoreBackendStats {
        &self.backend_stats
    }

    pub fn shared_latents_deduped(&self) -> usize {
        self.shared_latents_deduped
    }

    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Execute one autoregressive forward step for `token_id` at `position`, updating `cache`.
    /// Uses single-submission Full GPU-Resident execution whenever [`QCoreBackend::GpuFull`] is active.
    pub fn forward_token(
        &self,
        token_id: usize,
        position: usize,
        cache: &mut KvCache,
    ) -> Result<Vec<f64>, RuntimeError> {
        if let Some(full) = &self.gpu_full {
            if token_id >= self.config.vocab_size {
                return Err(RuntimeError::TokenOutOfRange {
                    token: token_id,
                    vocab_size: self.config.vocab_size,
                });
            }
            if position >= self.config.max_seq_len {
                return Err(RuntimeError::ContextLengthExceeded {
                    position,
                    max_seq_len: self.config.max_seq_len,
                });
            }
            let embed = self.embed_tokens.decode_row(token_id)?;
            let gpu_kv_mode = match cache.mode {
                KvCacheMode::Fp64 => GpuKvMode::Fp32,
                KvCacheMode::Int8 => GpuKvMode::Int8,
            };
            let gkv = cache.gpu_kv.get_or_insert_with(|| {
                full.create_kv_cache_with_mode(self.config.max_seq_len.min(4096), gpu_kv_mode)
            });
            return full
                .forward_embedding(&embed, position, gkv)
                .map_err(|e| RuntimeError::InvalidConfig(format!("GPU Full error: {e}")));
        }
        Ok(self
            .forward_token_with_trace(token_id, position, cache, None)?
            .logits)
    }

    /// Collect per-linear-layer input activation RMS and `64×64` block-diagonal Hessian covariances
    /// (`H_b = (2/N) X_bᵀ X_b`) across calibration token sequences for activation-aware GPTQ/AWQ/ECHO.
    pub fn collect_calibration_profile(
        &self,
        model_id: &str,
        sequences: &[Vec<usize>],
    ) -> Result<ModelCalibrationProfile, RuntimeError> {
        let bs = DEFAULT_HESSIAN_BLOCK_SIZE;
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let mut attn_in_accs: Vec<LayerCalibrationAccumulator> = (0..self.layers.len())
            .map(|l| LayerCalibrationAccumulator::new(format!("layer.{l}.attn_in"), h, bs))
            .collect();
        let mut o_in_accs: Vec<LayerCalibrationAccumulator> = (0..self.layers.len())
            .map(|l| {
                LayerCalibrationAccumulator::new(
                    format!("model.layers.{l}.self_attn.o_proj.weight"),
                    h,
                    bs,
                )
            })
            .collect();
        let mut mlp_in_accs: Vec<LayerCalibrationAccumulator> = (0..self.layers.len())
            .map(|l| LayerCalibrationAccumulator::new(format!("layer.{l}.mlp_in"), h, bs))
            .collect();
        let mut down_in_accs: Vec<LayerCalibrationAccumulator> = (0..self.layers.len())
            .map(|l| {
                LayerCalibrationAccumulator::new(
                    format!("model.layers.{l}.mlp.down_proj.weight"),
                    inter,
                    bs,
                )
            })
            .collect();
        let mut lm_head_acc = LayerCalibrationAccumulator::new("lm_head.weight", h, bs);

        let head_dim = self.config.head_dim();
        let kv_groups = self.config.num_heads / self.config.num_kv_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut total_tokens = 0usize;

        for seq in sequences {
            let mut cache = KvCache::new(self.config.num_layers);
            for (pos, &tok) in seq.iter().enumerate() {
                if tok >= self.config.vocab_size || pos >= self.config.max_seq_len {
                    break;
                }
                total_tokens += 1;
                let mut hidden = self.embed_tokens.decode_row(tok)?;
                for (layer_idx, block) in self.layers.iter().enumerate() {
                    let normed = rms_norm(&hidden, &block.input_norm, self.config.rms_norm_eps);
                    attn_in_accs[layer_idx].observe(&normed);

                    let mut q = block.q_proj.matvec(&normed)?;
                    if let Some(b) = &block.q_bias {
                        for (qv, &bv) in q.iter_mut().zip(b) {
                            *qv += bv;
                        }
                    }
                    let mut k = block.k_proj.matvec(&normed)?;
                    if let Some(b) = &block.k_bias {
                        for (kv, &bv) in k.iter_mut().zip(b) {
                            *kv += bv;
                        }
                    }
                    let mut v = block.v_proj.matvec(&normed)?;
                    if let Some(b) = &block.v_bias {
                        for (vv, &bv) in v.iter_mut().zip(b) {
                            *vv += bv;
                        }
                    }
                    for hd in 0..self.config.num_heads {
                        apply_rope(
                            &mut q[hd * head_dim..(hd + 1) * head_dim],
                            pos,
                            self.config.rope_theta,
                        );
                    }
                    for kh in 0..self.config.num_kv_heads {
                        apply_rope(
                            &mut k[kh * head_dim..(kh + 1) * head_dim],
                            pos,
                            self.config.rope_theta,
                        );
                    }
                    cache.keys[layer_idx].push(k);
                    cache.values[layer_idx].push(v);
                    let seq_len = cache.keys[layer_idx].len();
                    let mut attn_out = vec![0.0_f64; h];
                    for hd in 0..self.config.num_heads {
                        let kv_h = hd / kv_groups;
                        let q_head = &q[hd * head_dim..(hd + 1) * head_dim];
                        let mut scores = Vec::with_capacity(seq_len);
                        for k_vec in &cache.keys[layer_idx] {
                            let k_head = &k_vec[kv_h * head_dim..(kv_h + 1) * head_dim];
                            let dot: f64 = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum();
                            scores.push(dot * scale);
                        }
                        let probs = softmax(&scores);
                        let out_head = &mut attn_out[hd * head_dim..(hd + 1) * head_dim];
                        for (t, &p) in probs.iter().enumerate() {
                            let v_head =
                                &cache.values[layer_idx][t][kv_h * head_dim..(kv_h + 1) * head_dim];
                            for (o, &vv) in out_head.iter_mut().zip(v_head) {
                                *o += p * vv;
                            }
                        }
                    }
                    o_in_accs[layer_idx].observe(&attn_out);
                    let proj_out = block.o_proj.matvec(&attn_out)?;
                    for (hv, pv) in hidden.iter_mut().zip(proj_out) {
                        *hv += pv;
                    }

                    let mlp_in = rms_norm(&hidden, &block.post_attn_norm, self.config.rms_norm_eps);
                    mlp_in_accs[layer_idx].observe(&mlp_in);
                    let gate = block.gate_proj.matvec(&mlp_in)?;
                    let up = block.up_proj.matvec(&mlp_in)?;
                    let activated: Vec<f64> =
                        gate.iter().zip(up).map(|(&g, u)| silu(g) * u).collect();
                    down_in_accs[layer_idx].observe(&activated);
                    let mlp_out = block.down_proj.matvec(&activated)?;
                    for (hv, mv) in hidden.iter_mut().zip(mlp_out) {
                        *hv += mv;
                    }
                }
                let final_hidden = rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps);
                lm_head_acc.observe(&final_hidden);
            }
        }

        let mut layers_map = BTreeMap::new();
        for (l, ((a_acc, o_acc), (m_acc, d_acc))) in attn_in_accs
            .into_iter()
            .zip(o_in_accs)
            .zip(mlp_in_accs.into_iter().zip(down_in_accs))
            .enumerate()
        {
            let a_stats = a_acc.finalize();
            for proj in ["q_proj", "k_proj", "v_proj"] {
                let name = format!("model.layers.{l}.self_attn.{proj}.weight");
                let mut s = a_stats.clone();
                s.tensor_name = name.clone();
                layers_map.insert(name, s);
            }
            let o_stats = o_acc.finalize();
            layers_map.insert(o_stats.tensor_name.clone(), o_stats);

            let m_stats = m_acc.finalize();
            for proj in ["gate_proj", "up_proj"] {
                let name = format!("model.layers.{l}.mlp.{proj}.weight");
                let mut s = m_stats.clone();
                s.tensor_name = name.clone();
                layers_map.insert(name, s);
            }
            let d_stats = d_acc.finalize();
            layers_map.insert(d_stats.tensor_name.clone(), d_stats);
        }
        let lm_stats = lm_head_acc.finalize();
        if self.config.tie_word_embeddings {
            let mut emb_s = lm_stats.clone();
            emb_s.tensor_name = "model.embed_tokens.weight".to_owned();
            layers_map.insert("model.embed_tokens.weight".to_owned(), emb_s);
        }
        layers_map.insert("lm_head.weight".to_owned(), lm_stats);

        Ok(ModelCalibrationProfile {
            model_id: model_id.to_owned(),
            calibration_tokens: total_tokens,
            block_size: bs,
            layers: layers_map,
        })
    }

    /// Execute one autoregressive forward step while capturing post-block hidden states
    /// for every transformer layer and optionally applying a reversible [`LayerIntervention`].
    pub fn forward_token_with_trace(
        &self,
        token_id: usize,
        position: usize,
        cache: &mut KvCache,
        intervention: Option<LayerIntervention>,
    ) -> Result<ForwardTrace, RuntimeError> {
        if token_id >= self.config.vocab_size {
            return Err(RuntimeError::TokenOutOfRange {
                token: token_id,
                vocab_size: self.config.vocab_size,
            });
        }
        if position >= self.config.max_seq_len {
            return Err(RuntimeError::ContextLengthExceeded {
                position,
                max_seq_len: self.config.max_seq_len,
            });
        }

        let head_dim = self.config.head_dim();
        let kv_groups = self.config.num_heads / self.config.num_kv_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let mut hidden = self.embed_tokens.decode_row(token_id)?;
        let mut layer_hidden = Vec::with_capacity(self.layers.len());

        for (layer_idx, block) in self.layers.iter().enumerate() {
            // Pre-attention RMSNorm
            let normed = rms_norm(&hidden, &block.input_norm, self.config.rms_norm_eps);

            // Direct compressed linear projections (+ optional Qwen2 attention biases)
            let mut q = block.q_proj.matvec(&normed)?;
            if let Some(b) = &block.q_bias {
                for (qv, &bv) in q.iter_mut().zip(b) {
                    *qv += bv;
                }
            }
            let mut k = block.k_proj.matvec(&normed)?;
            if let Some(b) = &block.k_bias {
                for (kv, &bv) in k.iter_mut().zip(b) {
                    *kv += bv;
                }
            }
            let mut v = block.v_proj.matvec(&normed)?;
            if let Some(b) = &block.v_bias {
                for (vv, &bv) in v.iter_mut().zip(b) {
                    *vv += bv;
                }
            }

            // Apply RoPE to each Q and K head
            for h in 0..self.config.num_heads {
                apply_rope(
                    &mut q[h * head_dim..(h + 1) * head_dim],
                    position,
                    self.config.rope_theta,
                );
            }
            for kh in 0..self.config.num_kv_heads {
                apply_rope(
                    &mut k[kh * head_dim..(kh + 1) * head_dim],
                    position,
                    self.config.rope_theta,
                );
            }

            // Multi-head / Grouped-query causal attention (supports both FP64 and INT8 KV-cache)
            let mut attn_out = vec![0.0_f64; self.config.hidden_size];
            match cache.mode {
                KvCacheMode::Fp64 => {
                    cache.keys[layer_idx].push(k);
                    cache.values[layer_idx].push(v);
                    let seq_len = cache.keys[layer_idx].len();
                    for h in 0..self.config.num_heads {
                        let kv_h = h / kv_groups;
                        let q_head = &q[h * head_dim..(h + 1) * head_dim];
                        let mut scores = Vec::with_capacity(seq_len);
                        for k_vec in &cache.keys[layer_idx] {
                            let k_head = &k_vec[kv_h * head_dim..(kv_h + 1) * head_dim];
                            let dot: f64 = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum();
                            scores.push(dot * scale);
                        }
                        let probs = softmax(&scores);
                        let out_head = &mut attn_out[h * head_dim..(h + 1) * head_dim];
                        for (t, &p) in probs.iter().enumerate() {
                            let v_head =
                                &cache.values[layer_idx][t][kv_h * head_dim..(kv_h + 1) * head_dim];
                            for (o, &vv) in out_head.iter_mut().zip(v_head) {
                                *o += p * vv;
                            }
                        }
                    }
                }
                KvCacheMode::Int8 => {
                    cache.keys_i8[layer_idx].push(Int8KvEntry::encode(&k));
                    cache.values_i8[layer_idx].push(Int8KvEntry::encode(&v));
                    let seq_len = cache.keys_i8[layer_idx].len();
                    for h in 0..self.config.num_heads {
                        let kv_h = h / kv_groups;
                        let q_head = &q[h * head_dim..(h + 1) * head_dim];
                        let mut scores = Vec::with_capacity(seq_len);
                        for k_entry in &cache.keys_i8[layer_idx] {
                            let ks = k_entry.scale();
                            let k_codes = &k_entry.codes[kv_h * head_dim..(kv_h + 1) * head_dim];
                            let dot: f64 = q_head
                                .iter()
                                .zip(k_codes)
                                .map(|(&qv, &kc)| qv * f64::from(kc) * ks)
                                .sum();
                            scores.push(dot * scale);
                        }
                        let probs = softmax(&scores);
                        let out_head = &mut attn_out[h * head_dim..(h + 1) * head_dim];
                        for (t, &p) in probs.iter().enumerate() {
                            let v_entry = &cache.values_i8[layer_idx][t];
                            let vs = v_entry.scale();
                            let v_codes = &v_entry.codes[kv_h * head_dim..(kv_h + 1) * head_dim];
                            for (o, &vc) in out_head.iter_mut().zip(v_codes) {
                                *o += p * (f64::from(vc) * vs);
                            }
                        }
                    }
                }
            }

            // Output projection + residual connection (optional reversible attention ablation)
            let ablate_attn = matches!(
                intervention,
                Some(LayerIntervention::AblateAttention { layer }) if layer == layer_idx
            );
            if !ablate_attn {
                let proj_out = block.o_proj.matvec(&attn_out)?;
                for (h_val, p_val) in hidden.iter_mut().zip(proj_out) {
                    *h_val += p_val;
                }
            }

            // Post-attention RMSNorm + SwiGLU MLP + residual connection (optional reversible MLP ablation)
            let ablate_mlp = matches!(
                intervention,
                Some(LayerIntervention::AblateMlp { layer }) if layer == layer_idx
            );
            if !ablate_mlp {
                let mlp_in = rms_norm(&hidden, &block.post_attn_norm, self.config.rms_norm_eps);
                let gate = block.gate_proj.matvec(&mlp_in)?;
                let up = block.up_proj.matvec(&mlp_in)?;
                let activated: Vec<f64> = gate.iter().zip(up).map(|(&g, u)| silu(g) * u).collect();
                let mlp_out = block.down_proj.matvec(&activated)?;
                for (h_val, m_val) in hidden.iter_mut().zip(mlp_out) {
                    *h_val += m_val;
                }
            }

            if let Some(LayerIntervention::ScaleResidual { layer, scale: s }) = intervention
                && layer == layer_idx
            {
                for h_val in &mut hidden {
                    *h_val *= s;
                }
            }

            layer_hidden.push(hidden.clone());
        }

        let final_hidden = rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps);
        let logits = self.lm_head.matvec(&final_hidden)?;
        Ok(ForwardTrace {
            logits,
            layer_hidden,
        })
    }

    /// Generate `max_new_tokens` autoregressively using greedy (`temperature <= 0.0`) or sampled decoding.
    pub fn generate(
        &self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<GenerationResult, RuntimeError> {
        self.generate_with_callback(
            prompt_tokens,
            max_new_tokens,
            temperature,
            None,
            seed,
            |_| true,
        )
    }

    /// Generate up to `max_new_tokens` with an explicit [`KvCacheMode`] (`Fp64`/`Fp32` or `Int8` GPU/CPU KV).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_kv_mode<F>(
        &self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        temperature: f64,
        eos_token: Option<usize>,
        kv_mode: KvCacheMode,
        seed: u64,
        mut on_token: F,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(usize) -> bool,
    {
        if prompt_tokens.is_empty() {
            return Err(RuntimeError::EmptyPrompt);
        }
        let start = Instant::now();
        let mut cache = KvCache::with_mode(self.config.num_layers, kv_mode);
        let mut rng = DeterministicRng::new(seed);
        let mut logits = Vec::new();

        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            logits = self.forward_token(tok, pos, &mut cache)?;
        }

        let mut generated = Vec::with_capacity(max_new_tokens);
        for step in 0..max_new_tokens {
            let next_tok = sample_token(&logits, temperature, &mut rng);
            generated.push(next_tok);
            let keep_going = on_token(next_tok);
            if !keep_going || eos_token == Some(next_tok) {
                break;
            }
            let pos = prompt_tokens.len() + step;
            if pos + 1 < self.config.max_seq_len && step + 1 < max_new_tokens {
                logits = self.forward_token(next_tok, pos, &mut cache)?;
            }
        }

        let elapsed_micros = start.elapsed().as_micros() as u64;
        let tps = generation_rate(generated.len(), elapsed_micros);

        Ok(GenerationResult {
            prompt_tokens: prompt_tokens.to_vec(),
            generated_tokens: generated,
            elapsed_micros,
            tokens_per_second: tps,
        })
    }

    /// Generate up to `max_new_tokens`, stopping early on `eos_token` or when
    /// `on_token(token_id)` returns `false` (enabling live streaming & cancellation).
    pub fn generate_with_callback<F>(
        &self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        temperature: f64,
        eos_token: Option<usize>,
        seed: u64,
        on_token: F,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(usize) -> bool,
    {
        self.generate_with_kv_mode(
            prompt_tokens,
            max_new_tokens,
            temperature,
            eos_token,
            KvCacheMode::Fp64,
            seed,
            on_token,
        )
    }
}

/// Compare logits, KL divergence, top-1 token agreement, and perplexity between two models.
pub fn evaluate_inference_fidelity(
    reference: &QCoreModel,
    transformed: &QCoreModel,
    token_sequence: &[usize],
) -> Result<InferenceComparison, RuntimeError> {
    if token_sequence.len() < 2 {
        return Err(RuntimeError::EmptyPrompt);
    }
    let mut ref_cache = KvCache::new(reference.config.num_layers);
    let mut tr_cache = KvCache::new(transformed.config.num_layers);

    let mut ref_nll = 0.0_f64;
    let mut tr_nll = 0.0_f64;
    let mut sum_kl = 0.0_f64;
    let mut max_kl = 0.0_f64;
    let mut sum_rmse = 0.0_f64;
    let mut top1_matches = 0_usize;
    let steps = token_sequence.len() - 1;

    for pos in 0..steps {
        let tok = token_sequence[pos];
        let target = token_sequence[pos + 1];
        let ref_logits = reference.forward_token(tok, pos, &mut ref_cache)?;
        let tr_logits = transformed.forward_token(tok, pos, &mut tr_cache)?;

        let ref_probs = softmax(&ref_logits);
        let tr_probs = softmax(&tr_logits);

        ref_nll -= ref_probs[target].max(1e-30).ln();
        tr_nll -= tr_probs[target].max(1e-30).ln();

        let kl: f64 = ref_probs
            .iter()
            .zip(&tr_probs)
            .map(|(&p, &q)| {
                if p > 1e-30 {
                    p * (p / q.max(1e-30)).ln()
                } else {
                    0.0
                }
            })
            .sum();
        sum_kl += kl;
        if kl > max_kl {
            max_kl = kl;
        }

        let mse = ref_logits
            .iter()
            .zip(&tr_logits)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / ref_logits.len() as f64;
        sum_rmse += mse.sqrt();

        if argmax(&ref_logits) == argmax(&tr_logits) {
            top1_matches += 1;
        }
    }

    let ref_ppl = (ref_nll / steps as f64).exp();
    let tr_ppl = (tr_nll / steps as f64).exp();

    Ok(InferenceComparison {
        sequence_len: token_sequence.len(),
        reference_perplexity: ref_ppl,
        transformed_perplexity: tr_ppl,
        perplexity_delta: tr_ppl - ref_ppl,
        mean_kl_divergence: sum_kl / steps as f64,
        max_kl_divergence: max_kl,
        mean_logit_rmse: sum_rmse / steps as f64,
        top1_token_agreement: top1_matches as f64 / steps as f64,
    })
}

fn rms_norm(x: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f64>() / x.len().max(1) as f64;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(weight)
        .map(|(&xi, &wi)| xi * inv_rms * wi)
        .collect()
}

fn apply_rope(head: &mut [f64], pos: usize, theta: f64) {
    let half = head.len() / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf((2 * i) as f64 / head.len() as f64);
        let angle = (pos as f64) * freq;
        let (sin_a, cos_a) = angle.sin_cos();
        let x0 = head[i];
        let x1 = head[i + half];
        head[i] = x0 * cos_a - x1 * sin_a;
        head[i + half] = x0 * sin_a + x1 * cos_a;
    }
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum::<f64>().max(1e-30);
    exps.into_iter().map(|e| e / sum).collect()
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map_or(0, |(idx, _)| idx)
}

fn sample_token(logits: &[f64], temperature: f64, rng: &mut DeterministicRng) -> usize {
    if temperature <= 1e-6 {
        return argmax(logits);
    }
    let scaled: Vec<f64> = logits.iter().map(|&v| v / temperature).collect();
    let probs = softmax(&scaled);
    let draw = rng.next_f64();
    let mut cdf = 0.0_f64;
    for (idx, &p) in probs.iter().enumerate() {
        cdf += p;
        if draw <= cdf {
            return idx;
        }
    }
    probs.len().saturating_sub(1)
}

/// Generate a complete, deterministic synthetic Llama-style Safetensors model for end-to-end
/// Q-Forge → QMF v1 → Q-Core inference & KL divergence validation.
pub fn generate_synthetic_llama_safetensors(
    config: &TransformerConfig,
    shared_rank: usize,
    seed: u64,
) -> Result<Vec<u8>, String> {
    config.validate().map_err(|e| e.to_string())?;
    let mut rng = DeterministicRng::new(seed);
    let h = config.hidden_size;
    let kv_dim = config.num_kv_heads * config.head_dim();
    let inter = config.intermediate_size;
    let vocab = config.vocab_size;

    let encode_f32 = |vals: &[f64]| -> Vec<u8> {
        let mut raw = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            raw.extend_from_slice(&(v as f32).to_le_bytes());
        }
        raw
    };

    let make_correlated_bank = |rng: &mut DeterministicRng,
                                rows: usize,
                                cols: usize,
                                count: usize,
                                r_in: usize|
     -> Vec<Vec<f64>> {
        let r = r_in.max(1).min(rows.min(cols));
        let u: Vec<f64> = (0..rows * r)
            .map(|_| rng.next_normal() / (rows as f64).sqrt())
            .collect();
        let vt: Vec<f64> = (0..r * cols)
            .map(|_| rng.next_normal() / (cols as f64).sqrt())
            .collect();
        let mut bank = Vec::with_capacity(count);
        for l in 0..count {
            let core: Vec<f64> = (0..r * r)
                .map(|_| rng.next_normal() * (0.4 + 0.05 * l as f64))
                .collect();
            let mut mat = vec![0.0_f64; rows * cols];
            for row in 0..rows {
                for col in 0..cols {
                    let mut sum = 0.0_f64;
                    for a in 0..r {
                        for b in 0..r {
                            sum += u[row * r + a] * core[a * r + b] * vt[b * cols + col];
                        }
                    }
                    mat[row * cols + col] = sum + rng.next_normal() * 0.015;
                }
            }
            bank.push(mat);
        }
        bank
    };

    let mut owned: Vec<(String, DType, Vec<u64>, Vec<u8>)> = Vec::new();

    // Embeddings and LM head
    let emb: Vec<f64> = (0..vocab * h).map(|_| rng.next_normal() * 0.15).collect();
    owned.push((
        "model.embed_tokens.weight".to_owned(),
        DType::F32,
        vec![vocab as u64, h as u64],
        encode_f32(&emb),
    ));

    let q_bank = make_correlated_bank(&mut rng, h, h, config.num_layers, shared_rank);
    let k_bank = make_correlated_bank(&mut rng, kv_dim, h, config.num_layers, shared_rank);
    let v_bank = make_correlated_bank(&mut rng, kv_dim, h, config.num_layers, shared_rank);
    let o_bank = make_correlated_bank(&mut rng, h, h, config.num_layers, shared_rank);
    let gate_bank = make_correlated_bank(&mut rng, inter, h, config.num_layers, shared_rank);
    let up_bank = make_correlated_bank(&mut rng, inter, h, config.num_layers, shared_rank);
    let down_bank = make_correlated_bank(&mut rng, h, inter, config.num_layers, shared_rank);

    for l in 0..config.num_layers {
        let norm1: Vec<f64> = (0..h).map(|_| 1.0 + rng.next_normal() * 0.02).collect();
        let norm2: Vec<f64> = (0..h).map(|_| 1.0 + rng.next_normal() * 0.02).collect();
        owned.push((
            format!("model.layers.{l}.input_layernorm.weight"),
            DType::F32,
            vec![h as u64],
            encode_f32(&norm1),
        ));
        owned.push((
            format!("model.layers.{l}.post_attention_layernorm.weight"),
            DType::F32,
            vec![h as u64],
            encode_f32(&norm2),
        ));
        owned.push((
            format!("model.layers.{l}.self_attn.q_proj.weight"),
            DType::F32,
            vec![h as u64, h as u64],
            encode_f32(&q_bank[l]),
        ));
        owned.push((
            format!("model.layers.{l}.self_attn.k_proj.weight"),
            DType::F32,
            vec![kv_dim as u64, h as u64],
            encode_f32(&k_bank[l]),
        ));
        owned.push((
            format!("model.layers.{l}.self_attn.v_proj.weight"),
            DType::F32,
            vec![kv_dim as u64, h as u64],
            encode_f32(&v_bank[l]),
        ));
        owned.push((
            format!("model.layers.{l}.self_attn.o_proj.weight"),
            DType::F32,
            vec![h as u64, h as u64],
            encode_f32(&o_bank[l]),
        ));
        owned.push((
            format!("model.layers.{l}.mlp.gate_proj.weight"),
            DType::F32,
            vec![inter as u64, h as u64],
            encode_f32(&gate_bank[l]),
        ));
        owned.push((
            format!("model.layers.{l}.mlp.up_proj.weight"),
            DType::F32,
            vec![inter as u64, h as u64],
            encode_f32(&up_bank[l]),
        ));
        owned.push((
            format!("model.layers.{l}.mlp.down_proj.weight"),
            DType::F32,
            vec![h as u64, inter as u64],
            encode_f32(&down_bank[l]),
        ));
    }

    let final_norm: Vec<f64> = (0..h).map(|_| 1.0 + rng.next_normal() * 0.02).collect();
    owned.push((
        "model.norm.weight".to_owned(),
        DType::F32,
        vec![h as u64],
        encode_f32(&final_norm),
    ));
    let lm_head: Vec<f64> = (0..vocab * h).map(|_| rng.next_normal() * 0.15).collect();
    owned.push((
        "lm_head.weight".to_owned(),
        DType::F32,
        vec![vocab as u64, h as u64],
        encode_f32(&lm_head),
    ));

    let refs: Vec<(&str, DType, &[u64], &[u8])> = owned
        .iter()
        .map(|(n, dt, sh, b)| (n.as_str(), *dt, sh.as_slice(), b.as_slice()))
        .collect();
    let mut meta = BTreeMap::new();
    meta.insert("architecture".to_owned(), "synthetic_llama".to_owned());
    meta.insert("seed".to_owned(), seed.to_string());
    encode_safetensors(&refs, &meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generation_rate_counts_only_emitted_tokens_over_total_time() {
        assert_eq!(generation_rate(20, 2_000_000), 10.0);
        assert_eq!(generation_rate(0, 2_000_000), 0.0);
        assert_eq!(generation_rate(20, 0), 0.0);
    }
    use qml_forge::{TransformStrategy, forge_safetensors_to_qmf};
    use qml_model::PayloadBudget;
    use qml_quant::FactorPrecision;
    use std::io::Cursor;

    #[test]
    fn qcore_executes_direct_compressed_inference_and_measures_kl_divergence() {
        let cfg = TransformerConfig {
            vocab_size: 32,
            hidden_size: 24,
            intermediate_size: 48,
            num_layers: 3,
            num_heads: 4,
            num_kv_heads: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_seq_len: 32,
            tie_word_embeddings: false,
        };
        let model_bytes = generate_synthetic_llama_safetensors(&cfg, 4, 2026).unwrap();
        let budget = PayloadBudget::default();

        // Reference FP16 QMF model
        let (ref_archive, _) = forge_safetensors_to_qmf(
            Cursor::new(&model_bytes),
            "mini-llama",
            "synthetic",
            "llama",
            &TransformStrategy::Fp16,
            &budget,
            42,
        )
        .unwrap();

        // Transformed RMQ Shared-Basis QMF model
        let (rmq_archive, rmq_report) = forge_safetensors_to_qmf(
            Cursor::new(&model_bytes),
            "mini-llama-rmq",
            "synthetic",
            "llama",
            &TransformStrategy::RmqSharedBasis {
                rank: 4,
                sparse_fraction: 0.01,
                basis_precision: FactorPrecision::Int8Sym,
                core_precision: FactorPrecision::Fp16,
            },
            &budget,
            42,
        )
        .unwrap();

        let ref_model = QCoreModel::from_qmf(&ref_archive, cfg.clone()).unwrap();
        let rmq_model = QCoreModel::from_qmf(&rmq_archive, cfg).unwrap();

        // Autoregressive generation is deterministic across identical runs.
        let gen1 = rmq_model.generate(&[1, 5, 12], 6, 0.0, 99).unwrap();
        let gen2 = rmq_model.generate(&[1, 5, 12], 6, 0.0, 99).unwrap();
        assert_eq!(gen1.generated_tokens, gen2.generated_tokens);
        assert_eq!(gen1.generated_tokens.len(), 6);
        assert_eq!(
            gen1.tokens_per_second,
            generation_rate(6, gen1.elapsed_micros)
        );

        // Evaluate logit KL divergence and perplexity fidelity directly on compressed RMQ execution!
        let eval =
            evaluate_inference_fidelity(&ref_model, &rmq_model, &[1, 5, 12, 3, 19, 8, 22, 4])
                .unwrap();
        assert!(eval.mean_kl_divergence.is_finite());
        assert!(
            eval.mean_kl_divergence < 0.15,
            "expected small KL divergence on correlated mini-llama, got {:.5}",
            eval.mean_kl_divergence
        );
        assert!(rmq_report.shared_latent_count >= 14);
        assert!(rmq_report.compression_ratio_vs_source > 1.5);

        // Verify GPU Hybrid backend matches CPU inference logits & generated tokens on RMQ!
        let rmq_gpu = QCoreModel::from_qmf_with_backend(
            &rmq_archive,
            rmq_model.config().clone(),
            QCoreBackend::GpuHybrid,
        )
        .unwrap();
        let gpu_gen = rmq_gpu.generate(&[1, 5, 12], 6, 0.0, 99).unwrap();
        assert_eq!(gen1.generated_tokens, gpu_gen.generated_tokens);
        let gpu_vs_cpu =
            evaluate_inference_fidelity(&rmq_model, &rmq_gpu, &[1, 5, 12, 3, 19, 8]).unwrap();
        assert!(
            gpu_vs_cpu.mean_logit_rmse < 1e-3,
            "GPU Hybrid Q-Core diverged from CPU: logit_rmse={:.6}",
            gpu_vs_cpu.mean_logit_rmse
        );
        if rmq_gpu.backend_stats().gpu_adapter.is_some() {
            assert!(rmq_gpu.backend_stats().gpu_resident_operators > 0);
            assert!(rmq_gpu.backend_stats().shared_latents_deduped_gpu > 0);
            assert!(rmq_gpu.backend_stats().vram_weight_bytes > 0);
        }
    }

    #[test]
    fn gpu_full_single_submission_and_int8_kv_and_calibration_match_cpu_reference() {
        let cfg = TransformerConfig {
            vocab_size: 48,
            hidden_size: 64,
            intermediate_size: 128,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_seq_len: 32,
            tie_word_embeddings: false,
        };
        let model_bytes = generate_synthetic_llama_safetensors(&cfg, 8, 9001).unwrap();
        let budget = PayloadBudget::default();

        // Forge Hadamard INT3 model (tests GPU Hadamard FWHT + GPU INT3 + GPU RMSNorm + GPU RoPE + GPU GQA + GPU SwiGLU!)
        let (int3_archive, _) = forge_safetensors_to_qmf(
            Cursor::new(&model_bytes),
            "mini-llama-int3-had",
            "synthetic",
            "llama",
            &TransformStrategy::Int3Hadamard {
                group_size: 64,
                hadamard_block: 64,
            },
            &budget,
            77,
        )
        .unwrap();

        let cpu_model = QCoreModel::from_qmf(&int3_archive, cfg.clone()).unwrap();
        let full_gpu_model =
            QCoreModel::from_qmf_with_backend(&int3_archive, cfg.clone(), QCoreBackend::GpuFull)
                .unwrap();

        // 1. Verify Full GPU-Resident single-submission matches CPU logits and greedy generation!
        let seq = [1, 7, 13, 21, 5, 33, 9];
        let cmp = evaluate_inference_fidelity(&cpu_model, &full_gpu_model, &seq).unwrap();
        assert!(
            cmp.mean_logit_rmse < 1e-3,
            "GpuFull diverged from CPU on Hadamard INT3 transformer: logit_rmse={:.6}",
            cmp.mean_logit_rmse
        );
        assert_eq!(
            cpu_model
                .generate(&[1, 7, 13], 5, 0.0, 42)
                .unwrap()
                .generated_tokens,
            full_gpu_model
                .generate(&[1, 7, 13], 5, 0.0, 42)
                .unwrap()
                .generated_tokens
        );
        if full_gpu_model.backend_stats().gpu_adapter.is_some() {
            assert!(
                full_gpu_model
                    .backend_stats()
                    .backend
                    .contains("Full-Resident"),
                "expected WGPU Full-Resident backend, got {}",
                full_gpu_model.backend_stats().backend
            );
        }

        // 2. Verify GPU-resident INT8 KV-cache (and CPU INT8 KV-cache) match FP32/FP64 KV tightly!
        let mut fp64_kv = KvCache::new(cfg.num_layers);
        let mut int8_kv = KvCache::new_int8(cfg.num_layers);
        let mut max_kv_diff = 0.0_f64;
        for (pos, &tok) in seq.iter().enumerate() {
            let l_fp = cpu_model.forward_token(tok, pos, &mut fp64_kv).unwrap();
            let l_i8 = cpu_model.forward_token(tok, pos, &mut int8_kv).unwrap();
            for (&a, &b) in l_fp.iter().zip(&l_i8) {
                max_kv_diff = max_kv_diff.max((a - b).abs());
            }
        }
        assert!(
            max_kv_diff < 5e-2,
            "CPU INT8 KV-cache drifted too far from FP64: max_diff={max_kv_diff:.6}"
        );

        let mut gpu_fp_kv = KvCache::new(cfg.num_layers);
        let mut gpu_i8_kv = KvCache::new_int8(cfg.num_layers);
        let mut max_gpu_kv_diff = 0.0_f64;
        for (pos, &tok) in seq.iter().enumerate() {
            let l_fp = full_gpu_model
                .forward_token(tok, pos, &mut gpu_fp_kv)
                .unwrap();
            let l_i8 = full_gpu_model
                .forward_token(tok, pos, &mut gpu_i8_kv)
                .unwrap();
            for (&a, &b) in l_fp.iter().zip(&l_i8) {
                max_gpu_kv_diff = max_gpu_kv_diff.max((a - b).abs());
            }
        }
        assert!(
            max_gpu_kv_diff < 5e-2,
            "GpuFull INT8 KV-cache drifted too far from FP32 KV: max_diff={max_gpu_kv_diff:.6}"
        );
        if let (Some(gfp), Some(gi8)) = (&gpu_fp_kv.gpu_kv, &gpu_i8_kv.gpu_kv) {
            assert_eq!(gi8.mode(), GpuKvMode::Int8);
            assert!(
                gi8.vram_bytes() < gfp.vram_bytes() / 2,
                "expected GPU INT8 KV VRAM ({}) < half FP32 KV VRAM ({})",
                gi8.vram_bytes(),
                gfp.vram_bytes()
            );
        }

        // 3. Verify calibration profile collection covers all layers and lm_head
        let cal = cpu_model
            .collect_calibration_profile("mini-llama", &[seq.to_vec()])
            .unwrap();
        assert_eq!(cal.calibration_tokens, seq.len());
        assert!(cal.get("model.layers.0.self_attn.q_proj.weight").is_some());
        assert!(cal.get("model.layers.1.mlp.down_proj.weight").is_some());
        assert!(cal.get("lm_head.weight").is_some());

        // 4. Verify Layer-by-Layer Streaming BF16 Calibration & Reference Evaluator matches full model!
        let tmp_st = std::env::temp_dir().join("qml_test_streaming_calib.safetensors");
        std::fs::write(&tmp_st, &model_bytes).unwrap();
        let (bf16_archive, _) = forge_safetensors_to_qmf(
            Cursor::new(&model_bytes),
            "mini-llama-bf16",
            "synthetic",
            "llama",
            &TransformStrategy::Bf16,
            &budget,
            77,
        )
        .unwrap();
        let bf16_model = QCoreModel::from_qmf(&bf16_archive, cfg.clone()).unwrap();
        let mut ref_kv = KvCache::new(cfg.num_layers);
        let mut expected_logits = Vec::new();
        for (pos, &tok) in seq.iter().take(seq.len() - 1).enumerate() {
            expected_logits.push(bf16_model.forward_token(tok, pos, &mut ref_kv).unwrap());
        }
        let streamed = stream_calibrate_and_reference(
            &tmp_st,
            "mini-llama-stream",
            &cfg,
            &[seq.to_vec()],
            &[seq.to_vec()],
            DEFAULT_HESSIAN_BLOCK_SIZE,
            true,
            &budget,
            |_, _, _| {},
        )
        .unwrap();
        let _ = std::fs::remove_file(&tmp_st);
        assert_eq!(streamed.calibration.calibration_tokens, seq.len());
        assert!(
            streamed
                .calibration
                .get("model.layers.0.self_attn.q_proj.weight")
                .is_some()
        );
        assert_eq!(streamed.eval_logits[0].len(), expected_logits.len());
        let mut max_stream_diff = 0.0_f64;
        for (a_vec, b_vec) in expected_logits.iter().zip(&streamed.eval_logits[0]) {
            for (&a, &b) in a_vec.iter().zip(b_vec) {
                max_stream_diff = max_stream_diff.max((a - b).abs());
            }
        }
        assert!(
            max_stream_diff < 5e-2,
            "Streaming BF16 evaluator diverged from full BF16 QCoreModel: max_diff={max_stream_diff:.6}"
        );
    }
}

#[cfg(test)]
mod aegis_projection_tests {
    use super::*;
    use std::io::Cursor;
    #[test]
    fn untied_missing_projection_errors_in_memory_reader_and_streaming() {
        let cfg = TransformerConfig {
            vocab_size: 8,
            hidden_size: 8,
            intermediate_size: 16,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 8,
            tie_word_embeddings: false,
        };
        let mut bytes = generate_synthetic_llama_safetensors(&cfg, 2, 7).unwrap();
        let pos = bytes
            .windows(b"lm_head.weight".len())
            .position(|b| b == b"lm_head.weight")
            .unwrap();
        bytes[pos..pos + 14].copy_from_slice(b"no_head.weight");
        let (archive, _) = qml_forge::forge_safetensors_to_qmf(
            Cursor::new(&bytes),
            "test",
            "synthetic",
            "llama",
            &qml_forge::TransformStrategy::Bf16,
            &qml_model::PayloadBudget::default(),
            42,
        )
        .unwrap();
        assert!(QCoreModel::from_qmf(&archive, cfg.clone()).is_err());
        let mut encoded = Vec::new();
        archive.write_to(&mut encoded).unwrap();
        let mut reader =
            QmfFileReader::open(Cursor::new(encoded), &qml_format::v1::QmfBudget::default())
                .unwrap();
        assert!(QCoreModel::from_qmf_reader(&mut reader, cfg.clone()).is_err());
        // Corrupt just the head blob; lazy loading must propagate its CRC error.
        let valid_bytes = generate_synthetic_llama_safetensors(&cfg, 2, 7).unwrap();
        let (valid, _) = qml_forge::forge_safetensors_to_qmf(
            Cursor::new(&valid_bytes),
            "test",
            "synthetic",
            "llama",
            &qml_forge::TransformStrategy::Bf16,
            &qml_model::PayloadBudget::default(),
            42,
        )
        .unwrap();
        let mut encoded = Vec::new();
        valid.write_to(&mut encoded).unwrap();
        let manifest_len = u64::from_le_bytes(encoded[8..16].try_into().unwrap()) as usize;
        let head = valid
            .manifest()
            .tensors
            .iter()
            .find(|t| t.name == "lm_head.weight")
            .unwrap();
        if let qml_format::v1::QmfRepresentation::DenseBf16 { data } = &head.representation {
            encoded[32 + manifest_len + data.offset as usize] ^= 1;
        } else {
            panic!("expected BF16 head");
        }
        let mut bad_reader =
            QmfFileReader::open(Cursor::new(encoded), &qml_format::v1::QmfBudget::default())
                .unwrap();
        assert!(QCoreModel::from_qmf_reader(&mut bad_reader, cfg.clone()).is_err());
        let mut tied = cfg.clone();
        tied.tie_word_embeddings = true;
        assert!(QCoreModel::from_qmf(&archive, tied).is_ok());
        let temp = qml_core::publication::stage(std::path::Path::new("test-model")).unwrap();
        std::fs::write(temp.path(), bytes).unwrap();
        let err = stream_calibrate_and_reference(
            temp.path(),
            "test",
            &cfg,
            &[],
            &[vec![0, 1]],
            2,
            false,
            &qml_model::PayloadBudget::default(),
            |_, _, _| {},
        )
        .err()
        .unwrap();
        assert!(err.to_string().contains("lm_head.weight"));
    }

    #[test]
    fn streaming_rejects_nonfinite_output_projection() {
        let cfg = TransformerConfig {
            vocab_size: 8,
            hidden_size: 8,
            intermediate_size: 16,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 8,
            tie_word_embeddings: false,
        };
        let mut bytes = generate_synthetic_llama_safetensors(&cfg, 2, 7).unwrap();
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
        assert_eq!(header["lm_head.weight"]["dtype"], "F32");
        let offset = header["lm_head.weight"]["data_offsets"][0]
            .as_u64()
            .unwrap() as usize;
        let start = 8 + header_len + offset;
        bytes[start..start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let temp =
            qml_core::publication::stage(std::path::Path::new("test-nonfinite-model")).unwrap();
        std::fs::write(temp.path(), bytes).unwrap();
        let error = stream_calibrate_and_reference(
            temp.path(),
            "test",
            &cfg,
            &[],
            &[vec![0, 1]],
            2,
            false,
            &qml_model::PayloadBudget::default(),
            |_, _, _| {},
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("non-finite"));
    }
}
