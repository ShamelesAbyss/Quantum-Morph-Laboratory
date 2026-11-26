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

use qml_core::{DeterministicRng, MatrixError};
use qml_format::{QmfArchiveV1, QmfError, QmfFileReader, QmfLatentCache, QmfOperator};
use qml_hardware::{PersistentGpuMatrix, WgpuMatvecEngine};
use qml_model::{DType, encode_safetensors};
use qml_quant::{EchoBaseOperator, LinearOperator, QuantizedFactor, SparseResidual};
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
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Parse a standard Hugging Face `config.json` into a `TransformerConfig`.
    pub fn from_hf_config_bytes(bytes: &[u8]) -> Result<Self, RuntimeError> {
        #[derive(Deserialize)]
        struct HfLlamaConfig {
            vocab_size: usize,
            hidden_size: usize,
            intermediate_size: usize,
            num_hidden_layers: usize,
            num_attention_heads: usize,
            #[serde(default)]
            num_key_value_heads: Option<usize>,
            #[serde(default)]
            rms_norm_eps: Option<f64>,
            #[serde(default)]
            rope_theta: Option<f64>,
            #[serde(default)]
            max_position_embeddings: Option<usize>,
            #[serde(default)]
            tie_word_embeddings: bool,
        }
        let raw: HfLlamaConfig = serde_json::from_slice(bytes)
            .map_err(|e| RuntimeError::InvalidConfig(format!("config.json parse error: {e}")))?;
        let cfg = Self {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_layers: raw.num_hidden_layers,
            num_heads: raw.num_attention_heads,
            num_kv_heads: raw.num_key_value_heads.unwrap_or(raw.num_attention_heads),
            rms_norm_eps: raw.rms_norm_eps.unwrap_or(1e-5),
            rope_theta: raw.rope_theta.unwrap_or(10_000.0),
            max_seq_len: raw.max_position_embeddings.unwrap_or(2048),
            tie_word_embeddings: raw.tie_word_embeddings,
        };
        cfg.validate()?;
        Ok(cfg)
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

/// Incremental key-value cache for autoregressive decoding.
#[derive(Debug, Clone)]
pub struct KvCache {
    keys: Vec<Vec<Vec<f64>>>,   // [layer][pos][kv_dim]
    values: Vec<Vec<Vec<f64>>>, // [layer][pos][kv_dim]
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            keys: (0..num_layers).map(|_| Vec::new()).collect(),
            values: (0..num_layers).map(|_| Vec::new()).collect(),
        }
    }

    pub fn seq_len(&self) -> usize {
        self.keys.first().map_or(0, Vec::len)
    }
}

/// Execution backend selection for Q-Core inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum QCoreBackend {
    /// Multi-threaded Rayon CPU execution over compressed QMF operators.
    #[default]
    Cpu,
    /// Persistent GPU-resident compressed operators via WGPU compute shaders
    /// (zero per-token weight transfer or buffer allocation), with automatic
    /// graceful fallback to CPU if no compatible GPU adapter is present.
    GpuHybrid,
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
    Single(PersistentGpuMatrix),
    LowRankSvd {
        left: PersistentGpuMatrix,
        right: PersistentGpuMatrix,
    },
    RmqSharedBasis {
        u_shared: Arc<PersistentGpuMatrix>,
        core: PersistentGpuMatrix,
        vt_shared: Arc<PersistentGpuMatrix>,
        sparse_residual: Option<SparseResidual>,
    },
    RmqSharedCore {
        left: PersistentGpuMatrix,
        shared_core: Arc<PersistentGpuMatrix>,
        right: PersistentGpuMatrix,
        sparse_residual: Option<SparseResidual>,
    },
    EchoCorrected {
        base: PersistentGpuMatrix,
        low_rank: Option<(PersistentGpuMatrix, PersistentGpuMatrix)>,
        sparse_residual: Option<SparseResidual>,
    },
}

impl GpuOperator {
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
                GpuOperator::Single(p)
            }),
            QmfOperator::DenseF16(m) => self.engine.upload_fp16(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(p)
            }),
            QmfOperator::DenseBf16(m) => self.engine.upload_bf16(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(p)
            }),
            QmfOperator::Int8Sym(m) => self.engine.upload_int8_sym(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(p)
            }),
            QmfOperator::Int4Sym(m) => self.engine.upload_int4_sym(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(p)
            }),
            QmfOperator::Int4Asym(m) => self.engine.upload_int4_asym(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(p)
            }),
            QmfOperator::LowRankSvd(svd) => {
                let left = self.engine.upload_quantized_factor(svd.left()).ok();
                let right = self.engine.upload_quantized_factor(svd.right()).ok();
                match (left, right) {
                    (Some(l), Some(r)) => {
                        self.vram_weight_bytes += l.vram_bytes() + r.vram_bytes();
                        Some(GpuOperator::LowRankSvd { left: l, right: r })
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
                            core: c,
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
                            left: l,
                            shared_core: s,
                            right: r,
                            sparse_residual: sparse_residual.clone(),
                        })
                    }
                    _ => None,
                }
            }
            QmfOperator::Int3Asym(m) => self.engine.upload_int3_asym(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(p)
            }),
            QmfOperator::Int2Asym(m) => self.engine.upload_int2_asym(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(p)
            }),
            QmfOperator::TernaryPacked(m) => self.engine.upload_ternary_packed(m).ok().map(|p| {
                self.vram_weight_bytes += p.vram_bytes();
                GpuOperator::Single(p)
            }),
            QmfOperator::VectorQuant(_) => {
                // LUT-accelerated CPU path executes zero-multiply codebook matvec directly.
                None
            }
            QmfOperator::EchoCorrected(echo) => {
                let base_gpu = match echo.base() {
                    EchoBaseOperator::Int3Asym(m) => self.engine.upload_int3_asym(m).ok(),
                    EchoBaseOperator::Int2Asym(m) => self.engine.upload_int2_asym(m).ok(),
                    EchoBaseOperator::Ternary(m) => self.engine.upload_ternary_packed(m).ok(),
                    EchoBaseOperator::VectorQuant(_) => None,
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
                                Some((l, r))
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    Some(GpuOperator::EchoCorrected {
                        base,
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
    k_proj: RuntimeOperator,
    v_proj: RuntimeOperator,
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
/// shared RMQ latent structures across layers via `Arc`, and optionally executes
/// persistent compressed weights directly in GPU VRAM via WGPU compute shaders.
pub struct QCoreModel {
    config: TransformerConfig,
    embed_tokens: Arc<RuntimeOperator>,
    layers: Vec<TransformerBlock>,
    final_norm: Vec<f64>,
    lm_head: Arc<RuntimeOperator>,
    shared_latents_deduped: usize,
    backend_stats: QCoreBackendStats,
}

/// Output of autoregressive generation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GenerationResult {
    pub prompt_tokens: Vec<usize>,
    pub generated_tokens: Vec<usize>,
    pub elapsed_micros: u64,
    pub tokens_per_second: f64,
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
            QCoreBackend::GpuHybrid => WgpuMatvecEngine::try_new()
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
        // Only upload embed_tokens to VRAM when it is also tied as lm_head (decode_row runs on CPU).
        let embed_op = if config.tie_word_embeddings {
            Arc::new(wrap_op(embed_raw))
        } else {
            Arc::new(RuntimeOperator::cpu_only(embed_raw))
        };

        let mut layers = Vec::with_capacity(config.num_layers);
        for l in 0..config.num_layers {
            let p = format!("model.layers.{l}");
            layers.push(TransformerBlock {
                input_norm: reader.load_vector_f64(&format!("{p}.input_layernorm.weight"))?,
                q_proj: wrap_op(reader.load_operator_cached(
                    &format!("{p}.self_attn.q_proj.weight"),
                    &mut latent_cache,
                )?),
                k_proj: wrap_op(reader.load_operator_cached(
                    &format!("{p}.self_attn.k_proj.weight"),
                    &mut latent_cache,
                )?),
                v_proj: wrap_op(reader.load_operator_cached(
                    &format!("{p}.self_attn.v_proj.weight"),
                    &mut latent_cache,
                )?),
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
            match reader.load_operator_cached("lm_head.weight", &mut latent_cache) {
                Ok(op) => Arc::new(wrap_op(op)),
                Err(_) => Arc::clone(&embed_op),
            }
        };

        let backend_stats = match gpu_ctx {
            Some(ctx) => QCoreBackendStats {
                backend: format!(
                    "WGPU Hybrid ({} / {})",
                    ctx.engine.adapter_name(),
                    ctx.engine.backend_name()
                ),
                gpu_adapter: Some(ctx.engine.adapter_name().to_owned()),
                gpu_resident_operators: ctx.gpu_resident_operators,
                cpu_fallback_operators: ctx.cpu_fallback_operators,
                vram_weight_bytes: ctx.vram_weight_bytes,
                shared_latents_deduped_gpu: ctx.latent_gpu_cache.len(),
            },
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
            shared_latents_deduped: latent_cache.len(),
            backend_stats,
        })
    }

    fn build_model<FOp, FVec>(
        config: TransformerConfig,
        backend: QCoreBackend,
        load_op: FOp,
        load_vec: FVec,
    ) -> Result<Self, RuntimeError>
    where
        FOp: Fn(&str, &mut QmfLatentCache) -> Result<QmfOperator, QmfError>,
        FVec: Fn(&str) -> Result<Vec<f64>, QmfError>,
    {
        config.validate()?;
        let mut latent_cache = QmfLatentCache::new();
        let mut gpu_ctx = match backend {
            QCoreBackend::Cpu => None,
            QCoreBackend::GpuHybrid => WgpuMatvecEngine::try_new()
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

        let mut layers = Vec::with_capacity(config.num_layers);
        for l in 0..config.num_layers {
            layers.push(TransformerBlock {
                input_norm: load_vec(&format!("model.layers.{l}.input_layernorm.weight"))?,
                q_proj: wrap_op(load_op(
                    &format!("model.layers.{l}.self_attn.q_proj.weight"),
                    &mut latent_cache,
                )?),
                k_proj: wrap_op(load_op(
                    &format!("model.layers.{l}.self_attn.k_proj.weight"),
                    &mut latent_cache,
                )?),
                v_proj: wrap_op(load_op(
                    &format!("model.layers.{l}.self_attn.v_proj.weight"),
                    &mut latent_cache,
                )?),
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
            match load_op("lm_head.weight", &mut latent_cache) {
                Ok(op) => Arc::new(wrap_op(op)),
                Err(_) => Arc::clone(&embed_op),
            }
        };

        let backend_stats = match gpu_ctx {
            Some(ctx) => QCoreBackendStats {
                backend: format!(
                    "WGPU Hybrid ({} / {})",
                    ctx.engine.adapter_name(),
                    ctx.engine.backend_name()
                ),
                gpu_adapter: Some(ctx.engine.adapter_name().to_owned()),
                gpu_resident_operators: ctx.gpu_resident_operators,
                cpu_fallback_operators: ctx.cpu_fallback_operators,
                vram_weight_bytes: ctx.vram_weight_bytes,
                shared_latents_deduped_gpu: ctx.latent_gpu_cache.len(),
            },
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
    pub fn forward_token(
        &self,
        token_id: usize,
        position: usize,
        cache: &mut KvCache,
    ) -> Result<Vec<f64>, RuntimeError> {
        Ok(self
            .forward_token_with_trace(token_id, position, cache, None)?
            .logits)
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

            // Direct compressed linear projections (no dense weight reconstruction!)
            let mut q = block.q_proj.matvec(&normed)?;
            let mut k = block.k_proj.matvec(&normed)?;
            let v = block.v_proj.matvec(&normed)?;

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

            cache.keys[layer_idx].push(k);
            cache.values[layer_idx].push(v);
            let seq_len = cache.keys[layer_idx].len();

            // Multi-head / Grouped-query causal attention
            let mut attn_out = vec![0.0_f64; self.config.hidden_size];
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

    /// Generate up to `max_new_tokens`, stopping early on `eos_token` or when
    /// `on_token(token_id)` returns `false` (enabling live streaming & cancellation).
    pub fn generate_with_callback<F>(
        &self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        temperature: f64,
        eos_token: Option<usize>,
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
        let mut cache = KvCache::new(self.config.num_layers);
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
        let total_tokens = prompt_tokens.len() + generated.len();
        let tps = if elapsed_micros > 0 {
            (total_tokens as f64) / (elapsed_micros as f64 / 1_000_000.0)
        } else {
            0.0
        };

        Ok(GenerationResult {
            prompt_tokens: prompt_tokens.to_vec(),
            generated_tokens: generated,
            elapsed_micros,
            tokens_per_second: tps,
        })
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
}
