//! Full GPU-Resident Transformer Compute Engine (`qml-hardware::gpu_full`).
//!
//! Executes an entire decoder-only transformer forward step — RMSNorm, optional
//! Qwen2 Q/K/V biases, RoPE, persistent VRAM KV-cache (FP32 or INT8-compressed),
//! Grouped-Query Attention, SwiGLU MLP, residual connections, GPU blockwise
//! Fast Walsh–Hadamard pre-rotation, and direct compressed matvecs — inside a
//! **SINGLE `wgpu::CommandEncoder` submission per token** with zero intermediate
//! CPU↔GPU readbacks.

use crate::{GpuComputeError, PersistentGpuMatrix, WgpuMatvecEngine};
use qml_quant::hadamard_signs;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsNormParams {
    len: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HadamardParams {
    len: u32,
    block_size: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StepParams {
    pos: u32,
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    rope_theta: f32,
    inv_sqrt_head_dim: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VecLenParams {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// WGSL RMSNorm: single workgroup computes exact mean-square in shared memory, then scales by weight.
const RMSNORM_WGSL: &str = r#"
struct Params {
    len: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_in: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> y_out: array<f32>;

var<workgroup> sh_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var local_sq: f32 = 0.0;
    var i = tid;
    while (i < params.len) {
        let v = x_in[i];
        local_sq = local_sq + v * v;
        i = i + 256u;
    }
    sh_sum[tid] = local_sq;
    workgroupBarrier();

    var stride: u32 = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            sh_sum[tid] = sh_sum[tid] + sh_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    let inv_rms = inverseSqrt(sh_sum[0] / f32(max(params.len, 1u)) + params.eps);
    var j = tid;
    while (j < params.len) {
        y_out[j] = x_in[j] * inv_rms * weight[j];
        j = j + 256u;
    }
}
"#;

/// WGSL blockwise Randomized Fast Walsh-Hadamard Transform (`block_size <= 256` power of 2).
/// Uses pre-uploaded bit-packed signs (`1 bit/elem`, 100% bit-identical to CPU `hadamard_signs`).
const HADAMARD_FWD_WGSL: &str = r#"
struct Params {
    len: u32,
    block_size: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_in: array<f32>;
@group(0) @binding(2) var<storage, read> signs_bits: array<u32>;
@group(0) @binding(3) var<storage, read_write> y_out: array<f32>;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let bs = params.block_size;
    let base = wid.x * bs;
    let tid = lid.x;
    if (tid >= bs) {
        return;
    }
    let idx = base + tid;
    if (base + bs > params.len) {
        // Tail block smaller than block_size: pass through unchanged (matches CPU apply_hadamard_forward).
        if (idx < params.len) {
            y_out[idx] = x_in[idx];
        }
        return;
    }
    let sign_word = signs_bits[idx >> 5u];
    let is_neg = ((sign_word >> (idx & 31u)) & 1u) != 0u;
    let s = select(1.0, -1.0, is_neg);
    sh[tid] = x_in[idx] * s;
    workgroupBarrier();

    var h: u32 = 1u;
    while (h < bs) {
        let partner = tid ^ h;
        let a = sh[tid];
        let b = sh[partner];
        workgroupBarrier();
        if ((tid & h) == 0u) {
            sh[tid] = a + b;
        } else {
            sh[tid] = b - a;
        }
        workgroupBarrier();
        h = h << 1u;
    }

    let inv_sqrt = inverseSqrt(f32(bs));
    y_out[idx] = sh[tid] * inv_sqrt;
}
"#;

/// WGSL in-place RoPE on Q and K heads + write K and V into persistent VRAM KV-cache at `pos`.
const ROPE_AND_KV_APPEND_WGSL: &str = r#"
struct StepParams {
    pos: u32,
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    rope_theta: f32,
    inv_sqrt_head_dim: f32,
};

@group(0) @binding(0) var<uniform> sp: StepParams;
@group(0) @binding(1) var<storage, read_write> q_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_buf: array<f32>;
@group(0) @binding(3) var<storage, read> v_buf: array<f32>;
@group(0) @binding(4) var<storage, read_write> kv_k: array<f32>;
@group(0) @binding(5) var<storage, read_write> kv_v: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = sp.head_dim >> 1u;
    let total_q_pairs = sp.num_heads * half;
    let total_k_pairs = sp.num_kv_heads * half;
    let idx = gid.x;

    // 1. Apply RoPE to Q
    if (idx < total_q_pairs) {
        let h = idx / half;
        let i = idx % half;
        let freq = 1.0 / pow(sp.rope_theta, f32(2u * i) / f32(sp.head_dim));
        let angle = f32(sp.pos) * freq;
        let c = cos(angle);
        let s = sin(angle);
        let off = h * sp.head_dim;
        let x0 = q_buf[off + i];
        let x1 = q_buf[off + i + half];
        q_buf[off + i] = x0 * c - x1 * s;
        q_buf[off + i + half] = x0 * s + x1 * c;
    }

    // 2. Apply RoPE to K and write both rotated K and V into persistent KV-cache slot `sp.pos`
    if (idx < total_k_pairs) {
        let kh = idx / half;
        let i = idx % half;
        let freq = 1.0 / pow(sp.rope_theta, f32(2u * i) / f32(sp.head_dim));
        let angle = f32(sp.pos) * freq;
        let c = cos(angle);
        let s = sin(angle);
        let off = kh * sp.head_dim;
        let x0 = k_buf[off + i];
        let x1 = k_buf[off + i + half];
        let r0 = x0 * c - x1 * s;
        let r1 = x0 * s + x1 * c;
        k_buf[off + i] = r0;
        k_buf[off + i + half] = r1;

        let cache_base = sp.pos * sp.kv_dim + off;
        kv_k[cache_base + i] = r0;
        kv_k[cache_base + i + half] = r1;
        kv_v[cache_base + i] = v_buf[off + i];
        kv_v[cache_base + i + half] = v_buf[off + i + half];
    }
}
"#;

/// WGSL Grouped-Query Attention (causal, numerically stable online softmax across arbitrary seq_len).
/// 1 workgroup per query head `h` (`@workgroup_size(64)`).
const GQA_ATTN_WGSL: &str = r#"
struct StepParams {
    pos: u32,
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    rope_theta: f32,
    inv_sqrt_head_dim: f32,
};

@group(0) @binding(0) var<uniform> sp: StepParams;
@group(0) @binding(1) var<storage, read> q_buf: array<f32>;
@group(0) @binding(2) var<storage, read> kv_k: array<f32>;
@group(0) @binding(3) var<storage, read> kv_v: array<f32>;
@group(0) @binding(4) var<storage, read_write> attn_out: array<f32>;

var<workgroup> sh_dot: array<f32, 64>;
var<workgroup> sh_prob: f32;
var<workgroup> sh_alpha: f32;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wid.x;
    if (h >= sp.num_heads) {
        return;
    }
    let tid = lid.x;
    let kv_groups = sp.num_heads / sp.num_kv_heads;
    let kv_h = h / kv_groups;
    let q_off = h * sp.head_dim;
    let kv_off = kv_h * sp.head_dim;

    // Each thread accumulates output elements `d = tid, tid + 64, ...` (up to head_dim <= 256 -> 4 regs).
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    var running_max: f32 = -1e30;
    var running_sum: f32 = 0.0;

    for (var t: u32 = 0u; t < sp.seq_len; t = t + 1u) {
        let k_base = t * sp.kv_dim + kv_off;
        var part: f32 = 0.0;
        var d = tid;
        while (d < sp.head_dim) {
            part = part + q_buf[q_off + d] * kv_k[k_base + d];
            d = d + 64u;
        }
        sh_dot[tid] = part;
        workgroupBarrier();

        var s: u32 = 32u;
        while (s > 0u) {
            if (tid < s) {
                sh_dot[tid] = sh_dot[tid] + sh_dot[tid + s];
            }
            workgroupBarrier();
            s = s >> 1u;
        }

        if (tid == 0u) {
            let score = sh_dot[0] * sp.inv_sqrt_head_dim;
            let new_max = max(running_max, score);
            let alpha = exp(running_max - new_max);
            let p = exp(score - new_max);
            running_sum = running_sum * alpha + p;
            running_max = new_max;
            sh_alpha = alpha;
            sh_prob = p;
        }
        workgroupBarrier();

        let alpha = sh_alpha;
        let p = sh_prob;
        let v_base = t * sp.kv_dim + kv_off;
        if (tid < sp.head_dim) {
            acc0 = acc0 * alpha + p * kv_v[v_base + tid];
        }
        if (tid + 64u < sp.head_dim) {
            acc1 = acc1 * alpha + p * kv_v[v_base + tid + 64u];
        }
        if (tid + 128u < sp.head_dim) {
            acc2 = acc2 * alpha + p * kv_v[v_base + tid + 128u];
        }
        if (tid + 192u < sp.head_dim) {
            acc3 = acc3 * alpha + p * kv_v[v_base + tid + 192u];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        sh_alpha = 1.0 / max(running_sum, 1e-30);
    }
    workgroupBarrier();
    let inv_sum = sh_alpha;

    if (tid < sp.head_dim) {
        attn_out[q_off + tid] = acc0 * inv_sum;
    }
    if (tid + 64u < sp.head_dim) {
        attn_out[q_off + tid + 64u] = acc1 * inv_sum;
    }
    if (tid + 128u < sp.head_dim) {
        attn_out[q_off + tid + 128u] = acc2 * inv_sum;
    }
    if (tid + 192u < sp.head_dim) {
        attn_out[q_off + tid + 192u] = acc3 * inv_sum;
    }
}
"#;

/// WGSL SwiGLU activation: `gate[i] = silu(gate[i]) * up[i]`.
const SWIGLU_WGSL: &str = r#"
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) {
        return;
    }
    let g = gate[i];
    gate[i] = (g / (1.0 + exp(-g))) * up[i];
}
"#;

/// WGSL in-place vector addition: `dst[i] = dst[i] + src[i]` (residuals, biases, ECHO adapters).
const VEC_ADD_WGSL: &str = r#"
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) {
        return;
    }
    dst[i] = dst[i] + src[i];
}
"#;

/// Configuration for a Full GPU-Resident Transformer model.
#[derive(Debug, Clone)]
pub struct GpuFullConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_kv_len: usize,
}

/// GPU-resident linear projection supported inside a single-submission Full GPU forward pass.
pub enum GpuFullOperator {
    Single(Arc<PersistentGpuMatrix>),
    EchoLowRank {
        base: Arc<PersistentGpuMatrix>,
        left: Arc<PersistentGpuMatrix>,
        right: Arc<PersistentGpuMatrix>,
    },
}

/// Layer weights & optional biases handed to [`GpuFullModel::new`].
pub struct GpuFullLayerSpec {
    pub input_norm: Vec<f64>,
    pub q_proj: GpuFullOperator,
    pub q_bias: Option<Vec<f64>>,
    pub k_proj: GpuFullOperator,
    pub k_bias: Option<Vec<f64>>,
    pub v_proj: GpuFullOperator,
    pub v_bias: Option<Vec<f64>>,
    pub o_proj: GpuFullOperator,
    pub post_attn_norm: Vec<f64>,
    pub gate_proj: GpuFullOperator,
    pub up_proj: GpuFullOperator,
    pub down_proj: GpuFullOperator,
}

/// Persistent GPU-resident KV-cache (`[layer][max_kv_len * kv_dim]` f32s in VRAM).
pub struct GpuKvCache {
    max_kv_len: usize,
    seq_len: usize,
    kv_bytes: u64,
    _k_bufs: Vec<wgpu::Buffer>,
    _v_bufs: Vec<wgpu::Buffer>,
    rope_kv_bgs: Vec<wgpu::BindGroup>,
    attn_bgs: Vec<wgpu::BindGroup>,
}

impl GpuKvCache {
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn max_kv_len(&self) -> usize {
        self.max_kv_len
    }

    pub fn vram_bytes(&self) -> u64 {
        self.kv_bytes
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
    }
}

struct BoundMatvec {
    matrix: Arc<PersistentGpuMatrix>,
    hadamard_bg: Option<(wgpu::BindGroup, u32)>,
    matvec_bg: wgpu::BindGroup,
    _extra_bufs: Vec<wgpu::Buffer>,
}

struct BoundOperator {
    primary: BoundMatvec,
    echo_right: Option<BoundMatvec>,
    echo_left: Option<BoundMatvec>,
    echo_add_bg: Option<(wgpu::BindGroup, u32)>,
}

struct BoundLayer {
    input_norm_bg: wgpu::BindGroup,
    q_proj: BoundOperator,
    q_bias_add: Option<(wgpu::BindGroup, u32)>,
    k_proj: BoundOperator,
    k_bias_add: Option<(wgpu::BindGroup, u32)>,
    v_proj: BoundOperator,
    v_bias_add: Option<(wgpu::BindGroup, u32)>,
    o_proj: BoundOperator,
    post_attn_norm_bg: wgpu::BindGroup,
    gate_proj: BoundOperator,
    up_proj: BoundOperator,
    down_proj: BoundOperator,
    _owned_bufs: Vec<wgpu::Buffer>,
}

/// Complete Full GPU-Resident Transformer runtime executing all layers + `lm_head` in a
/// single `CommandEncoder` submission per token with zero intermediate CPU readbacks.
pub struct GpuFullModel {
    engine: Arc<WgpuMatvecEngine>,
    config: GpuFullConfig,
    pipeline_rmsnorm: wgpu::ComputePipeline,
    pipeline_hadamard: wgpu::ComputePipeline,
    pipeline_rope_kv: wgpu::ComputePipeline,
    pipeline_gqa_attn: wgpu::ComputePipeline,
    pipeline_swiglu: wgpu::ComputePipeline,
    pipeline_vec_add: wgpu::ComputePipeline,
    bgl_rope_kv: wgpu::BindGroupLayout,
    bgl_gqa_attn: wgpu::BindGroupLayout,
    step_params_buf: wgpu::Buffer,
    hidden_buf: wgpu::Buffer,
    q_buf: wgpu::Buffer,
    k_buf: wgpu::Buffer,
    v_buf: wgpu::Buffer,
    attn_out_buf: wgpu::Buffer,
    logits_buf: wgpu::Buffer,
    logits_readback_buf: wgpu::Buffer,
    attn_res_add_bg: wgpu::BindGroup,
    mlp_res_add_bg: wgpu::BindGroup,
    swiglu_bg: wgpu::BindGroup,
    final_norm_bg: wgpu::BindGroup,
    lm_head: BoundOperator,
    layers: Vec<BoundLayer>,
    _shared_bufs: Vec<wgpu::Buffer>,
    vram_runtime_bytes: u64,
}

fn pack_hadamard_signs_bits(cols: usize, seed: u64) -> Vec<u32> {
    let signs = hadamard_signs(cols, seed);
    let words = cols.div_ceil(32).max(1);
    let mut out = vec![0u32; words];
    for (i, &s) in signs.iter().enumerate() {
        if s < 0.0 {
            out[i >> 5] |= 1u32 << (i & 31);
        }
    }
    out
}

impl GpuFullModel {
    pub fn new(
        engine: Arc<WgpuMatvecEngine>,
        config: GpuFullConfig,
        layer_specs: Vec<GpuFullLayerSpec>,
        final_norm: Vec<f64>,
        lm_head: GpuFullOperator,
    ) -> Result<Self, GpuComputeError> {
        let device = &engine.device;
        let head_dim = config.hidden_size / config.num_heads;
        if head_dim > 256 {
            return Err(GpuComputeError::DimensionExceedsScratch {
                dim: head_dim,
                max_dim: 256,
            });
        }
        let kv_dim = config.num_kv_heads * head_dim;
        let max_inner = config.hidden_size.max(config.intermediate_size);

        let mk_shader = |label, src: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let sh_norm = mk_shader("qml-full-rmsnorm", RMSNORM_WGSL);
        let sh_had = mk_shader("qml-full-hadamard", HADAMARD_FWD_WGSL);
        let sh_rope = mk_shader("qml-full-rope-kv", ROPE_AND_KV_APPEND_WGSL);
        let sh_attn = mk_shader("qml-full-gqa-attn", GQA_ATTN_WGSL);
        let sh_swiglu = mk_shader("qml-full-swiglu", SWIGLU_WGSL);
        let sh_add = mk_shader("qml-full-vec-add", VEC_ADD_WGSL);

        let u_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let ro = |b| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let rw = |b| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl_norm = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("full-norm-bgl"),
            entries: &[u_entry, ro(1), ro(2), rw(3)],
        });
        let bgl_had = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("full-had-bgl"),
            entries: &[u_entry, ro(1), ro(2), rw(3)],
        });
        let bgl_rope_kv = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("full-rope-kv-bgl"),
            entries: &[u_entry, rw(1), rw(2), ro(3), rw(4), rw(5)],
        });
        let bgl_gqa_attn = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("full-gqa-attn-bgl"),
            entries: &[u_entry, ro(1), ro(2), ro(3), rw(4)],
        });
        let bgl_2vec = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("full-2vec-bgl"),
            entries: &[u_entry, rw(1), ro(2)],
        });

        let mk_pipe = |label, bgl: &wgpu::BindGroupLayout, sh: &wgpu::ShaderModule| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: sh,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let pipeline_rmsnorm = mk_pipe("full-rmsnorm-pipe", &bgl_norm, &sh_norm);
        let pipeline_hadamard = mk_pipe("full-had-pipe", &bgl_had, &sh_had);
        let pipeline_rope_kv = mk_pipe("full-rope-kv-pipe", &bgl_rope_kv, &sh_rope);
        let pipeline_gqa_attn = mk_pipe("full-gqa-attn-pipe", &bgl_gqa_attn, &sh_attn);
        let pipeline_swiglu = mk_pipe("full-swiglu-pipe", &bgl_2vec, &sh_swiglu);
        let pipeline_vec_add = mk_pipe("full-vec-add-pipe", &bgl_2vec, &sh_add);

        let f32_sz = std::mem::size_of::<f32>() as u64;
        let mk_buf = |label, elems: usize, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (elems.max(1) as u64) * f32_sz,
                usage: wgpu::BufferUsages::STORAGE | extra,
                mapped_at_creation: false,
            })
        };

        let hidden_buf = mk_buf(
            "full-hidden",
            config.hidden_size,
            wgpu::BufferUsages::COPY_DST,
        );
        let norm_buf = mk_buf("full-norm", config.hidden_size, wgpu::BufferUsages::empty());
        let rot_buf = mk_buf("full-rot", max_inner, wgpu::BufferUsages::empty());
        let q_buf = mk_buf("full-q", config.hidden_size, wgpu::BufferUsages::empty());
        let k_buf = mk_buf("full-k", kv_dim, wgpu::BufferUsages::empty());
        let v_buf = mk_buf("full-v", kv_dim, wgpu::BufferUsages::empty());
        let attn_out_buf = mk_buf(
            "full-attn-out",
            config.hidden_size,
            wgpu::BufferUsages::empty(),
        );
        let proj_tmp_buf = mk_buf(
            "full-proj-tmp",
            config.hidden_size,
            wgpu::BufferUsages::empty(),
        );
        let gate_buf = mk_buf(
            "full-gate",
            config.intermediate_size,
            wgpu::BufferUsages::empty(),
        );
        let up_buf = mk_buf(
            "full-up",
            config.intermediate_size,
            wgpu::BufferUsages::empty(),
        );
        let echo_mid_buf = mk_buf("full-echo-mid", 256, wgpu::BufferUsages::empty());
        let echo_delta_buf = mk_buf(
            "full-echo-delta",
            max_inner.max(config.vocab_size),
            wgpu::BufferUsages::empty(),
        );
        let logits_buf = mk_buf(
            "full-logits",
            config.vocab_size,
            wgpu::BufferUsages::COPY_SRC,
        );
        let logits_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("full-logits-readback"),
            size: (config.vocab_size.max(1) as u64) * f32_sz,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let step_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("full-step-params"),
            size: std::mem::size_of::<StepParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let norm_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("full-norm-params"),
            contents: bytemuck::bytes_of(&RmsNormParams {
                len: config.hidden_size as u32,
                eps: config.rms_norm_eps as f32,
                _pad0: 0,
                _pad1: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let hidden_len_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("full-hidden-len"),
            contents: bytemuck::bytes_of(&VecLenParams {
                len: config.hidden_size as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let kv_len_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("full-kv-len"),
            contents: bytemuck::bytes_of(&VecLenParams {
                len: kv_dim as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let inter_len_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("full-inter-len"),
            contents: bytemuck::bytes_of(&VecLenParams {
                len: config.intermediate_size as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let mut vram_runtime_bytes: u64 = ((config.hidden_size * 5
            + kv_dim * 2
            + max_inner * 2
            + config.intermediate_size * 2
            + config.vocab_size * 2
            + 256) as u64)
            * f32_sz;

        let bind_matvec = |mat: Arc<PersistentGpuMatrix>,
                           in_buf: &wgpu::Buffer,
                           out_buf: &wgpu::Buffer|
         -> BoundMatvec {
            let had = mat.hadamard();
            let mut extra = Vec::new();
            let (hadamard_bg, effective_in) = if had.is_enabled() && had.block_size <= 256 {
                let hp = HadamardParams {
                    len: mat.cols() as u32,
                    block_size: had.block_size as u32,
                    _pad0: 0,
                    _pad1: 0,
                };
                let hp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("full-had-params"),
                    contents: bytemuck::bytes_of(&hp),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let sign_bits = pack_hadamard_signs_bits(mat.cols(), had.seed);
                let sb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("full-had-signs"),
                    contents: bytemuck::cast_slice(&sign_bits),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("full-had-bg"),
                    layout: &bgl_had,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: hp_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: in_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: sb_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: rot_buf.as_entire_binding(),
                        },
                    ],
                });
                let wg = (mat.cols() as u32).div_ceil(had.block_size as u32);
                extra.push(hp_buf);
                extra.push(sb_buf);
                (Some((bg, wg)), &rot_buf)
            } else {
                (None, in_buf)
            };
            let matvec_bg = mat.create_io_bind_group(effective_in, out_buf);
            BoundMatvec {
                matrix: mat,
                hadamard_bg,
                matvec_bg,
                _extra_bufs: extra,
            }
        };

        let bind_op = |op: GpuFullOperator,
                       in_buf: &wgpu::Buffer,
                       out_buf: &wgpu::Buffer,
                       out_len_buf: &wgpu::Buffer|
         -> BoundOperator {
            match op {
                GpuFullOperator::Single(m) => BoundOperator {
                    primary: bind_matvec(m, in_buf, out_buf),
                    echo_right: None,
                    echo_left: None,
                    echo_add_bg: None,
                },
                GpuFullOperator::EchoLowRank { base, left, right } => {
                    let rows = base.rows() as u32;
                    let primary = bind_matvec(base, in_buf, out_buf);
                    let echo_right = bind_matvec(right, in_buf, &echo_mid_buf);
                    let echo_left = bind_matvec(left, &echo_mid_buf, &echo_delta_buf);
                    let add_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("full-echo-add-bg"),
                        layout: &bgl_2vec,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: out_len_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: echo_delta_buf.as_entire_binding(),
                            },
                        ],
                    });
                    BoundOperator {
                        primary,
                        echo_right: Some(echo_right),
                        echo_left: Some(echo_left),
                        echo_add_bg: Some((add_bg, rows.div_ceil(64))),
                    }
                }
            }
        };

        let mut layers = Vec::with_capacity(layer_specs.len());
        for spec in layer_specs {
            let mut owned = Vec::new();
            let in_norm_f32: Vec<f32> = spec.input_norm.iter().map(|&v| v as f32).collect();
            let in_norm_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("layer-in-norm"),
                contents: bytemuck::cast_slice(&in_norm_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
            vram_runtime_bytes += (in_norm_f32.len() as u64) * f32_sz;
            let input_norm_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("layer-in-norm-bg"),
                layout: &bgl_norm,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: norm_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: hidden_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: in_norm_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: norm_buf.as_entire_binding(),
                    },
                ],
            });
            owned.push(in_norm_buf);

            let post_norm_f32: Vec<f32> = spec.post_attn_norm.iter().map(|&v| v as f32).collect();
            let post_norm_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("layer-post-norm"),
                contents: bytemuck::cast_slice(&post_norm_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
            vram_runtime_bytes += (post_norm_f32.len() as u64) * f32_sz;
            let post_attn_norm_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("layer-post-norm-bg"),
                layout: &bgl_norm,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: norm_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: hidden_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: post_norm_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: norm_buf.as_entire_binding(),
                    },
                ],
            });
            owned.push(post_norm_buf);

            let mut make_bias = |bias: Option<Vec<f64>>,
                                 dst_buf: &wgpu::Buffer,
                                 len_buf: &wgpu::Buffer|
             -> Option<(wgpu::BindGroup, u32)> {
                let b = bias?;
                let bf32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
                let wg = (bf32.len() as u32).div_ceil(64);
                let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("layer-bias"),
                    contents: bytemuck::cast_slice(&bf32),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("layer-bias-bg"),
                    layout: &bgl_2vec,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: len_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: dst_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: b_buf.as_entire_binding(),
                        },
                    ],
                });
                owned.push(b_buf);
                Some((bg, wg))
            };

            let q_bias_add = make_bias(spec.q_bias, &q_buf, &hidden_len_buf);
            let k_bias_add = make_bias(spec.k_bias, &k_buf, &kv_len_buf);
            let v_bias_add = make_bias(spec.v_bias, &v_buf, &kv_len_buf);

            layers.push(BoundLayer {
                input_norm_bg,
                q_proj: bind_op(spec.q_proj, &norm_buf, &q_buf, &hidden_len_buf),
                q_bias_add,
                k_proj: bind_op(spec.k_proj, &norm_buf, &k_buf, &kv_len_buf),
                k_bias_add,
                v_proj: bind_op(spec.v_proj, &norm_buf, &v_buf, &kv_len_buf),
                v_bias_add,
                o_proj: bind_op(spec.o_proj, &attn_out_buf, &proj_tmp_buf, &hidden_len_buf),
                post_attn_norm_bg,
                gate_proj: bind_op(spec.gate_proj, &norm_buf, &gate_buf, &inter_len_buf),
                up_proj: bind_op(spec.up_proj, &norm_buf, &up_buf, &inter_len_buf),
                down_proj: bind_op(spec.down_proj, &gate_buf, &proj_tmp_buf, &hidden_len_buf),
                _owned_bufs: owned,
            });
        }

        let final_norm_f32: Vec<f32> = final_norm.iter().map(|&v| v as f32).collect();
        let final_norm_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("final-norm-w"),
            contents: bytemuck::cast_slice(&final_norm_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let final_norm_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("final-norm-bg"),
            layout: &bgl_norm,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: norm_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hidden_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: final_norm_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: norm_buf.as_entire_binding(),
                },
            ],
        });

        let vocab_len_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("full-vocab-len"),
            contents: bytemuck::bytes_of(&VecLenParams {
                len: config.vocab_size as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let lm_head_bound = bind_op(lm_head, &norm_buf, &logits_buf, &vocab_len_buf);

        let attn_res_add_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attn-res-add-bg"),
            layout: &bgl_2vec,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hidden_len_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hidden_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: proj_tmp_buf.as_entire_binding(),
                },
            ],
        });
        let mlp_res_add_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mlp-res-add-bg"),
            layout: &bgl_2vec,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hidden_len_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hidden_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: proj_tmp_buf.as_entire_binding(),
                },
            ],
        });
        let swiglu_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("swiglu-bg"),
            layout: &bgl_2vec,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inter_len_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gate_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: up_buf.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            engine,
            config,
            pipeline_rmsnorm,
            pipeline_hadamard,
            pipeline_rope_kv,
            pipeline_gqa_attn,
            pipeline_swiglu,
            pipeline_vec_add,
            bgl_rope_kv,
            bgl_gqa_attn,
            step_params_buf,
            hidden_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            logits_buf,
            logits_readback_buf,
            attn_res_add_bg,
            mlp_res_add_bg,
            swiglu_bg,
            final_norm_bg,
            lm_head: lm_head_bound,
            layers,
            _shared_bufs: vec![
                norm_buf,
                rot_buf,
                proj_tmp_buf,
                gate_buf,
                up_buf,
                echo_mid_buf,
                echo_delta_buf,
                norm_params_buf,
                hidden_len_buf,
                kv_len_buf,
                inter_len_buf,
                vocab_len_buf,
                final_norm_buf,
            ],
            vram_runtime_bytes,
        })
    }

    pub fn vram_runtime_bytes(&self) -> u64 {
        self.vram_runtime_bytes
    }

    /// Allocate a persistent GPU-resident KV-cache for up to `max_kv_len` tokens.
    pub fn create_kv_cache(&self, max_kv_len: usize) -> GpuKvCache {
        let len = max_kv_len.min(self.config.max_kv_len).max(1);
        let head_dim = self.config.hidden_size / self.config.num_heads;
        let kv_dim = self.config.num_kv_heads * head_dim;
        let per_buf_bytes = (len * kv_dim * std::mem::size_of::<f32>()) as u64;
        let mut k_bufs = Vec::with_capacity(self.config.num_layers);
        let mut v_bufs = Vec::with_capacity(self.config.num_layers);
        let mut rope_kv_bgs = Vec::with_capacity(self.config.num_layers);
        let mut attn_bgs = Vec::with_capacity(self.config.num_layers);

        for _ in 0..self.config.num_layers {
            let kb = self.engine.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-kv-k"),
                size: per_buf_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let vb = self.engine.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-kv-v"),
                size: per_buf_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let r_bg = self
                .engine
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gpu-rope-kv-bg"),
                    layout: &self.bgl_rope_kv,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.step_params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.q_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.k_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.v_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: kb.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: vb.as_entire_binding(),
                        },
                    ],
                });
            let a_bg = self
                .engine
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gpu-gqa-attn-bg"),
                    layout: &self.bgl_gqa_attn,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.step_params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.q_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: kb.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: vb.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.attn_out_buf.as_entire_binding(),
                        },
                    ],
                });
            k_bufs.push(kb);
            v_bufs.push(vb);
            rope_kv_bgs.push(r_bg);
            attn_bgs.push(a_bg);
        }

        GpuKvCache {
            max_kv_len: len,
            seq_len: 0,
            kv_bytes: per_buf_bytes * 2 * self.config.num_layers as u64,
            _k_bufs: k_bufs,
            _v_bufs: v_bufs,
            rope_kv_bgs,
            attn_bgs,
        }
    }

    fn dispatch_matvec<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>, bm: &'a BoundMatvec) {
        if let Some((hbg, wg)) = &bm.hadamard_bg {
            pass.set_pipeline(&self.pipeline_hadamard);
            pass.set_bind_group(0, hbg, &[]);
            pass.dispatch_workgroups(*wg, 1, 1);
        }
        pass.set_pipeline(bm.matrix.pipeline());
        pass.set_bind_group(0, &bm.matvec_bg, &[]);
        pass.dispatch_workgroups((bm.matrix.rows() as u32).div_ceil(64), 1, 1);
    }

    fn dispatch_op<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>, op: &'a BoundOperator) {
        self.dispatch_matvec(pass, &op.primary);
        if let (Some(right), Some(left), Some((add_bg, wg))) =
            (&op.echo_right, &op.echo_left, &op.echo_add_bg)
        {
            self.dispatch_matvec(pass, right);
            self.dispatch_matvec(pass, left);
            pass.set_pipeline(&self.pipeline_vec_add);
            pass.set_bind_group(0, add_bg, &[]);
            pass.dispatch_workgroups(*wg, 1, 1);
        }
    }

    /// Execute a COMPLETE transformer forward pass for one token (`embed_vec` at `position`)
    /// inside a **SINGLE GPU `CommandEncoder` submission**, returning `vocab_size` logits.
    pub fn forward_embedding(
        &self,
        embed_vec: &[f64],
        position: usize,
        cache: &mut GpuKvCache,
    ) -> Result<Vec<f64>, GpuComputeError> {
        if position >= cache.max_kv_len {
            return Err(GpuComputeError::DimensionExceedsScratch {
                dim: position,
                max_dim: cache.max_kv_len,
            });
        }
        let head_dim = self.config.hidden_size / self.config.num_heads;
        let kv_dim = self.config.num_kv_heads * head_dim;
        let seq_len = position + 1;
        cache.seq_len = cache.seq_len.max(seq_len);

        let _guard = self
            .engine
            .exec_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let embed_f32: Vec<f32> = embed_vec.iter().map(|&v| v as f32).collect();
        self.engine
            .queue
            .write_buffer(&self.hidden_buf, 0, bytemuck::cast_slice(&embed_f32));

        let step = StepParams {
            pos: position as u32,
            seq_len: seq_len as u32,
            num_heads: self.config.num_heads as u32,
            num_kv_heads: self.config.num_kv_heads as u32,
            head_dim: head_dim as u32,
            kv_dim: kv_dim as u32,
            rope_theta: self.config.rope_theta as f32,
            inv_sqrt_head_dim: 1.0 / (head_dim as f32).sqrt(),
        };
        self.engine
            .queue
            .write_buffer(&self.step_params_buf, 0, bytemuck::bytes_of(&step));

        let mut encoder =
            self.engine
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("qml-gpu-full-step"),
                });
        let hidden_wg = (self.config.hidden_size as u32).div_ceil(64);
        let inter_wg = (self.config.intermediate_size as u32).div_ceil(64);
        let rope_wg = ((self.config.num_heads * (head_dim / 2)) as u32).div_ceil(64);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("qml-gpu-full-pass"),
                timestamp_writes: None,
            });

            for (l_idx, layer) in self.layers.iter().enumerate() {
                // 1. Pre-attention RMSNorm
                pass.set_pipeline(&self.pipeline_rmsnorm);
                pass.set_bind_group(0, &layer.input_norm_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);

                // 2. Q, K, V compressed projections (+ optional Qwen2 biases)
                self.dispatch_op(&mut pass, &layer.q_proj);
                if let Some((bg, wg)) = &layer.q_bias_add {
                    pass.set_pipeline(&self.pipeline_vec_add);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(*wg, 1, 1);
                }
                self.dispatch_op(&mut pass, &layer.k_proj);
                if let Some((bg, wg)) = &layer.k_bias_add {
                    pass.set_pipeline(&self.pipeline_vec_add);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(*wg, 1, 1);
                }
                self.dispatch_op(&mut pass, &layer.v_proj);
                if let Some((bg, wg)) = &layer.v_bias_add {
                    pass.set_pipeline(&self.pipeline_vec_add);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(*wg, 1, 1);
                }

                // 3. In-place RoPE + append K/V to VRAM KV-cache
                pass.set_pipeline(&self.pipeline_rope_kv);
                pass.set_bind_group(0, &cache.rope_kv_bgs[l_idx], &[]);
                pass.dispatch_workgroups(rope_wg, 1, 1);

                // 4. Causal GQA Attention in VRAM
                pass.set_pipeline(&self.pipeline_gqa_attn);
                pass.set_bind_group(0, &cache.attn_bgs[l_idx], &[]);
                pass.dispatch_workgroups(self.config.num_heads as u32, 1, 1);

                // 5. O projection + residual add
                self.dispatch_op(&mut pass, &layer.o_proj);
                pass.set_pipeline(&self.pipeline_vec_add);
                pass.set_bind_group(0, &self.attn_res_add_bg, &[]);
                pass.dispatch_workgroups(hidden_wg, 1, 1);

                // 6. Post-attention RMSNorm + SwiGLU MLP + residual add
                pass.set_pipeline(&self.pipeline_rmsnorm);
                pass.set_bind_group(0, &layer.post_attn_norm_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);

                self.dispatch_op(&mut pass, &layer.gate_proj);
                self.dispatch_op(&mut pass, &layer.up_proj);

                pass.set_pipeline(&self.pipeline_swiglu);
                pass.set_bind_group(0, &self.swiglu_bg, &[]);
                pass.dispatch_workgroups(inter_wg, 1, 1);

                self.dispatch_op(&mut pass, &layer.down_proj);
                pass.set_pipeline(&self.pipeline_vec_add);
                pass.set_bind_group(0, &self.mlp_res_add_bg, &[]);
                pass.dispatch_workgroups(hidden_wg, 1, 1);
            }

            // 7. Final RMSNorm + LM Head
            pass.set_pipeline(&self.pipeline_rmsnorm);
            pass.set_bind_group(0, &self.final_norm_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);

            self.dispatch_op(&mut pass, &self.lm_head);
        }

        let out_bytes = (self.config.vocab_size * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(&self.logits_buf, 0, &self.logits_readback_buf, 0, out_bytes);
        self.engine.queue.submit(Some(encoder.finish()));

        let slice = self.logits_readback_buf.slice(0..out_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.engine.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuComputeError::BufferMapFailed(e.to_string()))?
            .map_err(|e| GpuComputeError::BufferMapFailed(format!("{e:?}")))?;

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let result: Vec<f64> = floats.iter().map(|&v| f64::from(v)).collect();
        drop(data);
        self.logits_readback_buf.unmap();
        Ok(result)
    }
}
