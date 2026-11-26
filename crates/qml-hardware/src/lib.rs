//! Host Hardware Observatory & Experimental WGPU Compute Backend (`qml-hardware`).
//!
//! Provides cross-platform CPU/SIMD/RAM telemetry, WGPU GPU adapter discovery,
//! resource preset calculation, and a validated WGSL compute shader matrix-vector
//! multiplication prototype (`y = A * x`) with graceful CPU fallback when no
//! GPU adapter is available (such as on headless CI runners).

pub mod gpu_full;

pub use gpu_full::{GpuFullConfig, GpuFullLayerSpec, GpuFullModel, GpuFullOperator, GpuKvCache};

use qml_core::{DenseMatrix, MatrixError};
use qml_quant::{
    Bf16Matrix, Fp16Matrix, HadamardSpec, Int2AsymGroupMatrix, Int3AsymGroupMatrix,
    Int4AsymGroupMatrix, Int4SymGroupMatrix, Int8SymMatrix, LinearOperator, QuantizedFactor,
    TernaryPackedMatrix, apply_hadamard_forward,
};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::{Arc, Mutex};
use sysinfo::System;

/// Summary of host CPU, SIMD, memory, and discovered GPU adapters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareReport {
    pub os: String,
    pub arch: String,
    pub cpu_brand: String,
    pub logical_cores: usize,
    pub simd_features: Vec<String>,
    pub total_ram_bytes: u64,
    pub available_ram_bytes: u64,
    pub gpus: Vec<GpuAdapterInfo>,
    pub recommended_backend: String,
}

/// Metadata for a discovered WGPU graphics/compute adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAdapterInfo {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: String,
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u32,
    pub max_compute_workgroup_size_x: u32,
}

/// Resource budget preset derived from host hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourcePreset {
    Conservative,
    Balanced,
    Performance,
}

/// Concrete RAM and thread limits computed from a [`ResourcePreset`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBudgetPlan {
    pub preset: ResourcePreset,
    pub ram_budget_bytes: u64,
    pub max_threads: usize,
    pub prefer_gpu_compute: bool,
}

impl ResourcePreset {
    pub fn label(self) -> &'static str {
        match self {
            Self::Conservative => "Conservative (25% RAM, 50% CPU)",
            Self::Balanced => "Balanced (50% RAM, 75% CPU)",
            Self::Performance => "Performance (75% RAM, 100% CPU)",
        }
    }

    pub fn plan(self, hw: &HardwareReport) -> ResourceBudgetPlan {
        let (ram_frac, cpu_frac) = match self {
            Self::Conservative => (0.25, 0.50),
            Self::Balanced => (0.50, 0.75),
            Self::Performance => (0.75, 1.00),
        };
        let ram_budget_bytes = ((hw.total_ram_bytes as f64) * ram_frac) as u64;
        let max_threads = ((hw.logical_cores as f64) * cpu_frac).round().max(1.0) as usize;
        let prefer_gpu_compute = !hw.gpus.is_empty();
        ResourceBudgetPlan {
            preset: self,
            ram_budget_bytes: ram_budget_bytes.max(512 * 1024 * 1024),
            max_threads,
            prefer_gpu_compute,
        }
    }
}

/// Detect x86/x86_64 CPU SIMD instruction sets safely at runtime.
pub fn detect_cpu_simd_features() -> Vec<String> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let mut features = Vec::new();
        if std::arch::is_x86_feature_detected!("sse4.2") {
            features.push("SSE4.2".to_owned());
        }
        if std::arch::is_x86_feature_detected!("avx") {
            features.push("AVX".to_owned());
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            features.push("AVX2".to_owned());
        }
        if std::arch::is_x86_feature_detected!("fma") {
            features.push("FMA".to_owned());
        }
        if std::arch::is_x86_feature_detected!("f16c") {
            features.push("F16C".to_owned());
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            features.push("AVX-512F".to_owned());
        }
        features
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        Vec::new()
    }
}

/// Enumerate all available WGPU adapters without panicking on headless hosts.
pub fn discover_gpu_adapters() -> Vec<GpuAdapterInfo> {
    let instance = wgpu::Instance::default();
    let mut out = Vec::new();
    for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
        let info = adapter.get_info();
        let limits = adapter.limits();
        out.push(GpuAdapterInfo {
            name: info.name,
            vendor_id: info.vendor,
            device_id: info.device,
            device_type: format!("{:?}", info.device_type),
            backend: format!("{:?}", info.backend),
            driver: info.driver,
            driver_info: info.driver_info,
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
        });
    }
    // Deduplicate adapters by (name, backend) and prefer DiscreteGpu first.
    out.sort_by(|a, b| {
        let rank = |t: &str| match t {
            "DiscreteGpu" => 0,
            "IntegratedGpu" => 1,
            "VirtualGpu" => 2,
            _ => 3,
        };
        rank(&a.device_type)
            .cmp(&rank(&b.device_type))
            .then_with(|| a.name.cmp(&b.name))
            .then_with(|| a.backend.cmp(&b.backend))
    });
    out
}

/// Collect a complete host hardware and GPU telemetry snapshot.
pub fn probe_hardware() -> HardwareReport {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.refresh_cpu_list(sysinfo::CpuRefreshKind::nothing());

    let cpu_brand = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("{} processor", std::env::consts::ARCH));
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or_else(|_| sys.cpus().len().max(1));

    let gpus = discover_gpu_adapters();
    let recommended_backend =
        if let Some(discrete) = gpus.iter().find(|g| g.device_type == "DiscreteGpu") {
            format!(
                "WGPU Compute ({} / {}) + Rayon CPU",
                discrete.name, discrete.backend
            )
        } else if let Some(hw) = gpus
            .iter()
            .find(|g| matches!(g.device_type.as_str(), "IntegratedGpu" | "VirtualGpu"))
        {
            format!("WGPU Compute ({} / {}) + Rayon CPU", hw.name, hw.backend)
        } else {
            "Rayon Multi-Threaded CPU (100% safe Rust)".to_owned()
        };

    HardwareReport {
        os: std::env::consts::OS.to_owned(),
        arch: std::env::consts::ARCH.to_owned(),
        cpu_brand,
        logical_cores,
        simd_features: detect_cpu_simd_features(),
        total_ram_bytes: sys.total_memory(),
        available_ram_bytes: sys.available_memory(),
        gpus,
        recommended_backend,
    }
}

/// Errors returned by the experimental WGPU compute engine.
#[derive(Debug)]
pub enum GpuComputeError {
    NoAdapterAvailable,
    DeviceRequestFailed(String),
    Matrix(MatrixError),
    BufferMapFailed(String),
    DimensionExceedsScratch { dim: usize, max_dim: usize },
}

impl fmt::Display for GpuComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapterAvailable => write!(f, "no WGPU compute adapter available on this host"),
            Self::DeviceRequestFailed(msg) => write!(f, "WGPU device request failed: {msg}"),
            Self::Matrix(err) => write!(f, "matrix error: {err}"),
            Self::BufferMapFailed(msg) => write!(f, "WGPU buffer readback failed: {msg}"),
            Self::DimensionExceedsScratch { dim, max_dim } => write!(
                f,
                "dimension {dim} exceeds preallocated GPU scratch capacity {max_dim}"
            ),
        }
    }
}

impl std::error::Error for GpuComputeError {}

impl From<MatrixError> for GpuComputeError {
    fn from(value: MatrixError) -> Self {
        Self::Matrix(value)
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatvecUniforms {
    rows: u32,
    cols: u32,
    group_size: u32,
    _pad1: u32,
}

const MATVEC_WGSL: &str = r#"
struct Dims {
    rows: u32,
    cols: u32,
    group_size: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> matrix: array<f32>;
@group(0) @binding(2) var<storage, read> vector: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.rows) {
        return;
    }
    var acc: f32 = 0.0;
    let base = row * dims.cols;
    for (var c: u32 = 0u; c < dims.cols; c = c + 1u) {
        acc = acc + matrix[base + c] * vector[c];
    }
    output[row] = acc;
}
"#;

const MATVEC_INT8_WGSL: &str = r#"
struct Dims {
    rows: u32,
    cols: u32,
    group_size: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> codes_packed: array<u32>;
@group(0) @binding(2) var<storage, read> scales_packed: array<u32>;
@group(0) @binding(3) var<storage, read> vector: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

fn decode_f16_scale(g: u32) -> f32 {
    let pair = unpack2x16float(scales_packed[g >> 1u]);
    return select(pair.x, pair.y, (g & 1u) != 0u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.rows) {
        return;
    }
    var acc: f32 = 0.0;
    let base = row * dims.cols;
    for (var c: u32 = 0u; c < dims.cols; c = c + 1u) {
        let idx = base + c;
        let word = codes_packed[idx >> 2u];
        let q = extractBits(i32(word), (idx & 3u) * 8u, 8u);
        let scale = decode_f16_scale(idx / dims.group_size);
        acc = acc + (f32(q) * scale) * vector[c];
    }
    output[row] = acc;
}
"#;

const MATVEC_Q4_0_WGSL: &str = r#"
struct Dims {
    rows: u32,
    cols: u32,
    group_size: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> nibbles_packed: array<u32>;
@group(0) @binding(2) var<storage, read> scales_packed: array<u32>;
@group(0) @binding(3) var<storage, read> vector: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

fn decode_f16_scale(g: u32) -> f32 {
    let pair = unpack2x16float(scales_packed[g >> 1u]);
    return select(pair.x, pair.y, (g & 1u) != 0u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.rows) {
        return;
    }
    var acc: f32 = 0.0;
    let base = row * dims.cols;
    for (var c: u32 = 0u; c < dims.cols; c = c + 1u) {
        let idx = base + c;
        let word = nibbles_packed[idx >> 3u];
        let nibble = (word >> ((idx & 7u) * 4u)) & 0xFu;
        let q = i32(nibble) - 8;
        let scale = decode_f16_scale(idx / dims.group_size);
        acc = acc + (f32(q) * scale) * vector[c];
    }
    output[row] = acc;
}
"#;

const MATVEC_Q4_1_WGSL: &str = r#"
struct Dims {
    rows: u32,
    cols: u32,
    group_size: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> nibbles_packed: array<u32>;
@group(0) @binding(2) var<storage, read> scale_min_packed: array<u32>;
@group(0) @binding(3) var<storage, read> vector: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.rows) {
        return;
    }
    var acc: f32 = 0.0;
    let base = row * dims.cols;
    for (var c: u32 = 0u; c < dims.cols; c = c + 1u) {
        let idx = base + c;
        let word = nibbles_packed[idx >> 3u];
        let nibble = (word >> ((idx & 7u) * 4u)) & 0xFu;
        let g = idx / dims.group_size;
        let sm = unpack2x16float(scale_min_packed[g]);
        acc = acc + (f32(nibble) * sm.x + sm.y) * vector[c];
    }
    output[row] = acc;
}
"#;

const MATVEC_FP16_WGSL: &str = r#"
struct Dims {
    rows: u32,
    cols: u32,
    group_size: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> weights_packed: array<u32>;
@group(0) @binding(2) var<storage, read> vector: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.rows) {
        return;
    }
    var acc: f32 = 0.0;
    let base = row * dims.cols;
    for (var c: u32 = 0u; c < dims.cols; c = c + 1u) {
        let idx = base + c;
        let pair = unpack2x16float(weights_packed[idx >> 1u]);
        let w = select(pair.x, pair.y, (idx & 1u) != 0u);
        acc = acc + w * vector[c];
    }
    output[row] = acc;
}
"#;

const MATVEC_BF16_WGSL: &str = r#"
struct Dims {
    rows: u32,
    cols: u32,
    group_size: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> weights_packed: array<u32>;
@group(0) @binding(2) var<storage, read> vector: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.rows) {
        return;
    }
    var acc: f32 = 0.0;
    let base = row * dims.cols;
    for (var c: u32 = 0u; c < dims.cols; c = c + 1u) {
        let idx = base + c;
        let word = weights_packed[idx >> 1u];
        let bits = select((word & 0xFFFFu) << 16u, word & 0xFFFF0000u, (idx & 1u) != 0u);
        let w = bitcast<f32>(bits);
        acc = acc + w * vector[c];
    }
    output[row] = acc;
}
"#;

/// Preallocated reusable GPU scratch vector capacity (262,144 f32s = 1 MiB per buffer).
/// Covers vocabularies up to 262K tokens (SmolLM2 49K, Llama-3 128K, Qwen2.5 152K, Gemma 256K)
/// with zero per-token allocations.
pub const GPU_SCRATCH_MAX_ELEMS: usize = 262_144;

fn pack_bytes_u32(bytes: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(bytes.len().div_ceil(4).max(1));
    for chunk in bytes.chunks(4) {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        out.push(u32::from_le_bytes(word));
    }
    if out.is_empty() {
        out.push(0);
    }
    out
}

fn pack_u16_u32(vals: &[u16]) -> Vec<u32> {
    let mut out = Vec::with_capacity(vals.len().div_ceil(2).max(1));
    for chunk in vals.chunks(2) {
        let lo = u32::from(chunk[0]);
        let hi = if chunk.len() > 1 {
            u32::from(chunk[1])
        } else {
            0
        };
        out.push(lo | (hi << 16));
    }
    if out.is_empty() {
        out.push(0);
    }
    out
}

/// Direct compressed 2-bit asymmetric (`INT2_ASYM`) WGSL compute shader (4 codes/byte + interleaved f16 scale/min).
const MATVEC_INT2_ASYM_WGSL: &str = r#"
struct Dims {
    rows: u32,
    cols: u32,
    group_size: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> packed_codes: array<u32>;
@group(0) @binding(2) var<storage, read> scale_mins: array<u32>;
@group(0) @binding(3) var<storage, read> vector: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.rows) {
        return;
    }
    let cols = dims.cols;
    let gsize = dims.group_size;
    let gpr = (cols + gsize - 1u) / gsize;
    let prb = (cols + 3u) / 4u;
    let row_byte_base = row * prb;
    var sum: f32 = 0.0;
    for (var g: u32 = 0u; g < gpr; g = g + 1u) {
        let sm = unpack2x16float(scale_mins[row * gpr + g]);
        let start = g * gsize;
        let end = min(start + gsize, cols);
        var qx: f32 = 0.0;
        var xsum: f32 = 0.0;
        for (var c: u32 = start; c < end; c = c + 1u) {
            let b_idx = row_byte_base + (c >> 2u);
            let word = packed_codes[b_idx >> 2u];
            let bval = (word >> ((b_idx & 3u) * 8u)) & 0xFFu;
            let q = f32((bval >> ((c & 3u) * 2u)) & 3u);
            let xv = vector[c];
            qx = qx + q * xv;
            xsum = xsum + xv;
        }
        sum = sum + sm.x * qx + sm.y * xsum;
    }
    output[row] = sum;
}
"#;

/// Direct compressed 3-bit asymmetric (`INT3_ASYM`) WGSL compute shader (8 codes/3 bytes + interleaved f16 scale/min).
const MATVEC_INT3_ASYM_WGSL: &str = r#"
struct Dims {
    rows: u32,
    cols: u32,
    group_size: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> packed_codes: array<u32>;
@group(0) @binding(2) var<storage, read> scale_mins: array<u32>;
@group(0) @binding(3) var<storage, read> vector: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

fn read_packed_byte(b_idx: u32) -> u32 {
    let word = packed_codes[b_idx >> 2u];
    return (word >> ((b_idx & 3u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.rows) {
        return;
    }
    let cols = dims.cols;
    let gsize = dims.group_size;
    let gpr = (cols + gsize - 1u) / gsize;
    let prb = ((cols + 7u) / 8u) * 3u;
    let row_byte_base = row * prb;
    var sum: f32 = 0.0;
    for (var g: u32 = 0u; g < gpr; g = g + 1u) {
        let sm = unpack2x16float(scale_mins[row * gpr + g]);
        let start = g * gsize;
        let end = min(start + gsize, cols);
        var qx: f32 = 0.0;
        var xsum: f32 = 0.0;
        for (var c: u32 = start; c < end; c = c + 1u) {
            let blk = c >> 3u;
            let idx = c & 7u;
            let b0 = row_byte_base + blk * 3u;
            let bits = read_packed_byte(b0) | (read_packed_byte(b0 + 1u) << 8u) | (read_packed_byte(b0 + 2u) << 16u);
            let q = f32((bits >> (idx * 3u)) & 7u);
            let xv = vector[c];
            qx = qx + q * xv;
            xsum = xsum + xv;
        }
        sum = sum + sm.x * qx + sm.y * xsum;
    }
    output[row] = sum;
}
"#;

/// Direct compressed packed base-3 ternary (`TERNARY_PACKED`, 5 trits/byte) WGSL compute shader.
const MATVEC_TERNARY_WGSL: &str = r#"
struct Dims {
    rows: u32,
    cols: u32,
    group_size: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> packed_trits: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read> vector: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.rows) {
        return;
    }
    let cols = dims.cols;
    let gsize = dims.group_size;
    let gpr = (cols + gsize - 1u) / gsize;
    let prb = (cols + 4u) / 5u;
    let row_byte_base = row * prb;
    var pow3 = array<u32, 5>(1u, 3u, 9u, 27u, 81u);
    var sum: f32 = 0.0;
    for (var g: u32 = 0u; g < gpr; g = g + 1u) {
        let s_idx = row * gpr + g;
        let s_pair = unpack2x16float(scales[s_idx >> 1u]);
        let scale = select(s_pair.x, s_pair.y, (s_idx & 1u) == 1u);
        let start = g * gsize;
        let end = min(start + gsize, cols);
        var acc: f32 = 0.0;
        for (var c: u32 = start; c < end; c = c + 1u) {
            let b_idx = row_byte_base + (c / 5u);
            let word = packed_trits[b_idx >> 2u];
            let bval = (word >> ((b_idx & 3u) * 8u)) & 0xFFu;
            let d = (bval / pow3[c % 5u]) % 3u;
            let w = f32(i32(d) - 1);
            acc = acc + w * vector[c];
        }
        sum = sum + scale * acc;
    }
    output[row] = sum;
}
"#;

/// Experimental WGPU compute shader engine for dense `f32` and direct compressed
/// `FP16` / `BF16` / `INT8_SYM` / `Q4_0` / `Q4_1` / `INT3_ASYM` / `INT2_ASYM` / `TERNARY_PACKED`
/// matrix-vector multiplication (zero dense weight reconstruction in VRAM) with preallocated
/// reusable scratch buffers for zero-allocation per-token inference.
pub struct WgpuMatvecEngine {
    adapter_name: String,
    backend_name: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_f32: wgpu::ComputePipeline,
    pipeline_fp16: wgpu::ComputePipeline,
    pipeline_bf16: wgpu::ComputePipeline,
    bgl_f32: wgpu::BindGroupLayout,
    pipeline_int8: wgpu::ComputePipeline,
    pipeline_q4_0: wgpu::ComputePipeline,
    pipeline_q4_1: wgpu::ComputePipeline,
    pipeline_int3: wgpu::ComputePipeline,
    pipeline_int2: wgpu::ComputePipeline,
    pipeline_ternary: wgpu::ComputePipeline,
    bgl_quant: wgpu::BindGroupLayout,
    scratch_vec_buf: wgpu::Buffer,
    scratch_out_buf: wgpu::Buffer,
    scratch_readback_buf: wgpu::Buffer,
    exec_lock: Mutex<()>,
}

/// Format of a persistent GPU-resident matrix uploaded once to VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMatrixKind {
    Fp32,
    Fp16,
    Bf16,
    Int8Sym,
    Int4Sym,
    Int4Asym,
    Int3Asym,
    Int2Asym,
    TernaryPacked,
}

/// Persistent compressed weight matrix stored once in GPU VRAM with a pre-bound
/// compute `BindGroup` targeting the engine's reusable scratch buffers.
///
/// Executing [`Self::matvec`] applies any Hadamard input pre-rotation in `O(cols log b)`,
/// uploads only the small input vector (`cols * 4` bytes), and performs zero weight
/// transfers, zero GPU buffer allocations, and zero bind-group allocations per token.
pub struct PersistentGpuMatrix {
    engine: Arc<WgpuMatvecEngine>,
    kind: GpuMatrixKind,
    rows: usize,
    cols: usize,
    hadamard: HadamardSpec,
    vram_bytes: u64,
    bind_group: wgpu::BindGroup,
    _weight_buffers: Vec<wgpu::Buffer>,
}

impl PersistentGpuMatrix {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn kind(&self) -> GpuMatrixKind {
        self.kind
    }

    pub fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }

    pub fn hadamard(&self) -> HadamardSpec {
        self.hadamard
    }

    pub fn engine(&self) -> &Arc<WgpuMatvecEngine> {
        &self.engine
    }

    pub(crate) fn pipeline(&self) -> &wgpu::ComputePipeline {
        match self.kind {
            GpuMatrixKind::Fp32 => &self.engine.pipeline_f32,
            GpuMatrixKind::Fp16 => &self.engine.pipeline_fp16,
            GpuMatrixKind::Bf16 => &self.engine.pipeline_bf16,
            GpuMatrixKind::Int8Sym => &self.engine.pipeline_int8,
            GpuMatrixKind::Int4Sym => &self.engine.pipeline_q4_0,
            GpuMatrixKind::Int4Asym => &self.engine.pipeline_q4_1,
            GpuMatrixKind::Int3Asym => &self.engine.pipeline_int3,
            GpuMatrixKind::Int2Asym => &self.engine.pipeline_int2,
            GpuMatrixKind::TernaryPacked => &self.engine.pipeline_ternary,
        }
    }

    /// Create a compute bind group binding this matrix's persistent VRAM weights to arbitrary
    /// GPU-resident `input_buf` and `output_buf` storage buffers (used by Full GPU-Resident inference).
    pub(crate) fn create_io_bind_group(
        &self,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        match self.kind {
            GpuMatrixKind::Fp32 | GpuMatrixKind::Fp16 | GpuMatrixKind::Bf16 => self
                .engine
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("full-gpu-matvec-1buf-bg"),
                    layout: &self.engine.bgl_f32,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self._weight_buffers[0].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self._weight_buffers[1].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: output_buf.as_entire_binding(),
                        },
                    ],
                }),
            _ => self
                .engine
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("full-gpu-matvec-2buf-bg"),
                    layout: &self.engine.bgl_quant,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self._weight_buffers[0].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self._weight_buffers[1].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self._weight_buffers[2].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: output_buf.as_entire_binding(),
                        },
                    ],
                }),
        }
    }

    /// Execute `y = W * x` directly on the GPU-resident compressed matrix using
    /// preallocated reusable scratch buffers (zero weight transfer per call).
    pub fn matvec(&self, vector: &[f64]) -> Result<Vec<f64>, GpuComputeError> {
        if vector.len() != self.cols {
            return Err(GpuComputeError::Matrix(MatrixError::DimensionMismatch {
                expected: self.cols,
                actual: vector.len(),
            }));
        }
        let pipeline = match self.kind {
            GpuMatrixKind::Fp32 => &self.engine.pipeline_f32,
            GpuMatrixKind::Fp16 => &self.engine.pipeline_fp16,
            GpuMatrixKind::Bf16 => &self.engine.pipeline_bf16,
            GpuMatrixKind::Int8Sym => &self.engine.pipeline_int8,
            GpuMatrixKind::Int4Sym => &self.engine.pipeline_q4_0,
            GpuMatrixKind::Int4Asym => &self.engine.pipeline_q4_1,
            GpuMatrixKind::Int3Asym => &self.engine.pipeline_int3,
            GpuMatrixKind::Int2Asym => &self.engine.pipeline_int2,
            GpuMatrixKind::TernaryPacked => &self.engine.pipeline_ternary,
        };
        let rotated_vec;
        let active_vec = if self.hadamard.is_enabled() {
            rotated_vec = {
                let mut v = vector.to_vec();
                apply_hadamard_forward(&mut v, self.hadamard.block_size, self.hadamard.seed);
                v
            };
            &rotated_vec[..]
        } else {
            vector
        };
        let _guard = self
            .engine
            .exec_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let vec_f32: Vec<f32> = active_vec.iter().map(|&v| v as f32).collect();
        self.engine.queue.write_buffer(
            &self.engine.scratch_vec_buf,
            0,
            bytemuck::cast_slice(&vec_f32),
        );
        let out_bytes = (self.rows * std::mem::size_of::<f32>()) as u64;
        self.engine.dispatch_and_readback_partial(
            pipeline,
            &self.bind_group,
            self.rows as u32,
            &self.engine.scratch_out_buf,
            &self.engine.scratch_readback_buf,
            out_bytes,
        )
    }
}

impl WgpuMatvecEngine {
    /// Initializes a WGPU high-performance compute device and compiles the WGSL matvec kernel.
    pub fn try_new() -> Result<Self, GpuComputeError> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or(GpuComputeError::NoAdapterAvailable)?;

        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            // Software CPU rasterizers (e.g. Microsoft WARP / llvmpipe) have no VRAM,
            // run far slower than native Rayon CPU, and can fault on multi-dispatch
            // compute graphs; decline them so Q-Core uses native Rayon CPU cleanly.
            return Err(GpuComputeError::NoAdapterAvailable);
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("qml-wgpu-compute"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| GpuComputeError::DeviceRequestFailed(e.to_string()))?;

        let shader_f32 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qml-matvec-wgsl"),
            source: wgpu::ShaderSource::Wgsl(MATVEC_WGSL.into()),
        });
        let shader_fp16 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qml-matvec-fp16-wgsl"),
            source: wgpu::ShaderSource::Wgsl(MATVEC_FP16_WGSL.into()),
        });
        let shader_bf16 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qml-matvec-bf16-wgsl"),
            source: wgpu::ShaderSource::Wgsl(MATVEC_BF16_WGSL.into()),
        });
        let shader_int8 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qml-matvec-int8-wgsl"),
            source: wgpu::ShaderSource::Wgsl(MATVEC_INT8_WGSL.into()),
        });
        let shader_q4_0 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qml-matvec-q4-0-wgsl"),
            source: wgpu::ShaderSource::Wgsl(MATVEC_Q4_0_WGSL.into()),
        });
        let shader_q4_1 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qml-matvec-q4-1-wgsl"),
            source: wgpu::ShaderSource::Wgsl(MATVEC_Q4_1_WGSL.into()),
        });
        let shader_int3 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qml-matvec-int3-wgsl"),
            source: wgpu::ShaderSource::Wgsl(MATVEC_INT3_ASYM_WGSL.into()),
        });
        let shader_int2 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qml-matvec-int2-wgsl"),
            source: wgpu::ShaderSource::Wgsl(MATVEC_INT2_ASYM_WGSL.into()),
        });
        let shader_ternary = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qml-matvec-ternary-wgsl"),
            source: wgpu::ShaderSource::Wgsl(MATVEC_TERNARY_WGSL.into()),
        });

        let ro_storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let rw_storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl_f32 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qml-matvec-f32-bgl"),
            entries: &[uniform_entry, ro_storage(1), ro_storage(2), rw_storage(3)],
        });
        let bgl_quant = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qml-matvec-quant-bgl"),
            entries: &[
                uniform_entry,
                ro_storage(1),
                ro_storage(2),
                ro_storage(3),
                rw_storage(4),
            ],
        });

        let pl_f32 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qml-matvec-f32-pl"),
            bind_group_layouts: &[&bgl_f32],
            push_constant_ranges: &[],
        });
        let pl_quant = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qml-matvec-quant-pl"),
            bind_group_layouts: &[&bgl_quant],
            push_constant_ranges: &[],
        });

        let pipeline_f32 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("qml-matvec-f32-pipeline"),
            layout: Some(&pl_f32),
            module: &shader_f32,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_fp16 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("qml-matvec-fp16-pipeline"),
            layout: Some(&pl_f32),
            module: &shader_fp16,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_bf16 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("qml-matvec-bf16-pipeline"),
            layout: Some(&pl_f32),
            module: &shader_bf16,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_int8 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("qml-matvec-int8-pipeline"),
            layout: Some(&pl_quant),
            module: &shader_int8,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_q4_0 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("qml-matvec-q4-0-pipeline"),
            layout: Some(&pl_quant),
            module: &shader_q4_0,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_q4_1 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("qml-matvec-q4-1-pipeline"),
            layout: Some(&pl_quant),
            module: &shader_q4_1,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_int3 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("qml-matvec-int3-pipeline"),
            layout: Some(&pl_quant),
            module: &shader_int3,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_int2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("qml-matvec-int2-pipeline"),
            layout: Some(&pl_quant),
            module: &shader_int2,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_ternary = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("qml-matvec-ternary-pipeline"),
            layout: Some(&pl_quant),
            module: &shader_ternary,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let scratch_bytes = (GPU_SCRATCH_MAX_ELEMS * std::mem::size_of::<f32>()) as u64;
        let scratch_vec_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qml-scratch-vec"),
            size: scratch_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scratch_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qml-scratch-out"),
            size: scratch_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scratch_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qml-scratch-readback"),
            size: scratch_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            adapter_name: info.name,
            backend_name: format!("{:?}", info.backend),
            device,
            queue,
            pipeline_f32,
            pipeline_fp16,
            pipeline_bf16,
            bgl_f32,
            pipeline_int8,
            pipeline_q4_0,
            pipeline_q4_1,
            pipeline_int3,
            pipeline_int2,
            pipeline_ternary,
            bgl_quant,
            scratch_vec_buf,
            scratch_out_buf,
            scratch_readback_buf,
            exec_lock: Mutex::new(()),
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub fn backend_name(&self) -> &str {
        &self.backend_name
    }

    /// Executes `y = matrix * vector` on the GPU in `f32` and returns `Vec<f64>`.
    pub fn matvec(
        &self,
        matrix: &DenseMatrix,
        vector: &[f64],
    ) -> Result<Vec<f64>, GpuComputeError> {
        if vector.len() != matrix.cols() {
            return Err(GpuComputeError::Matrix(MatrixError::DimensionMismatch {
                expected: matrix.cols(),
                actual: vector.len(),
            }));
        }
        use wgpu::util::DeviceExt;

        let rows = matrix.rows() as u32;
        let cols = matrix.cols() as u32;
        let uniforms = MatvecUniforms {
            rows,
            cols,
            group_size: 1,
            _pad1: 0,
        };
        let mat_f32: Vec<f32> = matrix.as_slice().iter().map(|&v| v as f32).collect();
        let vec_f32: Vec<f32> = vector.iter().map(|&v| v as f32).collect();
        let out_bytes = (matrix.rows() * std::mem::size_of::<f32>()) as u64;

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dims"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mat_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matrix"),
                contents: bytemuck::cast_slice(&mat_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let vec_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vector"),
                contents: bytemuck::cast_slice(&vec_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qml-matvec-f32-bg"),
            layout: &self.bgl_f32,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mat_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vec_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch_and_readback(
            &self.pipeline_f32,
            &bind_group,
            rows,
            &out_buf,
            &readback_buf,
            out_bytes,
        )
    }

    /// Executes direct compressed `INT8_SYM` matrix-vector multiplication on the GPU
    /// without ever reconstructing dense floating-point weights in VRAM.
    pub fn matvec_int8_sym(
        &self,
        matrix: &Int8SymMatrix,
        vector: &[f64],
    ) -> Result<Vec<f64>, GpuComputeError> {
        if vector.len() != matrix.cols() {
            return Err(GpuComputeError::Matrix(MatrixError::DimensionMismatch {
                expected: matrix.cols(),
                actual: vector.len(),
            }));
        }
        let code_bytes: Vec<u8> = matrix.codes().iter().map(|&q| q as u8).collect();
        let packed_codes = pack_bytes_u32(&code_bytes);
        let packed_scales = pack_u16_u32(matrix.scales_f16());
        self.run_quantized_matvec(
            &self.pipeline_int8,
            matrix.rows() as u32,
            matrix.cols() as u32,
            matrix.group_size() as u32,
            &packed_codes,
            &packed_scales,
            vector,
        )
    }

    /// Executes direct compressed `Q4_0 (INT4_SYM)` matrix-vector multiplication on the GPU
    /// without ever reconstructing dense floating-point weights in VRAM.
    pub fn matvec_int4_sym(
        &self,
        matrix: &Int4SymGroupMatrix,
        vector: &[f64],
    ) -> Result<Vec<f64>, GpuComputeError> {
        if vector.len() != matrix.cols() {
            return Err(GpuComputeError::Matrix(MatrixError::DimensionMismatch {
                expected: matrix.cols(),
                actual: vector.len(),
            }));
        }
        let packed_nibbles = pack_bytes_u32(matrix.packed_nibbles());
        let packed_scales = pack_u16_u32(matrix.scales_f16());
        self.run_quantized_matvec(
            &self.pipeline_q4_0,
            matrix.rows() as u32,
            matrix.cols() as u32,
            matrix.group_size() as u32,
            &packed_nibbles,
            &packed_scales,
            vector,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_quantized_matvec(
        &self,
        pipeline: &wgpu::ComputePipeline,
        rows: u32,
        cols: u32,
        group_size: u32,
        packed_weights: &[u32],
        packed_scales: &[u32],
        vector: &[f64],
    ) -> Result<Vec<f64>, GpuComputeError> {
        use wgpu::util::DeviceExt;

        let uniforms = MatvecUniforms {
            rows,
            cols,
            group_size,
            _pad1: 0,
        };
        let vec_f32: Vec<f32> = vector.iter().map(|&v| v as f32).collect();
        let out_bytes = (rows as usize * std::mem::size_of::<f32>()) as u64;

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quant-dims"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let weights_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quant-weights"),
                contents: bytemuck::cast_slice(packed_weights),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let scales_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quant-scales"),
                contents: bytemuck::cast_slice(packed_scales),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let vec_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quant-vector"),
                contents: bytemuck::cast_slice(&vec_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quant-output"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quant-readback"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qml-matvec-quant-bg"),
            layout: &self.bgl_quant,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scales_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: vec_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch_and_readback(
            pipeline,
            &bind_group,
            rows,
            &out_buf,
            &readback_buf,
            out_bytes,
        )
    }

    fn check_scratch_dims(rows: usize, cols: usize) -> Result<(), GpuComputeError> {
        if rows > GPU_SCRATCH_MAX_ELEMS {
            return Err(GpuComputeError::DimensionExceedsScratch {
                dim: rows,
                max_dim: GPU_SCRATCH_MAX_ELEMS,
            });
        }
        if cols > GPU_SCRATCH_MAX_ELEMS {
            return Err(GpuComputeError::DimensionExceedsScratch {
                dim: cols,
                max_dim: GPU_SCRATCH_MAX_ELEMS,
            });
        }
        Ok(())
    }

    /// Upload a dense `f32` matrix persistently to GPU VRAM with a pre-bound compute bind group.
    pub fn upload_f32(
        self: &Arc<Self>,
        matrix: &DenseMatrix,
    ) -> Result<PersistentGpuMatrix, GpuComputeError> {
        Self::check_scratch_dims(matrix.rows(), matrix.cols())?;
        use wgpu::util::DeviceExt;
        let uniforms = MatvecUniforms {
            rows: matrix.rows() as u32,
            cols: matrix.cols() as u32,
            group_size: 1,
            _pad1: 0,
        };
        let mat_f32: Vec<f32> = matrix.as_slice().iter().map(|&v| v as f32).collect();
        let bytes = bytemuck::cast_slice::<f32, u8>(&mat_f32);
        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("persist-f32-dims"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mat_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("persist-f32-weights"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("persist-f32-bg"),
            layout: &self.bgl_f32,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mat_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.scratch_vec_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.scratch_out_buf.as_entire_binding(),
                },
            ],
        });
        Ok(PersistentGpuMatrix {
            engine: Arc::clone(self),
            kind: GpuMatrixKind::Fp32,
            rows: matrix.rows(),
            cols: matrix.cols(),
            hadamard: HadamardSpec::NONE,
            vram_bytes: bytes.len() as u64 + std::mem::size_of::<MatvecUniforms>() as u64,
            bind_group,
            _weight_buffers: vec![uniform_buf, mat_buf],
        })
    }

    /// Upload a packed `FP16` matrix persistently to GPU VRAM (2 bytes/weight, zero FP32 expansion).
    pub fn upload_fp16(
        self: &Arc<Self>,
        matrix: &Fp16Matrix,
    ) -> Result<PersistentGpuMatrix, GpuComputeError> {
        self.upload_packed_16bit(
            GpuMatrixKind::Fp16,
            matrix.rows(),
            matrix.cols(),
            matrix.raw_bits(),
        )
    }

    /// Upload a packed `BF16` matrix persistently to GPU VRAM (2 bytes/weight, zero FP32 expansion).
    pub fn upload_bf16(
        self: &Arc<Self>,
        matrix: &Bf16Matrix,
    ) -> Result<PersistentGpuMatrix, GpuComputeError> {
        self.upload_packed_16bit(
            GpuMatrixKind::Bf16,
            matrix.rows(),
            matrix.cols(),
            matrix.raw_bits(),
        )
    }

    fn upload_packed_16bit(
        self: &Arc<Self>,
        kind: GpuMatrixKind,
        rows: usize,
        cols: usize,
        raw_bits: &[u16],
    ) -> Result<PersistentGpuMatrix, GpuComputeError> {
        Self::check_scratch_dims(rows, cols)?;
        use wgpu::util::DeviceExt;
        let uniforms = MatvecUniforms {
            rows: rows as u32,
            cols: cols as u32,
            group_size: 1,
            _pad1: 0,
        };
        let packed = pack_u16_u32(raw_bits);
        let bytes = bytemuck::cast_slice::<u32, u8>(&packed);
        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("persist-16b-dims"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mat_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("persist-16b-weights"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("persist-16b-bg"),
            layout: &self.bgl_f32,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mat_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.scratch_vec_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.scratch_out_buf.as_entire_binding(),
                },
            ],
        });
        Ok(PersistentGpuMatrix {
            engine: Arc::clone(self),
            kind,
            rows,
            cols,
            hadamard: HadamardSpec::NONE,
            vram_bytes: bytes.len() as u64 + std::mem::size_of::<MatvecUniforms>() as u64,
            bind_group,
            _weight_buffers: vec![uniform_buf, mat_buf],
        })
    }

    /// Upload a compressed `INT8_SYM` matrix persistently to GPU VRAM (~1.03 bytes/weight).
    pub fn upload_int8_sym(
        self: &Arc<Self>,
        matrix: &Int8SymMatrix,
    ) -> Result<PersistentGpuMatrix, GpuComputeError> {
        Self::check_scratch_dims(matrix.rows(), matrix.cols())?;
        let code_bytes: Vec<u8> = matrix.codes().iter().map(|&q| q as u8).collect();
        let packed_codes = pack_bytes_u32(&code_bytes);
        let packed_scales = pack_u16_u32(matrix.scales_f16());
        self.upload_quantized_2buf(
            GpuMatrixKind::Int8Sym,
            matrix.rows(),
            matrix.cols(),
            matrix.group_size(),
            &packed_codes,
            &packed_scales,
        )
    }

    /// Upload a compressed `Q4_0 (INT4_SYM)` matrix persistently to GPU VRAM (~0.56 bytes/weight).
    pub fn upload_int4_sym(
        self: &Arc<Self>,
        matrix: &Int4SymGroupMatrix,
    ) -> Result<PersistentGpuMatrix, GpuComputeError> {
        Self::check_scratch_dims(matrix.rows(), matrix.cols())?;
        let packed_nibbles = pack_bytes_u32(matrix.packed_nibbles());
        let packed_scales = pack_u16_u32(matrix.scales_f16());
        self.upload_quantized_2buf(
            GpuMatrixKind::Int4Sym,
            matrix.rows(),
            matrix.cols(),
            matrix.group_size(),
            &packed_nibbles,
            &packed_scales,
        )
    }

    fn upload_quantized_2buf(
        self: &Arc<Self>,
        kind: GpuMatrixKind,
        rows: usize,
        cols: usize,
        group_size: usize,
        packed_weights: &[u32],
        packed_scales: &[u32],
    ) -> Result<PersistentGpuMatrix, GpuComputeError> {
        use wgpu::util::DeviceExt;
        let uniforms = MatvecUniforms {
            rows: rows as u32,
            cols: cols as u32,
            group_size: group_size as u32,
            _pad1: 0,
        };
        let w_bytes = bytemuck::cast_slice::<u32, u8>(packed_weights);
        let s_bytes = bytemuck::cast_slice::<u32, u8>(packed_scales);
        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("persist-q-dims"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let weights_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("persist-q-weights"),
                contents: w_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let scales_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("persist-q-scales"),
                contents: s_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("persist-q-bg"),
            layout: &self.bgl_quant,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scales_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.scratch_vec_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.scratch_out_buf.as_entire_binding(),
                },
            ],
        });
        Ok(PersistentGpuMatrix {
            engine: Arc::clone(self),
            kind,
            rows,
            cols,
            hadamard: HadamardSpec::NONE,
            vram_bytes: w_bytes.len() as u64
                + s_bytes.len() as u64
                + std::mem::size_of::<MatvecUniforms>() as u64,
            bind_group,
            _weight_buffers: vec![uniform_buf, weights_buf, scales_buf],
        })
    }

    /// Upload a compressed `Q4_1 (INT4_ASYM)` matrix persistently to GPU VRAM (~0.625 bytes/weight).
    /// Interleaves `(scale_f16, min_f16)` into a single `u32` per group for 1-instruction `unpack2x16float`.
    pub fn upload_int4_asym(
        self: &Arc<Self>,
        matrix: &Int4AsymGroupMatrix,
    ) -> Result<PersistentGpuMatrix, GpuComputeError> {
        Self::check_scratch_dims(matrix.rows(), matrix.cols())?;
        let packed_nibbles = pack_bytes_u32(matrix.packed_nibbles());
        let packed_scale_min: Vec<u32> = matrix
            .scales_f16()
            .iter()
            .zip(matrix.mins_f16())
            .map(|(&s, &m)| u32::from(s) | (u32::from(m) << 16))
            .collect();
        self.upload_quantized_2buf(
            GpuMatrixKind::Int4Asym,
            matrix.rows(),
            matrix.cols(),
            matrix.group_size(),
            &packed_nibbles,
            &packed_scale_min,
        )
    }

    /// Upload a compressed `INT3_ASYM` (+ optional Hadamard) matrix persistently to GPU VRAM (~0.41-0.44 bytes/weight).
    pub fn upload_int3_asym(
        self: &Arc<Self>,
        matrix: &Int3AsymGroupMatrix,
    ) -> Result<PersistentGpuMatrix, GpuComputeError> {
        Self::check_scratch_dims(matrix.rows(), matrix.cols())?;
        let packed = pack_bytes_u32(matrix.packed());
        let packed_scale_min: Vec<u32> = matrix
            .scales_f16()
            .iter()
            .zip(matrix.mins_f16())
            .map(|(&s, &m)| u32::from(s) | (u32::from(m) << 16))
            .collect();
        let mut p = self.upload_quantized_2buf(
            GpuMatrixKind::Int3Asym,
            matrix.rows(),
            matrix.cols(),
            matrix.group_size(),
            &packed,
            &packed_scale_min,
        )?;
        p.hadamard = matrix.hadamard();
        Ok(p)
    }

    /// Upload a compressed `INT2_ASYM` (+ optional Hadamard) matrix persistently to GPU VRAM (~0.28-0.31 bytes/weight).
    pub fn upload_int2_asym(
        self: &Arc<Self>,
        matrix: &Int2AsymGroupMatrix,
    ) -> Result<PersistentGpuMatrix, GpuComputeError> {
        Self::check_scratch_dims(matrix.rows(), matrix.cols())?;
        let packed = pack_bytes_u32(matrix.packed());
        let packed_scale_min: Vec<u32> = matrix
            .scales_f16()
            .iter()
            .zip(matrix.mins_f16())
            .map(|(&s, &m)| u32::from(s) | (u32::from(m) << 16))
            .collect();
        let mut p = self.upload_quantized_2buf(
            GpuMatrixKind::Int2Asym,
            matrix.rows(),
            matrix.cols(),
            matrix.group_size(),
            &packed,
            &packed_scale_min,
        )?;
        p.hadamard = matrix.hadamard();
        Ok(p)
    }

    /// Upload a packed base-3 `TERNARY_PACKED` (+ optional Hadamard) matrix persistently to GPU VRAM (~0.21-0.23 bytes/weight).
    pub fn upload_ternary_packed(
        self: &Arc<Self>,
        matrix: &TernaryPackedMatrix,
    ) -> Result<PersistentGpuMatrix, GpuComputeError> {
        Self::check_scratch_dims(matrix.rows(), matrix.cols())?;
        let packed = pack_bytes_u32(matrix.packed());
        let packed_scales = pack_u16_u32(matrix.scales_f16());
        let mut p = self.upload_quantized_2buf(
            GpuMatrixKind::TernaryPacked,
            matrix.rows(),
            matrix.cols(),
            matrix.group_size(),
            &packed,
            &packed_scales,
        )?;
        p.hadamard = matrix.hadamard();
        Ok(p)
    }

    /// Upload a [`QuantizedFactor`] (used by SVD and RMQ Shared-Basis/Core latents) persistently to VRAM.
    pub fn upload_quantized_factor(
        self: &Arc<Self>,
        factor: &QuantizedFactor,
    ) -> Result<PersistentGpuMatrix, GpuComputeError> {
        match factor {
            QuantizedFactor::Fp32(m) => self.upload_f32(m),
            QuantizedFactor::Fp16(m) => self.upload_fp16(m),
            QuantizedFactor::Int8Sym(m) => self.upload_int8_sym(m),
        }
    }

    fn dispatch_and_readback(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        rows: u32,
        out_buf: &wgpu::Buffer,
        readback_buf: &wgpu::Buffer,
        out_bytes: u64,
    ) -> Result<Vec<f64>, GpuComputeError> {
        self.dispatch_and_readback_partial(
            pipeline,
            bind_group,
            rows,
            out_buf,
            readback_buf,
            out_bytes,
        )
    }

    fn dispatch_and_readback_partial(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        rows: u32,
        out_buf: &wgpu::Buffer,
        readback_buf: &wgpu::Buffer,
        out_bytes: u64,
    ) -> Result<Vec<f64>, GpuComputeError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qml-matvec-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("qml-matvec-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(rows.div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(out_buf, 0, readback_buf, 0, out_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback_buf.slice(0..out_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuComputeError::BufferMapFailed(e.to_string()))?
            .map_err(|e| GpuComputeError::BufferMapFailed(format!("{e:?}")))?;

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let result: Vec<f64> = floats.iter().map(|&v| f64::from(v)).collect();
        drop(data);
        readback_buf.unmap();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qml_core::DeterministicRng;

    #[test]
    fn hardware_probe_reports_nonzero_cpu_and_ram() {
        let report = probe_hardware();
        assert!(report.logical_cores >= 1);
        assert!(report.total_ram_bytes > 0);
        let plan = ResourcePreset::Balanced.plan(&report);
        assert!(plan.max_threads >= 1);
        assert!(plan.ram_budget_bytes >= 512 * 1024 * 1024);
    }

    #[test]
    fn wgpu_dense_and_compressed_kernels_match_cpu_reference_when_adapter_available() {
        // Initialize a single WGPU device in-process so concurrent DX12/Vulkan driver
        // initialization across parallel test threads never deadlocks.
        let Ok(engine) = WgpuMatvecEngine::try_new() else {
            // Headless CI runners without a GPU adapter cleanly skip GPU execution.
            return;
        };
        let mut rng = DeterministicRng::new(2026);
        let rows = 128;
        let cols = 96;
        let mat = DenseMatrix::from_flat(
            rows,
            cols,
            (0..rows * cols).map(|_| rng.next_normal() * 0.25).collect(),
        )
        .unwrap();
        let x: Vec<f64> = (0..cols).map(|_| rng.next_normal()).collect();

        // 1. Dense FP32 on GPU vs CPU
        let cpu = mat.matvec(&x).unwrap();
        let gpu = engine.matvec(&mat, &x).unwrap();
        assert_eq!(cpu.len(), gpu.len());
        let max_err = cpu
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-4,
            "WGPU matvec on {} ({}) diverged from CPU: max_err={max_err}",
            engine.adapter_name(),
            engine.backend_name()
        );

        // 2. Direct compressed INT8_SYM on GPU vs CPU
        let q8 = Int8SymMatrix::quantize(&mat, 64).unwrap();
        let cpu8 = q8.matvec(&x).unwrap();
        let gpu8 = engine.matvec_int8_sym(&q8, &x).unwrap();
        let err8 = cpu8
            .iter()
            .zip(&gpu8)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            err8 < 1e-4,
            "WGPU INT8_SYM matvec diverged from CPU: max_err={err8}"
        );

        // 3. Direct compressed Q4_0 (INT4_SYM) on GPU vs CPU
        let q4 = Int4SymGroupMatrix::quantize(&mat, 32).unwrap();
        let cpu4 = q4.matvec(&x).unwrap();
        let gpu4 = engine.matvec_int4_sym(&q4, &x).unwrap();
        let err4 = cpu4
            .iter()
            .zip(&gpu4)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            err4 < 1e-4,
            "WGPU Q4_0 (INT4_SYM) matvec diverged from CPU: max_err={err4}"
        );

        // 4. Persistent GPU-resident compressed operators (FP32, FP16, BF16, INT8, Q4_0, Q4_1,
        //    plus PROJECT RESONANCE INT3+Hadamard, INT2+Hadamard, and Ternary+Hadamard)
        //    reused across multiple input vectors with zero per-call weight uploads.
        let engine = Arc::new(engine);
        let fp16 = Fp16Matrix::quantize(&mat);
        let bf16 = Bf16Matrix::quantize(&mat);
        let q4_1 = Int4AsymGroupMatrix::quantize(&mat, 32).unwrap();
        let had = HadamardSpec::new(64, 0xCAFE);
        let q3_had = Int3AsymGroupMatrix::quantize(&mat, 64, had).unwrap();
        let q2_had = Int2AsymGroupMatrix::quantize(&mat, 64, had).unwrap();
        let tern_had = TernaryPackedMatrix::quantize(&mat, 64, had).unwrap();

        let p_f32 = engine.upload_f32(&mat).unwrap();
        let p_fp16 = engine.upload_fp16(&fp16).unwrap();
        let p_bf16 = engine.upload_bf16(&bf16).unwrap();
        let p_q8 = engine.upload_int8_sym(&q8).unwrap();
        let p_q4_0 = engine.upload_int4_sym(&q4).unwrap();
        let p_q4_1 = engine.upload_int4_asym(&q4_1).unwrap();
        let p_q3 = engine.upload_int3_asym(&q3_had).unwrap();
        let p_q2 = engine.upload_int2_asym(&q2_had).unwrap();
        let p_tern = engine.upload_ternary_packed(&tern_had).unwrap();

        assert!(p_q4_0.vram_bytes() < p_f32.vram_bytes() / 5);
        assert!(p_q2.vram_bytes() < p_q4_0.vram_bytes());
        assert!(p_tern.vram_bytes() < p_q2.vram_bytes());
        assert!(p_q8.vram_bytes() < p_f32.vram_bytes() / 3);
        assert!(p_bf16.vram_bytes() <= p_f32.vram_bytes() / 2 + 64);

        let x2: Vec<f64> = (0..cols).map(|_| rng.next_normal()).collect();
        for probe in [&x, &x2] {
            let checks: [(&str, Vec<f64>, Vec<f64>); 9] = [
                (
                    "FP32",
                    mat.matvec(probe).unwrap(),
                    p_f32.matvec(probe).unwrap(),
                ),
                (
                    "FP16",
                    fp16.matvec(probe).unwrap(),
                    p_fp16.matvec(probe).unwrap(),
                ),
                (
                    "BF16",
                    bf16.matvec(probe).unwrap(),
                    p_bf16.matvec(probe).unwrap(),
                ),
                (
                    "INT8",
                    q8.matvec(probe).unwrap(),
                    p_q8.matvec(probe).unwrap(),
                ),
                (
                    "Q4_0",
                    q4.matvec(probe).unwrap(),
                    p_q4_0.matvec(probe).unwrap(),
                ),
                (
                    "Q4_1",
                    q4_1.matvec(probe).unwrap(),
                    p_q4_1.matvec(probe).unwrap(),
                ),
                (
                    "INT3_HADAMARD",
                    q3_had.matvec(probe).unwrap(),
                    p_q3.matvec(probe).unwrap(),
                ),
                (
                    "INT2_HADAMARD",
                    q2_had.matvec(probe).unwrap(),
                    p_q2.matvec(probe).unwrap(),
                ),
                (
                    "TERNARY_HADAMARD",
                    tern_had.matvec(probe).unwrap(),
                    p_tern.matvec(probe).unwrap(),
                ),
            ];
            for (name, cpu_y, gpu_y) in checks {
                let err = cpu_y
                    .iter()
                    .zip(&gpu_y)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    err < 1e-4,
                    "Persistent GPU {name} matvec diverged from CPU: max_err={err}"
                );
            }
        }
    }
}
