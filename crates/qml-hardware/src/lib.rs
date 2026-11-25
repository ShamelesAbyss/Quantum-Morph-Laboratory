//! Host Hardware Observatory & Experimental WGPU Compute Backend (`qml-hardware`).
//!
//! Provides cross-platform CPU/SIMD/RAM telemetry, WGPU GPU adapter discovery,
//! resource preset calculation, and a validated WGSL compute shader matrix-vector
//! multiplication prototype (`y = A * x`) with graceful CPU fallback when no
//! GPU adapter is available (such as on headless CI runners).

use qml_core::{DenseMatrix, MatrixError};
use qml_quant::{Int4SymGroupMatrix, Int8SymMatrix, LinearOperator};
use serde::{Deserialize, Serialize};
use std::fmt;
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
        } else if let Some(first) = gpus.first() {
            format!(
                "WGPU Compute ({} / {}) + Rayon CPU",
                first.name, first.backend
            )
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
}

impl fmt::Display for GpuComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapterAvailable => write!(f, "no WGPU compute adapter available on this host"),
            Self::DeviceRequestFailed(msg) => write!(f, "WGPU device request failed: {msg}"),
            Self::Matrix(err) => write!(f, "matrix error: {err}"),
            Self::BufferMapFailed(msg) => write!(f, "WGPU buffer readback failed: {msg}"),
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

/// Experimental WGPU compute shader engine for dense `f32` and direct compressed
/// `INT8_SYM` / `Q4_0 (INT4_SYM)` matrix-vector multiplication (zero dense reconstruction in VRAM).
pub struct WgpuMatvecEngine {
    adapter_name: String,
    backend_name: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_f32: wgpu::ComputePipeline,
    bgl_f32: wgpu::BindGroupLayout,
    pipeline_int8: wgpu::ComputePipeline,
    pipeline_q4_0: wgpu::ComputePipeline,
    bgl_quant: wgpu::BindGroupLayout,
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
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("qml-wgpu-compute"),
                required_features: wgpu::Features::empty(),
                required_limits:
                    wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| GpuComputeError::DeviceRequestFailed(e.to_string()))?;

        let shader_f32 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qml-matvec-wgsl"),
            source: wgpu::ShaderSource::Wgsl(MATVEC_WGSL.into()),
        });
        let shader_int8 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qml-matvec-int8-wgsl"),
            source: wgpu::ShaderSource::Wgsl(MATVEC_INT8_WGSL.into()),
        });
        let shader_q4_0 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qml-matvec-q4-0-wgsl"),
            source: wgpu::ShaderSource::Wgsl(MATVEC_Q4_0_WGSL.into()),
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

        Ok(Self {
            adapter_name: info.name,
            backend_name: format!("{:?}", info.backend),
            device,
            queue,
            pipeline_f32,
            bgl_f32,
            pipeline_int8,
            pipeline_q4_0,
            bgl_quant,
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

    fn dispatch_and_readback(
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

        let slice = readback_buf.slice(..);
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
    }
}
