//! Host Hardware Observatory & Experimental WGPU Compute Backend (`qml-hardware`).
//!
//! Provides cross-platform CPU/SIMD/RAM telemetry, WGPU GPU adapter discovery,
//! resource preset calculation, and a validated WGSL compute shader matrix-vector
//! multiplication prototype (`y = A * x`) with graceful CPU fallback when no
//! GPU adapter is available (such as on headless CI runners).

use qml_core::{DenseMatrix, MatrixError};
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
    _pad0: u32,
    _pad1: u32,
}

const MATVEC_WGSL: &str = r#"
struct Dims {
    rows: u32,
    cols: u32,
    _pad0: u32,
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

/// Experimental WGPU compute shader engine for dense `f32` matrix-vector multiplication.
pub struct WgpuMatvecEngine {
    adapter_name: String,
    backend_name: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qml-matvec-wgsl"),
            source: wgpu::ShaderSource::Wgsl(MATVEC_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qml-matvec-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qml-matvec-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("qml-matvec-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            adapter_name: info.name,
            backend_name: format!("{:?}", info.backend),
            device,
            queue,
            pipeline,
            bind_group_layout,
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
            _pad0: 0,
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
            label: Some("qml-matvec-bg"),
            layout: &self.bind_group_layout,
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(rows.div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &readback_buf, 0, out_bytes);
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
    fn wgpu_matvec_matches_cpu_reference_when_adapter_available() {
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
    }
}
