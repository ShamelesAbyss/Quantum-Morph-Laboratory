//! Multi-Pass WGPU 3D Scientific Visualization Renderer (`PrismGpuRenderer`) +
//! Safe 2.5D CPU Fallback Painter.
//!
//! Renders instanced 3D crystalline geometry (with Depth32Float Z-buffering),
//! luminous animated neural filaments, GPU-driven orbital particles, and 2-pass
//! separable Gaussian HDR selective bloom directly inside any egui viewport via
//! `eframe::egui_wgpu::Callback`.

use crate::prism::camera::{OrbitCamera, Vec3};
use crate::prism::scene::{GraphicsSettings, PrismScene};
use crate::prism::shaders::{
    BACKDROP_WGSL, BLOOM_BLUR_WGSL, BLOOM_THRESHOLD_WGSL, COMPOSITE_WGSL, CRYSTAL_WGSL,
    FILAMENT_WGSL, PARTICLE_WGSL,
};
use eframe::egui::{self, Color32, Painter, Rect, Stroke};
use eframe::egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const SCENE_HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MAX_INSTANCES_PER_SHAPE: usize = 4096;
const MAX_FILAMENTS: usize = 4096;
const MAX_PARTICLES: usize = 8192;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    camera_right: [f32; 4],
    camera_up: [f32; 4],
    params: [f32; 4],
    bloom_params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshVertex {
    pos: [f32; 3],
    normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CrystalInstanceGpu {
    pos: [f32; 3],
    scale: [f32; 3],
    rot: [f32; 3],
    color: [f32; 4],
    meta: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FilamentGpu {
    p0: [f32; 3],
    p1: [f32; 3],
    color0: [f32; 4],
    color1: [f32; 4],
    meta: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleGpu {
    center: [f32; 3],
    orbit: [f32; 4],
    shape: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurUniforms {
    texel_dir: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeUniforms {
    params: [f32; 4],
}

struct MeshBuffers {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
}

struct OffscreenTargets {
    width: u32,
    height: u32,
    bloom_w: u32,
    bloom_h: u32,
    scene_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    bloom_a_view: wgpu::TextureView,
    bloom_b_view: wgpu::TextureView,
    threshold_bg: wgpu::BindGroup,
    blur_h_bg: wgpu::BindGroup,
    blur_v_bg: wgpu::BindGroup,
    composite_bg: wgpu::BindGroup,
}

/// Persistent WGPU multi-pass 3D renderer stored once in `eframe`'s `CallbackResources`.
pub struct PrismGpuRenderer {
    frame_uniform_buf: wgpu::Buffer,
    frame_bg: wgpu::BindGroup,
    blur_h_uniform_buf: wgpu::Buffer,
    blur_v_uniform_buf: wgpu::Buffer,
    composite_uniform_buf: wgpu::Buffer,
    linear_sampler: wgpu::Sampler,

    backdrop_pipeline: ComputeOrRender,
    crystal_pipeline: wgpu::RenderPipeline,
    filament_pipeline: wgpu::RenderPipeline,
    particle_pipeline: wgpu::RenderPipeline,
    threshold_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,

    threshold_bgl: wgpu::BindGroupLayout,
    blur_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,

    meshes: [MeshBuffers; 4],
    instance_bufs: [wgpu::Buffer; 4],
    instance_counts: [u32; 4],
    filament_buf: wgpu::Buffer,
    filament_count: u32,
    particle_buf: wgpu::Buffer,
    particle_count: u32,

    targets: Option<OffscreenTargets>,
    bloom_enabled_this_frame: bool,
}

struct ComputeOrRender(wgpu::RenderPipeline);

impl PrismGpuRenderer {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let frame_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-frame-uniforms"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism-frame-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let frame_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-frame-bg"),
            layout: &frame_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_uniform_buf.as_entire_binding(),
            }],
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("prism-linear-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let blur_h_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-blur-h-u"),
            size: std::mem::size_of::<BlurUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blur_v_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-blur-v-u"),
            size: std::mem::size_of::<BlurUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let composite_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-composite-u"),
            size: std::mem::size_of::<CompositeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let samp_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let uni_frag = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let threshold_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism-threshold-bgl"),
            entries: &[tex_entry(0), samp_entry(1)],
        });
        let blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism-blur-bgl"),
            entries: &[uni_frag(0), tex_entry(1), samp_entry(2)],
        });
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism-composite-bgl"),
            entries: &[uni_frag(0), tex_entry(1), tex_entry(2), samp_entry(3)],
        });

        let frame_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-frame-pl"),
            bind_group_layouts: &[&frame_bgl],
            push_constant_ranges: &[],
        });

        // 1. Backdrop pipeline
        let backdrop_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-backdrop-wgsl"),
            source: wgpu::ShaderSource::Wgsl(BACKDROP_WGSL.into()),
        });
        let backdrop_pipeline = ComputeOrRender(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("prism-backdrop-pipe"),
                layout: Some(&frame_pl),
                vertex: wgpu::VertexState {
                    module: &backdrop_mod,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &backdrop_mod,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: SCENE_HDR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            },
        ));

        // 2. Instanced 3D Crystal pipeline (with true Depth32Float Z-buffer)
        let crystal_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-crystal-wgsl"),
            source: wgpu::ShaderSource::Wgsl(CRYSTAL_WGSL.into()),
        });
        let crystal_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prism-crystal-pipe"),
            layout: Some(&frame_pl),
            vertex: wgpu::VertexState {
                module: &crystal_mod,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<MeshVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<CrystalInstanceGpu>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![
                            2 => Float32x3,
                            3 => Float32x3,
                            4 => Float32x3,
                            5 => Float32x4,
                            6 => Float32x4
                        ],
                    },
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &crystal_mod,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_HDR_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let additive_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        };

        // 3. Luminous Neural Filaments pipeline (depth-tested additive emission)
        let filament_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-filament-wgsl"),
            source: wgpu::ShaderSource::Wgsl(FILAMENT_WGSL.into()),
        });
        let filament_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prism-filament-pipe"),
            layout: Some(&frame_pl),
            vertex: wgpu::VertexState {
                module: &filament_mod,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<FilamentGpu>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3,
                        1 => Float32x3,
                        2 => Float32x4,
                        3 => Float32x4,
                        4 => Float32x4
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &filament_mod,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_HDR_FORMAT,
                    blend: Some(additive_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // 4. GPU-Driven Orbital Particle System pipeline
        let particle_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-particle-wgsl"),
            source: wgpu::ShaderSource::Wgsl(PARTICLE_WGSL.into()),
        });
        let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prism-particle-pipe"),
            layout: Some(&frame_pl),
            vertex: wgpu::VertexState {
                module: &particle_mod,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ParticleGpu>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3,
                        1 => Float32x4,
                        2 => Float32x4,
                        3 => Float32x4
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &particle_mod,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_HDR_FORMAT,
                    blend: Some(additive_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // 5. Selective Bloom Threshold + Separable Blur + ACES Composite pipelines
        let threshold_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-bloom-thresh-wgsl"),
            source: wgpu::ShaderSource::Wgsl(BLOOM_THRESHOLD_WGSL.into()),
        });
        let threshold_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-threshold-pl"),
            bind_group_layouts: &[&threshold_bgl],
            push_constant_ranges: &[],
        });
        let threshold_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prism-threshold-pipe"),
            layout: Some(&threshold_pl),
            vertex: wgpu::VertexState {
                module: &threshold_mod,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &threshold_mod,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let blur_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-bloom-blur-wgsl"),
            source: wgpu::ShaderSource::Wgsl(BLOOM_BLUR_WGSL.into()),
        });
        let blur_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-blur-pl"),
            bind_group_layouts: &[&blur_bgl],
            push_constant_ranges: &[],
        });
        let blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prism-blur-pipe"),
            layout: Some(&blur_pl),
            vertex: wgpu::VertexState {
                module: &blur_mod,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blur_mod,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let composite_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-composite-wgsl"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_WGSL.into()),
        });
        let composite_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-composite-pl"),
            bind_group_layouts: &[&composite_bgl],
            push_constant_ranges: &[],
        });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prism-composite-pipe"),
            layout: Some(&composite_pl),
            vertex: wgpu::VertexState {
                module: &composite_mod,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_mod,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Procedural 3D meshes uploaded ONCE to VRAM
        let meshes = [
            upload_mesh(device, "octahedron", &build_octahedron_mesh()),
            upload_mesh(device, "prism-slab", &build_box_mesh()),
            upload_mesh(device, "ico-sphere", &build_uv_sphere_mesh(16, 12)),
            upload_mesh(device, "torus-ring", &build_torus_mesh(48, 10, 1.0, 0.032)),
        ];

        let make_inst_buf = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (MAX_INSTANCES_PER_SHAPE * std::mem::size_of::<CrystalInstanceGpu>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let instance_bufs = [
            make_inst_buf("prism-inst-octa"),
            make_inst_buf("prism-inst-slab"),
            make_inst_buf("prism-inst-sphere"),
            make_inst_buf("prism-inst-torus"),
        ];
        let filament_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-filaments"),
            size: (MAX_FILAMENTS * std::mem::size_of::<FilamentGpu>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let particle_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-particles"),
            size: (MAX_PARTICLES * std::mem::size_of::<ParticleGpu>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            frame_uniform_buf,
            frame_bg,
            blur_h_uniform_buf,
            blur_v_uniform_buf,
            composite_uniform_buf,
            linear_sampler,
            backdrop_pipeline,
            crystal_pipeline,
            filament_pipeline,
            particle_pipeline,
            threshold_pipeline,
            blur_pipeline,
            composite_pipeline,
            threshold_bgl,
            blur_bgl,
            composite_bgl,
            meshes,
            instance_bufs,
            instance_counts: [0; 4],
            filament_buf,
            filament_count: 0,
            particle_buf,
            particle_count: 0,
            targets: None,
            bloom_enabled_this_frame: true,
        }
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let max_dim = device.limits().max_texture_dimension_2d.min(2560);
        let w = width.clamp(16, max_dim);
        let h = height.clamp(16, max_dim);
        if let Some(t) = &self.targets
            && t.width == w
            && t.height == h
        {
            return;
        }
        let bw = (w / 2).max(16);
        let bh = (h / 2).max(16);

        let make_color = |label: &str, tw: u32, th: u32| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: tw,
                    height: th,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: SCENE_HDR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            tex.create_view(&wgpu::TextureViewDescriptor::default())
        };

        let scene_view = make_color("prism-scene-hdr", w, h);
        let bloom_a_view = make_color("prism-bloom-a", bw, bh);
        let bloom_b_view = make_color("prism-bloom-b", bw, bh);
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("prism-depth32"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let threshold_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-thresh-bg"),
            layout: &self.threshold_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        });
        let blur_h_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-blur-h-bg"),
            layout: &self.blur_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.blur_h_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&bloom_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        });
        let blur_v_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-blur-v-bg"),
            layout: &self.blur_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.blur_v_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&bloom_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        });
        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-composite-bg"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.composite_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&bloom_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        });

        self.targets = Some(OffscreenTargets {
            width: w,
            height: h,
            bloom_w: bw,
            bloom_h: bh,
            scene_view,
            depth_view,
            bloom_a_view,
            bloom_b_view,
            threshold_bg,
            blur_h_bg,
            blur_v_bg,
            composite_bg,
        });
    }
}

/// Per-frame paint callback submitted into `egui`'s WGPU render pass.
pub struct PrismViewportCallback {
    pub scene: Arc<PrismScene>,
    pub camera: OrbitCamera,
    pub settings: GraphicsSettings,
    pub time_sec: f32,
    pub rect_px: [u32; 2],
}

impl CallbackTrait for PrismViewportCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(renderer) = callback_resources.get_mut::<PrismGpuRenderer>() else {
            return Vec::new();
        };
        if self.rect_px[0] < 8 || self.rect_px[1] < 8 {
            return Vec::new();
        }
        let scale = self.settings.resolution_scale.clamp(0.45, 1.0);
        let w = ((self.rect_px[0] as f32) * scale).round().max(16.0) as u32;
        let h = ((self.rect_px[1] as f32) * scale).round().max(16.0) as u32;
        renderer.ensure_targets(device, w, h);

        let aspect = (self.rect_px[0] as f32 / self.rect_px[1].max(1) as f32).max(0.05);
        let eye = self.camera.eye();
        let (right, up, _) = self.camera.basis();
        let anim_t = self.time_sec * self.settings.effective_anim_scale();

        let uniforms = FrameUniforms {
            view_proj: self.camera.view_proj(aspect).0,
            camera_pos: [eye.x(), eye.y(), eye.z(), 1.0],
            camera_right: [right.x(), right.y(), right.z(), 0.0],
            camera_up: [up.x(), up.y(), up.z(), 0.0],
            params: [
                anim_t,
                self.scene.energy.clamp(0.0, 1.5),
                self.scene.background as u32 as f32,
                aspect,
            ],
            bloom_params: [
                if self.settings.bloom_enabled {
                    self.settings.bloom_intensity
                } else {
                    0.0
                },
                w as f32,
                h as f32,
                if self.settings.reduced_motion {
                    1.0
                } else {
                    0.0
                },
            ],
        };
        queue.write_buffer(
            &renderer.frame_uniform_buf,
            0,
            bytemuck::bytes_of(&uniforms),
        );

        // Bucket instances by procedural mesh archetype
        let mut buckets: [Vec<CrystalInstanceGpu>; 4] = Default::default();
        for inst in &self.scene.instances {
            let idx = inst.shape as usize;
            if buckets[idx].len() < MAX_INSTANCES_PER_SHAPE {
                buckets[idx].push(CrystalInstanceGpu {
                    pos: inst.position,
                    scale: inst.scale,
                    rot: inst.rotation,
                    color: inst.color,
                    meta: [inst.emission, inst.selected, 0.0, 0.0],
                });
            }
        }
        for (i, bucket) in buckets.iter().enumerate() {
            renderer.instance_counts[i] = bucket.len() as u32;
            if !bucket.is_empty() {
                queue.write_buffer(&renderer.instance_bufs[i], 0, bytemuck::cast_slice(bucket));
            }
        }

        // Upload luminous neural filaments
        let fil_gpu: Vec<FilamentGpu> = self
            .scene
            .filaments
            .iter()
            .take(MAX_FILAMENTS)
            .map(|f| FilamentGpu {
                p0: f.p0,
                p1: f.p1,
                color0: f.color0,
                color1: f.color1,
                meta: [f.thickness, f.pulse_speed, f.pulse_phase, f.intensity],
            })
            .collect();
        renderer.filament_count = fil_gpu.len() as u32;
        if !fil_gpu.is_empty() {
            queue.write_buffer(&renderer.filament_buf, 0, bytemuck::cast_slice(&fil_gpu));
        }

        // Upload GPU-driven orbital particles scaled by user density setting
        let max_p = ((self.scene.particles.len() as f32)
            * self.settings.particle_density.clamp(0.0, 1.0))
        .round() as usize;
        let part_gpu: Vec<ParticleGpu> = self
            .scene
            .particles
            .iter()
            .take(max_p.min(MAX_PARTICLES))
            .map(|p| ParticleGpu {
                center: p.center,
                orbit: [
                    p.orbit_radius,
                    p.inclination,
                    p.azimuth_offset,
                    p.angular_speed,
                ],
                shape: [p.vertical_amp, p.size, p.emission, 0.0],
                color: p.color,
            })
            .collect();
        renderer.particle_count = part_gpu.len() as u32;
        if !part_gpu.is_empty() {
            queue.write_buffer(&renderer.particle_buf, 0, bytemuck::cast_slice(&part_gpu));
        }

        let Some(targets) = &renderer.targets else {
            return Vec::new();
        };

        let blur_h = BlurUniforms {
            texel_dir: [1.35 / targets.bloom_w as f32, 0.0, 0.0, 0.0],
        };
        let blur_v = BlurUniforms {
            texel_dir: [0.0, 1.35 / targets.bloom_h as f32, 0.0, 0.0],
        };
        let comp_u = CompositeUniforms {
            params: [
                if self.settings.bloom_enabled {
                    self.settings.bloom_intensity
                } else {
                    0.0
                },
                0.52,
                0.0,
                0.0,
            ],
        };
        queue.write_buffer(&renderer.blur_h_uniform_buf, 0, bytemuck::bytes_of(&blur_h));
        queue.write_buffer(&renderer.blur_v_uniform_buf, 0, bytemuck::bytes_of(&blur_v));
        queue.write_buffer(
            &renderer.composite_uniform_buf,
            0,
            bytemuck::bytes_of(&comp_u),
        );
        renderer.bloom_enabled_this_frame = self.settings.bloom_enabled;

        // Pass 1–4: Render 3D scene into HDR scene_view + Depth32Float
        {
            let mut pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("prism-3d-hdr-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.scene_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.012,
                            g: 0.015,
                            b: 0.032,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // 1. Procedural Singularity / Observatory backdrop
            pass.set_pipeline(&renderer.backdrop_pipeline.0);
            pass.set_bind_group(0, &renderer.frame_bg, &[]);
            pass.draw(0..3, 0..1);

            // 2. Depth-aware instanced 3D crystals / slabs / spheres / rings
            pass.set_pipeline(&renderer.crystal_pipeline);
            pass.set_bind_group(0, &renderer.frame_bg, &[]);
            for i in 0..4 {
                let count = renderer.instance_counts[i];
                if count > 0 {
                    pass.set_vertex_buffer(0, renderer.meshes[i].vertex_buf.slice(..));
                    pass.set_vertex_buffer(1, renderer.instance_bufs[i].slice(..));
                    pass.set_index_buffer(
                        renderer.meshes[i].index_buf.slice(..),
                        wgpu::IndexFormat::Uint16,
                    );
                    pass.draw_indexed(0..renderer.meshes[i].index_count, 0, 0..count);
                }
            }

            // 3. Luminous animated neural filaments
            if renderer.filament_count > 0 {
                pass.set_pipeline(&renderer.filament_pipeline);
                pass.set_bind_group(0, &renderer.frame_bg, &[]);
                pass.set_vertex_buffer(0, renderer.filament_buf.slice(..));
                pass.draw(0..6, 0..renderer.filament_count);
            }

            // 4. GPU-driven orbital particle halo
            if renderer.particle_count > 0 {
                pass.set_pipeline(&renderer.particle_pipeline);
                pass.set_bind_group(0, &renderer.frame_bg, &[]);
                pass.set_vertex_buffer(0, renderer.particle_buf.slice(..));
                pass.draw(0..6, 0..renderer.particle_count);
            }
        }

        // Pass 5: Selective Gaussian Bloom (Threshold -> H-Blur -> V-Blur)
        if renderer.bloom_enabled_this_frame {
            {
                let mut pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("prism-bloom-thresh"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &targets.bloom_a_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&renderer.threshold_pipeline);
                pass.set_bind_group(0, &targets.threshold_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            {
                let mut pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("prism-bloom-blur-h"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &targets.bloom_b_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&renderer.blur_pipeline);
                pass.set_bind_group(0, &targets.blur_h_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            {
                let mut pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("prism-bloom-blur-v"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &targets.bloom_a_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&renderer.blur_pipeline);
                pass.set_bind_group(0, &targets.blur_v_bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
    ) {
        let Some(renderer) = callback_resources.get::<PrismGpuRenderer>() else {
            return;
        };
        let Some(targets) = &renderer.targets else {
            return;
        };
        render_pass.set_pipeline(&renderer.composite_pipeline);
        render_pass.set_bind_group(0, &targets.composite_bg, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

/// Register the persistent [`PrismGpuRenderer`] in `eframe`'s `wgpu_render_state` at startup.
pub fn init_prism_renderer(cc: &eframe::CreationContext<'_>) -> bool {
    let Some(render_state) = cc.wgpu_render_state.as_ref() else {
        return false;
    };
    let renderer = PrismGpuRenderer::new(&render_state.device, render_state.target_format);
    render_state
        .renderer
        .write()
        .callback_resources
        .insert(renderer);
    true
}

/// Create an egui `PaintCallback` invoking the native WGPU 3D renderer inside `rect`.
pub fn make_wgpu_paint_callback(
    rect: Rect,
    pixels_per_point: f32,
    scene: Arc<PrismScene>,
    camera: OrbitCamera,
    settings: GraphicsSettings,
    time_sec: f32,
) -> egui::PaintCallback {
    let w = (rect.width() * pixels_per_point).round().max(1.0) as u32;
    let h = (rect.height() * pixels_per_point).round().max(1.0) as u32;
    egui_wgpu::Callback::new_paint_callback(
        rect,
        PrismViewportCallback {
            scene,
            camera,
            settings,
            time_sec,
            rect_px: [w, h],
        },
    )
}

/// Graceful 2.5D projected CPU fallback painter used when `GraphicsPreset::Minimal` is selected
/// or if a headless environment has no WGPU render state.
pub fn paint_cpu_fallback(painter: &Painter, rect: Rect, camera: &OrbitCamera, scene: &PrismScene) {
    painter.rect_filled(rect, 8.0, Color32::from_rgb(7, 8, 16));
    // Project filaments
    for edge in &scene.filaments {
        if let (Some((a, _)), Some((b, _))) = (
            camera.project_to_screen(edge.p0, rect),
            camera.project_to_screen(edge.p1, rect),
        ) {
            let col = Color32::from_rgba_unmultiplied(
                (edge.color0[0] * 255.0) as u8,
                (edge.color0[1] * 255.0) as u8,
                (edge.color0[2] * 255.0) as u8,
                ((edge.color0[3] * 180.0).min(255.0)) as u8,
            );
            painter.line_segment(
                [a, b],
                Stroke::new((edge.thickness * 18.0).clamp(0.8, 3.5), col),
            );
        }
    }
    // Depth-sort instances back-to-front
    let mut projected: Vec<(f32, egui::Pos2, f32, Color32, f32)> = scene
        .instances
        .iter()
        .filter_map(|inst| {
            let (pos, z) = camera.project_to_screen(inst.position, rect)?;
            let avg_s = (inst.scale[0] + inst.scale[1] + inst.scale[2]) / 3.0;
            let px_r = (avg_s * rect.height() / (camera.distance * 1.4).max(1.0)).clamp(3.0, 34.0);
            let col = Color32::from_rgba_unmultiplied(
                (inst.color[0] * 255.0) as u8,
                (inst.color[1] * 255.0) as u8,
                (inst.color[2] * 255.0) as u8,
                ((inst.color[3] * 220.0).min(255.0)) as u8,
            );
            Some((z, pos, px_r, col, inst.selected))
        })
        .collect();
    projected.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (_, pos, r, col, sel) in projected {
        painter.circle_filled(pos, r, col);
        if sel > 0.1 {
            painter.circle_stroke(
                pos,
                r + 2.5,
                Stroke::new(1.5_f32, Color32::from_rgb(34, 211, 238)),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Procedural 3D Mesh Generators (100% pure safe Rust, zero external assets)
// ---------------------------------------------------------------------------

fn upload_mesh(
    device: &wgpu::Device,
    label: &str,
    (verts, indices): &(Vec<MeshVertex>, Vec<u16>),
) -> MeshBuffers {
    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("prism-vb-{label}")),
        contents: bytemuck::cast_slice(verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("prism-ib-{label}")),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    MeshBuffers {
        vertex_buf,
        index_buf,
        index_count: indices.len() as u32,
    }
}

fn build_octahedron_mesh() -> (Vec<MeshVertex>, Vec<u16>) {
    let corners = [
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, -1.0, 0.0),
    ];
    let tris: [[usize; 3]; 8] = [
        [0, 2, 1],
        [0, 3, 2],
        [0, 4, 3],
        [0, 1, 4],
        [5, 1, 2],
        [5, 2, 3],
        [5, 3, 4],
        [5, 4, 1],
    ];
    let mut verts = Vec::with_capacity(24);
    let mut idx = Vec::with_capacity(24);
    for [a, b, c] in tris {
        let va = corners[a];
        let vb = corners[b];
        let vc = corners[c];
        let n = vb.sub(va).cross(vc.sub(va)).normalize();
        let base = verts.len() as u16;
        for v in [va, vb, vc] {
            verts.push(MeshVertex {
                pos: v.0,
                normal: n.0,
            });
        }
        idx.extend_from_slice(&[base, base + 1, base + 2]);
    }
    (verts, idx)
}

fn build_box_mesh() -> (Vec<MeshVertex>, Vec<u16>) {
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        (
            [0.0, 0.0, 1.0],
            [
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
        ),
        (
            [0.0, 0.0, -1.0],
            [
                [0.5, -0.5, -0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
            ],
        ),
        (
            [1.0, 0.0, 0.0],
            [
                [0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ],
        ),
        (
            [-1.0, 0.0, 0.0],
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
            ],
        ),
        (
            [0.0, 1.0, 0.0],
            [
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
            ],
        ),
        (
            [0.0, -1.0, 0.0],
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [-0.5, -0.5, 0.5],
            ],
        ),
    ];
    let mut verts = Vec::with_capacity(24);
    let mut idx = Vec::with_capacity(36);
    for (n, corners) in faces {
        let base = verts.len() as u16;
        for pos in corners {
            verts.push(MeshVertex { pos, normal: n });
        }
        idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (verts, idx)
}

fn build_uv_sphere_mesh(segs: usize, rings: usize) -> (Vec<MeshVertex>, Vec<u16>) {
    let mut verts = Vec::new();
    let mut idx = Vec::new();
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let phi = v * std::f32::consts::PI;
        let sp = phi.sin();
        let cp = phi.cos();
        for s in 0..=segs {
            let u = s as f32 / segs as f32;
            let theta = u * std::f32::consts::TAU;
            let x = sp * theta.cos();
            let y = cp;
            let z = sp * theta.sin();
            verts.push(MeshVertex {
                pos: [x, y, z],
                normal: [x, y, z],
            });
        }
    }
    let stride = (segs + 1) as u16;
    for r in 0..rings as u16 {
        for s in 0..segs as u16 {
            let a = r * stride + s;
            let b = a + 1;
            let c = (r + 1) * stride + s;
            let d = c + 1;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    (verts, idx)
}

fn build_torus_mesh(
    major_segs: usize,
    minor_segs: usize,
    major_r: f32,
    minor_r: f32,
) -> (Vec<MeshVertex>, Vec<u16>) {
    let mut verts = Vec::new();
    let mut idx = Vec::new();
    for i in 0..=major_segs {
        let u = (i as f32 / major_segs as f32) * std::f32::consts::TAU;
        let cu = u.cos();
        let su = u.sin();
        for j in 0..=minor_segs {
            let v = (j as f32 / minor_segs as f32) * std::f32::consts::TAU;
            let cv = v.cos();
            let sv = v.sin();
            let rr = major_r + minor_r * cv;
            let pos = [rr * cu, minor_r * sv, rr * su];
            let normal = [cv * cu, sv, cv * su];
            verts.push(MeshVertex { pos, normal });
        }
    }
    let stride = (minor_segs + 1) as u16;
    for i in 0..major_segs as u16 {
        for j in 0..minor_segs as u16 {
            let a = i * stride + j;
            let b = a + 1;
            let c = (i + 1) * stride + j;
            let d = c + 1;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    (verts, idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prism_wgpu_pipelines_and_wgsl_shaders_validate_when_adapter_available() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = instance
            .enumerate_adapters(wgpu::Backends::all())
            .into_iter()
            .next()
        else {
            return;
        };
        // Validate every WGSL shader & pipeline compilation on a real WGPU device!
        let Ok((device, _queue)) = eframe::egui_wgpu::wgpu::Adapter::request_device(
            &adapter,
            &wgpu::DeviceDescriptor {
                label: Some("prism-shader-test"),
                ..Default::default()
            },
            None,
        )
        .await_or_block() else {
            return;
        };
        let mut renderer = PrismGpuRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
        renderer.ensure_targets(&device, 640, 360);
        assert!(renderer.targets.is_some());
    }

    trait BlockOnExt: std::future::Future + Sized {
        fn await_or_block(self) -> Self::Output;
    }

    impl<F: std::future::Future> BlockOnExt for F {
        fn await_or_block(self) -> Self::Output {
            use std::task::{Context, Poll, Waker};
            let waker = Waker::noop();
            let mut cx = Context::from_waker(waker);
            let mut fut = std::pin::pin!(self);
            loop {
                match fut.as_mut().poll(&mut cx) {
                    Poll::Ready(out) => return out,
                    Poll::Pending => std::thread::yield_now(),
                }
            }
        }
    }
}
