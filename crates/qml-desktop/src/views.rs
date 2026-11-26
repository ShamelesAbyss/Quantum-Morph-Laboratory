//! All 10 scientific laboratory views for `qml-desktop`.
//! Every control invokes real Rust laboratory engines (`qml-scope`, `qml-quant`,
//! `qml-forge`, `qml-format`, `qml-runtime`, `qml-tokenizer`, `qml-genesis`,
//! `qml-hardware`). Unsupported/experimental statuses are explicitly labeled.

use crate::prism::scenes::{
    ScopeVisualMode, build_hardware_anatomy_scene, build_qbehavior_dual_scene,
    build_qentangle_constellation_scene, build_qforge_morphogenesis_scene,
    build_qgenesis_crucible_scene, build_qscope_anatomy_scene, build_singularity_scene,
};
use crate::prism::{GraphicsPreset, GraphicsSettings, render_prism_viewport};
use crate::state::{ModelFormat, QmfInspection, save_persisted_state};
use crate::theme::{
    self, AMBER, AMETHYST, BORDER_SUBTLE, CRIMSON, CYAN, EMERALD, TEXT_MUTED, TEXT_PRIMARY,
    ULTRAVIOLET, accent_card, draw_ambient_inference_pulse, draw_qmf_storage_visuals, format_bytes,
    glass_card, metric_tile, pill_badge, spectral_color,
};
use crate::{ForgeMethodChoice, NavTab, QmlDesktopApp};
use eframe::egui::{self, Color32, RichText, Stroke, Vec2};
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoints, Points};
use qml_format::{QmfRepresentation, SharedLatentEncoding};
use qml_genesis::GenesisProfile;
use qml_hardware::ResourcePreset;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn export_json_report<T: serde::Serialize>(
    ui: &mut egui::Ui,
    button_label: &str,
    default_filename: &str,
    value: &T,
) -> Option<String> {
    if ui.button(button_label).clicked()
        && let Some(path) = rfd::FileDialog::new()
            .add_filter("JSON Report", &["json"])
            .set_file_name(default_filename)
            .save_file()
    {
        return match serde_json::to_string_pretty(value)
            .map_err(|e| e.to_string())
            .and_then(|json| std::fs::write(&path, json).map_err(|e| e.to_string()))
        {
            Ok(()) => Some(format!("Exported JSON report to {}", path.display())),
            Err(err) => Some(format!("Failed to export JSON report: {err}")),
        };
    }
    None
}

fn latent_encoding_bytes(enc: &SharedLatentEncoding) -> u64 {
    match enc {
        SharedLatentEncoding::Fp32 { data } | SharedLatentEncoding::Fp16 { data } => data.len,
        SharedLatentEncoding::Int8Sym { scales, codes, .. } => scales.len + codes.len,
    }
}

fn representation_summary(rep: &QmfRepresentation) -> (&'static str, u64) {
    match rep {
        QmfRepresentation::DenseF32 { data } => ("DENSE_F32", data.len),
        QmfRepresentation::DenseF16 { data } => ("FP16", data.len),
        QmfRepresentation::DenseBf16 { data } => ("BF16", data.len),
        QmfRepresentation::Int8SymGroup { scales, codes, .. } => {
            ("INT8_SYM", scales.len + codes.len)
        }
        QmfRepresentation::Int4SymGroup {
            scales, nibbles, ..
        } => ("INT4_SYM (Q4_0)", scales.len + nibbles.len),
        QmfRepresentation::Int4AsymGroup {
            scales,
            mins,
            nibbles,
            ..
        } => ("INT4_ASYM (Q4_1)", scales.len + mins.len + nibbles.len),
        QmfRepresentation::LowRankSvd { left, right, .. } => (
            "LOW_RANK_SVD",
            latent_encoding_bytes(left) + latent_encoding_bytes(right),
        ),
        QmfRepresentation::RmqSharedBasis {
            core,
            sparse_residual,
            ..
        } => {
            let sp = sparse_residual
                .as_ref()
                .map(|s| s.indices.len + s.values_f16.len)
                .unwrap_or(0);
            ("RMQ_SHARED_BASIS", latent_encoding_bytes(core) + sp)
        }
        QmfRepresentation::RmqSharedCore {
            left,
            right,
            sparse_residual,
            ..
        } => {
            let sp = sparse_residual
                .as_ref()
                .map(|s| s.indices.len + s.values_f16.len)
                .unwrap_or(0);
            (
                "RMQ_SHARED_CORE",
                latent_encoding_bytes(left) + latent_encoding_bytes(right) + sp,
            )
        }
    }
}

fn section_header(ui: &mut egui::Ui, title: &str, subtitle: &str, badge: Option<(&str, Color32)>) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(title).color(TEXT_PRIMARY).size(22.0).strong());
        if let Some((text, color)) = badge {
            ui.add_space(6.0);
            pill_badge(ui, text, color);
        }
    });
    ui.label(RichText::new(subtitle).color(TEXT_MUTED).size(13.0));
    ui.add_space(10.0);
}

pub fn show_command_center(app: &mut QmlDesktopApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    let time_sec = app.start_time.elapsed().as_secs_f32();
    let dt = (1.0 / app.smoothed_fps.max(15.0)).clamp(0.004, 0.05);
    let gpu_name = app.hardware.gpus.first().map(|g| g.name.as_str());
    let scene = build_singularity_scene(
        time_sec,
        app.is_busy(),
        app.busy_label.as_deref(),
        app.persisted.library.len(),
        gpu_name,
    );

    let viewport_h = if app.fullscreen_3d {
        (ui.available_height() - 12.0).max(520.0)
    } else {
        420.0
    };
    let w = ui.available_width();
    let res = render_prism_viewport(
        ui,
        ctx,
        "cmd_singularity",
        Vec2::new(w, viewport_h),
        &mut app.cam_singularity,
        scene,
        &app.persisted.graphics,
        app.wgpu_ready,
        time_sec,
        dt,
        None,
        app.fullscreen_3d,
    );
    if res.fullscreen_toggled {
        app.fullscreen_3d = !app.fullscreen_3d;
    }
    if let Some(clicked) = res.clicked_id {
        match clicked.as_str() {
            "qscope" => app.active_tab = NavTab::QScope,
            "qentangle" => app.active_tab = NavTab::QEntangle,
            "qforge" => app.active_tab = NavTab::QForge,
            "qgenesis" => app.active_tab = NavTab::QGenesis,
            "qcore" => app.active_tab = NavTab::QCore,
            "qbehavior" => app.active_tab = NavTab::QBehavior,
            "qmf" => app.active_tab = NavTab::QmfExplorer,
            "library" => app.active_tab = NavTab::ModelLibrary,
            "rmq" => app.active_tab = NavTab::Benchmarks,
            "hardware" => app.active_tab = NavTab::SystemObservatory,
            _ => {}
        }
    }
    if app.fullscreen_3d {
        return;
    }

    ui.add_space(12.0);

    // Top real-telemetry metric strip (symmetric 4-column grid)
    ui.columns(4, |cols| {
        metric_tile(
            &mut cols[0],
            "LOCAL LIBRARY",
            &format!("{} Artifacts", app.persisted.library.len()),
            "Safetensors, GGUF & QMF (read-only)",
            CYAN,
        );
        metric_tile(
            &mut cols[1],
            "COMPUTE HOST",
            &format!("{} Threads", app.hardware.logical_cores),
            &app.hardware.cpu_brand,
            AMETHYST,
        );
        let gpu_val = app
            .hardware
            .gpus
            .first()
            .map(|g| g.name.clone())
            .unwrap_or_else(|| "CPU Only".to_owned());
        metric_tile(
            &mut cols[2],
            "PRIMARY ACCELERATOR",
            &gpu_val,
            &format!("{} WGPU adapter(s) detected", app.hardware.gpus.len()),
            EMERALD,
        );
        metric_tile(
            &mut cols[3],
            "SYSTEM MEMORY",
            &format!(
                "{:.1} / {:.1} GiB",
                app.hardware.available_ram_bytes as f64 / 1073741824.0,
                app.hardware.total_ram_bytes as f64 / 1073741824.0
            ),
            app.persisted.resource_preset.label(),
            ULTRAVIOLET,
        );
    });

    ui.add_space(12.0);

    // Quick-launch laboratory portals
    accent_card().show(ui, |ui| {
        ui.label(
            RichText::new("DIRECT OBSERVATORY PORTALS")
                .color(CYAN)
                .size(11.5)
                .strong(),
        );
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            if ui.button("Import Local Model File...").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("Neural Models", &["safetensors", "gguf", "qmf"])
                    .pick_file()
            {
                app.add_model_file(path);
            }
            if ui.button("Scan Model Folder...").clicked()
                && let Some(dir) = rfd::FileDialog::new().pick_folder()
            {
                app.scan_directory(dir);
            }
            if ui.button("Q-Scope 3D Anatomy").clicked() {
                app.active_tab = NavTab::QScope;
            }
            if ui.button("Q-Forge Morphogenesis").clicked() {
                app.active_tab = NavTab::QForge;
            }
            if ui.button("Q-Genesis Crucible").clicked() {
                app.active_tab = NavTab::QGenesis;
            }
            if ui.button("Q-Core Inference Nexus").clicked() {
                app.active_tab = NavTab::QCore;
            }
            if ui.button("Q-Behavior Observatory").clicked() {
                app.active_tab = NavTab::QBehavior;
            }
            if ui
                .add_enabled(!app.is_busy(), egui::Button::new("Run RMQ Ablation Demo"))
                .clicked()
            {
                app.active_tab = NavTab::Benchmarks;
                app.start_ablation_job(ctx.clone());
            }
        });
    });

    ui.add_space(12.0);

    // Two-column overview: Subsystems status + Activity Log
    ui.columns(2, |cols| {
        glass_card().show(&mut cols[0], |ui| {
            ui.label(
                RichText::new("LABORATORY SUBSYSTEM STATUS (13 SAFE-RUST CRATES)")
                    .color(AMETHYST)
                    .size(11.5)
                    .strong(),
            );
            ui.add_space(6.0);
            let rows = [
                (
                    "Q-Scope & Q-Entangle",
                    "3D Neural Anatomy, streaming stats, SVD & Grassmannian constellation",
                    "ACTIVE",
                    EMERALD,
                ),
                (
                    "Q-Forge Pipeline",
                    "Atomic out-of-core Safetensors -> QMF v1 + two-pass streaming MAP",
                    "ACTIVE",
                    EMERALD,
                ),
                (
                    "Q-Core GPU Hybrid",
                    "Persistent VRAM compressed inference (FP16/BF16/INT8/Q4_0/Q4_1/RMQ)",
                    "ACTIVE",
                    EMERALD,
                ),
                (
                    "Q-Behavior Observatory",
                    "Layer activation trajectories, RSA & reversible causal interventions",
                    "ACTIVE",
                    CYAN,
                ),
                (
                    "Q-Genesis Crucible",
                    "Role-stratified 3D NSGA-II evolution + Genome-to-QMF v1 export",
                    "EXPERIMENTAL",
                    AMBER,
                ),
                (
                    "PRISM 3D Renderer",
                    "Multi-pass WGPU HDR instanced geometry + separable Gaussian bloom",
                    "ACTIVE",
                    ULTRAVIOLET,
                ),
            ];
            for (name, desc, status, col) in rows {
                ui.horizontal(|ui| {
                    pill_badge(ui, status, col);
                    ui.label(RichText::new(name).color(TEXT_PRIMARY).strong());
                    ui.label(RichText::new(desc).color(TEXT_MUTED).size(11.5));
                });
                ui.add_space(2.0);
            }
        });

        glass_card().show(&mut cols[1], |ui| {
            ui.label(
                RichText::new("LABORATORY TELEMETRY LOG")
                    .color(CYAN)
                    .size(11.5)
                    .strong(),
            );
            ui.add_space(6.0);
            egui::ScrollArea::vertical()
                .max_height(210.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in app.activity_log.iter().rev().take(25).rev() {
                        ui.label(RichText::new(line).monospace().size(11.5).color(TEXT_MUTED));
                    }
                });
        });
    });
}

pub fn show_model_library(app: &mut QmlDesktopApp, ui: &mut egui::Ui) {
    section_header(
        ui,
        "Local Model Library",
        "Manage local Safetensors, GGUF, and QMF artifacts. Source model files are always opened strictly read-only.",
        Some(("READ-ONLY SAFE", EMERALD)),
    );

    glass_card().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("+ Import Model File (.safetensors / .gguf / .qmf)")
                .clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("Model Artifacts", &["safetensors", "gguf", "qmf"])
                    .pick_file()
            {
                app.add_model_file(path);
            }
            if ui.button("Scan Custom Directory...").clicked()
                && let Some(dir) = rfd::FileDialog::new().pick_folder()
            {
                app.scan_directory(dir);
            }
            if ui
                .button("Scan Standard Local Caches (HF / LM Studio / Ollama)")
                .clicked()
            {
                let dirs = crate::state::default_discovery_directories();
                if dirs.is_empty() {
                    app.log("No standard local model cache directories found.");
                } else {
                    for d in dirs {
                        app.scan_directory(d);
                    }
                }
            }
        });
    });

    ui.add_space(10.0);

    if app.persisted.library.is_empty() {
        glass_card().show(ui, |ui| {
            ui.label(
                RichText::new("No local models imported yet. Click 'Import Model File' or 'Scan Custom Directory' to add Safetensors, GGUF, or QMF artifacts.")
                    .color(TEXT_MUTED),
            );
        });
        return;
    }

    let mut action_scope: Option<PathBuf> = None;
    let mut action_forge: Option<PathBuf> = None;
    let mut action_qmf: Option<PathBuf> = None;
    let mut action_infer: Option<(PathBuf, Option<PathBuf>, Option<PathBuf>)> = None;
    let mut remove_idx: Option<usize> = None;

    for (idx, item) in app.persisted.library.iter().enumerate() {
        glass_card().show(ui, |ui| {
            ui.horizontal(|ui| {
                let fmt_col = match item.format {
                    ModelFormat::Safetensors => CYAN,
                    ModelFormat::Gguf => AMBER,
                    ModelFormat::Qmf => ULTRAVIOLET,
                };
                pill_badge(ui, item.format.label(), fmt_col);
                ui.label(
                    RichText::new(&item.name)
                        .color(TEXT_PRIMARY)
                        .size(15.0)
                        .strong(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Remove").clicked() {
                        remove_idx = Some(idx);
                    }
                    match item.format {
                        ModelFormat::Safetensors => {
                            if ui.small_button("Transform in Q-Forge").clicked() {
                                action_forge = Some(item.path.clone());
                            }
                            if ui.small_button("Inspect in Q-Scope").clicked() {
                                action_scope = Some(item.path.clone());
                            }
                        }
                        ModelFormat::Qmf => {
                            if ui.small_button("Load in Q-Core").clicked() {
                                action_infer = Some((
                                    item.path.clone(),
                                    item.config_path.clone(),
                                    item.tokenizer_path.clone(),
                                ));
                            }
                            if ui.small_button("Explore QMF").clicked() {
                                action_qmf = Some(item.path.clone());
                            }
                        }
                        ModelFormat::Gguf => {
                            ui.label(
                                RichText::new("Metadata Inspection Supported")
                                    .color(TEXT_MUTED)
                                    .size(11.5),
                            );
                        }
                    }
                });
            });
            ui.add_space(4.0);
            ui.label(
                RichText::new(format!(
                    "{}  •  {} tensors  •  {}",
                    format_bytes(item.file_bytes),
                    item.tensor_count,
                    item.detail
                ))
                .color(TEXT_MUTED)
                .size(12.0),
            );
            ui.label(
                RichText::new(item.path.display().to_string())
                    .monospace()
                    .size(11.0)
                    .color(Color32::from_rgb(112, 124, 160)),
            );
        });
        ui.add_space(6.0);
    }

    if let Some(i) = remove_idx {
        app.persisted.library.remove(i);
        save_persisted_state(&app.persisted);
    }
    if let Some(p) = action_scope {
        app.scope_source_path = p.display().to_string();
        app.active_tab = NavTab::QScope;
    }
    if let Some(p) = action_forge {
        app.forge_source_path = p.display().to_string();
        if let Some(parent) = p.parent() {
            app.forge_output_path = parent.join("morphed-output.qmf").display().to_string();
        }
        app.active_tab = NavTab::QForge;
    }
    if let Some(p) = action_qmf {
        app.qmf_inspect_path = p.display().to_string();
        app.qmf_inspection = QmfInspection::inspect_file(&p).ok();
        app.active_tab = NavTab::QmfExplorer;
    }
    if let Some((qmf, cfg, tok)) = action_infer {
        app.infer_qmf_path = qmf.display().to_string();
        if let Some(c) = cfg {
            app.infer_config_path = c.display().to_string();
        }
        if let Some(t) = tok {
            app.infer_tokenizer_path = t.display().to_string();
        }
        app.active_tab = NavTab::QCore;
    }
}

pub fn show_qscope(app: &mut QmlDesktopApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    section_header(
        ui,
        "Q-Scope Neural Model Observatory",
        "Streaming distributional statistics, outlier crest factors, quantization susceptibility, and randomized SVD spectra.",
        Some(("REAL ENGINE", CYAN)),
    );

    glass_card().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("Safetensors File:");
            ui.add(
                egui::TextEdit::singleline(&mut app.scope_source_path)
                    .desired_width(480.0)
                    .hint_text("Select a .safetensors model file..."),
            );
            if ui.button("Browse...").clicked()
                && let Some(p) = rfd::FileDialog::new()
                    .add_filter("Safetensors", &["safetensors"])
                    .pick_file()
            {
                app.scope_source_path = p.display().to_string();
            }
            ui.label("SVD Rank:");
            ui.add(egui::DragValue::new(&mut app.scope_svd_rank).range(4..=48));
            if ui
                .add_enabled(!app.is_busy(), egui::Button::new("Run Q-Scope Analysis"))
                .clicked()
            {
                app.start_scope_job(ctx.clone());
            }
            if let Some(rep) = &app.scope_report
                && let Some(msg) =
                    export_json_report(ui, "Export JSON...", "qscope-report.json", rep)
            {
                app.log(msg);
            }
        });
    });

    ui.add_space(10.0);

    // 6-Mode 3D Neural Anatomy Observatory (interactive WGPU crystal picking)
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new("3D Anatomy Mode:").color(CYAN).strong());
        for mode in ScopeVisualMode::ALL {
            ui.selectable_value(&mut app.scope_visual_mode, mode, mode.label());
        }
    });
    ui.add_space(6.0);

    let time_sec = app.start_time.elapsed().as_secs_f32();
    let dt = (1.0 / app.smoothed_fps.max(15.0)).clamp(0.004, 0.05);
    let scope_scene = build_qscope_anatomy_scene(
        app.scope_report.as_ref(),
        app.scope_visual_mode,
        app.scope_selected_tensor,
        time_sec,
    );
    let sel_str = app.scope_selected_tensor.to_string();
    let v_h = if app.fullscreen_3d {
        (ui.available_height() - 12.0).max(520.0)
    } else {
        360.0
    };
    let v_res = render_prism_viewport(
        ui,
        ctx,
        "qscope_3d_anatomy",
        Vec2::new(ui.available_width(), v_h),
        &mut app.cam_qscope,
        scope_scene,
        &app.persisted.graphics,
        app.wgpu_ready,
        time_sec,
        dt,
        Some(&sel_str),
        app.fullscreen_3d,
    );
    if v_res.fullscreen_toggled {
        app.fullscreen_3d = !app.fullscreen_3d;
    }
    if let Some(cid) = v_res.clicked_id
        && let Ok(idx) = cid.parse::<usize>()
    {
        app.scope_selected_tensor = idx;
    }
    if app.fullscreen_3d {
        return;
    }
    ui.add_space(10.0);

    let Some(report) = &app.scope_report else {
        glass_card().show(ui, |ui| {
            ui.label(
                RichText::new("No Q-Scope report loaded yet. Choose a local .safetensors model and click 'Run Q-Scope Analysis' to replace the illustrative 3D preview with real tensor telemetry.")
                    .color(TEXT_MUTED),
            );
        });
        return;
    };

    ui.horizontal_wrapped(|ui| {
        metric_tile(
            ui,
            "TOTAL PARAMETERS",
            &format!(
                "{:.2} M",
                report.summary.total_parameters as f64 / 1_000_000.0
            ),
            &format!(
                "{} tensors ({} SVD)",
                report.summary.total_tensors, report.summary.svd_analyzed_matrices
            ),
            CYAN,
        );
        metric_tile(
            ui,
            "MEAN CROSS-LAYER AFFINITY",
            &format!(
                "{:.4}",
                report.summary.mean_cross_layer_shared_score.unwrap_or(0.0)
            ),
            &format!("{} pairwise subspaces", report.cross_layer_relations.len()),
            ULTRAVIOLET,
        );
        if let (Some(peak_name), Some(peak_score)) = (
            &report.summary.max_quantization_susceptibility_tensor,
            report.summary.max_quantization_susceptibility_score,
        ) {
            metric_tile(
                ui,
                "PEAK SUSCEPTIBILITY",
                &format!("{peak_score:.2}"),
                peak_name,
                AMBER,
            );
        }
    });

    ui.add_space(10.0);

    ui.columns(2, |cols| {
        // Left: searchable tensor list
        glass_card().show(&mut cols[0], |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("TENSOR INSPECTOR").color(AMETHYST).strong());
                ui.add(
                    egui::TextEdit::singleline(&mut app.scope_filter)
                        .hint_text("Filter by layer/role..."),
                );
            });
            ui.add_space(6.0);
            let filter_lower = app.scope_filter.to_ascii_lowercase();
            egui::ScrollArea::vertical()
                .max_height(420.0)
                .show(ui, |ui| {
                    for (idx, t) in report.tensors.iter().enumerate() {
                        if !filter_lower.is_empty()
                            && !t.name.to_ascii_lowercase().contains(&filter_lower)
                        {
                            continue;
                        }
                        let selected = app.scope_selected_tensor == idx;
                        let (susc, crest) = t
                            .stats
                            .as_ref()
                            .map(|s| (s.quantization_susceptibility_score, s.crest_factor))
                            .unwrap_or((0.0, 0.0));
                        let label = format!(
                            "{}  [{:?}]  susc={susc:.1}  crest={crest:.1}",
                            t.name, t.shape
                        );
                        if ui
                            .selectable_label(selected, RichText::new(label).monospace().size(11.5))
                            .clicked()
                        {
                            app.scope_selected_tensor = idx;
                        }
                    }
                });
        });

        // Right: live plots for selected tensor
        glass_card().show(&mut cols[1], |ui| {
            if let Some(t) = report.tensors.get(app.scope_selected_tensor) {
                ui.label(RichText::new(&t.name).color(CYAN).size(14.0).strong());
                if let Some(stats) = &t.stats {
                    ui.label(
                        RichText::new(format!(
                            "DType: {} | Shape: {:?} | Role: {:?} | RMS: {:.5} | AbsMax: {:.4} | >6σ Outliers: {} | Entropy: {:.2} bits",
                            t.dtype,
                            t.shape,
                            t.role,
                            stats.rms,
                            stats.absmax,
                            stats.outlier_6sigma_count,
                            stats.shannon_entropy_bits
                        ))
                        .color(TEXT_MUTED)
                        .size(11.5),
                    );
                    ui.add_space(8.0);

                    // 32-bin weight distribution histogram
                    ui.label(
                        RichText::new("Weight Value Distribution (32-bin histogram)")
                            .color(TEXT_PRIMARY)
                            .size(12.0),
                    );
                    let bars: Vec<Bar> = stats
                        .histogram_counts
                        .iter()
                        .enumerate()
                        .map(|(i, &c)| Bar::new(i as f64, c as f64).width(0.85).fill(ULTRAVIOLET))
                        .collect();
                    Plot::new("qscope_histogram_plot")
                        .height(170.0)
                        .show_axes([true, true])
                        .show(ui, |plot_ui| {
                            plot_ui.bar_chart(BarChart::new(bars).name("Weight Bins"));
                        });
                }

                ui.add_space(8.0);

                // Singular value spectrum if available
                if let Some(svd) = &t.matrix_structure {
                    ui.label(
                        RichText::new(format!(
                            "Singular Value Spectrum (σ₁={:.3}, stable_rank={:.2}, eff_rank={:.2})",
                            svd.spectral_norm, svd.stable_rank, svd.effective_rank
                        ))
                        .color(TEXT_PRIMARY)
                        .size(12.0),
                    );
                    let pts: PlotPoints<'_> = svd
                        .singular_values
                        .iter()
                        .enumerate()
                        .map(|(i, &sv)| [(i + 1) as f64, sv])
                        .collect();
                    Plot::new("qscope_svd_plot")
                        .height(170.0)
                        .show(ui, |plot_ui| {
                            plot_ui.line(
                                Line::new(pts)
                                    .name("Singular Values")
                                    .color(CYAN)
                                    .width(2.2_f32),
                            );
                        });
                } else {
                    ui.label(
                        RichText::new("SVD spectrum not computed for this tensor (1D norm or above SVD tensor cap).")
                            .color(TEXT_MUTED)
                            .size(11.5),
                    );
                }
            }
        });
    });
}

pub fn show_qentangle(app: &mut QmlDesktopApp, ui: &mut egui::Ui) {
    section_header(
        ui,
        "Q-Entangle — The Latent Constellation",
        "Spatial 3D visualization of classical cross-layer Grassmannian subspace affinity (not physical quantum entanglement).",
        Some(("GRASSMANNIAN CONSTELLATION", CYAN)),
    );

    ui.horizontal_wrapped(|ui| {
        ui.selectable_value(&mut app.entangle_view_mode, 0, "3D Latent Constellation");
        ui.selectable_value(&mut app.entangle_view_mode, 1, "2D Affinity Heatmap");
        ui.selectable_value(&mut app.entangle_view_mode, 2, "RMQ Cluster Overview");
        ui.selectable_value(&mut app.entangle_view_mode, 3, "Pair Comparison");
        ui.separator();
        ui.label("Min Affinity Filter:");
        ui.add(egui::Slider::new(&mut app.entangle_threshold, 0.0..=0.65).fixed_decimals(2));
    });
    ui.add_space(8.0);

    if app.entangle_view_mode == 0 {
        let ctx = ui.ctx().clone();
        let time_sec = app.start_time.elapsed().as_secs_f32();
        let dt = (1.0 / app.smoothed_fps.max(15.0)).clamp(0.004, 0.05);
        let scene = build_qentangle_constellation_scene(
            app.scope_report.as_ref(),
            app.entangle_selected_pair,
            app.entangle_threshold,
            time_sec,
        );
        let sel_key = app.entangle_selected_pair.map(|i| format!("pair:{i}"));
        let v_h = if app.fullscreen_3d {
            (ui.available_height() - 12.0).max(520.0)
        } else {
            420.0
        };
        let res = render_prism_viewport(
            ui,
            &ctx,
            "qentangle_constellation",
            Vec2::new(ui.available_width(), v_h),
            &mut app.cam_qentangle,
            scene,
            &app.persisted.graphics,
            app.wgpu_ready,
            time_sec,
            dt,
            sel_key.as_deref(),
            app.fullscreen_3d,
        );
        if res.fullscreen_toggled {
            app.fullscreen_3d = !app.fullscreen_3d;
        }
        if let Some(cid) = res.clicked_id
            && let Some(idx_s) = cid.strip_prefix("pair:")
            && let Ok(idx) = idx_s.parse::<usize>()
        {
            app.entangle_selected_pair = Some(idx);
        }
        if app.fullscreen_3d {
            return;
        }
        ui.add_space(10.0);
    }

    let Some(report) = &app.scope_report else {
        glass_card().show(ui, |ui| {
            ui.label(
                RichText::new("Run Q-Scope on a Safetensors model first to populate cross-layer Grassmannian subspace affinity measurements.")
                    .color(TEXT_MUTED),
            );
            if ui.button("Go to Q-Scope Observatory").clicked() {
                app.active_tab = NavTab::QScope;
            }
        });
        return;
    };

    if report.cross_layer_relations.is_empty() {
        glass_card().show(ui, |ui| {
            ui.label(
                RichText::new("No cross-layer pairs were found in the current Q-Scope report (requires multiple 2D matrices of matching shape).")
                    .color(TEXT_MUTED),
            );
        });
        return;
    }

    if let Some(sel_i) = app.entangle_selected_pair
        && let Some(rel) = report.cross_layer_relations.get(sel_i)
    {
        accent_card().show(ui, |ui| {
            ui.label(
                RichText::new(format!(
                    "SELECTED RELATIONSHIP: {}  <->  {}",
                    rel.tensor_a, rel.tensor_b
                ))
                .color(CYAN)
                .strong(),
            );
            ui.horizontal_wrapped(|ui| {
                metric_tile(
                    ui,
                    "SHARED AFFINITY",
                    &format!("{:.4}", rel.shared_structure_score),
                    "Mean(Left, Right)",
                    CYAN,
                );
                metric_tile(
                    ui,
                    "LEFT SUBSPACE",
                    &format!("{:.4}", rel.left_subspace_affinity),
                    "Column-space Grassmannian",
                    ULTRAVIOLET,
                );
                metric_tile(
                    ui,
                    "RIGHT SUBSPACE",
                    &format!("{:.4}", rel.right_subspace_affinity),
                    "Row-space Grassmannian",
                    AMETHYST,
                );
                metric_tile(
                    ui,
                    "ELEMENT COSINE",
                    &format!("{:.4}", rel.weight_cosine_similarity),
                    "Direct weight correlation",
                    EMERALD,
                );
            });
        });
        ui.add_space(8.0);
    }

    if app.entangle_view_mode == 2 {
        // RMQ Cluster Overview grouped by LayerRole
        let mut by_role: BTreeMap<&'static str, (usize, f64, f64)> = BTreeMap::new();
        for rel in &report.cross_layer_relations {
            let e = by_role.entry(rel.role.as_str()).or_insert((0, 0.0, 0.0));
            e.0 += 1;
            e.1 += rel.shared_structure_score;
            e.2 = e.2.max(rel.shared_structure_score);
        }
        glass_card().show(ui, |ui| {
            ui.label(
                RichText::new("RMQ CLUSTER ELIGIBILITY BY TRANSFORMER ROLE")
                    .color(CYAN)
                    .strong(),
            );
            ui.add_space(6.0);
            for (role, (cnt, sum, peak)) in by_role {
                let mean = sum / cnt.max(1) as f64;
                ui.horizontal(|ui| {
                    pill_badge(
                        ui,
                        if peak >= 0.25 {
                            "RMQ CANDIDATE"
                        } else {
                            "LOW AFFINITY"
                        },
                        if peak >= 0.25 { EMERALD } else { TEXT_MUTED },
                    );
                    ui.label(RichText::new(role).monospace().color(TEXT_PRIMARY).strong());
                    ui.label(
                        RichText::new(format!("pairs={cnt}  •  mean={mean:.4}  •  peak={peak:.4}"))
                            .color(TEXT_MUTED),
                    );
                });
            }
        });
        return;
    }

    // Build an interactive heatmap of the top layers participating in relationships
    let mut unique_names: Vec<String> = Vec::new();
    for rel in &report.cross_layer_relations {
        if !unique_names.contains(&rel.tensor_a) && unique_names.len() < 16 {
            unique_names.push(rel.tensor_a.clone());
        }
        if !unique_names.contains(&rel.tensor_b) && unique_names.len() < 16 {
            unique_names.push(rel.tensor_b.clone());
        }
    }
    let n = unique_names.len();

    ui.columns(2, |cols| {
        glass_card().show(&mut cols[0], |ui| {
            ui.label(
                RichText::new("SUBSPACE AFFINITY HEATMAP (OBSIDIAN -> ULTRAVIOLET -> CYAN)")
                    .color(CYAN)
                    .size(12.0)
                    .strong(),
            );
            ui.add_space(6.0);
            let cell = 22.0_f32;
            let total = (n as f32) * cell;
            let (rect, _resp) = ui.allocate_exact_size(Vec2::splat(total.max(120.0)), egui::Sense::hover());
            let painter = ui.painter();
            for r in 0..n {
                for c in 0..n {
                    let score = if r == c {
                        1.0
                    } else {
                        report
                            .cross_layer_relations
                            .iter()
                            .find(|rel| {
                                (rel.tensor_a == unique_names[r] && rel.tensor_b == unique_names[c])
                                    || (rel.tensor_a == unique_names[c]
                                        && rel.tensor_b == unique_names[r])
                            })
                            .map(|rel| rel.shared_structure_score)
                            .unwrap_or(0.0)
                    };
                    let cell_rect = egui::Rect::from_min_size(
                        egui::pos2(rect.left() + c as f32 * cell, rect.top() + r as f32 * cell),
                        Vec2::splat(cell - 1.5),
                    );
                    painter.rect_filled(cell_rect, 2.0, spectral_color(score));
                }
            }
            ui.add_space(6.0);
            ui.label(
                RichText::new(format!(
                    "Mean affinity across {} pairs: {:.4}. High affinity (>0.25) indicates strong RMQ shared-subspace compressibility.",
                    report.cross_layer_relations.len(),
                    report.summary.mean_cross_layer_shared_score.unwrap_or(0.0)
                ))
                .color(TEXT_MUTED)
                .size(11.5),
            );
        });

        glass_card().show(&mut cols[1], |ui| {
            ui.label(
                RichText::new("TOP ENTANGLED LAYER PAIRS")
                    .color(AMETHYST)
                    .size(12.0)
                    .strong(),
            );
            ui.add_space(6.0);
            egui::ScrollArea::vertical()
                .max_height(380.0)
                .show(ui, |ui| {
                    for rel in report.cross_layer_relations.iter().take(20) {
                        ui.horizontal(|ui| {
                            pill_badge(
                                ui,
                                &format!("{:.3}", rel.shared_structure_score),
                                if rel.shared_structure_score >= 0.25 {
                                    CYAN
                                } else {
                                    ULTRAVIOLET
                                },
                            );
                            ui.vertical(|ui| {
                                ui.label(
                                    RichText::new(format!("{}  <->  {}", rel.tensor_a, rel.tensor_b))
                                        .monospace()
                                        .size(11.0)
                                        .color(TEXT_PRIMARY),
                                );
                                ui.label(
                                    RichText::new(format!(
                                        "left={:.3} | right={:.3} | cos={:.4}",
                                        rel.left_subspace_affinity,
                                        rel.right_subspace_affinity,
                                        rel.weight_cosine_similarity
                                    ))
                                    .size(10.5)
                                    .color(TEXT_MUTED),
                                );
                            });
                        });
                        ui.add_space(3.0);
                    }
                });
        });
    });
}

pub fn show_qforge(app: &mut QmlDesktopApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    section_header(
        ui,
        "Q-Forge — The Morphogenesis Chamber",
        "Streaming, atomic Safetensors -> QMF v1 representation transformation with live progress coupling and exact byte accounting.",
        Some(("MORPHOGENESIS CHAMBER", ULTRAVIOLET)),
    );

    let time_sec = app.start_time.elapsed().as_secs_f32();
    let dt = (1.0 / app.smoothed_fps.max(15.0)).clamp(0.004, 0.05);
    let (cr, shared_l) = app.forge_report.as_ref().map_or((None, 0), |(r, _)| {
        (Some(r.compression_ratio_vs_source), r.shared_latent_count)
    });
    let forge_scene = build_qforge_morphogenesis_scene(
        app.forge_method.label(),
        app.is_busy(),
        app.forge_progress_frac,
        cr,
        shared_l,
        time_sec,
    );
    let v_h = if app.fullscreen_3d {
        (ui.available_height() - 12.0).max(520.0)
    } else {
        320.0
    };
    let v_res = render_prism_viewport(
        ui,
        ctx,
        "qforge_morphogenesis",
        Vec2::new(ui.available_width(), v_h),
        &mut app.cam_qforge,
        forge_scene,
        &app.persisted.graphics,
        app.wgpu_ready,
        time_sec,
        dt,
        None,
        app.fullscreen_3d,
    );
    if v_res.fullscreen_toggled {
        app.fullscreen_3d = !app.fullscreen_3d;
    }
    if app.fullscreen_3d {
        return;
    }
    ui.add_space(10.0);

    glass_card().show(ui, |ui| {
        egui::Grid::new("qforge_config_grid")
            .num_columns(3)
            .spacing([10.0, 8.0])
            .show(ui, |ui| {
                ui.label("Source Safetensors:");
                ui.add(egui::TextEdit::singleline(&mut app.forge_source_path).desired_width(460.0));
                if ui.button("Browse...").clicked()
                    && let Some(p) = rfd::FileDialog::new()
                        .add_filter("Safetensors", &["safetensors"])
                        .pick_file()
                {
                    app.forge_source_path = p.display().to_string();
                }
                ui.end_row();

                ui.label("Output QMF v1 Path:");
                ui.add(egui::TextEdit::singleline(&mut app.forge_output_path).desired_width(460.0));
                if ui.button("Save As...").clicked()
                    && let Some(p) = rfd::FileDialog::new()
                        .add_filter("QMF v1", &["qmf"])
                        .set_file_name("morphed-model.qmf")
                        .save_file()
                {
                    app.forge_output_path = p.display().to_string();
                }
                ui.end_row();

                ui.label("Transformation Method:");
                egui::ComboBox::from_id_salt("forge_method_combo")
                    .width(340.0)
                    .selected_text(app.forge_method.label())
                    .show_ui(ui, |ui| {
                        for m in ForgeMethodChoice::ALL {
                            ui.selectable_value(&mut app.forge_method, m, m.label());
                        }
                    });
                ui.label("");
                ui.end_row();
            });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            match app.forge_method {
                ForgeMethodChoice::MapAdaptive => {
                    ui.label("Target Avg BPP:");
                    ui.add(
                        egui::DragValue::new(&mut app.forge_target_bpp)
                            .speed(0.05)
                            .range(4.15..=16.0),
                    );
                }
                ForgeMethodChoice::RmqAffinityGated => {
                    ui.label("RMQ Rank:");
                    ui.add(egui::DragValue::new(&mut app.forge_rank).range(2..=128));
                    ui.label("Min Affinity:");
                    ui.add(
                        egui::DragValue::new(&mut app.forge_affinity_threshold)
                            .speed(0.02)
                            .range(0.05..=0.90),
                    );
                    ui.label("Sparse %:");
                    ui.add(
                        egui::DragValue::new(&mut app.forge_sparse_pct)
                            .speed(0.1)
                            .range(0.0..=10.0),
                    );
                }
                ForgeMethodChoice::RmqSharedBasis | ForgeMethodChoice::RmqSharedCore => {
                    ui.label("RMQ Rank:");
                    ui.add(egui::DragValue::new(&mut app.forge_rank).range(2..=128));
                    ui.label("Sparse Outlier %:");
                    ui.add(
                        egui::DragValue::new(&mut app.forge_sparse_pct)
                            .speed(0.1)
                            .range(0.0..=10.0),
                    );
                }
                ForgeMethodChoice::LowRankSvd => {
                    ui.label("SVD Rank:");
                    ui.add(egui::DragValue::new(&mut app.forge_rank).range(2..=256));
                }
                _ => {}
            }
            if ui
                .add_enabled(!app.is_busy(), egui::Button::new("Forge QMF v1 Artifact"))
                .clicked()
            {
                app.start_forge_job(ctx.clone());
            }
        });
    });

    ui.add_space(12.0);

    let mut export_msg: Option<String> = None;
    if let Some((rep, out_path)) = &app.forge_report {
        accent_card().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("TRANSFORMATION COMPLETE — VERIFIED ROUND-TRIP REPORT")
                        .color(EMERALD)
                        .size(12.5)
                        .strong(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    export_msg =
                        export_json_report(ui, "Export JSON...", "qforge-report.json", rep);
                });
            });
            ui.add_space(6.0);
            ui.horizontal_wrapped(|ui| {
                metric_tile(
                    ui,
                    "COMPLETE QMF BPP",
                    &format!("{:.3} bpp", rep.effective_qmf_bpp),
                    &format!("{:.2}x vs source", rep.compression_ratio_vs_source),
                    CYAN,
                );
                metric_tile(
                    ui,
                    "OUTPUT FILE SIZE",
                    &format_bytes(rep.qmf_serialized_bytes as u64),
                    &format!("Source: {}", format_bytes(rep.source_file_bytes)),
                    ULTRAVIOLET,
                );
                metric_tile(
                    ui,
                    "MEAN MATVEC REL ERR",
                    &format!("{:.5}", rep.mean_matvec_rel_error),
                    &format!("Frobenius: {:.5}", rep.mean_rel_frobenius_error),
                    AMETHYST,
                );
                metric_tile(
                    ui,
                    "SHARED LATENTS",
                    &format!("{}", rep.shared_latent_count),
                    &format!("{} ms elapsed", rep.elapsed_millis),
                    EMERALD,
                );
            });
            ui.add_space(6.0);
            ui.label(
                RichText::new(format!("SHA-256: {}", rep.qmf_sha256))
                    .monospace()
                    .size(11.5)
                    .color(TEXT_MUTED),
            );
            ui.label(
                RichText::new(format!("Artifact: {}", out_path.display()))
                    .monospace()
                    .size(11.5)
                    .color(TEXT_MUTED),
            );
        });
    }
    if let Some(msg) = export_msg {
        app.log(msg);
    }
}

pub fn show_qgenesis(app: &mut QmlDesktopApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    section_header(
        ui,
        "Q-Genesis — The Evolutionary Crucible",
        "Interactive 3D objective-space Pareto evolution (BPP × Matvec Error × FLOPs) with recorded parent-child lineage and direct Genome-to-QMF v1 export.",
        Some(("3D EVOLUTIONARY CRUCIBLE", EMERALD)),
    );

    glass_card().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(
                &mut app.genesis_use_synthetic,
                "Use Synthetic Correlated Cluster",
            );
            if !app.genesis_use_synthetic {
                ui.add(
                    egui::TextEdit::singleline(&mut app.genesis_source_path)
                        .desired_width(300.0)
                        .hint_text("Local .safetensors file..."),
                );
                if ui.button("Browse...").clicked()
                    && let Some(p) = rfd::FileDialog::new()
                        .add_filter("Safetensors", &["safetensors"])
                        .pick_file()
                {
                    app.genesis_source_path = p.display().to_string();
                }
            }
            ui.label("Population:");
            ui.add(egui::DragValue::new(&mut app.genesis_population).range(6..=32));
            ui.label("Generations:");
            ui.add(egui::DragValue::new(&mut app.genesis_generations).range(2..=15));
            ui.label("Profile:");
            egui::ComboBox::from_id_salt("genesis_profile_combo")
                .selected_text(app.genesis_profile.label())
                .show_ui(ui, |ui| {
                    for p in [
                        GenesisProfile::Balanced,
                        GenesisProfile::Smallest,
                        GenesisProfile::BestQuality,
                        GenesisProfile::Fastest,
                    ] {
                        ui.selectable_value(&mut app.genesis_profile, p, p.label());
                    }
                });
            if ui
                .add_enabled(!app.is_busy(), egui::Button::new("Run Q-Genesis Evolution"))
                .clicked()
            {
                app.start_genesis_job(ctx.clone());
            }
            if let Some(arch) = &app.genesis_archive {
                if let Some(msg) =
                    export_json_report(ui, "Export JSON...", "qgenesis-archive.json", arch)
                {
                    app.log(msg);
                }
                if !app.genesis_use_synthetic
                    && ui
                        .add_enabled(
                            !app.is_busy(),
                            egui::Button::new("Export Winner to QMF v1..."),
                        )
                        .clicked()
                    && let Some(out) = rfd::FileDialog::new()
                        .add_filter("QMF v1", &["qmf"])
                        .set_file_name("genesis-winner.qmf")
                        .save_file()
                {
                    app.start_genesis_export_job(ctx.clone(), out);
                }
            }
        });
    });

    ui.add_space(10.0);

    // 3D Evolutionary Crucible Viewport
    let time_sec = app.start_time.elapsed().as_secs_f32();
    let dt = (1.0 / app.smoothed_fps.max(15.0)).clamp(0.004, 0.05);
    let gen_scene = build_qgenesis_crucible_scene(
        app.genesis_archive.as_ref(),
        app.genesis_profile,
        app.genesis_selected_id,
        app.genesis_gen_filter,
        time_sec,
    );
    let sel_s = app.genesis_selected_id.map(|id| id.to_string());
    let v_h = if app.fullscreen_3d {
        (ui.available_height() - 12.0).max(520.0)
    } else {
        380.0
    };
    let v_res = render_prism_viewport(
        ui,
        ctx,
        "qgenesis_3d_crucible",
        Vec2::new(ui.available_width(), v_h),
        &mut app.cam_qgenesis,
        gen_scene,
        &app.persisted.graphics,
        app.wgpu_ready,
        time_sec,
        dt,
        sel_s.as_deref(),
        app.fullscreen_3d,
    );
    if v_res.fullscreen_toggled {
        app.fullscreen_3d = !app.fullscreen_3d;
    }
    if let Some(cid) = v_res.clicked_id
        && let Ok(id) = cid.parse::<usize>()
    {
        app.genesis_selected_id = Some(id);
    }
    if app.fullscreen_3d {
        return;
    }

    ui.add_space(10.0);

    let Some(archive) = &app.genesis_archive else {
        glass_card().show(ui, |ui| {
            ui.label(
                RichText::new("Click 'Run Q-Genesis Evolution' to evolve hybrid precision / SVD / RMQ genomes and explore the 3D objective-space crucible.")
                    .color(TEXT_MUTED),
            );
        });
        return;
    };

    // Generation scrubber for inspecting recorded evolutionary transitions
    ui.horizontal(|ui| {
        let mut filter_on = app.genesis_gen_filter.is_some();
        if ui.checkbox(&mut filter_on, "Scrub Generation:").changed() {
            app.genesis_gen_filter = if filter_on {
                Some(archive.config.generations)
            } else {
                None
            };
        }
        if let Some(ref mut g) = app.genesis_gen_filter {
            ui.add(egui::Slider::new(g, 0..=archive.config.generations));
        }
        if let Some(win) = archive.select_best(app.genesis_profile) {
            ui.separator();
            pill_badge(
                ui,
                &format!(
                    "Profile Winner #{}: {:.3} bpp • MV err {:.5}",
                    win.id, win.fitness.effective_bpp, win.fitness.mean_matvec_rel_error
                ),
                EMERALD,
            );
        }
    });
    ui.add_space(6.0);

    if let Some(sel_id) = app.genesis_selected_id
        && let Some(cand) = archive.all_candidates.iter().find(|c| c.id == sel_id)
    {
        accent_card().show(ui, |ui| {
            ui.label(
                RichText::new(format!(
                    "INSPECTING GENOME #{} (Generation {} • Parents: {:?})",
                    cand.id, cand.generation, cand.parent_ids
                ))
                .color(CYAN)
                .strong(),
            );
            ui.label(
                RichText::new(format!(
                    "Eff BPP: {:.3}  •  Matvec Rel Err: {:.5}  •  Est. FLOPs: {}  •  Pareto: {}",
                    cand.fitness.effective_bpp,
                    cand.fitness.mean_matvec_rel_error,
                    cand.fitness.estimated_matvec_flops,
                    cand.on_pareto_frontier
                ))
                .color(TEXT_PRIMARY),
            );
            ui.label(
                RichText::new(format!("Genes: {:?}", cand.genes))
                    .monospace()
                    .size(10.5)
                    .color(TEXT_MUTED),
            );
        });
        ui.add_space(6.0);
    }

    glass_card().show(ui, |ui| {
        ui.label(
            RichText::new(format!(
                "PARETO FRONTIER ({} non-dominated / {} total evaluated across {} generations)",
                archive.pareto_frontier.len(),
                archive.total_candidates_evaluated,
                archive.config.generations
            ))
            .color(CYAN)
            .size(12.5)
            .strong(),
        );
        ui.add_space(6.0);
        let dom_pts: PlotPoints<'_> = archive
            .all_candidates
            .iter()
            .map(|c| [c.fitness.effective_bpp, c.fitness.mean_matvec_rel_error])
            .collect();
        let pareto_pts: PlotPoints<'_> = archive
            .pareto_frontier
            .iter()
            .map(|c| [c.fitness.effective_bpp, c.fitness.mean_matvec_rel_error])
            .collect();
        Plot::new("qgenesis_pareto_plot")
            .height(260.0)
            .x_axis_label("Complete Effective BPP (lower is smaller)")
            .y_axis_label("Mean Matvec Rel Error (lower is better)")
            .show(ui, |plot_ui| {
                plot_ui.points(
                    Points::new(dom_pts)
                        .name("Evaluated Trials")
                        .color(Color32::from_rgb(100, 110, 150))
                        .radius(3.5_f32),
                );
                plot_ui.points(
                    Points::new(pareto_pts)
                        .name("Pareto Frontier")
                        .color(CYAN)
                        .radius(6.0_f32),
                );
            });
    });

    ui.add_space(8.0);
    glass_card().show(ui, |ui| {
        ui.label(
            RichText::new("DISCOVERED NON-DOMINATED GENOMES")
                .color(AMETHYST)
                .strong(),
        );
        ui.add_space(4.0);
        egui::ScrollArea::vertical()
            .max_height(200.0)
            .show(ui, |ui| {
                for c in &archive.pareto_frontier {
                    ui.label(
                        RichText::new(format!(
                            "#{:<3} Gen {:<2} | {:6.3} bpp | mv_err {:.5} | FLOPs {:>9} | {:?}",
                            c.id,
                            c.generation,
                            c.fitness.effective_bpp,
                            c.fitness.mean_matvec_rel_error,
                            c.fitness.estimated_matvec_flops,
                            c.genes
                        ))
                        .monospace()
                        .size(11.0)
                        .color(TEXT_PRIMARY),
                    );
                }
            });
    });
}

pub fn show_qmf_explorer(app: &mut QmlDesktopApp, ui: &mut egui::Ui) {
    section_header(
        ui,
        "QMF Container Explorer",
        "Streaming header inspection, provenance audit, shared latent tables, and CRC-32 integrity verification for QMF v0/v1.",
        Some(("MULTI-GB STREAMING", CYAN)),
    );

    glass_card().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("QMF File:");
            ui.add(egui::TextEdit::singleline(&mut app.qmf_inspect_path).desired_width(480.0));
            if ui.button("Browse...").clicked()
                && let Some(p) = rfd::FileDialog::new()
                    .add_filter("QMF Container", &["qmf"])
                    .pick_file()
            {
                app.qmf_inspect_path = p.display().to_string();
                app.qmf_inspection = QmfInspection::inspect_file(&p).ok();
            }
            if ui.button("Inspect Header").clicked() {
                let p = PathBuf::from(app.qmf_inspect_path.trim());
                match QmfInspection::inspect_file(&p) {
                    Ok(info) => {
                        app.qmf_inspection = Some(info);
                        app.log(format!("Inspected QMF container: {}", p.display()));
                    }
                    Err(e) => app.log(format!("QMF inspection failed: {e}")),
                }
            }
        });
    });

    ui.add_space(10.0);

    match &app.qmf_inspection {
        Some(QmfInspection::V1 {
            manifest,
            manifest_bytes,
            payload_bytes,
        }) => {
            ui.horizontal_wrapped(|ui| {
                metric_tile(
                    ui,
                    "FORMAT VERSION",
                    "QMF v1",
                    &manifest.transformation,
                    ULTRAVIOLET,
                );
                metric_tile(
                    ui,
                    "MODEL ID",
                    &manifest.model_id,
                    &manifest.architecture,
                    CYAN,
                );
                metric_tile(
                    ui,
                    "TENSORS",
                    &format!("{}", manifest.tensors.len()),
                    &format!("{} shared latents", manifest.shared_latents.len()),
                    EMERALD,
                );
                metric_tile(
                    ui,
                    "TOTAL PAYLOAD",
                    &format_bytes(*payload_bytes),
                    &format!("Manifest: {}", format_bytes(*manifest_bytes)),
                    AMETHYST,
                );
            });
            ui.add_space(8.0);
            glass_card().show(ui, |ui| {
                ui.label(
                    RichText::new(format!(
                        "Source Format: {}  •  License: {}  •  Seed: {}",
                        manifest.source_format, manifest.source_license, manifest.random_seed
                    ))
                    .color(TEXT_MUTED),
                );
                if !manifest.extra_metadata.is_empty() {
                    ui.add_space(4.0);
                    let meta_str: Vec<String> = manifest
                        .extra_metadata
                        .iter()
                        .map(|(k, v)| format!("{k}={v}"))
                        .collect();
                    ui.label(
                        RichText::new(meta_str.join("  •  "))
                            .monospace()
                            .size(11.0)
                            .color(AMETHYST),
                    );
                }
            });

            // Compute per-representation byte and count breakdown
            let mut rep_stats: BTreeMap<&'static str, (usize, u64)> = BTreeMap::new();
            for t in &manifest.tensors {
                let (kind, bytes) = representation_summary(&t.representation);
                let entry = rep_stats.entry(kind).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += bytes;
            }
            let shared_bytes: u64 = manifest
                .shared_latents
                .iter()
                .map(|l| latent_encoding_bytes(&l.encoding))
                .sum();
            if shared_bytes > 0 {
                rep_stats.insert(
                    "SHARED_LATENTS",
                    (manifest.shared_latents.len(), shared_bytes),
                );
            }
            let palette = [CYAN, ULTRAVIOLET, AMETHYST, EMERALD, AMBER, CRIMSON];
            let mut segments: Vec<(String, u64, Color32)> = rep_stats
                .iter()
                .enumerate()
                .map(|(i, (name, (_, bytes)))| {
                    ((*name).to_owned(), *bytes, palette[i % palette.len()])
                })
                .collect();
            if *manifest_bytes > 0 {
                segments.push((
                    "JSON_MANIFEST".to_owned(),
                    *manifest_bytes,
                    Color32::from_rgb(130, 145, 190),
                ));
            }
            let grand_total: u64 = segments.iter().map(|(_, b, _)| *b).sum();
            ui.add_space(8.0);
            accent_card().show(ui, |ui| {
                draw_qmf_storage_visuals(ui, &segments, grand_total);
            });

            ui.add_space(8.0);
            ui.columns(2, |cols| {
                glass_card().show(&mut cols[0], |ui| {
                    ui.label(
                        RichText::new("REPRESENTATION STORAGE BREAKDOWN")
                            .color(CYAN)
                            .size(12.0)
                            .strong(),
                    );
                    ui.add_space(6.0);
                    for (kind, (count, bytes)) in &rep_stats {
                        let pct = if *payload_bytes > 0 {
                            (*bytes as f64 / *payload_bytes as f64) * 100.0
                        } else {
                            0.0
                        };
                        ui.horizontal(|ui| {
                            pill_badge(ui, kind, ULTRAVIOLET);
                            ui.label(
                                RichText::new(format!(
                                    "{count} entries  •  {} ({pct:.1}%)",
                                    format_bytes(*bytes)
                                ))
                                .monospace()
                                .size(11.5)
                                .color(TEXT_PRIMARY),
                            );
                        });
                    }
                    ui.add_space(6.0);
                    let bars: Vec<Bar> = rep_stats
                        .iter()
                        .enumerate()
                        .map(|(i, (_k, (_, bytes)))| {
                            Bar::new(i as f64, *bytes as f64 / (1024.0 * 1024.0))
                                .width(0.65)
                                .fill(CYAN)
                        })
                        .collect();
                    Plot::new("qmf_rep_breakdown_plot")
                        .height(150.0)
                        .y_axis_label("MiB")
                        .show(ui, |plot_ui| {
                            plot_ui.bar_chart(BarChart::new(bars).name("Payload MiB"));
                        });
                });

                glass_card().show(&mut cols[1], |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("TENSOR DIRECTORY")
                                .color(AMETHYST)
                                .size(12.0)
                                .strong(),
                        );
                        ui.add(
                            egui::TextEdit::singleline(&mut app.qmf_filter)
                                .hint_text("Filter tensors..."),
                        );
                    });
                    ui.add_space(6.0);
                    let filter = app.qmf_filter.to_ascii_lowercase();
                    egui::ScrollArea::vertical()
                        .max_height(260.0)
                        .show(ui, |ui| {
                            for t in &manifest.tensors {
                                if !filter.is_empty()
                                    && !t.name.to_ascii_lowercase().contains(&filter)
                                {
                                    continue;
                                }
                                let (kind, bytes) = representation_summary(&t.representation);
                                ui.label(
                                    RichText::new(format!(
                                        "{}  {:?}  [{}]  {}",
                                        t.name,
                                        t.shape,
                                        kind,
                                        format_bytes(bytes)
                                    ))
                                    .monospace()
                                    .size(11.0)
                                    .color(TEXT_PRIMARY),
                                );
                            }
                        });
                });
            });

            if !manifest.shared_latents.is_empty() {
                ui.add_space(8.0);
                glass_card().show(ui, |ui| {
                    ui.label(
                        RichText::new(format!(
                            "SHARED LATENT STRUCTURE TABLE ({} deduplicated Arc factors)",
                            manifest.shared_latents.len()
                        ))
                        .color(EMERALD)
                        .size(12.0)
                        .strong(),
                    );
                    ui.add_space(4.0);
                    egui::ScrollArea::vertical()
                        .max_height(140.0)
                        .show(ui, |ui| {
                            for lat in &manifest.shared_latents {
                                ui.label(
                                    RichText::new(format!(
                                        "{}  [{} x {}]  {}",
                                        lat.id,
                                        lat.rows,
                                        lat.cols,
                                        format_bytes(latent_encoding_bytes(&lat.encoding))
                                    ))
                                    .monospace()
                                    .size(11.0)
                                    .color(TEXT_MUTED),
                                );
                            }
                        });
                });
            }
        }
        Some(QmfInspection::V0 { rows, cols, path }) => {
            glass_card().show(ui, |ui| {
                ui.label(
                    RichText::new(format!(
                        "QMF v0 Single Rank-One Container: {} x {} ({})",
                        rows,
                        cols,
                        path.display()
                    ))
                    .color(CYAN),
                );
            });
        }
        None => {
            glass_card().show(ui, |ui| {
                ui.label(
                    RichText::new("Select any .qmf file and click 'Inspect Header' to audit its structure without loading weights into RAM.")
                        .color(TEXT_MUTED),
                );
            });
        }
    }
}

pub fn show_qcore(app: &mut QmlDesktopApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            section_header(
                ui,
                "Q-Core — The Inference Nexus",
                "Executes compressed QMF v1 Transformers natively on Rayon CPU or Persistent GPU Hybrid VRAM with pure-Rust Byte-Level BPE.",
                Some(("ZERO DENSE RECONSTRUCTION", EMERALD)),
            );
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
            let zen_lbl = if app.qcore_zen_mode {
                "Exit Distraction-Free Mode [Esc]"
            } else {
                "Distraction-Free Mode"
            };
            if ui.button(RichText::new(zen_lbl).color(CYAN)).clicked() {
                app.qcore_zen_mode = !app.qcore_zen_mode;
            }
        });
    });
    draw_ambient_inference_pulse(ui, app.is_busy(), app.start_time.elapsed().as_secs_f64());
    ui.add_space(8.0);

    glass_card().show(ui, |ui| {
        egui::Grid::new("qcore_files_grid")
            .num_columns(3)
            .spacing([8.0, 6.0])
            .show(ui, |ui| {
                ui.label("Model QMF v1:");
                ui.add(egui::TextEdit::singleline(&mut app.infer_qmf_path).desired_width(480.0));
                if ui.button("Browse QMF...").clicked()
                    && let Some(p) = rfd::FileDialog::new()
                        .add_filter("QMF", &["qmf"])
                        .pick_file()
                {
                    app.infer_qmf_path = p.display().to_string();
                    if let Some(dir) = p.parent() {
                        if app.infer_config_path.is_empty() && dir.join("config.json").is_file() {
                            app.infer_config_path = dir.join("config.json").display().to_string();
                        }
                        if app.infer_tokenizer_path.is_empty()
                            && dir.join("tokenizer.json").is_file()
                        {
                            app.infer_tokenizer_path =
                                dir.join("tokenizer.json").display().to_string();
                        }
                    }
                }
                ui.end_row();

                ui.label("HF config.json:");
                ui.add(egui::TextEdit::singleline(&mut app.infer_config_path).desired_width(480.0));
                if ui.button("Browse Config...").clicked()
                    && let Some(p) = rfd::FileDialog::new()
                        .add_filter("JSON", &["json"])
                        .pick_file()
                {
                    app.infer_config_path = p.display().to_string();
                }
                ui.end_row();

                ui.label("HF tokenizer.json:");
                ui.add(
                    egui::TextEdit::singleline(&mut app.infer_tokenizer_path).desired_width(480.0),
                );
                if ui.button("Browse Tokenizer...").clicked()
                    && let Some(p) = rfd::FileDialog::new()
                        .add_filter("JSON", &["json"])
                        .pick_file()
                {
                    app.infer_tokenizer_path = p.display().to_string();
                }
                ui.end_row();

                ui.label("Compare Ref QMF (opt):");
                ui.add(
                    egui::TextEdit::singleline(&mut app.infer_compare_qmf_path)
                        .desired_width(480.0)
                        .hint_text(
                            "Optional BF16/FP16 reference .qmf for live KL divergence & PPL",
                        ),
                );
                if ui.button("Browse Ref...").clicked()
                    && let Some(p) = rfd::FileDialog::new()
                        .add_filter("QMF", &["qmf"])
                        .pick_file()
                {
                    app.infer_compare_qmf_path = p.display().to_string();
                }
                ui.end_row();
            });

        ui.add_space(8.0);
        ui.label("Prompt:");
        ui.add(
            egui::TextEdit::multiline(&mut app.infer_prompt)
                .desired_rows(3)
                .desired_width(f32::INFINITY),
        );
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut app.infer_use_chatml, "Apply ChatML Instruct Template");
            ui.checkbox(
                &mut app.infer_use_gpu,
                "GPU Hybrid (Persistent Compressed VRAM)",
            );
            ui.label("Max Tokens:");
            ui.add(egui::DragValue::new(&mut app.infer_max_tokens).range(4..=256));
            ui.label("Temperature:");
            ui.add(
                egui::DragValue::new(&mut app.infer_temperature)
                    .speed(0.05)
                    .range(0.0..=1.5),
            );
            if ui
                .add_enabled(!app.is_busy(), egui::Button::new("Generate Tokens"))
                .clicked()
            {
                app.start_inference_job(ctx.clone());
            }
        });
    });

    ui.add_space(10.0);

    accent_card().show(ui, |ui| {
        ui.label(
            RichText::new("LIVE STREAMING OUTPUT")
                .color(CYAN)
                .size(12.0)
                .strong(),
        );
        ui.add_space(6.0);
        let text = if app.infer_output_text.is_empty() {
            "Awaiting generation..."
        } else {
            &app.infer_output_text
        };
        ui.label(RichText::new(text).size(14.5).color(TEXT_PRIMARY));

        if let Some((toks, tps, ms)) = app.infer_stats {
            ui.add_space(8.0);
            ui.label(
                RichText::new(format!(
                    "Generated {toks} tokens  •  {tps:.2} tok/s  •  {ms:.1} ms total"
                ))
                .color(EMERALD)
                .size(12.0),
            );
        }

        if let Some(fid) = &app.infer_fidelity {
            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                metric_tile(
                    ui,
                    "MEAN KL DIVERGENCE",
                    &format!("{:.5}", fid.mean_kl_divergence),
                    "vs Reference QMF",
                    CYAN,
                );
                metric_tile(
                    ui,
                    "LOGIT RMSE",
                    &format!("{:.5}", fid.mean_logit_rmse),
                    "Next-token logits",
                    AMETHYST,
                );
                metric_tile(
                    ui,
                    "TOP-1 AGREEMENT",
                    &format!("{:.1}%", fid.top1_token_agreement * 100.0),
                    "Greedy argmax match",
                    EMERALD,
                );
                metric_tile(
                    ui,
                    "PERPLEXITY",
                    &format!("{:.3}", fid.transformed_perplexity),
                    &format!("Ref PPL: {:.3}", fid.reference_perplexity),
                    ULTRAVIOLET,
                );
            });
        }
    });
}

pub fn show_qbehavior(app: &mut QmlDesktopApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    section_header(
        ui,
        "Q-Behavior — The Consciousness Observatory",
        "Dual-column 3D activation observatory measuring classical layer cosine similarity, relative L2 drift, RSA geometry, and strictly reversible causal interventions.",
        Some(("CLASSICAL ACTIVATION SCIENCE", CYAN)),
    );

    let time_sec = app.start_time.elapsed().as_secs_f32();
    let dt = (1.0 / app.smoothed_fps.max(15.0)).clamp(0.004, 0.05);
    let bhv_scene = build_qbehavior_dual_scene(
        app.behavior_report.as_ref(),
        app.intervention_report.as_ref(),
        app.behavior_selected_layer,
        time_sec,
    );
    let sel_l = app.behavior_selected_layer.to_string();
    let v_h = if app.fullscreen_3d {
        (ui.available_height() - 12.0).max(520.0)
    } else {
        360.0
    };
    let v_res = render_prism_viewport(
        ui,
        ctx,
        "qbehavior_3d_dual",
        Vec2::new(ui.available_width(), v_h),
        &mut app.cam_qbehavior,
        bhv_scene,
        &app.persisted.graphics,
        app.wgpu_ready,
        time_sec,
        dt,
        Some(&sel_l),
        app.fullscreen_3d,
    );
    if v_res.fullscreen_toggled {
        app.fullscreen_3d = !app.fullscreen_3d;
    }
    if let Some(cid) = v_res.clicked_id
        && let Ok(l) = cid.parse::<usize>()
    {
        app.behavior_selected_layer = l;
        app.behavior_intervention_layer = l;
    }
    if app.fullscreen_3d {
        return;
    }
    ui.add_space(10.0);

    glass_card().show(ui, |ui| {
        ui.label(
            RichText::new("Uses the Model QMF, Reference QMF, config.json, tokenizer.json, and Prompt configured in Q-Core Inference (or edit directly below).")
                .color(TEXT_MUTED)
                .size(12.0),
        );
        ui.add_space(6.0);
        egui::Grid::new("qbehavior_grid")
            .num_columns(2)
            .spacing([8.0, 6.0])
            .show(ui, |ui| {
                ui.label("Reference QMF (e.g. BF16):");
                ui.add(egui::TextEdit::singleline(&mut app.infer_compare_qmf_path).desired_width(520.0));
                ui.end_row();
                ui.label("Candidate QMF:");
                ui.add(egui::TextEdit::singleline(&mut app.infer_qmf_path).desired_width(520.0));
                ui.end_row();
            });
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(
                    !app.is_busy(),
                    egui::Button::new("Compare Layer Activations & RSA"),
                )
                .clicked()
            {
                app.start_behavior_compare_job(ctx.clone());
            }
            ui.separator();
            ui.label("Intervention Layer:");
            ui.add(egui::DragValue::new(&mut app.behavior_intervention_layer).range(0..=127));
            if ui
                .add_enabled(!app.is_busy(), egui::Button::new("Ablate MLP (Reversible)"))
                .clicked()
            {
                let l = app.behavior_intervention_layer;
                app.start_intervention_job(ctx.clone(), qml_runtime::LayerIntervention::AblateMlp { layer: l });
            }
            if ui
                .add_enabled(
                    !app.is_busy(),
                    egui::Button::new("Ablate Attention (Reversible)"),
                )
                .clicked()
            {
                let l = app.behavior_intervention_layer;
                app.start_intervention_job(
                    ctx.clone(),
                    qml_runtime::LayerIntervention::AblateAttention { layer: l },
                );
            }
        });
    });

    ui.add_space(10.0);

    let mut export_msg: Option<String> = None;
    if let Some(rep) = &app.behavior_report {
        accent_card().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("LAYER-BY-LAYER ACTIVATION DRIFT & RSA GEOMETRY")
                        .color(CYAN)
                        .strong(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    export_msg =
                        export_json_report(ui, "Export JSON...", "qbehavior-report.json", rep);
                });
            });
            ui.add_space(6.0);
            ui.horizontal_wrapped(|ui| {
                metric_tile(
                    ui,
                    "PEAK DRIFT LAYER",
                    &format!("Layer {}", rep.peak_divergence_layer),
                    &format!("Rel L2: {:.4}", rep.peak_rel_l2_divergence),
                    AMETHYST,
                );
                metric_tile(
                    ui,
                    "FINAL LAYER COSINE",
                    &format!("{:.4}", rep.final_layer_cosine),
                    &format!("{} layers traced", rep.num_layers),
                    EMERALD,
                );
                metric_tile(
                    ui,
                    "MEAN LOGIT KL",
                    &format!("{:.5}", rep.mean_logit_kl_divergence),
                    &format!("Top-1: {:.1}%", rep.top1_token_agreement * 100.0),
                    CYAN,
                );
            });
            ui.add_space(8.0);
            let cos_pts: PlotPoints = rep
                .layers
                .iter()
                .map(|l| [l.layer_index as f64, l.mean_cosine_similarity])
                .collect();
            let rsa_pts: PlotPoints = rep
                .layers
                .iter()
                .map(|l| [l.layer_index as f64, l.rsa_correlation])
                .collect();
            let l2_pts: PlotPoints = rep
                .layers
                .iter()
                .map(|l| [l.layer_index as f64, l.mean_rel_l2_divergence])
                .collect();
            Plot::new("qbehavior_layer_plot")
                .height(220.0)
                .legend(egui_plot::Legend::default())
                .x_axis_label("Transformer Layer")
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new(cos_pts).name("Mean Cosine Sim").color(EMERALD));
                    plot_ui.line(Line::new(rsa_pts).name("RSA Correlation").color(CYAN));
                    plot_ui.line(Line::new(l2_pts).name("Mean Rel L2 Drift").color(AMETHYST));
                });
        });
    }

    if let Some(intv) = &app.intervention_report {
        ui.add_space(10.0);
        glass_card().show(ui, |ui| {
            ui.label(
                RichText::new(format!(
                    "REVERSIBLE CAUSAL INTERVENTION: {:?}",
                    intv.intervention
                ))
                .color(AMBER)
                .strong(),
            );
            ui.label(
                RichText::new(format!(
                    "Tokens: {}  •  Mean Logit KL: {:.5}  •  Top-1 Prediction Flip Rate: {:.1}%",
                    intv.sequence_len,
                    intv.mean_logit_kl_divergence,
                    intv.top1_flip_rate * 100.0
                ))
                .color(TEXT_PRIMARY),
            );
        });
    }
    if let Some(msg) = export_msg {
        app.log(msg);
    }
}

pub fn show_benchmarks(app: &mut QmlDesktopApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    section_header(
        ui,
        "Benchmarks & Falsifiable RMQ Ablations",
        "Complete byte-for-byte storage accounting across correlated shared-subspace clusters AND uncorrelated i.i.d. Gaussian controls.",
        Some(("FALSIFIABLE SCIENCE", CYAN)),
    );

    glass_card().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label("Layers:");
            ui.add(egui::DragValue::new(&mut app.ablation_layers).range(2..=12));
            ui.label("Matrix Dim:");
            ui.add(egui::DragValue::new(&mut app.ablation_dim).range(32..=128));
            ui.label("RMQ Rank:");
            ui.add(egui::DragValue::new(&mut app.ablation_rank).range(2..=24));
            if ui
                .add_enabled(!app.is_busy(), egui::Button::new("Run Live RMQ Ablation"))
                .clicked()
            {
                app.start_ablation_job(ctx.clone());
            }
        });
    });

    ui.add_space(10.0);

    if let Some((corr, gauss)) = &app.ablation_result {
        ui.columns(2, |cols| {
            glass_card().show(&mut cols[0], |ui| {
                ui.label(
                    RichText::new("CORRELATED SHARED-SUBSPACE CLUSTER (+0.5% OUTLIERS)")
                        .color(EMERALD)
                        .size(12.0)
                        .strong(),
                );
                ui.add_space(6.0);
                for row in &corr.rows {
                    ui.label(
                        RichText::new(format!(
                            "{:<28} | {:6.3} bpp | err {:.5}",
                            row.method, row.effective_bpp, row.mean_rel_frobenius_error
                        ))
                        .monospace()
                        .size(11.5)
                        .color(TEXT_PRIMARY),
                    );
                }
            });

            glass_card().show(&mut cols[1], |ui| {
                ui.label(
                    RichText::new("UNCORRELATED I.I.D. GAUSSIAN CONTROL (FALSIFIABILITY CHECK)")
                        .color(CRIMSON)
                        .size(12.0)
                        .strong(),
                );
                ui.add_space(6.0);
                for row in &gauss.rows {
                    ui.label(
                        RichText::new(format!(
                            "{:<28} | {:6.3} bpp | err {:.5}",
                            row.method, row.effective_bpp, row.mean_rel_frobenius_error
                        ))
                        .monospace()
                        .size(11.5)
                        .color(TEXT_PRIMARY),
                    );
                }
            });
        });
    } else {
        glass_card().show(ui, |ui| {
            ui.label(
                RichText::new("Click 'Run Live RMQ Ablation' to execute head-to-head RMQ Shared-Basis, RMQ Shared-Core, Q4_0, Q4_1, and INT8 baselines in real time.")
                    .color(TEXT_MUTED),
            );
        });
    }
}

pub fn show_system_observatory(app: &mut QmlDesktopApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    section_header(
        ui,
        "Hardware Observatory — Computational Anatomy",
        "Procedural 3D silicon schematic, real CPU/SIMD/RAM/WGPU telemetry, PROJECT PRISM graphics presets, and live WGSL compute shader verification.",
        Some(("COMPUTATIONAL ANATOMY", EMERALD)),
    );

    let time_sec = app.start_time.elapsed().as_secs_f32();
    let dt = (1.0 / app.smoothed_fps.max(15.0)).clamp(0.004, 0.05);
    let hw_scene = build_hardware_anatomy_scene(&app.hardware, time_sec);
    let v_h = if app.fullscreen_3d {
        (ui.available_height() - 12.0).max(520.0)
    } else {
        320.0
    };
    let v_res = render_prism_viewport(
        ui,
        ctx,
        "hardware_3d_anatomy",
        Vec2::new(ui.available_width(), v_h),
        &mut app.cam_hardware,
        hw_scene,
        &app.persisted.graphics,
        app.wgpu_ready,
        time_sec,
        dt,
        None,
        app.fullscreen_3d,
    );
    if v_res.fullscreen_toggled {
        app.fullscreen_3d = !app.fullscreen_3d;
    }
    if app.fullscreen_3d {
        return;
    }
    ui.add_space(10.0);

    // PROJECT PRISM Graphics Quality Presets & Controls (persisted cleanly)
    accent_card().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("PROJECT PRISM — 3D GRAPHICS QUALITY & ADAPTIVE CONTROLS")
                    .color(CYAN)
                    .size(12.0)
                    .strong(),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                pill_badge(
                    ui,
                    &format!("Measured Frame Rate: {:.0} FPS", app.smoothed_fps),
                    EMERALD,
                );
            });
        });
        ui.add_space(6.0);
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            for p in GraphicsPreset::ALL {
                if ui
                    .selectable_value(&mut app.persisted.graphics.preset, p, p.label())
                    .changed()
                {
                    app.persisted.graphics = GraphicsSettings::from_preset(p);
                    changed = true;
                }
            }
        });
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            ui.label("Resolution Scale:");
            changed |= ui
                .add(egui::Slider::new(&mut app.persisted.graphics.resolution_scale, 0.50..=1.0).fixed_decimals(2))
                .changed();
            ui.label("Particle Density:");
            changed |= ui
                .add(egui::Slider::new(&mut app.persisted.graphics.particle_density, 0.0..=1.0).fixed_decimals(2))
                .changed();
            changed |= ui
                .checkbox(&mut app.persisted.graphics.bloom_enabled, "HDR Bloom")
                .changed();
            if app.persisted.graphics.bloom_enabled {
                changed |= ui
                    .add(egui::Slider::new(&mut app.persisted.graphics.bloom_intensity, 0.0..=1.6).fixed_decimals(2))
                    .changed();
            }
            ui.label("Animation:");
            changed |= ui
                .add(egui::Slider::new(&mut app.persisted.graphics.animation_intensity, 0.0..=1.5).fixed_decimals(2))
                .changed();
            changed |= ui
                .checkbox(&mut app.persisted.graphics.reduced_motion, "Reduced Motion")
                .changed();
            ui.label("FPS Cap:");
            for cap in [30u32, 60, 120, 144] {
                changed |= ui
                    .selectable_value(&mut app.persisted.graphics.fps_limit, cap, format!("{cap}"))
                    .changed();
            }
        });
        if changed {
            save_persisted_state(&app.persisted);
        }
        ui.add_space(4.0);
        ui.label(
            RichText::new("Telemetry Honesty: GPU temperature, board power draw, and live OS-wide VRAM usage are not exposed by safe cross-platform WGPU and are intentionally never fabricated.")
                .color(TEXT_MUTED)
                .size(11.0),
        );
    });
    ui.add_space(10.0);

    ui.columns(3, |cols| {
        metric_tile(
            &mut cols[0],
            "PROCESSOR",
            &format!("{} Threads", app.hardware.logical_cores),
            &app.hardware.cpu_brand,
            CYAN,
        );
        metric_tile(
            &mut cols[1],
            "SIMD INSTRUCTION SETS",
            &format!("{} Detected", app.hardware.simd_features.len()),
            &app.hardware.simd_features.join(", "),
            AMETHYST,
        );
        metric_tile(
            &mut cols[2],
            "PHYSICAL RAM",
            &format_bytes(app.hardware.total_ram_bytes),
            &format!(
                "Available: {}",
                format_bytes(app.hardware.available_ram_bytes)
            ),
            ULTRAVIOLET,
        );
    });

    ui.add_space(10.0);

    glass_card().show(ui, |ui| {
        ui.label(
            RichText::new("RESOURCE BUDGET PRESET")
                .color(CYAN)
                .size(12.0)
                .strong(),
        );
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            for preset in [
                ResourcePreset::Conservative,
                ResourcePreset::Balanced,
                ResourcePreset::Performance,
            ] {
                if ui
                    .selectable_value(&mut app.persisted.resource_preset, preset, preset.label())
                    .changed()
                {
                    save_persisted_state(&app.persisted);
                }
            }
        });
        let plan = app.persisted.resource_preset.plan(&app.hardware);
        ui.add_space(4.0);
        ui.label(
            RichText::new(format!(
                "Active Plan: RAM cap {}  •  Worker threads {}  •  GPU compute preferred: {}",
                format_bytes(plan.ram_budget_bytes),
                plan.max_threads,
                plan.prefer_gpu_compute
            ))
            .color(TEXT_MUTED)
            .size(12.0),
        );
    });

    ui.add_space(10.0);

    glass_card().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("DISCOVERED WGPU ADAPTERS & COMPUTE SELF-TEST")
                    .color(EMERALD)
                    .size(12.0)
                    .strong(),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(!app.is_busy(), egui::Button::new("Run WGPU Compute Self-Test"))
                    .clicked()
                {
                    app.start_gpu_selftest(ctx.clone());
                }
            });
        });
        ui.add_space(6.0);
        if app.hardware.gpus.is_empty() {
            ui.label(RichText::new("No WGPU adapters detected — CPU Rayon backend active.").color(AMBER));
        } else {
            for gpu in &app.hardware.gpus {
                egui::Frame::new()
                    .fill(theme::PANEL_BG)
                    .stroke(Stroke::new(1.0_f32, BORDER_SUBTLE))
                    .corner_radius(egui::CornerRadius::same(6))
                    .inner_margin(egui::Margin::same(10))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            pill_badge(ui, &gpu.backend, CYAN);
                            pill_badge(ui, &gpu.device_type, ULTRAVIOLET);
                            ui.label(RichText::new(&gpu.name).color(TEXT_PRIMARY).strong());
                        });
                        let max_buf_label = if gpu.max_buffer_size > 1_099_511_627_776 {
                            "Device Max (>1 TiB)".to_owned()
                        } else {
                            format_bytes(gpu.max_buffer_size)
                        };
                        ui.label(
                            RichText::new(format!(
                                "Driver: {} ({})  •  Max Buffer: {}  •  Max Workgroup X: {}",
                                gpu.driver,
                                gpu.driver_info,
                                max_buf_label,
                                gpu.max_compute_workgroup_size_x
                            ))
                            .color(TEXT_MUTED)
                            .size(11.5),
                        );
                    });
                ui.add_space(4.0);
            }
        }

        if let Some(res) = &app.gpu_selftest {
            ui.add_space(8.0);
            accent_card().show(ui, |ui| {
                ui.label(
                    RichText::new(format!(
                        "WGPU MULTI-KERNEL SELF-TEST PASSED on {} ({}) — Matrix {}x{}",
                        res.adapter_name, res.backend_name, res.matrix_shape.0, res.matrix_shape.1
                    ))
                    .color(EMERALD)
                    .strong(),
                );
                ui.add_space(6.0);
                for k in &res.kernels {
                    egui::Frame::new()
                        .fill(theme::PANEL_BG)
                        .stroke(Stroke::new(1.0_f32, BORDER_SUBTLE))
                        .corner_radius(egui::CornerRadius::same(6))
                        .inner_margin(egui::Margin::same(8))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                pill_badge(
                                    ui,
                                    &format!("{:.2}x VRAM", k.vram_compression_vs_f32),
                                    if k.vram_compression_vs_f32 > 1.5 {
                                        CYAN
                                    } else {
                                        ULTRAVIOLET
                                    },
                                );
                                ui.label(RichText::new(&k.kernel_name).color(TEXT_PRIMARY).strong());
                            });
                            ui.label(
                                RichText::new(format!(
                                    "VRAM Buffer: {}  •  Max Abs Err vs CPU f64: {:.2e}  •  GPU (incl. staging): {:.2} ms  •  CPU: {:.2} ms",
                                    format_bytes(k.vram_weight_bytes as u64),
                                    k.max_abs_error,
                                    k.gpu_elapsed_ms,
                                    k.cpu_elapsed_ms
                                ))
                                .monospace()
                                .color(TEXT_MUTED)
                                .size(11.5),
                            );
                        });
                    ui.add_space(4.0);
                }
            });
        }
    });
}
