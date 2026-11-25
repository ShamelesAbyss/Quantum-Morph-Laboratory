//! All 10 scientific laboratory views for `qml-desktop`.
//! Every control invokes real Rust laboratory engines (`qml-scope`, `qml-quant`,
//! `qml-forge`, `qml-format`, `qml-runtime`, `qml-tokenizer`, `qml-genesis`,
//! `qml-hardware`). Unsupported/experimental statuses are explicitly labeled.

use crate::state::{ModelFormat, QmfInspection, save_persisted_state};
use crate::theme::{
    self, AMBER, AMETHYST, BORDER_SUBTLE, CRIMSON, CYAN, EMERALD, TEXT_MUTED, TEXT_PRIMARY,
    ULTRAVIOLET, accent_card, format_bytes, glass_card, metric_tile, pill_badge, spectral_color,
};
use crate::{ForgeMethodChoice, NavTab, QmlDesktopApp};
use eframe::egui::{self, Color32, RichText, Stroke, Vec2};
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoints, Points};
use qml_hardware::ResourcePreset;
use std::path::PathBuf;

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
    section_header(
        ui,
        "Laboratory Command Center",
        "Don't just compress intelligence. Transform its representation.",
        Some(("PROJECT SINGULARITY", ULTRAVIOLET)),
    );

    // Top metric strip
    ui.horizontal_wrapped(|ui| {
        metric_tile(
            ui,
            "LOCAL LIBRARY",
            &format!("{} Artifacts", app.persisted.library.len()),
            "Safetensors, GGUF & QMF (read-only)",
            CYAN,
        );
        metric_tile(
            ui,
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
            ui,
            "PRIMARY ACCELERATOR",
            &gpu_val,
            &format!("{} WGPU adapter(s) detected", app.hardware.gpus.len()),
            EMERALD,
        );
        metric_tile(
            ui,
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

    // Quick-launch laboratory workflows
    accent_card().show(ui, |ui| {
        ui.label(
            RichText::new("QUICK LABORATORY ACTIONS")
                .color(CYAN)
                .size(12.0)
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
            if ui.button("Open Q-Scope Observatory").clicked() {
                app.active_tab = NavTab::QScope;
            }
            if ui.button("Open Q-Forge Studio").clicked() {
                app.active_tab = NavTab::QForge;
            }
            if ui.button("Open Q-Core Inference").clicked() {
                app.active_tab = NavTab::QCore;
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
                RichText::new("RESEARCH SUBSYSTEM STATUS")
                    .color(AMETHYST)
                    .size(12.0)
                    .strong(),
            );
            ui.add_space(6.0);
            let rows = [
                (
                    "Q-Scope Observatory",
                    "Streaming stats, SVD & Grassmannian affinity",
                    "ACTIVE",
                    EMERALD,
                ),
                (
                    "Q-Forge Pipeline",
                    "Memory-bounded Safetensors -> QMF v1",
                    "ACTIVE",
                    EMERALD,
                ),
                (
                    "Q-Core Runtime",
                    "Direct compressed Transformer + BPE tokenizer",
                    "ACTIVE",
                    EMERALD,
                ),
                (
                    "MAP & RMQ Engines",
                    "Adaptive precision & Shared-Basis/Core RMQ",
                    "EXPERIMENTAL",
                    AMBER,
                ),
                (
                    "Q-Genesis Evolution",
                    "3-objective NSGA-II Pareto representation search",
                    "EXPERIMENTAL",
                    AMBER,
                ),
                (
                    "WGPU Compute Kernel",
                    "Validated WGSL matvec prototype (f32)",
                    "PROTOTYPE",
                    CYAN,
                ),
            ];
            for (name, desc, status, col) in rows {
                ui.horizontal(|ui| {
                    pill_badge(ui, status, col);
                    ui.label(RichText::new(name).color(TEXT_PRIMARY).strong());
                    ui.label(RichText::new(desc).color(TEXT_MUTED).size(12.0));
                });
                ui.add_space(2.0);
            }
        });

        glass_card().show(&mut cols[1], |ui| {
            ui.label(
                RichText::new("LABORATORY TELEMETRY LOG")
                    .color(CYAN)
                    .size(12.0)
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
        });
    });

    ui.add_space(10.0);

    let Some(report) = &app.scope_report else {
        glass_card().show(ui, |ui| {
            ui.label(
                RichText::new("No Q-Scope report loaded yet. Choose a local .safetensors model and click 'Run Q-Scope Analysis'.")
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
        "Q-Entangle — Cross-Layer Grassmannian Subspace Affinity",
        "Visualizes shared left/right singular subspaces across transformer layers to identify genuine RMQ compression candidates.",
        Some(("GRASSMANNIAN", ULTRAVIOLET)),
    );

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
        "Q-Forge Transformation Studio",
        "Memory-bounded streaming transformation from immutable Safetensors into verified QMF v1 compressed containers.",
        Some(("STREAMING PIPELINE", EMERALD)),
    );

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

    if let Some((rep, out_path)) = &app.forge_report {
        accent_card().show(ui, |ui| {
            ui.label(
                RichText::new("TRANSFORMATION COMPLETE — VERIFIED ROUND-TRIP REPORT")
                    .color(EMERALD)
                    .size(12.5)
                    .strong(),
            );
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
}

pub fn show_qgenesis(app: &mut QmlDesktopApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    section_header(
        ui,
        "Q-Genesis Evolutionary Representation Lab",
        "Multi-objective NSGA-II Pareto search optimizing Complete BPP vs Direct Matvec Error vs Estimated FLOPs.",
        Some(("EXPERIMENTAL", AMBER)),
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
                        .desired_width(320.0)
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
            if ui
                .add_enabled(!app.is_busy(), egui::Button::new("Run Q-Genesis Evolution"))
                .clicked()
            {
                app.start_genesis_job(ctx.clone());
            }
        });
    });

    ui.add_space(10.0);

    let Some(archive) = &app.genesis_archive else {
        glass_card().show(ui, |ui| {
            ui.label(
                RichText::new("Click 'Run Q-Genesis Evolution' to evolve hybrid precision / SVD / RMQ genomes and plot the live Pareto frontier.")
                    .color(TEXT_MUTED),
            );
        });
        return;
    };

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
            });
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
    section_header(
        ui,
        "Q-Core Native Compressed Inference Chamber",
        "Executes Transformer models directly over compressed QMF v1 operators with pure-Rust Byte-Level BPE tokenization and live streaming output.",
        Some(("ZERO DENSE RECONSTRUCTION", EMERALD)),
    );

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
        "System Observatory & Hardware Acceleration",
        "Real-time CPU, SIMD, RAM, and WGPU GPU adapter diagnostics with live WGSL compute shader verification.",
        Some(("WGPU + RAYON", EMERALD)),
    );

    ui.horizontal_wrapped(|ui| {
        metric_tile(
            ui,
            "PROCESSOR",
            &format!("{} Threads", app.hardware.logical_cores),
            &app.hardware.cpu_brand,
            CYAN,
        );
        metric_tile(
            ui,
            "SIMD INSTRUCTION SETS",
            &format!("{} Detected", app.hardware.simd_features.len()),
            &app.hardware.simd_features.join(", "),
            AMETHYST,
        );
        metric_tile(
            ui,
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
                        ui.label(
                            RichText::new(format!(
                                "Driver: {} ({})  •  Max Buffer: {}  •  Max Workgroup X: {}",
                                gpu.driver,
                                gpu.driver_info,
                                format_bytes(gpu.max_buffer_size),
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
                        "WGPU SELF-TEST PASSED on {} ({})",
                        res.adapter_name, res.backend_name
                    ))
                    .color(EMERALD)
                    .strong(),
                );
                ui.label(
                    RichText::new(format!(
                        "Matrix {}x{} matvec  •  Max Abs Error vs CPU f64: {:.2e}  •  GPU (incl. staging): {:.2} ms  •  CPU: {:.2} ms",
                        res.matrix_shape.0,
                        res.matrix_shape.1,
                        res.max_abs_error,
                        res.gpu_elapsed_ms,
                        res.cpu_elapsed_ms
                    ))
                    .color(TEXT_PRIMARY)
                    .size(12.0),
                );
            });
        }
    });
}
