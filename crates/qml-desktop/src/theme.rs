//! Visual identity and scientific laboratory theme for Quantum Morph Laboratory.
//!
//! Palette: Deep Obsidian (`#07080F`), Elevated Glass (`#0E111E`),
//! Ultraviolet (`#8B5CF6`), Amethyst (`#A78BFA`), Electric Cyan (`#22D3EE`),
//! Spectral Emerald (`#10B981`), Amber (`#F59E0B`), Crimson (`#F43F5E`).

use eframe::egui::{self, Color32, CornerRadius, Margin, RichText, Stroke, Vec2};

pub const OBSIDIAN_BG: Color32 = Color32::from_rgb(7, 8, 15);
pub const PANEL_BG: Color32 = Color32::from_rgb(13, 16, 29);
pub const CARD_BG: Color32 = Color32::from_rgb(18, 22, 40);
pub const BORDER_SUBTLE: Color32 = Color32::from_rgb(38, 45, 76);
pub const BORDER_GLOW: Color32 = Color32::from_rgb(82, 65, 148);

pub const ULTRAVIOLET: Color32 = Color32::from_rgb(139, 92, 246);
pub const AMETHYST: Color32 = Color32::from_rgb(167, 139, 250);
pub const CYAN: Color32 = Color32::from_rgb(34, 211, 238);
pub const EMERALD: Color32 = Color32::from_rgb(16, 185, 129);
pub const AMBER: Color32 = Color32::from_rgb(245, 158, 11);
pub const CRIMSON: Color32 = Color32::from_rgb(244, 63, 94);

pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(236, 240, 255);
pub const TEXT_MUTED: Color32 = Color32::from_rgb(148, 160, 198);

/// Apply the Deep Obsidian / Ultraviolet / Electric Cyan laboratory visual theme.
pub fn apply_laboratory_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.dark_mode = true;
    visuals.panel_fill = OBSIDIAN_BG;
    visuals.window_fill = PANEL_BG;
    visuals.extreme_bg_color = Color32::from_rgb(5, 6, 12);
    visuals.faint_bg_color = Color32::from_rgb(15, 19, 34);
    visuals.code_bg_color = Color32::from_rgb(11, 14, 26);

    visuals.selection.bg_fill = Color32::from_rgba_unmultiplied(139, 92, 246, 85);
    visuals.selection.stroke = Stroke::new(1.0_f32, CYAN);
    visuals.hyperlink_color = CYAN;

    visuals.widgets.noninteractive.bg_fill = PANEL_BG;
    visuals.widgets.noninteractive.weak_bg_fill = PANEL_BG;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, BORDER_SUBTLE);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, TEXT_PRIMARY);

    visuals.widgets.inactive.bg_fill = CARD_BG;
    visuals.widgets.inactive.weak_bg_fill = CARD_BG;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, BORDER_SUBTLE);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT_PRIMARY);
    visuals.widgets.inactive.corner_radius = CornerRadius::same(6);

    visuals.widgets.hovered.bg_fill = Color32::from_rgb(28, 34, 60);
    visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(28, 34, 60);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.2_f32, ULTRAVIOLET);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0_f32, Color32::WHITE);
    visuals.widgets.hovered.corner_radius = CornerRadius::same(6);

    visuals.widgets.active.bg_fill = Color32::from_rgb(45, 36, 84);
    visuals.widgets.active.weak_bg_fill = Color32::from_rgb(45, 36, 84);
    visuals.widgets.active.bg_stroke = Stroke::new(1.5_f32, CYAN);
    visuals.widgets.active.fg_stroke = Stroke::new(1.0_f32, Color32::WHITE);
    visuals.widgets.active.corner_radius = CornerRadius::same(6);

    visuals.window_stroke = Stroke::new(1.0_f32, BORDER_GLOW);
    visuals.window_corner_radius = CornerRadius::same(8);

    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(10.0, 8.0);
    style.spacing.button_padding = Vec2::new(12.0, 6.0);
    style.spacing.window_margin = Margin::same(14);
    ctx.set_style(style);
}

/// Luminous dark glass card container.
pub fn glass_card() -> egui::Frame {
    egui::Frame::new()
        .fill(CARD_BG)
        .stroke(Stroke::new(1.0_f32, BORDER_SUBTLE))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::same(14))
}

/// Highlighted ultraviolet-bordered card container.
pub fn accent_card() -> egui::Frame {
    egui::Frame::new()
        .fill(Color32::from_rgb(17, 19, 38))
        .stroke(Stroke::new(1.2_f32, BORDER_GLOW))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::same(14))
}

/// Render a small status or maturity pill badge.
pub fn pill_badge(ui: &mut egui::Ui, label: &str, color: Color32) {
    let bg = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 32);
    egui::Frame::new()
        .fill(bg)
        .stroke(Stroke::new(1.0_f32, color))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::symmetric(8, 2))
        .show(ui, |ui| {
            ui.label(RichText::new(label).color(color).size(11.0).strong());
        });
}

/// Render a structured metric tile inside a grid or horizontal layout.
pub fn metric_tile(ui: &mut egui::Ui, title: &str, value: &str, subtitle: &str, accent: Color32) {
    glass_card().show(ui, |ui| {
        ui.set_min_width(180.0);
        ui.label(RichText::new(title).color(TEXT_MUTED).size(11.5));
        ui.add_space(3.0);
        ui.label(RichText::new(value).color(accent).size(20.0).strong());
        if !subtitle.is_empty() {
            ui.add_space(2.0);
            ui.label(RichText::new(subtitle).color(TEXT_MUTED).size(11.0));
        }
    });
}

/// Render a subtle procedural Quantum Morph Laboratory vector emblem.
pub fn draw_lab_emblem(ui: &mut egui::Ui, size: f32, time_sec: f64) {
    let (rect, _resp) = ui.allocate_exact_size(Vec2::splat(size), egui::Sense::hover());
    let painter = ui.painter();
    let center = rect.center();
    let r = size * 0.42;

    // Outer ultraviolet ring
    painter.circle_stroke(center, r, Stroke::new(1.6_f32, ULTRAVIOLET));

    // Rotating orbital ellipse vertices
    let phase = (time_sec * 0.6) as f32;
    let mut pts = Vec::with_capacity(24);
    for i in 0..=24 {
        let a = (i as f32 / 24.0) * std::f32::consts::TAU;
        let x = a.cos() * r * 0.92;
        let y = a.sin() * r * 0.42;
        let rx = x * phase.cos() - y * phase.sin();
        let ry = x * phase.sin() + y * phase.cos();
        pts.push(egui::pos2(center.x + rx, center.y + ry));
    }
    painter.add(egui::Shape::line(pts, Stroke::new(1.3_f32, CYAN)));

    // Inner core nucleus
    painter.circle_filled(center, r * 0.22, CYAN);
    painter.circle_stroke(center, r * 0.36, Stroke::new(1.0_f32, AMETHYST));
}

/// Map a `[0.0, 1.0]` spectral affinity value to an Obsidian -> Ultraviolet -> Electric Cyan color.
pub fn spectral_color(value: f64) -> Color32 {
    let t = value.clamp(0.0, 1.0) as f32;
    if t < 0.5 {
        let u = t * 2.0;
        let r = (15.0 + (139.0 - 15.0) * u) as u8;
        let g = (19.0 + (92.0 - 19.0) * u) as u8;
        let b = (36.0 + (246.0 - 36.0) * u) as u8;
        Color32::from_rgb(r, g, b)
    } else {
        let u = (t - 0.5) * 2.0;
        let r = (139.0 + (34.0 - 139.0) * u) as u8;
        let g = (92.0 + (211.0 - 92.0) * u) as u8;
        let b = (246.0 + (238.0 - 246.0) * u) as u8;
        Color32::from_rgb(r, g, b)
    }
}

/// Human-readable byte size formatting.
pub fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.2} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.1} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

/// Generate a crisp procedural 64×64 RGBA Singularity Prism window icon (zero external asset deps).
pub fn generate_app_icon_data() -> egui::IconData {
    let size = 64u32;
    let mut rgba = vec![0u8; (size * size * 4) as usize];
    let half = size as f32 * 0.5;
    for y in 0..size {
        for x in 0..size {
            let nx = (x as f32 + 0.5 - half) / half;
            let ny = (y as f32 + 0.5 - half) / half;
            let r = (nx * nx + ny * ny).sqrt();
            let idx = ((y * size + x) * 4) as usize;
            if r > 0.98 {
                rgba[idx..idx + 4].copy_from_slice(&[0, 0, 0, 0]);
                continue;
            }
            // Deep obsidian disc
            let mut cr = 8.0_f32;
            let mut cg = 10.0_f32;
            let mut cb = 22.0_f32;

            // Outer ultraviolet halo ring
            let ring = (-((r - 0.78) * 14.0).powi(2)).exp();
            cr += 139.0 * ring;
            cg += 92.0 * ring;
            cb += 246.0 * ring;

            // Tilted cyan orbital ellipse
            let rx = nx * 0.866 + ny * 0.5;
            let ry = (-nx * 0.5 + ny * 0.866) * 2.15;
            let er = (rx * rx + ry * ry).sqrt();
            let orbit = (-((er - 0.68) * 15.0).powi(2)).exp();
            cr += 34.0 * orbit;
            cg += 211.0 * orbit;
            cb += 238.0 * orbit;

            // Central diamond crystal prism
            let d = nx.abs() + ny.abs();
            if d < 0.36 {
                let f = 1.0 - d / 0.36;
                cr = cr * (1.0 - f) + (167.0 * (1.0 - f * 0.5) + 34.0 * f * 0.5) * f;
                cg = cg * (1.0 - f) + (139.0 * (1.0 - f * 0.6) + 235.0 * f * 0.6) * f;
                cb = cb * (1.0 - f) + 252.0 * f;
            }

            let alpha = ((1.0 - ((r - 0.90) / 0.08).clamp(0.0, 1.0)) * 255.0) as u8;
            rgba[idx] = cr.clamp(0.0, 255.0) as u8;
            rgba[idx + 1] = cg.clamp(0.0, 255.0) as u8;
            rgba[idx + 2] = cb.clamp(0.0, 255.0) as u8;
            rgba[idx + 3] = alpha;
        }
    }
    egui::IconData {
        rgba,
        width: size,
        height: size,
    }
}

/// Draw an original minimalist scientific vector icon for each laboratory subsystem tab.
pub fn draw_subsystem_icon(ui: &mut egui::Ui, index: usize, size: f32, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(size), egui::Sense::hover());
    let p = ui.painter();
    let c = rect.center();
    let r = size * 0.38;
    let st = Stroke::new(1.35_f32, color);

    match index {
        0 => {
            // Singularity Command Center: concentric horizon + nucleus
            p.circle_stroke(c, r, st);
            p.circle_stroke(c, r * 0.55, Stroke::new(1.0_f32, AMETHYST));
            p.circle_filled(c, r * 0.24, color);
        }
        1 => {
            // Model Library: stacked data strata
            for dy in [-r * 0.65, 0.0, r * 0.65] {
                p.line_segment(
                    [egui::pos2(c.x - r, c.y + dy), egui::pos2(c.x + r, c.y + dy)],
                    st,
                );
            }
        }
        2 => {
            // Q-Scope: octahedral prism crystal
            let pts = vec![
                egui::pos2(c.x, c.y - r),
                egui::pos2(c.x + r * 0.85, c.y),
                egui::pos2(c.x, c.y + r),
                egui::pos2(c.x - r * 0.85, c.y),
                egui::pos2(c.x, c.y - r),
            ];
            p.add(egui::Shape::line(pts, st));
            p.line_segment(
                [
                    egui::pos2(c.x - r * 0.85, c.y),
                    egui::pos2(c.x + r * 0.85, c.y),
                ],
                Stroke::new(0.9_f32, color),
            );
        }
        3 => {
            // Q-Entangle: 3-node subspace constellation
            let a = egui::pos2(c.x - r * 0.8, c.y + r * 0.55);
            let b = egui::pos2(c.x + r * 0.8, c.y + r * 0.35);
            let d = egui::pos2(c.x, c.y - r * 0.8);
            p.line_segment([a, b], st);
            p.line_segment([b, d], st);
            p.line_segment([d, a], st);
            for pt in [a, b, d] {
                p.circle_filled(pt, 2.2, color);
            }
        }
        4 => {
            // Q-Forge: morphogenesis chevron transform
            p.line_segment(
                [
                    egui::pos2(c.x - r * 0.8, c.y - r * 0.6),
                    egui::pos2(c.x, c.y),
                ],
                st,
            );
            p.line_segment(
                [
                    egui::pos2(c.x - r * 0.8, c.y + r * 0.6),
                    egui::pos2(c.x, c.y),
                ],
                st,
            );
            p.line_segment(
                [
                    egui::pos2(c.x + r * 0.1, c.y - r * 0.6),
                    egui::pos2(c.x + r * 0.9, c.y),
                ],
                st,
            );
            p.line_segment(
                [
                    egui::pos2(c.x + r * 0.1, c.y + r * 0.6),
                    egui::pos2(c.x + r * 0.9, c.y),
                ],
                st,
            );
        }
        5 => {
            // Q-Genesis: Pareto frontier arc + star nodes
            let p1 = egui::pos2(c.x - r * 0.75, c.y - r * 0.7);
            let p2 = egui::pos2(c.x - r * 0.15, c.y + r * 0.15);
            let p3 = egui::pos2(c.x + r * 0.8, c.y + r * 0.7);
            p.line_segment([p1, p2], st);
            p.line_segment([p2, p3], st);
            for pt in [p1, p2, p3] {
                p.circle_filled(pt, 2.4, color);
            }
        }
        6 => {
            // QMF Explorer: isometric container cube
            p.rect_stroke(
                egui::Rect::from_center_size(c, Vec2::splat(r * 1.45)),
                2.0,
                st,
                egui::StrokeKind::Inside,
            );
            p.line_segment(
                [
                    egui::pos2(c.x, c.y - r * 0.72),
                    egui::pos2(c.x, c.y + r * 0.72),
                ],
                Stroke::new(0.9_f32, color),
            );
        }
        7 => {
            // Q-Core Inference Nexus: luminous neural spark
            p.line_segment([egui::pos2(c.x - r, c.y), egui::pos2(c.x + r, c.y)], st);
            p.line_segment([egui::pos2(c.x, c.y - r), egui::pos2(c.x, c.y + r)], st);
            p.circle_filled(c, 2.5, CYAN);
        }
        8 => {
            // Q-Behavior: twin parallel activation columns + bridge
            p.line_segment(
                [
                    egui::pos2(c.x - r * 0.6, c.y - r),
                    egui::pos2(c.x - r * 0.6, c.y + r),
                ],
                st,
            );
            p.line_segment(
                [
                    egui::pos2(c.x + r * 0.6, c.y - r),
                    egui::pos2(c.x + r * 0.6, c.y + r),
                ],
                st,
            );
            for dy in [-r * 0.5, 0.0, r * 0.5] {
                p.line_segment(
                    [
                        egui::pos2(c.x - r * 0.6, c.y + dy),
                        egui::pos2(c.x + r * 0.6, c.y + dy),
                    ],
                    Stroke::new(1.0_f32, AMETHYST),
                );
            }
        }
        9 => {
            // Benchmarks: 3 bar columns
            for (i, h) in [0.45_f32, 0.95, 0.7].iter().enumerate() {
                let x = c.x - r * 0.7 + i as f32 * r * 0.7;
                p.line_segment(
                    [
                        egui::pos2(x, c.y + r * 0.8),
                        egui::pos2(x, c.y + r * 0.8 - r * 1.6 * *h),
                    ],
                    Stroke::new(2.2_f32, color),
                );
            }
        }
        _ => {
            // Hardware Observatory: silicon die with pins
            p.rect_stroke(
                egui::Rect::from_center_size(c, Vec2::splat(r * 1.25)),
                2.0,
                st,
                egui::StrokeKind::Inside,
            );
            p.circle_filled(c, 2.2, color);
        }
    }
}

/// Interactive horizontal treemap + radial composition ring for QMF v1 Storage Architecture.
pub fn draw_qmf_storage_visuals(
    ui: &mut egui::Ui,
    segments: &[(String, u64, Color32)],
    total_bytes: u64,
) {
    if segments.is_empty() || total_bytes == 0 {
        return;
    }
    ui.horizontal_top(|ui| {
        // Left: Radial Precision & Overhead Ring Diagram
        let ring_size = 148.0;
        let (ring_rect, _) = ui.allocate_exact_size(Vec2::splat(ring_size), egui::Sense::hover());
        let p = ui.painter();
        let center = ring_rect.center();
        let rad = ring_size * 0.40;
        let mut start_a = -std::f32::consts::FRAC_PI_2;
        for (_, bytes, col) in segments {
            let frac = (*bytes as f64 / total_bytes as f64).clamp(0.0, 1.0) as f32;
            let sweep = frac * std::f32::consts::TAU;
            let steps = ((sweep * 18.0).ceil() as usize).max(2);
            let mut pts = Vec::with_capacity(steps + 1);
            for s in 0..=steps {
                let a = start_a + sweep * (s as f32 / steps as f32);
                pts.push(egui::pos2(
                    center.x + rad * a.cos(),
                    center.y + rad * a.sin(),
                ));
            }
            p.add(egui::Shape::line(pts, Stroke::new(11.0_f32, *col)));
            start_a += sweep;
        }
        p.text(
            center,
            egui::Align2::CENTER_CENTER,
            format_bytes(total_bytes),
            egui::FontId::monospace(11.5),
            TEXT_PRIMARY,
        );

        ui.add_space(14.0);

        // Right: Proportional Storage Composition Treemap Strip + Legend
        ui.vertical(|ui| {
            ui.label(
                RichText::new("STORAGE ARCHITECTURE COMPOSITION (PROPORTIONAL BYTES)")
                    .color(CYAN)
                    .size(11.5)
                    .strong(),
            );
            ui.add_space(6.0);
            let bar_w = (ui.available_width() - 8.0).max(260.0);
            let (bar_rect, _) =
                ui.allocate_exact_size(Vec2::new(bar_w, 34.0), egui::Sense::hover());
            let bp = ui.painter();
            bp.rect_filled(bar_rect, 6.0, OBSIDIAN_BG);
            let mut cursor_x = bar_rect.left();
            for (name, bytes, col) in segments {
                let frac = (*bytes as f64 / total_bytes as f64).clamp(0.0, 1.0) as f32;
                let seg_w = (frac * bar_rect.width()).max(2.0);
                let seg_rect = egui::Rect::from_min_size(
                    egui::pos2(cursor_x, bar_rect.top()),
                    Vec2::new(seg_w.min(bar_rect.right() - cursor_x), bar_rect.height()),
                );
                bp.rect_filled(seg_rect.shrink(1.0), 4.0, *col);
                if seg_w > 54.0 {
                    bp.text(
                        seg_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        format!("{:.0}%", frac * 100.0),
                        egui::FontId::monospace(10.5),
                        OBSIDIAN_BG,
                    );
                }
                cursor_x += seg_w;
                let _ = name;
            }
            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                for (name, bytes, col) in segments {
                    let pct = (*bytes as f64 / total_bytes as f64) * 100.0;
                    pill_badge(
                        ui,
                        &format!("{name}: {} ({pct:.1}%)", format_bytes(*bytes)),
                        *col,
                    );
                }
            });
        });
    });
}

/// Compact ambient neural-pulse strip for Q-Core Inference Nexus (explicitly artistic/ambient).
pub fn draw_ambient_inference_pulse(ui: &mut egui::Ui, active: bool, time_sec: f64) {
    let w = ui.available_width().max(240.0);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(w, 28.0), egui::Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, 6.0, Color32::from_rgb(9, 11, 22));
    p.rect_stroke(
        rect,
        6.0,
        Stroke::new(0.9_f32, BORDER_SUBTLE),
        egui::StrokeKind::Inside,
    );

    let speed = if active { 3.6 } else { 0.7 };
    let amp = if active { 8.5 } else { 2.5 };
    let samples = 72;
    let mut pts = Vec::with_capacity(samples);
    for i in 0..samples {
        let u = i as f32 / (samples - 1) as f32;
        let x = rect.left() + 8.0 + u * (rect.width() - 16.0);
        let env = (u * std::f32::consts::PI).sin();
        let wave = ((u * 18.0 - time_sec as f32 * speed).sin()
            + 0.45 * (u * 31.0 + time_sec as f32 * speed * 1.4).cos())
            * amp
            * env;
        pts.push(egui::pos2(x, rect.center().y + wave));
    }
    let col = if active { CYAN } else { ULTRAVIOLET };
    p.add(egui::Shape::line(pts, Stroke::new(1.6_f32, col)));
    p.text(
        egui::pos2(rect.right() - 10.0, rect.center().y),
        egui::Align2::RIGHT_CENTER,
        if active {
            "AMBIENT STREAMING PULSE (ILLUSTRATIVE)"
        } else {
            "AMBIENT IDLE FIELD"
        },
        egui::FontId::monospace(9.5),
        TEXT_MUTED,
    );
}
