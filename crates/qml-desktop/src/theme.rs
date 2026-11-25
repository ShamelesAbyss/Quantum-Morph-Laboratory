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
