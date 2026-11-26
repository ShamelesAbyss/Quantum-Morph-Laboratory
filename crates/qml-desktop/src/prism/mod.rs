//! PROJECT PRISM — Native WGPU 3D Scientific Visualization Subsystem.

pub mod camera;
pub mod renderer;
pub mod scene;
pub mod scenes;
pub mod shaders;

pub use camera::{OrbitCamera, ray_sphere_hit};
pub use renderer::{init_prism_renderer, make_wgpu_paint_callback, paint_cpu_fallback};
pub use scene::{GraphicsPreset, GraphicsSettings, PickableNode, PrismScene};

use crate::theme::{self, BORDER_GLOW, CYAN, TEXT_MUTED, TEXT_PRIMARY};
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use std::sync::Arc;

/// Result of rendering an interactive 3D PRISM viewport for one frame.
pub struct PrismViewportResult {
    pub clicked_id: Option<String>,
    pub fullscreen_toggled: bool,
}

/// Render an interactive 3D WGPU PRISM viewport with Orbit/Pan/Zoom, 3D ray picking,
/// screen-projected scientific HUD labels, explanatory legend, and camera controls.
#[allow(clippy::too_many_arguments)]
pub fn render_prism_viewport(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    id_salt: &str,
    desired_size: Vec2,
    camera: &mut OrbitCamera,
    scene: PrismScene,
    settings: &GraphicsSettings,
    wgpu_ready: bool,
    time_sec: f32,
    dt: f32,
    selected_id: Option<&str>,
    is_fullscreen: bool,
) -> PrismViewportResult {
    let avail = ui.available_size();
    let size = Vec2::new(
        desired_size.x.min(avail.x).max(220.0),
        desired_size.y.max(200.0),
    );
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click_and_drag());

    // Guard against zero-sized surfaces during minimize/resize
    if rect.width() < 8.0 || rect.height() < 8.0 {
        return PrismViewportResult {
            clicked_id: None,
            fullscreen_toggled: false,
        };
    }

    let animate = settings.effective_anim_scale() > 0.0;
    camera.update_from_interaction(ui, &resp, dt, animate);

    let scene_arc = Arc::new(scene);

    // 3D ray picking for hover & click selection
    let mut hovered_node: Option<&PickableNode> = None;
    if let Some(pointer) = resp.hover_pos() {
        let (ro, rd) = camera.screen_ray(pointer, rect);
        let mut best_t = f32::INFINITY;
        for node in &scene_arc.pickables {
            if let Some(t) = ray_sphere_hit(ro, rd, node.position, node.radius)
                && t < best_t
            {
                best_t = t;
                hovered_node = Some(node);
            }
        }
    }
    let hovered_id = hovered_node.map(|n| n.id.clone());
    let clicked_id = if resp.clicked() {
        hovered_id.clone()
    } else {
        None
    };

    let painter = ui.painter_at(rect);

    // Subtle dark glass border around the 3D observatory viewport
    painter.rect_filled(rect, 10.0, Color32::from_rgb(6, 7, 14));

    if wgpu_ready && settings.preset != GraphicsPreset::Minimal {
        let cb = make_wgpu_paint_callback(
            rect.shrink(1.0),
            ctx.pixels_per_point(),
            Arc::clone(&scene_arc),
            camera.clone(),
            settings.clone(),
            time_sec,
        );
        painter.add(cb);
    } else {
        paint_cpu_fallback(&painter, rect.shrink(1.0), camera, &scene_arc);
    }

    // Screen-projected 3D scientific callout labels
    for lbl in &scene_arc.labels {
        if let Some((pos, _)) = camera.project_to_screen(lbl.world_pos, rect)
            && rect.shrink(18.0).contains(pos)
        {
            let font = FontId::monospace(if lbl.prominent { 11.5 } else { 10.0 });
            let galley = painter.layout_no_wrap(lbl.text.clone(), font, lbl.color);
            let bg_rect = Rect::from_center_size(
                Pos2::new(pos.x, pos.y - 12.0),
                galley.size() + Vec2::new(10.0, 4.0),
            );
            painter.rect_filled(
                bg_rect,
                4.0,
                Color32::from_rgba_unmultiplied(8, 10, 20, 195),
            );
            painter.rect_stroke(
                bg_rect,
                4.0,
                Stroke::new(
                    0.8_f32,
                    Color32::from_rgba_unmultiplied(
                        lbl.color.r(),
                        lbl.color.g(),
                        lbl.color.b(),
                        140,
                    ),
                ),
                egui::StrokeKind::Inside,
            );
            painter.galley(bg_rect.min + Vec2::new(5.0, 2.0), galley, lbl.color);
        }
    }

    // Highlight hovered or selected 3D node with a clean scientific reticle + tooltip
    if let Some(node) = hovered_node
        && let Some((pos, _)) = camera.project_to_screen(node.position, rect)
    {
        painter.circle_stroke(pos, 14.0, Stroke::new(1.4_f32, CYAN));
        painter.circle_stroke(
            pos,
            20.0,
            Stroke::new(0.8_f32, Color32::from_rgba_unmultiplied(167, 139, 250, 160)),
        );
        let tip = format!("{}  •  {}", node.label, node.detail);
        let galley = painter.layout_no_wrap(tip, FontId::proportional(12.0), TEXT_PRIMARY);
        let tip_pos = Pos2::new(
            (pos.x + 18.0)
                .min(rect.right() - galley.size().x - 18.0)
                .max(rect.left() + 10.0),
            (pos.y - 26.0).max(rect.top() + 44.0),
        );
        let tip_rect = Rect::from_min_size(tip_pos, galley.size() + Vec2::new(14.0, 8.0));
        painter.rect_filled(
            tip_rect,
            6.0,
            Color32::from_rgba_unmultiplied(10, 12, 24, 232),
        );
        painter.rect_stroke(
            tip_rect,
            6.0,
            Stroke::new(1.0_f32, node.accent),
            egui::StrokeKind::Inside,
        );
        painter.galley(tip_rect.min + Vec2::new(7.0, 4.0), galley, TEXT_PRIMARY);
    } else if let Some(sel) = selected_id
        && let Some(node) = scene_arc.pickables.iter().find(|n| n.id == sel)
        && let Some((pos, _)) = camera.project_to_screen(node.position, rect)
        && rect.shrink(12.0).contains(pos)
    {
        painter.circle_stroke(pos, 13.0, Stroke::new(1.3_f32, CYAN));
    }

    // Outer crystalline frame stroke
    painter.rect_stroke(
        rect,
        10.0,
        Stroke::new(1.1_f32, BORDER_GLOW),
        egui::StrokeKind::Inside,
    );

    // Top-left floating HUD title + scientific provenance pill
    let hud_top_left = rect.left_top() + Vec2::new(14.0, 12.0);
    let badge_galley = painter.layout_no_wrap(
        scene_arc.provenance_badge.clone(),
        FontId::monospace(10.0),
        scene_arc.provenance_color,
    );
    let badge_rect = Rect::from_min_size(hud_top_left, badge_galley.size() + Vec2::new(12.0, 4.0));
    painter.rect_filled(
        badge_rect,
        6.0,
        Color32::from_rgba_unmultiplied(8, 10, 20, 210),
    );
    painter.rect_stroke(
        badge_rect,
        6.0,
        Stroke::new(0.9_f32, scene_arc.provenance_color),
        egui::StrokeKind::Inside,
    );
    painter.galley(
        badge_rect.min + Vec2::new(6.0, 2.0),
        badge_galley,
        scene_arc.provenance_color,
    );

    if !scene_arc.title.is_empty() {
        painter.text(
            Pos2::new(hud_top_left.x, badge_rect.bottom() + 6.0),
            Align2::LEFT_TOP,
            &scene_arc.title,
            FontId::proportional(14.5),
            TEXT_PRIMARY,
        );
    }
    if !scene_arc.subtitle.is_empty() {
        painter.text(
            Pos2::new(hud_top_left.x, badge_rect.bottom() + 24.0),
            Align2::LEFT_TOP,
            &scene_arc.subtitle,
            FontId::proportional(11.5),
            TEXT_MUTED,
        );
    }

    // Bottom-left explanatory legend (for scientific color-mapped modes)
    if !scene_arc.legend.is_empty() {
        let mut x = rect.left() + 14.0;
        let y = rect.bottom() - 22.0;
        let bg_w = scene_arc
            .legend
            .iter()
            .map(|l| l.label.len() as f32 * 6.8 + 26.0)
            .sum::<f32>()
            .min(rect.width() - 240.0);
        let lrect = Rect::from_min_size(Pos2::new(x - 6.0, y - 6.0), Vec2::new(bg_w + 12.0, 22.0));
        painter.rect_filled(lrect, 6.0, Color32::from_rgba_unmultiplied(8, 10, 20, 200));
        for item in &scene_arc.legend {
            if x > rect.right() - 220.0 {
                break;
            }
            painter.circle_filled(Pos2::new(x + 4.0, y + 5.0), 4.0, item.color);
            let g = painter.layout_no_wrap(
                item.label.clone(),
                FontId::proportional(10.5),
                TEXT_PRIMARY,
            );
            let w = g.size().x;
            painter.galley(Pos2::new(x + 12.0, y - 1.0), g, TEXT_PRIMARY);
            x += w + 22.0;
        }
    }

    // Top-right compact interactive camera toolbar (Orbit/Ortho/Reset/Fullscreen)
    // IMPORTANT: use `new_child` (not `scope_builder`) so the parent UI's vertical cursor
    // remains at `rect.bottom()` and subsequent cards render cleanly below the 3D viewport.
    let mut fullscreen_toggled = false;
    let bar_rect = Rect::from_min_size(
        Pos2::new(rect.right() - 238.0, rect.top() + 10.0),
        Vec2::new(228.0, 28.0),
    );
    let mut bar_ui = ui.new_child(
        egui::UiBuilder::new()
            .id_salt((id_salt, "prism_cam_bar"))
            .max_rect(bar_rect),
    );
    bar_ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        let fs_label = if is_fullscreen {
            "Exit [F]"
        } else {
            "Expand [F]"
        };
        if ui
            .small_button(RichText::new(fs_label).size(10.5).color(CYAN))
            .clicked()
        {
            fullscreen_toggled = true;
        }
        if ui
            .small_button(RichText::new("Reset").size(10.5).color(TEXT_PRIMARY))
            .clicked()
        {
            camera.reset();
        }
        let proj_label = if camera.orthographic {
            "Ortho"
        } else {
            "Persp"
        };
        if ui
            .small_button(RichText::new(proj_label).size(10.5).color(theme::AMETHYST))
            .on_hover_text("Toggle Perspective / Orthographic projection [P]")
            .clicked()
        {
            camera.orthographic = !camera.orthographic;
        }
    });

    // Bottom-right subtle interaction hint
    painter.text(
        Pos2::new(rect.right() - 12.0, rect.bottom() - 10.0),
        Align2::RIGHT_BOTTOM,
        "Drag: Orbit  •  Shift/Right-Drag: Pan  •  Scroll: Zoom  •  Click: Inspect",
        FontId::proportional(10.0),
        Color32::from_rgba_unmultiplied(155, 168, 208, 165),
    );

    PrismViewportResult {
        clicked_id,
        fullscreen_toggled,
    }
}
