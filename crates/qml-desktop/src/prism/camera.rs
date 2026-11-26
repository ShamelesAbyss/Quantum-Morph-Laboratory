//! Pure safe-Rust 3D linear algebra, Perspective/Orthographic Orbit Camera, and Ray Picking.

use eframe::egui::{self, Pos2, Rect, Vec2};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default)]
pub struct Vec3(pub [f32; 3]);

impl Vec3 {
    pub const ZERO: Self = Self([0.0, 0.0, 0.0]);
    pub const UP: Self = Self([0.0, 1.0, 0.0]);

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self([x, y, z])
    }

    pub fn x(self) -> f32 {
        self.0[0]
    }
    pub fn y(self) -> f32 {
        self.0[1]
    }
    pub fn z(self) -> f32 {
        self.0[2]
    }

    pub fn add(self, o: Self) -> Self {
        Self([self.0[0] + o.0[0], self.0[1] + o.0[1], self.0[2] + o.0[2]])
    }

    pub fn sub(self, o: Self) -> Self {
        Self([self.0[0] - o.0[0], self.0[1] - o.0[1], self.0[2] - o.0[2]])
    }

    pub fn scale(self, s: f32) -> Self {
        Self([self.0[0] * s, self.0[1] * s, self.0[2] * s])
    }

    pub fn dot(self, o: Self) -> f32 {
        self.0[0] * o.0[0] + self.0[1] * o.0[1] + self.0[2] * o.0[2]
    }

    pub fn cross(self, o: Self) -> Self {
        Self([
            self.0[1] * o.0[2] - self.0[2] * o.0[1],
            self.0[2] * o.0[0] - self.0[0] * o.0[2],
            self.0[0] * o.0[1] - self.0[1] * o.0[0],
        ])
    }

    pub fn len_sq(self) -> f32 {
        self.dot(self)
    }

    pub fn len(self) -> f32 {
        self.len_sq().sqrt()
    }

    pub fn normalize(self) -> Self {
        let l = self.len();
        if l > 1e-7 {
            self.scale(1.0 / l)
        } else {
            Self([0.0, 0.0, 1.0])
        }
    }
}

/// Column-major 4x4 matrix matching WGSL `mat4x4<f32>` layout.
#[derive(Debug, Clone, Copy)]
pub struct Mat4(pub [[f32; 4]; 4]);

impl Mat4 {
    pub fn mul(self, rhs: Self) -> Self {
        let mut out = [[0.0_f32; 4]; 4];
        for (c, out_col) in out.iter_mut().enumerate() {
            for (r, cell) in out_col.iter_mut().enumerate() {
                *cell = self.0[0][r] * rhs.0[c][0]
                    + self.0[1][r] * rhs.0[c][1]
                    + self.0[2][r] * rhs.0[c][2]
                    + self.0[3][r] * rhs.0[c][3];
            }
        }
        Self(out)
    }

    pub fn mul_vec4(self, v: [f32; 4]) -> [f32; 4] {
        [
            self.0[0][0] * v[0] + self.0[1][0] * v[1] + self.0[2][0] * v[2] + self.0[3][0] * v[3],
            self.0[0][1] * v[0] + self.0[1][1] * v[1] + self.0[2][1] * v[2] + self.0[3][1] * v[3],
            self.0[0][2] * v[0] + self.0[1][2] * v[1] + self.0[2][2] * v[2] + self.0[3][2] * v[3],
            self.0[0][3] * v[0] + self.0[1][3] * v[1] + self.0[2][3] * v[2] + self.0[3][3] * v[3],
        ]
    }

    /// Right-handed look-at view matrix.
    pub fn look_at_rh(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let f = target.sub(eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        Self([
            [s.x(), u.x(), -f.x(), 0.0],
            [s.y(), u.y(), -f.y(), 0.0],
            [s.z(), u.z(), -f.z(), 0.0],
            [-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0],
        ])
    }

    /// Right-handed perspective projection for WGPU depth range `[0, 1]`.
    pub fn perspective_rh_wgpu(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y_rad * 0.5).tan();
        let nf = 1.0 / (near - far);
        Self([
            [f / aspect.max(1e-4), 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, far * nf, -1.0],
            [0.0, 0.0, far * near * nf, 0.0],
        ])
    }

    /// Right-handed orthographic projection for WGPU depth range `[0, 1]`.
    pub fn orthographic_rh_wgpu(half_h: f32, aspect: f32, near: f32, far: f32) -> Self {
        let half_w = (half_h * aspect.max(1e-4)).max(1e-4);
        let hh = half_h.max(1e-4);
        let nf = 1.0 / (near - far);
        Self([
            [1.0 / half_w, 0.0, 0.0, 0.0],
            [0.0, 1.0 / hh, 0.0, 0.0],
            [0.0, 0.0, nf, 0.0],
            [0.0, 0.0, near * nf, 1.0],
        ])
    }
}

/// Interactive 3D scientific camera with orbit, pan, zoom, and perspective/orthographic toggle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: [f32; 3],
    pub fov_y_deg: f32,
    pub orthographic: bool,
    pub auto_rotate: f32,
    pub default_yaw: f32,
    pub default_pitch: f32,
    pub default_distance: f32,
}

impl OrbitCamera {
    pub fn new(yaw: f32, pitch: f32, distance: f32, auto_rotate: f32) -> Self {
        Self {
            yaw,
            pitch,
            distance,
            target: [0.0, 0.0, 0.0],
            fov_y_deg: 44.0,
            orthographic: false,
            auto_rotate,
            default_yaw: yaw,
            default_pitch: pitch,
            default_distance: distance,
        }
    }

    pub fn reset(&mut self) {
        self.yaw = self.default_yaw;
        self.pitch = self.default_pitch;
        self.distance = self.default_distance;
        self.target = [0.0, 0.0, 0.0];
    }

    pub fn eye(&self) -> Vec3 {
        let cp = self.pitch.cos();
        let x = self.distance * cp * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * cp * self.yaw.cos();
        Vec3::new(self.target[0] + x, self.target[1] + y, self.target[2] + z)
    }

    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let eye = self.eye();
        let target = Vec3(self.target);
        let forward = target.sub(eye).normalize();
        let right = forward.cross(Vec3::UP).normalize();
        let up = right.cross(forward).normalize();
        (right, up, forward)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), Vec3(self.target), Vec3::UP);
        let proj = if self.orthographic {
            let half_h = self.distance * (self.fov_y_deg.to_radians() * 0.5).tan();
            Mat4::orthographic_rh_wgpu(half_h, aspect, 0.05, 250.0)
        } else {
            Mat4::perspective_rh_wgpu(self.fov_y_deg.to_radians(), aspect, 0.05, 250.0)
        };
        proj.mul(view)
    }

    /// Project a 3D world point into egui screen coordinates inside `rect`.
    pub fn project_to_screen(&self, world: [f32; 3], rect: Rect) -> Option<(Pos2, f32)> {
        let aspect = (rect.width() / rect.height().max(1.0)).max(0.01);
        let clip = self
            .view_proj(aspect)
            .mul_vec4([world[0], world[1], world[2], 1.0]);
        if clip[3] <= 1e-4 {
            return None;
        }
        let ndc_x = clip[0] / clip[3];
        let ndc_y = clip[1] / clip[3];
        let ndc_z = clip[2] / clip[3];
        if !(0.0..=1.0).contains(&ndc_z) {
            return None;
        }
        let sx = rect.left() + (ndc_x * 0.5 + 0.5) * rect.width();
        let sy = rect.top() + (0.5 - ndc_y * 0.5) * rect.height();
        Some((Pos2::new(sx, sy), ndc_z))
    }

    /// Compute a world-space picking ray `(origin, dir)` from an egui screen position inside `rect`.
    pub fn screen_ray(&self, pos: Pos2, rect: Rect) -> (Vec3, Vec3) {
        let aspect = (rect.width() / rect.height().max(1.0)).max(0.01);
        let ndc_x = ((pos.x - rect.left()) / rect.width().max(1.0)) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((pos.y - rect.top()) / rect.height().max(1.0)) * 2.0;
        let (right, up, forward) = self.basis();
        let tan_half = (self.fov_y_deg.to_radians() * 0.5).tan();
        if self.orthographic {
            let half_h = self.distance * tan_half;
            let half_w = half_h * aspect;
            let origin = self
                .eye()
                .add(right.scale(ndc_x * half_w))
                .add(up.scale(ndc_y * half_h));
            (origin, forward)
        } else {
            let dir = forward
                .add(right.scale(ndc_x * tan_half * aspect))
                .add(up.scale(ndc_y * tan_half))
                .normalize();
            (self.eye(), dir)
        }
    }

    /// Apply smooth orbit, pan, zoom, and optional auto-rotation from egui pointer events.
    pub fn update_from_interaction(
        &mut self,
        ui: &egui::Ui,
        resp: &egui::Response,
        dt: f32,
        animate: bool,
    ) {
        if resp.dragged_by(egui::PointerButton::Primary) {
            let delta = resp.drag_delta();
            if ui.input(|i| i.modifiers.shift) {
                self.pan(delta);
            } else {
                self.yaw -= delta.x * 0.0065;
                self.pitch = (self.pitch + delta.y * 0.0065).clamp(-1.42, 1.42);
            }
        } else if resp.dragged_by(egui::PointerButton::Secondary)
            || resp.dragged_by(egui::PointerButton::Middle)
        {
            self.pan(resp.drag_delta());
        } else if animate && self.auto_rotate.abs() > 1e-5 {
            self.yaw += self.auto_rotate * dt;
        }

        if resp.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll.abs() > 0.1 {
                let factor = (-scroll * 0.0018).exp();
                self.distance = (self.distance * factor).clamp(1.8, 42.0);
            }
        }

        if resp.double_clicked() {
            self.reset();
        }
    }

    fn pan(&mut self, delta: Vec2) {
        let (right, up, _) = self.basis();
        let speed = self.distance * 0.0016;
        let shift = right.scale(-delta.x * speed).add(up.scale(delta.y * speed));
        self.target[0] += shift.x();
        self.target[1] += shift.y();
        self.target[2] += shift.z();
    }
}

/// Ray-sphere intersection returning closest positive distance along ray, if hit.
pub fn ray_sphere_hit(origin: Vec3, dir: Vec3, center: [f32; 3], radius: f32) -> Option<f32> {
    let oc = origin.sub(Vec3(center));
    let b = oc.dot(dir);
    let c = oc.len_sq() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    if t > 0.0 { Some(t) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit_camera_projects_origin_to_viewport_center_and_picks_sphere() {
        let cam = OrbitCamera::new(0.0, 0.0, 10.0, 0.0);
        let rect = Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(1000.0, 500.0));
        let (pos, z) = cam.project_to_screen([0.0, 0.0, 0.0], rect).unwrap();
        assert!((pos.x - 500.0).abs() < 1e-3);
        assert!((pos.y - 250.0).abs() < 1e-3);
        assert!(z > 0.0 && z < 1.0);

        let (ro, rd) = cam.screen_ray(Pos2::new(500.0, 250.0), rect);
        let hit = ray_sphere_hit(ro, rd, [0.0, 0.0, 0.0], 1.0);
        assert!(hit.is_some());
        assert!((hit.unwrap() - 9.0).abs() < 1e-3);
    }
}
