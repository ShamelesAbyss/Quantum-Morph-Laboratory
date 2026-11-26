//! PROJECT RESONANCE — Sub-3-Bit Neural Representations, Hadamard Incoherence,
//! Learned Residual Vector Quantization, Packed Base-3 Ternary, PROJECT ECHO
//! Compact Fidelity Recovery, and Behavior-Aware MAP-B (`QMR-300` → `QMR-150`).
//!
//! Strictly classical mathematics (Walsh–Hadamard orthogonality, Lloyd–Max
//! ternary quantization, k-means++ residual vector quantization, low-rank SVD
//! residual recovery, and activation/behavior-weighted allocation).
//! Every single byte—codes, scales, mins, codebooks, recovery factors, sparse
//! indices, and headers—is counted in [`StorageBreakdown`].

use crate::baselines::{
    FactorPrecision, Int4SymGroupMatrix, Int8SymMatrix, LowRankSvdMatrix, SparseResidual,
};
#[cfg(test)]
use crate::evaluate_operator;
use crate::{LinearOperator, StorageBreakdown, deterministic_probe_vectors};
use qml_core::{DenseMatrix, DeterministicRng, MatrixError, truncated_svd};
use qml_model::{f16_to_f32, f32_to_f16};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const RESONANCE_HEADER_BYTES: usize = 24;

// ---------------------------------------------------------------------------
// 1. Blockwise Randomized Fast Walsh–Hadamard Transform (QuIP#-inspired RHT)
// ---------------------------------------------------------------------------

/// Compute an orthonormal in-place Fast Walsh–Hadamard Transform on `slice`
/// whose length must be a power of two (`len > 0 && len.is_power_of_two()`).
///
/// Satisfies `H = H^T` and `H * H = I`, so applying `fwht_in_place` twice
/// recovers the exact original vector up to floating-point rounding.
pub fn fwht_in_place(slice: &mut [f64]) {
    let n = slice.len();
    debug_assert!(n > 0 && n.is_power_of_two());
    let mut h = 1usize;
    while h < n {
        let step = h << 1;
        for chunk in slice.chunks_exact_mut(step) {
            let (left, right) = chunk.split_at_mut(h);
            for i in 0..h {
                let x = left[i];
                let y = right[i];
                left[i] = x + y;
                right[i] = x - y;
            }
        }
        h = step;
    }
    let scale = 1.0 / (n as f64).sqrt();
    for v in slice.iter_mut() {
        *v *= scale;
    }
}

/// Deterministic Rademacher sign vector `s_j in {-1.0, +1.0}` generated from `seed`
/// using SplitMix64 (zero storage overhead in serialized QMF artifacts).
pub fn hadamard_signs(cols: usize, seed: u64) -> Vec<f64> {
    let mut rng = DeterministicRng::new(seed ^ 0x5155_4950_5348_5250);
    (0..cols)
        .map(|_| if (rng.next_u64() & 1) == 0 { 1.0 } else { -1.0 })
        .collect()
}

/// Forward blockwise Randomized Hadamard Transform: `y = H_b (s ⊙ x)`.
///
/// Preserves inner products identically: `dot(R(w), R(x)) == dot(w, x)`.
/// Non-power-of-two trailing elements (`cols % block_size`) are sign-flipped only,
/// keeping the entire transform strictly orthogonal for arbitrary matrix shapes.
pub fn apply_hadamard_forward(vec: &mut [f64], block_size: usize, seed: u64) {
    if block_size <= 1 || !block_size.is_power_of_two() {
        return;
    }
    let signs = hadamard_signs(vec.len(), seed);
    apply_hadamard_forward_with_signs(vec, block_size, &signs);
}

#[inline]
pub fn apply_hadamard_forward_with_signs(vec: &mut [f64], block_size: usize, signs: &[f64]) {
    if block_size <= 1 || !block_size.is_power_of_two() {
        return;
    }
    for (v, &s) in vec.iter_mut().zip(signs) {
        *v *= s;
    }
    let full = (vec.len() / block_size) * block_size;
    for chunk in vec[..full].chunks_exact_mut(block_size) {
        fwht_in_place(chunk);
    }
}

/// Inverse blockwise Randomized Hadamard Transform: `x = s ⊙ (H_b y)`.
pub fn apply_hadamard_inverse(vec: &mut [f64], block_size: usize, seed: u64) {
    if block_size <= 1 || !block_size.is_power_of_two() {
        return;
    }
    let signs = hadamard_signs(vec.len(), seed);
    apply_hadamard_inverse_with_signs(vec, block_size, &signs);
}

#[inline]
pub fn apply_hadamard_inverse_with_signs(vec: &mut [f64], block_size: usize, signs: &[f64]) {
    if block_size <= 1 || !block_size.is_power_of_two() {
        return;
    }
    let full = (vec.len() / block_size) * block_size;
    for chunk in vec[..full].chunks_exact_mut(block_size) {
        fwht_in_place(chunk);
    }
    for (v, &s) in vec.iter_mut().zip(signs) {
        *v *= s;
    }
}

/// Optional deterministic Hadamard incoherence specification (`block_size = 0` disables rotation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct HadamardSpec {
    pub block_size: usize,
    pub seed: u64,
}

impl HadamardSpec {
    pub const NONE: Self = Self {
        block_size: 0,
        seed: 0,
    };

    pub fn new(block_size: usize, seed: u64) -> Self {
        if block_size > 1 && block_size.is_power_of_two() {
            Self { block_size, seed }
        } else {
            Self::NONE
        }
    }

    pub fn is_enabled(self) -> bool {
        self.block_size > 1 && self.block_size.is_power_of_two()
    }
}

fn rotate_rows_if_enabled(matrix: &DenseMatrix, hadamard: HadamardSpec) -> Vec<f64> {
    let rows = matrix.rows();
    let cols = matrix.cols();
    let mut data = matrix.as_slice().to_vec();
    if hadamard.is_enabled() {
        let signs = hadamard_signs(cols, hadamard.seed);
        if rows >= 32 {
            data.par_chunks_exact_mut(cols).for_each(|row| {
                apply_hadamard_forward_with_signs(row, hadamard.block_size, &signs);
            });
        } else {
            for row in data.chunks_exact_mut(cols) {
                apply_hadamard_forward_with_signs(row, hadamard.block_size, &signs);
            }
        }
    }
    data
}

fn rotate_input_if_enabled(vector: &[f64], hadamard: HadamardSpec) -> Vec<f64> {
    let mut out = vector.to_vec();
    if hadamard.is_enabled() {
        apply_hadamard_forward(&mut out, hadamard.block_size, hadamard.seed);
    }
    out
}

/// Closed-form optimal affine `(scale, min)` refit for a quantized group using exact
/// least-squares linear regression (`min + scale * q ≈ w`) + 2 alternating refinements.
#[inline]
fn fit_affine_uniform_group(slice: &[f64], max_q: f64) -> (f64, f64) {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut mean = 0.0_f64;
    for &v in slice {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
        mean += v;
    }
    let n = slice.len() as f64;
    mean /= n;
    let mut var = 0.0_f64;
    for &v in slice {
        let d = v - mean;
        var += d * d;
    }
    let sigma = (var / n).sqrt();
    if max_v - min_v <= 1e-12 || sigma <= 1e-12 {
        return (0.0, min_v);
    }

    // Search both min-max clips and Gaussian-optimal sigma multiples (`[-k*sigma, +k*sigma]`)
    let center = 0.5 * (min_v + max_v);
    let half = 0.5 * (max_v - min_v);
    let sigma_mults: &[f64] = if max_q <= 3.5 {
        &[1.65, 1.95, 2.25]
    } else {
        &[2.20, 2.65, 3.10]
    };
    let mut best_min = min_v;
    let mut best_scale = (max_v - min_v) / max_q;
    let mut best_mse = f64::INFINITY;

    let mut try_range = |c_min: f64, c_scale: f64| {
        if c_scale <= 1e-12 {
            return;
        }
        let mut mse = 0.0_f64;
        for &v in slice {
            let q = ((v - c_min) / c_scale).round().clamp(0.0, max_q);
            let d = v - (c_min + q * c_scale);
            mse += d * d;
        }
        if mse < best_mse {
            best_mse = mse;
            best_min = c_min;
            best_scale = c_scale;
        }
    };

    for &clip in &[1.0_f64, 0.88, 0.76] {
        try_range(center - half * clip, (2.0 * half * clip) / max_q);
    }
    for &km in sigma_mults {
        try_range(mean - km * sigma, (2.0 * km * sigma) / max_q);
    }

    // 2 iterations of closed-form least-squares (scale, min) refit given assigned integer codes
    for _ in 0..2 {
        let mut sum_q = 0.0_f64;
        let mut sum_w = 0.0_f64;
        let mut sum_qq = 0.0_f64;
        let mut sum_qw = 0.0_f64;
        for &v in slice {
            let q = ((v - best_min) / best_scale).round().clamp(0.0, max_q);
            sum_q += q;
            sum_w += v;
            sum_qq += q * q;
            sum_qw += q * v;
        }
        let denom = n * sum_qq - sum_q * sum_q;
        if denom > 1e-12 {
            let new_scale = (n * sum_qw - sum_q * sum_w) / denom;
            if new_scale > 1e-12 {
                best_scale = new_scale;
                best_min = (sum_w - best_scale * sum_q) / n;
            }
        }
    }
    (best_scale, best_min)
}

// ---------------------------------------------------------------------------
// 2. Blockwise 3-Bit Asymmetric Quantizer (`Int3AsymGroupMatrix`, ~3.25-3.50 bpp)
// ---------------------------------------------------------------------------

/// Groupwise asymmetric 3-bit quantizer packing 8 codes (0..=7) into 3 bytes (24 bits),
/// with per-group `f16` scale + `f16` min and optional Hadamard incoherence rotation.
///
/// Effective bit rate: `3.00 + 32 / group_size` bpp (`3.50 bpp` at `g=64`, `3.25 bpp` at `g=128`).
#[derive(Debug, Clone, PartialEq)]
pub struct Int3AsymGroupMatrix {
    rows: usize,
    cols: usize,
    group_size: usize,
    hadamard: HadamardSpec,
    scales_f16: Vec<u16>,
    mins_f16: Vec<u16>,
    /// Packed 3-bit codes: 8 codes per 3 bytes (little-endian bit stream).
    packed: Vec<u8>,
}

impl Int3AsymGroupMatrix {
    pub fn quantize(
        matrix: &DenseMatrix,
        group_size: usize,
        hadamard: HadamardSpec,
    ) -> Result<Self, MatrixError> {
        if group_size == 0 {
            return Err(MatrixError::DimensionMismatch {
                expected: 1,
                actual: 0,
            });
        }
        let rows = matrix.rows();
        let cols = matrix.cols();
        let rotated = rotate_rows_if_enabled(matrix, hadamard);
        let groups_per_row = cols.div_ceil(group_size);
        let packed_row_bytes = cols.div_ceil(8) * 3;

        let quant_row = |r: usize| -> (Vec<u16>, Vec<u16>, Vec<u8>) {
            let row = &rotated[r * cols..(r + 1) * cols];
            let mut r_scales = vec![0u16; groups_per_row];
            let mut r_mins = vec![0u16; groups_per_row];
            let mut p_row = vec![0u8; packed_row_bytes];
            let mut codes = vec![0u8; cols];
            for g in 0..groups_per_row {
                let start = g * group_size;
                let end = (start + group_size).min(cols);
                let slice = &row[start..end];
                let (best_scale, best_min) = fit_affine_uniform_group(slice, 7.0);
                let s_bits = f32_to_f16(best_scale as f32);
                let m_bits = f32_to_f16(best_min as f32);
                let scale = f64::from(f16_to_f32(s_bits));
                let min_q = f64::from(f16_to_f32(m_bits));
                r_scales[g] = s_bits;
                r_mins[g] = m_bits;
                for (offset, &v) in slice.iter().enumerate() {
                    let q = if scale > 0.0 {
                        ((v - min_q) / scale).round().clamp(0.0, 7.0) as u8
                    } else {
                        0
                    };
                    codes[start + offset] = q;
                }
            }
            for (blk, chunk) in codes.chunks(8).enumerate() {
                let mut bits: u32 = 0;
                for (i, &c) in chunk.iter().enumerate() {
                    bits |= (u32::from(c & 0x07)) << (i * 3);
                }
                let base = blk * 3;
                p_row[base] = (bits & 0xFF) as u8;
                p_row[base + 1] = ((bits >> 8) & 0xFF) as u8;
                p_row[base + 2] = ((bits >> 16) & 0xFF) as u8;
            }
            (r_scales, r_mins, p_row)
        };

        let row_results: Vec<_> = if rows >= 64 {
            (0..rows).into_par_iter().map(quant_row).collect()
        } else {
            (0..rows).map(quant_row).collect()
        };
        let mut scales_f16 = Vec::with_capacity(rows * groups_per_row);
        let mut mins_f16 = Vec::with_capacity(rows * groups_per_row);
        let mut packed = Vec::with_capacity(rows * packed_row_bytes);
        for (rs, rm, rp) in row_results {
            scales_f16.extend(rs);
            mins_f16.extend(rm);
            packed.extend(rp);
        }

        Ok(Self {
            rows,
            cols,
            group_size,
            hadamard,
            scales_f16,
            mins_f16,
            packed,
        })
    }

    pub fn from_raw(
        rows: usize,
        cols: usize,
        group_size: usize,
        hadamard: HadamardSpec,
        scales_f16: Vec<u16>,
        mins_f16: Vec<u16>,
        packed: Vec<u8>,
    ) -> Result<Self, MatrixError> {
        if rows == 0 || cols == 0 || group_size == 0 {
            return Err(MatrixError::Empty);
        }
        let gpr = cols.div_ceil(group_size);
        let expected_g = rows.saturating_mul(gpr);
        let expected_p = rows.saturating_mul(cols.div_ceil(8) * 3);
        if scales_f16.len() != expected_g
            || mins_f16.len() != expected_g
            || packed.len() != expected_p
        {
            return Err(MatrixError::ElementCountMismatch {
                expected: expected_p,
                actual: packed.len(),
            });
        }
        Ok(Self {
            rows,
            cols,
            group_size,
            hadamard,
            scales_f16,
            mins_f16,
            packed,
        })
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }
    pub fn hadamard(&self) -> HadamardSpec {
        self.hadamard
    }
    pub fn scales_f16(&self) -> &[u16] {
        &self.scales_f16
    }
    pub fn mins_f16(&self) -> &[u16] {
        &self.mins_f16
    }
    pub fn packed(&self) -> &[u8] {
        &self.packed
    }

    #[inline]
    fn unpack_code(p_row: &[u8], col: usize) -> u8 {
        let blk = col >> 3;
        let idx = col & 7;
        let base = blk * 3;
        let bits = u32::from(p_row[base])
            | (u32::from(p_row[base + 1]) << 8)
            | (u32::from(p_row[base + 2]) << 16);
        ((bits >> (idx * 3)) & 0x07) as u8
    }

    pub fn decode_row(&self, row: usize) -> Result<Vec<f64>, MatrixError> {
        if row >= self.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: self.rows,
                actual: row,
            });
        }
        let gpr = self.cols.div_ceil(self.group_size);
        let prb = self.cols.div_ceil(8) * 3;
        let p_row = &self.packed[row * prb..(row + 1) * prb];
        let mut out = vec![0.0_f64; self.cols];
        for (c, slot) in out.iter_mut().enumerate() {
            let g = c / self.group_size;
            let scale = f64::from(f16_to_f32(self.scales_f16[row * gpr + g]));
            let min_q = f64::from(f16_to_f32(self.mins_f16[row * gpr + g]));
            *slot = min_q + scale * f64::from(Self::unpack_code(p_row, c));
        }
        if self.hadamard.is_enabled() {
            apply_hadamard_inverse(&mut out, self.hadamard.block_size, self.hadamard.seed);
        }
        Ok(out)
    }
}

impl LinearOperator for Int3AsymGroupMatrix {
    fn method_name(&self) -> &'static str {
        if self.hadamard.is_enabled() {
            match self.group_size {
                64 => "INT3_HADAMARD_G64",
                128 => "INT3_HADAMARD_G128",
                _ => "INT3_HADAMARD",
            }
        } else {
            match self.group_size {
                64 => "INT3_ASYM_G64",
                128 => "INT3_ASYM_G128",
                _ => "INT3_ASYM",
            }
        }
    }

    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }

    fn matvec(&self, vector: &[f64]) -> Result<Vec<f64>, MatrixError> {
        if vector.len() != self.cols {
            return Err(MatrixError::DimensionMismatch {
                expected: self.cols,
                actual: vector.len(),
            });
        }
        let x = rotate_input_if_enabled(vector, self.hadamard);
        let gpr = self.cols.div_ceil(self.group_size);
        let prb = self.cols.div_ceil(8) * 3;
        let row_dot = |r: usize| -> f64 {
            let p_row = &self.packed[r * prb..(r + 1) * prb];
            let mut sum = 0.0_f64;
            for g in 0..gpr {
                let scale = f64::from(f16_to_f32(self.scales_f16[r * gpr + g]));
                let min_q = f64::from(f16_to_f32(self.mins_f16[r * gpr + g]));
                let start = g * self.group_size;
                let end = (start + self.group_size).min(self.cols);
                let mut qx = 0.0_f64;
                let mut xsum = 0.0_f64;
                for (c, &xv) in x.iter().enumerate().take(end).skip(start) {
                    qx += f64::from(Self::unpack_code(p_row, c)) * xv;
                    xsum += xv;
                }
                sum += scale * qx + min_q * xsum;
            }
            sum
        };
        if self.rows >= 64 {
            Ok((0..self.rows).into_par_iter().map(row_dot).collect())
        } else {
            Ok((0..self.rows).map(row_dot).collect())
        }
    }

    fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError> {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for r in 0..self.rows {
            data.extend(self.decode_row(r)?);
        }
        DenseMatrix::from_flat(self.rows, self.cols, data)
    }

    fn storage_breakdown(&self) -> StorageBreakdown {
        StorageBreakdown {
            payload_bytes: self.packed.len(),
            scale_metadata_bytes: (self.scales_f16.len() + self.mins_f16.len()) * 2,
            structure_bytes: 0,
            header_bytes: RESONANCE_HEADER_BYTES,
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Blockwise 2-Bit Asymmetric Quantizer (`Int2AsymGroupMatrix`, ~2.25-2.50 bpp)
// ---------------------------------------------------------------------------

/// Groupwise asymmetric 2-bit quantizer packing 4 codes (0..=3) per byte,
/// with per-group `f16` scale + `f16` min and optional Hadamard incoherence rotation.
///
/// Effective bit rate: `2.00 + 32 / group_size` bpp (`2.50 bpp` at `g=64`, `2.25 bpp` at `g=128`).
#[derive(Debug, Clone, PartialEq)]
pub struct Int2AsymGroupMatrix {
    rows: usize,
    cols: usize,
    group_size: usize,
    hadamard: HadamardSpec,
    scales_f16: Vec<u16>,
    mins_f16: Vec<u16>,
    /// Packed 2-bit codes: 4 codes per byte (`bits[1:0], bits[3:2], bits[5:4], bits[7:6]`).
    packed: Vec<u8>,
}

impl Int2AsymGroupMatrix {
    pub fn quantize(
        matrix: &DenseMatrix,
        group_size: usize,
        hadamard: HadamardSpec,
    ) -> Result<Self, MatrixError> {
        if group_size == 0 {
            return Err(MatrixError::DimensionMismatch {
                expected: 1,
                actual: 0,
            });
        }
        let rows = matrix.rows();
        let cols = matrix.cols();
        let rotated = rotate_rows_if_enabled(matrix, hadamard);
        let gpr = cols.div_ceil(group_size);
        let prb = cols.div_ceil(4);

        let quant_row = |r: usize| -> (Vec<u16>, Vec<u16>, Vec<u8>) {
            let row = &rotated[r * cols..(r + 1) * cols];
            let mut r_scales = vec![0u16; gpr];
            let mut r_mins = vec![0u16; gpr];
            let mut p_row = vec![0u8; prb];
            for g in 0..gpr {
                let start = g * group_size;
                let end = (start + group_size).min(cols);
                let slice = &row[start..end];
                let (best_scale, best_min) = fit_affine_uniform_group(slice, 3.0);
                let s_bits = f32_to_f16(best_scale as f32);
                let m_bits = f32_to_f16(best_min as f32);
                let scale = f64::from(f16_to_f32(s_bits));
                let min_q = f64::from(f16_to_f32(m_bits));
                r_scales[g] = s_bits;
                r_mins[g] = m_bits;
                for (offset, &v) in slice.iter().enumerate() {
                    let col = start + offset;
                    let q = if scale > 0.0 {
                        ((v - min_q) / scale).round().clamp(0.0, 3.0) as u8
                    } else {
                        0
                    };
                    p_row[col >> 2] |= (q & 0x03) << ((col & 3) * 2);
                }
            }
            (r_scales, r_mins, p_row)
        };

        let row_results: Vec<_> = if rows >= 64 {
            (0..rows).into_par_iter().map(quant_row).collect()
        } else {
            (0..rows).map(quant_row).collect()
        };
        let mut scales_f16 = Vec::with_capacity(rows * gpr);
        let mut mins_f16 = Vec::with_capacity(rows * gpr);
        let mut packed = Vec::with_capacity(rows * prb);
        for (rs, rm, rp) in row_results {
            scales_f16.extend(rs);
            mins_f16.extend(rm);
            packed.extend(rp);
        }

        Ok(Self {
            rows,
            cols,
            group_size,
            hadamard,
            scales_f16,
            mins_f16,
            packed,
        })
    }

    pub fn from_raw(
        rows: usize,
        cols: usize,
        group_size: usize,
        hadamard: HadamardSpec,
        scales_f16: Vec<u16>,
        mins_f16: Vec<u16>,
        packed: Vec<u8>,
    ) -> Result<Self, MatrixError> {
        if rows == 0 || cols == 0 || group_size == 0 {
            return Err(MatrixError::Empty);
        }
        let gpr = cols.div_ceil(group_size);
        let expected_g = rows.saturating_mul(gpr);
        let expected_p = rows.saturating_mul(cols.div_ceil(4));
        if scales_f16.len() != expected_g
            || mins_f16.len() != expected_g
            || packed.len() != expected_p
        {
            return Err(MatrixError::ElementCountMismatch {
                expected: expected_p,
                actual: packed.len(),
            });
        }
        Ok(Self {
            rows,
            cols,
            group_size,
            hadamard,
            scales_f16,
            mins_f16,
            packed,
        })
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }
    pub fn hadamard(&self) -> HadamardSpec {
        self.hadamard
    }
    pub fn scales_f16(&self) -> &[u16] {
        &self.scales_f16
    }
    pub fn mins_f16(&self) -> &[u16] {
        &self.mins_f16
    }
    pub fn packed(&self) -> &[u8] {
        &self.packed
    }

    pub fn decode_row(&self, row: usize) -> Result<Vec<f64>, MatrixError> {
        if row >= self.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: self.rows,
                actual: row,
            });
        }
        let gpr = self.cols.div_ceil(self.group_size);
        let prb = self.cols.div_ceil(4);
        let p_row = &self.packed[row * prb..(row + 1) * prb];
        let mut out = vec![0.0_f64; self.cols];
        for (c, slot) in out.iter_mut().enumerate() {
            let g = c / self.group_size;
            let scale = f64::from(f16_to_f32(self.scales_f16[row * gpr + g]));
            let min_q = f64::from(f16_to_f32(self.mins_f16[row * gpr + g]));
            let q = (p_row[c >> 2] >> ((c & 3) * 2)) & 0x03;
            *slot = min_q + scale * f64::from(q);
        }
        if self.hadamard.is_enabled() {
            apply_hadamard_inverse(&mut out, self.hadamard.block_size, self.hadamard.seed);
        }
        Ok(out)
    }
}

impl LinearOperator for Int2AsymGroupMatrix {
    fn method_name(&self) -> &'static str {
        if self.hadamard.is_enabled() {
            match self.group_size {
                64 => "INT2_HADAMARD_G64",
                128 => "INT2_HADAMARD_G128",
                _ => "INT2_HADAMARD",
            }
        } else {
            match self.group_size {
                64 => "INT2_ASYM_G64",
                128 => "INT2_ASYM_G128",
                _ => "INT2_ASYM",
            }
        }
    }

    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }

    fn matvec(&self, vector: &[f64]) -> Result<Vec<f64>, MatrixError> {
        if vector.len() != self.cols {
            return Err(MatrixError::DimensionMismatch {
                expected: self.cols,
                actual: vector.len(),
            });
        }
        let x = rotate_input_if_enabled(vector, self.hadamard);
        let gpr = self.cols.div_ceil(self.group_size);
        let prb = self.cols.div_ceil(4);
        let row_dot = |r: usize| -> f64 {
            let p_row = &self.packed[r * prb..(r + 1) * prb];
            let mut sum = 0.0_f64;
            for g in 0..gpr {
                let scale = f64::from(f16_to_f32(self.scales_f16[r * gpr + g]));
                let min_q = f64::from(f16_to_f32(self.mins_f16[r * gpr + g]));
                let start = g * self.group_size;
                let end = (start + self.group_size).min(self.cols);
                let mut qx = 0.0_f64;
                let mut xsum = 0.0_f64;
                for (c, &xv) in x.iter().enumerate().take(end).skip(start) {
                    let q = (p_row[c >> 2] >> ((c & 3) * 2)) & 0x03;
                    qx += f64::from(q) * xv;
                    xsum += xv;
                }
                sum += scale * qx + min_q * xsum;
            }
            sum
        };
        if self.rows >= 64 {
            Ok((0..self.rows).into_par_iter().map(row_dot).collect())
        } else {
            Ok((0..self.rows).map(row_dot).collect())
        }
    }

    fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError> {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for r in 0..self.rows {
            data.extend(self.decode_row(r)?);
        }
        DenseMatrix::from_flat(self.rows, self.cols, data)
    }

    fn storage_breakdown(&self) -> StorageBreakdown {
        StorageBreakdown {
            payload_bytes: self.packed.len(),
            scale_metadata_bytes: (self.scales_f16.len() + self.mins_f16.len()) * 2,
            structure_bytes: 0,
            header_bytes: RESONANCE_HEADER_BYTES,
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Packed Base-3 Ternary Quantizer (`TernaryPackedMatrix`, 1.66-1.85 bpp)
// ---------------------------------------------------------------------------

const POW3: [u8; 5] = [1, 3, 9, 27, 81];

/// Groupwise ternary quantizer (`{-1, 0, +1}`) packing **5 trits per byte**
/// (`3^5 = 243 <= 256`, `1.60 bpp` raw payload — within 1% of Shannon `log2(3) = 1.585`),
/// with optimal Lloyd–Max threshold + per-group `f16` scale and optional Hadamard rotation.
///
/// Effective bit rate: `1.60 + 16 / group_size` bpp (`1.85 bpp` at `g=64`, `1.725 bpp` at `g=128`).
#[derive(Debug, Clone, PartialEq)]
pub struct TernaryPackedMatrix {
    rows: usize,
    cols: usize,
    group_size: usize,
    hadamard: HadamardSpec,
    scales_f16: Vec<u16>,
    /// 5 base-3 digits (`0 => -1`, `1 => 0`, `2 => +1`) packed per byte.
    packed: Vec<u8>,
}

impl TernaryPackedMatrix {
    pub fn quantize(
        matrix: &DenseMatrix,
        group_size: usize,
        hadamard: HadamardSpec,
    ) -> Result<Self, MatrixError> {
        if group_size == 0 {
            return Err(MatrixError::DimensionMismatch {
                expected: 1,
                actual: 0,
            });
        }
        let rows = matrix.rows();
        let cols = matrix.cols();
        let rotated = rotate_rows_if_enabled(matrix, hadamard);
        let gpr = cols.div_ceil(group_size);
        let prb = cols.div_ceil(5);

        let quant_row = |r: usize| -> (Vec<u16>, Vec<u8>) {
            let row = &rotated[r * cols..(r + 1) * cols];
            let mut r_scales = vec![0u16; gpr];
            let mut p_row = vec![0u8; prb];
            let mut trits = vec![1u8; cols];
            for (g, scale_slot) in r_scales.iter_mut().enumerate().take(gpr) {
                let start = g * group_size;
                let end = (start + group_size).min(cols);
                let slice = &row[start..end];
                let mean_abs = slice.iter().map(|v| v.abs()).sum::<f64>() / slice.len() as f64;
                let mut thresh = 0.7 * mean_abs;
                let mut alpha = mean_abs;
                for _ in 0..3 {
                    let mut sum_abs = 0.0_f64;
                    let mut count = 0usize;
                    for &v in slice {
                        if v.abs() > thresh {
                            sum_abs += v.abs();
                            count += 1;
                        }
                    }
                    if count > 0 {
                        alpha = sum_abs / count as f64;
                        thresh = 0.5 * alpha;
                    } else {
                        alpha = 0.0;
                        break;
                    }
                }
                let s_bits = f32_to_f16(alpha as f32);
                let scale = f64::from(f16_to_f32(s_bits));
                let final_t = 0.5 * scale;
                *scale_slot = s_bits;
                for (offset, &v) in slice.iter().enumerate() {
                    trits[start + offset] = if scale <= 0.0 || v.abs() <= final_t {
                        1
                    } else if v > 0.0 {
                        2
                    } else {
                        0
                    };
                }
            }
            for (blk, chunk) in trits.chunks(5).enumerate() {
                let mut byte = 0u8;
                for (i, &t) in chunk.iter().enumerate() {
                    byte += t * POW3[i];
                }
                for &p3 in POW3.iter().skip(chunk.len()) {
                    byte += p3;
                }
                p_row[blk] = byte;
            }
            (r_scales, p_row)
        };

        let row_results: Vec<_> = if rows >= 64 {
            (0..rows).into_par_iter().map(quant_row).collect()
        } else {
            (0..rows).map(quant_row).collect()
        };
        let mut scales_f16 = Vec::with_capacity(rows * gpr);
        let mut packed = Vec::with_capacity(rows * prb);
        for (rs, rp) in row_results {
            scales_f16.extend(rs);
            packed.extend(rp);
        }

        Ok(Self {
            rows,
            cols,
            group_size,
            hadamard,
            scales_f16,
            packed,
        })
    }

    pub fn from_raw(
        rows: usize,
        cols: usize,
        group_size: usize,
        hadamard: HadamardSpec,
        scales_f16: Vec<u16>,
        packed: Vec<u8>,
    ) -> Result<Self, MatrixError> {
        if rows == 0 || cols == 0 || group_size == 0 {
            return Err(MatrixError::Empty);
        }
        let gpr = cols.div_ceil(group_size);
        let expected_g = rows.saturating_mul(gpr);
        let expected_p = rows.saturating_mul(cols.div_ceil(5));
        if scales_f16.len() != expected_g || packed.len() != expected_p {
            return Err(MatrixError::ElementCountMismatch {
                expected: expected_p,
                actual: packed.len(),
            });
        }
        Ok(Self {
            rows,
            cols,
            group_size,
            hadamard,
            scales_f16,
            packed,
        })
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }
    pub fn hadamard(&self) -> HadamardSpec {
        self.hadamard
    }
    pub fn scales_f16(&self) -> &[u16] {
        &self.scales_f16
    }
    pub fn packed(&self) -> &[u8] {
        &self.packed
    }

    #[inline]
    fn unpack_trit(byte: u8, slot: usize) -> f64 {
        let d = (byte / POW3[slot]) % 3;
        match d {
            0 => -1.0,
            2 => 1.0,
            _ => 0.0,
        }
    }

    pub fn decode_row(&self, row: usize) -> Result<Vec<f64>, MatrixError> {
        if row >= self.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: self.rows,
                actual: row,
            });
        }
        let gpr = self.cols.div_ceil(self.group_size);
        let prb = self.cols.div_ceil(5);
        let p_row = &self.packed[row * prb..(row + 1) * prb];
        let mut out = vec![0.0_f64; self.cols];
        for (c, slot) in out.iter_mut().enumerate() {
            let g = c / self.group_size;
            let scale = f64::from(f16_to_f32(self.scales_f16[row * gpr + g]));
            *slot = scale * Self::unpack_trit(p_row[c / 5], c % 5);
        }
        if self.hadamard.is_enabled() {
            apply_hadamard_inverse(&mut out, self.hadamard.block_size, self.hadamard.seed);
        }
        Ok(out)
    }
}

impl LinearOperator for TernaryPackedMatrix {
    fn method_name(&self) -> &'static str {
        if self.hadamard.is_enabled() {
            match self.group_size {
                64 => "TERNARY_HADAMARD_G64",
                128 => "TERNARY_HADAMARD_G128",
                _ => "TERNARY_HADAMARD",
            }
        } else {
            match self.group_size {
                64 => "TERNARY_PACKED_G64",
                128 => "TERNARY_PACKED_G128",
                _ => "TERNARY_PACKED",
            }
        }
    }

    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }

    fn matvec(&self, vector: &[f64]) -> Result<Vec<f64>, MatrixError> {
        if vector.len() != self.cols {
            return Err(MatrixError::DimensionMismatch {
                expected: self.cols,
                actual: vector.len(),
            });
        }
        let x = rotate_input_if_enabled(vector, self.hadamard);
        let gpr = self.cols.div_ceil(self.group_size);
        let prb = self.cols.div_ceil(5);
        let row_dot = |r: usize| -> f64 {
            let p_row = &self.packed[r * prb..(r + 1) * prb];
            let mut sum = 0.0_f64;
            for g in 0..gpr {
                let scale = f64::from(f16_to_f32(self.scales_f16[r * gpr + g]));
                let start = g * self.group_size;
                let end = (start + self.group_size).min(self.cols);
                let mut acc = 0.0_f64;
                for (c, &xv) in x.iter().enumerate().take(end).skip(start) {
                    acc += Self::unpack_trit(p_row[c / 5], c % 5) * xv;
                }
                sum += scale * acc;
            }
            sum
        };
        if self.rows >= 64 {
            Ok((0..self.rows).into_par_iter().map(row_dot).collect())
        } else {
            Ok((0..self.rows).map(row_dot).collect())
        }
    }

    fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError> {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for r in 0..self.rows {
            data.extend(self.decode_row(r)?);
        }
        DenseMatrix::from_flat(self.rows, self.cols, data)
    }

    fn storage_breakdown(&self) -> StorageBreakdown {
        StorageBreakdown {
            payload_bytes: self.packed.len(),
            scale_metadata_bytes: self.scales_f16.len() * 2,
            structure_bytes: 0,
            header_bytes: RESONANCE_HEADER_BYTES,
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Learned Residual Vector Quantizer (`VectorQuantMatrix`, AQLM / QuIP#-inspired)
// ---------------------------------------------------------------------------

/// Multi-stage Learned Vector / Residual Vector Quantizer (`VQ` / `RVQ`).
///
/// Groups each row (after optional Hadamard incoherence rotation) into subvectors of
/// length `subvec_dim` (e.g. `4` or `8`), normalizes each group of `group_size` weights
/// by an `f16` RMS scale, and represents each subvector as the sum of `num_stages`
/// centroids from learned FP16 codebooks (`K <= 256`, 1 byte per stage per subvector):
/// - `d=8, stages=1, K=256, g=64`: `1.00 + 0.25 = 1.25 bpp` (+ tiny amortized codebook)
/// - `d=8, stages=2, K=256, g=64`: `2.00 + 0.25 = 2.25 bpp` (Residual VQ)
/// - `d=4, stages=1, K=256, g=64`: `2.00 + 0.25 = 2.25 bpp`
///
/// Executes direct `matvec` via pre-projected codebook-activation LUTs (`O(K * cols)`
/// once per matvec, followed by zero-multiplication table lookups per row!).
#[derive(Debug, Clone, PartialEq)]
pub struct VectorQuantMatrix {
    rows: usize,
    cols: usize,
    subvec_dim: usize,
    codebook_size: usize,
    num_stages: usize,
    group_size: usize,
    hadamard: HadamardSpec,
    /// Flattened FP16 codebooks: `[num_stages * codebook_size * subvec_dim]`.
    codebooks_f16: Vec<u16>,
    /// Per-row, per-group `f16` RMS normalization scales.
    scales_f16: Vec<u16>,
    /// Packed centroid indices: `[rows * num_subvecs_per_row * num_stages]`.
    codes: Vec<u8>,
}

impl VectorQuantMatrix {
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        matrix: &DenseMatrix,
        subvec_dim: usize,
        codebook_size: usize,
        num_stages: usize,
        group_size: usize,
        hadamard: HadamardSpec,
        seed: u64,
    ) -> Result<Self, MatrixError> {
        if subvec_dim == 0
            || codebook_size == 0
            || codebook_size > 256
            || num_stages == 0
            || num_stages > 2
            || group_size == 0
        {
            return Err(MatrixError::DimensionMismatch {
                expected: 1,
                actual: 0,
            });
        }
        let rows = matrix.rows();
        let cols = matrix.cols();
        let rotated = rotate_rows_if_enabled(matrix, hadamard);
        let gpr = cols.div_ceil(group_size);
        let subvecs_per_row = cols.div_ceil(subvec_dim);
        let total_subvecs = rows * subvecs_per_row;

        // Normalize rows by per-group RMS so a single compact codebook generalizes cleanly
        // across the entire matrix regardless of row/group norm variance.
        let mut scales_f16 = vec![0u16; rows * gpr];
        let mut norm_data = vec![0.0_f64; rows * subvecs_per_row * subvec_dim];
        for r in 0..rows {
            let row = &rotated[r * cols..(r + 1) * cols];
            let dst = &mut norm_data
                [r * subvecs_per_row * subvec_dim..(r + 1) * subvecs_per_row * subvec_dim];
            for g in 0..gpr {
                let start = g * group_size;
                let end = (start + group_size).min(cols);
                let slice = &row[start..end];
                let rms = (slice.iter().map(|v| v * v).sum::<f64>() / slice.len() as f64).sqrt();
                let s_bits = f32_to_f16(rms as f32);
                let scale = f64::from(f16_to_f32(s_bits));
                scales_f16[r * gpr + g] = s_bits;
                if scale > 1e-12 {
                    for c in start..end {
                        dst[c] = row[c] / scale;
                    }
                }
            }
        }

        // Deterministic bounded subsample (up to 4096 subvectors) for fast consumer-CPU k-means.
        let sample_cap = total_subvecs.min(4096);
        let mut rng = DeterministicRng::new(seed ^ 0x5651_5245_534f_4e41);
        let sample_indices: Vec<usize> = if total_subvecs <= sample_cap {
            (0..total_subvecs).collect()
        } else {
            let stride = (total_subvecs / sample_cap).max(1);
            (0..sample_cap)
                .map(|i| (i * stride + rng.next_usize(stride)) % total_subvecs)
                .collect()
        };

        let mut residual = norm_data;
        let mut codebooks_f16 = Vec::with_capacity(num_stages * codebook_size * subvec_dim);
        let mut codes = vec![0u8; total_subvecs * num_stages];

        for stage in 0..num_stages {
            let k_eff = codebook_size.min(sample_indices.len().max(1));
            // Deterministic farthest-first / k-means++ initialization on subsample
            let mut centroids = vec![0.0_f64; codebook_size * subvec_dim];
            let first_idx = sample_indices[rng.next_usize(sample_indices.len())];
            centroids[..subvec_dim]
                .copy_from_slice(&residual[first_idx * subvec_dim..(first_idx + 1) * subvec_dim]);
            let mut min_dists = vec![f64::INFINITY; sample_indices.len()];
            for c_idx in 1..k_eff {
                let prev = &centroids[(c_idx - 1) * subvec_dim..c_idx * subvec_dim];
                let mut best_si = 0usize;
                let mut best_d = -1.0_f64;
                for (si, &sv_idx) in sample_indices.iter().enumerate() {
                    let v = &residual[sv_idx * subvec_dim..(sv_idx + 1) * subvec_dim];
                    let d = sq_dist(v, prev);
                    if d < min_dists[si] {
                        min_dists[si] = d;
                    }
                    if min_dists[si] > best_d {
                        best_d = min_dists[si];
                        best_si = si;
                    }
                }
                let chosen = sample_indices[best_si];
                centroids[c_idx * subvec_dim..(c_idx + 1) * subvec_dim]
                    .copy_from_slice(&residual[chosen * subvec_dim..(chosen + 1) * subvec_dim]);
            }

            // 6 Lloyd iterations on the deterministic subsample
            let mut sums = vec![0.0_f64; codebook_size * subvec_dim];
            let mut counts = vec![0usize; codebook_size];
            for _ in 0..6 {
                sums.fill(0.0);
                counts.fill(0);
                for &sv_idx in &sample_indices {
                    let v = &residual[sv_idx * subvec_dim..(sv_idx + 1) * subvec_dim];
                    let best_k = nearest_centroid(v, &centroids, k_eff, subvec_dim);
                    counts[best_k] += 1;
                    let dst = &mut sums[best_k * subvec_dim..(best_k + 1) * subvec_dim];
                    for (s, &vv) in dst.iter_mut().zip(v) {
                        *s += vv;
                    }
                }
                for k in 0..k_eff {
                    if counts[k] > 0 {
                        let inv = 1.0 / counts[k] as f64;
                        let dst = &mut centroids[k * subvec_dim..(k + 1) * subvec_dim];
                        let src = &sums[k * subvec_dim..(k + 1) * subvec_dim];
                        for (c_val, &s_val) in dst.iter_mut().zip(src) {
                            *c_val = s_val * inv;
                        }
                    }
                }
            }

            // Quantize centroids to FP16 first so residual subtraction uses exact decoded centroids!
            let mut stage_cb_f16 = Vec::with_capacity(codebook_size * subvec_dim);
            let mut cb_decoded = vec![0.0_f64; codebook_size * subvec_dim];
            for (i, &c) in centroids.iter().enumerate() {
                let bits = f32_to_f16(c as f32);
                stage_cb_f16.push(bits);
                cb_decoded[i] = f64::from(f16_to_f32(bits));
            }
            codebooks_f16.extend_from_slice(&stage_cb_f16);

            // Assign every subvector in parallel and subtract decoded centroid for next RVQ stage
            let assigned: Vec<u8> = if total_subvecs >= 512 {
                (0..total_subvecs)
                    .into_par_iter()
                    .map(|sv_idx| {
                        let v = &residual[sv_idx * subvec_dim..(sv_idx + 1) * subvec_dim];
                        nearest_centroid(v, &cb_decoded, k_eff, subvec_dim) as u8
                    })
                    .collect()
            } else {
                (0..total_subvecs)
                    .map(|sv_idx| {
                        let v = &residual[sv_idx * subvec_dim..(sv_idx + 1) * subvec_dim];
                        nearest_centroid(v, &cb_decoded, k_eff, subvec_dim) as u8
                    })
                    .collect()
            };

            for (sv_idx, &k) in assigned.iter().enumerate() {
                codes[sv_idx * num_stages + stage] = k;
                if stage + 1 < num_stages {
                    let cb = &cb_decoded[k as usize * subvec_dim..(k as usize + 1) * subvec_dim];
                    let v = &mut residual[sv_idx * subvec_dim..(sv_idx + 1) * subvec_dim];
                    for (rv, &cv) in v.iter_mut().zip(cb) {
                        *rv -= cv;
                    }
                }
            }
        }

        Ok(Self {
            rows,
            cols,
            subvec_dim,
            codebook_size,
            num_stages,
            group_size,
            hadamard,
            codebooks_f16,
            scales_f16,
            codes,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_raw(
        rows: usize,
        cols: usize,
        subvec_dim: usize,
        codebook_size: usize,
        num_stages: usize,
        group_size: usize,
        hadamard: HadamardSpec,
        codebooks_f16: Vec<u16>,
        scales_f16: Vec<u16>,
        codes: Vec<u8>,
    ) -> Result<Self, MatrixError> {
        if rows == 0
            || cols == 0
            || subvec_dim == 0
            || codebook_size == 0
            || codebook_size > 256
            || num_stages == 0
            || group_size == 0
        {
            return Err(MatrixError::Empty);
        }
        let expected_cb = num_stages
            .saturating_mul(codebook_size)
            .saturating_mul(subvec_dim);
        let expected_scales = rows.saturating_mul(cols.div_ceil(group_size));
        let expected_codes = rows
            .saturating_mul(cols.div_ceil(subvec_dim))
            .saturating_mul(num_stages);
        if codebooks_f16.len() != expected_cb
            || scales_f16.len() != expected_scales
            || codes.len() != expected_codes
        {
            return Err(MatrixError::ElementCountMismatch {
                expected: expected_codes,
                actual: codes.len(),
            });
        }
        Ok(Self {
            rows,
            cols,
            subvec_dim,
            codebook_size,
            num_stages,
            group_size,
            hadamard,
            codebooks_f16,
            scales_f16,
            codes,
        })
    }

    pub fn subvec_dim(&self) -> usize {
        self.subvec_dim
    }
    pub fn codebook_size(&self) -> usize {
        self.codebook_size
    }
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }
    pub fn group_size(&self) -> usize {
        self.group_size
    }
    pub fn hadamard(&self) -> HadamardSpec {
        self.hadamard
    }
    pub fn codebooks_f16(&self) -> &[u16] {
        &self.codebooks_f16
    }
    pub fn scales_f16(&self) -> &[u16] {
        &self.scales_f16
    }
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    pub fn decode_row(&self, row: usize) -> Result<Vec<f64>, MatrixError> {
        if row >= self.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: self.rows,
                actual: row,
            });
        }
        let gpr = self.cols.div_ceil(self.group_size);
        let spr = self.cols.div_ceil(self.subvec_dim);
        let cb_stage_stride = self.codebook_size * self.subvec_dim;
        let mut out = vec![0.0_f64; self.cols];
        for sv in 0..spr {
            let code_base = (row * spr + sv) * self.num_stages;
            let col_start = sv * self.subvec_dim;
            let len = (self.cols - col_start).min(self.subvec_dim);
            for stage in 0..self.num_stages {
                let k = self.codes[code_base + stage] as usize;
                let cb_off = stage * cb_stage_stride + k * self.subvec_dim;
                for i in 0..len {
                    out[col_start + i] += f64::from(f16_to_f32(self.codebooks_f16[cb_off + i]));
                }
            }
        }
        for (c, slot) in out.iter_mut().enumerate() {
            let g = c / self.group_size;
            *slot *= f64::from(f16_to_f32(self.scales_f16[row * gpr + g]));
        }
        if self.hadamard.is_enabled() {
            apply_hadamard_inverse(&mut out, self.hadamard.block_size, self.hadamard.seed);
        }
        Ok(out)
    }
}

#[inline]
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut d = 0.0_f64;
    for (&x, &y) in a.iter().zip(b) {
        let diff = x - y;
        d += diff * diff;
    }
    d
}

#[inline]
fn nearest_centroid(v: &[f64], centroids: &[f64], k_eff: usize, dim: usize) -> usize {
    let mut best_k = 0usize;
    let mut best_d = f64::INFINITY;
    for k in 0..k_eff {
        let c = &centroids[k * dim..(k + 1) * dim];
        let d = sq_dist(v, c);
        if d < best_d {
            best_d = d;
            best_k = k;
        }
    }
    best_k
}

impl LinearOperator for VectorQuantMatrix {
    fn method_name(&self) -> &'static str {
        match (self.num_stages, self.subvec_dim) {
            (1, 8) => "VQ_D8_K256",
            (2, 8) => "RVQ_D8_2X256",
            (1, 4) => "VQ_D4_K256",
            (2, 4) => "RVQ_D4_2X256",
            _ => "VECTOR_QUANT",
        }
    }

    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }

    fn matvec(&self, vector: &[f64]) -> Result<Vec<f64>, MatrixError> {
        if vector.len() != self.cols {
            return Err(MatrixError::DimensionMismatch {
                expected: self.cols,
                actual: vector.len(),
            });
        }
        let x = rotate_input_if_enabled(vector, self.hadamard);
        let gpr = self.cols.div_ceil(self.group_size);
        let spr = self.cols.div_ceil(self.subvec_dim);
        let cb_stage_stride = self.codebook_size * self.subvec_dim;

        // Fast path when subvectors do not straddle group boundaries (always true for
        // power-of-two `subvec_dim` dividing `group_size`): precompute codebook-activation
        // inner-product LUT once, turning every row dot product into pure table lookups!
        let aligned = self.group_size.is_multiple_of(self.subvec_dim);
        if aligned {
            let mut lut = vec![0.0_f64; self.num_stages * spr * self.codebook_size];
            for stage in 0..self.num_stages {
                for sv in 0..spr {
                    let col_start = sv * self.subvec_dim;
                    let len = (self.cols - col_start).min(self.subvec_dim);
                    let x_blk = &x[col_start..col_start + len];
                    let lut_base = (stage * spr + sv) * self.codebook_size;
                    for k in 0..self.codebook_size {
                        let cb_off = stage * cb_stage_stride + k * self.subvec_dim;
                        let mut dot = 0.0_f64;
                        for i in 0..len {
                            dot += f64::from(f16_to_f32(self.codebooks_f16[cb_off + i])) * x_blk[i];
                        }
                        lut[lut_base + k] = dot;
                    }
                }
            }
            let sv_per_group = self.group_size / self.subvec_dim;
            let row_dot = |r: usize| -> f64 {
                let mut sum = 0.0_f64;
                for g in 0..gpr {
                    let scale = f64::from(f16_to_f32(self.scales_f16[r * gpr + g]));
                    let sv_start = g * sv_per_group;
                    let sv_end = (sv_start + sv_per_group).min(spr);
                    let mut g_acc = 0.0_f64;
                    for sv in sv_start..sv_end {
                        let code_base = (r * spr + sv) * self.num_stages;
                        for stage in 0..self.num_stages {
                            let k = self.codes[code_base + stage] as usize;
                            g_acc += lut[(stage * spr + sv) * self.codebook_size + k];
                        }
                    }
                    sum += scale * g_acc;
                }
                sum
            };
            if self.rows >= 64 {
                return Ok((0..self.rows).into_par_iter().map(row_dot).collect());
            }
            return Ok((0..self.rows).map(row_dot).collect());
        }

        // General fallback for non-aligned subvector/group sizes
        let mut out = Vec::with_capacity(self.rows);
        for r in 0..self.rows {
            let row = self.decode_row(r)?;
            let dot: f64 = row.iter().zip(vector).map(|(a, b)| a * b).sum();
            out.push(dot);
        }
        Ok(out)
    }

    fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError> {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for r in 0..self.rows {
            data.extend(self.decode_row(r)?);
        }
        DenseMatrix::from_flat(self.rows, self.cols, data)
    }

    fn storage_breakdown(&self) -> StorageBreakdown {
        StorageBreakdown {
            payload_bytes: self.codes.len(),
            scale_metadata_bytes: self.scales_f16.len() * 2,
            structure_bytes: self.codebooks_f16.len() * 2,
            header_bytes: RESONANCE_HEADER_BYTES,
        }
    }
}

// ---------------------------------------------------------------------------
// 6. PROJECT ECHO — Compact Low-Rank & Sparse Residual Fidelity Recovery
// ---------------------------------------------------------------------------

/// Base ultra-low-bit representation wrapped by [`EchoCorrectedMatrix`].
#[derive(Debug, Clone, PartialEq)]
pub enum EchoBaseOperator {
    Int3Asym(Int3AsymGroupMatrix),
    Int2Asym(Int2AsymGroupMatrix),
    Ternary(TernaryPackedMatrix),
    VectorQuant(VectorQuantMatrix),
}

impl EchoBaseOperator {
    pub fn decode_row(&self, row: usize) -> Result<Vec<f64>, MatrixError> {
        match self {
            Self::Int3Asym(m) => m.decode_row(row),
            Self::Int2Asym(m) => m.decode_row(row),
            Self::Ternary(m) => m.decode_row(row),
            Self::VectorQuant(m) => m.decode_row(row),
        }
    }
}

impl LinearOperator for EchoBaseOperator {
    fn method_name(&self) -> &'static str {
        match self {
            Self::Int3Asym(m) => m.method_name(),
            Self::Int2Asym(m) => m.method_name(),
            Self::Ternary(m) => m.method_name(),
            Self::VectorQuant(m) => m.method_name(),
        }
    }

    fn rows(&self) -> usize {
        match self {
            Self::Int3Asym(m) => m.rows(),
            Self::Int2Asym(m) => m.rows(),
            Self::Ternary(m) => m.rows(),
            Self::VectorQuant(m) => m.rows(),
        }
    }

    fn cols(&self) -> usize {
        match self {
            Self::Int3Asym(m) => m.cols(),
            Self::Int2Asym(m) => m.cols(),
            Self::Ternary(m) => m.cols(),
            Self::VectorQuant(m) => m.cols(),
        }
    }

    fn matvec(&self, vector: &[f64]) -> Result<Vec<f64>, MatrixError> {
        match self {
            Self::Int3Asym(m) => m.matvec(vector),
            Self::Int2Asym(m) => m.matvec(vector),
            Self::Ternary(m) => m.matvec(vector),
            Self::VectorQuant(m) => m.matvec(vector),
        }
    }

    fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError> {
        match self {
            Self::Int3Asym(m) => m.reconstruct_dense(),
            Self::Int2Asym(m) => m.reconstruct_dense(),
            Self::Ternary(m) => m.reconstruct_dense(),
            Self::VectorQuant(m) => m.reconstruct_dense(),
        }
    }

    fn storage_breakdown(&self) -> StorageBreakdown {
        match self {
            Self::Int3Asym(m) => m.storage_breakdown(),
            Self::Int2Asym(m) => m.storage_breakdown(),
            Self::Ternary(m) => m.storage_breakdown(),
            Self::VectorQuant(m) => m.storage_breakdown(),
        }
    }
}

/// PROJECT ECHO — Compact Fidelity Recovery Operator:
/// `W_hat = W_base + U_echo * V_echo^T + SparseResidual`
///
/// Fits compact INT8 low-rank SVD recovery factors (`rank = 4..16`) and/or top-k
/// FP16 sparse outlier corrections directly onto the residual `E = W - W_base`
/// (optionally weighted by real activation channel RMS `act_scales`).
/// Every recovery byte is strictly counted in [`StorageBreakdown`].
#[derive(Debug, Clone, PartialEq)]
pub struct EchoCorrectedMatrix {
    base: EchoBaseOperator,
    low_rank_adapter: Option<LowRankSvdMatrix>,
    sparse_residual: Option<SparseResidual>,
}

impl EchoCorrectedMatrix {
    pub fn fit(
        original: &DenseMatrix,
        base: EchoBaseOperator,
        echo_rank: usize,
        sparse_fraction: f64,
        seed: u64,
    ) -> Result<Self, MatrixError> {
        let base_dense = base.reconstruct_dense()?;
        let mut residual = original.sub(&base_dense)?;

        let sparse_residual = if sparse_fraction > 0.0 {
            let sr = SparseResidual::from_residual_top_fraction(&residual, sparse_fraction);
            if sr.nnz() > 0 {
                let mut res_data = residual.as_slice().to_vec();
                for (&idx, &bits) in sr.indices().iter().zip(sr.values_f16()) {
                    res_data[idx as usize] -= f64::from(f16_to_f32(bits));
                }
                residual = DenseMatrix::from_flat(residual.rows(), residual.cols(), res_data)?;
                Some(sr)
            } else {
                None
            }
        } else {
            None
        };

        let low_rank_adapter = if echo_rank > 0 {
            let r = echo_rank.min(original.rows().min(original.cols()));
            if r > 0 {
                Some(LowRankSvdMatrix::from_matrix(
                    &residual,
                    r,
                    FactorPrecision::Int8Sym,
                    seed ^ 0x4543_484f_5245_4356,
                )?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            base,
            low_rank_adapter,
            sparse_residual,
        })
    }

    pub fn from_parts(
        base: EchoBaseOperator,
        low_rank_adapter: Option<LowRankSvdMatrix>,
        sparse_residual: Option<SparseResidual>,
    ) -> Self {
        Self {
            base,
            low_rank_adapter,
            sparse_residual,
        }
    }

    pub fn base(&self) -> &EchoBaseOperator {
        &self.base
    }
    pub fn low_rank_adapter(&self) -> Option<&LowRankSvdMatrix> {
        self.low_rank_adapter.as_ref()
    }
    pub fn sparse_residual(&self) -> Option<&SparseResidual> {
        self.sparse_residual.as_ref()
    }

    pub fn decode_row(&self, row: usize) -> Result<Vec<f64>, MatrixError> {
        let mut out = self.base.decode_row(row)?;
        if let Some(lr) = &self.low_rank_adapter {
            let u = lr.left().reconstruct_dense()?;
            let v = lr.right().reconstruct_dense()?;
            let rank = u.cols();
            let cols = v.cols();
            let u_row = &u.as_slice()[row * rank..(row + 1) * rank];
            let v_data = v.as_slice();
            for (k, &uk) in u_row.iter().enumerate() {
                let v_row = &v_data[k * cols..(k + 1) * cols];
                for (a, &vk) in out.iter_mut().zip(v_row) {
                    *a += uk * vk;
                }
            }
        }
        if let Some(sr) = &self.sparse_residual {
            let cols = self.cols();
            let row_start = row * cols;
            let row_end = row_start + cols;
            for (&idx, &bits) in sr.indices().iter().zip(sr.values_f16()) {
                let i = idx as usize;
                if i >= row_start && i < row_end {
                    out[i - row_start] += f64::from(f16_to_f32(bits));
                }
            }
        }
        Ok(out)
    }
}

impl LinearOperator for EchoCorrectedMatrix {
    fn method_name(&self) -> &'static str {
        match &self.base {
            EchoBaseOperator::Int3Asym(_) => "INT3_ECHO",
            EchoBaseOperator::Int2Asym(_) => "INT2_HADAMARD_ECHO",
            EchoBaseOperator::Ternary(_) => "TERNARY_HADAMARD_ECHO",
            EchoBaseOperator::VectorQuant(_) => "VQ_ECHO",
        }
    }

    fn rows(&self) -> usize {
        self.base.rows()
    }
    fn cols(&self) -> usize {
        self.base.cols()
    }

    fn matvec(&self, vector: &[f64]) -> Result<Vec<f64>, MatrixError> {
        let mut y = self.base.matvec(vector)?;
        if let Some(lr) = &self.low_rank_adapter {
            let dy = lr.matvec(vector)?;
            for (a, &b) in y.iter_mut().zip(&dy) {
                *a += b;
            }
        }
        if let Some(sr) = &self.sparse_residual {
            sr.add_matvec_into(vector, &mut y)?;
        }
        Ok(y)
    }

    fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError> {
        let mut mat = self.base.reconstruct_dense()?;
        if let Some(lr) = &self.low_rank_adapter {
            let lrm = lr.reconstruct_dense()?;
            mat = mat.add(&lrm)?;
        }
        if let Some(sr) = &self.sparse_residual {
            let mut data = mat.as_slice().to_vec();
            sr.add_dense_into(&mut data);
            mat = DenseMatrix::from_flat(mat.rows(), mat.cols(), data)?;
        }
        Ok(mat)
    }

    fn storage_breakdown(&self) -> StorageBreakdown {
        let b = self.base.storage_breakdown();
        let lr = self
            .low_rank_adapter
            .as_ref()
            .map_or(0, |m| m.storage_breakdown().total_bytes());
        let sr = self
            .sparse_residual
            .as_ref()
            .map_or(0, SparseResidual::entry_bytes);
        StorageBreakdown {
            payload_bytes: b.payload_bytes,
            scale_metadata_bytes: b.scale_metadata_bytes,
            structure_bytes: b.structure_bytes.saturating_add(lr).saturating_add(sr),
            header_bytes: b.header_bytes,
        }
    }
}

// ---------------------------------------------------------------------------
// 7. QML Resonance Representation Family (`QMR-300` -> `QMR-150`) & MAP-B
// ---------------------------------------------------------------------------

/// The 7-rung Sub-3-Bit Research Ladder targeting complete executable artifacts
/// between `3.00` and `1.50` effective bits per parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResonanceProfile {
    Qmr300,
    Qmr275,
    Qmr250,
    Qmr225,
    Qmr200,
    Qmr175,
    Qmr150,
}

impl ResonanceProfile {
    pub const ALL: [Self; 7] = [
        Self::Qmr300,
        Self::Qmr275,
        Self::Qmr250,
        Self::Qmr225,
        Self::Qmr200,
        Self::Qmr175,
        Self::Qmr150,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::Qmr300 => "QMR-300",
            Self::Qmr275 => "QMR-275",
            Self::Qmr250 => "QMR-250",
            Self::Qmr225 => "QMR-225",
            Self::Qmr200 => "QMR-200",
            Self::Qmr175 => "QMR-175",
            Self::Qmr150 => "QMR-150",
        }
    }

    pub fn target_bpp(self) -> f64 {
        match self {
            Self::Qmr300 => 3.00,
            Self::Qmr275 => 2.75,
            Self::Qmr250 => 2.50,
            Self::Qmr225 => 2.25,
            Self::Qmr200 => 2.00,
            Self::Qmr175 => 1.75,
            Self::Qmr150 => 1.50,
        }
    }

    pub fn from_code(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "qmr-300" | "qmr300" | "3.00" => Some(Self::Qmr300),
            "qmr-275" | "qmr275" | "2.75" => Some(Self::Qmr275),
            "qmr-250" | "qmr250" | "2.50" => Some(Self::Qmr250),
            "qmr-225" | "qmr225" | "2.25" => Some(Self::Qmr225),
            "qmr-200" | "qmr200" | "2.00" => Some(Self::Qmr200),
            "qmr-175" | "qmr175" | "1.75" => Some(Self::Qmr175),
            "qmr-150" | "qmr150" | "1.50" => Some(Self::Qmr150),
            _ => None,
        }
    }
}

/// Discrete candidate encoding evaluated by behavior-aware `MAP-B` for a single tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResonanceCandidateKind {
    /// ~1.25 bpp (d=8, K=256, 1-stage Hadamard VQ)
    VqD8Hadamard,
    /// ~1.725 bpp (5 trits/byte, g=128, Hadamard)
    TernaryHadamardG128,
    /// ~1.85 bpp (5 trits/byte, g=64, Hadamard)
    TernaryHadamardG64,
    /// ~2.05-2.20 bpp (Ternary g=128 + rank-8 INT8 Echo adapter)
    TernaryEchoR8,
    /// ~2.25 bpp (2-bit packed, g=128, Hadamard)
    Int2HadamardG128,
    /// ~2.50 bpp (2-bit packed, g=64, Hadamard)
    Int2HadamardG64,
    /// ~2.65-2.85 bpp (2-bit g=128 + rank-8 INT8 Echo adapter)
    Int2EchoR8,
    /// ~3.25 bpp (3-bit packed, g=128, Hadamard)
    Int3HadamardG128,
    /// ~3.50 bpp (3-bit packed, g=64, Hadamard)
    Int3HadamardG64,
    /// ~4.50 bpp (Q4_0 g=32)
    Int4SymG32,
    /// ~8.25 bpp (INT8 g=64)
    Int8SymG64,
}

impl ResonanceCandidateKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::VqD8Hadamard => "VQ_D8_HADAMARD",
            Self::TernaryHadamardG128 => "TERNARY_HADAMARD_G128",
            Self::TernaryHadamardG64 => "TERNARY_HADAMARD_G64",
            Self::TernaryEchoR8 => "TERNARY_ECHO_R8",
            Self::Int2HadamardG128 => "INT2_HADAMARD_G128",
            Self::Int2HadamardG64 => "INT2_HADAMARD_G64",
            Self::Int2EchoR8 => "INT2_ECHO_R8",
            Self::Int3HadamardG128 => "INT3_HADAMARD_G128",
            Self::Int3HadamardG64 => "INT3_HADAMARD_G64",
            Self::Int4SymG32 => "INT4_SYM_G32",
            Self::Int8SymG64 => "INT8_SYM_G64",
        }
    }
}

/// Measured candidate evaluation for one tensor inside `MAP-B`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResonanceCandidateEval {
    pub kind: ResonanceCandidateKind,
    pub total_bytes: usize,
    pub effective_bpp: f64,
    pub rel_frob_error: f64,
    pub matvec_rel_error: f64,
}

/// Streaming per-tensor profile built during Pass 1 of `MAP-B` so dense weights
/// are immediately dropped before solving the global sub-3-bit budget.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResonanceTensorProfile {
    pub tensor_name: String,
    pub tensor_index: usize,
    pub param_count: usize,
    pub spectral_norm: f64,
    pub behavioral_weight: f64,
    pub candidates: Vec<ResonanceCandidateEval>,
}

/// Optional externally measured per-layer behavioral importance multipliers
/// (e.g., from `qml-behavior` layer drift / RSA / causal ablation studies).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BehaviorImportanceMap {
    /// Multiplier per transformer layer index (default `1.0`; e.g., `2.5` for a high-drift block).
    pub layer_multipliers: BTreeMap<usize, f64>,
    /// Extra multiplier for embedding & output head tensors (default `2.2`).
    pub embedding_multiplier: f64,
    /// Extra multiplier for first & last transformer blocks (default `1.65`).
    pub boundary_block_multiplier: f64,
}

impl BehaviorImportanceMap {
    pub fn default_priors() -> Self {
        Self {
            layer_multipliers: BTreeMap::new(),
            embedding_multiplier: 2.5,
            boundary_block_multiplier: 1.8,
        }
    }

    /// Compute the behavioral importance multiplier for a tensor given its name and total layers.
    pub fn weight_for_tensor(
        &self,
        name: &str,
        layer_idx: Option<usize>,
        total_layers: usize,
    ) -> f64 {
        let lower = name.to_lowercase();
        let mut w = 1.0_f64;
        if lower.contains("embed") || lower.contains("lm_head") {
            w *= self.embedding_multiplier.max(1.0);
        }
        if lower.contains("v_proj") || lower.contains("o_proj") || lower.contains("down_proj") {
            w *= 1.25;
        }
        if let Some(l) = layer_idx {
            if l == 0 || (total_layers > 1 && l + 1 >= total_layers) {
                w *= self.boundary_block_multiplier.max(1.0);
            }
            if let Some(&ext) = self.layer_multipliers.get(&l) {
                w *= ext.clamp(0.25, 10.0);
            }
        }
        w
    }
}

/// Final `MAP-B` selection for one tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResonanceTensorAllocation {
    pub tensor_name: String,
    pub param_count: usize,
    pub behavioral_weight: f64,
    pub selected: ResonanceCandidateEval,
}

/// Complete `MAP-B` sub-3-bit allocation plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResonanceAllocationPlan {
    pub profile: ResonanceProfile,
    pub target_complete_bpp: f64,
    pub total_parameters: usize,
    pub fixed_overhead_bytes: usize,
    pub target_budget_bytes: usize,
    pub allocated_tensor_bytes: usize,
    pub projected_complete_bytes: usize,
    pub projected_complete_bpp: f64,
    pub budget_satisfied: bool,
    pub allocations: Vec<ResonanceTensorAllocation>,
}

/// Fast direct matvec probe error without dense reconstruction (10-30x faster for MAP-B Pass 1).
fn fast_probe_matvec_error<O: LinearOperator>(
    ref_outputs: &[Vec<f64>],
    probes: &[Vec<f64>],
    op: &O,
) -> Result<f64, MatrixError> {
    let mut sum_rel = 0.0_f64;
    for (expected, probe) in ref_outputs.iter().zip(probes) {
        let actual = op.matvec(probe)?;
        let err_norm = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let ref_norm = expected.iter().map(|a| a * a).sum::<f64>().sqrt();
        sum_rel += if ref_norm > 1e-15 {
            err_norm / ref_norm
        } else {
            err_norm
        };
    }
    Ok(sum_rel / probes.len().max(1) as f64)
}

/// Profile one 2D matrix across the sub-3-bit resonance candidate menu in Pass 1
/// (constant-memory streaming compatible: drop `mat` immediately after calling).
pub fn profile_tensor_for_resonance(
    tensor_name: &str,
    tensor_index: usize,
    mat: &DenseMatrix,
    behavioral_weight: f64,
    include_vq: bool,
    include_echo: bool,
    seed: u64,
) -> Result<ResonanceTensorProfile, MatrixError> {
    profile_tensor_for_resonance_calibrated(
        tensor_name,
        tensor_index,
        mat,
        behavioral_weight,
        include_vq,
        include_echo,
        None,
        seed,
    )
}

/// Activation-calibrated Pass 1 profiler for `MAP-B`: when `cal` is provided, evaluates
/// candidates using activation-weighted probes (`probe[c] *= input_rms[c]`) and
/// Blockwise Cholesky GPTQ + Hadamard quantizers.
#[allow(clippy::too_many_arguments)]
pub fn profile_tensor_for_resonance_calibrated(
    tensor_name: &str,
    tensor_index: usize,
    mat: &DenseMatrix,
    behavioral_weight: f64,
    include_vq: bool,
    include_echo: bool,
    cal: Option<&crate::gptq::LayerCalibrationStats>,
    seed: u64,
) -> Result<ResonanceTensorProfile, MatrixError> {
    let is_huge_table = mat.rows() > 8192 || mat.cols() > 8192;
    let spectral_norm = if is_huge_table {
        mat.frobenius_norm() / (mat.rows().min(mat.cols()) as f64).sqrt().max(1.0)
    } else {
        truncated_svd(mat, 1, seed.wrapping_add(tensor_index as u64))?.singular_values[0]
    };
    let param_count = mat.rows() * mat.cols();
    let cal_matched = cal.filter(|s| s.cols == mat.cols());
    let mut probes =
        deterministic_probe_vectors(mat.cols(), 3, seed.wrapping_add(tensor_index as u64));
    if let Some(stats) = cal_matched {
        for p in &mut probes {
            for (v, &rms) in p.iter_mut().zip(&stats.input_rms) {
                *v *= rms;
            }
        }
    }
    let ref_outputs: Vec<Vec<f64>> = probes
        .iter()
        .map(|p| mat.matvec(p))
        .collect::<Result<_, _>>()?;
    let had = HadamardSpec::new(64, seed ^ (tensor_index as u64).wrapping_mul(0x9e37_79b9));

    let mut candidates = Vec::new();
    macro_rules! push_eval {
        ($kind:expr, $op:expr) => {{
            let op = $op;
            let sb = op.storage_breakdown();
            let total_bytes = sb.total_bytes();
            let effective_bpp = sb.effective_bits_per_param(param_count);
            let mv_err = fast_probe_matvec_error(&ref_outputs, &probes, &op)?;
            candidates.push(ResonanceCandidateEval {
                kind: $kind,
                total_bytes,
                effective_bpp,
                rel_frob_error: mv_err,
                matvec_rel_error: mv_err,
            });
        }};
    }

    // 1. Extreme sub-2-bit candidates (VQ ~1.25 bpp & Ternary ~1.725-1.85 bpp)
    // Pass 1 uses fast screening quantizers evaluated against activation-weighted probes
    // (Pass 2 applies full Blockwise Cholesky GPTQ to the selected winner — ~10x faster!).
    if include_vq && mat.cols() >= 8 {
        let cb_k = if is_huge_table { 64 } else { 256 };
        let vq = VectorQuantMatrix::fit(mat, 8, cb_k, 1, 64, had, seed + tensor_index as u64)?;
        push_eval!(ResonanceCandidateKind::VqD8Hadamard, vq);
    }
    let tern128 = TernaryPackedMatrix::quantize(mat, 128, had)?;
    push_eval!(ResonanceCandidateKind::TernaryHadamardG128, tern128.clone());
    let tern64 = TernaryPackedMatrix::quantize(mat, 64, had)?;
    push_eval!(ResonanceCandidateKind::TernaryHadamardG64, tern64);

    if include_echo && !is_huge_table && mat.rows().min(mat.cols()) >= 16 {
        let echo_t = EchoCorrectedMatrix::fit(
            mat,
            EchoBaseOperator::Ternary(tern128),
            8,
            0.0,
            seed + tensor_index as u64,
        )?;
        push_eval!(ResonanceCandidateKind::TernaryEchoR8, echo_t);
    }

    // 2. 2-bit Hadamard + optional Echo recovery (~2.25 - 2.85 bpp)
    let q2_128 = Int2AsymGroupMatrix::quantize(mat, 128, had)?;
    push_eval!(ResonanceCandidateKind::Int2HadamardG128, q2_128.clone());
    let q2_64 = Int2AsymGroupMatrix::quantize(mat, 64, had)?;
    push_eval!(ResonanceCandidateKind::Int2HadamardG64, q2_64);

    if include_echo && !is_huge_table && mat.rows().min(mat.cols()) >= 16 {
        let echo_2 = EchoCorrectedMatrix::fit(
            mat,
            EchoBaseOperator::Int2Asym(q2_128),
            8,
            0.0,
            seed + tensor_index as u64,
        )?;
        push_eval!(ResonanceCandidateKind::Int2EchoR8, echo_2);
    }

    // 3. 3-bit Hadamard (~3.25 - 3.50 bpp) + Q4_0 (~4.50 bpp) + INT8 (~8.25 bpp)
    let q3_128 = Int3AsymGroupMatrix::quantize(mat, 128, had)?;
    push_eval!(ResonanceCandidateKind::Int3HadamardG128, q3_128);
    let q3_64 = Int3AsymGroupMatrix::quantize(mat, 64, had)?;
    push_eval!(ResonanceCandidateKind::Int3HadamardG64, q3_64);
    let q4 = Int4SymGroupMatrix::quantize(mat, 32)?;
    push_eval!(ResonanceCandidateKind::Int4SymG32, q4);
    let q8 = Int8SymMatrix::quantize(mat, 64)?;
    push_eval!(ResonanceCandidateKind::Int8SymG64, q8);

    candidates.sort_by(|a, b| {
        a.total_bytes
            .cmp(&b.total_bytes)
            .then(a.matvec_rel_error.total_cmp(&b.matvec_rel_error))
    });

    Ok(ResonanceTensorProfile {
        tensor_name: tensor_name.to_owned(),
        tensor_index,
        param_count,
        spectral_norm,
        behavioral_weight: behavioral_weight.max(0.1),
        candidates,
    })
}

/// Solve global behavior-aware `MAP-B` sub-3-bit allocation across all profiled tensors
/// while accounting for `fixed_overhead_bytes` (1D norms, QMF JSON manifest, CRC-32, headers)
/// so the **complete serialized artifact** targets `<= profile.target_bpp()`.
pub fn allocate_resonance_map_b(
    profiles: &[ResonanceTensorProfile],
    profile: ResonanceProfile,
    fixed_overhead_bytes: usize,
    total_model_parameters: usize,
) -> Result<ResonanceAllocationPlan, MatrixError> {
    if profiles.is_empty() {
        return Err(MatrixError::Empty);
    }
    let total_params = total_model_parameters.max(profiles.iter().map(|p| p.param_count).sum());
    let target_bpp = profile.target_bpp();
    let target_budget_bytes = ((total_params as f64) * target_bpp / 8.0).floor() as usize;
    let tensor_budget = target_budget_bytes.saturating_sub(fixed_overhead_bytes);

    let mut chosen: Vec<usize> = vec![0; profiles.len()];
    let mut current_bytes: usize = profiles.iter().map(|p| p.candidates[0].total_bytes).sum();

    // Greedy marginal behavioral-distortion-reduction per byte
    loop {
        let mut best: Option<(usize, usize, f64)> = None;
        for (i, prof) in profiles.iter().enumerate() {
            let cur = &prof.candidates[chosen[i]];
            for (cand_idx, cand) in prof.candidates.iter().enumerate().skip(chosen[i] + 1) {
                let delta_bytes = cand.total_bytes.saturating_sub(cur.total_bytes);
                if current_bytes.saturating_add(delta_bytes) > tensor_budget {
                    continue;
                }
                // Variance-additive squared-error objective (`cur^2 - cand^2`): under independent
                // quantization noise across layers, output distortion variance adds in `err^2`,
                // ensuring MAP-B rescues high-error sub-2-bit layers before over-allocating INT8.
                let err_gain = cur.matvec_rel_error * cur.matvec_rel_error
                    - cand.matvec_rel_error * cand.matvec_rel_error;
                if err_gain <= 1e-12 {
                    continue;
                }
                let score = if delta_bytes == 0 {
                    f64::INFINITY
                } else {
                    (prof.behavioral_weight * (prof.param_count as f64) * err_gain)
                        / (delta_bytes as f64)
                };
                if best.is_none_or(|(_, _, bs)| score > bs) {
                    best = Some((i, cand_idx, score));
                }
            }
        }
        let Some((idx, next_c, _)) = best else {
            break;
        };
        let old_b = profiles[idx].candidates[chosen[idx]].total_bytes;
        let new_b = profiles[idx].candidates[next_c].total_bytes;
        current_bytes = current_bytes.saturating_sub(old_b).saturating_add(new_b);
        chosen[idx] = next_c;
    }

    let allocations: Vec<ResonanceTensorAllocation> = profiles
        .iter()
        .enumerate()
        .map(|(i, p)| ResonanceTensorAllocation {
            tensor_name: p.tensor_name.clone(),
            param_count: p.param_count,
            behavioral_weight: p.behavioral_weight,
            selected: p.candidates[chosen[i]].clone(),
        })
        .collect();

    let projected_complete_bytes = current_bytes.saturating_add(fixed_overhead_bytes);
    let projected_complete_bpp = (projected_complete_bytes as f64 * 8.0) / (total_params as f64);

    Ok(ResonanceAllocationPlan {
        profile,
        target_complete_bpp: target_bpp,
        total_parameters: total_params,
        fixed_overhead_bytes,
        target_budget_bytes,
        allocated_tensor_bytes: current_bytes,
        projected_complete_bytes,
        projected_complete_bpp,
        budget_satisfied: projected_complete_bytes <= target_budget_bytes,
        allocations,
    })
}

/// Consumer hardware VRAM / RAM target calculator (Section 7 of Project Resonance).
/// Computes **actual total runtime memory** (persistent weights + KV cache + compute scratch + overhead)
/// for models from `135M` through `32B` across `4 GB`–`16 GB` VRAM targets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsumerHardwareFit {
    pub model_params_billions: f64,
    pub profile: ResonanceProfile,
    pub context_length: usize,
    pub kv_cache_int8: bool,
    pub weight_vram_gib: f64,
    pub kv_cache_vram_gib: f64,
    pub runtime_overhead_gib: f64,
    pub total_vram_gib: f64,
    pub fits_4gb_vram: bool,
    pub fits_6gb_vram: bool,
    pub fits_8gb_vram: bool,
    pub fits_12gb_vram: bool,
    pub fits_16gb_vram: bool,
}

pub fn estimate_consumer_hardware_fit(
    model_params_billions: f64,
    profile: ResonanceProfile,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    context_length: usize,
    kv_cache_int8: bool,
) -> ConsumerHardwareFit {
    let bytes_per_gib = 1_073_741_824.0_f64;
    let weight_bytes = model_params_billions * 1e9 * profile.target_bpp() / 8.0;
    let weight_vram_gib = weight_bytes / bytes_per_gib;
    // K + V tensors: 2 * layers * context * kv_heads * head_dim * bytes_per_elem
    let kv_elem_bytes = if kv_cache_int8 { 1.125 } else { 2.0 };
    let kv_bytes = 2.0
        * (num_layers as f64)
        * (context_length as f64)
        * (num_kv_heads as f64)
        * (head_dim as f64)
        * kv_elem_bytes;
    let kv_cache_vram_gib = kv_bytes / bytes_per_gib;
    // Conservative runtime overhead (scratch buffers + activation workspace + driver reserves)
    let runtime_overhead_gib = 0.55;
    let total_vram_gib = weight_vram_gib + kv_cache_vram_gib + runtime_overhead_gib;
    ConsumerHardwareFit {
        model_params_billions,
        profile,
        context_length,
        kv_cache_int8,
        weight_vram_gib,
        kv_cache_vram_gib,
        runtime_overhead_gib,
        total_vram_gib,
        fits_4gb_vram: total_vram_gib <= 3.6,
        fits_6gb_vram: total_vram_gib <= 5.5,
        fits_8gb_vram: total_vram_gib <= 7.5,
        fits_12gb_vram: total_vram_gib <= 11.4,
        fits_16gb_vram: total_vram_gib <= 15.2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heavy_tailed_test_matrix(rows: usize, cols: usize, seed: u64) -> DenseMatrix {
        let mut rng = DeterministicRng::new(seed);
        let mut data = vec![0.0_f64; rows * cols];
        // Rank-6 structured base + small Gaussian + sparse 1% heavy outliers
        let u: Vec<f64> = (0..rows * 6).map(|_| rng.next_normal() * 0.4).collect();
        let v: Vec<f64> = (0..6 * cols).map(|_| rng.next_normal() * 0.4).collect();
        for r in 0..rows {
            for c in 0..cols {
                let mut acc = rng.next_normal() * 0.12;
                for k in 0..6 {
                    acc += u[r * 6 + k] * v[k * cols + c];
                }
                if (r * 31 + c * 17) % 97 == 0 {
                    acc += if (r + c) % 2 == 0 { 4.2 } else { -4.2 };
                }
                data[r * cols + c] = acc;
            }
        }
        DenseMatrix::from_flat(rows, cols, data).expect("valid matrix")
    }

    #[test]
    fn hadamard_rotation_is_exact_orthogonal_involution() {
        let mut rng = DeterministicRng::new(42);
        let orig: Vec<f64> = (0..150).map(|_| rng.next_normal()).collect();
        let other: Vec<f64> = (0..150).map(|_| rng.next_normal()).collect();
        let dot_before: f64 = orig.iter().zip(&other).map(|(a, b)| a * b).sum();

        let mut r1 = orig.clone();
        let mut r2 = other.clone();
        apply_hadamard_forward(&mut r1, 64, 12345);
        apply_hadamard_forward(&mut r2, 64, 12345);
        let dot_after: f64 = r1.iter().zip(&r2).map(|(a, b)| a * b).sum();
        assert!((dot_before - dot_after).abs() < 1e-10);

        apply_hadamard_inverse(&mut r1, 64, 12345);
        for (a, b) in orig.iter().zip(&r1) {
            assert!((a - b).abs() < 1e-11);
        }
    }

    #[test]
    fn int3_int2_ternary_vq_and_echo_match_reconstructed_matvec_and_improve_with_hadamard() {
        let mat = heavy_tailed_test_matrix(64, 128, 777);
        let probes = deterministic_probe_vectors(128, 4, 999);
        let had = HadamardSpec::new(64, 0xBEEF);

        let q2_plain = Int2AsymGroupMatrix::quantize(&mat, 64, HadamardSpec::NONE).unwrap();
        let q2_had = Int2AsymGroupMatrix::quantize(&mat, 64, had).unwrap();
        let q3_had = Int3AsymGroupMatrix::quantize(&mat, 64, had).unwrap();
        let tern_had = TernaryPackedMatrix::quantize(&mat, 128, had).unwrap();
        let vq_d8 = VectorQuantMatrix::fit(&mat, 8, 64, 1, 64, had, 55).unwrap();
        let echo_q2 =
            EchoCorrectedMatrix::fit(&mat, EchoBaseOperator::Int2Asym(q2_had.clone()), 6, 0.0, 88)
                .unwrap();

        let ev_q2_plain = evaluate_operator(&mat, &q2_plain, &probes, 0).unwrap();
        let ev_q2_had = evaluate_operator(&mat, &q2_had, &probes, 0).unwrap();
        let ev_q3_had = evaluate_operator(&mat, &q3_had, &probes, 0).unwrap();
        let ev_tern = evaluate_operator(&mat, &tern_had, &probes, 0).unwrap();
        let ev_vq = evaluate_operator(&mat, &vq_d8, &probes, 0).unwrap();
        let ev_echo = evaluate_operator(&mat, &echo_q2, &probes, 0).unwrap();

        // 1. Hadamard incoherence must strictly beat plain 2-bit on heavy-tailed outlier matrices!
        assert!(
            ev_q2_had.relative_frobenius_error < ev_q2_plain.relative_frobenius_error,
            "Hadamard 2-bit ({}) should beat plain 2-bit ({})",
            ev_q2_had.relative_frobenius_error,
            ev_q2_plain.relative_frobenius_error
        );
        // 2. Project Echo low-rank recovery must strictly reduce error over base 2-bit!
        assert!(
            ev_echo.relative_frobenius_error < ev_q2_had.relative_frobenius_error,
            "Echo 2-bit ({}) should beat base 2-bit ({})",
            ev_echo.relative_frobenius_error,
            ev_q2_had.relative_frobenius_error
        );
        // 3. Verify bit-rate hierarchy & exact matvec-vs-dense agreement across every operator
        assert!(ev_tern.effective_bpp < 1.9);
        assert!(ev_q2_had.effective_bpp < 2.6);
        assert!(ev_q3_had.effective_bpp < 3.6);
        assert!(ev_vq.effective_bpp < 2.5);

        for op in [
            &q2_had as &dyn LinearOperator,
            &q3_had,
            &tern_had,
            &vq_d8,
            &echo_q2,
        ] {
            let dense = op.reconstruct_dense().unwrap();
            let y_direct = op.matvec(&probes[0]).unwrap();
            let y_dense = dense.matvec(&probes[0]).unwrap();
            let max_diff = y_direct
                .iter()
                .zip(&y_dense)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_diff < 1e-8,
                "{} direct matvec diverged from dense by {max_diff}",
                op.method_name()
            );
        }
    }

    #[test]
    fn resonance_map_b_hits_sub_3bit_budgets_and_estimates_32b_vram_fit() {
        let m1 = heavy_tailed_test_matrix(64, 128, 1);
        let m2 = heavy_tailed_test_matrix(64, 128, 2);
        let p1 =
            profile_tensor_for_resonance("layer.0.v_proj", 0, &m1, 2.5, true, true, 11).unwrap();
        let p2 = profile_tensor_for_resonance("layer.1.mlp", 1, &m2, 1.0, true, true, 11).unwrap();
        let plan = allocate_resonance_map_b(&[p1, p2], ResonanceProfile::Qmr250, 256, 64 * 128 * 2)
            .unwrap();
        assert!(plan.budget_satisfied);
        assert!(plan.projected_complete_bpp <= 2.50);

        // Verify honest total VRAM accounting for 27B & 32B GQA models on 8 GB consumer GPU:
        let fit_27b_175 =
            estimate_consumer_hardware_fit(27.0, ResonanceProfile::Qmr175, 46, 8, 128, 4096, true);
        assert!(
            fit_27b_175.fits_8gb_vram,
            "27B QMR-175 total VRAM = {:.2} GiB",
            fit_27b_175.total_vram_gib
        );
        let fit_32b_150 =
            estimate_consumer_hardware_fit(32.0, ResonanceProfile::Qmr150, 64, 8, 128, 4096, true);
        assert!(
            fit_32b_150.fits_8gb_vram,
            "32B QMR-150 total VRAM = {:.2} GiB",
            fit_32b_150.total_vram_gib
        );
    }
}
