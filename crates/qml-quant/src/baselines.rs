//! Reproducible conventional quantization and low-rank baselines with exact byte accounting.
//!
//! Includes:
//! - `FP16` and `BF16` IEEE/bfloat representations
//! - `INT8` symmetric and asymmetric per-group quantization
//! - `INT4` symmetric (`Q4_0`-style) and asymmetric (`Q4_1`-style) packed nibble quantization
//! - Truncated SVD low-rank baseline with `FP32`, `FP16`, or `INT8` quantized factors
//! - Sparse top-k residual representation (`SparseResidual`)

use crate::{LinearOperator, StorageBreakdown};
use qml_core::{DenseMatrix, MatrixError, truncated_svd};
use qml_model::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};
use serde::{Deserialize, Serialize};

const BASE_HEADER_BYTES: usize = 16;

/// Dense IEEE-754 binary16 (`FP16`) baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct Fp16Matrix {
    rows: usize,
    cols: usize,
    data: Vec<u16>,
}

impl Fp16Matrix {
    pub fn quantize(matrix: &DenseMatrix) -> Self {
        let data = matrix
            .as_slice()
            .iter()
            .map(|&v| f32_to_f16(v as f32))
            .collect();
        Self {
            rows: matrix.rows(),
            cols: matrix.cols(),
            data,
        }
    }

    pub fn from_raw(rows: usize, cols: usize, data: Vec<u16>) -> Result<Self, MatrixError> {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::Empty);
        }
        if data.len() != rows.saturating_mul(cols) {
            return Err(MatrixError::ElementCountMismatch {
                expected: rows.saturating_mul(cols),
                actual: data.len(),
            });
        }
        Ok(Self { rows, cols, data })
    }

    pub fn raw_bits(&self) -> &[u16] {
        &self.data
    }

    /// Decode a single row without reconstructing the full matrix (for compressed embeddings).
    pub fn decode_row(&self, row: usize) -> Result<Vec<f64>, MatrixError> {
        if row >= self.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: self.rows,
                actual: row,
            });
        }
        let start = row * self.cols;
        Ok(self.data[start..start + self.cols]
            .iter()
            .map(|&bits| f64::from(f16_to_f32(bits)))
            .collect())
    }
}

impl LinearOperator for Fp16Matrix {
    fn method_name(&self) -> &'static str {
        "FP16"
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
        let mut out = vec![0.0_f64; self.rows];
        for (r, slot) in out.iter_mut().enumerate() {
            let row = &self.data[r * self.cols..(r + 1) * self.cols];
            *slot = row
                .iter()
                .zip(vector)
                .map(|(&bits, &x)| f64::from(f16_to_f32(bits)) * x)
                .sum();
        }
        Ok(out)
    }

    fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError> {
        let data = self
            .data
            .iter()
            .map(|&bits| f64::from(f16_to_f32(bits)))
            .collect();
        DenseMatrix::from_flat(self.rows, self.cols, data)
    }

    fn storage_breakdown(&self) -> StorageBreakdown {
        StorageBreakdown {
            payload_bytes: self.data.len() * 2,
            scale_metadata_bytes: 0,
            structure_bytes: 0,
            header_bytes: BASE_HEADER_BYTES,
        }
    }
}

/// Dense brain-float (`BF16`) baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct Bf16Matrix {
    rows: usize,
    cols: usize,
    data: Vec<u16>,
}

impl Bf16Matrix {
    pub fn quantize(matrix: &DenseMatrix) -> Self {
        let data = matrix
            .as_slice()
            .iter()
            .map(|&v| f32_to_bf16(v as f32))
            .collect();
        Self {
            rows: matrix.rows(),
            cols: matrix.cols(),
            data,
        }
    }

    pub fn from_raw(rows: usize, cols: usize, data: Vec<u16>) -> Result<Self, MatrixError> {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::Empty);
        }
        if data.len() != rows.saturating_mul(cols) {
            return Err(MatrixError::ElementCountMismatch {
                expected: rows.saturating_mul(cols),
                actual: data.len(),
            });
        }
        Ok(Self { rows, cols, data })
    }

    pub fn raw_bits(&self) -> &[u16] {
        &self.data
    }

    /// Decode a single row without reconstructing the full matrix (for compressed embeddings).
    pub fn decode_row(&self, row: usize) -> Result<Vec<f64>, MatrixError> {
        if row >= self.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: self.rows,
                actual: row,
            });
        }
        let start = row * self.cols;
        Ok(self.data[start..start + self.cols]
            .iter()
            .map(|&bits| f64::from(bf16_to_f32(bits)))
            .collect())
    }
}

impl LinearOperator for Bf16Matrix {
    fn method_name(&self) -> &'static str {
        "BF16"
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
        let mut out = vec![0.0_f64; self.rows];
        for (r, slot) in out.iter_mut().enumerate() {
            let row = &self.data[r * self.cols..(r + 1) * self.cols];
            *slot = row
                .iter()
                .zip(vector)
                .map(|(&bits, &x)| f64::from(bf16_to_f32(bits)) * x)
                .sum();
        }
        Ok(out)
    }

    fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError> {
        let data = self
            .data
            .iter()
            .map(|&bits| f64::from(bf16_to_f32(bits)))
            .collect();
        DenseMatrix::from_flat(self.rows, self.cols, data)
    }

    fn storage_breakdown(&self) -> StorageBreakdown {
        StorageBreakdown {
            payload_bytes: self.data.len() * 2,
            scale_metadata_bytes: 0,
            structure_bytes: 0,
            header_bytes: BASE_HEADER_BYTES,
        }
    }
}

/// Symmetric per-group `INT8` quantization with `FP16` block scales (`q in [-127, 127]`).
#[derive(Debug, Clone, PartialEq)]
pub struct Int8SymMatrix {
    rows: usize,
    cols: usize,
    group_size: usize,
    /// `f16` scale bits per group along row-major flattened elements.
    scales_f16: Vec<u16>,
    codes: Vec<i8>,
}

impl Int8SymMatrix {
    pub fn quantize(matrix: &DenseMatrix, group_size: usize) -> Result<Self, MatrixError> {
        let g = group_size.max(1);
        let data = matrix.as_slice();
        let num_groups = data.len().div_ceil(g);
        let mut scales_f16 = Vec::with_capacity(num_groups);
        let mut codes = Vec::with_capacity(data.len());

        for chunk in data.chunks(g) {
            let absmax = chunk.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            let scale = if absmax > 0.0 { absmax / 127.0 } else { 0.0 };
            let scale_bits = f32_to_f16(scale as f32);
            let eff_scale = f64::from(f16_to_f32(scale_bits));
            scales_f16.push(scale_bits);
            for &val in chunk {
                let q = if eff_scale > 0.0 {
                    (val / eff_scale).round().clamp(-127.0, 127.0) as i8
                } else {
                    0
                };
                codes.push(q);
            }
        }
        Ok(Self {
            rows: matrix.rows(),
            cols: matrix.cols(),
            group_size: g,
            scales_f16,
            codes,
        })
    }

    pub fn from_parts(
        rows: usize,
        cols: usize,
        group_size: usize,
        scales_f16: Vec<u16>,
        codes: Vec<i8>,
    ) -> Result<Self, MatrixError> {
        if rows == 0 || cols == 0 || group_size == 0 {
            return Err(MatrixError::Empty);
        }
        let expected = rows.saturating_mul(cols);
        if codes.len() != expected || scales_f16.len() != expected.div_ceil(group_size) {
            return Err(MatrixError::ElementCountMismatch {
                expected,
                actual: codes.len(),
            });
        }
        Ok(Self {
            rows,
            cols,
            group_size,
            scales_f16,
            codes,
        })
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    pub fn scales_f16(&self) -> &[u16] {
        &self.scales_f16
    }

    pub fn codes(&self) -> &[i8] {
        &self.codes
    }

    /// Decode a single row without reconstructing the full matrix.
    pub fn decode_row(&self, row: usize) -> Result<Vec<f64>, MatrixError> {
        if row >= self.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: self.rows,
                actual: row,
            });
        }
        let base = row * self.cols;
        Ok((0..self.cols)
            .map(|c| {
                let idx = base + c;
                let scale = f64::from(f16_to_f32(self.scales_f16[idx / self.group_size]));
                f64::from(self.codes[idx]) * scale
            })
            .collect())
    }
}

impl LinearOperator for Int8SymMatrix {
    fn method_name(&self) -> &'static str {
        "INT8_SYM"
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
        let mut out = vec![0.0_f64; self.rows];
        for (r, slot) in out.iter_mut().enumerate() {
            let base = r * self.cols;
            let mut acc = 0.0_f64;
            for (c, &x) in vector.iter().enumerate() {
                let idx = base + c;
                let scale = f64::from(f16_to_f32(self.scales_f16[idx / self.group_size]));
                acc += (f64::from(self.codes[idx]) * scale) * x;
            }
            *slot = acc;
        }
        Ok(out)
    }

    fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError> {
        let data: Vec<f64> = self
            .codes
            .iter()
            .enumerate()
            .map(|(idx, &q)| {
                let scale = f64::from(f16_to_f32(self.scales_f16[idx / self.group_size]));
                f64::from(q) * scale
            })
            .collect();
        DenseMatrix::from_flat(self.rows, self.cols, data)
    }

    fn storage_breakdown(&self) -> StorageBreakdown {
        StorageBreakdown {
            payload_bytes: self.codes.len(),
            scale_metadata_bytes: self.scales_f16.len() * 2,
            structure_bytes: 0,
            header_bytes: BASE_HEADER_BYTES + 4,
        }
    }
}

/// Asymmetric (affine) per-group `INT8` quantization with `FP16` scale and `FP16` minimum.
#[derive(Debug, Clone, PartialEq)]
pub struct Int8AsymMatrix {
    rows: usize,
    cols: usize,
    group_size: usize,
    scales_f16: Vec<u16>,
    mins_f16: Vec<u16>,
    codes: Vec<u8>,
}

impl Int8AsymMatrix {
    pub fn quantize(matrix: &DenseMatrix, group_size: usize) -> Result<Self, MatrixError> {
        let g = group_size.max(1);
        let data = matrix.as_slice();
        let num_groups = data.len().div_ceil(g);
        let mut scales_f16 = Vec::with_capacity(num_groups);
        let mut mins_f16 = Vec::with_capacity(num_groups);
        let mut codes = Vec::with_capacity(data.len());

        for chunk in data.chunks(g) {
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            for &val in chunk {
                if val < min_v {
                    min_v = val;
                }
                if val > max_v {
                    max_v = val;
                }
            }
            let min_bits = f32_to_f16(min_v as f32);
            let eff_min = f64::from(f16_to_f32(min_bits));
            let range = (max_v - eff_min).max(0.0);
            let scale_bits = f32_to_f16((range / 255.0) as f32);
            let eff_scale = f64::from(f16_to_f32(scale_bits));
            scales_f16.push(scale_bits);
            mins_f16.push(min_bits);
            for &val in chunk {
                let q = if eff_scale > 0.0 {
                    ((val - eff_min) / eff_scale).round().clamp(0.0, 255.0) as u8
                } else {
                    0
                };
                codes.push(q);
            }
        }
        Ok(Self {
            rows: matrix.rows(),
            cols: matrix.cols(),
            group_size: g,
            scales_f16,
            mins_f16,
            codes,
        })
    }
}

impl LinearOperator for Int8AsymMatrix {
    fn method_name(&self) -> &'static str {
        "INT8_ASYM"
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
        let mut out = vec![0.0_f64; self.rows];
        for (r, slot) in out.iter_mut().enumerate() {
            let base = r * self.cols;
            let mut acc = 0.0_f64;
            for (c, &x) in vector.iter().enumerate() {
                let idx = base + c;
                let g = idx / self.group_size;
                let scale = f64::from(f16_to_f32(self.scales_f16[g]));
                let min_v = f64::from(f16_to_f32(self.mins_f16[g]));
                acc += (f64::from(self.codes[idx]) * scale + min_v) * x;
            }
            *slot = acc;
        }
        Ok(out)
    }

    fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError> {
        let data: Vec<f64> = self
            .codes
            .iter()
            .enumerate()
            .map(|(idx, &q)| {
                let g = idx / self.group_size;
                let scale = f64::from(f16_to_f32(self.scales_f16[g]));
                let min_v = f64::from(f16_to_f32(self.mins_f16[g]));
                f64::from(q) * scale + min_v
            })
            .collect();
        DenseMatrix::from_flat(self.rows, self.cols, data)
    }

    fn storage_breakdown(&self) -> StorageBreakdown {
        StorageBreakdown {
            payload_bytes: self.codes.len(),
            scale_metadata_bytes: (self.scales_f16.len() + self.mins_f16.len()) * 2,
            structure_bytes: 0,
            header_bytes: BASE_HEADER_BYTES + 4,
        }
    }
}

/// Symmetric packed 4-bit block quantization (GGML `Q4_0`-style: centered `[-8, 7]`, `FP16` scale).
///
/// Two 4-bit codes are packed per byte (low nibble = even index, high nibble = odd index).
/// With `group_size = 32`: 16 packed bytes + 2 scale bytes = 18 bytes / 32 weights = 4.50 bpp
/// (plus fixed tensor header).
#[derive(Debug, Clone, PartialEq)]
pub struct Int4SymGroupMatrix {
    rows: usize,
    cols: usize,
    group_size: usize,
    scales_f16: Vec<u16>,
    packed_nibbles: Vec<u8>,
}

impl Int4SymGroupMatrix {
    pub fn quantize(matrix: &DenseMatrix, group_size: usize) -> Result<Self, MatrixError> {
        let g = group_size.max(2);
        let data = matrix.as_slice();
        let num_groups = data.len().div_ceil(g);
        let mut scales_f16 = Vec::with_capacity(num_groups);
        let mut packed_nibbles = vec![0_u8; data.len().div_ceil(2)];

        for (group_idx, chunk) in data.chunks(g).enumerate() {
            let absmax = chunk.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            let scale = if absmax > 0.0 { absmax / 7.0 } else { 0.0 };
            let scale_bits = f32_to_f16(scale as f32);
            let eff_scale = f64::from(f16_to_f32(scale_bits));
            scales_f16.push(scale_bits);

            let base_idx = group_idx * g;
            for (offset, &val) in chunk.iter().enumerate() {
                let q_signed = if eff_scale > 0.0 {
                    (val / eff_scale).round().clamp(-8.0, 7.0) as i8
                } else {
                    0
                };
                let nibble = ((q_signed + 8) as u8) & 0x0f;
                let flat_idx = base_idx + offset;
                if flat_idx.is_multiple_of(2) {
                    packed_nibbles[flat_idx / 2] |= nibble;
                } else {
                    packed_nibbles[flat_idx / 2] |= nibble << 4;
                }
            }
        }

        Ok(Self {
            rows: matrix.rows(),
            cols: matrix.cols(),
            group_size: g,
            scales_f16,
            packed_nibbles,
        })
    }

    pub fn from_parts(
        rows: usize,
        cols: usize,
        group_size: usize,
        scales_f16: Vec<u16>,
        packed_nibbles: Vec<u8>,
    ) -> Result<Self, MatrixError> {
        if rows == 0 || cols == 0 || group_size == 0 {
            return Err(MatrixError::Empty);
        }
        let expected = rows.saturating_mul(cols);
        if packed_nibbles.len() != expected.div_ceil(2)
            || scales_f16.len() != expected.div_ceil(group_size)
        {
            return Err(MatrixError::ElementCountMismatch {
                expected: expected.div_ceil(2),
                actual: packed_nibbles.len(),
            });
        }
        Ok(Self {
            rows,
            cols,
            group_size,
            scales_f16,
            packed_nibbles,
        })
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    pub fn scales_f16(&self) -> &[u16] {
        &self.scales_f16
    }

    pub fn packed_nibbles(&self) -> &[u8] {
        &self.packed_nibbles
    }

    /// Decode a single row without reconstructing the full matrix.
    pub fn decode_row(&self, row: usize) -> Result<Vec<f64>, MatrixError> {
        if row >= self.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: self.rows,
                actual: row,
            });
        }
        let base = row * self.cols;
        Ok((0..self.cols).map(|c| self.decode_at(base + c)).collect())
    }

    #[inline]
    fn decode_at(&self, flat_idx: usize) -> f64 {
        let byte = self.packed_nibbles[flat_idx / 2];
        let nibble = if flat_idx.is_multiple_of(2) {
            byte & 0x0f
        } else {
            (byte >> 4) & 0x0f
        };
        let q = i32::from(nibble) - 8;
        let scale = f64::from(f16_to_f32(self.scales_f16[flat_idx / self.group_size]));
        f64::from(q) * scale
    }
}

impl LinearOperator for Int4SymGroupMatrix {
    fn method_name(&self) -> &'static str {
        match self.group_size {
            32 => "INT4_SYM_G32(Q4_0)",
            64 => "INT4_SYM_G64",
            128 => "INT4_SYM_G128",
            _ => "INT4_SYM",
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
        let mut out = vec![0.0_f64; self.rows];
        for (r, slot) in out.iter_mut().enumerate() {
            let base = r * self.cols;
            let mut acc = 0.0_f64;
            for (c, &x) in vector.iter().enumerate() {
                acc += self.decode_at(base + c) * x;
            }
            *slot = acc;
        }
        Ok(out)
    }

    fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError> {
        let total = self.rows * self.cols;
        let data = (0..total).map(|idx| self.decode_at(idx)).collect();
        DenseMatrix::from_flat(self.rows, self.cols, data)
    }

    fn storage_breakdown(&self) -> StorageBreakdown {
        StorageBreakdown {
            payload_bytes: self.packed_nibbles.len(),
            scale_metadata_bytes: self.scales_f16.len() * 2,
            structure_bytes: 0,
            header_bytes: BASE_HEADER_BYTES + 4,
        }
    }
}

/// Asymmetric packed 4-bit block quantization (GGML `Q4_1`-style: `q in [0, 15]`, `FP16` scale + min).
///
/// With `group_size = 32`: 16 packed bytes + 4 metadata bytes = 20 bytes / 32 weights = 5.00 bpp
/// (plus fixed tensor header).
#[derive(Debug, Clone, PartialEq)]
pub struct Int4AsymGroupMatrix {
    rows: usize,
    cols: usize,
    group_size: usize,
    scales_f16: Vec<u16>,
    mins_f16: Vec<u16>,
    packed_nibbles: Vec<u8>,
}

impl Int4AsymGroupMatrix {
    pub fn quantize(matrix: &DenseMatrix, group_size: usize) -> Result<Self, MatrixError> {
        let g = group_size.max(2);
        let data = matrix.as_slice();
        let num_groups = data.len().div_ceil(g);
        let mut scales_f16 = Vec::with_capacity(num_groups);
        let mut mins_f16 = Vec::with_capacity(num_groups);
        let mut packed_nibbles = vec![0_u8; data.len().div_ceil(2)];

        for (group_idx, chunk) in data.chunks(g).enumerate() {
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            for &val in chunk {
                if val < min_v {
                    min_v = val;
                }
                if val > max_v {
                    max_v = val;
                }
            }
            let min_bits = f32_to_f16(min_v as f32);
            let eff_min = f64::from(f16_to_f32(min_bits));
            let range = (max_v - eff_min).max(0.0);
            let scale_bits = f32_to_f16((range / 15.0) as f32);
            let eff_scale = f64::from(f16_to_f32(scale_bits));
            scales_f16.push(scale_bits);
            mins_f16.push(min_bits);

            let base_idx = group_idx * g;
            for (offset, &val) in chunk.iter().enumerate() {
                let nibble = if eff_scale > 0.0 {
                    ((val - eff_min) / eff_scale).round().clamp(0.0, 15.0) as u8
                } else {
                    0
                };
                let flat_idx = base_idx + offset;
                if flat_idx.is_multiple_of(2) {
                    packed_nibbles[flat_idx / 2] |= nibble & 0x0f;
                } else {
                    packed_nibbles[flat_idx / 2] |= (nibble & 0x0f) << 4;
                }
            }
        }

        Ok(Self {
            rows: matrix.rows(),
            cols: matrix.cols(),
            group_size: g,
            scales_f16,
            mins_f16,
            packed_nibbles,
        })
    }

    pub fn from_parts(
        rows: usize,
        cols: usize,
        group_size: usize,
        scales_f16: Vec<u16>,
        mins_f16: Vec<u16>,
        packed_nibbles: Vec<u8>,
    ) -> Result<Self, MatrixError> {
        if rows == 0 || cols == 0 || group_size == 0 {
            return Err(MatrixError::Empty);
        }
        let expected = rows.saturating_mul(cols);
        let groups = expected.div_ceil(group_size);
        if packed_nibbles.len() != expected.div_ceil(2)
            || scales_f16.len() != groups
            || mins_f16.len() != groups
        {
            return Err(MatrixError::ElementCountMismatch {
                expected: expected.div_ceil(2),
                actual: packed_nibbles.len(),
            });
        }
        Ok(Self {
            rows,
            cols,
            group_size,
            scales_f16,
            mins_f16,
            packed_nibbles,
        })
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    pub fn scales_f16(&self) -> &[u16] {
        &self.scales_f16
    }

    pub fn mins_f16(&self) -> &[u16] {
        &self.mins_f16
    }

    pub fn packed_nibbles(&self) -> &[u8] {
        &self.packed_nibbles
    }

    /// Decode a single row without reconstructing the full matrix.
    pub fn decode_row(&self, row: usize) -> Result<Vec<f64>, MatrixError> {
        if row >= self.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: self.rows,
                actual: row,
            });
        }
        let base = row * self.cols;
        Ok((0..self.cols).map(|c| self.decode_at(base + c)).collect())
    }

    #[inline]
    fn decode_at(&self, flat_idx: usize) -> f64 {
        let byte = self.packed_nibbles[flat_idx / 2];
        let nibble = if flat_idx.is_multiple_of(2) {
            byte & 0x0f
        } else {
            (byte >> 4) & 0x0f
        };
        let g = flat_idx / self.group_size;
        let scale = f64::from(f16_to_f32(self.scales_f16[g]));
        let min_v = f64::from(f16_to_f32(self.mins_f16[g]));
        f64::from(nibble) * scale + min_v
    }
}

impl LinearOperator for Int4AsymGroupMatrix {
    fn method_name(&self) -> &'static str {
        match self.group_size {
            32 => "INT4_ASYM_G32(Q4_1)",
            64 => "INT4_ASYM_G64",
            128 => "INT4_ASYM_G128",
            _ => "INT4_ASYM",
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
        let mut out = vec![0.0_f64; self.rows];
        for (r, slot) in out.iter_mut().enumerate() {
            let base = r * self.cols;
            let mut acc = 0.0_f64;
            for (c, &x) in vector.iter().enumerate() {
                acc += self.decode_at(base + c) * x;
            }
            *slot = acc;
        }
        Ok(out)
    }

    fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError> {
        let total = self.rows * self.cols;
        let data = (0..total).map(|idx| self.decode_at(idx)).collect();
        DenseMatrix::from_flat(self.rows, self.cols, data)
    }

    fn storage_breakdown(&self) -> StorageBreakdown {
        StorageBreakdown {
            payload_bytes: self.packed_nibbles.len(),
            scale_metadata_bytes: (self.scales_f16.len() + self.mins_f16.len()) * 2,
            structure_bytes: 0,
            header_bytes: BASE_HEADER_BYTES + 4,
        }
    }
}

/// Storage precision for low-rank / RMQ factor matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactorPrecision {
    Fp32,
    Fp16,
    Int8Sym,
}

/// Internal quantized matrix container used for factor matrices.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantizedFactor {
    Fp32(DenseMatrix),
    Fp16(Fp16Matrix),
    Int8Sym(Int8SymMatrix),
}

impl QuantizedFactor {
    pub fn encode(matrix: &DenseMatrix, precision: FactorPrecision) -> Result<Self, MatrixError> {
        Ok(match precision {
            FactorPrecision::Fp32 => {
                // Round through f32 so reconstruction matches serialized f32 exactly.
                let data: Vec<f64> = matrix
                    .as_slice()
                    .iter()
                    .map(|&v| f64::from(v as f32))
                    .collect();
                Self::Fp32(DenseMatrix::from_flat(matrix.rows(), matrix.cols(), data)?)
            }
            FactorPrecision::Fp16 => Self::Fp16(Fp16Matrix::quantize(matrix)),
            FactorPrecision::Int8Sym => Self::Int8Sym(Int8SymMatrix::quantize(matrix, 64)?),
        })
    }

    pub fn precision(&self) -> FactorPrecision {
        match self {
            Self::Fp32(_) => FactorPrecision::Fp32,
            Self::Fp16(_) => FactorPrecision::Fp16,
            Self::Int8Sym(_) => FactorPrecision::Int8Sym,
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::Fp32(m) => m.rows(),
            Self::Fp16(m) => m.rows(),
            Self::Int8Sym(m) => m.rows(),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Fp32(m) => m.cols(),
            Self::Fp16(m) => m.cols(),
            Self::Int8Sym(m) => m.cols(),
        }
    }

    pub fn matvec(&self, vector: &[f64]) -> Result<Vec<f64>, MatrixError> {
        match self {
            Self::Fp32(m) => m.matvec(vector),
            Self::Fp16(m) => m.matvec(vector),
            Self::Int8Sym(m) => m.matvec(vector),
        }
    }

    pub fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError> {
        match self {
            Self::Fp32(m) => Ok(m.clone()),
            Self::Fp16(m) => m.reconstruct_dense(),
            Self::Int8Sym(m) => m.reconstruct_dense(),
        }
    }

    pub fn raw_storage_bytes(&self) -> (usize, usize) {
        // Returns (payload_bytes, scale_metadata_bytes)
        match self {
            Self::Fp32(m) => (m.rows() * m.cols() * 4, 0),
            Self::Fp16(m) => (m.rows() * m.cols() * 2, 0),
            Self::Int8Sym(m) => {
                let sb = m.storage_breakdown();
                (sb.payload_bytes, sb.scale_metadata_bytes)
            }
        }
    }
}

/// Conventional truncated-SVD low-rank baseline `W ≈ L * R` with quantized factors.
///
/// Absorbs `sqrt(Σ)` symmetrically into `L = U Σ^{1/2}` and `R = Σ^{1/2} Vᵀ` for balanced dynamic range.
#[derive(Debug, Clone, PartialEq)]
pub struct LowRankSvdMatrix {
    rows: usize,
    cols: usize,
    rank: usize,
    precision: FactorPrecision,
    left: QuantizedFactor,
    right: QuantizedFactor,
}

impl LowRankSvdMatrix {
    pub fn from_matrix(
        matrix: &DenseMatrix,
        rank: usize,
        precision: FactorPrecision,
        seed: u64,
    ) -> Result<Self, MatrixError> {
        let svd = truncated_svd(matrix, rank, seed)?;
        let rows = matrix.rows();
        let cols = matrix.cols();
        let r = svd.rank();

        let mut left_data = vec![0.0_f64; rows * r];
        let mut right_data = vec![0.0_f64; r * cols];
        for k in 0..r {
            let s_sqrt = svd.singular_values[k].max(0.0).sqrt();
            for row in 0..rows {
                left_data[row * r + k] = svd.u.as_slice()[row * r + k] * s_sqrt;
            }
            for col in 0..cols {
                right_data[k * cols + col] = svd.vt.as_slice()[k * cols + col] * s_sqrt;
            }
        }
        let left_dense = DenseMatrix::from_flat(rows, r, left_data)?;
        let right_dense = DenseMatrix::from_flat(r, cols, right_data)?;
        Ok(Self {
            rows,
            cols,
            rank: r,
            precision,
            left: QuantizedFactor::encode(&left_dense, precision)?,
            right: QuantizedFactor::encode(&right_dense, precision)?,
        })
    }

    pub fn from_factors(
        left: QuantizedFactor,
        right: QuantizedFactor,
    ) -> Result<Self, MatrixError> {
        if left.cols() != right.rows() {
            return Err(MatrixError::DimensionMismatch {
                expected: left.cols(),
                actual: right.rows(),
            });
        }
        Ok(Self {
            rows: left.rows(),
            cols: right.cols(),
            rank: left.cols(),
            precision: left.precision(),
            left,
            right,
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn left(&self) -> &QuantizedFactor {
        &self.left
    }

    pub fn right(&self) -> &QuantizedFactor {
        &self.right
    }
}

impl LinearOperator for LowRankSvdMatrix {
    fn method_name(&self) -> &'static str {
        match self.precision {
            FactorPrecision::Fp32 => "SVD_FP32",
            FactorPrecision::Fp16 => "SVD_FP16",
            FactorPrecision::Int8Sym => "SVD_INT8",
        }
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn matvec(&self, vector: &[f64]) -> Result<Vec<f64>, MatrixError> {
        let mid = self.right.matvec(vector)?;
        self.left.matvec(&mid)
    }

    fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError> {
        let l = self.left.reconstruct_dense()?;
        let r = self.right.reconstruct_dense()?;
        l.matmul(&r)
    }

    fn storage_breakdown(&self) -> StorageBreakdown {
        let (lp, ls) = self.left.raw_storage_bytes();
        let (rp, rs) = self.right.raw_storage_bytes();
        StorageBreakdown {
            payload_bytes: 0,
            scale_metadata_bytes: ls + rs,
            structure_bytes: lp + rp,
            header_bytes: BASE_HEADER_BYTES + 8,
        }
    }
}

/// Sparse COO outlier residual storing top-k magnitude entries as `(u32 flat_index, u16 f16_value)`.
///
/// Each nonzero consumes exactly 6 bytes (4-byte index + 2-byte FP16 value).
#[derive(Debug, Clone, PartialEq)]
pub struct SparseResidual {
    rows: usize,
    cols: usize,
    indices: Vec<u32>,
    values_f16: Vec<u16>,
}

impl SparseResidual {
    pub fn empty(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            indices: Vec::new(),
            values_f16: Vec::new(),
        }
    }

    /// Extract the top `fraction` (e.g., `0.01` = 1%) largest-magnitude residual entries.
    pub fn from_residual_top_fraction(residual: &DenseMatrix, fraction: f64) -> Self {
        let total = residual.rows() * residual.cols();
        let k = ((total as f64) * fraction.clamp(0.0, 1.0)).round() as usize;
        Self::from_residual_top_k(residual, k)
    }

    pub fn from_residual_top_k(residual: &DenseMatrix, k: usize) -> Self {
        if k == 0 {
            return Self::empty(residual.rows(), residual.cols());
        }
        let slice = residual.as_slice();
        let mut order: Vec<usize> = (0..slice.len()).collect();
        let keep = k.min(slice.len());
        order.sort_by(|&a, &b| slice[b].abs().total_cmp(&slice[a].abs()).then(a.cmp(&b)));
        order.truncate(keep);
        order.sort_unstable();

        let mut indices = Vec::with_capacity(keep);
        let mut values_f16 = Vec::with_capacity(keep);
        for idx in order {
            let v = slice[idx];
            if v == 0.0 {
                continue;
            }
            if let Ok(u32_idx) = u32::try_from(idx) {
                indices.push(u32_idx);
                values_f16.push(f32_to_f16(v as f32));
            }
        }
        Self {
            rows: residual.rows(),
            cols: residual.cols(),
            indices,
            values_f16,
        }
    }

    pub fn from_parts(
        rows: usize,
        cols: usize,
        indices: Vec<u32>,
        values_f16: Vec<u16>,
    ) -> Result<Self, MatrixError> {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::Empty);
        }
        if indices.len() != values_f16.len() {
            return Err(MatrixError::ElementCountMismatch {
                expected: indices.len(),
                actual: values_f16.len(),
            });
        }
        let limit = rows.saturating_mul(cols);
        for &idx in &indices {
            if (idx as usize) >= limit {
                return Err(MatrixError::ElementCountMismatch {
                    expected: limit,
                    actual: idx as usize,
                });
            }
        }
        Ok(Self {
            rows,
            cols,
            indices,
            values_f16,
        })
    }

    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn values_f16(&self) -> &[u16] {
        &self.values_f16
    }

    /// Add `self * vector` directly into `output` without reconstructing a dense matrix.
    pub fn add_matvec_into(&self, vector: &[f64], output: &mut [f64]) -> Result<(), MatrixError> {
        if vector.len() != self.cols {
            return Err(MatrixError::DimensionMismatch {
                expected: self.cols,
                actual: vector.len(),
            });
        }
        if output.len() != self.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: self.rows,
                actual: output.len(),
            });
        }
        for (&idx, &val_bits) in self.indices.iter().zip(&self.values_f16) {
            let flat = idx as usize;
            let row = flat / self.cols;
            let col = flat % self.cols;
            output[row] += f64::from(f16_to_f32(val_bits)) * vector[col];
        }
        Ok(())
    }

    /// Add this sparse residual into a mutable dense buffer.
    pub fn add_dense_into(&self, dense_data: &mut [f64]) {
        for (&idx, &val_bits) in self.indices.iter().zip(&self.values_f16) {
            dense_data[idx as usize] += f64::from(f16_to_f32(val_bits));
        }
    }

    /// Exact byte cost of sparse entries (4-byte index + 2-byte f16 value per nnz).
    pub fn entry_bytes(&self) -> usize {
        self.indices.len() * 6
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{benchmark_conventional_baselines, deterministic_probe_vectors};
    use qml_core::DeterministicRng;

    fn sample_matrix(rows: usize, cols: usize, seed: u64) -> DenseMatrix {
        let mut rng = DeterministicRng::new(seed);
        let data = (0..rows * cols).map(|_| rng.next_normal() * 0.25).collect();
        DenseMatrix::from_flat(rows, cols, data).unwrap()
    }

    #[test]
    fn direct_matvec_matches_reconstructed_dense_matvec_across_all_baselines() {
        let mat = sample_matrix(32, 64, 77);
        let x = &deterministic_probe_vectors(64, 1, 99)[0];

        let ops: Vec<Box<dyn LinearOperator>> = vec![
            Box::new(Fp16Matrix::quantize(&mat)),
            Box::new(Bf16Matrix::quantize(&mat)),
            Box::new(Int8SymMatrix::quantize(&mat, 32).unwrap()),
            Box::new(Int8AsymMatrix::quantize(&mat, 32).unwrap()),
            Box::new(Int4SymGroupMatrix::quantize(&mat, 32).unwrap()),
            Box::new(Int4AsymGroupMatrix::quantize(&mat, 32).unwrap()),
            Box::new(LowRankSvdMatrix::from_matrix(&mat, 8, FactorPrecision::Fp16, 7).unwrap()),
            Box::new(LowRankSvdMatrix::from_matrix(&mat, 8, FactorPrecision::Int8Sym, 7).unwrap()),
        ];

        for op in ops {
            let direct = op.matvec(x).unwrap();
            let dense_y = op.reconstruct_dense().unwrap().matvec(x).unwrap();
            for (a, b) in direct.iter().zip(dense_y) {
                assert!(
                    (a - b).abs() < 1e-10,
                    "method {} direct vs dense mismatch: {a} vs {b}",
                    op.method_name()
                );
            }
        }
    }

    #[test]
    fn storage_accounting_matches_theoretical_block_sizes() {
        let mat = sample_matrix(64, 64, 11); // 4096 params
        let q4_0 = Int4SymGroupMatrix::quantize(&mat, 32).unwrap();
        let sb = q4_0.storage_breakdown();
        // 4096 / 2 = 2048 packed bytes; 128 groups * 2B = 256 scale bytes -> 2304B = 4.50 bpp + 20B header
        assert_eq!(sb.payload_bytes, 2048);
        assert_eq!(sb.scale_metadata_bytes, 256);
        assert!(
            sb.effective_bits_per_param(4096) > 4.50 && sb.effective_bits_per_param(4096) < 4.60
        );

        let q4_1 = Int4AsymGroupMatrix::quantize(&mat, 32).unwrap();
        let sb1 = q4_1.storage_breakdown();
        assert_eq!(sb1.payload_bytes, 2048);
        assert_eq!(sb1.scale_metadata_bytes, 512);
        assert!(
            sb1.effective_bits_per_param(4096) > 5.00 && sb1.effective_bits_per_param(4096) < 5.10
        );
    }

    #[test]
    fn baseline_error_orders_sensibly_from_fp16_to_int8_to_int4() {
        let mat = sample_matrix(64, 64, 55);
        let report = benchmark_conventional_baselines(&mat, 16, 42).unwrap();
        let err = |name: &str| {
            report
                .iter()
                .find(|r| r.method == name)
                .unwrap()
                .relative_frobenius_error
        };
        assert!(err("FP16") < err("BF16"));
        assert!(err("BF16") < err("INT8_SYM"));
        assert!(err("INT8_SYM") < err("INT4_SYM_G32(Q4_0)"));
        assert!(err("INT4_SYM_G32(Q4_0)") < 0.15);
    }
}
