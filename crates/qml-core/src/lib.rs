//! Small, safe mathematical building blocks for Quantum Morph Laboratory.
//!
//! These are conventional foundations, not novel compression methods. The
//! rank-one pivot approximation here is a simple baseline, not an optimal
//! truncated-SVD approximation.

pub mod rng;
pub mod svd;

pub use rng::DeterministicRng;
pub use svd::{
    SpectrumMetrics, SvdResult, exact_thin_svd, randomized_svd, spectrum_metrics_from_values,
    subspace_affinity_columns, truncated_svd,
};

use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixError {
    Empty,
    RaggedRows,
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    ShapeMismatch {
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    ElementCountMismatch {
        expected: usize,
        actual: usize,
    },
    NonFiniteValue {
        index: usize,
    },
    InvalidRank {
        requested: usize,
        max_rank: usize,
    },
}

impl Display for MatrixError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "matrix and factor dimensions must be non-zero"),
            Self::RaggedRows => write!(f, "all matrix rows must have the same length"),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "vector length mismatch: expected {expected}, received {actual}"
            ),
            Self::ShapeMismatch {
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            } => write!(
                f,
                "matrix shape mismatch: expected {expected_rows}x{expected_cols}, received {actual_rows}x{actual_cols}"
            ),
            Self::ElementCountMismatch { expected, actual } => write!(
                f,
                "matrix element count mismatch: expected {expected}, received {actual}"
            ),
            Self::NonFiniteValue { index } => {
                write!(f, "non-finite value at flat element index {index}")
            }
            Self::InvalidRank {
                requested,
                max_rank,
            } => write!(f, "invalid rank {requested}: must be within 1..={max_rank}"),
        }
    }
}

impl Error for MatrixError {}

/// A row-major dense matrix stored in host `f64` values.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl DenseMatrix {
    /// Construct from row slices, rejecting empty, inconsistent, or non-finite input.
    pub fn from_rows(rows: &[&[f64]]) -> Result<Self, MatrixError> {
        let Some(first) = rows.first() else {
            return Err(MatrixError::Empty);
        };
        let cols = first.len();
        if cols == 0 {
            return Err(MatrixError::Empty);
        }
        if rows.iter().any(|row| row.len() != cols) {
            return Err(MatrixError::RaggedRows);
        }
        let data = rows.iter().flat_map(|row| row.iter().copied()).collect();
        Self::from_flat(rows.len(), cols, data)
    }

    /// Construct from row-major data after validating shape and values.
    pub fn from_flat(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, MatrixError> {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::Empty);
        }
        let Some(expected) = rows.checked_mul(cols) else {
            return Err(MatrixError::ElementCountMismatch {
                expected: usize::MAX,
                actual: data.len(),
            });
        };
        if data.len() != expected {
            return Err(MatrixError::ElementCountMismatch {
                expected,
                actual: data.len(),
            });
        }
        if let Some(index) = data.iter().position(|value| !value.is_finite()) {
            return Err(MatrixError::NonFiniteValue { index });
        }
        Ok(Self { rows, cols, data })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        Some(self.data[row * self.cols + col])
    }

    /// Compute `self * vector` on the CPU without allocating an intermediate matrix.
    pub fn matvec(&self, vector: &[f64]) -> Result<Vec<f64>, MatrixError> {
        if vector.len() != self.cols {
            return Err(MatrixError::DimensionMismatch {
                expected: self.cols,
                actual: vector.len(),
            });
        }
        let mut output = vec![0.0; self.rows];
        for (row_index, output_value) in output.iter_mut().enumerate() {
            let start = row_index * self.cols;
            *output_value = self.data[start..start + self.cols]
                .iter()
                .zip(vector)
                .map(|(weight, value)| weight * value)
                .sum();
        }
        Ok(output)
    }

    /// Return the Frobenius norm, `sqrt(sum(x^2))`.
    pub fn frobenius_norm(&self) -> f64 {
        self.data
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
    }

    /// Construct an all-zero matrix with validated non-zero dimensions.
    pub fn zeros(rows: usize, cols: usize) -> Result<Self, MatrixError> {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::Empty);
        }
        let Some(count) = rows.checked_mul(cols) else {
            return Err(MatrixError::ElementCountMismatch {
                expected: usize::MAX,
                actual: 0,
            });
        };
        Ok(Self {
            rows,
            cols,
            data: vec![0.0; count],
        })
    }

    /// Return the transpose `selfᵀ` (`cols × rows`).
    pub fn transpose(&self) -> Self {
        let mut out = vec![0.0_f64; self.data.len()];
        for r in 0..self.rows {
            let row_start = r * self.cols;
            for c in 0..self.cols {
                out[c * self.rows + r] = self.data[row_start + c];
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            data: out,
        }
    }

    /// Compute `selfᵀ * vector` without materializing the transpose.
    pub fn matvec_transposed(&self, vector: &[f64]) -> Result<Vec<f64>, MatrixError> {
        if vector.len() != self.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: self.rows,
                actual: vector.len(),
            });
        }
        let mut output = vec![0.0_f64; self.cols];
        for (r, &v) in vector.iter().enumerate() {
            if v == 0.0 {
                continue;
            }
            let row = &self.data[r * self.cols..(r + 1) * self.cols];
            for (out, &w) in output.iter_mut().zip(row) {
                *out += w * v;
            }
        }
        Ok(output)
    }

    /// Compute dense matrix-matrix product `self * other`.
    pub fn matmul(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: self.cols,
                actual: other.rows,
            });
        }
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;
        let Some(count) = m.checked_mul(n) else {
            return Err(MatrixError::ElementCountMismatch {
                expected: usize::MAX,
                actual: 0,
            });
        };
        let mut out = vec![0.0_f64; count];
        for i in 0..m {
            let a_row = &self.data[i * k..(i + 1) * k];
            let c_row = &mut out[i * n..(i + 1) * n];
            for (p, &a_ip) in a_row.iter().enumerate() {
                if a_ip == 0.0 {
                    continue;
                }
                let b_row = &other.data[p * n..(p + 1) * n];
                for (c_ij, &b_pj) in c_row.iter_mut().zip(b_row) {
                    *c_ij += a_ip * b_pj;
                }
            }
        }
        Self::from_flat(m, n, out)
    }

    /// Element-wise addition `self + other`.
    pub fn add(&self, other: &Self) -> Result<Self, MatrixError> {
        self.check_same_shape(other)?;
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Self::from_flat(self.rows, self.cols, data)
    }

    /// Element-wise subtraction `self - other`.
    pub fn sub(&self, other: &Self) -> Result<Self, MatrixError> {
        self.check_same_shape(other)?;
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();
        Self::from_flat(self.rows, self.cols, data)
    }

    /// Scalar multiplication `factor * self`.
    pub fn scale(&self, factor: f64) -> Result<Self, MatrixError> {
        if !factor.is_finite() {
            return Err(MatrixError::NonFiniteValue { index: 0 });
        }
        let data = self.data.iter().map(|v| v * factor).collect();
        Self::from_flat(self.rows, self.cols, data)
    }

    /// Frobenius inner product `sum(self[i,j] * other[i,j])`.
    pub fn frobenius_inner(&self, other: &Self) -> Result<f64, MatrixError> {
        self.check_same_shape(other)?;
        Ok(self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum())
    }

    /// Cosine similarity between flattened weight matrices (`0.0` if either norm is zero).
    pub fn cosine_similarity(&self, other: &Self) -> Result<f64, MatrixError> {
        let dot = self.frobenius_inner(other)?;
        let denom = self.frobenius_norm() * other.frobenius_norm();
        if denom <= 1e-30 {
            Ok(0.0)
        } else {
            Ok((dot / denom).clamp(-1.0, 1.0))
        }
    }

    fn check_same_shape(&self, other: &Self) -> Result<(), MatrixError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::ShapeMismatch {
                expected_rows: self.rows,
                expected_cols: self.cols,
                actual_rows: other.rows,
                actual_cols: other.cols,
            });
        }
        Ok(())
    }

    /// Build a rank-one cross approximation using the largest-magnitude pivot.
    ///
    /// For a pivot at `(i, j)`, this returns `left[r] = W[r,j]` and
    /// `right[c] = W[i,c] / W[i,j]`. It exactly reconstructs rank-one matrices
    /// when the selected pivot is non-zero, but is not guaranteed to minimize
    /// Frobenius error for arbitrary matrices.
    pub fn rank_one_pivot_approximation(&self) -> RankOneFactors {
        let Some((pivot_index, pivot_value)) = self
            .data
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
        else {
            unreachable!("validated matrices are non-empty")
        };

        if pivot_value == 0.0 {
            return RankOneFactors {
                left: vec![0.0; self.rows],
                right: vec![0.0; self.cols],
            };
        }

        let pivot_row = pivot_index / self.cols;
        let pivot_col = pivot_index % self.cols;
        let left = (0..self.rows)
            .map(|row| self.data[row * self.cols + pivot_col])
            .collect();
        let right = (0..self.cols)
            .map(|col| self.data[pivot_row * self.cols + col] / pivot_value)
            .collect();
        RankOneFactors { left, right }
    }

    /// Compute Frobenius reconstruction error without materializing the approximation.
    pub fn approximation_error(&self, factors: &RankOneFactors) -> Result<f64, MatrixError> {
        if self.rows != factors.rows() || self.cols != factors.cols() {
            return Err(MatrixError::ShapeMismatch {
                expected_rows: self.rows,
                expected_cols: self.cols,
                actual_rows: factors.rows(),
                actual_cols: factors.cols(),
            });
        }
        let squared_error = self
            .data
            .iter()
            .enumerate()
            .map(|(index, original)| {
                let row = index / self.cols;
                let col = index % self.cols;
                let difference = original - factors.left[row] * factors.right[col];
                difference * difference
            })
            .sum::<f64>();
        Ok(squared_error.sqrt())
    }
}

/// Rank-one matrix representation `left * rightᵀ`.
#[derive(Debug, Clone, PartialEq)]
pub struct RankOneFactors {
    left: Vec<f64>,
    right: Vec<f64>,
}

impl RankOneFactors {
    pub fn new(left: Vec<f64>, right: Vec<f64>) -> Result<Self, MatrixError> {
        if left.is_empty() || right.is_empty() {
            return Err(MatrixError::Empty);
        }
        if let Some(index) = left
            .iter()
            .chain(&right)
            .position(|value| !value.is_finite())
        {
            return Err(MatrixError::NonFiniteValue { index });
        }
        Ok(Self { left, right })
    }

    pub fn rows(&self) -> usize {
        self.left.len()
    }

    pub fn cols(&self) -> usize {
        self.right.len()
    }

    pub fn left(&self) -> &[f64] {
        &self.left
    }

    pub fn right(&self) -> &[f64] {
        &self.right
    }

    pub fn parameter_count(&self) -> usize {
        self.left.len() + self.right.len()
    }

    pub fn parameter_bytes_f64(&self) -> usize {
        self.parameter_count() * std::mem::size_of::<f64>()
    }

    /// Compute `(left * rightᵀ) * vector` without constructing the dense matrix.
    pub fn matvec(&self, vector: &[f64]) -> Result<Vec<f64>, MatrixError> {
        if vector.len() != self.cols() {
            return Err(MatrixError::DimensionMismatch {
                expected: self.cols(),
                actual: vector.len(),
            });
        }
        let projection = self
            .right
            .iter()
            .zip(vector)
            .map(|(weight, value)| weight * value)
            .sum::<f64>();
        Ok(self.left.iter().map(|value| value * projection).collect())
    }

    pub fn reconstruct(&self) -> Result<DenseMatrix, MatrixError> {
        let data = self
            .left
            .iter()
            .flat_map(|left| self.right.iter().map(move |right| left * right))
            .collect();
        DenseMatrix::from_flat(self.rows(), self.cols(), data)
    }
}

#[cfg(test)]
mod tests {
    use super::{DenseMatrix, MatrixError, RankOneFactors};

    #[test]
    fn constructs_matrix_and_multiplies_vector() {
        let matrix = DenseMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(matrix.matvec(&[2.0, 1.0]).unwrap(), vec![4.0, 10.0]);
    }

    #[test]
    fn rejects_empty_ragged_mismatched_and_non_finite_input() {
        assert_eq!(DenseMatrix::from_rows(&[]), Err(MatrixError::Empty));
        assert_eq!(DenseMatrix::from_rows(&[&[]]), Err(MatrixError::Empty));
        assert_eq!(
            DenseMatrix::from_rows(&[&[1.0], &[2.0, 3.0]]),
            Err(MatrixError::RaggedRows)
        );
        assert_eq!(
            DenseMatrix::from_flat(2, 2, vec![1.0, 2.0, 3.0]),
            Err(MatrixError::ElementCountMismatch {
                expected: 4,
                actual: 3
            })
        );
        assert_eq!(
            DenseMatrix::from_flat(1, 2, vec![1.0, f64::NAN]),
            Err(MatrixError::NonFiniteValue { index: 1 })
        );
    }

    #[test]
    fn rejects_wrong_vector_length_and_checks_bounds() {
        let matrix = DenseMatrix::from_rows(&[&[1.0, 2.0]]).unwrap();
        assert_eq!(
            matrix.matvec(&[1.0]),
            Err(MatrixError::DimensionMismatch {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(matrix.get(1, 0), None);
        assert_eq!(matrix.get(0, 2), None);
    }

    #[test]
    fn computes_frobenius_norm() {
        let matrix = DenseMatrix::from_rows(&[&[3.0, 4.0]]).unwrap();
        assert_eq!(matrix.frobenius_norm(), 5.0);
    }

    #[test]
    fn pivot_cross_factorization_exactly_recovers_a_rank_one_matrix() {
        let matrix = DenseMatrix::from_rows(&[&[2.0, 4.0, 6.0], &[3.0, 6.0, 9.0]]).unwrap();
        let factors = matrix.rank_one_pivot_approximation();
        assert!(matrix.approximation_error(&factors).unwrap() < 1e-12);
        assert_eq!(factors.parameter_count(), matrix.rows() + matrix.cols());
    }

    #[test]
    fn factorized_matvec_matches_dense_matvec_without_reconstruction() {
        let matrix = DenseMatrix::from_rows(&[&[2.0, 4.0, 6.0], &[3.0, 6.0, 9.0]]).unwrap();
        let factors = matrix.rank_one_pivot_approximation();
        let expected = matrix.matvec(&[0.5, -1.0, 2.0]).unwrap();
        let actual = factors.matvec(&[0.5, -1.0, 2.0]).unwrap();
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn zero_matrix_and_non_rank_one_matrix_have_defined_error() {
        let zero = DenseMatrix::from_rows(&[&[0.0, 0.0], &[0.0, 0.0]]).unwrap();
        assert_eq!(
            zero.approximation_error(&zero.rank_one_pivot_approximation())
                .unwrap(),
            0.0
        );

        let identity = DenseMatrix::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]).unwrap();
        assert!(
            (identity
                .approximation_error(&identity.rank_one_pivot_approximation())
                .unwrap()
                - 1.0)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn factor_constructor_rejects_invalid_values() {
        assert_eq!(
            RankOneFactors::new(vec![], vec![1.0]),
            Err(MatrixError::Empty)
        );
        assert_eq!(
            RankOneFactors::new(vec![1.0, f64::INFINITY], vec![1.0]),
            Err(MatrixError::NonFiniteValue { index: 1 })
        );
    }
}
