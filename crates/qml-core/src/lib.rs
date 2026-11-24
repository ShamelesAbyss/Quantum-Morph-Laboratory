//! Small, safe mathematical building blocks for Quantum Morph Laboratory.
//!
//! These primitives are conventional foundations, not novel compression methods.

use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixError {
    Empty,
    RaggedRows,
    DimensionMismatch { expected: usize, actual: usize },
}

impl Display for MatrixError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "matrix must have at least one row and one column"),
            Self::RaggedRows => write!(f, "all matrix rows must have the same length"),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "vector length mismatch: expected {expected}, received {actual}"
            ),
        }
    }
}

impl Error for MatrixError {}

/// A finite, row-major dense matrix stored in host `f64` values.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl DenseMatrix {
    /// Construct from row slices, rejecting empty and inconsistent dimensions.
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

        let mut data = Vec::with_capacity(rows.len() * cols);
        for row in rows {
            data.extend_from_slice(row);
        }
        Ok(Self {
            rows: rows.len(),
            cols,
            data,
        })
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
}

#[cfg(test)]
mod tests {
    use super::{DenseMatrix, MatrixError};

    #[test]
    fn constructs_matrix_and_multiplies_vector() {
        let matrix = DenseMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(matrix.matvec(&[2.0, 1.0]).unwrap(), vec![4.0, 10.0]);
    }

    #[test]
    fn rejects_empty_and_ragged_input() {
        assert_eq!(DenseMatrix::from_rows(&[]), Err(MatrixError::Empty));
        assert_eq!(DenseMatrix::from_rows(&[&[]]), Err(MatrixError::Empty));
        assert_eq!(
            DenseMatrix::from_rows(&[&[1.0], &[2.0, 3.0]]),
            Err(MatrixError::RaggedRows)
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
}
