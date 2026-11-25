//! Deterministic singular-value decomposition and spectral structure metrics.
//!
//! Provides:
//! - Exact deterministic Jacobi eigendecomposition on the smaller Gram matrix
//!   (`WᵀW` or `WWᵀ`) for small-to-medium matrices (`min(m, n) <= 128`), and
//! - Deterministic randomized range-finder + subspace-iteration SVD
//!   (Halko–Martinsson–Tropp, 2011) for larger matrices or truncated targets.
//!
//! All operations are 100% safe Rust with deterministic seeding.

use crate::rng::DeterministicRng;
use crate::{DenseMatrix, MatrixError};

/// Truncated or thin singular-value decomposition `W ≈ U * diag(singular_values) * Vᵀ`.
#[derive(Debug, Clone, PartialEq)]
pub struct SvdResult {
    /// Left singular vectors (`rows × rank`), orthonormal columns.
    pub u: DenseMatrix,
    /// Singular values sorted in non-increasing order (`length == rank`, all `>= 0.0`).
    pub singular_values: Vec<f64>,
    /// Right singular vectors transposed (`rank × cols`), orthonormal rows.
    pub vt: DenseMatrix,
    /// Exact Frobenius norm of the original matrix before truncation.
    pub original_frobenius_norm: f64,
}

/// Summary metrics derived from a matrix's singular-value spectrum.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectrumMetrics {
    /// Computed top singular values in descending order.
    pub singular_values: Vec<f64>,
    /// Largest singular value (`σ_1`), or `0.0` for the zero matrix.
    pub spectral_norm: f64,
    /// Exact Frobenius norm `||W||_F`.
    pub frobenius_norm: f64,
    /// Stable rank `||W||_F² / σ_1²` (in `[1, min(m, n)]` for non-zero matrices, `0.0` for zero).
    pub stable_rank: f64,
    /// Roy–Vetterli (2007) entropy effective rank `exp(-Σ p_i ln p_i)`, `p_i = σ_i / Σ σ_j`.
    pub effective_rank: f64,
    /// Estimated condition number `σ_1 / σ_min` when full spectrum was computed and `σ_min > 1e-15`.
    pub condition_number_est: Option<f64>,
    /// Minimum rank capturing >= 90% of `||W||_F²` (capped by computed singular values).
    pub rank_90_energy: usize,
    /// Minimum rank capturing >= 95% of `||W||_F²`.
    pub rank_95_energy: usize,
    /// Minimum rank capturing >= 99% of `||W||_F²`.
    pub rank_99_energy: usize,
    /// Relative Frobenius error of best rank-1 approximation `sqrt(max(0, ||W||_F² - σ_1²)) / ||W||_F`.
    pub rank1_relative_error: f64,
    /// Fraction of total Frobenius energy `||W||_F²` captured by the computed singular values.
    pub captured_energy_ratio: f64,
    /// Whether the full spectrum (`min(m, n)` singular values) was computed.
    pub is_full_spectrum: bool,
}

impl SvdResult {
    pub fn rows(&self) -> usize {
        self.u.rows()
    }

    pub fn cols(&self) -> usize {
        self.vt.cols()
    }

    pub fn rank(&self) -> usize {
        self.singular_values.len()
    }

    /// Direct factorized matrix-vector product `U * (diag(Σ) * (Vᵀ * x))` without dense reconstruction.
    pub fn matvec(&self, vector: &[f64]) -> Result<Vec<f64>, MatrixError> {
        if vector.len() != self.cols() {
            return Err(MatrixError::DimensionMismatch {
                expected: self.cols(),
                actual: vector.len(),
            });
        }
        let mut projected = self.vt.matvec(vector)?;
        for (value, sigma) in projected.iter_mut().zip(&self.singular_values) {
            *value *= *sigma;
        }
        self.u.matvec(&projected)
    }

    /// Reconstruct the dense approximation `U * diag(Σ) * Vᵀ`.
    pub fn reconstruct(&self) -> Result<DenseMatrix, MatrixError> {
        let cols = self.cols();
        let rank = self.rank();
        let mut scaled_vt = vec![0.0; rank * cols];
        for r in 0..rank {
            let sigma = self.singular_values[r];
            let src_start = r * cols;
            for c in 0..cols {
                scaled_vt[src_start + c] = sigma * self.vt.as_slice()[src_start + c];
            }
        }
        let scaled = DenseMatrix::from_flat(rank, cols, scaled_vt)?;
        self.u.matmul(&scaled)
    }

    /// Truncate to a smaller rank `1 <= new_rank <= self.rank()`.
    pub fn truncate(&self, new_rank: usize) -> Result<Self, MatrixError> {
        if new_rank == 0 || new_rank > self.rank() {
            return Err(MatrixError::InvalidRank {
                requested: new_rank,
                max_rank: self.rank(),
            });
        }
        let rows = self.rows();
        let cols = self.cols();
        let old_rank = self.rank();
        let mut u_data = Vec::with_capacity(rows * new_rank);
        for r in 0..rows {
            let start = r * old_rank;
            u_data.extend_from_slice(&self.u.as_slice()[start..start + new_rank]);
        }
        let vt_data = self.vt.as_slice()[..new_rank * cols].to_vec();
        Ok(Self {
            u: DenseMatrix::from_flat(rows, new_rank, u_data)?,
            singular_values: self.singular_values[..new_rank].to_vec(),
            vt: DenseMatrix::from_flat(new_rank, cols, vt_data)?,
            original_frobenius_norm: self.original_frobenius_norm,
        })
    }

    /// Compute spectral summary metrics from this decomposition.
    pub fn spectrum_metrics(&self) -> SpectrumMetrics {
        let full_dim = self.rows().min(self.cols());
        let is_full_spectrum = self.rank() >= full_dim;
        spectrum_metrics_from_values(
            &self.singular_values,
            self.original_frobenius_norm,
            is_full_spectrum,
        )
    }

    /// Compute Grassmannian left-subspace affinity `||U_aᵀ U_b||_F² / min(k_a, k_b)` in `[0, 1]`.
    pub fn left_subspace_affinity(&self, other: &Self) -> Result<f64, MatrixError> {
        subspace_affinity_columns(&self.u, &other.u)
    }

    /// Compute Grassmannian right-subspace affinity `||V_aᵀ V_b||_F² / min(k_a, k_b)` in `[0, 1]`.
    pub fn right_subspace_affinity(&self, other: &Self) -> Result<f64, MatrixError> {
        if self.cols() != other.cols() {
            return Err(MatrixError::DimensionMismatch {
                expected: self.cols(),
                actual: other.cols(),
            });
        }
        let k_a = self.rank();
        let k_b = other.rank();
        let cols = self.cols();
        let mut sum_sq = 0.0_f64;
        for i in 0..k_a {
            let row_a = &self.vt.as_slice()[i * cols..(i + 1) * cols];
            for j in 0..k_b {
                let row_b = &other.vt.as_slice()[j * cols..(j + 1) * cols];
                let dot: f64 = row_a.iter().zip(row_b).map(|(a, b)| a * b).sum();
                sum_sq += dot * dot;
            }
        }
        let denom = k_a.min(k_b) as f64;
        Ok((sum_sq / denom).clamp(0.0, 1.0))
    }
}

/// Compute Grassmannian subspace affinity between orthonormal column spaces `U_a` and `U_b`.
pub fn subspace_affinity_columns(u_a: &DenseMatrix, u_b: &DenseMatrix) -> Result<f64, MatrixError> {
    if u_a.rows() != u_b.rows() {
        return Err(MatrixError::DimensionMismatch {
            expected: u_a.rows(),
            actual: u_b.rows(),
        });
    }
    let rows = u_a.rows();
    let k_a = u_a.cols();
    let k_b = u_b.cols();
    let mut sum_sq = 0.0_f64;
    for i in 0..k_a {
        for j in 0..k_b {
            let mut dot = 0.0_f64;
            for r in 0..rows {
                dot += u_a.as_slice()[r * k_a + i] * u_b.as_slice()[r * k_b + j];
            }
            sum_sq += dot * dot;
        }
    }
    let denom = k_a.min(k_b) as f64;
    Ok((sum_sq / denom).clamp(0.0, 1.0))
}

/// Compute spectral summary metrics from descending singular values and exact Frobenius norm.
pub fn spectrum_metrics_from_values(
    singular_values: &[f64],
    frobenius_norm: f64,
    is_full_spectrum: bool,
) -> SpectrumMetrics {
    let spectral_norm = singular_values.first().copied().unwrap_or(0.0);
    let frob_sq = frobenius_norm * frobenius_norm;
    let stable_rank = if spectral_norm > 1e-15 {
        (frob_sq / (spectral_norm * spectral_norm)).max(1.0)
    } else {
        0.0
    };

    let sum_sigma: f64 = singular_values.iter().copied().filter(|&s| s > 0.0).sum();
    let effective_rank = if sum_sigma > 1e-15 {
        let entropy: f64 = singular_values
            .iter()
            .copied()
            .filter(|&s| s > 0.0)
            .map(|s| {
                let p = s / sum_sigma;
                if p > 0.0 { -p * p.ln() } else { 0.0 }
            })
            .sum();
        entropy.exp()
    } else {
        0.0
    };

    let condition_number_est = if is_full_spectrum && !singular_values.is_empty() {
        let min_s = singular_values.last().copied().unwrap_or(0.0);
        if min_s > 1e-15 {
            Some(spectral_norm / min_s)
        } else {
            None
        }
    } else {
        None
    };

    let mut cumulative = 0.0_f64;
    let mut rank_90 = singular_values.len();
    let mut rank_95 = singular_values.len();
    let mut rank_99 = singular_values.len();
    let mut found_90 = false;
    let mut found_95 = false;
    let mut found_99 = false;
    if frob_sq <= 1e-30 {
        rank_90 = 0;
        rank_95 = 0;
        rank_99 = 0;
    } else {
        for (idx, &s) in singular_values.iter().enumerate() {
            cumulative += s * s;
            let ratio = cumulative / frob_sq;
            if !found_90 && ratio >= 0.90 {
                rank_90 = idx + 1;
                found_90 = true;
            }
            if !found_95 && ratio >= 0.95 {
                rank_95 = idx + 1;
                found_95 = true;
            }
            if !found_99 && ratio >= 0.99 {
                rank_99 = idx + 1;
                found_99 = true;
            }
        }
    }

    let captured_energy_ratio = if frob_sq > 1e-30 {
        (cumulative / frob_sq).clamp(0.0, 1.0)
    } else {
        1.0
    };

    let rank1_relative_error = if frobenius_norm > 1e-15 {
        let residual_sq = (frob_sq - spectral_norm * spectral_norm).max(0.0);
        residual_sq.sqrt() / frobenius_norm
    } else {
        0.0
    };

    SpectrumMetrics {
        singular_values: singular_values.to_vec(),
        spectral_norm,
        frobenius_norm,
        stable_rank,
        effective_rank,
        condition_number_est,
        rank_90_energy: rank_90,
        rank_95_energy: rank_95,
        rank_99_energy: rank_99,
        rank1_relative_error,
        captured_energy_ratio,
        is_full_spectrum,
    }
}

/// Compute truncated SVD of `matrix` up to `target_rank` singular components.
///
/// Uses exact Gram-Jacobi SVD when `min(rows, cols) <= 96`, and deterministic
/// randomized subspace-iteration SVD (Halko–Martinsson–Tropp) for larger matrices.
pub fn truncated_svd(
    matrix: &DenseMatrix,
    target_rank: usize,
    seed: u64,
) -> Result<SvdResult, MatrixError> {
    let min_dim = matrix.rows().min(matrix.cols());
    if target_rank == 0 || target_rank > min_dim {
        return Err(MatrixError::InvalidRank {
            requested: target_rank,
            max_rank: min_dim,
        });
    }
    if min_dim <= 96 {
        let full = exact_thin_svd(matrix)?;
        if target_rank < full.rank() {
            full.truncate(target_rank)
        } else {
            Ok(full)
        }
    } else {
        randomized_svd(matrix, target_rank, 8, 2, seed)
    }
}

/// Compute exact thin SVD via one-sided Jacobi orthogonalization directly on `W`
/// (avoids squaring the condition number in normal equations `WᵀW`).
pub fn exact_thin_svd(matrix: &DenseMatrix) -> Result<SvdResult, MatrixError> {
    let m = matrix.rows();
    let n = matrix.cols();
    let frob = matrix.frobenius_norm();

    if m >= n {
        // Store columns of A (m × n) and V (n × n) in column-major layout for cache-friendly rotations.
        let mut a_cols: Vec<Vec<f64>> = (0..n)
            .map(|c| (0..m).map(|r| matrix.as_slice()[r * n + c]).collect())
            .collect();
        let mut v_cols: Vec<Vec<f64>> = (0..n)
            .map(|c| {
                let mut col = vec![0.0_f64; n];
                col[c] = 1.0;
                col
            })
            .collect();

        let max_sweeps = 64;
        for _sweep in 0..max_sweeps {
            let mut converged = true;
            for p in 0..n {
                let (left_a, right_a) = a_cols.split_at_mut(p + 1);
                let (left_v, right_v) = v_cols.split_at_mut(p + 1);
                let col_ap = &mut left_a[p];
                let col_vp = &mut left_v[p];
                for (col_aq, col_vq) in right_a.iter_mut().zip(right_v.iter_mut()) {
                    let mut a_pp = 0.0_f64;
                    let mut a_qq = 0.0_f64;
                    let mut a_pq = 0.0_f64;
                    for (&ap, &aq) in col_ap.iter().zip(col_aq.iter()) {
                        a_pp += ap * ap;
                        a_qq += aq * aq;
                        a_pq += ap * aq;
                    }
                    let denom = (a_pp * a_qq).sqrt();
                    if denom <= 1e-30 || a_pq.abs() <= 1e-14 * denom {
                        continue;
                    }
                    converged = false;
                    let tau = (a_qq - a_pp) / (2.0 * a_pq);
                    let t = if tau >= 0.0 {
                        1.0 / (tau + (1.0 + tau * tau).sqrt())
                    } else {
                        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                    };
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;

                    for (ap, aq) in col_ap.iter_mut().zip(col_aq.iter_mut()) {
                        let old_p = *ap;
                        let old_q = *aq;
                        *ap = c * old_p - s * old_q;
                        *aq = s * old_p + c * old_q;
                    }
                    for (vp, vq) in col_vp.iter_mut().zip(col_vq.iter_mut()) {
                        let old_p = *vp;
                        let old_q = *vq;
                        *vp = c * old_p - s * old_q;
                        *vq = s * old_p + c * old_q;
                    }
                }
            }
            if converged {
                break;
            }
        }

        let norms: Vec<f64> = a_cols
            .iter()
            .map(|col| col.iter().map(|&x| x * x).sum::<f64>().sqrt())
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| norms[b].total_cmp(&norms[a]));

        let mut singular_values = Vec::with_capacity(n);
        let mut vt_data = vec![0.0_f64; n * n];
        let mut u_data = vec![0.0_f64; m * n];

        for (k, &col_idx) in order.iter().enumerate() {
            let sigma = norms[col_idx];
            singular_values.push(sigma);
            for col in 0..n {
                vt_data[k * n + col] = v_cols[col_idx][col];
            }
            if sigma > 1e-14 * frob.max(1.0) {
                let inv_sigma = 1.0 / sigma;
                for r in 0..m {
                    u_data[r * n + k] = a_cols[col_idx][r] * inv_sigma;
                }
            }
        }
        orthonormalize_columns(&mut u_data, m, n);

        Ok(SvdResult {
            u: DenseMatrix::from_flat(m, n, u_data)?,
            singular_values,
            vt: DenseMatrix::from_flat(n, n, vt_data)?,
            original_frobenius_norm: frob,
        })
    } else {
        // Compute thin SVD on Wᵀ and transpose results: W = V Σ Uᵀ.
        let transposed = matrix.transpose();
        let svd_t = exact_thin_svd(&transposed)?;
        let u = svd_t.vt.transpose();
        let vt = svd_t.u.transpose();
        Ok(SvdResult {
            u,
            singular_values: svd_t.singular_values,
            vt,
            original_frobenius_norm: frob,
        })
    }
}

/// Deterministic randomized SVD (Halko–Martinsson–Tropp 2011) with subspace power iterations.
pub fn randomized_svd(
    matrix: &DenseMatrix,
    target_rank: usize,
    oversampling: usize,
    power_iterations: usize,
    seed: u64,
) -> Result<SvdResult, MatrixError> {
    let m = matrix.rows();
    let n = matrix.cols();
    let min_dim = m.min(n);
    if target_rank == 0 || target_rank > min_dim {
        return Err(MatrixError::InvalidRank {
            requested: target_rank,
            max_rank: min_dim,
        });
    }
    let sketch_rank = (target_rank + oversampling).min(min_dim);
    let mut rng = DeterministicRng::new(seed);

    // Random Gaussian test matrix Ω ∈ R^{n × sketch_rank}
    let mut omega_data = Vec::with_capacity(n * sketch_rank);
    for _ in 0..(n * sketch_rank) {
        omega_data.push(rng.next_normal());
    }
    let omega = DenseMatrix::from_flat(n, sketch_rank, omega_data)?;

    // Y = W * Ω (m × sketch_rank)
    let mut y = matrix.matmul(&omega)?;
    let mut q = orthonormalize_matrix_columns(&y)?;

    // Subspace power iterations: Q = orth(W * (Wᵀ * Q))
    for _ in 0..power_iterations {
        let mut z_data = vec![0.0_f64; n * sketch_rank];
        for k in 0..sketch_rank {
            let col_q: Vec<f64> = (0..m).map(|r| q.as_slice()[r * sketch_rank + k]).collect();
            let wt_q = matrix.matvec_transposed(&col_q)?;
            for c in 0..n {
                z_data[c * sketch_rank + k] = wt_q[c];
            }
        }
        let z = DenseMatrix::from_flat(n, sketch_rank, z_data)?;
        let q_z = orthonormalize_matrix_columns(&z)?;
        y = matrix.matmul(&q_z)?;
        q = orthonormalize_matrix_columns(&y)?;
    }

    // B = Qᵀ * W (sketch_rank × n)
    let qt = q.transpose();
    let b = qt.matmul(matrix)?;
    let b_svd = exact_thin_svd(&b)?;

    // U = Q * U_b
    let u_full = q.matmul(&b_svd.u)?;
    let svd_full = SvdResult {
        u: u_full,
        singular_values: b_svd.singular_values,
        vt: b_svd.vt,
        original_frobenius_norm: matrix.frobenius_norm(),
    };
    if target_rank < svd_full.rank() {
        svd_full.truncate(target_rank)
    } else {
        Ok(svd_full)
    }
}

/// Modified Gram-Schmidt with two passes and deterministic canonical basis completion for zero columns.
fn orthonormalize_columns(data: &mut [f64], rows: usize, cols: usize) {
    for k in 0..cols {
        // Two passes of modified Gram-Schmidt for numerical stability.
        for _pass in 0..2 {
            for j in 0..k {
                let mut dot = 0.0_f64;
                for r in 0..rows {
                    dot += data[r * cols + j] * data[r * cols + k];
                }
                for r in 0..rows {
                    data[r * cols + k] -= dot * data[r * cols + j];
                }
            }
        }
        let mut norm_sq = 0.0_f64;
        for r in 0..rows {
            let val = data[r * cols + k];
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for r in 0..rows {
                data[r * cols + k] *= inv;
            }
        } else {
            // Find a deterministic standard basis vector not in span(0..k).
            for basis in 0..rows {
                for r in 0..rows {
                    data[r * cols + k] = if r == basis { 1.0 } else { 0.0 };
                }
                for _pass in 0..2 {
                    for j in 0..k {
                        let mut dot = 0.0_f64;
                        for r in 0..rows {
                            dot += data[r * cols + j] * data[r * cols + k];
                        }
                        for r in 0..rows {
                            data[r * cols + k] -= dot * data[r * cols + j];
                        }
                    }
                }
                let mut candidate_norm_sq = 0.0_f64;
                for r in 0..rows {
                    let val = data[r * cols + k];
                    candidate_norm_sq += val * val;
                }
                let candidate_norm = candidate_norm_sq.sqrt();
                if candidate_norm > 1e-10 {
                    let inv = 1.0 / candidate_norm;
                    for r in 0..rows {
                        data[r * cols + k] *= inv;
                    }
                    break;
                }
            }
        }
    }
}

fn orthonormalize_matrix_columns(matrix: &DenseMatrix) -> Result<DenseMatrix, MatrixError> {
    let mut data = matrix.as_slice().to_vec();
    orthonormalize_columns(&mut data, matrix.rows(), matrix.cols());
    DenseMatrix::from_flat(matrix.rows(), matrix.cols(), data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_svd_recovers_rectangular_matrix_and_matvec() {
        let matrix = DenseMatrix::from_rows(&[
            &[3.0, 2.0, 2.0],
            &[2.0, 3.0, -2.0],
            &[1.0, -1.0, 4.0],
            &[-2.0, 5.0, 1.0],
        ])
        .unwrap();
        let svd = exact_thin_svd(&matrix).unwrap();
        let reconstructed = svd.reconstruct().unwrap();
        let diff = matrix.sub(&reconstructed).unwrap().frobenius_norm();
        assert!(diff < 1e-10, "reconstruction diff={diff}");

        let x = [0.7, -1.3, 2.1];
        let dense_y = matrix.matvec(&x).unwrap();
        let svd_y = svd.matvec(&x).unwrap();
        for (a, b) in dense_y.iter().zip(svd_y) {
            assert!((a - b).abs() < 1e-10);
        }

        // Verify singular values match Frobenius norm.
        let sv_frob = svd
            .singular_values
            .iter()
            .map(|s| s * s)
            .sum::<f64>()
            .sqrt();
        assert!((sv_frob - matrix.frobenius_norm()).abs() < 1e-10);
    }

    #[test]
    fn exact_svd_handles_wide_matrices() {
        let matrix =
            DenseMatrix::from_rows(&[&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]]).unwrap();
        let svd = exact_thin_svd(&matrix).unwrap();
        assert_eq!(svd.rank(), 2);
        let reconstructed = svd.reconstruct().unwrap();
        let diff = matrix.sub(&reconstructed).unwrap().frobenius_norm();
        assert!(diff < 1e-10, "wide reconstruction diff={diff}");
    }

    #[test]
    fn truncated_svd_matches_eckart_young_energy_and_metrics() {
        // Construct an exact rank-2 matrix of shape 12 × 10 with known singular values 10 and 2.
        let mut rng = DeterministicRng::new(99);
        let m = 12;
        let n = 10;
        let mut a_raw: Vec<f64> = (0..m * 2).map(|_| rng.next_normal()).collect();
        let mut b_raw: Vec<f64> = (0..n * 2).map(|_| rng.next_normal()).collect();
        orthonormalize_columns(&mut a_raw, m, 2);
        orthonormalize_columns(&mut b_raw, n, 2);

        let mut data = vec![0.0_f64; m * n];
        for r in 0..m {
            for c in 0..n {
                data[r * n + c] =
                    10.0 * a_raw[r * 2] * b_raw[c * 2] + 2.0 * a_raw[r * 2 + 1] * b_raw[c * 2 + 1];
            }
        }
        let matrix = DenseMatrix::from_flat(m, n, data).unwrap();
        let svd = exact_thin_svd(&matrix).unwrap();
        assert!((svd.singular_values[0] - 10.0).abs() < 1e-9);
        assert!((svd.singular_values[1] - 2.0).abs() < 1e-9);
        assert!(svd.singular_values[2] < 1e-9);

        let metrics = svd.spectrum_metrics();
        // stable_rank = (100 + 4) / 100 = 1.04
        assert!((metrics.stable_rank - 1.04).abs() < 1e-9);
        assert_eq!(metrics.rank_90_energy, 1); // 100/104 = 96.15%
        assert_eq!(metrics.rank_95_energy, 1);
        assert_eq!(metrics.rank_99_energy, 2);
        assert!(metrics.effective_rank > 1.4 && metrics.effective_rank < 1.8);
    }

    #[test]
    fn randomized_svd_recovers_low_rank_large_matrix_deterministically() {
        let m = 140;
        let n = 120;
        let r = 4;
        let mut rng = DeterministicRng::new(2026);
        let u_true =
            DenseMatrix::from_flat(m, r, (0..m * r).map(|_| rng.next_normal() * 0.5).collect())
                .unwrap();
        let v_true =
            DenseMatrix::from_flat(r, n, (0..r * n).map(|_| rng.next_normal() * 0.5).collect())
                .unwrap();
        let matrix = u_true.matmul(&v_true).unwrap();

        let svd1 = truncated_svd(&matrix, 4, 7).unwrap();
        let svd2 = truncated_svd(&matrix, 4, 7).unwrap();
        assert_eq!(svd1, svd2);
        let reconstructed = svd1.reconstruct().unwrap();
        let rel_err =
            matrix.sub(&reconstructed).unwrap().frobenius_norm() / matrix.frobenius_norm();
        assert!(rel_err < 1e-8, "randomized svd rel_err={rel_err}");
    }

    #[test]
    fn subspace_affinity_is_one_for_identical_and_zero_for_orthogonal() {
        let m1 = DenseMatrix::from_rows(&[
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
        ])
        .unwrap();
        let m2 = DenseMatrix::from_rows(&[
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
        ])
        .unwrap();
        let s1 = truncated_svd(&m1, 2, 1).unwrap();
        let s2 = truncated_svd(&m2, 2, 1).unwrap();
        assert!((s1.left_subspace_affinity(&s1).unwrap() - 1.0).abs() < 1e-10);
        assert!(s1.left_subspace_affinity(&s2).unwrap() < 1e-10);
    }
}
