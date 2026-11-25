//! RMQ — Relational Morph Quantization experimental engine.
//!
//! Investigates whether multiple neural network layers can share compressed
//! latent mathematical structures rather than being quantized independently.
//!
//! Implements and compares two explicit formulations:
//! 1. **RMQ Shared-Basis (`W_l ≈ U_shared * C_l * V_sharedᵀ + R_l`)**:
//!    Shared orthonormal left/right subspace dictionaries `U_shared (m × p)` and
//!    `V_shared (n × q)` extracted across `L` clustered layers via deterministic
//!    2D-HOSVD / Higher-Order Orthogonal Iteration (HOOI), quantized once and
//!    amortized over all `L` layers; each layer stores only a compact quantized
//!    core `C_l (p × q)` plus an optional residual `R_l` (sparse top-k outliers
//!    or packed INT4 residual backbone).
//! 2. **RMQ Shared-Core (`W_l ≈ A_l * S * B_l + R_l`)**:
//!    Directly evaluates the directive's initial proposed equation with a shared
//!    spectral core `S (r × r)` across `L` layers, layer-specific quantized
//!    transforms `A_l (m × r)` and `B_l (r × n)`, plus an optional residual `R_l`.
//!
//! Both formulations execute `y = Ŵ_l x` directly through their compressed
//! representations without dense weight reconstruction, and count every byte
//! including amortized shared structures, cores, transforms, scales, sparse
//! indices, and headers.

use crate::baselines::{
    FactorPrecision, Int4SymGroupMatrix, Int8SymMatrix, LowRankSvdMatrix, QuantizedFactor,
    SparseResidual,
};
use crate::{
    EvaluationMetrics, LinearOperator, StorageBreakdown, deterministic_probe_vectors,
    evaluate_operator,
};
use qml_core::{DenseMatrix, MatrixError, truncated_svd};
use serde::{Deserialize, Serialize};
use std::time::Instant;

const RMQ_LAYER_HEADER_BYTES: usize = 24;
const RMQ_SHARED_HEADER_BYTES: usize = 24;

/// Configurable residual correction attached to an RMQ layer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RmqResidualKind {
    None,
    /// Sparse top-k outlier residual storing `fraction` (e.g., `0.01` = 1%) of entries.
    SparseTopFraction {
        fraction: f64,
    },
    /// Quantized dense INT4 residual backbone (CALDERA-comparable hybrid).
    Int4Sym {
        group_size: usize,
    },
}

/// Runtime storage of a layer's residual component `R_l`.
#[derive(Debug, Clone, PartialEq)]
pub enum RmqResidual {
    None,
    Sparse(SparseResidual),
    Int4Sym(Int4SymGroupMatrix),
}

impl RmqResidual {
    pub fn fit(residual: &DenseMatrix, kind: RmqResidualKind) -> Result<Self, MatrixError> {
        Ok(match kind {
            RmqResidualKind::None => Self::None,
            RmqResidualKind::SparseTopFraction { fraction } => Self::Sparse(
                SparseResidual::from_residual_top_fraction(residual, fraction),
            ),
            RmqResidualKind::Int4Sym { group_size } => {
                Self::Int4Sym(Int4SymGroupMatrix::quantize(residual, group_size)?)
            }
        })
    }

    pub fn add_matvec_into(&self, x: &[f64], output: &mut [f64]) -> Result<(), MatrixError> {
        match self {
            Self::None => Ok(()),
            Self::Sparse(sp) => sp.add_matvec_into(x, output),
            Self::Int4Sym(q4) => {
                let r_y = q4.matvec(x)?;
                for (o, r) in output.iter_mut().zip(r_y) {
                    *o += r;
                }
                Ok(())
            }
        }
    }

    pub fn add_dense_into(&self, base: &DenseMatrix) -> Result<DenseMatrix, MatrixError> {
        match self {
            Self::None => Ok(base.clone()),
            Self::Sparse(sp) => {
                let mut data = base.as_slice().to_vec();
                sp.add_dense_into(&mut data);
                DenseMatrix::from_flat(base.rows(), base.cols(), data)
            }
            Self::Int4Sym(q4) => {
                let r = q4.reconstruct_dense()?;
                base.add(&r)
            }
        }
    }

    /// Returns `(payload_bytes, scale_bytes, structure_bytes)`.
    pub fn storage_bytes(&self) -> (usize, usize, usize) {
        match self {
            Self::None => (0, 0, 0),
            Self::Sparse(sp) => (0, 0, sp.entry_bytes()),
            Self::Int4Sym(q4) => {
                let sb = q4.storage_breakdown();
                (sb.payload_bytes, sb.scale_metadata_bytes, 0)
            }
        }
    }
}

/// Shared-basis RMQ cluster: `W_l ≈ U_shared * C_l * V_sharedᵀ + R_l` for `l = 0..L`.
#[derive(Debug, Clone, PartialEq)]
pub struct RmqSharedBasisCluster {
    pub rows: usize,
    pub cols: usize,
    pub left_rank: usize,
    pub right_rank: usize,
    pub u_shared: QuantizedFactor,   // m × p
    pub vt_shared: QuantizedFactor,  // q × n
    pub cores: Vec<QuantizedFactor>, // each p × q
    pub residuals: Vec<RmqResidual>,
}

/// Borrowed view of one layer inside an [`RmqSharedBasisCluster`] implementing [`LinearOperator`].
pub struct RmqSharedBasisLayer<'a> {
    cluster: &'a RmqSharedBasisCluster,
    layer_index: usize,
}

/// Configuration for fitting a shared-basis RMQ cluster.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmqSharedBasisConfig {
    pub left_rank: usize,
    pub right_rank: usize,
    pub hooi_iters: usize,
    pub basis_precision: FactorPrecision,
    pub core_precision: FactorPrecision,
    pub residual_kind: RmqResidualKind,
    pub seed: u64,
}

impl RmqSharedBasisCluster {
    /// Fit a shared-basis RMQ representation across `layers` via deterministic 2D-HOOI.
    ///
    /// Every matrix in `layers` must share the same shape `(m, n)`.
    pub fn fit(layers: &[&DenseMatrix], config: RmqSharedBasisConfig) -> Result<Self, MatrixError> {
        let Some(first) = layers.first() else {
            return Err(MatrixError::Empty);
        };
        let m = first.rows();
        let n = first.cols();
        for mat in layers.iter().skip(1) {
            if mat.rows() != m || mat.cols() != n {
                return Err(MatrixError::ShapeMismatch {
                    expected_rows: m,
                    expected_cols: n,
                    actual_rows: mat.rows(),
                    actual_cols: mat.cols(),
                });
            }
        }
        let p = config.left_rank.max(1).min(m);
        let q = config.right_rank.max(1).min(n);

        // Initialize Vᵀ via stacked vertical matrix [W_0; W_1; ...; W_{L-1}] -> top-q right singular vectors.
        let mut stacked_v_data = Vec::with_capacity(layers.len() * m * n);
        for mat in layers {
            stacked_v_data.extend_from_slice(mat.as_slice());
        }
        let stacked_v = DenseMatrix::from_flat(layers.len() * m, n, stacked_v_data)?;
        let mut vt = truncated_svd(&stacked_v, q, config.seed)?.vt; // q × n

        // Initialize U via horizontally concatenated [W_0 V | W_1 V | ... | W_{L-1} V] (m × L*q).
        let mut u = compute_shared_u(layers, &vt, p, config.seed.wrapping_add(1))?;

        // Higher-Order Orthogonal Iteration refinement.
        for iter in 0..config.hooi_iters {
            vt = compute_shared_vt(layers, &u, q, seed_offset(config.seed, 10 + iter as u64))?;
            u = compute_shared_u(layers, &vt, p, seed_offset(config.seed, 100 + iter as u64))?;
        }

        // Robust Outlier-Peeling Refinement (RPCA-inspired): when a sparse residual is enabled,
        // peel top-k residual outliers (plus initial extreme magnitude outliers) and refit the
        // shared subspaces on `(W_l - R_l)` so isolated weight outliers cannot steal shared
        // subspace dimensions from the layer cluster.
        if let RmqResidualKind::SparseTopFraction { fraction } = config.residual_kind
            && fraction > 0.0
        {
            let mut peeled_storage: Vec<DenseMatrix> = Vec::with_capacity(layers.len());
            for mat in layers {
                // First peel obvious heavy-tailed outliers directly from W_l so they cannot dominate SVD.
                let sp = SparseResidual::from_residual_top_fraction(mat, fraction);
                let mut clean_data = mat.as_slice().to_vec();
                for (&idx, &val_bits) in sp.indices().iter().zip(sp.values_f16()) {
                    clean_data[idx as usize] -= f64::from(qml_model::f16_to_f32(val_bits));
                }
                peeled_storage.push(DenseMatrix::from_flat(m, n, clean_data)?);
            }
            let peeled_refs: Vec<&DenseMatrix> = peeled_storage.iter().collect();
            let mut stacked_p = Vec::with_capacity(layers.len() * m * n);
            for mat in &peeled_refs {
                stacked_p.extend_from_slice(mat.as_slice());
            }
            let stacked_mat = DenseMatrix::from_flat(layers.len() * m, n, stacked_p)?;
            vt = truncated_svd(&stacked_mat, q, seed_offset(config.seed, 150))?.vt;
            u = compute_shared_u(&peeled_refs, &vt, p, seed_offset(config.seed, 160))?;
            for iter in 0..config.hooi_iters.max(1) {
                vt = compute_shared_vt(
                    &peeled_refs,
                    &u,
                    q,
                    seed_offset(config.seed, 200 + iter as u64),
                )?;
                u = compute_shared_u(
                    &peeled_refs,
                    &vt,
                    p,
                    seed_offset(config.seed, 300 + iter as u64),
                )?;
            }
        }

        // Quantize shared bases once, then project cores against the *quantized* bases
        // so quantization error in U_shared / V_shared is compensated in C_l!
        let u_q = QuantizedFactor::encode(&u, config.basis_precision)?;
        let vt_q = QuantizedFactor::encode(&vt, config.basis_precision)?;
        let u_eff = u_q.reconstruct_dense()?;
        let vt_eff = vt_q.reconstruct_dense()?;
        let ut_eff = u_eff.transpose();
        let v_eff = vt_eff.transpose();

        let mut cores = Vec::with_capacity(layers.len());
        let mut residuals = Vec::with_capacity(layers.len());
        for mat in layers {
            // Core C_l = U_effᵀ * W_l * V_eff (p × q), refined after peeling sparse outliers if active.
            let mut core_dense = ut_eff.matmul(mat)?.matmul(&v_eff)?;
            if let RmqResidualKind::SparseTopFraction { fraction } = config.residual_kind
                && fraction > 0.0
            {
                let approx0 = u_eff.matmul(&core_dense)?.matmul(&vt_eff)?;
                let diff0 = mat.sub(&approx0)?;
                let sp0 = SparseResidual::from_residual_top_fraction(&diff0, fraction);
                let mut peeled = mat.as_slice().to_vec();
                for (&idx, &val_bits) in sp0.indices().iter().zip(sp0.values_f16()) {
                    peeled[idx as usize] -= f64::from(qml_model::f16_to_f32(val_bits));
                }
                let peeled_mat = DenseMatrix::from_flat(m, n, peeled)?;
                core_dense = ut_eff.matmul(&peeled_mat)?.matmul(&v_eff)?;
            }
            let core_q = QuantizedFactor::encode(&core_dense, config.core_precision)?;

            let residual = if matches!(config.residual_kind, RmqResidualKind::None) {
                RmqResidual::None
            } else {
                let core_eff = core_q.reconstruct_dense()?;
                let approx = u_eff.matmul(&core_eff)?.matmul(&vt_eff)?;
                let diff = mat.sub(&approx)?;
                RmqResidual::fit(&diff, config.residual_kind)?
            };

            cores.push(core_q);
            residuals.push(residual);
        }

        Ok(Self {
            rows: m,
            cols: n,
            left_rank: p,
            right_rank: q,
            u_shared: u_q,
            vt_shared: vt_q,
            cores,
            residuals,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.cores.len()
    }

    pub fn layer(&self, index: usize) -> Option<RmqSharedBasisLayer<'_>> {
        if index < self.cores.len() {
            Some(RmqSharedBasisLayer {
                cluster: self,
                layer_index: index,
            })
        } else {
            None
        }
    }

    /// Exact total bytes of the entire cluster (shared bases + all cores + all residuals + headers).
    pub fn total_cluster_bytes(&self) -> usize {
        let (up, us) = self.u_shared.raw_storage_bytes();
        let (vp, vs) = self.vt_shared.raw_storage_bytes();
        let mut total = RMQ_SHARED_HEADER_BYTES + up + us + vp + vs;
        for (core, res) in self.cores.iter().zip(&self.residuals) {
            let (cp, cs) = core.raw_storage_bytes();
            let (rp, rs, rst) = res.storage_bytes();
            total += RMQ_LAYER_HEADER_BYTES + cp + cs + rp + rs + rst;
        }
        total
    }
}

impl LinearOperator for RmqSharedBasisLayer<'_> {
    fn method_name(&self) -> &'static str {
        match &self.cluster.residuals[self.layer_index] {
            RmqResidual::None => "RMQ_SHARED_BASIS",
            RmqResidual::Sparse(_) => "RMQ_SHARED_BASIS+SPARSE",
            RmqResidual::Int4Sym(_) => "RMQ_SHARED_BASIS+INT4",
        }
    }

    fn rows(&self) -> usize {
        self.cluster.rows
    }

    fn cols(&self) -> usize {
        self.cluster.cols
    }

    /// Direct compressed matvec: `y = U_shared * (C_l * (V_sharedᵀ * x)) + R_l * x`.
    fn matvec(&self, vector: &[f64]) -> Result<Vec<f64>, MatrixError> {
        let vtx = self.cluster.vt_shared.matvec(vector)?;
        let c_vtx = self.cluster.cores[self.layer_index].matvec(&vtx)?;
        let mut y = self.cluster.u_shared.matvec(&c_vtx)?;
        self.cluster.residuals[self.layer_index].add_matvec_into(vector, &mut y)?;
        Ok(y)
    }

    fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError> {
        let u = self.cluster.u_shared.reconstruct_dense()?;
        let c = self.cluster.cores[self.layer_index].reconstruct_dense()?;
        let vt = self.cluster.vt_shared.reconstruct_dense()?;
        let base = u.matmul(&c)?.matmul(&vt)?;
        self.cluster.residuals[self.layer_index].add_dense_into(&base)
    }

    fn storage_breakdown(&self) -> StorageBreakdown {
        let n_layers = self.cluster.cores.len().max(1);
        let (up, us) = self.cluster.u_shared.raw_storage_bytes();
        let (vp, vs) = self.cluster.vt_shared.raw_storage_bytes();
        let shared_payload_share = (up + vp).div_ceil(n_layers);
        let shared_scale_share = (us + vs).div_ceil(n_layers);
        let shared_header_share = RMQ_SHARED_HEADER_BYTES.div_ceil(n_layers);

        let (cp, cs) = self.cluster.cores[self.layer_index].raw_storage_bytes();
        let (rp, rs, rst) = self.cluster.residuals[self.layer_index].storage_bytes();

        StorageBreakdown {
            payload_bytes: rp,
            scale_metadata_bytes: shared_scale_share + cs + rs,
            structure_bytes: shared_payload_share + cp + rst,
            header_bytes: RMQ_LAYER_HEADER_BYTES + shared_header_share,
        }
    }
}

/// Shared-core RMQ cluster directly implementing `W_l ≈ A_l * S * B_l + R_l`.
#[derive(Debug, Clone, PartialEq)]
pub struct RmqSharedCoreCluster {
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    pub shared_core: QuantizedFactor,           // r × r
    pub left_transforms: Vec<QuantizedFactor>,  // each m × r
    pub right_transforms: Vec<QuantizedFactor>, // each r × n
    pub residuals: Vec<RmqResidual>,
}

/// Borrowed view of one layer inside an [`RmqSharedCoreCluster`] implementing [`LinearOperator`].
pub struct RmqSharedCoreLayer<'a> {
    cluster: &'a RmqSharedCoreCluster,
    layer_index: usize,
}

impl RmqSharedCoreCluster {
    /// Fit `W_l ≈ A_l * S * B_l + R_l` with a shared spectral core `S ∈ R^{r × r}`.
    ///
    /// Computes per-layer truncated SVDs `W_l ≈ U_l Σ_l V_lᵀ`, defines the shared
    /// diagonal spectral core `S = diag(mean_l(Σ_l))`, absorbs per-layer singular-value
    /// ratios `Σ_l / S` symmetrically into `A_l` and `B_l`, quantizes `S`, `A_l`, `B_l`,
    /// and optionally fits residual `R_l`.
    pub fn fit(
        layers: &[&DenseMatrix],
        rank: usize,
        factor_precision: FactorPrecision,
        residual_kind: RmqResidualKind,
        seed: u64,
    ) -> Result<Self, MatrixError> {
        let Some(first) = layers.first() else {
            return Err(MatrixError::Empty);
        };
        let m = first.rows();
        let n = first.cols();
        for mat in layers.iter().skip(1) {
            if mat.rows() != m || mat.cols() != n {
                return Err(MatrixError::ShapeMismatch {
                    expected_rows: m,
                    expected_cols: n,
                    actual_rows: mat.rows(),
                    actual_cols: mat.cols(),
                });
            }
        }
        let r = rank.max(1).min(m.min(n));

        let mut svds = Vec::with_capacity(layers.len());
        let mut mean_sigma = vec![0.0_f64; r];
        for (idx, mat) in layers.iter().enumerate() {
            let svd = truncated_svd(mat, r, seed.wrapping_add(idx as u64))?;
            for (k, &s) in svd.singular_values.iter().enumerate() {
                mean_sigma[k] += s / layers.len() as f64;
            }
            svds.push(svd);
        }

        let mut s_data = vec![0.0_f64; r * r];
        for k in 0..r {
            s_data[k * r + k] = mean_sigma[k].max(1e-8);
        }
        let s_dense = DenseMatrix::from_flat(r, r, s_data)?;
        let shared_core = QuantizedFactor::encode(&s_dense, FactorPrecision::Fp16)?;
        let s_eff = shared_core.reconstruct_dense()?;

        let mut left_transforms = Vec::with_capacity(layers.len());
        let mut right_transforms = Vec::with_capacity(layers.len());
        let mut residuals = Vec::with_capacity(layers.len());

        for (mat, svd) in layers.iter().zip(svds) {
            let mut a_data = vec![0.0_f64; m * r];
            let mut b_data = vec![0.0_f64; r * n];
            for k in 0..r {
                let s_shared = s_eff.as_slice()[k * r + k].max(1e-8);
                let ratio_sqrt = (svd.singular_values[k] / s_shared).max(0.0).sqrt();
                for row in 0..m {
                    a_data[row * r + k] = svd.u.as_slice()[row * r + k] * ratio_sqrt;
                }
                for col in 0..n {
                    b_data[k * n + col] = svd.vt.as_slice()[k * n + col] * ratio_sqrt;
                }
            }
            let a_q =
                QuantizedFactor::encode(&DenseMatrix::from_flat(m, r, a_data)?, factor_precision)?;
            let b_q =
                QuantizedFactor::encode(&DenseMatrix::from_flat(r, n, b_data)?, factor_precision)?;

            let residual = if matches!(residual_kind, RmqResidualKind::None) {
                RmqResidual::None
            } else {
                let a_eff = a_q.reconstruct_dense()?;
                let b_eff = b_q.reconstruct_dense()?;
                let approx = a_eff.matmul(&s_eff)?.matmul(&b_eff)?;
                let diff = mat.sub(&approx)?;
                RmqResidual::fit(&diff, residual_kind)?
            };

            left_transforms.push(a_q);
            right_transforms.push(b_q);
            residuals.push(residual);
        }

        Ok(Self {
            rows: m,
            cols: n,
            rank: r,
            shared_core,
            left_transforms,
            right_transforms,
            residuals,
        })
    }

    pub fn layer(&self, index: usize) -> Option<RmqSharedCoreLayer<'_>> {
        if index < self.left_transforms.len() {
            Some(RmqSharedCoreLayer {
                cluster: self,
                layer_index: index,
            })
        } else {
            None
        }
    }
}

impl LinearOperator for RmqSharedCoreLayer<'_> {
    fn method_name(&self) -> &'static str {
        match &self.cluster.residuals[self.layer_index] {
            RmqResidual::None => "RMQ_SHARED_CORE",
            RmqResidual::Sparse(_) => "RMQ_SHARED_CORE+SPARSE",
            RmqResidual::Int4Sym(_) => "RMQ_SHARED_CORE+INT4",
        }
    }

    fn rows(&self) -> usize {
        self.cluster.rows
    }

    fn cols(&self) -> usize {
        self.cluster.cols
    }

    /// Direct compressed matvec: `y = A_l * (S * (B_l * x)) + R_l * x`.
    fn matvec(&self, vector: &[f64]) -> Result<Vec<f64>, MatrixError> {
        let bx = self.cluster.right_transforms[self.layer_index].matvec(vector)?;
        let s_bx = self.cluster.shared_core.matvec(&bx)?;
        let mut y = self.cluster.left_transforms[self.layer_index].matvec(&s_bx)?;
        self.cluster.residuals[self.layer_index].add_matvec_into(vector, &mut y)?;
        Ok(y)
    }

    fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError> {
        let a = self.cluster.left_transforms[self.layer_index].reconstruct_dense()?;
        let s = self.cluster.shared_core.reconstruct_dense()?;
        let b = self.cluster.right_transforms[self.layer_index].reconstruct_dense()?;
        let base = a.matmul(&s)?.matmul(&b)?;
        self.cluster.residuals[self.layer_index].add_dense_into(&base)
    }

    fn storage_breakdown(&self) -> StorageBreakdown {
        let n_layers = self.cluster.left_transforms.len().max(1);
        let (sp, ss) = self.cluster.shared_core.raw_storage_bytes();
        let (ap, as_) = self.cluster.left_transforms[self.layer_index].raw_storage_bytes();
        let (bp, bs) = self.cluster.right_transforms[self.layer_index].raw_storage_bytes();
        let (rp, rs, rst) = self.cluster.residuals[self.layer_index].storage_bytes();
        StorageBreakdown {
            payload_bytes: rp,
            scale_metadata_bytes: ss.div_ceil(n_layers) + as_ + bs + rs,
            structure_bytes: sp.div_ceil(n_layers) + ap + bp + rst,
            header_bytes: RMQ_LAYER_HEADER_BYTES + RMQ_SHARED_HEADER_BYTES.div_ceil(n_layers),
        }
    }
}

fn seed_offset(base: u64, delta: u64) -> u64 {
    base.wrapping_add(delta)
}

fn compute_shared_u(
    layers: &[&DenseMatrix],
    vt: &DenseMatrix,
    target_p: usize,
    seed: u64,
) -> Result<DenseMatrix, MatrixError> {
    let m = layers[0].rows();
    let q = vt.rows();
    let v = vt.transpose(); // n × q
    let total_cols = layers.len() * q;
    let mut concat = vec![0.0_f64; m * total_cols];
    for (idx, mat) in layers.iter().enumerate() {
        let proj = mat.matmul(&v)?; // m × q
        for r in 0..m {
            let dst_start = r * total_cols + idx * q;
            let src_start = r * q;
            concat[dst_start..dst_start + q]
                .copy_from_slice(&proj.as_slice()[src_start..src_start + q]);
        }
    }
    let mat_concat = DenseMatrix::from_flat(m, total_cols, concat)?;
    Ok(truncated_svd(&mat_concat, target_p, seed)?.u)
}

fn compute_shared_vt(
    layers: &[&DenseMatrix],
    u: &DenseMatrix,
    target_q: usize,
    seed: u64,
) -> Result<DenseMatrix, MatrixError> {
    let n = layers[0].cols();
    let p = u.cols();
    let ut = u.transpose(); // p × m
    let mut concat = Vec::with_capacity(layers.len() * p * n);
    for mat in layers {
        let proj = ut.matmul(mat)?; // p × n
        concat.extend_from_slice(proj.as_slice());
    }
    let mat_concat = DenseMatrix::from_flat(layers.len() * p, n, concat)?;
    Ok(truncated_svd(&mat_concat, target_q, seed)?.vt)
}

/// Summary row comparing one compression method across a multi-layer cluster.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RmqComparisonRow {
    pub method: String,
    pub total_bytes_all_layers: usize,
    pub effective_bpp: f64,
    pub mean_rel_frobenius_error: f64,
    pub mean_matvec_rel_error: f64,
    pub total_encode_micros: u64,
}

/// Reproducible head-to-head ablation comparing RMQ variants against conventional baselines.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RmqAblationReport {
    pub cluster_label: String,
    pub num_layers: usize,
    pub shape: [usize; 2],
    pub total_parameters: usize,
    pub rows: Vec<RmqComparisonRow>,
}

/// Compare RMQ Shared-Basis, RMQ Shared-Core, independent SVD, INT4, and INT8 on a layer cluster.
pub fn compare_rmq_vs_baselines(
    cluster_label: &str,
    layers: &[&DenseMatrix],
    rank: usize,
    seed: u64,
) -> Result<RmqAblationReport, MatrixError> {
    let Some(first) = layers.first() else {
        return Err(MatrixError::Empty);
    };
    let m = first.rows();
    let n = first.cols();
    let total_params = layers.len() * m * n;
    let probes = deterministic_probe_vectors(n, 6, seed);
    let mut rows = Vec::new();

    // Helper to aggregate per-layer evaluation metrics.
    let aggregate = |name: &str, metrics: &[EvaluationMetrics], encode_micros: u64| {
        let total_bytes: usize = metrics.iter().map(|e| e.total_bytes).sum();
        let mean_frob = metrics
            .iter()
            .map(|e| e.relative_frobenius_error)
            .sum::<f64>()
            / metrics.len() as f64;
        let mean_mv =
            metrics.iter().map(|e| e.mean_matvec_rel_error).sum::<f64>() / metrics.len() as f64;
        RmqComparisonRow {
            method: name.to_owned(),
            total_bytes_all_layers: total_bytes,
            effective_bpp: (total_bytes as f64 * 8.0) / total_params as f64,
            mean_rel_frobenius_error: mean_frob,
            mean_matvec_rel_error: mean_mv,
            total_encode_micros: encode_micros,
        }
    };

    // 1. Conventional INT8_SYM g64
    {
        let t0 = Instant::now();
        let mut evals = Vec::new();
        for mat in layers {
            let op = Int8SymMatrix::quantize(mat, 64)?;
            evals.push(evaluate_operator(mat, &op, &probes, 0)?);
        }
        rows.push(aggregate(
            "INT8_SYM_G64",
            &evals,
            t0.elapsed().as_micros() as u64,
        ));
    }

    // 2. Conventional INT4_SYM g32 (Q4_0)
    {
        let t0 = Instant::now();
        let mut evals = Vec::new();
        for mat in layers {
            let op = Int4SymGroupMatrix::quantize(mat, 32)?;
            evals.push(evaluate_operator(mat, &op, &probes, 0)?);
        }
        rows.push(aggregate(
            "INT4_SYM_G32(Q4_0)",
            &evals,
            t0.elapsed().as_micros() as u64,
        ));
    }

    // 3. Independent per-layer SVD (Int8 factors)
    {
        let t0 = Instant::now();
        let mut evals = Vec::new();
        for (idx, mat) in layers.iter().enumerate() {
            let op = LowRankSvdMatrix::from_matrix(
                mat,
                rank,
                FactorPrecision::Int8Sym,
                seed.wrapping_add(idx as u64),
            )?;
            evals.push(evaluate_operator(mat, &op, &probes, 0)?);
        }
        rows.push(aggregate(
            "INDEPENDENT_SVD_INT8",
            &evals,
            t0.elapsed().as_micros() as u64,
        ));
    }

    // 4. RMQ Shared-Basis (no residual)
    {
        let t0 = Instant::now();
        let cluster = RmqSharedBasisCluster::fit(
            layers,
            RmqSharedBasisConfig {
                left_rank: rank,
                right_rank: rank,
                hooi_iters: 2,
                basis_precision: FactorPrecision::Int8Sym,
                core_precision: FactorPrecision::Fp16,
                residual_kind: RmqResidualKind::None,
                seed,
            },
        )?;
        let enc = t0.elapsed().as_micros() as u64;
        let mut evals = Vec::new();
        for (i, mat) in layers.iter().enumerate() {
            let op = cluster.layer(i).unwrap();
            evals.push(evaluate_operator(mat, &op, &probes, 0)?);
        }
        rows.push(aggregate("RMQ_SHARED_BASIS", &evals, enc));
    }

    // 5. RMQ Shared-Basis + 1% Sparse Outlier Residual
    {
        let t0 = Instant::now();
        let cluster = RmqSharedBasisCluster::fit(
            layers,
            RmqSharedBasisConfig {
                left_rank: rank,
                right_rank: rank,
                hooi_iters: 2,
                basis_precision: FactorPrecision::Int8Sym,
                core_precision: FactorPrecision::Fp16,
                residual_kind: RmqResidualKind::SparseTopFraction { fraction: 0.01 },
                seed,
            },
        )?;
        let enc = t0.elapsed().as_micros() as u64;
        let mut evals = Vec::new();
        for (i, mat) in layers.iter().enumerate() {
            let op = cluster.layer(i).unwrap();
            evals.push(evaluate_operator(mat, &op, &probes, 0)?);
        }
        rows.push(aggregate("RMQ_SHARED_BASIS+SPARSE1%", &evals, enc));
    }

    // 6. RMQ Shared-Core (A_l * S * B_l + 1% Sparse)
    {
        let t0 = Instant::now();
        let cluster = RmqSharedCoreCluster::fit(
            layers,
            rank,
            FactorPrecision::Int8Sym,
            RmqResidualKind::SparseTopFraction { fraction: 0.01 },
            seed,
        )?;
        let enc = t0.elapsed().as_micros() as u64;
        let mut evals = Vec::new();
        for (i, mat) in layers.iter().enumerate() {
            let op = cluster.layer(i).unwrap();
            evals.push(evaluate_operator(mat, &op, &probes, 0)?);
        }
        rows.push(aggregate("RMQ_SHARED_CORE+SPARSE1%", &evals, enc));
    }

    Ok(RmqAblationReport {
        cluster_label: cluster_label.to_owned(),
        num_layers: layers.len(),
        shape: [m, n],
        total_parameters: total_params,
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use qml_core::DeterministicRng;

    fn make_correlated_cluster(
        num_layers: usize,
        m: usize,
        n: usize,
        true_rank: usize,
        noise_std: f64,
        seed: u64,
    ) -> Vec<DenseMatrix> {
        let mut rng = DeterministicRng::new(seed);
        let u_base = DenseMatrix::from_flat(
            m,
            true_rank,
            (0..m * true_rank)
                .map(|_| rng.next_normal() / (m as f64).sqrt())
                .collect(),
        )
        .unwrap();
        let vt_base = DenseMatrix::from_flat(
            true_rank,
            n,
            (0..true_rank * n)
                .map(|_| rng.next_normal() / (n as f64).sqrt())
                .collect(),
        )
        .unwrap();

        let mut layers = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let core_data: Vec<f64> = (0..true_rank * true_rank)
                .map(|_| rng.next_normal() * (1.0 + 0.1 * l as f64))
                .collect();
            let core = DenseMatrix::from_flat(true_rank, true_rank, core_data).unwrap();
            let clean = u_base.matmul(&core).unwrap().matmul(&vt_base).unwrap();
            let mut noisy = clean.as_slice().to_vec();
            for val in &mut noisy {
                *val += rng.next_normal() * noise_std;
            }
            // Add a couple of strong outliers to test sparse residual capture.
            noisy[0] += 3.5;
            noisy[n + 1] -= 2.8;
            layers.push(DenseMatrix::from_flat(m, n, noisy).unwrap());
        }
        layers
    }

    #[test]
    fn rmq_shared_basis_and_core_direct_matvec_match_reconstructed_dense() {
        let layers = make_correlated_cluster(4, 32, 32, 4, 0.01, 42);
        let refs: Vec<&DenseMatrix> = layers.iter().collect();
        let x = &deterministic_probe_vectors(32, 1, 7)[0];

        let basis_cluster = RmqSharedBasisCluster::fit(
            &refs,
            RmqSharedBasisConfig {
                left_rank: 4,
                right_rank: 4,
                hooi_iters: 2,
                basis_precision: FactorPrecision::Fp16,
                core_precision: FactorPrecision::Fp16,
                residual_kind: RmqResidualKind::SparseTopFraction { fraction: 0.01 },
                seed: 11,
            },
        )
        .unwrap();

        let core_cluster = RmqSharedCoreCluster::fit(
            &refs,
            4,
            FactorPrecision::Fp16,
            RmqResidualKind::SparseTopFraction { fraction: 0.01 },
            11,
        )
        .unwrap();

        for i in 0..refs.len() {
            let b_layer = basis_cluster.layer(i).unwrap();
            let b_direct = b_layer.matvec(x).unwrap();
            let b_dense = b_layer.reconstruct_dense().unwrap().matvec(x).unwrap();
            for (a, b) in b_direct.iter().zip(b_dense) {
                assert!((a - b).abs() < 1e-10);
            }

            let c_layer = core_cluster.layer(i).unwrap();
            let c_direct = c_layer.matvec(x).unwrap();
            let c_dense = c_layer.reconstruct_dense().unwrap().matvec(x).unwrap();
            for (a, b) in c_direct.iter().zip(c_dense) {
                assert!((a - b).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn rmq_shared_basis_achieves_sub_2bpp_and_beats_independent_svd_on_correlated_layers() {
        let layers = make_correlated_cluster(8, 64, 64, 6, 0.005, 101);
        let refs: Vec<&DenseMatrix> = layers.iter().collect();
        let ablation = compare_rmq_vs_baselines("correlated_8x64x64", &refs, 6, 77).unwrap();

        let rmq_sparse = ablation
            .rows
            .iter()
            .find(|r| r.method == "RMQ_SHARED_BASIS+SPARSE1%")
            .unwrap();
        let svd_indep = ablation
            .rows
            .iter()
            .find(|r| r.method == "INDEPENDENT_SVD_INT8")
            .unwrap();
        let q4 = ablation
            .rows
            .iter()
            .find(|r| r.method == "INT4_SYM_G32(Q4_0)")
            .unwrap();

        // RMQ Shared-Basis + 1% sparse residual should use far fewer bits/param than Q4_0 and independent SVD
        assert!(
            rmq_sparse.effective_bpp < 2.0,
            "expected < 2.0 bpp, got {:.3}",
            rmq_sparse.effective_bpp
        );
        assert!(
            rmq_sparse.effective_bpp < svd_indep.effective_bpp,
            "RMQ ({:.3} bpp) should be smaller than independent SVD ({:.3} bpp)",
            rmq_sparse.effective_bpp,
            svd_indep.effective_bpp
        );
        assert!(
            rmq_sparse.mean_rel_frobenius_error < q4.mean_rel_frobenius_error * 2.5,
            "RMQ error {:.4} vs Q4_0 {:.4}",
            rmq_sparse.mean_rel_frobenius_error,
            q4.mean_rel_frobenius_error
        );
    }

    #[test]
    fn rmq_honestly_shows_higher_error_on_uncorrelated_full_rank_gaussian_control() {
        let mut rng = DeterministicRng::new(999);
        let layers: Vec<DenseMatrix> = (0..4)
            .map(|_| {
                DenseMatrix::from_flat(32, 32, (0..1024).map(|_| rng.next_normal()).collect())
                    .unwrap()
            })
            .collect();
        let refs: Vec<&DenseMatrix> = layers.iter().collect();
        let ablation =
            compare_rmq_vs_baselines("uncorrelated_gaussian_control", &refs, 4, 5).unwrap();

        let rmq = ablation
            .rows
            .iter()
            .find(|r| r.method == "RMQ_SHARED_BASIS")
            .unwrap();
        let q4 = ablation
            .rows
            .iter()
            .find(|r| r.method == "INT4_SYM_G32(Q4_0)")
            .unwrap();
        // On i.i.d. full-rank Gaussian weights, low-rank shared basis CANNOT capture full spectrum:
        // Q4_0 must have much lower reconstruction error than rank-4 RMQ!
        assert!(q4.mean_rel_frobenius_error < rmq.mean_rel_frobenius_error);
    }
}
