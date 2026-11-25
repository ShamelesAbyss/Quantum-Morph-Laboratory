//! Q-Forge — Memory-Bounded Model Transformation Engine for Quantum Morph Laboratory.
//!
//! Transforms immutable Safetensors models into experimental QMF v1 artifacts
//! with tensor-by-tensor and cluster-by-cluster bounded memory execution,
//! real SHA-256 provenance hashing, complete on-disk byte accounting, and
//! direct-execution verification against the original weights.

use qml_core::{DenseMatrix, sha256_hex, truncated_svd};
use qml_format::{QmfArchiveV1, QmfError, QmfLatentCache, QmfV1Builder};
use qml_model::{PayloadBudget, PayloadError, SafetensorsFile};
use qml_quant::{
    Bf16Matrix, FactorPrecision, Fp16Matrix, Int4AsymGroupMatrix, Int4SymGroupMatrix,
    Int8SymMatrix, LinearOperator, LowRankSvdMatrix, MapConfig, RmqResidualKind,
    RmqSharedBasisCluster, RmqSharedBasisConfig, RmqSharedCoreCluster, allocate_from_profiles,
    deterministic_probe_vectors, evaluate_operator, profile_layer_for_map,
};
use qml_scope::{LayerRole, classify_tensor_name};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::io::{Read, Seek};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Supported model-level transformation strategies in Q-Forge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum TransformStrategy {
    Fp16,
    Bf16,
    Int8Sym {
        group_size: usize,
    },
    Int4Sym {
        group_size: usize,
    },
    Int4Asym {
        group_size: usize,
    },
    LowRankSvd {
        rank: usize,
        precision: FactorPrecision,
    },
    /// Clusters same-role, same-shape 2D tensors across layers into shared-basis RMQ
    /// representations (`U_shared * C_l * V_sharedᵀ + SparseResidual`); processed one
    /// cluster at a time to bound working RAM.
    RmqSharedBasis {
        rank: usize,
        sparse_fraction: f64,
        basis_precision: FactorPrecision,
        core_precision: FactorPrecision,
    },
    /// Clusters same-role, same-shape 2D tensors across layers into shared-core RMQ
    /// representations (`A_l * S * B_l + SparseResidual`); processed one cluster at a time.
    RmqSharedCore {
        rank: usize,
        sparse_fraction: f64,
        factor_precision: FactorPrecision,
    },
    /// Runs Morphological Adaptive Precision (MAP) across 2D tensors under a global
    /// target average bits-per-parameter budget.
    MapAdaptive {
        target_avg_bpp: f64,
        propagation_weight: f64,
        include_svd_rank: Option<usize>,
    },
    /// Selective Affinity-Gated Hybrid RMQ + Quantization: computes pairwise Grassmannian
    /// subspace affinity across each same-role cluster, applies RMQ Shared-Basis + sparse
    /// outlier residual ONLY when mean affinity `>= affinity_threshold`, and routes
    /// low-affinity clusters (and embeddings/heads) to blockwise `INT4_SYM (Q4_0)`.
    RmqAffinityGated {
        rank: usize,
        affinity_threshold: f64,
        sparse_fraction: f64,
        fallback_group_size: usize,
    },
}

impl TransformStrategy {
    pub fn descriptor(&self) -> TransformDescriptor {
        match self {
            Self::Fp16 => TransformDescriptor {
                name: "fp16",
                assumptions: "IEEE-754 binary16 range [-65504, 65504]; tensor-by-tensor streaming",
                deterministic: true,
                maturity: "conventional-baseline",
            },
            Self::Bf16 => TransformDescriptor {
                name: "bf16",
                assumptions: "BFloat16 dynamic range with 7-bit mantissa; tensor-by-tensor streaming",
                deterministic: true,
                maturity: "conventional-baseline",
            },
            Self::Int8Sym { .. } => TransformDescriptor {
                name: "int8_sym",
                assumptions: "Symmetric groupwise uniform quantization with f16 scales; tensor-by-tensor",
                deterministic: true,
                maturity: "conventional-baseline",
            },
            Self::Int4Sym { .. } => TransformDescriptor {
                name: "int4_sym",
                assumptions: "GGML Q4_0-style symmetric packed 4-bit blocks with f16 scales; tensor-by-tensor",
                deterministic: true,
                maturity: "conventional-baseline",
            },
            Self::Int4Asym { .. } => TransformDescriptor {
                name: "int4_asym",
                assumptions: "GGML Q4_1-style affine packed 4-bit blocks with f16 scale+min; tensor-by-tensor",
                deterministic: true,
                maturity: "conventional-baseline",
            },
            Self::LowRankSvd { .. } => TransformDescriptor {
                name: "low_rank_svd",
                assumptions: "Per-layer truncated SVD with symmetrically scaled quantized factors; tensor-by-tensor",
                deterministic: true,
                maturity: "conventional-baseline",
            },
            Self::RmqSharedBasis { .. } => TransformDescriptor {
                name: "rmq_shared_basis",
                assumptions: "Cluster-by-cluster 2D-HOOI shared orthonormal bases + outlier peeling + sparse residual",
                deterministic: true,
                maturity: "experimental-proposal",
            },
            Self::RmqSharedCore { .. } => TransformDescriptor {
                name: "rmq_shared_core",
                assumptions: "Cluster-by-cluster shared spectral core S with layer-specific A_l, B_l + sparse residual",
                deterministic: true,
                maturity: "experimental-proposal",
            },
            Self::MapAdaptive { .. } => TransformDescriptor {
                name: "map_adaptive",
                assumptions: "Sensitivity + downstream spectral error-propagation greedy byte-budget allocator",
                deterministic: true,
                maturity: "experimental-proposal",
            },
            Self::RmqAffinityGated { .. } => TransformDescriptor {
                name: "rmq_affinity_gated",
                assumptions: "Grassmannian subspace affinity gating: RMQ on correlated clusters, Q4_0/INT8 fallback on uncorrelated layers",
                deterministic: true,
                maturity: "experimental-proposal",
            },
        }
    }
}

/// Self-declared transformation metadata required by the QML directive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransformDescriptor {
    pub name: &'static str,
    pub assumptions: &'static str,
    pub deterministic: bool,
    pub maturity: &'static str,
}

/// Per-tensor summary recorded inside a [`ForgeReport`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TransformedTensorSummary {
    pub name: String,
    pub shape: Vec<usize>,
    pub representation: String,
    pub relative_frobenius_error: Option<f64>,
    pub mean_matvec_rel_error: Option<f64>,
}

/// Reproducible experimental report produced by Q-Forge.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ForgeReport {
    pub descriptor: TransformDescriptor,
    pub model_id: String,
    pub random_seed: u64,
    pub source_file_bytes: u64,
    pub qmf_serialized_bytes: usize,
    pub qmf_sha256: String,
    pub total_parameters: u64,
    pub effective_qmf_bpp: f64,
    pub compression_ratio_vs_source: f64,
    pub shared_latent_count: usize,
    pub transformed_tensor_count: usize,
    pub mean_rel_frobenius_error: f64,
    pub mean_matvec_rel_error: f64,
    pub elapsed_millis: u64,
    pub tensors: Vec<TransformedTensorSummary>,
}

/// Verification report produced by comparing a QMF v1 artifact directly against its source model.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CompareReport {
    pub qmf_model_id: String,
    pub transformation: String,
    pub qmf_sha256: String,
    pub tensors_compared: usize,
    pub source_file_bytes: u64,
    pub qmf_file_bytes: usize,
    pub total_parameters: u64,
    pub effective_qmf_bpp: f64,
    pub mean_rel_frobenius_error: f64,
    pub max_rel_frobenius_error: f64,
    pub mean_matvec_rel_error: f64,
    pub max_matvec_rel_error: f64,
}

#[derive(Debug)]
pub enum ForgeError {
    Payload(PayloadError),
    Qmf(QmfError),
    Matrix(qml_core::MatrixError),
    Cancelled,
    Message(String),
}

impl fmt::Display for ForgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Payload(e) => write!(f, "model payload error: {e}"),
            Self::Qmf(e) => write!(f, "QMF error: {e}"),
            Self::Matrix(e) => write!(f, "matrix error: {e}"),
            Self::Cancelled => write!(f, "transformation cancelled by user"),
            Self::Message(m) => write!(f, "{m}"),
        }
    }
}

/// Progress callback invoked by Q-Forge as `(completed_tensors, total_tensors, stage_label)`.
pub type ForgeProgressCallback = Arc<dyn Fn(usize, usize, &str) + Send + Sync>;

/// Cooperative cancellation and progress reporting handle for [`forge_safetensors_to_qmf_with_control`].
pub struct ForgeControl {
    pub cancel: Arc<AtomicBool>,
    pub on_progress: Option<ForgeProgressCallback>,
}

impl Default for ForgeControl {
    fn default() -> Self {
        Self {
            cancel: Arc::new(AtomicBool::new(false)),
            on_progress: None,
        }
    }
}

impl ForgeControl {
    fn check_cancelled(&self) -> Result<(), ForgeError> {
        if self.cancel.load(Ordering::Relaxed) {
            Err(ForgeError::Cancelled)
        } else {
            Ok(())
        }
    }

    fn report(&self, done: usize, total: usize, stage: &str) {
        if let Some(cb) = &self.on_progress {
            cb(done, total, stage);
        }
    }
}

impl std::error::Error for ForgeError {}
impl From<PayloadError> for ForgeError {
    fn from(v: PayloadError) -> Self {
        Self::Payload(v)
    }
}
impl From<QmfError> for ForgeError {
    fn from(v: QmfError) -> Self {
        Self::Qmf(v)
    }
}
impl From<qml_core::MatrixError> for ForgeError {
    fn from(v: qml_core::MatrixError) -> Self {
        Self::Matrix(v)
    }
}

/// Transform a read-only Safetensors stream into a validated QMF v1 archive and report.
///
/// Single-tensor strategies (`Fp16`, `Bf16`, `Int8Sym`, `Int4Sym`, `Int4Asym`, `LowRankSvd`)
/// and two-pass streaming `MapAdaptive` execute strictly **tensor-by-tensor** (holding only one
/// decoded matrix in memory at a time). RMQ strategies execute **cluster-by-cluster**.
pub fn forge_safetensors_to_qmf<R: Read + Seek>(
    reader: R,
    model_id: &str,
    source_license: &str,
    architecture: &str,
    strategy: &TransformStrategy,
    budget: &PayloadBudget,
    seed: u64,
) -> Result<(QmfArchiveV1, ForgeReport), ForgeError> {
    forge_safetensors_to_qmf_with_control(
        reader,
        model_id,
        source_license,
        architecture,
        strategy,
        budget,
        seed,
        &ForgeControl::default(),
    )
}

/// Transform a read-only Safetensors stream into a validated QMF v1 archive with
/// cooperative cancellation and per-tensor progress reporting.
#[allow(clippy::too_many_arguments)]
pub fn forge_safetensors_to_qmf_with_control<R: Read + Seek>(
    reader: R,
    model_id: &str,
    source_license: &str,
    architecture: &str,
    strategy: &TransformStrategy,
    budget: &PayloadBudget,
    seed: u64,
    control: &ForgeControl,
) -> Result<(QmfArchiveV1, ForgeReport), ForgeError> {
    control.check_cancelled()?;
    let start = Instant::now();
    let mut file = SafetensorsFile::open(reader, budget)?;
    let meta = file.metadata().clone();
    let descriptor = strategy.descriptor();
    let total_tensors = meta.tensors.len().max(1);
    let mut processed_tensors = 0usize;

    let mut builder = QmfV1Builder::new(
        model_id,
        "safetensors",
        source_license,
        architecture,
        descriptor.name,
        seed,
    );
    builder.insert_metadata("assumptions", descriptor.assumptions);
    builder.insert_metadata("maturity", descriptor.maturity);

    let mut total_params = 0_u64;
    let mut tensor_summaries: Vec<TransformedTensorSummary> =
        Vec::with_capacity(meta.tensors.len());
    let mut sum_frob = 0.0_f64;
    let mut sum_mv = 0.0_f64;
    let mut mat_count = 0_usize;

    // Separate tensor names by dimensionality using metadata only (zero payload loaded yet).
    let mut matrix_infos = Vec::new();
    let mut vector_infos = Vec::new();
    for info in &meta.tensors {
        let count = qml_model::tensor_element_count(info)?;
        total_params = total_params.saturating_add(count);
        if info.shape.len() == 2 && info.shape[0] > 0 && info.shape[1] > 0 {
            matrix_infos.push(info.clone());
        } else {
            vector_infos.push(info.clone());
        }
    }

    // 1. Stream 1D/non-2D vectors one by one and drop immediately after encoding.
    for info in &vector_infos {
        control.check_cancelled()?;
        let vals = file.read_tensor_f64(&info.name, budget)?;
        let shape_usize: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
        let f32_vals: Vec<f32> = vals.into_iter().map(|v| v as f32).collect();
        builder.add_dense_f32(info.name.clone(), shape_usize.clone(), &f32_vals)?;
        tensor_summaries.push(TransformedTensorSummary {
            name: info.name.clone(),
            shape: shape_usize,
            representation: "QMF_DENSE_F32".to_owned(),
            relative_frobenius_error: Some(0.0),
            mean_matvec_rel_error: None,
        });
        processed_tensors += 1;
        control.report(processed_tensors, total_tensors, &info.name);
    }

    // 2. Transform 2D matrices with memory-bounded streaming per strategy.
    match strategy {
        TransformStrategy::Fp16
        | TransformStrategy::Bf16
        | TransformStrategy::Int8Sym { .. }
        | TransformStrategy::Int4Sym { .. }
        | TransformStrategy::Int4Asym { .. }
        | TransformStrategy::LowRankSvd { .. } => {
            for (idx, info) in matrix_infos.iter().enumerate() {
                control.check_cancelled()?;
                let mat = file.read_tensor_matrix(&info.name, budget)?;
                let probes = deterministic_probe_vectors(mat.cols(), 4, seed);
                let (rep_name, rel_frob, mv_err) = match strategy {
                    TransformStrategy::Fp16 => {
                        let op = Fp16Matrix::quantize(&mat);
                        let ev = evaluate_operator(&mat, &op, &probes, 0)?;
                        builder.add_fp16(info.name.clone(), &op)?;
                        (
                            "QMF_FP16".to_owned(),
                            ev.relative_frobenius_error,
                            ev.mean_matvec_rel_error,
                        )
                    }
                    TransformStrategy::Bf16 => {
                        let op = Bf16Matrix::quantize(&mat);
                        let ev = evaluate_operator(&mat, &op, &probes, 0)?;
                        builder.add_bf16(info.name.clone(), &op)?;
                        (
                            "QMF_BF16".to_owned(),
                            ev.relative_frobenius_error,
                            ev.mean_matvec_rel_error,
                        )
                    }
                    TransformStrategy::Int8Sym { group_size } => {
                        let op = Int8SymMatrix::quantize(&mat, *group_size)?;
                        let ev = evaluate_operator(&mat, &op, &probes, 0)?;
                        builder.add_int8_sym(info.name.clone(), &op)?;
                        (
                            "QMF_INT8_SYM".to_owned(),
                            ev.relative_frobenius_error,
                            ev.mean_matvec_rel_error,
                        )
                    }
                    TransformStrategy::Int4Sym { group_size } => {
                        let op = Int4SymGroupMatrix::quantize(&mat, *group_size)?;
                        let ev = evaluate_operator(&mat, &op, &probes, 0)?;
                        builder.add_int4_sym(info.name.clone(), &op)?;
                        (
                            "QMF_INT4_SYM".to_owned(),
                            ev.relative_frobenius_error,
                            ev.mean_matvec_rel_error,
                        )
                    }
                    TransformStrategy::Int4Asym { group_size } => {
                        let op = Int4AsymGroupMatrix::quantize(&mat, *group_size)?;
                        let ev = evaluate_operator(&mat, &op, &probes, 0)?;
                        builder.add_int4_asym(info.name.clone(), &op)?;
                        (
                            "QMF_INT4_ASYM".to_owned(),
                            ev.relative_frobenius_error,
                            ev.mean_matvec_rel_error,
                        )
                    }
                    TransformStrategy::LowRankSvd { rank, precision } => {
                        let r = (*rank).max(1).min(mat.rows().min(mat.cols()));
                        let op = LowRankSvdMatrix::from_matrix(
                            &mat,
                            r,
                            *precision,
                            seed.wrapping_add(idx as u64),
                        )?;
                        let ev = evaluate_operator(&mat, &op, &probes, 0)?;
                        builder.add_low_rank_svd(info.name.clone(), &op)?;
                        (
                            "QMF_SVD".to_owned(),
                            ev.relative_frobenius_error,
                            ev.mean_matvec_rel_error,
                        )
                    }
                    _ => unreachable!(),
                };
                sum_frob += rel_frob;
                sum_mv += mv_err;
                mat_count += 1;
                tensor_summaries.push(TransformedTensorSummary {
                    name: info.name.clone(),
                    shape: vec![mat.rows(), mat.cols()],
                    representation: rep_name,
                    relative_frobenius_error: Some(rel_frob),
                    mean_matvec_rel_error: Some(mv_err),
                });
                // `mat` drops here before loading the next tensor!
            }
        }
        TransformStrategy::RmqSharedBasis {
            rank,
            sparse_fraction,
            basis_precision,
            core_precision,
        } => {
            // Group metadata by (LayerRole, rows, cols) and process ONE cluster at a time!
            type ClusterKey = (LayerRole, usize, usize);
            let mut groups: BTreeMap<ClusterKey, Vec<String>> = BTreeMap::new();
            for info in &matrix_infos {
                let (_, role) = classify_tensor_name(&info.name);
                groups
                    .entry((role, info.shape[0] as usize, info.shape[1] as usize))
                    .or_default()
                    .push(info.name.clone());
            }
            for ((role, rows, cols), names) in groups {
                control.check_cancelled()?;
                control.report(
                    processed_tensors,
                    total_tensors,
                    &format!("RMQ gated cluster: {} ({}x{})", role.as_str(), rows, cols),
                );
                let cluster_mats: Vec<DenseMatrix> = names
                    .iter()
                    .map(|n| file.read_tensor_matrix(n, budget))
                    .collect::<Result<_, _>>()?;
                let probes = deterministic_probe_vectors(cols, 4, seed);
                if cluster_mats.len() >= 2 {
                    let refs: Vec<&DenseMatrix> = cluster_mats.iter().collect();
                    let r = (*rank).max(1).min(rows.min(cols));
                    let residual_kind = if *sparse_fraction > 0.0 {
                        RmqResidualKind::SparseTopFraction {
                            fraction: *sparse_fraction,
                        }
                    } else {
                        RmqResidualKind::None
                    };
                    let cluster = RmqSharedBasisCluster::fit(
                        &refs,
                        RmqSharedBasisConfig {
                            left_rank: r,
                            right_rank: r,
                            hooi_iters: 2,
                            basis_precision: *basis_precision,
                            core_precision: *core_precision,
                            residual_kind,
                            seed,
                        },
                    )?;
                    for (local_i, (name, mat)) in names.iter().zip(&cluster_mats).enumerate() {
                        let op = cluster.layer(local_i).unwrap();
                        let ev = evaluate_operator(mat, &op, &probes, 0)?;
                        sum_frob += ev.relative_frobenius_error;
                        sum_mv += ev.mean_matvec_rel_error;
                        mat_count += 1;
                        tensor_summaries.push(TransformedTensorSummary {
                            name: name.clone(),
                            shape: vec![rows, cols],
                            representation: op.method_name().to_owned(),
                            relative_frobenius_error: Some(ev.relative_frobenius_error),
                            mean_matvec_rel_error: Some(ev.mean_matvec_rel_error),
                        });
                    }
                    let cluster_id = format!("{}_{rows}x{cols}", role.as_str());
                    builder.add_rmq_shared_basis_cluster(&cluster_id, &names, &cluster)?;
                } else {
                    let mat = &cluster_mats[0];
                    let op = Int8SymMatrix::quantize(mat, 64)?;
                    let ev = evaluate_operator(mat, &op, &probes, 0)?;
                    builder.add_int8_sym(names[0].clone(), &op)?;
                    sum_frob += ev.relative_frobenius_error;
                    sum_mv += ev.mean_matvec_rel_error;
                    mat_count += 1;
                    tensor_summaries.push(TransformedTensorSummary {
                        name: names[0].clone(),
                        shape: vec![rows, cols],
                        representation: "QMF_INT8_SYM".to_owned(),
                        relative_frobenius_error: Some(ev.relative_frobenius_error),
                        mean_matvec_rel_error: Some(ev.mean_matvec_rel_error),
                    });
                }
                // Entire cluster drops here before loading the next role cluster!
            }
        }
        TransformStrategy::RmqSharedCore {
            rank,
            sparse_fraction,
            factor_precision,
        } => {
            type ClusterKey = (LayerRole, usize, usize);
            let mut groups: BTreeMap<ClusterKey, Vec<String>> = BTreeMap::new();
            for info in &matrix_infos {
                let (_, role) = classify_tensor_name(&info.name);
                groups
                    .entry((role, info.shape[0] as usize, info.shape[1] as usize))
                    .or_default()
                    .push(info.name.clone());
            }
            for ((role, rows, cols), names) in groups {
                control.check_cancelled()?;
                control.report(
                    processed_tensors,
                    total_tensors,
                    &format!("RMQ gated cluster: {} ({}x{})", role.as_str(), rows, cols),
                );
                let cluster_mats: Vec<DenseMatrix> = names
                    .iter()
                    .map(|n| file.read_tensor_matrix(n, budget))
                    .collect::<Result<_, _>>()?;
                let probes = deterministic_probe_vectors(cols, 4, seed);
                if cluster_mats.len() >= 2 {
                    let refs: Vec<&DenseMatrix> = cluster_mats.iter().collect();
                    let r = (*rank).max(1).min(rows.min(cols));
                    let residual_kind = if *sparse_fraction > 0.0 {
                        RmqResidualKind::SparseTopFraction {
                            fraction: *sparse_fraction,
                        }
                    } else {
                        RmqResidualKind::None
                    };
                    let cluster = RmqSharedCoreCluster::fit(
                        &refs,
                        r,
                        *factor_precision,
                        residual_kind,
                        seed,
                    )?;
                    for (local_i, (name, mat)) in names.iter().zip(&cluster_mats).enumerate() {
                        let op = cluster.layer(local_i).unwrap();
                        let ev = evaluate_operator(mat, &op, &probes, 0)?;
                        sum_frob += ev.relative_frobenius_error;
                        sum_mv += ev.mean_matvec_rel_error;
                        mat_count += 1;
                        tensor_summaries.push(TransformedTensorSummary {
                            name: name.clone(),
                            shape: vec![rows, cols],
                            representation: op.method_name().to_owned(),
                            relative_frobenius_error: Some(ev.relative_frobenius_error),
                            mean_matvec_rel_error: Some(ev.mean_matvec_rel_error),
                        });
                    }
                    let cluster_id = format!("{}_{rows}x{cols}", role.as_str());
                    builder.add_rmq_shared_core_cluster(&cluster_id, &names, &cluster)?;
                } else {
                    let mat = &cluster_mats[0];
                    let op = Int8SymMatrix::quantize(mat, 64)?;
                    let ev = evaluate_operator(mat, &op, &probes, 0)?;
                    builder.add_int8_sym(names[0].clone(), &op)?;
                    sum_frob += ev.relative_frobenius_error;
                    sum_mv += ev.mean_matvec_rel_error;
                    mat_count += 1;
                    tensor_summaries.push(TransformedTensorSummary {
                        name: names[0].clone(),
                        shape: vec![rows, cols],
                        representation: "QMF_INT8_SYM".to_owned(),
                        relative_frobenius_error: Some(ev.relative_frobenius_error),
                        mean_matvec_rel_error: Some(ev.mean_matvec_rel_error),
                    });
                }
            }
        }
        TransformStrategy::MapAdaptive {
            target_avg_bpp,
            propagation_weight,
            include_svd_rank,
        } => {
            if !matrix_infos.is_empty() {
                let map_cfg = MapConfig {
                    target_avg_bpp: *target_avg_bpp,
                    propagation_weight: *propagation_weight,
                    include_svd_rank: *include_svd_rank,
                    seed,
                };
                // Pass 1 (streaming): profile one matrix at a time and drop dense weights immediately.
                let mut profiles = Vec::with_capacity(matrix_infos.len());
                for (idx, info) in matrix_infos.iter().enumerate() {
                    control.check_cancelled()?;
                    control.report(
                        processed_tensors,
                        total_tensors,
                        &format!("MAP pass 1/2: {}", info.name),
                    );
                    let mat = file.read_tensor_matrix(&info.name, budget)?;
                    profiles.push(profile_layer_for_map(&info.name, idx, &mat, &map_cfg)?);
                }
                let plan = allocate_from_profiles(&profiles, &map_cfg)?;

                // Pass 2 (streaming): re-read one matrix at a time and encode ONLY its chosen tier.
                for (idx, alloc) in plan.allocations.iter().enumerate() {
                    control.check_cancelled()?;
                    let name = &matrix_infos[idx].name;
                    let mat = file.read_tensor_matrix(name, budget)?;
                    let probes = deterministic_probe_vectors(mat.cols(), 4, seed);
                    let (rep_name, rel_frob, mv_err) = match alloc.selected.method.as_str() {
                        "FP16" => {
                            let op = Fp16Matrix::quantize(&mat);
                            let ev = evaluate_operator(&mat, &op, &probes, 0)?;
                            builder.add_fp16(name.clone(), &op)?;
                            (
                                "QMF_FP16",
                                ev.relative_frobenius_error,
                                ev.mean_matvec_rel_error,
                            )
                        }
                        "BF16" => {
                            let op = Bf16Matrix::quantize(&mat);
                            let ev = evaluate_operator(&mat, &op, &probes, 0)?;
                            builder.add_bf16(name.clone(), &op)?;
                            (
                                "QMF_BF16",
                                ev.relative_frobenius_error,
                                ev.mean_matvec_rel_error,
                            )
                        }
                        "INT8_SYM" => {
                            let op = Int8SymMatrix::quantize(&mat, 64)?;
                            let ev = evaluate_operator(&mat, &op, &probes, 0)?;
                            builder.add_int8_sym(name.clone(), &op)?;
                            (
                                "QMF_INT8_SYM",
                                ev.relative_frobenius_error,
                                ev.mean_matvec_rel_error,
                            )
                        }
                        "INT4_ASYM_G32(Q4_1)" => {
                            let op = Int4AsymGroupMatrix::quantize(&mat, 32)?;
                            let ev = evaluate_operator(&mat, &op, &probes, 0)?;
                            builder.add_int4_asym(name.clone(), &op)?;
                            (
                                "QMF_INT4_ASYM",
                                ev.relative_frobenius_error,
                                ev.mean_matvec_rel_error,
                            )
                        }
                        "INT4_SYM_G128" => {
                            let op = Int4SymGroupMatrix::quantize(&mat, 128)?;
                            let ev = evaluate_operator(&mat, &op, &probes, 0)?;
                            builder.add_int4_sym(name.clone(), &op)?;
                            (
                                "QMF_INT4_SYM",
                                ev.relative_frobenius_error,
                                ev.mean_matvec_rel_error,
                            )
                        }
                        "SVD_INT8" => {
                            let r = include_svd_rank
                                .unwrap_or(8)
                                .max(1)
                                .min(mat.rows().min(mat.cols()));
                            let op = LowRankSvdMatrix::from_matrix(
                                &mat,
                                r,
                                FactorPrecision::Int8Sym,
                                seed + idx as u64,
                            )?;
                            let ev = evaluate_operator(&mat, &op, &probes, 0)?;
                            builder.add_low_rank_svd(name.clone(), &op)?;
                            (
                                "QMF_SVD",
                                ev.relative_frobenius_error,
                                ev.mean_matvec_rel_error,
                            )
                        }
                        _ => {
                            let op = Int4SymGroupMatrix::quantize(&mat, 32)?;
                            let ev = evaluate_operator(&mat, &op, &probes, 0)?;
                            builder.add_int4_sym(name.clone(), &op)?;
                            (
                                "QMF_INT4_SYM",
                                ev.relative_frobenius_error,
                                ev.mean_matvec_rel_error,
                            )
                        }
                    };
                    sum_frob += rel_frob;
                    sum_mv += mv_err;
                    mat_count += 1;
                    processed_tensors += 1;
                    control.report(processed_tensors, total_tensors, name);
                    tensor_summaries.push(TransformedTensorSummary {
                        name: name.clone(),
                        shape: vec![mat.rows(), mat.cols()],
                        representation: rep_name.to_owned(),
                        relative_frobenius_error: Some(rel_frob),
                        mean_matvec_rel_error: Some(mv_err),
                    });
                }
            }
        }
        TransformStrategy::RmqAffinityGated {
            rank,
            affinity_threshold,
            sparse_fraction,
            fallback_group_size,
        } => {
            type ClusterKey = (LayerRole, usize, usize);
            let mut groups: BTreeMap<ClusterKey, Vec<String>> = BTreeMap::new();
            for info in &matrix_infos {
                let (_, role) = classify_tensor_name(&info.name);
                groups
                    .entry((role, info.shape[0] as usize, info.shape[1] as usize))
                    .or_default()
                    .push(info.name.clone());
            }

            let mut rmq_clusters_selected = 0usize;
            let mut fallback_clusters = 0usize;
            for ((role, rows, cols), names) in groups {
                control.check_cancelled()?;
                control.report(
                    processed_tensors,
                    total_tensors,
                    &format!("RMQ gated cluster: {} ({}x{})", role.as_str(), rows, cols),
                );
                let cluster_mats: Vec<DenseMatrix> = names
                    .iter()
                    .map(|n| file.read_tensor_matrix(n, budget))
                    .collect::<Result<_, _>>()?;
                let probes = deterministic_probe_vectors(cols, 4, seed);

                // Never apply RMQ to embeddings or lm_head; gate other clusters by measured affinity.
                let eligible = cluster_mats.len() >= 2
                    && !matches!(role, LayerRole::Embedding | LayerRole::LmHead);
                let mean_affinity = if eligible {
                    let probe_r = (*rank).clamp(2, 16).min(rows.min(cols));
                    let svds: Vec<_> = cluster_mats
                        .iter()
                        .take(8)
                        .enumerate()
                        .map(|(i, m)| truncated_svd(m, probe_r, seed.wrapping_add(i as u64)))
                        .collect::<Result<_, _>>()?;
                    let mut sum_aff = 0.0_f64;
                    let mut pairs = 0usize;
                    for a in 0..svds.len() {
                        let va = svds[a].vt.transpose();
                        let uat = svds[a].u.transpose();
                        let vat = va.transpose();
                        for b in (a + 1)..svds.len() {
                            let vb = svds[b].vt.transpose();
                            let k = svds[a].u.cols().min(svds[b].u.cols()).max(1) as f64;
                            let left = (uat.matmul(&svds[b].u)?.frobenius_norm().powi(2) / k)
                                .clamp(0.0, 1.0);
                            let right =
                                (vat.matmul(&vb)?.frobenius_norm().powi(2) / k).clamp(0.0, 1.0);
                            sum_aff += 0.5 * (left + right);
                            pairs += 1;
                        }
                    }
                    if pairs > 0 {
                        sum_aff / pairs as f64
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                if eligible && mean_affinity >= *affinity_threshold {
                    rmq_clusters_selected += 1;
                    let refs: Vec<&DenseMatrix> = cluster_mats.iter().collect();
                    let r = (*rank).max(1).min(rows.min(cols));
                    let residual_kind = if *sparse_fraction > 0.0 {
                        RmqResidualKind::SparseTopFraction {
                            fraction: *sparse_fraction,
                        }
                    } else {
                        RmqResidualKind::None
                    };
                    let cluster = RmqSharedBasisCluster::fit(
                        &refs,
                        RmqSharedBasisConfig {
                            left_rank: r,
                            right_rank: r,
                            hooi_iters: 4,
                            basis_precision: FactorPrecision::Int8Sym,
                            core_precision: FactorPrecision::Fp16,
                            residual_kind,
                            seed,
                        },
                    )?;
                    for (local_i, (name, mat)) in names.iter().zip(&cluster_mats).enumerate() {
                        let op = cluster.layer(local_i).unwrap();
                        let ev = evaluate_operator(mat, &op, &probes, 0)?;
                        sum_frob += ev.relative_frobenius_error;
                        sum_mv += ev.mean_matvec_rel_error;
                        mat_count += 1;
                        tensor_summaries.push(TransformedTensorSummary {
                            name: name.clone(),
                            shape: vec![rows, cols],
                            representation: op.method_name().to_owned(),
                            relative_frobenius_error: Some(ev.relative_frobenius_error),
                            mean_matvec_rel_error: Some(ev.mean_matvec_rel_error),
                        });
                    }
                    let cluster_id = format!("{}_{rows}x{cols}", role.as_str());
                    builder.add_rmq_shared_basis_cluster(&cluster_id, &names, &cluster)?;
                } else {
                    fallback_clusters += 1;
                    let g = (*fallback_group_size).max(8);
                    for (name, mat) in names.iter().zip(&cluster_mats) {
                        let (rep_name, rel_frob, mv_err) =
                            if matches!(role, LayerRole::Embedding | LayerRole::LmHead) {
                                let op = Int8SymMatrix::quantize(mat, 64)?;
                                let ev = evaluate_operator(mat, &op, &probes, 0)?;
                                builder.add_int8_sym(name.clone(), &op)?;
                                (
                                    "QMF_INT8_SYM",
                                    ev.relative_frobenius_error,
                                    ev.mean_matvec_rel_error,
                                )
                            } else {
                                let op = Int4SymGroupMatrix::quantize(mat, g)?;
                                let ev = evaluate_operator(mat, &op, &probes, 0)?;
                                builder.add_int4_sym(name.clone(), &op)?;
                                (
                                    "QMF_INT4_SYM",
                                    ev.relative_frobenius_error,
                                    ev.mean_matvec_rel_error,
                                )
                            };
                        sum_frob += rel_frob;
                        sum_mv += mv_err;
                        mat_count += 1;
                        tensor_summaries.push(TransformedTensorSummary {
                            name: name.clone(),
                            shape: vec![rows, cols],
                            representation: rep_name.to_owned(),
                            relative_frobenius_error: Some(rel_frob),
                            mean_matvec_rel_error: Some(mv_err),
                        });
                    }
                }
            }
            builder.insert_metadata("rmq_clusters_selected", rmq_clusters_selected.to_string());
            builder.insert_metadata("fallback_clusters", fallback_clusters.to_string());
            builder.insert_metadata("affinity_threshold", format!("{affinity_threshold:.4}"));
        }
    }

    let archive = builder.build()?;
    let mut serialized = Vec::new();
    let qmf_bytes = archive.write_to(&mut serialized)?;
    let qmf_sha256 = sha256_hex(&serialized);
    let verified = QmfArchiveV1::from_bytes(&serialized)?;
    tensor_summaries.sort_by(|a, b| a.name.cmp(&b.name));

    let effective_qmf_bpp = if total_params > 0 {
        (qmf_bytes as f64 * 8.0) / total_params as f64
    } else {
        0.0
    };
    let report = ForgeReport {
        descriptor,
        model_id: model_id.to_owned(),
        random_seed: seed,
        source_file_bytes: meta.file_bytes,
        qmf_serialized_bytes: qmf_bytes,
        qmf_sha256,
        total_parameters: total_params,
        effective_qmf_bpp,
        compression_ratio_vs_source: meta.file_bytes as f64 / qmf_bytes.max(1) as f64,
        shared_latent_count: verified.manifest().shared_latents.len(),
        transformed_tensor_count: verified.manifest().tensors.len(),
        mean_rel_frobenius_error: if mat_count > 0 {
            sum_frob / mat_count as f64
        } else {
            0.0
        },
        mean_matvec_rel_error: if mat_count > 0 {
            sum_mv / mat_count as f64
        } else {
            0.0
        },
        elapsed_millis: start.elapsed().as_millis() as u64,
        tensors: tensor_summaries,
    };
    Ok((verified, report))
}

/// Independently compare an original Safetensors file against a QMF v1 artifact.
pub fn compare_safetensors_and_qmf<R: Read + Seek>(
    source_reader: R,
    qmf_bytes: &[u8],
    budget: &PayloadBudget,
    seed: u64,
) -> Result<CompareReport, ForgeError> {
    let mut source = SafetensorsFile::open(source_reader, budget)?;
    let meta = source.metadata().clone();
    let archive = QmfArchiveV1::from_bytes(qmf_bytes)?;
    let qmf_sha256 = sha256_hex(qmf_bytes);
    let mut latent_cache = QmfLatentCache::new();

    let mut total_params = 0_u64;
    let mut compared = 0_usize;
    let mut sum_frob = 0.0_f64;
    let mut max_frob = 0.0_f64;
    let mut sum_mv = 0.0_f64;
    let mut max_mv = 0.0_f64;

    for info in &meta.tensors {
        total_params = total_params.saturating_add(qml_model::tensor_element_count(info)?);
        if info.shape.len() == 2 && info.shape[0] > 0 && info.shape[1] > 0 {
            let orig = source.read_tensor_matrix(&info.name, budget)?;
            let op = archive.load_operator_cached(&info.name, &mut latent_cache)?;
            let probes = deterministic_probe_vectors(orig.cols(), 4, seed);
            let ev = evaluate_operator(&orig, &op, &probes, 0)?;
            sum_frob += ev.relative_frobenius_error;
            if ev.relative_frobenius_error > max_frob {
                max_frob = ev.relative_frobenius_error;
            }
            sum_mv += ev.mean_matvec_rel_error;
            if ev.mean_matvec_rel_error > max_mv {
                max_mv = ev.mean_matvec_rel_error;
            }
            compared += 1;
        }
    }

    Ok(CompareReport {
        qmf_model_id: archive.manifest().model_id.clone(),
        transformation: archive.manifest().transformation.clone(),
        qmf_sha256,
        tensors_compared: compared,
        source_file_bytes: meta.file_bytes,
        qmf_file_bytes: qmf_bytes.len(),
        total_parameters: total_params,
        effective_qmf_bpp: if total_params > 0 {
            (qmf_bytes.len() as f64 * 8.0) / total_params as f64
        } else {
            0.0
        },
        mean_rel_frobenius_error: if compared > 0 {
            sum_frob / compared as f64
        } else {
            0.0
        },
        max_rel_frobenius_error: max_frob,
        mean_matvec_rel_error: if compared > 0 {
            sum_mv / compared as f64
        } else {
            0.0
        },
        max_matvec_rel_error: max_mv,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use qml_scope::generate_synthetic_transformer_safetensors;
    use std::io::Cursor;

    #[test]
    fn forge_transforms_synthetic_transformer_with_rmq_basis_core_and_map() {
        let model_bytes = generate_synthetic_transformer_safetensors(4, 32, 64, 4, 77).unwrap();
        let budget = PayloadBudget::default();

        // 1. RMQ Shared-Basis transformation -> QMF v1
        let rmq_strategy = TransformStrategy::RmqSharedBasis {
            rank: 4,
            sparse_fraction: 0.01,
            basis_precision: FactorPrecision::Int8Sym,
            core_precision: FactorPrecision::Fp16,
        };
        let (archive, report) = forge_safetensors_to_qmf(
            Cursor::new(&model_bytes),
            "synthetic-4L",
            "generated-test",
            "mini_llama",
            &rmq_strategy,
            &budget,
            42,
        )
        .unwrap();

        assert_eq!(report.qmf_sha256.len(), 64);
        assert!(report.shared_latent_count >= 8);
        assert!(report.compression_ratio_vs_source > 2.0);

        let mut qmf_buf = Vec::new();
        archive.write_to(&mut qmf_buf).unwrap();
        let cmp =
            compare_safetensors_and_qmf(Cursor::new(&model_bytes), &qmf_buf, &budget, 42).unwrap();
        assert_eq!(cmp.qmf_sha256, report.qmf_sha256);
        assert_eq!(cmp.tensors_compared, 20);
        assert!((cmp.mean_matvec_rel_error - report.mean_matvec_rel_error).abs() < 1e-12);

        // 2. RMQ Shared-Core transformation -> QMF v1
        let core_strategy = TransformStrategy::RmqSharedCore {
            rank: 4,
            sparse_fraction: 0.01,
            factor_precision: FactorPrecision::Int8Sym,
        };
        let (core_archive, core_report) = forge_safetensors_to_qmf(
            Cursor::new(&model_bytes),
            "synthetic-4L-core",
            "generated-test",
            "mini_llama",
            &core_strategy,
            &budget,
            42,
        )
        .unwrap();
        assert!(core_report.shared_latent_count >= 4);
        let mut core_buf = Vec::new();
        core_archive.write_to(&mut core_buf).unwrap();
        let core_cmp =
            compare_safetensors_and_qmf(Cursor::new(&model_bytes), &core_buf, &budget, 42).unwrap();
        assert_eq!(core_cmp.tensors_compared, 20);
        assert!((core_cmp.mean_matvec_rel_error - core_report.mean_matvec_rel_error).abs() < 1e-12);

        // 3. MAP Adaptive transformation -> QMF v1
        let map_strategy = TransformStrategy::MapAdaptive {
            target_avg_bpp: 6.0,
            propagation_weight: 0.35,
            include_svd_rank: Some(4),
        };
        let (map_archive, map_report) = forge_safetensors_to_qmf(
            Cursor::new(&model_bytes),
            "synthetic-4L",
            "generated-test",
            "mini_llama",
            &map_strategy,
            &budget,
            42,
        )
        .unwrap();
        let payload_bpp =
            (map_archive.payload_bytes() as f64 * 8.0) / map_report.total_parameters as f64;
        assert!(payload_bpp <= 6.5);
        assert!(map_report.qmf_serialized_bytes < map_report.source_file_bytes as usize);
        assert!(map_report.mean_matvec_rel_error < 0.15);

        // 4. Affinity-Gated Hybrid RMQ: gates strongly correlated clusters to RMQ and falls back cleanly
        let gated_strategy = TransformStrategy::RmqAffinityGated {
            rank: 4,
            affinity_threshold: 0.50,
            sparse_fraction: 0.01,
            fallback_group_size: 32,
        };
        let (gated_archive, gated_report) = forge_safetensors_to_qmf(
            Cursor::new(&model_bytes),
            "synthetic-4L-gated",
            "generated-test",
            "mini_llama",
            &gated_strategy,
            &budget,
            42,
        )
        .unwrap();
        assert!(gated_report.shared_latent_count > 0);
        assert!(
            gated_report.mean_matvec_rel_error <= report.mean_matvec_rel_error + 1e-6,
            "affinity-gated RMQ ({}) should not be worse than blind RMQ ({})",
            gated_report.mean_matvec_rel_error,
            report.mean_matvec_rel_error
        );
        let mut gated_buf = Vec::new();
        gated_archive.write_to(&mut gated_buf).unwrap();
        let gated_cmp =
            compare_safetensors_and_qmf(Cursor::new(&model_bytes), &gated_buf, &budget, 42)
                .unwrap();
        assert_eq!(gated_cmp.tensors_compared, 20);
        assert!(
            (gated_cmp.mean_matvec_rel_error - gated_report.mean_matvec_rel_error).abs() < 1e-12
        );

        // High threshold (0.999) forces all clusters to Q4_0/INT8 fallback (zero shared latents).
        let strict_gate = TransformStrategy::RmqAffinityGated {
            rank: 4,
            affinity_threshold: 0.999,
            sparse_fraction: 0.01,
            fallback_group_size: 32,
        };
        let (_, fallback_report) = forge_safetensors_to_qmf(
            Cursor::new(&model_bytes),
            "synthetic-4L-fallback",
            "generated-test",
            "mini_llama",
            &strict_gate,
            &budget,
            42,
        )
        .unwrap();
        assert_eq!(fallback_report.shared_latent_count, 0);
        assert!(fallback_report.mean_matvec_rel_error < 0.15);
    }

    #[test]
    fn forge_honors_cooperative_cancellation() {
        let model_bytes = generate_synthetic_transformer_safetensors(2, 16, 32, 2, 11).unwrap();
        let budget = PayloadBudget::default();
        let cancel = Arc::new(AtomicBool::new(true));
        let control = ForgeControl {
            cancel,
            on_progress: None,
        };
        let err = forge_safetensors_to_qmf_with_control(
            Cursor::new(&model_bytes),
            "cancelled-model",
            "generated-test",
            "mini_llama",
            &TransformStrategy::Int4Sym { group_size: 32 },
            &budget,
            42,
            &control,
        );
        assert!(matches!(err, Err(ForgeError::Cancelled)));
    }
}
