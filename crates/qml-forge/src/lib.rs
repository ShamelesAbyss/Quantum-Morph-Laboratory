//! Q-Forge — Modular Model Transformation Engine for Quantum Morph Laboratory.
//!
//! Transforms immutable Safetensors models into experimental QMF v1 artifacts
//! with full parameter provenance, complete on-disk byte accounting, and
//! direct-execution verification against the original weights.

use qml_core::DenseMatrix;
use qml_format::{QmfArchiveV1, QmfError, QmfV1Builder};
use qml_model::{PayloadBudget, PayloadError, SafetensorsFile};
use qml_quant::{
    Bf16Matrix, FactorPrecision, Fp16Matrix, Int4AsymGroupMatrix, Int4SymGroupMatrix,
    Int8SymMatrix, LinearOperator, LowRankSvdMatrix, MapConfig, RmqResidualKind,
    RmqSharedBasisCluster, RmqSharedBasisConfig, allocate_adaptive_precision,
    deterministic_probe_vectors, evaluate_operator,
};
use qml_scope::{LayerRole, classify_tensor_name};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::io::{Read, Seek};
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
    /// representations (`U_shared * C_l * V_sharedᵀ + SparseResidual`); unclustered 2D
    /// matrices fall back to `Int8Sym(64)` and 1D vectors remain `DenseF32`.
    RmqSharedBasis {
        rank: usize,
        sparse_fraction: f64,
        basis_precision: FactorPrecision,
        core_precision: FactorPrecision,
    },
    /// Runs Morphological Adaptive Precision (MAP) across all 2D tensors under a global
    /// target average bits-per-parameter budget.
    MapAdaptive {
        target_avg_bpp: f64,
        propagation_weight: f64,
        include_svd_rank: Option<usize>,
    },
}

impl TransformStrategy {
    pub fn descriptor(&self) -> TransformDescriptor {
        match self {
            Self::Fp16 => TransformDescriptor {
                name: "fp16",
                assumptions: "IEEE-754 binary16 range [-65504, 65504]",
                deterministic: true,
                maturity: "conventional-baseline",
            },
            Self::Bf16 => TransformDescriptor {
                name: "bf16",
                assumptions: "BFloat16 dynamic range with 7-bit mantissa",
                deterministic: true,
                maturity: "conventional-baseline",
            },
            Self::Int8Sym { .. } => TransformDescriptor {
                name: "int8_sym",
                assumptions: "Symmetric groupwise uniform quantization with f16 scales",
                deterministic: true,
                maturity: "conventional-baseline",
            },
            Self::Int4Sym { .. } => TransformDescriptor {
                name: "int4_sym",
                assumptions: "GGML Q4_0-style symmetric packed 4-bit blocks with f16 scales",
                deterministic: true,
                maturity: "conventional-baseline",
            },
            Self::Int4Asym { .. } => TransformDescriptor {
                name: "int4_asym",
                assumptions: "GGML Q4_1-style affine packed 4-bit blocks with f16 scale+min",
                deterministic: true,
                maturity: "conventional-baseline",
            },
            Self::LowRankSvd { .. } => TransformDescriptor {
                name: "low_rank_svd",
                assumptions: "Per-layer truncated SVD with symmetrically scaled quantized factors",
                deterministic: true,
                maturity: "conventional-baseline",
            },
            Self::RmqSharedBasis { .. } => TransformDescriptor {
                name: "rmq_shared_basis",
                assumptions: "Cross-layer 2D-HOOI shared orthonormal bases + layer cores + optional sparse residual; assumes cross-layer subspace overlap",
                deterministic: true,
                maturity: "experimental-proposal",
            },
            Self::MapAdaptive { .. } => TransformDescriptor {
                name: "map_adaptive",
                assumptions: "Sensitivity + downstream spectral error-propagation greedy byte-budget allocator",
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
    Message(String),
}

impl fmt::Display for ForgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Payload(e) => write!(f, "model payload error: {e}"),
            Self::Qmf(e) => write!(f, "QMF error: {e}"),
            Self::Matrix(e) => write!(f, "matrix error: {e}"),
            Self::Message(m) => write!(f, "{m}"),
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
pub fn forge_safetensors_to_qmf<R: Read + Seek>(
    reader: R,
    model_id: &str,
    source_license: &str,
    architecture: &str,
    strategy: &TransformStrategy,
    budget: &PayloadBudget,
    seed: u64,
) -> Result<(QmfArchiveV1, ForgeReport), ForgeError> {
    let start = Instant::now();
    let mut file = SafetensorsFile::open(reader, budget)?;
    let meta = file.metadata().clone();
    let descriptor = strategy.descriptor();

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

    // Load 2D matrices and 1D/other vectors within the explicit PayloadBudget.
    let mut matrices: Vec<(String, DenseMatrix)> = Vec::new();
    let mut vectors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut total_params = 0_u64;

    for info in &meta.tensors {
        let count = qml_model::tensor_element_count(info)?;
        total_params = total_params.saturating_add(count);
        if info.shape.len() == 2 && info.shape[0] > 0 && info.shape[1] > 0 {
            let mat = file.read_tensor_matrix(&info.name, budget)?;
            matrices.push((info.name.clone(), mat));
        } else {
            let vals = file.read_tensor_f64(&info.name, budget)?;
            let shape_usize: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
            let f32_vals: Vec<f32> = vals.into_iter().map(|v| v as f32).collect();
            vectors.push((info.name.clone(), shape_usize, f32_vals));
        }
    }

    for (name, shape, vals) in &vectors {
        builder.add_dense_f32(name.clone(), shape.clone(), vals)?;
    }

    match strategy {
        TransformStrategy::Fp16 => {
            for (name, mat) in &matrices {
                builder.add_fp16(name.clone(), &Fp16Matrix::quantize(mat))?;
            }
        }
        TransformStrategy::Bf16 => {
            for (name, mat) in &matrices {
                builder.add_bf16(name.clone(), &Bf16Matrix::quantize(mat))?;
            }
        }
        TransformStrategy::Int8Sym { group_size } => {
            for (name, mat) in &matrices {
                builder.add_int8_sym(name.clone(), &Int8SymMatrix::quantize(mat, *group_size)?)?;
            }
        }
        TransformStrategy::Int4Sym { group_size } => {
            for (name, mat) in &matrices {
                builder.add_int4_sym(
                    name.clone(),
                    &Int4SymGroupMatrix::quantize(mat, *group_size)?,
                )?;
            }
        }
        TransformStrategy::Int4Asym { group_size } => {
            for (name, mat) in &matrices {
                builder.add_int4_asym(
                    name.clone(),
                    &Int4AsymGroupMatrix::quantize(mat, *group_size)?,
                )?;
            }
        }
        TransformStrategy::LowRankSvd { rank, precision } => {
            for (idx, (name, mat)) in matrices.iter().enumerate() {
                let r = (*rank).max(1).min(mat.rows().min(mat.cols()));
                let svd = LowRankSvdMatrix::from_matrix(
                    mat,
                    r,
                    *precision,
                    seed.wrapping_add(idx as u64),
                )?;
                builder.add_low_rank_svd(name.clone(), &svd)?;
            }
        }
        TransformStrategy::RmqSharedBasis {
            rank,
            sparse_fraction,
            basis_precision,
            core_precision,
        } => {
            // Group 2D matrices by (LayerRole, rows, cols); clusters of size >= 2 share bases!
            type ClusterKey = (LayerRole, usize, usize);
            let mut groups: BTreeMap<ClusterKey, Vec<usize>> = BTreeMap::new();
            for (idx, (name, mat)) in matrices.iter().enumerate() {
                let (_, role) = classify_tensor_name(name);
                groups
                    .entry((role, mat.rows(), mat.cols()))
                    .or_default()
                    .push(idx);
            }
            for ((role, rows, cols), indices) in groups {
                if indices.len() >= 2 {
                    let refs: Vec<&DenseMatrix> = indices.iter().map(|&i| &matrices[i].1).collect();
                    let names: Vec<String> =
                        indices.iter().map(|&i| matrices[i].0.clone()).collect();
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
                    let cluster_id = format!("{}_{rows}x{cols}", role.as_str());
                    builder.add_rmq_shared_basis_cluster(&cluster_id, &names, &cluster)?;
                } else {
                    for idx in indices {
                        let (name, mat) = &matrices[idx];
                        builder.add_int8_sym(name.clone(), &Int8SymMatrix::quantize(mat, 64)?)?;
                    }
                }
            }
        }
        TransformStrategy::MapAdaptive {
            target_avg_bpp,
            propagation_weight,
            include_svd_rank,
        } => {
            if !matrices.is_empty() {
                let pairs: Vec<(&str, &DenseMatrix)> =
                    matrices.iter().map(|(n, m)| (n.as_str(), m)).collect();
                let map_cfg = MapConfig {
                    target_avg_bpp: *target_avg_bpp,
                    propagation_weight: *propagation_weight,
                    include_svd_rank: *include_svd_rank,
                    seed,
                };
                let plan = allocate_adaptive_precision(&pairs, &map_cfg)?;
                for (idx, alloc) in plan.allocations.iter().enumerate() {
                    let (name, mat) = &matrices[idx];
                    match alloc.selected.method.as_str() {
                        "FP16" => builder.add_fp16(name.clone(), &Fp16Matrix::quantize(mat))?,
                        "BF16" => builder.add_bf16(name.clone(), &Bf16Matrix::quantize(mat))?,
                        "INT8_SYM" => builder
                            .add_int8_sym(name.clone(), &Int8SymMatrix::quantize(mat, 64)?)?,
                        "INT4_ASYM_G32(Q4_1)" => builder.add_int4_asym(
                            name.clone(),
                            &Int4AsymGroupMatrix::quantize(mat, 32)?,
                        )?,
                        "INT4_SYM_G128" => builder
                            .add_int4_sym(name.clone(), &Int4SymGroupMatrix::quantize(mat, 128)?)?,
                        "SVD_INT8" => {
                            let r = include_svd_rank
                                .unwrap_or(8)
                                .max(1)
                                .min(mat.rows().min(mat.cols()));
                            builder.add_low_rank_svd(
                                name.clone(),
                                &LowRankSvdMatrix::from_matrix(
                                    mat,
                                    r,
                                    FactorPrecision::Int8Sym,
                                    seed + idx as u64,
                                )?,
                            )?;
                        }
                        _ => builder
                            .add_int4_sym(name.clone(), &Int4SymGroupMatrix::quantize(mat, 32)?)?,
                    }
                }
            }
        }
    }

    let archive = builder.build()?;
    let mut serialized = Vec::new();
    let qmf_bytes = archive.write_to(&mut serialized)?;
    // Reload from exact serialized bytes to guarantee 100% round-trip fidelity in reported metrics.
    let verified = QmfArchiveV1::from_bytes(&serialized)?;

    let mut tensor_summaries = Vec::with_capacity(verified.manifest().tensors.len());
    let mut sum_frob = 0.0_f64;
    let mut sum_mv = 0.0_f64;
    let mut mat_count = 0_usize;

    for (name, mat) in &matrices {
        let op = verified.load_operator(name)?;
        let probes = deterministic_probe_vectors(mat.cols(), 6, seed);
        let ev = evaluate_operator(mat, &op, &probes, 0)?;
        sum_frob += ev.relative_frobenius_error;
        sum_mv += ev.mean_matvec_rel_error;
        mat_count += 1;
        tensor_summaries.push(TransformedTensorSummary {
            name: name.clone(),
            shape: vec![mat.rows(), mat.cols()],
            representation: op.method_name().to_owned(),
            relative_frobenius_error: Some(ev.relative_frobenius_error),
            mean_matvec_rel_error: Some(ev.mean_matvec_rel_error),
        });
    }
    for (name, shape, _) in &vectors {
        tensor_summaries.push(TransformedTensorSummary {
            name: name.clone(),
            shape: shape.clone(),
            representation: "QMF_DENSE_F32".to_owned(),
            relative_frobenius_error: Some(0.0),
            mean_matvec_rel_error: None,
        });
    }
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
            let op = archive.load_operator(&info.name)?;
            let probes = deterministic_probe_vectors(orig.cols(), 6, seed);
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
    fn forge_transforms_synthetic_transformer_with_rmq_and_map_and_verifies_via_compare() {
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

        assert!(report.shared_latent_count >= 8); // U+Vt for each of the 4 attn roles + up_proj
        assert!(
            report.compression_ratio_vs_source > 2.0,
            "expected >2x complete QMF file compression, got {:.2}x",
            report.compression_ratio_vs_source
        );

        let mut qmf_buf = Vec::new();
        archive.write_to(&mut qmf_buf).unwrap();
        let cmp =
            compare_safetensors_and_qmf(Cursor::new(&model_bytes), &qmf_buf, &budget, 42).unwrap();
        assert_eq!(cmp.tensors_compared, 20); // 4 layers * 5 2D matrices
        assert!((cmp.mean_matvec_rel_error - report.mean_matvec_rel_error).abs() < 1e-12);

        // 2. MAP Adaptive transformation -> QMF v1
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
        // Verify payload strictly obeys MAP budget while total file accounts for JSON manifest overhead.
        let payload_bpp =
            (map_archive.payload_bytes() as f64 * 8.0) / map_report.total_parameters as f64;
        assert!(
            payload_bpp <= 6.5,
            "expected MAP payload bpp <= 6.5, got {payload_bpp:.3}"
        );
        assert!(
            map_report.qmf_serialized_bytes < map_report.source_file_bytes as usize,
            "expected complete QMF file ({}) smaller than source ({})",
            map_report.qmf_serialized_bytes,
            map_report.source_file_bytes
        );
        assert!(map_report.mean_matvec_rel_error < 0.15);
    }
}
