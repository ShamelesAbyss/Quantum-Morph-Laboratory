//! Quantization baselines, complete byte accounting, RMQ, and MAP engines.
//!
//! Every representation in this crate implements [`LinearOperator`], supporting:
//! - Direct compressed matrix-vector multiplication without full dense reconstruction
//! - Exact complete storage accounting (payload + scales/zero-points + factors/indices + headers)
//! - Reproducible reconstruction and matvec error evaluation

pub mod baselines;
pub mod gptq;
pub mod map;
pub mod resonance;
pub mod rmq;

pub use baselines::{
    Bf16Matrix, FactorPrecision, Fp16Matrix, Int4AsymGroupMatrix, Int4SymGroupMatrix,
    Int8AsymMatrix, Int8SymMatrix, LowRankSvdMatrix, QuantizedFactor, SparseResidual,
};
pub use gptq::{
    DEFAULT_HESSIAN_BLOCK_SIZE, LayerCalibrationAccumulator, LayerCalibrationStats,
    ModelCalibrationProfile, activation_output_rel_error, fit_echo_calibrated,
    fit_weighted_affine_group, quantize_int2_gptq, quantize_int3_gptq, quantize_int4_sym_gptq,
    quantize_ternary_gptq, rotate_hessian_block_hadamard,
};
pub use map::{
    LayerAllocation, LayerCandidate, MapAllocationReport, MapConfig, MapLayerProfile,
    allocate_adaptive_precision, allocate_from_profiles, profile_layer_for_map,
};
pub use resonance::{
    BehaviorImportanceMap, ConsumerHardwareFit, EchoBaseOperator, EchoCorrectedMatrix,
    HadamardSpec, Int2AsymGroupMatrix, Int3AsymGroupMatrix, ResonanceAllocationPlan,
    ResonanceCandidateEval, ResonanceCandidateKind, ResonanceProfile, ResonanceTensorAllocation,
    ResonanceTensorProfile, TernaryPackedMatrix, VectorQuantMatrix, allocate_resonance_map_b,
    apply_hadamard_forward, apply_hadamard_inverse, estimate_consumer_hardware_fit, fwht_in_place,
    hadamard_signs, profile_tensor_for_resonance, profile_tensor_for_resonance_calibrated,
};
pub use rmq::{
    RmqAblationReport, RmqComparisonRow, RmqResidualKind, RmqSharedBasisCluster,
    RmqSharedBasisConfig, RmqSharedBasisLayer, RmqSharedCoreCluster, RmqSharedCoreLayer,
    compare_rmq_vs_baselines,
};

use qml_core::{DenseMatrix, DeterministicRng, MatrixError};
use serde::Serialize;
use std::time::Instant;

/// Exact accounting of every byte required to serialize a representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StorageBreakdown {
    /// Raw packed codes or quantized element values.
    pub payload_bytes: usize,
    /// Scales, zero-points, and per-group min/max metadata.
    pub scale_metadata_bytes: usize,
    /// Shared basis amortized share, low-rank factors, or sparse coordinate indices.
    pub structure_bytes: usize,
    /// Fixed descriptor / shape / encoding header bytes.
    pub header_bytes: usize,
}

impl StorageBreakdown {
    pub fn total_bytes(&self) -> usize {
        self.payload_bytes
            .saturating_add(self.scale_metadata_bytes)
            .saturating_add(self.structure_bytes)
            .saturating_add(self.header_bytes)
    }

    pub fn effective_bits_per_param(&self, param_count: usize) -> f64 {
        if param_count == 0 {
            return 0.0;
        }
        (self.total_bytes() as f64 * 8.0) / (param_count as f64)
    }
}

/// Common interface for direct compressed linear operators (`y = Ŵ x`).
pub trait LinearOperator {
    fn method_name(&self) -> &'static str;
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn matvec(&self, vector: &[f64]) -> Result<Vec<f64>, MatrixError>;
    fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError>;
    fn storage_breakdown(&self) -> StorageBreakdown;
}

/// Comprehensive evaluation metrics comparing an operator against its dense reference.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvaluationMetrics {
    pub method: String,
    pub rows: usize,
    pub cols: usize,
    pub param_count: usize,
    pub storage: StorageBreakdown,
    pub total_bytes: usize,
    pub effective_bpp: f64,
    pub compression_ratio_vs_fp32: f64,
    pub compression_ratio_vs_fp16: f64,
    pub frobenius_error: f64,
    pub relative_frobenius_error: f64,
    pub mse: f64,
    pub rmse: f64,
    pub max_abs_error: f64,
    pub snr_db: f64,
    pub mean_matvec_rel_error: f64,
    pub max_matvec_rel_error: f64,
    pub encode_micros: u64,
    pub matvec_micros: u64,
}

/// Generate deterministic probe vectors (mix of standard Gaussian and outlier-channel activations).
pub fn deterministic_probe_vectors(cols: usize, count: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = DeterministicRng::new(seed);
    let mut vectors = Vec::with_capacity(count);
    for idx in 0..count {
        let mut v: Vec<f64> = (0..cols).map(|_| rng.next_normal()).collect();
        // Inject heavy-tailed activation outlier channel on half the probes to mirror LLM activations.
        if idx % 2 == 1 && !v.is_empty() {
            let channel = rng.next_usize(cols);
            v[channel] *= 8.0;
        }
        vectors.push(v);
    }
    vectors
}

/// Evaluate a [`LinearOperator`] against its original dense matrix on reconstruction and matvec probes.
pub fn evaluate_operator<O: LinearOperator + ?Sized>(
    original: &DenseMatrix,
    operator: &O,
    probe_vectors: &[Vec<f64>],
    encode_micros: u64,
) -> Result<EvaluationMetrics, MatrixError> {
    let reconstructed = operator.reconstruct_dense()?;
    let diff = original.sub(&reconstructed)?;
    let frob_err = diff.frobenius_norm();
    let orig_frob = original.frobenius_norm();
    let rel_frob_err = if orig_frob > 1e-15 {
        frob_err / orig_frob
    } else {
        frob_err
    };

    let param_count = original.rows() * original.cols();
    let mse = (frob_err * frob_err) / (param_count as f64);
    let rmse = mse.sqrt();
    let max_abs_error = diff
        .as_slice()
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    let snr_db = if frob_err <= 1e-15 {
        f64::INFINITY
    } else if orig_frob <= 1e-15 {
        0.0
    } else {
        20.0 * (orig_frob / frob_err).log10()
    };

    let start_mv = Instant::now();
    let mut sum_mv_err = 0.0_f64;
    let mut max_mv_err = 0.0_f64;
    for probe in probe_vectors {
        let expected = original.matvec(probe)?;
        let actual = operator.matvec(probe)?;
        let err_norm = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let ref_norm = expected.iter().map(|a| a * a).sum::<f64>().sqrt();
        let rel = if ref_norm > 1e-15 {
            err_norm / ref_norm
        } else {
            err_norm
        };
        sum_mv_err += rel;
        if rel > max_mv_err {
            max_mv_err = rel;
        }
    }
    let matvec_micros = start_mv.elapsed().as_micros() as u64;
    let mean_mv_err = if probe_vectors.is_empty() {
        0.0
    } else {
        sum_mv_err / probe_vectors.len() as f64
    };

    let storage = operator.storage_breakdown();
    let total_bytes = storage.total_bytes();
    let effective_bpp = storage.effective_bits_per_param(param_count);
    let fp32_bytes = (param_count * 4) as f64;
    let fp16_bytes = (param_count * 2) as f64;

    Ok(EvaluationMetrics {
        method: operator.method_name().to_owned(),
        rows: original.rows(),
        cols: original.cols(),
        param_count,
        storage,
        total_bytes,
        effective_bpp,
        compression_ratio_vs_fp32: fp32_bytes / total_bytes.max(1) as f64,
        compression_ratio_vs_fp16: fp16_bytes / total_bytes.max(1) as f64,
        frobenius_error: frob_err,
        relative_frobenius_error: rel_frob_err,
        mse,
        rmse,
        max_abs_error,
        snr_db,
        mean_matvec_rel_error: mean_mv_err,
        max_matvec_rel_error: max_mv_err,
        encode_micros,
        matvec_micros,
    })
}

/// Run the full conventional quantization baseline suite on a single dense matrix.
pub fn benchmark_conventional_baselines(
    matrix: &DenseMatrix,
    svd_rank: usize,
    seed: u64,
) -> Result<Vec<EvaluationMetrics>, MatrixError> {
    let probes = deterministic_probe_vectors(matrix.cols(), 8, seed);
    let mut results = Vec::new();

    macro_rules! run_case {
        ($build:expr) => {{
            let t0 = Instant::now();
            let op = $build?;
            let enc = t0.elapsed().as_micros() as u64;
            results.push(evaluate_operator(matrix, &op, &probes, enc)?);
        }};
    }

    run_case!(Ok::<_, MatrixError>(Fp16Matrix::quantize(matrix)));
    run_case!(Ok::<_, MatrixError>(Bf16Matrix::quantize(matrix)));
    run_case!(Int8SymMatrix::quantize(matrix, 64));
    run_case!(Int8AsymMatrix::quantize(matrix, 64));
    run_case!(Int4SymGroupMatrix::quantize(matrix, 32));
    run_case!(Int4AsymGroupMatrix::quantize(matrix, 32));
    run_case!(Int4SymGroupMatrix::quantize(matrix, 128));
    let clamped_rank = svd_rank.max(1).min(matrix.rows().min(matrix.cols()));
    run_case!(LowRankSvdMatrix::from_matrix(
        matrix,
        clamped_rank,
        FactorPrecision::Fp16,
        seed
    ));
    run_case!(LowRankSvdMatrix::from_matrix(
        matrix,
        clamped_rank,
        FactorPrecision::Int8Sym,
        seed
    ));

    Ok(results)
}
