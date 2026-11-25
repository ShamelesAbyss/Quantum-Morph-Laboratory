//! Q-Scope — Neural Model Observatory for Quantum Morph Laboratory.
//!
//! Performs streaming, memory-bounded mathematical analysis of Safetensors
//! model tensors:
//! - Distributional statistics (mean, std, skewness, excess kurtosis, norms)
//! - Sparsity & outlier detection (exact/near-zero fractions, 3σ/6σ outliers, crest factor)
//! - Histogram & binned Shannon entropy estimation
//! - 2D weight matrix singular-value spectrum, stable rank, and Roy–Vetterli effective rank
//! - Cross-layer mathematical relationship analysis (cosine similarity and
//!   Grassmannian principal-subspace affinity across transformer layers)
//!
//! All analysis is read-only and never modifies source model files.

use qml_core::{DenseMatrix, DeterministicRng, SvdResult, truncated_svd};
use qml_model::{
    DType, PayloadBudget, PayloadError, SafetensorsFile, encode_safetensors, f32_to_f16,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::io::{Read, Seek};

/// Configuration controlling memory budgets and mathematical analysis depth in Q-Scope.
#[derive(Debug, Clone)]
pub struct ScopeConfig {
    pub payload_budget: PayloadBudget,
    pub histogram_bins: usize,
    pub compute_svd: bool,
    pub svd_max_rank: usize,
    /// Maximum 2D matrix element count eligible for SVD analysis (default: 262,144 elements).
    pub svd_max_elements: u64,
    pub compute_cross_layer: bool,
    /// Maximum number of tensors per role compared pairwise in cross-layer analysis.
    pub max_cross_layer_members: usize,
    pub seed: u64,
}

impl Default for ScopeConfig {
    fn default() -> Self {
        Self {
            payload_budget: PayloadBudget::default(),
            histogram_bins: 32,
            compute_svd: true,
            svd_max_rank: 16,
            svd_max_elements: 262_144,
            compute_cross_layer: true,
            max_cross_layer_members: 16,
            seed: 0x514d_4c53_434f_5045,
        }
    }
}

/// Architectural role inferred from common transformer tensor naming conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerRole {
    QProj,
    KProj,
    VProj,
    OProj,
    QkvProj,
    GateProj,
    UpProj,
    DownProj,
    Embedding,
    LmHead,
    Norm,
    Bias,
    Other,
}

impl LayerRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::QProj => "q_proj",
            Self::KProj => "k_proj",
            Self::VProj => "v_proj",
            Self::OProj => "o_proj",
            Self::QkvProj => "qkv_proj",
            Self::GateProj => "gate_proj",
            Self::UpProj => "up_proj",
            Self::DownProj => "down_proj",
            Self::Embedding => "embedding",
            Self::LmHead => "lm_head",
            Self::Norm => "norm",
            Self::Bias => "bias",
            Self::Other => "other",
        }
    }
}

/// Infer `(layer_index, LayerRole)` from standard Hugging Face / GGML tensor names.
pub fn classify_tensor_name(name: &str) -> (Option<usize>, LayerRole) {
    let lower = name.to_ascii_lowercase();
    let layer_index = extract_layer_index(&lower);

    if lower.ends_with(".bias") || lower == "bias" {
        return (layer_index, LayerRole::Bias);
    }
    if lower.contains("embed_tokens")
        || lower.contains("wte")
        || lower.contains("token_embd")
        || lower.contains("word_embeddings")
    {
        return (layer_index, LayerRole::Embedding);
    }
    if lower.contains("lm_head") || lower.contains("output.weight") {
        return (layer_index, LayerRole::LmHead);
    }
    if lower.contains("norm") || lower.contains("ln_") || lower.contains("ln_f") {
        return (layer_index, LayerRole::Norm);
    }
    if lower.contains("qkv") || lower.contains("c_attn") {
        return (layer_index, LayerRole::QkvProj);
    }
    if lower.contains("q_proj") || lower.contains("attn_q") || lower.contains(".query.") {
        return (layer_index, LayerRole::QProj);
    }
    if lower.contains("k_proj") || lower.contains("attn_k") || lower.contains(".key.") {
        return (layer_index, LayerRole::KProj);
    }
    if lower.contains("v_proj") || lower.contains("attn_v") || lower.contains(".value.") {
        return (layer_index, LayerRole::VProj);
    }
    if lower.contains("o_proj")
        || lower.contains("attn_output")
        || lower.contains("out_proj")
        || lower.contains("c_proj")
    {
        return (layer_index, LayerRole::OProj);
    }
    if lower.contains("gate_proj") || lower.contains("ffn_gate") || lower.contains("w1") {
        return (layer_index, LayerRole::GateProj);
    }
    if lower.contains("up_proj")
        || lower.contains("ffn_up")
        || lower.contains("c_fc")
        || lower.contains("w3")
    {
        return (layer_index, LayerRole::UpProj);
    }
    if lower.contains("down_proj") || lower.contains("ffn_down") || lower.contains("w2") {
        return (layer_index, LayerRole::DownProj);
    }
    (layer_index, LayerRole::Other)
}

fn extract_layer_index(lower_name: &str) -> Option<usize> {
    let tokens: Vec<&str> = lower_name.split('.').collect();
    for window in tokens.windows(2) {
        if matches!(window[0], "layers" | "layer" | "h" | "blocks" | "blk")
            && let Ok(idx) = window[1].parse::<usize>()
        {
            return Some(idx);
        }
    }
    None
}

/// Streaming numerical statistics for one tensor.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TensorStats {
    pub finite_count: u64,
    pub nan_count: u64,
    pub inf_count: u64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub absmax: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub skewness: f64,
    pub excess_kurtosis: f64,
    pub l1_norm: f64,
    pub l2_norm: f64,
    pub rms: f64,
    pub exact_zero_fraction: f64,
    pub near_zero_1e4_fraction: f64,
    pub near_zero_1e3_fraction: f64,
    pub near_zero_1e2_fraction: f64,
    pub outlier_3sigma_count: u64,
    pub outlier_6sigma_count: u64,
    /// `absmax / rms` (measures peak-to-RMS outlier extremity; 0.0 for all-zero tensors).
    pub crest_factor: f64,
    /// Binned Shannon entropy estimate in bits over `[min, max]`.
    pub shannon_entropy_bits: f64,
    pub histogram_counts: Vec<u64>,
    pub histogram_edges: Vec<f64>,
    /// Heuristic quantization susceptibility proxy combining crest factor and excess kurtosis.
    /// Higher values indicate heavier tails/outliers that tend to challenge uniform quantization.
    pub quantization_susceptibility_score: f64,
}

/// Spectral structure summary for a 2D weight matrix.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MatrixStructureReport {
    pub rows: usize,
    pub cols: usize,
    pub computed_rank: usize,
    pub singular_values: Vec<f64>,
    pub spectral_norm: f64,
    pub frobenius_norm: f64,
    pub stable_rank: f64,
    pub effective_rank: f64,
    pub condition_number_est: Option<f64>,
    pub rank_90_energy: usize,
    pub rank_95_energy: usize,
    pub rank_99_energy: usize,
    pub rank1_relative_error: f64,
    pub captured_energy_ratio: f64,
    pub is_full_spectrum: bool,
}

/// Complete Q-Scope record for a single tensor.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TensorScopeReport {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub data_bytes: u64,
    pub layer_index: Option<usize>,
    pub role: LayerRole,
    pub stats: Option<TensorStats>,
    pub stats_skipped_reason: Option<String>,
    pub matrix_structure: Option<MatrixStructureReport>,
    pub matrix_structure_skipped_reason: Option<String>,
}

/// Mathematical relationship between two compatible same-role matrices across layers.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CrossLayerRelation {
    pub role: LayerRole,
    pub layer_a: Option<usize>,
    pub layer_b: Option<usize>,
    pub tensor_a: String,
    pub tensor_b: String,
    pub shape: [usize; 2],
    pub weight_cosine_similarity: f64,
    pub left_subspace_affinity: f64,
    pub right_subspace_affinity: f64,
    pub shared_structure_score: f64,
}

/// Aggregate model-level observatory summary.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScopeSummary {
    pub total_tensors: usize,
    pub analyzed_tensors: usize,
    pub svd_analyzed_matrices: usize,
    pub total_parameters: u64,
    pub total_payload_bytes: u64,
    pub dtype_parameter_counts: BTreeMap<String, u64>,
    pub role_parameter_counts: BTreeMap<String, u64>,
    pub mean_cross_layer_shared_score: Option<f64>,
    pub max_quantization_susceptibility_tensor: Option<String>,
    pub max_quantization_susceptibility_score: Option<f64>,
}

/// Top-level machine-readable Q-Scope observatory report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelScopeReport {
    pub file_bytes: u64,
    pub header_bytes: u64,
    pub data_bytes: u64,
    pub file_metadata: BTreeMap<String, String>,
    pub summary: ScopeSummary,
    pub tensors: Vec<TensorScopeReport>,
    pub cross_layer_relations: Vec<CrossLayerRelation>,
}

/// Analyze a Safetensors stream with Q-Scope using bounded, streaming tensor access.
pub fn analyze_safetensors<R: Read + Seek>(
    reader: R,
    config: &ScopeConfig,
) -> Result<ModelScopeReport, PayloadError> {
    let mut file = SafetensorsFile::open(reader, &config.payload_budget)?;
    let meta = file.metadata().clone();

    let mut tensor_reports = Vec::with_capacity(meta.tensors.len());
    let mut dtype_params: BTreeMap<String, u64> = BTreeMap::new();
    let mut role_params: BTreeMap<String, u64> = BTreeMap::new();
    let mut total_parameters = 0_u64;
    let mut analyzed_tensors = 0_usize;
    let mut svd_analyzed_matrices = 0_usize;

    // Store compact SVD/matrix handles grouped by (role, rows, cols) for cross-layer analysis.
    type GroupKey = (LayerRole, usize, usize);
    struct RoleCandidate {
        name: String,
        layer_index: Option<usize>,
        matrix: DenseMatrix,
        svd: SvdResult,
    }
    let mut role_groups: BTreeMap<GroupKey, Vec<RoleCandidate>> = BTreeMap::new();

    for info in &meta.tensors {
        let element_count = qml_model::tensor_element_count(info)?;
        total_parameters = total_parameters.saturating_add(element_count);
        *dtype_params.entry(info.dtype.clone()).or_insert(0) += element_count;
        let (layer_index, role) = classify_tensor_name(&info.name);
        *role_params.entry(role.as_str().to_owned()).or_insert(0) += element_count;

        let dtype_opt = DType::parse(&info.dtype);
        let (stats, stats_skipped_reason) = match dtype_opt {
            Some(dt) if dt.supports_f64_decode() => {
                let s = compute_streaming_tensor_stats(
                    &mut file,
                    &info.name,
                    config.payload_budget.stream_chunk_elements,
                    config.histogram_bins.max(4),
                )?;
                analyzed_tensors += 1;
                (Some(s), None)
            }
            Some(dt) => (
                None,
                Some(format!("dtype {dt} does not support scalar f64 decoding")),
            ),
            None => (None, Some(format!("unknown dtype {:?}", info.dtype))),
        };

        let mut matrix_structure = None;
        let mut matrix_structure_skipped_reason = None;

        if config.compute_svd && info.shape.len() == 2 {
            let rows = info.shape[0];
            let cols = info.shape[1];
            if rows == 0 || cols == 0 {
                matrix_structure_skipped_reason = Some("zero-sized matrix".to_owned());
            } else if element_count > config.svd_max_elements {
                matrix_structure_skipped_reason = Some(format!(
                    "element count {element_count} exceeds svd_max_elements {}",
                    config.svd_max_elements
                ));
            } else if stats
                .as_ref()
                .is_some_and(|s| s.nan_count > 0 || s.inf_count > 0)
            {
                matrix_structure_skipped_reason =
                    Some("tensor contains non-finite values".to_owned());
            } else if dtype_opt.is_some_and(DType::supports_f64_decode) {
                match file.read_tensor_matrix(&info.name, &config.payload_budget) {
                    Ok(matrix) => {
                        let min_dim = matrix.rows().min(matrix.cols());
                        let target_rank = config.svd_max_rank.max(1).min(min_dim);
                        let svd = truncated_svd(&matrix, target_rank, config.seed)?;
                        let metrics = svd.spectrum_metrics();
                        svd_analyzed_matrices += 1;
                        matrix_structure = Some(MatrixStructureReport {
                            rows: matrix.rows(),
                            cols: matrix.cols(),
                            computed_rank: svd.rank(),
                            singular_values: metrics.singular_values,
                            spectral_norm: metrics.spectral_norm,
                            frobenius_norm: metrics.frobenius_norm,
                            stable_rank: metrics.stable_rank,
                            effective_rank: metrics.effective_rank,
                            condition_number_est: metrics.condition_number_est,
                            rank_90_energy: metrics.rank_90_energy,
                            rank_95_energy: metrics.rank_95_energy,
                            rank_99_energy: metrics.rank_99_energy,
                            rank1_relative_error: metrics.rank1_relative_error,
                            captured_energy_ratio: metrics.captured_energy_ratio,
                            is_full_spectrum: metrics.is_full_spectrum,
                        });

                        if config.compute_cross_layer {
                            let group = role_groups
                                .entry((role, matrix.rows(), matrix.cols()))
                                .or_default();
                            if group.len() < config.max_cross_layer_members {
                                group.push(RoleCandidate {
                                    name: info.name.clone(),
                                    layer_index,
                                    matrix,
                                    svd,
                                });
                            }
                        }
                    }
                    Err(e) => {
                        matrix_structure_skipped_reason = Some(e.to_string());
                    }
                }
            }
        }

        tensor_reports.push(TensorScopeReport {
            name: info.name.clone(),
            dtype: info.dtype.clone(),
            shape: info.shape.clone(),
            element_count,
            data_bytes: info.data_bytes,
            layer_index,
            role,
            stats,
            stats_skipped_reason,
            matrix_structure,
            matrix_structure_skipped_reason,
        });
    }

    // Cross-layer relationship analysis across same-role, same-shape matrices.
    let mut cross_layer_relations = Vec::new();
    for ((role, rows, cols), mut candidates) in role_groups {
        if candidates.len() < 2 {
            continue;
        }
        candidates.sort_by(|a, b| a.layer_index.cmp(&b.layer_index).then(a.name.cmp(&b.name)));
        for i in 0..candidates.len() {
            for j in (i + 1)..candidates.len() {
                let a = &candidates[i];
                let b = &candidates[j];
                let cos = a.matrix.cosine_similarity(&b.matrix)?;
                let left_aff = a.svd.left_subspace_affinity(&b.svd)?;
                let right_aff = a.svd.right_subspace_affinity(&b.svd)?;
                let shared_score = 0.5 * (left_aff + right_aff);
                cross_layer_relations.push(CrossLayerRelation {
                    role,
                    layer_a: a.layer_index,
                    layer_b: b.layer_index,
                    tensor_a: a.name.clone(),
                    tensor_b: b.name.clone(),
                    shape: [rows, cols],
                    weight_cosine_similarity: cos,
                    left_subspace_affinity: left_aff,
                    right_subspace_affinity: right_aff,
                    shared_structure_score: shared_score,
                });
            }
        }
    }

    let mean_cross_layer_shared_score = if cross_layer_relations.is_empty() {
        None
    } else {
        Some(
            cross_layer_relations
                .iter()
                .map(|r| r.shared_structure_score)
                .sum::<f64>()
                / cross_layer_relations.len() as f64,
        )
    };

    let (max_susc_tensor, max_susc_score) = tensor_reports
        .iter()
        .filter_map(|t| {
            t.stats
                .as_ref()
                .map(|s| (t.name.clone(), s.quantization_susceptibility_score))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map_or((None, None), |(n, s)| (Some(n), Some(s)));

    let summary = ScopeSummary {
        total_tensors: meta.tensors.len(),
        analyzed_tensors,
        svd_analyzed_matrices,
        total_parameters,
        total_payload_bytes: meta.data_bytes,
        dtype_parameter_counts: dtype_params,
        role_parameter_counts: role_params,
        mean_cross_layer_shared_score,
        max_quantization_susceptibility_tensor: max_susc_tensor,
        max_quantization_susceptibility_score: max_susc_score,
    };

    Ok(ModelScopeReport {
        file_bytes: meta.file_bytes,
        header_bytes: meta.header_bytes,
        data_bytes: meta.data_bytes,
        file_metadata: meta.metadata,
        summary,
        tensors: tensor_reports,
        cross_layer_relations,
    })
}

/// Two-pass streaming statistics computation using constant working memory.
pub fn compute_streaming_tensor_stats<R: Read + Seek>(
    file: &mut SafetensorsFile<R>,
    name: &str,
    chunk_elements: usize,
    histogram_bins: usize,
) -> Result<TensorStats, PayloadError> {
    let mut finite_count = 0_u64;
    let mut nan_count = 0_u64;
    let mut inf_count = 0_u64;
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    let mut absmax = 0.0_f64;
    let mut l1_norm = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    let mut exact_zeros = 0_u64;
    let mut near_1e4 = 0_u64;
    let mut near_1e3 = 0_u64;
    let mut near_1e2 = 0_u64;

    // Pass 1: Welford online mean/variance, norms, range, sparsity.
    file.stream_tensor_f64(name, chunk_elements, |chunk| {
        for &x in chunk {
            if x.is_nan() {
                nan_count += 1;
                continue;
            }
            if !x.is_finite() {
                inf_count += 1;
                continue;
            }
            finite_count += 1;
            let ax = x.abs();
            if x < min_val {
                min_val = x;
            }
            if x > max_val {
                max_val = x;
            }
            if ax > absmax {
                absmax = ax;
            }
            l1_norm += ax;
            sum_sq += x * x;
            if x == 0.0 {
                exact_zeros += 1;
            }
            if ax <= 1e-4 {
                near_1e4 += 1;
            }
            if ax <= 1e-3 {
                near_1e3 += 1;
            }
            if ax <= 1e-2 {
                near_1e2 += 1;
            }
            let delta = x - mean;
            mean += delta / finite_count as f64;
            let delta2 = x - mean;
            m2 += delta * delta2;
        }
    })?;

    if finite_count == 0 {
        return Ok(TensorStats {
            finite_count: 0,
            nan_count,
            inf_count,
            min: None,
            max: None,
            absmax: 0.0,
            mean: 0.0,
            std_dev: 0.0,
            skewness: 0.0,
            excess_kurtosis: 0.0,
            l1_norm: 0.0,
            l2_norm: 0.0,
            rms: 0.0,
            exact_zero_fraction: 0.0,
            near_zero_1e4_fraction: 0.0,
            near_zero_1e3_fraction: 0.0,
            near_zero_1e2_fraction: 0.0,
            outlier_3sigma_count: 0,
            outlier_6sigma_count: 0,
            crest_factor: 0.0,
            shannon_entropy_bits: 0.0,
            histogram_counts: vec![0; histogram_bins],
            histogram_edges: vec![0.0; histogram_bins + 1],
            quantization_susceptibility_score: 0.0,
        });
    }

    let n = finite_count as f64;
    let variance = (m2 / n).max(0.0);
    let std_dev = variance.sqrt();
    let l2_norm = sum_sq.sqrt();
    let rms = (sum_sq / n).sqrt();
    let crest_factor = if rms > 1e-15 { absmax / rms } else { 0.0 };

    // Pass 2: higher moments (skewness, kurtosis), outlier counts, histogram.
    let mut sum_m3 = 0.0_f64;
    let mut sum_m4 = 0.0_f64;
    let mut outlier_3sigma = 0_u64;
    let mut outlier_6sigma = 0_u64;
    let mut hist_counts = vec![0_u64; histogram_bins];
    let range = max_val - min_val;
    let mut hist_edges = Vec::with_capacity(histogram_bins + 1);
    for b in 0..=histogram_bins {
        hist_edges.push(min_val + range * (b as f64 / histogram_bins as f64));
    }

    file.stream_tensor_f64(name, chunk_elements, |chunk| {
        for &x in chunk {
            if !x.is_finite() {
                continue;
            }
            let d = x - mean;
            let d2 = d * d;
            sum_m3 += d2 * d;
            sum_m4 += d2 * d2;
            if std_dev > 1e-15 {
                let z = d.abs() / std_dev;
                if z > 3.0 {
                    outlier_3sigma += 1;
                }
                if z > 6.0 {
                    outlier_6sigma += 1;
                }
            }
            let bin = if range <= 1e-15 {
                0
            } else {
                let frac = ((x - min_val) / range).clamp(0.0, 1.0);
                let idx = (frac * histogram_bins as f64) as usize;
                idx.min(histogram_bins - 1)
            };
            hist_counts[bin] += 1;
        }
    })?;

    let skewness = if std_dev > 1e-15 {
        (sum_m3 / n) / (std_dev * std_dev * std_dev)
    } else {
        0.0
    };
    let excess_kurtosis = if variance > 1e-30 {
        (sum_m4 / n) / (variance * variance) - 3.0
    } else {
        0.0
    };

    let shannon_entropy_bits: f64 = hist_counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum();

    // Transparent susceptibility proxy: combines crest factor (peak/RMS) and positive kurtosis.
    let quantization_susceptibility_score = crest_factor + 0.25 * excess_kurtosis.max(0.0).ln_1p();

    Ok(TensorStats {
        finite_count,
        nan_count,
        inf_count,
        min: Some(min_val),
        max: Some(max_val),
        absmax,
        mean,
        std_dev,
        skewness,
        excess_kurtosis,
        l1_norm,
        l2_norm,
        rms,
        exact_zero_fraction: exact_zeros as f64 / n,
        near_zero_1e4_fraction: near_1e4 as f64 / n,
        near_zero_1e3_fraction: near_1e3 as f64 / n,
        near_zero_1e2_fraction: near_1e2 as f64 / n,
        outlier_3sigma_count: outlier_3sigma,
        outlier_6sigma_count: outlier_6sigma,
        crest_factor,
        shannon_entropy_bits,
        histogram_counts: hist_counts,
        histogram_edges: hist_edges,
        quantization_susceptibility_score,
    })
}

/// Generate a deterministic, synthetic multi-layer transformer Safetensors fixture.
///
/// Contains correlated cross-layer attention projections (with shared latent subspaces
/// plus layer-specific perturbations and sparse outliers) and higher-rank MLP layers,
/// allowing reproducible end-to-end validation of Q-Scope, quantization baselines, RMQ,
/// and MAP without shipping pretrained weights.
pub fn generate_synthetic_transformer_safetensors(
    num_layers: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
    shared_rank: usize,
    seed: u64,
) -> Result<Vec<u8>, String> {
    let mut rng = DeterministicRng::new(seed);
    let r = shared_rank.max(1).min(hidden_dim);

    // Shared latent left/right bases for attention projections across layers.
    let shared_u: Vec<f64> = (0..hidden_dim * r)
        .map(|_| rng.next_normal() / (hidden_dim as f64).sqrt())
        .collect();
    let shared_v: Vec<f64> = (0..r * hidden_dim)
        .map(|_| rng.next_normal() / (hidden_dim as f64).sqrt())
        .collect();

    let mut owned_tensors: Vec<(String, DType, Vec<u64>, Vec<u8>)> = Vec::new();

    for layer in 0..num_layers {
        // q_proj and k_proj share dominant subspace + small layer noise + 1% outliers (F32).
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            let mut core = vec![0.0_f64; r * r];
            for i in 0..r {
                core[i * r + i] = (r - i) as f64 * (0.8 + 0.1 * layer as f64);
            }
            let mut raw = Vec::with_capacity(hidden_dim * hidden_dim * 4);
            for row in 0..hidden_dim {
                for col in 0..hidden_dim {
                    let mut val = 0.0_f64;
                    for a in 0..r {
                        val += shared_u[row * r + a]
                            * core[a * r + a]
                            * shared_v[a * hidden_dim + col];
                    }
                    val += rng.next_normal() * 0.03;
                    // Inject occasional structured outlier in layer 0 q_proj to test outlier detection.
                    if layer == 0 && proj == "q_proj" && row == 0 && col < 2 {
                        val += 4.5;
                    }
                    raw.extend_from_slice(&(val as f32).to_le_bytes());
                }
            }
            owned_tensors.push((
                format!("model.layers.{layer}.self_attn.{proj}.weight"),
                DType::F32,
                vec![hidden_dim as u64, hidden_dim as u64],
                raw,
            ));
        }

        // MLP up_proj stored in F16 to exercise multi-dtype decoding.
        let mut up_raw = Vec::with_capacity(intermediate_dim * hidden_dim * 2);
        for _ in 0..(intermediate_dim * hidden_dim) {
            let val = (rng.next_normal() * 0.15) as f32;
            up_raw.extend_from_slice(&f32_to_f16(val).to_le_bytes());
        }
        owned_tensors.push((
            format!("model.layers.{layer}.mlp.up_proj.weight"),
            DType::F16,
            vec![intermediate_dim as u64, hidden_dim as u64],
            up_raw,
        ));

        // LayerNorm scale vector (1D F32).
        let mut norm_raw = Vec::with_capacity(hidden_dim * 4);
        for _ in 0..hidden_dim {
            let val = (1.0 + rng.next_normal() * 0.05) as f32;
            norm_raw.extend_from_slice(&val.to_le_bytes());
        }
        owned_tensors.push((
            format!("model.layers.{layer}.input_layernorm.weight"),
            DType::F32,
            vec![hidden_dim as u64],
            norm_raw,
        ));
    }

    let refs: Vec<(&str, DType, &[u64], &[u8])> = owned_tensors
        .iter()
        .map(|(n, dt, sh, b)| (n.as_str(), *dt, sh.as_slice(), b.as_slice()))
        .collect();
    let mut meta = BTreeMap::new();
    meta.insert("qml_fixture".to_owned(), "synthetic_transformer".to_owned());
    meta.insert("seed".to_owned(), seed.to_string());
    encode_safetensors(&refs, &meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn classifies_standard_transformer_tensor_names() {
        assert_eq!(
            classify_tensor_name("model.layers.12.self_attn.q_proj.weight"),
            (Some(12), LayerRole::QProj)
        );
        assert_eq!(
            classify_tensor_name("blk.3.ffn_down.weight"),
            (Some(3), LayerRole::DownProj)
        );
        assert_eq!(
            classify_tensor_name("model.embed_tokens.weight"),
            (None, LayerRole::Embedding)
        );
        assert_eq!(
            classify_tensor_name("model.layers.0.input_layernorm.weight"),
            (Some(0), LayerRole::Norm)
        );
    }

    #[test]
    fn qscope_detects_outliers_low_rank_and_cross_layer_subspace_affinity() {
        let bytes = generate_synthetic_transformer_safetensors(3, 32, 64, 4, 42).unwrap();
        let cfg = ScopeConfig {
            svd_max_rank: 8,
            ..ScopeConfig::default()
        };
        let report = analyze_safetensors(Cursor::new(bytes), &cfg).unwrap();

        assert_eq!(report.summary.total_tensors, 18); // 3 layers * (4 attn + 1 up + 1 norm)
        assert_eq!(report.summary.analyzed_tensors, 18);
        assert_eq!(report.summary.svd_analyzed_matrices, 15); // 3 layers * 5 2D matrices

        // Layer 0 q_proj has injected outliers -> highest susceptibility score.
        assert_eq!(
            report
                .summary
                .max_quantization_susceptibility_tensor
                .as_deref(),
            Some("model.layers.0.self_attn.q_proj.weight")
        );
        let q0 = report
            .tensors
            .iter()
            .find(|t| t.name == "model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        let q0_stats = q0.stats.as_ref().unwrap();
        assert!(q0_stats.outlier_6sigma_count >= 2);
        assert!(q0_stats.crest_factor > 10.0);

        // Layer 1 q_proj is low-rank (rank 4 dominant + small noise) -> small rank_90_energy.
        let q1 = report
            .tensors
            .iter()
            .find(|t| t.name == "model.layers.1.self_attn.q_proj.weight")
            .unwrap();
        let q1_struct = q1.matrix_structure.as_ref().unwrap();
        assert!(q1_struct.rank_90_energy <= 4);

        // Cross-layer q_proj relations between layers 1 and 2 should show strong subspace affinity!
        let q12 = report
            .cross_layer_relations
            .iter()
            .find(|r| r.role == LayerRole::QProj && r.layer_a == Some(1) && r.layer_b == Some(2))
            .expect("q_proj layer 1-2 relation");
        assert!(
            q12.shared_structure_score > 0.45,
            "expected shared subspace affinity, got {}",
            q12.shared_structure_score
        );
    }
}
