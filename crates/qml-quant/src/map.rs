//! MAP — Morphological Adaptive Precision allocator.
//!
//! Assigns heterogeneous numerical/structural representations across layers
//! under an explicit global byte budget by combining:
//! - Layer sensitivity (Frobenius norm, crest factor / outlier extremity, and
//!   optional activation-weighted error proxy), and
//! - Downstream cross-layer error-propagation amplification (bounded spectral-norm
//!   product proxy across subsequent layers, ablatable via `propagation_weight`).
//!
//! Uses a deterministic greedy marginal distortion-per-byte allocator over the
//! discrete candidate menu and compares directly against uniform quantization at
//! matched total byte budgets.

use crate::baselines::{
    Bf16Matrix, FactorPrecision, Fp16Matrix, Int4AsymGroupMatrix, Int4SymGroupMatrix,
    Int8SymMatrix, LowRankSvdMatrix,
};
use crate::{deterministic_probe_vectors, evaluate_operator};
use qml_core::{DenseMatrix, MatrixError, truncated_svd};
use serde::Serialize;

/// Configuration for Morphological Adaptive Precision allocation.
#[derive(Debug, Clone)]
pub struct MapConfig {
    /// Target average bits per parameter across all layers (counting all overhead bytes).
    pub target_avg_bpp: f64,
    /// Weight on downstream spectral error-propagation factor (`0.0` disables cross-layer term).
    pub propagation_weight: f64,
    /// Candidate low-rank SVD rank to include in the heterogeneous menu (if `Some(r)`).
    pub include_svd_rank: Option<usize>,
    pub seed: u64,
}

impl Default for MapConfig {
    fn default() -> Self {
        Self {
            target_avg_bpp: 5.5,
            propagation_weight: 0.35,
            include_svd_rank: Some(8),
            seed: 0x4d41_5000_0001,
        }
    }
}

/// One candidate representation evaluated for a single layer.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayerCandidate {
    pub method: String,
    pub total_bytes: usize,
    pub effective_bpp: f64,
    pub relative_error: f64,
    pub matvec_rel_error: f64,
}

/// Final allocation selected by MAP for one layer.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayerAllocation {
    pub layer_name: String,
    pub layer_index: usize,
    pub param_count: usize,
    pub local_sensitivity: f64,
    pub propagation_factor: f64,
    pub combined_importance: f64,
    pub selected: LayerCandidate,
}

/// Complete MAP mixed-precision allocation report with uniform-baseline comparison.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MapAllocationReport {
    pub target_avg_bpp: f64,
    pub budget_bytes: usize,
    pub actual_total_bytes: usize,
    pub actual_avg_bpp: f64,
    pub total_parameters: usize,
    pub propagation_weight: f64,
    pub importance_weighted_error: f64,
    pub mean_matvec_rel_error: f64,
    pub uniform_int4_g32_bytes: usize,
    pub uniform_int4_g32_weighted_error: f64,
    pub uniform_int8_bytes: usize,
    pub uniform_int8_weighted_error: f64,
    pub allocations: Vec<LayerAllocation>,
}

/// Streaming per-layer profile computed in Pass 1 of MAP so dense matrices can be
/// dropped immediately before global budget allocation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MapLayerProfile {
    pub layer_name: String,
    pub layer_index: usize,
    pub param_count: usize,
    pub spectral_norm: f64,
    pub local_sensitivity: f64,
    pub uniform_int4_bytes: usize,
    pub uniform_int4_mv_err: f64,
    pub uniform_int8_bytes: usize,
    pub uniform_int8_mv_err: f64,
    pub candidates: Vec<LayerCandidate>,
}

/// Profile a single dense matrix for MAP allocation so the caller can drop `mat`
/// immediately afterwards (enabling constant-memory streaming MAP).
pub fn profile_layer_for_map(
    layer_name: &str,
    layer_index: usize,
    mat: &DenseMatrix,
    config: &MapConfig,
) -> Result<MapLayerProfile, MatrixError> {
    let svd1 = truncated_svd(mat, 1, config.seed.wrapping_add(layer_index as u64))?;
    let spectral_norm = svd1.singular_values[0];

    let frob = mat.frobenius_norm();
    let param_count = mat.rows() * mat.cols();
    let rms = (frob * frob / param_count as f64).sqrt();
    let absmax = mat
        .as_slice()
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    let crest = if rms > 1e-15 { absmax / rms } else { 1.0 };
    let local_sensitivity = frob.max(1e-6) * (1.0 + 0.15 * crest.ln_1p());

    let probes = deterministic_probe_vectors(mat.cols(), 4, config.seed + layer_index as u64);
    let mut candidates = Vec::new();

    macro_rules! eval_cand {
        ($op:expr) => {{
            let op = $op;
            let ev = evaluate_operator(mat, &op, &probes, 0)?;
            LayerCandidate {
                method: ev.method,
                total_bytes: ev.total_bytes,
                effective_bpp: ev.effective_bpp,
                relative_error: ev.relative_frobenius_error,
                matvec_rel_error: ev.mean_matvec_rel_error,
            }
        }};
    }

    if let Some(svd_r) = config.include_svd_rank {
        let r = svd_r.max(1).min(mat.rows().min(mat.cols()));
        candidates.push(eval_cand!(LowRankSvdMatrix::from_matrix(
            mat,
            r,
            FactorPrecision::Int8Sym,
            config.seed + layer_index as u64
        )?));
    }
    let q4_g128 = eval_cand!(Int4SymGroupMatrix::quantize(mat, 128)?);
    let q4_g32 = eval_cand!(Int4SymGroupMatrix::quantize(mat, 32)?);
    let q4_1_g32 = eval_cand!(Int4AsymGroupMatrix::quantize(mat, 32)?);
    let int8_g64 = eval_cand!(Int8SymMatrix::quantize(mat, 64)?);
    let bf16 = eval_cand!(Bf16Matrix::quantize(mat));
    let fp16 = eval_cand!(Fp16Matrix::quantize(mat));

    let uniform_int4_bytes = q4_g32.total_bytes;
    let uniform_int4_mv_err = q4_g32.matvec_rel_error;
    let uniform_int8_bytes = int8_g64.total_bytes;
    let uniform_int8_mv_err = int8_g64.matvec_rel_error;

    candidates.extend([q4_g128, q4_g32, q4_1_g32, int8_g64, bf16, fp16]);
    candidates.sort_by(|a, b| {
        a.total_bytes
            .cmp(&b.total_bytes)
            .then(a.matvec_rel_error.total_cmp(&b.matvec_rel_error))
    });

    Ok(MapLayerProfile {
        layer_name: layer_name.to_owned(),
        layer_index,
        param_count,
        spectral_norm,
        local_sensitivity,
        uniform_int4_bytes,
        uniform_int4_mv_err,
        uniform_int8_bytes,
        uniform_int8_mv_err,
        candidates,
    })
}

/// Solve global MAP allocation from precomputed streaming [`MapLayerProfile`] records
/// without holding any dense weight matrices in memory.
pub fn allocate_from_profiles(
    profiles: &[MapLayerProfile],
    config: &MapConfig,
) -> Result<MapAllocationReport, MatrixError> {
    if profiles.is_empty() {
        return Err(MatrixError::Empty);
    }
    let n_layers = profiles.len();
    let total_params: usize = profiles.iter().map(|p| p.param_count).sum();
    let budget_bytes = ((total_params as f64) * config.target_avg_bpp / 8.0).floor() as usize;

    let mut sorted_sigmas: Vec<f64> = profiles.iter().map(|p| p.spectral_norm).collect();
    sorted_sigmas.sort_by(|a, b| a.total_cmp(b));
    let median_sigma = sorted_sigmas[n_layers / 2].max(1e-6);

    let mut propagation_factors = vec![1.0_f64; n_layers];
    for (l, slot) in propagation_factors.iter_mut().enumerate() {
        let mut log_downstream = 0.0_f64;
        for p in profiles.iter().skip(l + 1) {
            let ratio = (p.spectral_norm / median_sigma).clamp(0.25, 4.0);
            log_downstream += ratio.ln().max(0.0);
        }
        let depth_factor = ((n_layers - 1 - l) as f64) / (n_layers.max(1) as f64);
        *slot = 1.0 + config.propagation_weight * (depth_factor + 0.5 * log_downstream);
    }

    let combined_importance: Vec<f64> = profiles
        .iter()
        .zip(&propagation_factors)
        .map(|(prof, &prop)| prof.local_sensitivity * prop)
        .collect();
    let total_importance: f64 = combined_importance.iter().sum::<f64>().max(1e-12);

    let mut uniform_int4_bytes = 0_usize;
    let mut uniform_int4_err = 0.0_f64;
    let mut uniform_int8_bytes = 0_usize;
    let mut uniform_int8_err = 0.0_f64;
    for (idx, p) in profiles.iter().enumerate() {
        let w = combined_importance[idx] / total_importance;
        uniform_int4_bytes += p.uniform_int4_bytes;
        uniform_int4_err += w * p.uniform_int4_mv_err;
        uniform_int8_bytes += p.uniform_int8_bytes;
        uniform_int8_err += w * p.uniform_int8_mv_err;
    }

    let mut chosen: Vec<usize> = vec![0; n_layers];
    let mut current_bytes: usize = profiles.iter().map(|p| p.candidates[0].total_bytes).sum();

    loop {
        let mut best_upgrade: Option<(usize, usize, f64)> = None;
        for l in 0..n_layers {
            let cur = &profiles[l].candidates[chosen[l]];
            let weight = combined_importance[l] / total_importance;
            for (cand_idx, cand) in profiles[l]
                .candidates
                .iter()
                .enumerate()
                .skip(chosen[l] + 1)
            {
                let delta_bytes = cand.total_bytes.saturating_sub(cur.total_bytes);
                if current_bytes.saturating_add(delta_bytes) > budget_bytes {
                    continue;
                }
                let err_gain = cur.matvec_rel_error - cand.matvec_rel_error;
                if err_gain <= 1e-12 {
                    continue;
                }
                let score = if delta_bytes == 0 {
                    f64::INFINITY
                } else {
                    (weight * err_gain) / (delta_bytes as f64)
                };
                if best_upgrade.is_none_or(|(_, _, best_s)| score > best_s) {
                    best_upgrade = Some((l, cand_idx, score));
                }
            }
        }
        let Some((layer_idx, cand_idx, _)) = best_upgrade else {
            break;
        };
        let old_bytes = profiles[layer_idx].candidates[chosen[layer_idx]].total_bytes;
        let new_bytes = profiles[layer_idx].candidates[cand_idx].total_bytes;
        current_bytes = current_bytes - old_bytes + new_bytes;
        chosen[layer_idx] = cand_idx;
    }

    let mut allocations = Vec::with_capacity(n_layers);
    let mut weighted_error = 0.0_f64;
    let mut sum_mv_error = 0.0_f64;
    for l in 0..n_layers {
        let selected = profiles[l].candidates[chosen[l]].clone();
        weighted_error += (combined_importance[l] / total_importance) * selected.matvec_rel_error;
        sum_mv_error += selected.matvec_rel_error;
        allocations.push(LayerAllocation {
            layer_name: profiles[l].layer_name.clone(),
            layer_index: profiles[l].layer_index,
            param_count: profiles[l].param_count,
            local_sensitivity: profiles[l].local_sensitivity,
            propagation_factor: propagation_factors[l],
            combined_importance: combined_importance[l],
            selected,
        });
    }

    Ok(MapAllocationReport {
        target_avg_bpp: config.target_avg_bpp,
        budget_bytes,
        actual_total_bytes: current_bytes,
        actual_avg_bpp: (current_bytes as f64 * 8.0) / total_params as f64,
        total_parameters: total_params,
        propagation_weight: config.propagation_weight,
        importance_weighted_error: weighted_error,
        mean_matvec_rel_error: sum_mv_error / n_layers as f64,
        uniform_int4_g32_bytes: uniform_int4_bytes,
        uniform_int4_g32_weighted_error: uniform_int4_err,
        uniform_int8_bytes,
        uniform_int8_weighted_error: uniform_int8_err,
        allocations,
    })
}

/// Run Morphological Adaptive Precision allocation across an ordered sequence of layers.
pub fn allocate_adaptive_precision(
    layers: &[(&str, &DenseMatrix)],
    config: &MapConfig,
) -> Result<MapAllocationReport, MatrixError> {
    if layers.is_empty() {
        return Err(MatrixError::Empty);
    }
    let profiles: Vec<MapLayerProfile> = layers
        .iter()
        .enumerate()
        .map(|(idx, (name, mat))| profile_layer_for_map(name, idx, mat, config))
        .collect::<Result<_, _>>()?;
    allocate_from_profiles(&profiles, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qml_core::DeterministicRng;

    #[test]
    fn map_prioritizes_high_sensitivity_outlier_layer_under_tight_budget() {
        let mut rng = DeterministicRng::new(314);
        // Layer 0: sensitive high-norm + strong outliers
        let mut l0_data: Vec<f64> = (0..32 * 32).map(|_| rng.next_normal() * 0.4).collect();
        l0_data[0] = 18.0;
        l0_data[17] = -15.0;
        let l0 = DenseMatrix::from_flat(32, 32, l0_data).unwrap();

        // Layers 1..3: low-norm, smooth, low-rank-friendly matrices
        let u = DenseMatrix::from_flat(32, 4, (0..128).map(|_| rng.next_normal() * 0.1).collect())
            .unwrap();
        let v = DenseMatrix::from_flat(4, 32, (0..128).map(|_| rng.next_normal() * 0.1).collect())
            .unwrap();
        let l1 = u.matmul(&v).unwrap();
        let l2 = l1.scale(0.8).unwrap();
        let l3 = l1.scale(0.6).unwrap();

        let layers = [
            ("layer.0.q_proj", &l0),
            ("layer.1.q_proj", &l1),
            ("layer.2.q_proj", &l2),
            ("layer.3.q_proj", &l3),
        ];

        // Budget near Q4_0 (~4.8 bpp): MAP should allocate higher precision to layer 0
        // and lower bits (SVD/INT4) to layers 1..3, beating uniform Q4_0 weighted error!
        let cfg = MapConfig {
            target_avg_bpp: 4.8,
            propagation_weight: 0.4,
            include_svd_rank: Some(4),
            seed: 42,
        };
        let report = allocate_adaptive_precision(&layers, &cfg).unwrap();
        assert!(report.actual_total_bytes <= report.budget_bytes);
        assert!(
            report.allocations[0].selected.effective_bpp
                > report.allocations[3].selected.effective_bpp,
            "layer 0 ({:.2} bpp) should get more precision than layer 3 ({:.2} bpp)",
            report.allocations[0].selected.effective_bpp,
            report.allocations[3].selected.effective_bpp
        );
        assert!(
            report.importance_weighted_error < report.uniform_int4_g32_weighted_error,
            "MAP weighted error {:.4} should beat uniform Q4_0 {:.4}",
            report.importance_weighted_error,
            report.uniform_int4_g32_weighted_error
        );
    }

    #[test]
    fn map_matched_exact_byte_budget_against_uniform_q4_0() {
        let mut rng = DeterministicRng::new(2026);
        let mut l0_data: Vec<f64> = (0..32 * 32).map(|_| rng.next_normal() * 0.4).collect();
        l0_data[0] = 20.0;
        let l0 = DenseMatrix::from_flat(32, 32, l0_data).unwrap();
        let u = DenseMatrix::from_flat(32, 3, (0..96).map(|_| rng.next_normal() * 0.1).collect())
            .unwrap();
        let v = DenseMatrix::from_flat(3, 32, (0..96).map(|_| rng.next_normal() * 0.1).collect())
            .unwrap();
        let l1 = u.matmul(&v).unwrap();
        let l2 = l1.scale(0.9).unwrap();
        let l3 = l1.scale(0.7).unwrap();
        let layers = [("l0", &l0), ("l1", &l1), ("l2", &l2), ("l3", &l3)];

        // Exact matched budget: Q4_0 uses 4.65625 bpp on 32x32 matrices -> set MAP target <= Q4_0!
        let cfg = MapConfig {
            target_avg_bpp: 4.65,
            propagation_weight: 0.35,
            include_svd_rank: Some(4),
            seed: 42,
        };
        let report = allocate_adaptive_precision(&layers, &cfg).unwrap();
        assert!(
            report.actual_total_bytes <= report.uniform_int4_g32_bytes,
            "MAP bytes ({}) must be <= uniform Q4_0 bytes ({}) for matched-budget comparison",
            report.actual_total_bytes,
            report.uniform_int4_g32_bytes
        );
        assert!(
            report.importance_weighted_error < report.uniform_int4_g32_weighted_error,
            "At matched byte budget, MAP ({:.4}) should beat uniform Q4_0 ({:.4})",
            report.importance_weighted_error,
            report.uniform_int4_g32_weighted_error
        );
    }
}
