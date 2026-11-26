//! Q-Behavior — Mechanistic Activation, Representational Geometry & Causal Intervention Observatory.
//!
//! Captures intermediate transformer hidden-state trajectories across every layer,
//! measures layer-by-layer cosine similarity, relative L2 activation divergence,
//! and Representational Similarity Analysis (RSA) between a reference and transformed
//! model, and executes strictly reversible runtime causal interventions (attention
//! ablation, MLP ablation, residual scaling) without ever modifying stored QMF weights.

use qml_runtime::{KvCache, LayerIntervention, QCoreModel, RuntimeError};
use serde::{Deserialize, Serialize};

/// Per-layer activation comparison between reference and candidate models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerBehaviorMetrics {
    pub layer_index: usize,
    pub mean_cosine_similarity: f64,
    pub min_cosine_similarity: f64,
    pub mean_rel_l2_divergence: f64,
    pub max_rel_l2_divergence: f64,
    /// Representational Similarity Analysis (Pearson correlation of pairwise token
    /// cosine similarity matrices at this layer; `1.0` when identical, `0.0` if < 3 tokens).
    pub rsa_correlation: f64,
}

/// Complete behavioral and mechanistic comparison between a reference and transformed model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BehaviorComparisonReport {
    pub sequence_len: usize,
    pub num_layers: usize,
    pub layers: Vec<LayerBehaviorMetrics>,
    pub peak_divergence_layer: usize,
    pub peak_rel_l2_divergence: f64,
    pub final_layer_cosine: f64,
    pub mean_logit_kl_divergence: f64,
    pub top1_token_agreement: f64,
}

/// Causal effect report for a strictly reversible runtime [`LayerIntervention`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionReport {
    pub intervention: LayerIntervention,
    pub sequence_len: usize,
    pub layers: Vec<LayerBehaviorMetrics>,
    pub mean_logit_kl_divergence: f64,
    pub top1_flip_rate: f64,
}

/// Compare layer-by-layer hidden activations, RSA geometry, and output logits between
/// a reference and a candidate (quantized/transformed) [`QCoreModel`] on identical tokens.
pub fn compare_model_behavior(
    reference: &QCoreModel,
    candidate: &QCoreModel,
    token_sequence: &[usize],
) -> Result<BehaviorComparisonReport, RuntimeError> {
    if token_sequence.is_empty() {
        return Err(RuntimeError::EmptyPrompt);
    }
    let num_layers = reference
        .config()
        .num_layers
        .min(candidate.config().num_layers);
    let mut ref_cache = KvCache::new(reference.config().num_layers);
    let mut cand_cache = KvCache::new(candidate.config().num_layers);

    let mut ref_acts: Vec<Vec<Vec<f64>>> = (0..num_layers).map(|_| Vec::new()).collect();
    let mut cand_acts: Vec<Vec<Vec<f64>>> = (0..num_layers).map(|_| Vec::new()).collect();
    let mut kl_sum = 0.0_f64;
    let mut top1_matches = 0usize;

    for (pos, &tok) in token_sequence.iter().enumerate() {
        let r_trace = reference.forward_token_with_trace(tok, pos, &mut ref_cache, None)?;
        let c_trace = candidate.forward_token_with_trace(tok, pos, &mut cand_cache, None)?;
        for l in 0..num_layers {
            ref_acts[l].push(r_trace.layer_hidden[l].clone());
            cand_acts[l].push(c_trace.layer_hidden[l].clone());
        }
        kl_sum += logit_kl_divergence(&r_trace.logits, &c_trace.logits);
        if argmax(&r_trace.logits) == argmax(&c_trace.logits) {
            top1_matches += 1;
        }
    }

    let layers = compute_layer_metrics(&ref_acts, &cand_acts);
    let (peak_divergence_layer, peak_rel_l2_divergence) = layers
        .iter()
        .max_by(|a, b| {
            a.mean_rel_l2_divergence
                .total_cmp(&b.mean_rel_l2_divergence)
        })
        .map_or((0, 0.0), |m| (m.layer_index, m.mean_rel_l2_divergence));
    let final_layer_cosine = layers.last().map_or(1.0, |m| m.mean_cosine_similarity);

    Ok(BehaviorComparisonReport {
        sequence_len: token_sequence.len(),
        num_layers,
        layers,
        peak_divergence_layer,
        peak_rel_l2_divergence,
        final_layer_cosine,
        mean_logit_kl_divergence: kl_sum / token_sequence.len() as f64,
        top1_token_agreement: top1_matches as f64 / token_sequence.len() as f64,
    })
}

/// Measure the exact causal effect of a strictly reversible [`LayerIntervention`]
/// on a single model without modifying any stored weights.
pub fn run_intervention_study(
    model: &QCoreModel,
    token_sequence: &[usize],
    intervention: LayerIntervention,
) -> Result<InterventionReport, RuntimeError> {
    if token_sequence.is_empty() {
        return Err(RuntimeError::EmptyPrompt);
    }
    let num_layers = model.config().num_layers;
    let mut base_cache = KvCache::new(num_layers);
    let mut int_cache = KvCache::new(num_layers);

    let mut base_acts: Vec<Vec<Vec<f64>>> = (0..num_layers).map(|_| Vec::new()).collect();
    let mut int_acts: Vec<Vec<Vec<f64>>> = (0..num_layers).map(|_| Vec::new()).collect();
    let mut kl_sum = 0.0_f64;
    let mut flips = 0usize;

    for (pos, &tok) in token_sequence.iter().enumerate() {
        let b_trace = model.forward_token_with_trace(tok, pos, &mut base_cache, None)?;
        let i_trace =
            model.forward_token_with_trace(tok, pos, &mut int_cache, Some(intervention))?;
        for l in 0..num_layers {
            base_acts[l].push(b_trace.layer_hidden[l].clone());
            int_acts[l].push(i_trace.layer_hidden[l].clone());
        }
        kl_sum += logit_kl_divergence(&b_trace.logits, &i_trace.logits);
        if argmax(&b_trace.logits) != argmax(&i_trace.logits) {
            flips += 1;
        }
    }

    let layers = compute_layer_metrics(&base_acts, &int_acts);
    Ok(InterventionReport {
        intervention,
        sequence_len: token_sequence.len(),
        layers,
        mean_logit_kl_divergence: kl_sum / token_sequence.len() as f64,
        top1_flip_rate: flips as f64 / token_sequence.len() as f64,
    })
}

/// Held-out capability category for Project Resonance behavioral evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityCategory {
    Reasoning,
    Coding,
    Knowledge,
    InstructionFollowing,
}

impl CapabilityCategory {
    pub fn label(self) -> &'static str {
        match self {
            Self::Reasoning => "Reasoning",
            Self::Coding => "Coding",
            Self::Knowledge => "Knowledge",
            Self::InstructionFollowing => "Instruction Following",
        }
    }
}

/// One held-out tokenized evaluation example in a capability retention suite.
#[derive(Debug, Clone)]
pub struct CapabilityExample {
    pub category: CapabilityCategory,
    pub label: String,
    pub tokens: Vec<usize>,
}

/// Per-category held-out evaluation summary comparing candidate against reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityCategorySummary {
    pub category: CapabilityCategory,
    pub token_count: usize,
    pub ref_perplexity: f64,
    pub cand_perplexity: f64,
    pub perplexity_ratio: f64,
    pub mean_logit_kl: f64,
    pub top1_agreement: f64,
    pub final_layer_cosine: f64,
    pub mean_rsa: f64,
}

/// Complete multi-category held-out capability retention report for Project Resonance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilitySuiteReport {
    pub categories: Vec<CapabilityCategorySummary>,
    pub overall_ref_perplexity: f64,
    pub overall_cand_perplexity: f64,
    pub overall_perplexity_ratio: f64,
    pub overall_mean_kl: f64,
    pub overall_top1_agreement: f64,
    pub overall_final_cosine: f64,
    /// Relative capability retention percentage: `100 * (ref_ppl / cand_ppl).min(1.0) * 0.5 + 100 * top1 * 0.5`,
    /// bounded strictly to `[0, 100]` and accompanied by raw perplexity/KL/Top-1 so catastrophic failures are never hidden.
    pub composite_retention_pct: f64,
    pub catastrophic_failure: bool,
}

/// Convert a measured [`BehaviorComparisonReport`] into per-layer importance multipliers
/// so `MAP-B` allocates additional representation capacity to high-drift / low-RSA blocks.
pub fn derive_layer_importance_multipliers(
    report: &BehaviorComparisonReport,
) -> std::collections::BTreeMap<usize, f64> {
    let mean_l2 = (report
        .layers
        .iter()
        .map(|l| l.mean_rel_l2_divergence)
        .sum::<f64>()
        / report.layers.len().max(1) as f64)
        .max(1e-6);
    let mut out = std::collections::BTreeMap::new();
    for l in &report.layers {
        let l2_ratio = (l.mean_rel_l2_divergence / mean_l2).clamp(0.5, 4.0);
        let rsa_penalty = 1.0 + (1.0 - l.rsa_correlation.clamp(0.0, 1.0)) * 3.0;
        let mult = (l2_ratio * rsa_penalty).clamp(0.75, 5.0);
        out.insert(l.layer_index, mult);
    }
    out
}

/// Evaluate a candidate model against a higher-precision reference across identical held-out
/// Reasoning, Coding, Knowledge, and Instruction-Following token sequences.
pub fn evaluate_capability_suite(
    reference: &QCoreModel,
    candidate: &QCoreModel,
    examples: &[CapabilityExample],
) -> Result<CapabilitySuiteReport, RuntimeError> {
    if examples.is_empty() {
        return Err(RuntimeError::EmptyPrompt);
    }
    let mut categories = Vec::with_capacity(examples.len());
    let mut total_tokens = 0usize;
    let mut total_ref_nll = 0.0_f64;
    let mut total_cand_nll = 0.0_f64;
    let mut total_kl = 0.0_f64;
    let mut total_top1 = 0usize;
    let mut total_cos = 0.0_f64;

    for ex in examples {
        if ex.tokens.len() < 2 {
            continue;
        }
        let num_layers = reference
            .config()
            .num_layers
            .min(candidate.config().num_layers);
        let mut r_cache = KvCache::new(reference.config().num_layers);
        let mut c_cache = KvCache::new(candidate.config().num_layers);
        let mut r_acts: Vec<Vec<Vec<f64>>> = (0..num_layers).map(|_| Vec::new()).collect();
        let mut c_acts: Vec<Vec<Vec<f64>>> = (0..num_layers).map(|_| Vec::new()).collect();
        let mut ref_nll = 0.0_f64;
        let mut cand_nll = 0.0_f64;
        let mut kl_sum = 0.0_f64;
        let mut top1 = 0usize;
        let eval_steps = ex.tokens.len() - 1;

        for pos in 0..eval_steps {
            let tok = ex.tokens[pos];
            let next_tok = ex.tokens[pos + 1];
            let rt = reference.forward_token_with_trace(tok, pos, &mut r_cache, None)?;
            let ct = candidate.forward_token_with_trace(tok, pos, &mut c_cache, None)?;
            for l in 0..num_layers {
                r_acts[l].push(rt.layer_hidden[l].clone());
                c_acts[l].push(ct.layer_hidden[l].clone());
            }
            let r_probs = softmax(&rt.logits);
            let c_probs = softmax(&ct.logits);
            ref_nll += -r_probs[next_tok].max(1e-12).ln();
            cand_nll += -c_probs[next_tok].max(1e-12).ln();
            kl_sum += logit_kl_divergence(&rt.logits, &ct.logits);
            if argmax(&rt.logits) == argmax(&ct.logits) {
                top1 += 1;
            }
        }
        let layer_metrics = compute_layer_metrics(&r_acts, &c_acts);
        let final_cos = layer_metrics
            .last()
            .map_or(1.0, |m| m.mean_cosine_similarity);
        let mean_rsa = layer_metrics.iter().map(|m| m.rsa_correlation).sum::<f64>()
            / layer_metrics.len().max(1) as f64;
        let ref_ppl = (ref_nll / eval_steps as f64).exp();
        let cand_ppl = (cand_nll / eval_steps as f64).exp();

        total_tokens += eval_steps;
        total_ref_nll += ref_nll;
        total_cand_nll += cand_nll;
        total_kl += kl_sum;
        total_top1 += top1;
        total_cos += final_cos * eval_steps as f64;

        categories.push(CapabilityCategorySummary {
            category: ex.category,
            token_count: eval_steps,
            ref_perplexity: ref_ppl,
            cand_perplexity: cand_ppl,
            perplexity_ratio: cand_ppl / ref_ppl.max(1e-9),
            mean_logit_kl: kl_sum / eval_steps as f64,
            top1_agreement: top1 as f64 / eval_steps as f64,
            final_layer_cosine: final_cos,
            mean_rsa,
        });
    }

    let n = total_tokens.max(1) as f64;
    let overall_ref_ppl = (total_ref_nll / n).exp();
    let overall_cand_ppl = (total_cand_nll / n).exp();
    let overall_ppl_ratio = overall_cand_ppl / overall_ref_ppl.max(1e-9);
    let overall_mean_kl = total_kl / n;
    let overall_top1 = total_top1 as f64 / n;
    let overall_cos = total_cos / n;
    let ppl_ret = (overall_ref_ppl / overall_cand_ppl.max(1e-9)).clamp(0.0, 1.0);
    let composite_retention_pct = (0.5 * ppl_ret + 0.5 * overall_top1) * 100.0;
    // Flag catastrophic collapse transparently: perplexity inflation > 5x OR KL > 5.0 OR Top-1 < 25%
    let catastrophic_failure =
        overall_ppl_ratio > 5.0 || overall_mean_kl > 5.0 || overall_top1 < 0.25;

    Ok(CapabilitySuiteReport {
        categories,
        overall_ref_perplexity: overall_ref_ppl,
        overall_cand_perplexity: overall_cand_ppl,
        overall_perplexity_ratio: overall_ppl_ratio,
        overall_mean_kl,
        overall_top1_agreement: overall_top1,
        overall_final_cosine: overall_cos,
        composite_retention_pct,
        catastrophic_failure,
    })
}

fn compute_layer_metrics(
    ref_acts: &[Vec<Vec<f64>>],
    cand_acts: &[Vec<Vec<f64>>],
) -> Vec<LayerBehaviorMetrics> {
    let mut out = Vec::with_capacity(ref_acts.len());
    for (layer_idx, (r_seq, c_seq)) in ref_acts.iter().zip(cand_acts).enumerate() {
        let mut cos_sum = 0.0_f64;
        let mut cos_min = 1.0_f64;
        let mut l2_sum = 0.0_f64;
        let mut l2_max = 0.0_f64;
        let n = r_seq.len().max(1) as f64;
        for (rv, cv) in r_seq.iter().zip(c_seq) {
            let cos = cosine_similarity(rv, cv);
            let rel_l2 = relative_l2_error(rv, cv);
            cos_sum += cos;
            cos_min = cos_min.min(cos);
            l2_sum += rel_l2;
            l2_max = l2_max.max(rel_l2);
        }
        let rsa = compute_rsa(r_seq, c_seq);
        out.push(LayerBehaviorMetrics {
            layer_index: layer_idx,
            mean_cosine_similarity: cos_sum / n,
            min_cosine_similarity: cos_min,
            mean_rel_l2_divergence: l2_sum / n,
            max_rel_l2_divergence: l2_max,
            rsa_correlation: rsa,
        });
    }
    out
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (&x, &y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom <= 1e-12 {
        1.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

fn relative_l2_error(reference: &[f64], candidate: &[f64]) -> f64 {
    let mut diff_sq = 0.0_f64;
    let mut ref_sq = 0.0_f64;
    for (&r, &c) in reference.iter().zip(candidate) {
        let d = r - c;
        diff_sq += d * d;
        ref_sq += r * r;
    }
    if ref_sq <= 1e-12 {
        diff_sq.sqrt()
    } else {
        diff_sq.sqrt() / ref_sq.sqrt()
    }
}

/// Representational Similarity Analysis: Pearson correlation of upper-triangular
/// pairwise token cosine similarities between reference and candidate representations.
fn compute_rsa(ref_seq: &[Vec<f64>], cand_seq: &[Vec<f64>]) -> f64 {
    let n = ref_seq.len();
    if n < 3 {
        return 1.0;
    }
    let mut r_pairs = Vec::with_capacity(n * (n - 1) / 2);
    let mut c_pairs = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            r_pairs.push(cosine_similarity(&ref_seq[i], &ref_seq[j]));
            c_pairs.push(cosine_similarity(&cand_seq[i], &cand_seq[j]));
        }
    }
    pearson_correlation(&r_pairs, &c_pairs)
}

fn pearson_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n < 2.0 {
        return 1.0;
    }
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0_f64;
    let mut vx = 0.0_f64;
    let mut vy = 0.0_f64;
    for (&x, &y) in xs.iter().zip(ys) {
        let dx = x - mx;
        let dy = y - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = vx.sqrt() * vy.sqrt();
    if denom <= 1e-12 {
        1.0
    } else {
        (cov / denom).clamp(-1.0, 1.0)
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_v = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut exps: Vec<f64> = logits.iter().map(|&v| (v - max_v).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

fn logit_kl_divergence(ref_logits: &[f64], cand_logits: &[f64]) -> f64 {
    let p = softmax(ref_logits);
    let q = softmax(cand_logits);
    let mut kl = 0.0_f64;
    for (&pi, &qi) in p.iter().zip(&q) {
        if pi > 1e-12 {
            kl += pi * (pi / qi.max(1e-12)).ln();
        }
    }
    kl.max(0.0)
}

fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map_or(0, |(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qml_forge::{TransformStrategy, forge_safetensors_to_qmf};
    use qml_model::PayloadBudget;
    use qml_runtime::{TransformerConfig, generate_synthetic_llama_safetensors};
    use std::io::Cursor;

    #[test]
    fn behavior_observatory_measures_layer_drift_rsa_and_reversible_interventions() {
        let cfg = TransformerConfig {
            vocab_size: 32,
            hidden_size: 24,
            intermediate_size: 48,
            num_layers: 3,
            num_heads: 4,
            num_kv_heads: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_seq_len: 32,
            tie_word_embeddings: false,
        };
        let model_bytes = generate_synthetic_llama_safetensors(&cfg, 4, 2026).unwrap();
        let budget = PayloadBudget::default();
        let (ref_qmf, _) = forge_safetensors_to_qmf(
            Cursor::new(&model_bytes),
            "ref",
            "MIT",
            "llama",
            &TransformStrategy::Fp16,
            &budget,
            42,
        )
        .unwrap();
        let (q4_qmf, _) = forge_safetensors_to_qmf(
            Cursor::new(&model_bytes),
            "q4",
            "MIT",
            "llama",
            &TransformStrategy::Int4Sym { group_size: 32 },
            &budget,
            42,
        )
        .unwrap();
        let ref_model = QCoreModel::from_qmf(&ref_qmf, cfg.clone()).unwrap();
        let q4_model = QCoreModel::from_qmf(&q4_qmf, cfg).unwrap();

        let seq = [1, 5, 12, 3, 19, 8];

        // Self-comparison must yield cosine=1.0, L2=0.0, RSA=1.0, KL=0.0.
        let self_rep = compare_model_behavior(&ref_model, &ref_model, &seq).unwrap();
        assert_eq!(self_rep.num_layers, 3);
        assert!((self_rep.final_layer_cosine - 1.0).abs() < 1e-12);
        assert!(self_rep.peak_rel_l2_divergence < 1e-12);
        assert!(self_rep.mean_logit_kl_divergence < 1e-12);
        for l in &self_rep.layers {
            assert!((l.rsa_correlation - 1.0).abs() < 1e-12);
        }

        // Quantized comparison must measure non-zero activation drift with high RSA.
        let q4_rep = compare_model_behavior(&ref_model, &q4_model, &seq).unwrap();
        assert!(q4_rep.peak_rel_l2_divergence > 0.0);
        assert!(q4_rep.final_layer_cosine > 0.85);
        assert!(q4_rep.layers[0].rsa_correlation > 0.85);

        // Reversible MLP ablation must induce measurable downstream shift without mutating model.
        let int_rep =
            run_intervention_study(&ref_model, &seq, LayerIntervention::AblateMlp { layer: 1 })
                .unwrap();
        assert!(int_rep.layers[0].mean_rel_l2_divergence < 1e-12);
        assert!(int_rep.layers[1].mean_rel_l2_divergence > 0.0);
        assert!(int_rep.mean_logit_kl_divergence > 0.0);
    }
}
