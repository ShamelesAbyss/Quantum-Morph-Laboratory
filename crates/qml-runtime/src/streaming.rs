//! Layer-by-Layer Streaming BF16 Calibration & Reference Evaluation (`qml-runtime::streaming`).
//!
//! Enables calibrating (`ModelCalibrationProfile`) and computing exact BF16 reference logits
//! (`ReferenceSuiteCache`) for **7B, 14B, and 27B–32B parameter models (`15–64 GB` BF16)** on
//! consumer 8 GB VRAM GPUs and 32 GB RAM laptops by holding **AT MOST ONE transformer layer**
//! (`~450–900 MiB` BF16) in VRAM/RAM at any instant!
//!
//! Pipeline:
//! 1. Read ONLY the distinct token rows needed from `model.embed_tokens.weight` (`< 10 MiB` RAM).
//! 2. Stream layers `0..num_layers` one by one:
//!    - Load that single layer's BF16 projections directly (no `f64` dense allocation!) and upload
//!      to temporary GPU BF16 buffers (or Rayon CPU fallback).
//!    - Advance all calibration sequences through the layer while observing `attn_in`, `o_in`,
//!      `mlp_in`, and `down_in` covariance blocks (`H_b = (2/N) X_bᵀ X_b`).
//!    - Advance all held-out evaluation sequences through the exact same layer.
//!    - Drop all 7 layer matrices before touching layer `l + 1`!
//! 3. Load `model.norm.weight` + `lm_head.weight` (chunked into `<= 32,768`-row horizontal blocks
//!    so even 152K–256K vocabularies stay under `<= 256 MiB` per GPU buffer!), calibrate `lm_head`,
//!    and compute exact BF16 reference logits for every evaluation token.

use crate::{RuntimeError, TransformerConfig, apply_rope, rms_norm, silu, softmax};
use qml_hardware::{PersistentGpuMatrix, WgpuMatvecEngine};
use qml_model::{PayloadBudget, SafetensorsSource};
use qml_quant::{Bf16Matrix, LayerCalibrationAccumulator, LinearOperator, ModelCalibrationProfile};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Result of a single layer-by-layer streaming BF16 calibration + reference pass.
#[derive(Debug, Clone)]
pub struct StreamingCalibAndRefResult {
    pub calibration: ModelCalibrationProfile,
    /// `[eval_seq_idx][step_pos]` -> exact BF16 reference logits (`vocab_size`) for `pos in 0..seq.len()-1`.
    pub eval_logits: Vec<Vec<Vec<f64>>>,
    pub peak_layer_vram_bytes: u64,
    pub used_gpu: bool,
    pub elapsed_secs: f64,
}

enum StreamingLinear {
    Gpu(PersistentGpuMatrix),
    Cpu(Bf16Matrix),
}

impl StreamingLinear {
    fn from_bf16(mat: Bf16Matrix, gpu: Option<&Arc<WgpuMatvecEngine>>) -> (Self, u64) {
        if let Some(engine) = gpu
            && let Ok(p) = engine.upload_bf16(&mat)
        {
            let bytes = p.vram_bytes();
            return (Self::Gpu(p), bytes);
        }
        (Self::Cpu(mat), 0)
    }

    fn matvec(&self, x: &[f64]) -> Result<Vec<f64>, RuntimeError> {
        match self {
            Self::Gpu(g) => g
                .matvec(x)
                .map_err(|e| RuntimeError::InvalidConfig(format!("streaming GPU matvec: {e}"))),
            Self::Cpu(c) => Ok(c.matvec(x)?),
        }
    }
}

/// Stream a single-file or multi-shard Safetensors checkpoint ONE layer at a time to collect
/// an exact [`ModelCalibrationProfile`] AND exact BF16 reference logits for held-out evaluation
/// sequences in `< 1 GiB` peak VRAM (scaling cleanly to 7B, 14B, and 32B models on 8 GB GPUs).
#[allow(clippy::too_many_arguments)]
pub fn stream_calibrate_and_reference<F>(
    model_path: &Path,
    model_id: &str,
    config: &TransformerConfig,
    calib_seqs: &[Vec<usize>],
    eval_seqs: &[Vec<usize>],
    hessian_block_size: usize,
    use_gpu: bool,
    budget: &PayloadBudget,
    mut on_progress: F,
) -> Result<StreamingCalibAndRefResult, RuntimeError>
where
    F: FnMut(usize, usize, &str),
{
    let t0 = Instant::now();
    config.validate()?;
    let mut source = SafetensorsSource::open(model_path, budget)
        .map_err(|e| RuntimeError::InvalidConfig(format!("safetensors open error: {e}")))?;

    let gpu_engine = if use_gpu {
        WgpuMatvecEngine::try_new().ok().map(Arc::new)
    } else {
        None
    };
    let used_gpu = gpu_engine.is_some();

    // 1. Collect all distinct token IDs across calibration + eval sequences and load ONLY those rows
    let mut distinct_tokens = BTreeSet::new();
    for seq in calib_seqs.iter().chain(eval_seqs.iter()) {
        for &tok in seq {
            if tok < config.vocab_size {
                distinct_tokens.insert(tok);
            }
        }
    }
    let token_list: Vec<usize> = distinct_tokens.into_iter().collect();
    let embed_rows = source
        .read_tensor_rows_f64("model.embed_tokens.weight", &token_list, budget)
        .map_err(|e| RuntimeError::InvalidConfig(format!("embed row read error: {e}")))?;

    let mut calib_hidden: Vec<Vec<Vec<f64>>> = calib_seqs
        .iter()
        .map(|seq| {
            seq.iter()
                .take(config.max_seq_len)
                .filter_map(|t| embed_rows.get(t).cloned())
                .collect()
        })
        .collect();
    let mut eval_hidden: Vec<Vec<Vec<f64>>> = eval_seqs
        .iter()
        .map(|seq| {
            let steps = seq.len().saturating_sub(1).min(config.max_seq_len);
            seq.iter()
                .take(steps)
                .filter_map(|t| embed_rows.get(t).cloned())
                .collect()
        })
        .collect();
    drop(embed_rows);

    let bs = hessian_block_size.max(1);
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let head_dim = config.head_dim();
    let kv_groups = config.num_heads / config.num_kv_heads;
    let scale = 1.0 / (head_dim as f64).sqrt();

    let mut layers_map: BTreeMap<String, qml_quant::LayerCalibrationStats> = BTreeMap::new();
    let mut peak_layer_vram_bytes: u64 = 0;

    let load_bf16_linear = |src: &mut SafetensorsSource,
                            name: &str,
                            eng: Option<&Arc<WgpuMatvecEngine>>|
     -> Result<(StreamingLinear, u64), RuntimeError> {
        let (rows, cols, bits) = src
            .read_tensor_bf16_bits(name, budget)
            .map_err(|e| RuntimeError::InvalidConfig(format!("failed reading {name}: {e}")))?;
        let mat = Bf16Matrix::from_raw(rows, cols, bits)?;
        Ok(StreamingLinear::from_bf16(mat, eng))
    };

    let load_opt_vec = |src: &mut SafetensorsSource, name: &str| -> Option<Vec<f64>> {
        src.read_tensor_f64(name, budget).ok()
    };

    // 2. Stream ONE transformer layer at a time through VRAM / RAM
    for l in 0..config.num_layers {
        on_progress(l + 1, config.num_layers + 1, "streaming layer");

        let input_norm = source
            .read_tensor_f64(&format!("model.layers.{l}.input_layernorm.weight"), budget)
            .map_err(|e| RuntimeError::InvalidConfig(format!("layer {l} input_norm: {e}")))?;
        let post_norm = source
            .read_tensor_f64(
                &format!("model.layers.{l}.post_attention_layernorm.weight"),
                budget,
            )
            .map_err(|e| RuntimeError::InvalidConfig(format!("layer {l} post_norm: {e}")))?;

        let q_bias = load_opt_vec(
            &mut source,
            &format!("model.layers.{l}.self_attn.q_proj.bias"),
        );
        let k_bias = load_opt_vec(
            &mut source,
            &format!("model.layers.{l}.self_attn.k_proj.bias"),
        );
        let v_bias = load_opt_vec(
            &mut source,
            &format!("model.layers.{l}.self_attn.v_proj.bias"),
        );

        let (q_proj, b0) = load_bf16_linear(
            &mut source,
            &format!("model.layers.{l}.self_attn.q_proj.weight"),
            gpu_engine.as_ref(),
        )?;
        let (k_proj, b1) = load_bf16_linear(
            &mut source,
            &format!("model.layers.{l}.self_attn.k_proj.weight"),
            gpu_engine.as_ref(),
        )?;
        let (v_proj, b2) = load_bf16_linear(
            &mut source,
            &format!("model.layers.{l}.self_attn.v_proj.weight"),
            gpu_engine.as_ref(),
        )?;
        let (o_proj, b3) = load_bf16_linear(
            &mut source,
            &format!("model.layers.{l}.self_attn.o_proj.weight"),
            gpu_engine.as_ref(),
        )?;
        let (gate_proj, b4) = load_bf16_linear(
            &mut source,
            &format!("model.layers.{l}.mlp.gate_proj.weight"),
            gpu_engine.as_ref(),
        )?;
        let (up_proj, b5) = load_bf16_linear(
            &mut source,
            &format!("model.layers.{l}.mlp.up_proj.weight"),
            gpu_engine.as_ref(),
        )?;
        let (down_proj, b6) = load_bf16_linear(
            &mut source,
            &format!("model.layers.{l}.mlp.down_proj.weight"),
            gpu_engine.as_ref(),
        )?;
        let layer_vram = b0 + b1 + b2 + b3 + b4 + b5 + b6;
        peak_layer_vram_bytes = peak_layer_vram_bytes.max(layer_vram);

        let mut attn_acc = LayerCalibrationAccumulator::new(
            format!("model.layers.{l}.self_attn.q_proj.weight"),
            h,
            bs,
        );
        let mut o_acc = LayerCalibrationAccumulator::new(
            format!("model.layers.{l}.self_attn.o_proj.weight"),
            h,
            bs,
        );
        let mut mlp_acc = LayerCalibrationAccumulator::new(
            format!("model.layers.{l}.mlp.gate_proj.weight"),
            h,
            bs,
        );
        let mut down_acc = LayerCalibrationAccumulator::new(
            format!("model.layers.{l}.mlp.down_proj.weight"),
            inter,
            bs,
        );

        let run_seq_through_layer = |seq_h: &mut [Vec<f64>],
                                     observe: bool,
                                     attn_acc: &mut LayerCalibrationAccumulator,
                                     o_acc: &mut LayerCalibrationAccumulator,
                                     mlp_acc: &mut LayerCalibrationAccumulator,
                                     down_acc: &mut LayerCalibrationAccumulator|
         -> Result<(), RuntimeError> {
            let mut k_cache: Vec<Vec<f64>> = Vec::with_capacity(seq_h.len());
            let mut v_cache: Vec<Vec<f64>> = Vec::with_capacity(seq_h.len());
            for (pos, hidden) in seq_h.iter_mut().enumerate() {
                let normed = rms_norm(hidden, &input_norm, config.rms_norm_eps);
                if observe {
                    attn_acc.observe(&normed);
                }
                let mut q = q_proj.matvec(&normed)?;
                if let Some(b) = &q_bias {
                    for (qv, &bv) in q.iter_mut().zip(b) {
                        *qv += bv;
                    }
                }
                let mut k = k_proj.matvec(&normed)?;
                if let Some(b) = &k_bias {
                    for (kv, &bv) in k.iter_mut().zip(b) {
                        *kv += bv;
                    }
                }
                let mut v = v_proj.matvec(&normed)?;
                if let Some(b) = &v_bias {
                    for (vv, &bv) in v.iter_mut().zip(b) {
                        *vv += bv;
                    }
                }
                for hd in 0..config.num_heads {
                    apply_rope(
                        &mut q[hd * head_dim..(hd + 1) * head_dim],
                        pos,
                        config.rope_theta,
                    );
                }
                for kh in 0..config.num_kv_heads {
                    apply_rope(
                        &mut k[kh * head_dim..(kh + 1) * head_dim],
                        pos,
                        config.rope_theta,
                    );
                }
                k_cache.push(k);
                v_cache.push(v);
                let seq_len = k_cache.len();
                let mut attn_out = vec![0.0_f64; h];
                for hd in 0..config.num_heads {
                    let kv_h = hd / kv_groups;
                    let q_head = &q[hd * head_dim..(hd + 1) * head_dim];
                    let mut scores = Vec::with_capacity(seq_len);
                    for k_vec in &k_cache {
                        let k_head = &k_vec[kv_h * head_dim..(kv_h + 1) * head_dim];
                        let dot: f64 = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum();
                        scores.push(dot * scale);
                    }
                    let probs = softmax(&scores);
                    let out_head = &mut attn_out[hd * head_dim..(hd + 1) * head_dim];
                    for (t, &p) in probs.iter().enumerate() {
                        let v_head = &v_cache[t][kv_h * head_dim..(kv_h + 1) * head_dim];
                        for (o, &vv) in out_head.iter_mut().zip(v_head) {
                            *o += p * vv;
                        }
                    }
                }
                if observe {
                    o_acc.observe(&attn_out);
                }
                let proj_out = o_proj.matvec(&attn_out)?;
                for (hv, pv) in hidden.iter_mut().zip(proj_out) {
                    *hv += pv;
                }

                let mlp_in = rms_norm(hidden, &post_norm, config.rms_norm_eps);
                if observe {
                    mlp_acc.observe(&mlp_in);
                }
                let gate = gate_proj.matvec(&mlp_in)?;
                let up = up_proj.matvec(&mlp_in)?;
                let activated: Vec<f64> = gate.iter().zip(up).map(|(&g, u)| silu(g) * u).collect();
                if observe {
                    down_acc.observe(&activated);
                }
                let mlp_out = down_proj.matvec(&activated)?;
                for (hv, mv) in hidden.iter_mut().zip(mlp_out) {
                    *hv += mv;
                }
            }
            Ok(())
        };

        for seq_h in &mut calib_hidden {
            run_seq_through_layer(
                seq_h,
                true,
                &mut attn_acc,
                &mut o_acc,
                &mut mlp_acc,
                &mut down_acc,
            )?;
        }
        for seq_h in &mut eval_hidden {
            run_seq_through_layer(
                seq_h,
                false,
                &mut attn_acc,
                &mut o_acc,
                &mut mlp_acc,
                &mut down_acc,
            )?;
        }

        // Finalize layer stats and drop all 7 layer weight buffers immediately!
        let a_stats = attn_acc.finalize();
        for proj in ["q_proj", "k_proj", "v_proj"] {
            let name = format!("model.layers.{l}.self_attn.{proj}.weight");
            let mut s = a_stats.clone();
            s.tensor_name = name.clone();
            layers_map.insert(name, s);
        }
        let o_stats = o_acc.finalize();
        layers_map.insert(o_stats.tensor_name.clone(), o_stats);
        let m_stats = mlp_acc.finalize();
        for proj in ["gate_proj", "up_proj"] {
            let name = format!("model.layers.{l}.mlp.{proj}.weight");
            let mut s = m_stats.clone();
            s.tensor_name = name.clone();
            layers_map.insert(name, s);
        }
        let d_stats = down_acc.finalize();
        layers_map.insert(d_stats.tensor_name.clone(), d_stats);
    }

    // 3. Final RMSNorm + chunked BF16 LM Head (stays <= 256 MiB per GPU buffer for 152K-256K vocabs!)
    on_progress(config.num_layers + 1, config.num_layers + 1, "lm_head");
    let final_norm = source
        .read_tensor_f64("model.norm.weight", budget)
        .map_err(|e| RuntimeError::InvalidConfig(format!("model.norm.weight: {e}")))?;

    let mut lm_head_acc = LayerCalibrationAccumulator::new("lm_head.weight", h, bs);
    let mut total_calib_tokens = 0usize;
    for seq_h in &calib_hidden {
        for hidden in seq_h {
            total_calib_tokens += 1;
            let fh = rms_norm(hidden, &final_norm, config.rms_norm_eps);
            lm_head_acc.observe(&fh);
        }
    }
    let lm_stats = lm_head_acc.finalize();
    if config.tie_word_embeddings {
        let mut emb_s = lm_stats.clone();
        emb_s.tensor_name = "model.embed_tokens.weight".to_owned();
        layers_map.insert("model.embed_tokens.weight".to_owned(), emb_s);
    }
    layers_map.insert("lm_head.weight".to_owned(), lm_stats);

    let eval_normed: Vec<Vec<Vec<f64>>> = eval_hidden
        .iter()
        .map(|seq_h| {
            seq_h
                .iter()
                .map(|hidden| rms_norm(hidden, &final_norm, config.rms_norm_eps))
                .collect()
        })
        .collect();

    let lm_name = if config.tie_word_embeddings {
        "model.embed_tokens.weight"
    } else {
        "lm_head.weight"
    };

    let (lm_rows, lm_cols, lm_bits) = source
        .read_tensor_bf16_bits(lm_name, budget)
        .map_err(|e| RuntimeError::InvalidConfig(format!("failed reading {lm_name}: {e}")))?;
    if lm_rows != config.vocab_size
        || lm_cols != config.hidden_size
        || lm_bits.iter().any(|bits| bits & 0x7f80 == 0x7f80)
    {
        return Err(RuntimeError::InvalidConfig(format!(
            "invalid dimensions or non-finite values in {lm_name}"
        )));
    }

    let mut eval_logits: Vec<Vec<Vec<f64>>> = eval_normed
        .iter()
        .map(|seq| seq.iter().map(|_| Vec::with_capacity(lm_rows)).collect())
        .collect();

    // Chunk lm_head into horizontal slices of <= 32,768 rows (~235 MiB for 7B, fits any GPU binding limit!)
    const LM_CHUNK_ROWS: usize = 32_768;
    let mut row_start = 0usize;
    while row_start < lm_rows {
        let chunk_rows = (lm_rows - row_start).min(LM_CHUNK_ROWS);
        let slice_bits = lm_bits[row_start * lm_cols..(row_start + chunk_rows) * lm_cols].to_vec();
        let chunk_mat = Bf16Matrix::from_raw(chunk_rows, lm_cols, slice_bits)?;
        let (chunk_op, chunk_vram) = StreamingLinear::from_bf16(chunk_mat, gpu_engine.as_ref());
        peak_layer_vram_bytes = peak_layer_vram_bytes.max(chunk_vram);

        for (s_idx, seq) in eval_normed.iter().enumerate() {
            for (p_idx, fh) in seq.iter().enumerate() {
                let part = chunk_op.matvec(fh)?;
                eval_logits[s_idx][p_idx].extend_from_slice(&part);
            }
        }
        row_start += chunk_rows;
    }

    Ok(StreamingCalibAndRefResult {
        calibration: ModelCalibrationProfile {
            model_id: model_id.to_owned(),
            calibration_tokens: total_calib_tokens,
            block_size: bs,
            layers: layers_map,
        },
        eval_logits,
        peak_layer_vram_bytes,
        used_gpu,
        elapsed_secs: t0.elapsed().as_secs_f64(),
    })
}
