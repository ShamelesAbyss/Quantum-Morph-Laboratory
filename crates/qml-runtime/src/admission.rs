//! Admission for the two existing decoder families, not a universal HF adapter.
use crate::{RuntimeError, TransformerConfig};
use serde::{
    Deserialize, Deserializer,
    de::{Error, MapAccess, Visitor},
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;

fn unique_extra<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<BTreeMap<String, Value>, D::Error> {
    struct Fields;
    impl<'de> Visitor<'de> for Fields {
        type Value = BTreeMap<String, Value>;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("configuration fields without duplicate keys")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
            let mut fields = BTreeMap::new();
            while let Some((key, value)) = access.next_entry::<String, Value>()? {
                if fields.insert(key.clone(), value).is_some() {
                    return Err(A::Error::custom(format!("duplicate config field {key:?}")));
                }
            }
            Ok(fields)
        }
    }
    deserializer.deserialize_map(Fields)
}

#[derive(Deserialize)]
struct RopeConfig {
    #[serde(rename = "type")]
    legacy_type: Option<String>,
    rope_type: Option<String>,
    factor: Option<f64>,
    partial_rotary_factor: Option<f64>,
    attention_factor: Option<f64>,
    rope_theta: Option<f64>,
    original_max_position_embeddings: Option<usize>,
    #[serde(flatten, deserialize_with = "unique_extra")]
    extra: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct HfConfig {
    model_type: Option<String>,
    architectures: Option<Vec<String>>,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    hidden_act: Option<String>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    max_position_embeddings: Option<usize>,
    #[serde(default)]
    tie_word_embeddings: bool,
    rope_scaling: Option<RopeConfig>,
    rope_parameters: Option<RopeConfig>,
    partial_rotary_factor: Option<f64>,
    rotary_dim: Option<usize>,
    #[serde(default)]
    rope_interleaved: bool,
    #[serde(default)]
    use_sliding_window: bool,
    #[serde(default = "default_sliding_window")]
    sliding_window: Option<usize>,
    max_window_layers: Option<usize>,
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(flatten, deserialize_with = "unique_extra")]
    extra: BTreeMap<String, Value>,
}

fn default_sliding_window() -> Option<usize> {
    Some(4096)
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    RuntimeError::InvalidConfig(message.into())
}

fn number(value: &Value, name: &str) -> Result<f64, RuntimeError> {
    value
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| invalid(format!("{name} must be a finite number")))
}

fn require_unit(value: f64, name: &str) -> Result<(), RuntimeError> {
    if value != 1.0 {
        return Err(invalid(format!(
            "{name} must be 1; partial/scaled execution is unsupported"
        )));
    }
    Ok(())
}

/// Only explicitly classified metadata may be ignored. Unknown computation controls fail closed.
fn validate_extra(extra: &BTreeMap<String, Value>) -> Result<(), RuntimeError> {
    for (name, value) in extra {
        match name.as_str() {
            "_name_or_path"
            | "name_or_path"
            | "_commit_hash"
            | "transformers_version"
            | "torch_dtype"
            | "dtype"
            | "bos_token_id"
            | "eos_token_id"
            | "pad_token_id"
            | "sep_token_id"
            | "decoder_start_token_id"
            | "tokenizer_class"
            | "initializer_range"
            | "finetuning_task"
            | "task_specific_params"
            | "id2label"
            | "label2id"
            | "problem_type"
            | "return_dict"
            | "output_attentions"
            | "output_hidden_states"
            | "use_cache"
            | "is_decoder"
            | "torchscript"
            | "_attn_implementation"
            | "_attn_implementation_internal"
            | "_attn_implementation_autoset" => {}
            "attention_dropout" | "hidden_dropout" | "hidden_dropout_prob" => {
                // Dropout is disabled in inference, but malformed rates are still rejected.
                if !value.is_null() && !(0.0..=1.0).contains(&number(value, name)?) {
                    return Err(invalid(format!("{name} must be in [0, 1]")));
                }
            }
            "pretraining_tp" => {
                // Partitioning the same stored weights does not change this decoder's mathematics.
                if !value.is_null() && value.as_u64().is_none_or(|v| v == 0) {
                    return Err(invalid("pretraining_tp must be a positive integer"));
                }
            }
            "is_encoder_decoder"
            | "add_cross_attention"
            | "tie_encoder_decoder"
            | "trust_remote_code"
            | "qk_layernorm"
            | "qk_norm"
            | "use_alibi"
            | "alibi"
            | "use_logn_attn" => {
                if !value.is_null() && value != &Value::Bool(false) {
                    return Err(invalid(format!("unsupported active feature {name}")));
                }
            }
            "quantization_config" | "auto_map" => {
                if !value.is_null() && !value.as_object().is_some_and(|v| v.is_empty()) {
                    return Err(invalid(format!("unsupported active feature {name}")));
                }
            }
            "cross_attention_hidden_size" => {
                if !value.is_null() {
                    return Err(invalid("cross-attention is unsupported"));
                }
            }
            "attn_logit_softcapping" | "final_logit_softcapping" => {
                if !value.is_null() && number(value, name)? != 0.0 {
                    return Err(invalid(format!("unsupported active feature {name}")));
                }
            }
            "position_embedding_type" => {
                if !matches!(value.as_str(), Some("rope" | "rotary")) {
                    return Err(invalid("only rotary position embeddings are supported"));
                }
            }
            _ => {
                return Err(invalid(format!(
                    "unrecognized config field {name:?}; its execution semantics are not verified"
                )));
            }
        }
    }
    Ok(())
}

fn validate_rope(
    value: &RopeConfig,
    name: &str,
    theta: &mut Option<f64>,
) -> Result<(), RuntimeError> {
    if let (Some(a), Some(b)) = (&value.legacy_type, &value.rope_type)
        && a != b
    {
        return Err(invalid(format!("conflicting {name} type and rope_type")));
    }
    let kind = value
        .rope_type
        .as_deref()
        .or(value.legacy_type.as_deref())
        .unwrap_or("default");
    // Linear scaling by exactly one is the identity; all other non-default variants are active.
    if kind != "default" && !(kind == "linear" && value.factor == Some(1.0)) {
        return Err(invalid(format!(
            "unsupported active {name} rope_type {kind:?}"
        )));
    }
    for (key, factor) in [
        ("factor", value.factor),
        ("partial_rotary_factor", value.partial_rotary_factor),
        ("attention_factor", value.attention_factor),
    ] {
        if let Some(factor) = factor {
            require_unit(factor, &format!("{name}.{key}"))?;
        }
    }
    if let Some(nested) = value.rope_theta {
        if theta.is_some_and(|top| top != nested) {
            return Err(invalid("conflicting RoPE theta values"));
        }
        *theta = Some(nested);
    }
    if value.original_max_position_embeddings == Some(0) {
        return Err(invalid(format!(
            "{name}.original_max_position_embeddings must be positive"
        )));
    }
    if let Some(key) = value.extra.keys().next() {
        return Err(invalid(format!("unsupported RoPE parameter {name}.{key}")));
    }
    Ok(())
}

pub(super) fn parse(bytes: &[u8]) -> Result<TransformerConfig, RuntimeError> {
    if bytes.len() > 1024 * 1024 {
        return Err(invalid("config.json exceeds the 1 MiB admission limit"));
    }
    let raw: HfConfig = serde_json::from_slice(bytes)
        .map_err(|e| invalid(format!("config.json parse error: {e}")))?;
    let from_arch = match raw.architectures.as_deref() {
        None | Some([]) => None,
        Some([name]) => match name.as_str() {
            "LlamaForCausalLM" => Some("llama"),
            "Qwen2ForCausalLM" => Some("qwen2"),
            _ => return Err(invalid(format!("unsupported architecture {name:?}"))),
        },
        Some(_) => {
            return Err(invalid(
                "architectures must identify exactly one supported causal decoder",
            ));
        }
    };
    let family = match (raw.model_type.as_deref(), from_arch) {
        (Some(model @ ("llama" | "qwen2")), arch) if arch.is_none_or(|arch| arch == model) => model,
        (None, Some(arch)) => arch,
        (None, None) => {
            return Err(invalid(
                "missing supported model_type or architectures identity",
            ));
        }
        _ => {
            return Err(invalid(
                "unsupported or conflicting model_type/architectures identity",
            ));
        }
    };
    if raw
        .hidden_act
        .as_deref()
        .is_some_and(|act| !matches!(act, "silu" | "swish"))
    {
        return Err(invalid("only SiLU (swish) activation is supported"));
    }
    if raw.attention_bias || raw.mlp_bias {
        return Err(invalid(
            "attention_bias/MLP bias layouts are unsupported; standard optional Qwen2 q/k/v biases remain supported",
        ));
    }
    if raw.rope_interleaved {
        return Err(invalid("interleaved RoPE is unsupported"));
    }
    if let Some(factor) = raw.partial_rotary_factor {
        require_unit(factor, "partial_rotary_factor")?;
    }
    validate_extra(&raw.extra)?;
    let mut theta = raw.rope_theta;
    for (name, value) in [
        ("rope_scaling", &raw.rope_scaling),
        ("rope_parameters", &raw.rope_parameters),
    ] {
        if let Some(value) = value {
            validate_rope(value, name, &mut theta)?;
        }
    }
    let cfg = TransformerConfig {
        vocab_size: raw.vocab_size,
        hidden_size: raw.hidden_size,
        intermediate_size: raw.intermediate_size,
        num_layers: raw.num_hidden_layers,
        num_heads: raw.num_attention_heads,
        num_kv_heads: raw.num_key_value_heads.unwrap_or(raw.num_attention_heads),
        rms_norm_eps: raw.rms_norm_eps.unwrap_or(1e-6),
        rope_theta: theta.unwrap_or(10_000.0),
        max_seq_len: raw.max_position_embeddings.unwrap_or(if family == "qwen2" {
            32768
        } else {
            2048
        }),
        tie_word_embeddings: raw.tie_word_embeddings,
    };
    cfg.validate()?;
    for (name, dim) in [("head_dim", raw.head_dim), ("rotary_dim", raw.rotary_dim)] {
        if dim.is_some_and(|dim| dim != cfg.head_dim()) {
            return Err(invalid(format!(
                "{name} must match hidden_size / num_attention_heads (full-head RoPE)"
            )));
        }
    }
    if let Some(types) = raw.layer_types {
        if types.len() != cfg.num_layers || types.iter().any(|kind| kind != "full_attention") {
            return Err(invalid(
                "layer_types must contain one full_attention entry per layer; sliding/hybrid attention is unsupported",
            ));
        }
    } else if raw.use_sliding_window
        && raw.sliding_window.is_some()
        && raw.max_window_layers.unwrap_or(28) < cfg.num_layers
    {
        return Err(invalid("active sliding-window attention is unsupported"));
    }
    Ok(cfg)
}
