//! Bounded parser for Hugging Face transformer `config.json` metadata.
//!
//! Identifies common decoder-only architectures (Llama, Mistral, Qwen2, GPT-2,
//! Phi, Gemma, SmolLM, etc.) without executing code or loading weights.

use serde::{Deserialize, Serialize};
use std::io::Read;

/// Maximum `config.json` file size accepted (1 MiB).
pub const MAX_CONFIG_BYTES: u64 = 1024 * 1024;

/// Parsed architectural summary of a transformer model configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub hidden_size: Option<usize>,
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub vocab_size: Option<usize>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub rms_norm_eps: Option<f64>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub torch_dtype: Option<String>,
}

impl ModelConfig {
    /// Parse a `config.json` stream with a strict 1 MiB ceiling.
    pub fn from_reader<R: Read>(reader: R) -> Result<Self, String> {
        let mut bytes = Vec::new();
        reader
            .take(MAX_CONFIG_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| format!("I/O error reading config.json: {e}"))?;
        if bytes.len() as u64 > MAX_CONFIG_BYTES {
            return Err(format!(
                "config.json exceeds maximum allowed size of {MAX_CONFIG_BYTES} bytes"
            ));
        }
        serde_json::from_slice(&bytes).map_err(|e| format!("invalid config.json: {e}"))
    }

    /// Infer the canonical architecture family label if recognized.
    pub fn architecture_family(&self) -> String {
        if let Some(ref mt) = self.model_type {
            return mt.clone();
        }
        if let Some(first) = self.architectures.first() {
            return first.clone();
        }
        "unknown".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::ModelConfig;
    use std::io::Cursor;

    #[test]
    fn parses_llama_style_config_safely() {
        let json = br#"{
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 576,
            "num_hidden_layers": 30,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "intermediate_size": 1536,
            "vocab_size": 49152,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": true,
            "torch_dtype": "bfloat16",
            "ignored_extra_field": 123
        }"#;
        let cfg = ModelConfig::from_reader(Cursor::new(json)).unwrap();
        assert_eq!(cfg.architecture_family(), "llama");
        assert_eq!(cfg.hidden_size, Some(576));
        assert_eq!(cfg.num_hidden_layers, Some(30));
        assert_eq!(cfg.num_key_value_heads, Some(3));
        assert_eq!(cfg.tie_word_embeddings, Some(true));
    }
}
