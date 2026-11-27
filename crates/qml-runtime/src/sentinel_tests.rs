use super::*;
use qml_format::v1::{QmfBudget, QmfManifest, QmfRepresentation};
use qml_model::PayloadBudget;
use serde_json::{Value, json};
use std::io::{self, Cursor, Read, Seek, SeekFrom};

#[test]
fn sentinel_large_declared_layer_count_does_not_reserve_an_unbacked_directory() {
    let archive = archive(&source(None));
    let mut declared = config();
    declared.num_layers = 1 << 30;
    declared.validate().unwrap(); // Addressable arithmetic does not imply those layers exist.
    let memory = QCoreModel::from_qmf(&archive, declared.clone())
        .err()
        .unwrap();
    let file = QCoreModel::from_qmf_reader(&mut reader(encoded(&archive)), declared)
        .err()
        .unwrap();
    for error in [memory, file] {
        assert!(error.to_string().contains("not found"), "{error}");
    }
}

fn config() -> TransformerConfig {
    TransformerConfig {
        vocab_size: 8,
        hidden_size: 8,
        intermediate_size: 16,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 1,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        max_seq_len: 8,
        tie_word_embeddings: false,
    }
}
fn hf() -> Value {
    json!({"model_type":"llama", "architectures":["LlamaForCausalLM"], "vocab_size":8,
        "hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,
        "num_attention_heads":2,"num_key_value_heads":1,"max_position_embeddings":8})
}
fn parse(value: &Value) -> Result<TransformerConfig, RuntimeError> {
    TransformerConfig::from_hf_config_bytes(&serde_json::to_vec(value).unwrap())
}
fn rejects(field: &str, value: Value) {
    let mut input = hf();
    input[field] = value;
    assert!(parse(&input).is_err(), "must reject {field}: {input}");
}
#[test]
fn sentinel_preserves_llama_and_qwen2_identity_and_architecture_defaults() {
    assert_eq!(parse(&hf()).unwrap(), config());
    let mut llama = hf();
    llama
        .as_object_mut()
        .unwrap()
        .remove("max_position_embeddings");
    assert_eq!(parse(&llama).unwrap().max_seq_len, 2048);
    let mut input = hf();
    input["model_type"] = json!("qwen2");
    input["architectures"] = json!(["Qwen2ForCausalLM"]);
    input
        .as_object_mut()
        .unwrap()
        .remove("max_position_embeddings");
    let qwen = parse(&input).unwrap();
    assert_eq!(qwen.rope_theta, 10_000.0);
    assert_eq!(qwen.rms_norm_eps, 1e-6);
    assert_eq!(qwen.max_seq_len, 32768);
    input.as_object_mut().unwrap().remove("model_type");
    assert_eq!(parse(&input).unwrap(), qwen);
    let mut input = hf();
    input["architectures"] = json!([]);
    assert_eq!(parse(&input).unwrap(), config());
    input.as_object_mut().unwrap().remove("architectures");
    input.as_object_mut().unwrap().remove("num_key_value_heads");
    assert_eq!(parse(&input).unwrap().num_kv_heads, 2);
}

#[test]
fn sentinel_admits_representative_existing_family_metadata_without_weight_allocation() {
    for input in [
        json!({"model_type":"llama","architectures":["LlamaForCausalLM"],
            "vocab_size":32000,"hidden_size":4096,"intermediate_size":11008,
            "num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":32,
            "hidden_act":"silu","max_position_embeddings":4096,"rms_norm_eps":1e-6,
            "rope_theta":10000,"rope_scaling":null,"tie_word_embeddings":false,
            "attention_bias":false,"mlp_bias":false,"pretraining_tp":1,"torch_dtype":"float16"}),
        json!({"model_type":"qwen2","architectures":["Qwen2ForCausalLM"],
            "vocab_size":152064,"hidden_size":3584,"intermediate_size":18944,
            "num_hidden_layers":28,"num_attention_heads":28,"num_key_value_heads":4,
            "hidden_act":"silu","max_position_embeddings":32768,"rms_norm_eps":1e-6,
            "rope_theta":1000000,"tie_word_embeddings":false,"attention_dropout":0.0,
            "use_sliding_window":false,"sliding_window":131072,"max_window_layers":28,
            "initializer_range":0.02,"bos_token_id":151643,"eos_token_id":151645,
            "torch_dtype":"bfloat16","transformers_version":"4.43.1","use_cache":true}),
    ] {
        let cfg = parse(&input).unwrap();
        assert_eq!(cfg.hidden_size, cfg.num_heads * cfg.head_dim());
        assert!(!cfg.tie_word_embeddings);
    }
}
#[test]
fn sentinel_rejects_unknown_conflicting_or_missing_architecture_identity() {
    for name in ["mistral", "gemma", "qwen3", "unknown", ""] {
        rejects("model_type", json!(name));
    }
    for names in [
        json!(["Qwen2ForCausalLM"]),
        json!(["LlamaModel"]),
        json!(["CustomForCausalLM"]),
        json!(["LlamaForCausalLM", "Qwen2ForCausalLM"]),
    ] {
        rejects("architectures", names);
    }
    let mut input = hf();
    input.as_object_mut().unwrap().remove("model_type");
    input["architectures"] = json!([]);
    assert!(parse(&input).unwrap_err().to_string().contains("identity"));
    let duplicate = serde_json::to_string(&hf()).unwrap().replace(
        "\"model_type\":\"llama\"",
        "\"model_type\":\"llama\",\"model_type\":\"qwen2\"",
    );
    assert!(TransformerConfig::from_hf_config_bytes(duplicate.as_bytes()).is_err());
    assert!(TransformerConfig::from_hf_config_bytes(b"[]").is_err());
    for duplicate in [
        "\"qk_norm\":true,\"qk_norm\":false",
        "\"auto_map\":{\"AutoModel\":\"custom.Model\"},\"auto_map\":{}",
        "\"rope_scaling\":{\"type\":\"linear\",\"factor\":2.0,\"factor\":1.0}",
        "\"rope_parameters\":{\"rope_type\":\"yarn\",\"rope_type\":\"default\"}",
    ] {
        let mut text = serde_json::to_string(&hf()).unwrap();
        text.pop();
        text.push(',');
        text.push_str(duplicate);
        text.push('}');
        let error = TransformerConfig::from_hf_config_bytes(text.as_bytes())
            .unwrap_err()
            .to_string();
        assert!(error.contains("duplicate"), "{error}");
    }
}
#[test]
fn sentinel_rejects_unsupported_activations_and_active_features() {
    for activation in ["relu", "gelu", "gelu_pytorch_tanh", ""] {
        rejects("hidden_act", json!(activation));
    }
    for activation in ["silu", "swish"] {
        let mut input = hf();
        input["hidden_act"] = json!(activation);
        assert_eq!(parse(&input).unwrap(), config());
    }
    for field in [
        "attention_bias",
        "mlp_bias",
        "is_encoder_decoder",
        "add_cross_attention",
        "tie_encoder_decoder",
        "trust_remote_code",
        "qk_layernorm",
        "qk_norm",
        "use_alibi",
        "alibi",
        "use_logn_attn",
    ] {
        rejects(field, json!(true));
    }
    rejects(
        "quantization_config",
        json!({"quant_method":"gptq","bits":4}),
    );
    rejects("auto_map", json!({"AutoModelForCausalLM":"custom.Model"}));
    rejects("cross_attention_hidden_size", json!(8));
    rejects("attn_logit_softcapping", json!(30.0));
    rejects("final_logit_softcapping", json!(30.0));
    rejects("position_embedding_type", json!("absolute"));
    rejects("unverified_attention_feature", json!(true));
}
#[test]
fn sentinel_accepts_classified_inactive_and_inference_metadata() {
    let mut input = hf();
    for (name, value) in [
        ("torch_dtype", json!("bfloat16")),
        ("transformers_version", json!("4.43.1")),
        ("initializer_range", json!(0.02)),
        ("bos_token_id", json!(1)),
        ("eos_token_id", json!([2, 3])),
        ("pad_token_id", Value::Null),
        ("use_cache", json!(false)),
        ("attention_dropout", json!(0.1)),
        ("pretraining_tp", json!(2)),
        ("is_encoder_decoder", json!(false)),
        ("qk_norm", json!(false)),
        ("quantization_config", Value::Null),
        ("auto_map", json!({})),
        ("attn_logit_softcapping", json!(0)),
        ("final_logit_softcapping", Value::Null),
    ] {
        input[name] = value;
    }
    assert_eq!(parse(&input).unwrap(), config());
    for rate in [json!(-0.1), json!(1.1), json!("not a rate")] {
        rejects("attention_dropout", rate);
    }
    rejects("pretraining_tp", json!(0));
}
#[test]
fn sentinel_validates_attention_geometry_without_panics() {
    for (field, values) in [
        ("hidden_size", vec![json!(0), json!(9), json!(6), json!(-1)]),
        ("num_attention_heads", vec![json!(0), json!(3), json!(16)]),
        ("num_key_value_heads", vec![json!(0), json!(3), json!(4)]),
        ("head_dim", vec![json!(0), json!(2), json!(6)]),
        ("rotary_dim", vec![json!(0), json!(2)]),
        ("vocab_size", vec![json!(0)]),
        ("intermediate_size", vec![json!(0)]),
        ("num_hidden_layers", vec![json!(0)]),
        ("max_position_embeddings", vec![json!(0)]),
    ] {
        for value in values {
            let result = std::panic::catch_unwind(|| {
                let mut input = hf();
                input[field] = value;
                parse(&input)
            });
            assert!(result.unwrap().is_err(), "{field} must be rejected");
        }
    }
    let mut input = hf();
    input["head_dim"] = json!(4);
    input["rotary_dim"] = json!(4);
    assert_eq!(parse(&input).unwrap(), config());
}
#[test]
fn sentinel_accepts_only_unscaled_full_head_rope() {
    for rope in [
        Value::Null,
        json!({}),
        json!({"rope_type":"default"}),
        json!({"type":"default","rope_type":"default","factor":1.0,"attention_factor":1.0,"partial_rotary_factor":1.0,
            "original_max_position_embeddings":8}),
        json!({"type":"linear","factor":1.0}),
    ] {
        let mut input = hf();
        input["rope_scaling"] = rope;
        input["partial_rotary_factor"] = json!(1.0);
        input["rope_interleaved"] = json!(false);
        assert_eq!(parse(&input).unwrap(), config());
    }
    let mut input = hf();
    input["rope_parameters"] = json!({"rope_type":"default","rope_theta":500000.0});
    assert_eq!(parse(&input).unwrap().rope_theta, 500000.0);
    for rope in [
        json!({"type":"linear","factor":2.0}),
        json!({"rope_type":"dynamic","factor":1.0}),
        json!({"rope_type":"yarn","factor":1.0}),
        json!({"rope_type":"longrope"}),
        json!({"rope_type":"llama3","factor":8.0}),
        json!({"rope_type":"unknown"}),
        json!({"type":"linear","rope_type":"default","factor":1.0}),
        json!({"rope_type":"default","factor":0.0}),
        json!({"rope_type":"default","factor":-1.0}),
        json!({"rope_type":"default","partial_rotary_factor":0.5}),
        json!({"rope_type":"default","attention_factor":2.0}),
        json!({"rope_type":"default","unknown_parameter":1}),
        json!({"full_attention":{"rope_type":"default"}}),
        json!("invalid"),
        json!([null, null, null, null, null, null, null]),
    ] {
        rejects("rope_scaling", rope.clone());
        rejects("rope_parameters", rope);
    }
    rejects("rope_interleaved", json!(true));
    rejects("partial_rotary_factor", json!(0.5));
    let mut input = hf();
    input["rope_theta"] = json!(10000);
    input["rope_parameters"] = json!({"rope_type":"default","rope_theta":500000});
    assert!(
        parse(&input)
            .unwrap_err()
            .to_string()
            .contains("conflicting")
    );
}
#[test]
fn sentinel_distinguishes_active_sliding_windows_from_inactive_metadata() {
    let mut input = hf();
    input["model_type"] = json!("qwen2");
    input["architectures"] = json!(["Qwen2ForCausalLM"]);
    input["rope_theta"] = json!(10000);
    input["use_sliding_window"] = json!(false);
    input["sliding_window"] = json!(131072);
    input["max_window_layers"] = json!(28);
    assert_eq!(parse(&input).unwrap(), config());
    input["use_sliding_window"] = json!(true);
    assert_eq!(parse(&input).unwrap(), config()); // No layers use the window.
    input["max_window_layers"] = json!(0);
    assert!(
        parse(&input)
            .unwrap_err()
            .to_string()
            .contains("sliding-window")
    );
    input["sliding_window"] = Value::Null;
    assert_eq!(parse(&input).unwrap(), config());
    input.as_object_mut().unwrap().remove("sliding_window"); // Qwen2 defaults to 4096.
    assert!(parse(&input).is_err());
    input["layer_types"] = json!(["full_attention"]);
    assert_eq!(parse(&input).unwrap(), config()); // Explicit layer types control execution.
    input["use_sliding_window"] = json!(false);
    for types in [
        json!(["sliding_attention"]),
        json!(["unknown"]),
        json!([]),
        json!(["full_attention", "full_attention"]),
    ] {
        input["layer_types"] = types;
        assert!(parse(&input).is_err());
    }
}
#[test]
fn sentinel_rejects_nonfinite_nonpositive_and_unrepresentable_constants() {
    for value in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -1.0,
        1e-300,
        1e300,
    ] {
        for field in ["rms_norm_eps", "rope_theta"] {
            let mut cfg = config();
            if field == "rope_theta" {
                cfg.rope_theta = value
            } else {
                cfg.rms_norm_eps = value
            }
            assert!(cfg.validate().is_err(), "{field}={value}");
        }
    }
    for field in ["rms_norm_eps", "rope_theta"] {
        rejects(field, json!(0));
        rejects(field, json!(-1));
    }
    let mut cfg = config();
    cfg.hidden_size = 256;
    cfg.rope_theta = 1e-38;
    cfg.max_seq_len = 128;
    assert!(
        cfg.validate()
            .unwrap_err()
            .to_string()
            .contains("RoPE angle")
    );
    let bad_json = serde_json::to_string(&hf())
        .unwrap()
        .replace("\"vocab_size\":8", "\"vocab_size\":1e309");
    assert!(TransformerConfig::from_hf_config_bytes(bad_json.as_bytes()).is_err());
}
#[test]
fn sentinel_rejects_dimension_products_before_allocation() {
    let mut cfg = config();
    cfg.vocab_size = 1 << 31;
    cfg.hidden_size = 1 << 30;
    assert!(
        cfg.validate()
            .unwrap_err()
            .to_string()
            .contains("embedding")
    );
    let mut cfg = config();
    cfg.max_seq_len = 1 << 30;
    cfg.num_layers = 1 << 30;
    assert!(cfg.validate().unwrap_err().to_string().contains("KV cache"));
    let mut cfg = config();
    cfg.num_layers = usize::MAX;
    assert!(cfg.validate().is_err());
    assert!(
        TransformerConfig::from_hf_config_bytes(&vec![b' '; 1024 * 1024 + 1])
            .unwrap_err()
            .to_string()
            .contains("1 MiB")
    );
}

#[test]
fn sentinel_execution_identity_binds_effective_settings_and_rejects_invalid_input() {
    let current = parse(&hf()).unwrap();
    let identity = current.execution_identity().unwrap();
    assert_eq!(identity, current.clone().execution_identity().unwrap());
    let mut previous_defaults = current.clone();
    previous_defaults.rms_norm_eps = 1e-5;
    assert_ne!(identity, previous_defaults.execution_identity().unwrap());
    let mut different_context = current.clone();
    different_context.max_seq_len = 4;
    assert_ne!(identity, different_context.execution_identity().unwrap());
    let mut invalid = current;
    invalid.num_heads = 0;
    assert!(invalid.execution_identity().is_err());
}

fn source(bias: Option<(&str, Vec<usize>, Vec<f32>)>) -> Vec<u8> {
    let bytes = generate_synthetic_llama_safetensors(&config(), 2, 7).unwrap();
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header: Value = serde_json::from_slice(&bytes[8..8 + len]).unwrap();
    let mut data = bytes[8 + len..].to_vec();
    if let Some((name, shape, values)) = bias {
        let start = data.len();
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header[name] = json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]});
    }
    pack_source(&header, &data)
}
fn pack_source(header: &Value, data: &[u8]) -> Vec<u8> {
    let mut text = serde_json::to_vec(header).unwrap();
    while !text.len().is_multiple_of(8) {
        text.push(b' ');
    }
    let mut result = (text.len() as u64).to_le_bytes().to_vec();
    result.extend(text);
    result.extend_from_slice(data);
    result
}
fn archive(bytes: &[u8]) -> QmfArchiveV1 {
    qml_forge::forge_safetensors_to_qmf(
        Cursor::new(bytes),
        "sentinel",
        "synthetic",
        "llama",
        &qml_forge::TransformStrategy::Bf16,
        &PayloadBudget::default(),
        42,
    )
    .unwrap()
    .0
}
fn encoded(archive: &QmfArchiveV1) -> Vec<u8> {
    let mut bytes = Vec::new();
    archive.write_to(&mut bytes).unwrap();
    bytes
}
fn reader(bytes: Vec<u8>) -> QmfFileReader<Cursor<Vec<u8>>> {
    QmfFileReader::open(Cursor::new(bytes), &QmfBudget::default()).unwrap()
}
fn crc(bytes: &[u8]) -> u32 {
    let mut state = !0_u32;
    for &byte in bytes {
        state ^= u32::from(byte);
        for _ in 0..8 {
            state = (state >> 1) ^ (0xedb88320 & 0_u32.wrapping_sub(state & 1));
        }
    }
    !state
}
fn rewrite(
    archive: &QmfArchiveV1,
    name: &str,
    mutate: impl FnOnce(&mut qml_format::v1::BlobRef, &mut Vec<u8>),
) -> Vec<u8> {
    let bytes = encoded(archive);
    let len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let mut payload = bytes[32 + len..bytes.len() - 4].to_vec();
    let mut manifest: QmfManifest = archive.manifest().clone();
    let entry = manifest
        .tensors
        .iter_mut()
        .find(|t| t.name == name)
        .unwrap();
    let QmfRepresentation::DenseF32 { data } = &mut entry.representation else {
        panic!("expected vector")
    };
    mutate(data, &mut payload);
    data.crc32 = crc(&payload[data.offset as usize..(data.offset + data.len) as usize]);
    let manifest = serde_json::to_vec(&manifest).unwrap();
    let mut result = bytes[..32].to_vec();
    result[8..16].copy_from_slice(&(manifest.len() as u64).to_le_bytes());
    result.extend(manifest);
    result.extend(payload);
    let sum = crc(&result);
    result.extend(sum.to_le_bytes());
    result
}
fn stream(bytes: &[u8], truncate_bias: bool) -> Result<StreamingCalibAndRefResult, RuntimeError> {
    let tmp = qml_core::publication::stage(std::path::Path::new("sentinel-test-model")).unwrap();
    std::fs::write(tmp.path(), bytes).unwrap();
    let mut truncated = false;
    stream_calibrate_and_reference(
        tmp.path(),
        "sentinel",
        &config(),
        &[],
        &[vec![0, 1, 2]],
        2,
        false,
        &PayloadBudget::default(),
        |_, _, _| {
            if truncate_bias && !truncated {
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(tmp.path())
                    .unwrap()
                    .set_len((bytes.len() - 1) as u64)
                    .unwrap();
                truncated = true;
            }
        },
    )
}
#[test]
fn sentinel_absent_and_valid_optional_biases_work_in_every_loader() {
    let base = source(None);
    let plain = archive(&base);
    let a = QCoreModel::from_qmf(&plain, config()).unwrap();
    assert!(
        a.layers[0].q_bias.is_none()
            && a.layers[0].k_bias.is_none()
            && a.layers[0].v_bias.is_none()
    );
    let no_bias_logits = stream(&base, false).unwrap().eval_logits;
    let logits = |model: &QCoreModel| {
        let mut cache = KvCache::new(1);
        [0, 1]
            .into_iter()
            .enumerate()
            .map(|(position, token)| model.forward_token(token, position, &mut cache).unwrap())
            .collect::<Vec<_>>()
    };
    let base_logits = logits(&a);
    for projection in ["q", "k", "v"] {
        let name = format!("model.layers.0.self_attn.{projection}_proj.bias");
        let len = if projection == "q" { 8 } else { 4 };
        let values: Vec<f32> = (0..len).map(|n| (n as f32 + 1.0) * 0.03).collect();
        let bytes = source(Some((&name, vec![len], values)));
        let valid = archive(&bytes);
        let memory = QCoreModel::from_qmf(&valid, config()).unwrap();
        let file = QCoreModel::from_qmf_reader(&mut reader(encoded(&valid)), config()).unwrap();
        let actual = match projection {
            "q" => &memory.layers[0].q_bias,
            "k" => &memory.layers[0].k_bias,
            _ => &memory.layers[0].v_bias,
        };
        assert_eq!(actual.as_ref().unwrap().len(), len);
        let m = logits(&memory);
        let f = logits(&file);
        assert_eq!(m, f);
        let streamed = stream(&bytes, false).unwrap().eval_logits;
        assert_ne!(m, base_logits);
        assert_ne!(streamed, no_bias_logits);
    }
}
#[test]
fn sentinel_wrong_bias_rank_and_length_fail_in_every_loader() {
    for projection in ["q", "k", "v"] {
        let name = format!("model.layers.0.self_attn.{projection}_proj.bias");
        let len = if projection == "q" { 8 } else { 4 };
        for shape in [vec![1, len], vec![len - 1], vec![len + 1]] {
            let count: usize = shape.iter().product();
            let bytes = source(Some((&name, shape, vec![0.0; count])));
            let invalid = archive(&bytes);
            let memory = QCoreModel::from_qmf(&invalid, config())
                .err()
                .unwrap()
                .to_string();
            let file = QCoreModel::from_qmf_reader(&mut reader(encoded(&invalid)), config())
                .err()
                .unwrap()
                .to_string();
            let streamed = stream(&bytes, false).unwrap_err().to_string();
            for error in [memory, file, streamed] {
                assert!(error.contains(&name) && error.contains("shape"), "{error}");
            }
        }
    }
}
#[test]
fn sentinel_nonfinite_biases_with_valid_checksums_fail_in_memory_and_file() {
    for projection in ["q", "k", "v"] {
        let name = format!("model.layers.0.self_attn.{projection}_proj.bias");
        let len = if projection == "q" { 8 } else { 4 };
        let valid = archive(&source(Some((&name, vec![len], vec![0.0; len]))));
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let bytes = rewrite(&valid, &name, |data, payload| {
                payload[data.offset as usize..data.offset as usize + 4]
                    .copy_from_slice(&value.to_le_bytes());
            });
            let archive = QmfArchiveV1::read_from(Cursor::new(&bytes)).unwrap();
            let memory = QCoreModel::from_qmf(&archive, config())
                .err()
                .unwrap()
                .to_string();
            let file = QCoreModel::from_qmf_reader(&mut reader(bytes), config())
                .err()
                .unwrap()
                .to_string();
            for error in [memory, file] {
                assert!(error.contains(&name), "{error}");
            }
        }
    }
}
#[test]
fn sentinel_nonfinite_and_truncated_streaming_biases_are_errors() {
    for projection in ["q", "k", "v"] {
        let name = format!("model.layers.0.self_attn.{projection}_proj.bias");
        let len = if projection == "q" { 8 } else { 4 };
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut values = vec![0.0; len];
            values[0] = value;
            let error = stream(&source(Some((&name, vec![len], values))), false)
                .unwrap_err()
                .to_string();
            assert!(error.contains(&name), "{error}");
        }
        let bytes = source(Some((&name, vec![len], vec![0.0; len])));
        let error = stream(&bytes, true).unwrap_err().to_string();
        assert!(error.contains(&name), "{error}");
    }
}
#[test]
fn sentinel_bias_payload_length_and_crc_failures_are_not_absence() {
    for projection in ["q", "k", "v"] {
        let name = format!("model.layers.0.self_attn.{projection}_proj.bias");
        let len = if projection == "q" { 8 } else { 4 };
        let valid = archive(&source(Some((&name, vec![len], vec![0.0; len]))));
        let shortened = rewrite(&valid, &name, |data, _| data.len -= 4);
        let memory = QmfArchiveV1::read_from(Cursor::new(&shortened)).unwrap();
        let header_len = u64::from_le_bytes(shortened[8..16].try_into().unwrap());
        let entry = valid
            .manifest()
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap();
        let QmfRepresentation::DenseF32 { data } = entry.representation else {
            panic!("expected vector")
        };
        let start = 32 + header_len + data.offset;
        let mut no_payload_read = QmfFileReader::open(
            FailingRead {
                inner: Cursor::new(shortened.clone()),
                range: start..start + data.len,
            },
            &QmfBudget::default(),
        )
        .unwrap();
        let error = QCoreModel::from_qmf_reader(&mut no_payload_read, config())
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("payload byte length") && !error.contains("injected"),
            "{error}"
        );
        assert!(
            QCoreModel::from_qmf(&memory, config())
                .err()
                .unwrap()
                .to_string()
                .contains(&name)
        );
        assert!(
            QCoreModel::from_qmf_reader(&mut reader(shortened), config())
                .err()
                .unwrap()
                .to_string()
                .contains(&name)
        );
        let mut corrupt = encoded(&valid);
        let manifest_len = u64::from_le_bytes(corrupt[8..16].try_into().unwrap()) as usize;
        let entry = valid
            .manifest()
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap();
        let QmfRepresentation::DenseF32 { data } = entry.representation else {
            panic!("expected vector")
        };
        corrupt[32 + manifest_len + data.offset as usize] ^= 1;
        assert!(QmfArchiveV1::read_from(Cursor::new(&corrupt)).is_err());
        assert!(
            QCoreModel::from_qmf_reader(&mut reader(corrupt), config())
                .err()
                .unwrap()
                .to_string()
                .contains(&name)
        );
    }
}
struct FailingRead {
    inner: Cursor<Vec<u8>>,
    range: std::ops::Range<u64>,
}
impl Read for FailingRead {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if self.range.contains(&self.inner.position()) {
            return Err(io::Error::other("injected optional tensor I/O failure"));
        }
        self.inner.read(output)
    }
}
impl Seek for FailingRead {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.inner.seek(position)
    }
}
#[test]
fn sentinel_file_bias_io_failures_keep_tensor_context() {
    for projection in ["q", "k", "v"] {
        let name = format!("model.layers.0.self_attn.{projection}_proj.bias");
        let len = if projection == "q" { 8 } else { 4 };
        let valid = archive(&source(Some((&name, vec![len], vec![0.0; len]))));
        let bytes = encoded(&valid);
        let manifest_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let entry = valid
            .manifest()
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap();
        let QmfRepresentation::DenseF32 { data } = entry.representation else {
            panic!("expected vector")
        };
        let start = 32 + manifest_len + data.offset;
        let mut reader = QmfFileReader::open(
            FailingRead {
                inner: Cursor::new(bytes),
                range: start..start + data.len,
            },
            &QmfBudget::default(),
        )
        .unwrap();
        let error = QCoreModel::from_qmf_reader(&mut reader, config())
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains(&name) && error.contains("injected"),
            "{error}"
        );
    }
}

#[test]
fn sentinel_half_precision_nonfinite_biases_are_errors_not_valid_optional_values() {
    for projection in ["q", "k", "v"] {
        let name = format!("model.layers.0.self_attn.{projection}_proj.bias");
        let len = if projection == "q" { 8 } else { 4 };
        let valid = archive(&source(Some((&name, vec![len], vec![0.0; len]))));
        for bf16 in [false, true] {
            for bits in if bf16 {
                [0x7fc0_u16, 0x7f80, 0xff80]
            } else {
                [0x7e00_u16, 0x7c00, 0xfc00]
            } {
                let encoded = rewrite(&valid, &name, |data, payload| {
                    data.len = (len * 2) as u64;
                    payload[data.offset as usize..data.offset as usize + 2]
                        .copy_from_slice(&bits.to_le_bytes());
                });
                let manifest_len = u64::from_le_bytes(encoded[8..16].try_into().unwrap()) as usize;
                let mut manifest: QmfManifest =
                    serde_json::from_slice(&encoded[32..32 + manifest_len]).unwrap();
                let entry = manifest
                    .tensors
                    .iter_mut()
                    .find(|entry| entry.name == name)
                    .unwrap();
                let QmfRepresentation::DenseF32 { data } = entry.representation else {
                    panic!("expected vector")
                };
                entry.representation = if bf16 {
                    QmfRepresentation::DenseBf16 { data }
                } else {
                    QmfRepresentation::DenseF16 { data }
                };
                let text = serde_json::to_vec(&manifest).unwrap();
                let mut bytes = encoded[..32].to_vec();
                bytes[8..16].copy_from_slice(&(text.len() as u64).to_le_bytes());
                bytes.extend(text);
                bytes.extend(&encoded[32 + manifest_len..encoded.len() - 4]);
                let checksum = crc(&bytes);
                bytes.extend(checksum.to_le_bytes());
                let memory = QmfArchiveV1::read_from(Cursor::new(&bytes)).unwrap();
                let error = QCoreModel::from_qmf(&memory, config())
                    .err()
                    .unwrap()
                    .to_string();
                assert!(error.contains(&name) && error.contains("finite"), "{error}");
                let error = QCoreModel::from_qmf_reader(&mut reader(bytes), config())
                    .err()
                    .unwrap()
                    .to_string();
                assert!(error.contains(&name) && error.contains("finite"), "{error}");
                let source = source(Some((&name, vec![len], vec![0.0; len])));
                let header_len = u64::from_le_bytes(source[..8].try_into().unwrap()) as usize;
                let mut header: Value = serde_json::from_slice(&source[8..8 + header_len]).unwrap();
                let offset = header[&name]["data_offsets"][0].as_u64().unwrap() as usize;
                let mut data = source[8 + header_len..8 + header_len + offset].to_vec();
                data.resize(offset + len * 2, 0);
                data[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
                header[&name]["dtype"] = json!(if bf16 { "BF16" } else { "F16" });
                header[&name]["data_offsets"] = json!([offset, data.len()]);
                let error = stream(&pack_source(&header, &data), false)
                    .unwrap_err()
                    .to_string();
                assert!(error.contains(&name) && error.contains("finite"), "{error}");
            }
        }
    }
}
