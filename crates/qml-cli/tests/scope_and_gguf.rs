use qml_scope::generate_synthetic_transformer_safetensors;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

fn temp_path(ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "qml-scope-cli-{}-{}.{ext}",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

#[test]
fn scope_cli_emits_human_and_json_reports() {
    let bytes = generate_synthetic_transformer_safetensors(2, 16, 32, 3, 123).unwrap();
    let path = temp_path("safetensors");
    fs::write(&path, bytes).unwrap();

    let text_out = Command::new(env!("CARGO_BIN_EXE_qml"))
        .args(["scope", "--svd-rank", "6", path.to_str().unwrap()])
        .output()
        .expect("run qml scope");
    assert!(
        text_out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&text_out.stderr)
    );
    let stdout = String::from_utf8(text_out.stdout).unwrap();
    assert!(stdout.contains("Q-Scope Neural Model Observatory Report"));
    assert!(stdout.contains("Cross-Layer Subspace Relationships"));

    let json_out = Command::new(env!("CARGO_BIN_EXE_qml"))
        .args(["scope", "--json", "--svd-rank", "6", path.to_str().unwrap()])
        .output()
        .expect("run qml scope --json");
    fs::remove_file(&path).unwrap();
    assert!(json_out.status.success());
    let parsed: serde_json::Value = serde_json::from_slice(&json_out.stdout).unwrap();
    assert_eq!(parsed["summary"]["total_tensors"], 12);
    assert_eq!(parsed["summary"]["svd_analyzed_matrices"], 10);
    assert!(parsed["cross_layer_relations"].as_array().unwrap().len() >= 4);
}

#[test]
fn doctor_reports_simd_and_parallelism() {
    let out = Command::new(env!("CARGO_BIN_EXE_qml"))
        .arg("doctor")
        .output()
        .expect("run qml doctor");
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("CPU SIMD features:"));
    assert!(stdout.contains("Available logical parallelism:"));
}
