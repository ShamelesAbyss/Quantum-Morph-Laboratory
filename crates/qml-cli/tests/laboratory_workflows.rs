use qml_scope::generate_synthetic_transformer_safetensors;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(0);

fn temp_file(ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "qml-lab-cli-{}-{}.{ext}",
        std::process::id(),
        NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

#[test]
fn morph_inspect_qmf_v1_and_compare_end_to_end() {
    let model_bytes = generate_synthetic_transformer_safetensors(3, 24, 48, 4, 99).unwrap();
    let src_path = temp_file("safetensors");
    let qmf_path = temp_file("qmf");
    fs::write(&src_path, model_bytes).unwrap();

    let morph_out = Command::new(env!("CARGO_BIN_EXE_qml"))
        .args([
            "morph",
            src_path.to_str().unwrap(),
            "--method",
            "rmq",
            "--rank",
            "4",
            "--output",
            qmf_path.to_str().unwrap(),
            "--json",
        ])
        .output()
        .expect("run qml morph");
    assert!(
        morph_out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&morph_out.stderr)
    );
    let morph_json: serde_json::Value = serde_json::from_slice(&morph_out.stdout).unwrap();
    assert!(morph_json["shared_latent_count"].as_u64().unwrap() >= 8);

    let inspect_out = Command::new(env!("CARGO_BIN_EXE_qml"))
        .args(["inspect-qmf", "--json", qmf_path.to_str().unwrap()])
        .output()
        .expect("run qml inspect-qmf");
    assert!(inspect_out.status.success());
    let inspect_json: serde_json::Value = serde_json::from_slice(&inspect_out.stdout).unwrap();
    assert_eq!(inspect_json["format_status"], "experimental-qmf-v1");

    let cmp_out = Command::new(env!("CARGO_BIN_EXE_qml"))
        .args([
            "compare",
            "--json",
            src_path.to_str().unwrap(),
            qmf_path.to_str().unwrap(),
        ])
        .output()
        .expect("run qml compare");
    fs::remove_file(&src_path).unwrap();
    fs::remove_file(&qmf_path).unwrap();
    assert!(cmp_out.status.success());
    let cmp_json: serde_json::Value = serde_json::from_slice(&cmp_out.stdout).unwrap();
    assert_eq!(cmp_json["tensors_compared"], 15);
}

#[test]
fn baseline_rmq_ablate_infer_demo_and_genesis_cli_succeed() {
    let base = Command::new(env!("CARGO_BIN_EXE_qml"))
        .args(["baseline", "--synthetic", "--json"])
        .output()
        .unwrap();
    assert!(base.status.success());
    let base_arr: serde_json::Value = serde_json::from_slice(&base.stdout).unwrap();
    assert!(base_arr.as_array().unwrap().len() >= 8);

    let ablate = Command::new(env!("CARGO_BIN_EXE_qml"))
        .args([
            "rmq-ablate",
            "--layers",
            "4",
            "--dim",
            "24",
            "--rank",
            "4",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(ablate.status.success());

    let infer = Command::new(env!("CARGO_BIN_EXE_qml"))
        .args(["infer-demo", "--json"])
        .output()
        .unwrap();
    assert!(infer.status.success());
    let infer_arr: serde_json::Value = serde_json::from_slice(&infer.stdout).unwrap();
    assert_eq!(infer_arr.as_array().unwrap().len(), 5);

    let genesis = Command::new(env!("CARGO_BIN_EXE_qml"))
        .args([
            "genesis",
            "--synthetic",
            "--population",
            "6",
            "--generations",
            "2",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(genesis.status.success());
    let g_val: serde_json::Value = serde_json::from_slice(&genesis.stdout).unwrap();
    assert!(!g_val["pareto_frontier"].as_array().unwrap().is_empty());
}
