use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

fn synthetic_file() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "qml-safetensors-cli-{}-{}.safetensors",
        std::process::id(),
        NEXT_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    let header = r#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"__metadata__":{"kind":"generated-test"}}"#;
    let padded_len = header.len().div_ceil(8) * 8;
    let mut bytes = Vec::with_capacity(8 + padded_len + 1);
    bytes.extend_from_slice(&(padded_len as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(8 + padded_len, b' ');
    bytes.push(7);
    fs::write(&path, bytes).expect("write temporary synthetic fixture");
    path
}

#[test]
fn emits_human_readable_header_summary() {
    let path = synthetic_file();
    let output = Command::new(env!("CARGO_BIN_EXE_qml"))
        .args(["inspect-safetensors", path.to_str().unwrap()])
        .output()
        .expect("run CLI");
    fs::remove_file(path).unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("payload not read"));
    assert!(stdout.contains("weight: U8 [1]"));
    assert!(stdout.contains("generated-test"));
}

#[test]
fn emits_machine_readable_json() {
    let path = synthetic_file();
    let output = Command::new(env!("CARGO_BIN_EXE_qml"))
        .args(["inspect-safetensors", "--json", path.to_str().unwrap()])
        .output()
        .expect("run CLI");
    fs::remove_file(path).unwrap();
    assert!(output.status.success());
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["data_bytes"], 1);
    assert_eq!(value["tensors"][0]["name"], "weight");
    assert_eq!(value["tensors"][0]["dtype"], "U8");
}

#[test]
fn reports_usage_errors_with_exit_code_two() {
    let output = Command::new(env!("CARGO_BIN_EXE_qml"))
        .arg("inspect-safetensors")
        .output()
        .expect("run CLI");
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("usage: qml inspect-safetensors")
    );
}
