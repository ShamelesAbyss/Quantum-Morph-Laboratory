use std::process::Command;

#[test]
fn morph_exposes_independent_limits_and_explicit_overwrite() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.safetensors");
    let output = dir.path().join("artifact.qmf");
    let scratch = dir.path().join("scratch");
    let bytes = qml_scope::generate_synthetic_transformer_safetensors(1, 16, 32, 4, 7).unwrap();
    std::fs::write(&source, &bytes).unwrap();
    std::fs::write(&output, b"existing artifact").unwrap();
    let command = || {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_qml"));
        cmd.arg("morph")
            .arg(&source)
            .args(["--method", "bf16", "--output"])
            .arg(&output)
            .arg("--scratch-dir")
            .arg(&scratch)
            .args([
                "--max-artifact-bytes",
                "68719476736",
                "--max-builder-memory-bytes",
                "1048576",
                "--max-temp-bytes",
                "1048576",
                "--max-tensor-bytes",
                "1048576",
                "--max-manifest-bytes",
                "16384",
                "--json",
            ]);
        cmd
    };
    assert!(!command().output().unwrap().status.success());
    assert_eq!(std::fs::read(&output).unwrap(), b"existing artifact");
    let result = command().arg("--overwrite").output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(
        report["qmf_serialized_bytes"].as_u64().unwrap(),
        std::fs::metadata(&output).unwrap().len()
    );
    let result = command()
        .args(["--overwrite", "--max-tensor-bytes", "1"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(std::fs::read(&source).unwrap(), bytes);
    assert_eq!(std::fs::read_dir(&scratch).unwrap().count(), 0);
}
