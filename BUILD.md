# Build

## Requirements

- A stable Rust toolchain with `rustc` and `cargo`.
- No Python, model weights, GPU, or network connection is required to build or test the current workspace.

## Validate

From the repository root:

```sh
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p qml-cli -- --help
cargo run -p qml-cli -- doctor
cargo run -p qml-cli -- demo-rank1 --output target/qml-demo.qmf
cargo run -p qml-cli -- inspect-qmf target/qml-demo.qmf
```

The demo uses a generated synthetic matrix, not a pretrained model or general benchmark. It reports the actual file and factor sizes for that fixture and checks matrix-vector output before and after serialization. QMF v0 is an unstable single-matrix prototype; see [its format note](research/qmf-experimental-v0.md), and do not use it for important artifacts.

`qml doctor` currently reports only OS, architecture, and logical parallelism. It does not inspect memory, GPU devices, disk space, or compute backends. No Python runtime or third-party Rust package is required by the current workspace.
