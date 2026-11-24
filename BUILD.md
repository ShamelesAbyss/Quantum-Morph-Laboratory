# Build

## Requirements

- A stable Rust toolchain with `rustc` and `cargo`.
- No Python, model weights, GPU, or network connection is required to build or test the current workspace.

## Validate

From the repository root:

```sh
cargo fmt --all -- --check
cargo test --workspace
cargo run -p qml-cli -- --help
cargo run -p qml-cli -- doctor
```

`qml doctor` currently reports only OS, architecture, and logical parallelism. It does not inspect memory, GPU devices, disk space, or compute backends.
