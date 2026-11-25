# Build and validation

## Requirements

A stable Rust toolchain with `rustc` and `cargo` is required. No Python, model weights, GPU, account, or cloud connection is required to build or run the application. Dependency checks need network access to update the RustSec advisory database. The `safetensors` crate is a development-only test oracle.

To run the dependency-policy check locally, install the same pinned tool version used in CI:

```sh
cargo install cargo-deny --version 0.21.0 --locked
```

## Validate

From the repository root:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --workspace
cargo deny check
```

`cargo deny check` validates security advisories, dependency sources, and third-party license policy. The project itself has no selected license. Workspace packages are marked unpublished until the owner selects a license; this does not grant or select a project license.

## CLI smoke tests

```sh
cargo run -p qml-cli -- --help
cargo run -p qml-cli -- doctor
cargo run -p qml-cli -- inspect-safetensors PATH
cargo run -p qml-cli -- inspect-safetensors --json PATH
cargo run -p qml-cli -- demo-rank1 --output target/qml-demo.qmf
cargo run -p qml-cli -- inspect-qmf target/qml-demo.qmf
```

Safetensors inspection seeks to determine file size and reads only the eight-byte prefix plus a bounded header (16 MiB maximum by default). It validates metadata layout but does not read or inspect payload values. It does not load model weights, infer architecture, or transform a model. See [the Safetensors format note](research/safetensors-inspection.md) and [the decision record](docs/decisions/0001-bounded-safetensors-inspection.md).

The rank-one demo uses a generated synthetic matrix, not a pretrained model or general benchmark. It reports the actual file and factor sizes for that fixture and checks matrix-vector output before and after serialization. QMF v0 is an unstable single-matrix prototype; see [its format note](research/qmf-experimental-v0.md), and do not use it for important artifacts.

`qml doctor` currently reports only OS, architecture, and logical parallelism. It does not inspect memory, GPU devices, disk space, or compute backends. Do not commit weights, credentials, private datasets, or personal data.
