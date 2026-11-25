# Quantum Morph Laboratory (QML)

**Don't just compress intelligence. Transform its representation.**

An experimental, local-first research laboratory for quantum-inspired neural model transformation, compression, and native inference research.

> **Scientific status:** QML is an early research prototype. “Quantum-inspired” refers to classical mathematics; it does not claim physical quantum computation or quantum advantage. RMQ, MAP, and QMF are proposals or experimental baselines, not established discoveries. No pretrained-model transformation or inference pipeline is implemented.

## What works today

- `qml-core` provides validated row-major dense matrices, matrix-vector multiplication, a deterministic largest-pivot rank-one cross approximation, factorized matrix-vector multiplication, and Frobenius reconstruction error.
- `qml-format` provides a bounded reader and deterministic writer for one rank-one `f64` matrix in experimental QMF v0. It is not a model container or compatibility-stable format.
- `qml-model` provides bounded, read-only Safetensors metadata inspection. It reads the header but never reads tensor payload bytes.
- `qml` provides help/version/diagnostic commands, a synthetic `demo-rank1`, `inspect-qmf`, and `inspect-safetensors` with text or JSON output.

The synthetic rank-one demo reports actual sizes and errors for a generated fixture only. It is not a pretrained-model result or general benchmark. The pivot approximation is a conventional baseline and is not guaranteed optimal for arbitrary matrices. `qml doctor` reports the operating system, architecture, and available logical parallelism; it does not inspect memory, GPUs, drivers, or accelerator backends.

## Research pipeline

```text
Safetensors file
       │
       ▼
Bounded header read ──► dtype / shape / offset validation ──► text or JSON metadata
       │                                                           │
       └── payload bytes are not read                              └── no model transformation yet

Synthetic matrix ──► conventional rank-one baseline ──► direct factorized matvec
                                                      └──► experimental QMF v0 round trip
```

The Safetensors inspector validates the UTF-8 JSON header, known dtype widths, checked shape arithmetic, byte alignment, offset spans, contiguous layout, and exact file length. It rejects duplicate JSON keys, unknown fields or dtypes, invalid offsets, excessive dimensions/counts, and malformed headers. The default header allocation cap is 16 MiB; callers may lower it, but not raise it. Header validation does not verify numerical tensor values or architecture. See [the source-backed format note](research/safetensors-inspection.md) and [ADR 0001](docs/decisions/0001-bounded-safetensors-inspection.md).

## Build, test, inspect

Install a stable Rust toolchain. The application does not require Python, model weights, a GPU, or a cloud service. Runtime dependencies are Rust crates; the upstream Safetensors parser is used only as a pinned test oracle.

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --workspace
cargo install cargo-deny --version 0.21.0 --locked
cargo deny check
cargo run -p qml-cli -- --help
cargo run -p qml-cli -- inspect-safetensors ./models/example.safetensors
cargo run -p qml-cli -- inspect-safetensors --json ./models/example.safetensors
cargo run -p qml-cli -- demo-rank1 --output target/qml-demo.qmf
cargo run -p qml-cli -- inspect-qmf target/qml-demo.qmf
```

The Safetensors commands require a local file path and do not download model files. Do not add private weights or datasets to Git. Use synthetic test fixtures when validating parser changes. More details are in [BUILD.md](BUILD.md).

## Research status

The first source-backed prior-art batch covers 17 topics spanning common quantization methods, tensor and low-rank representations, codebooks, rate-distortion coding, and mixed precision. The [prior-art index](research/prior-art-index.md) links each topic note; a separate [synthesis](research/prior-art-synthesis.md) compares those methods and documents evidence limits.

The literature review remains incomplete. Behavioral transformation, mechanistic interpretability, distillation, evolutionary search, quantum-inspired optimization, complex-valued networks, several named methods, and additional model formats still need review. The current batch establishes no novelty, priority, freedom to operate, or non-infringement conclusion for any QML proposal. The [novelty ledger](research/novelty-ledger.md) records the bounded status of RMQ, MAP, and QMF. No cited result has been independently reproduced by QML.

## Principles and roadmap

Rust is the primary implementation language; Python is not part of the application or development toolchain. Imported model artifacts should remain immutable by default, and weights or private experimental data must never be committed. Keep local-first behavior: no required account, cloud service, telemetry, API subscription, or remote model-weight transfer. Distinguish measured results from estimates and hypotheses. Count all representation overhead before claiming compression.

The next proof-of-concept work is a separately bounded tensor-payload reader for small legal fixtures, conventional numerical baselines on tensor data, and a defined structural approximation compared against those baselines. Tensor analysis, transformations, quantization, benchmarking, inference, hardware discovery, behavioral research, evolutionary search, and desktop UI remain future work. See [ROADMAP.md](ROADMAP.md) for current status and [MASTER_DEVELOPMENT_DIRECTIVE.md](MASTER_DEVELOPMENT_DIRECTIVE.md) for the complete user directive.

QMF v0 is experimental and unsuitable for important artifacts. Its CRC-32 detects accidental corruption but does not authenticate data.

## Contributing and licensing

Read [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md), and [HANDOFF.md](HANDOFF.md) before making changes. CI targets Linux and Windows and includes format, lint, build, tests, and dependency policy checks. The project has no chosen license. Public visibility does not grant permission to reuse or redistribute QML code; do not add a project license without the owner's approval. Third-party dependency licenses are checked separately.
