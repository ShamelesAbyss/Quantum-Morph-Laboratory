# Quantum Morph Laboratory (QML)

An experimental, local-first research laboratory for quantum-inspired neural model transformation, compression, evolutionary optimization, and native inference.

> **Research status:** early proof of concept. “Quantum-inspired” here means classical computation motivated by mathematical ideas; it is not a claim of physical quantum computation or quantum advantage. RMQ, MAP, and QMF are research proposals, not established discoveries. The current rank-one operation and QMF container are narrow experimental baselines; no pretrained-model transformation or inference pipeline is implemented.

## Current implementation

This repository currently contains a small Rust workspace with:

- `qml-core`: validated row-major dense matrices, matrix-vector multiplication, a deterministic largest-pivot rank-one cross approximation, factorized matrix-vector multiplication, and Frobenius reconstruction error.
- `qml-format`: a bounded reader and deterministic writer for a single rank-one `f64` matrix in experimental QMF v0. This is not a model container or compatibility-stable format.
- `qml`: help/version/diagnostic commands, a synthetic `demo-rank1`, and `inspect-qmf` for validating the narrow experimental container.

`qml demo-rank1` creates a generated rank-one matrix, writes and reloads its factors, and reports storage and error values for that fixture only. It is not a pretrained model or a general benchmark. The pivot approximation is a simple conventional baseline and is not guaranteed optimal for arbitrary matrices. `qml doctor` reports the operating system, architecture, and available logical parallelism; it does not detect memory, GPUs, drivers, or accelerator backends. Safetensors/GGUF inspection, model transformation, quantization, benchmarking, and inference are not implemented.

## Build and test

Install a stable Rust toolchain, then run:

```sh
cargo fmt --all -- --check
cargo test --workspace
cargo run -p qml-cli -- doctor
cargo run -p qml-cli -- demo-rank1 --output target/qml-demo.qmf
cargo run -p qml-cli -- inspect-qmf target/qml-demo.qmf
```

See [BUILD.md](BUILD.md) for more detail.

## Research status

The first source-backed prior-art batch covers 17 topics spanning common quantization methods, tensor and low-rank representations, codebooks, rate-distortion coding, and mixed precision. The [prior-art index](research/prior-art-index.md) links each topic note; a separate [synthesis](research/prior-art-synthesis.md) compares those methods and documents evidence limits.

This is not a complete literature review. Behavioral transformation, mechanistic interpretability, distillation, evolutionary search, quantum-inspired optimization, complex-valued networks, several named methods, and other model formats still need review. The current batch does not establish novelty, priority, freedom to operate, or non-infringement for any QML proposal. The [novelty ledger](research/novelty-ledger.md) records the bounded status of RMQ, MAP, and QMF.

No results in the review were independently reproduced by QML. Reported measurements are attributed to the original papers or maintainers. Research notes state where compute details, source licensing, model-weight rights, and methodology remain unverified.

The directive attachments received so far are incomplete: the master document ends mid-sentence in the QMF section, and the second attachment identifies itself as “Continuation 1 of 2.” The missing material has not been inferred or invented. Foundational work here is limited to requirements that are clear from the available text. QMF v0's current layout and bounds are documented in [the experimental format note](research/qmf-experimental-v0.md).

## Principles

Rust is the primary implementation language; Python is not part of the application or development toolchain. Imported model artifacts should remain immutable by default, and weights or private experimental data must never be committed. Transformations must be independently configurable and measurable, representation overhead must be counted, and estimates must be distinguished from measured results. Establish conventional baselines and review prior art before making novelty or performance claims. Local-first behavior means no required account, cloud service, telemetry, or remote model-weight transfer. QMF CRC-32 is an accidental-corruption check only, not authentication; do not use the prototype format for important artifacts.

## Licensing

The repository is public, but no project license has been selected yet. Until that decision is made, do not assume permission to reuse or redistribute this code beyond rights provided by applicable law. A license should be chosen before external contributions or releases.
