# Quantum Morph Laboratory (QML)

An experimental, local-first research laboratory for quantum-inspired neural model transformation, compression, evolutionary optimization, and native inference.

> **Research status:** early scaffold. “Quantum-inspired” here means classical computation motivated by mathematical ideas; it is not a claim of physical quantum computation or quantum advantage. RMQ, MAP, and QMF are research proposals, not established discoveries. No model transformation or inference pipeline is implemented yet.

## Current implementation

This repository currently contains a small Rust workspace with:

- `qml-core`: validated row-major dense matrix storage, matrix-vector multiplication, and Frobenius-norm primitives, with unit tests.
- `qml`: a minimal CLI with `qml --help`, `qml --version`, and `qml doctor`.

`qml doctor` reports the operating system, architecture, and available logical parallelism. It does not yet detect memory, GPUs, drivers, or accelerator backends. No command claims to inspect, transform, quantize, benchmark, or run a pretrained model.

## Build and test

Install a stable Rust toolchain, then run:

```sh
cargo fmt --all -- --check
cargo test --workspace
cargo run -p qml-cli -- doctor
```

See [BUILD.md](BUILD.md) for more detail.

## Research status

The first source-backed prior-art batch covers 17 topics spanning common quantization methods, tensor and low-rank representations, codebooks, rate-distortion coding, and mixed precision. The [prior-art index](research/prior-art-index.md) links each topic note; a separate [synthesis](research/prior-art-synthesis.md) compares those methods and documents evidence limits.

This is not a complete literature review. Behavioral transformation, mechanistic interpretability, distillation, evolutionary search, quantum-inspired optimization, complex-valued networks, several named methods, and other model formats still need review. The current batch does not establish novelty, priority, freedom to operate, or non-infringement for any QML proposal. The [novelty ledger](research/novelty-ledger.md) records the bounded status of RMQ, MAP, and QMF.

No results in the review were independently reproduced by QML. Reported measurements are attributed to the original papers or maintainers. Research notes state where compute details, source licensing, model-weight rights, and methodology remain unverified.

The directive attachments received so far are incomplete: the master document ends mid-sentence in the QMF section, and the second attachment identifies itself as “Continuation 1 of 2.” The missing material has not been inferred or invented. Foundational work here is limited to requirements that are clear from the available text.

## Principles

Rust is the primary implementation language; Python is not part of the application or development toolchain. Imported model artifacts should remain immutable by default, and weights or private experimental data must never be committed. Transformations must be independently configurable and measurable, representation overhead must be counted, and estimates must be distinguished from measured results. Establish conventional baselines and review prior art before making novelty or performance claims. Local-first behavior means no required account, cloud service, telemetry, or remote model-weight transfer.

## Licensing

The repository is public, but no project license has been selected yet. Until that decision is made, do not assume permission to reuse or redistribute this code beyond rights provided by applicable law. A license should be chosen before external contributions or releases.
