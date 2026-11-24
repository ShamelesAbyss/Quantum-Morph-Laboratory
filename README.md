# Quantum Morph Laboratory (QML)

An experimental, local-first research laboratory for quantum-inspired neural model transformation, compression, evolutionary optimization, and native inference.

> **Research status:** early scaffold. “Quantum-inspired” here means classical computation motivated by mathematical ideas; it is not a claim of physical quantum computation or quantum advantage. RMQ and MAP are working research proposals, not established discoveries. No model transformation or inference pipeline is implemented yet.

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

## Principles

- Rust is the primary implementation language; Python is not part of the application or development toolchain.
- Preserve imported model artifacts immutably by default; never commit model weights or private experimental data.
- Keep transformations independently configurable and measurable.
- Count all representation overhead and distinguish measured results from estimates.
- Establish conventional baselines and review prior art before making novelty or performance claims.
- Preserve failed experiments and document limitations.
- Local-first behavior: no required account, cloud service, telemetry, or remote model-weight transfer.

## Research status and limitations

The supplied development directive calls for a broad prior-art review and a research novelty ledger. The ledger has been initialized, but the comprehensive literature review has not yet been completed. No new method is claimed. See [`research/`](research/), [`ARCHITECTURE.md`](ARCHITECTURE.md), and [`ROADMAP.md`](ROADMAP.md).

The directive attachments received so far are incomplete: the master document ends mid-sentence in the QMF section, and the second attachment identifies itself as “Continuation 1 of 2.” The missing continuation has not been inferred or invented. Foundational work here is intentionally limited to requirements that are clear from the available text.

## Licensing

The repository is public, but no project license has been selected yet. Until that decision is made, do not assume permission to reuse or redistribute this code beyond rights provided by applicable law. A license should be chosen before external contributions or releases.
