# Architecture (initial)

## Current workspace

```text
crates/
├── qml-core/    Safe dense-matrix primitives and rank-one factorized operations
├── qml-format/  Narrow experimental QMF v0 reader/writer for one rank-one f64 matrix
└── qml-cli/     CLI help, basic diagnostics, synthetic demo, and QMF validation
```

The workspace stays small. A future crate should be introduced only when a stable responsibility warrants separation.

## Current data flow

`qml demo-rank1` creates a deterministic synthetic low-rank matrix, applies a largest-pivot rank-one cross approximation, performs matrix-vector products through the factors directly, writes the factors to an experimental `.qmf` file, reloads them, and compares output values. `qml inspect-qmf` validates and summarizes that narrow representation. The demo does not read model files or represent a pretrained model.

`qml-core` owns row-major `f64` matrices and rank-one factors. `qml-format` encodes factors using the experimental QMF v0 layout documented in [`research/qmf-experimental-v0.md`](research/qmf-experimental-v0.md). `qml-cli` composes those APIs. The workspace has no external Rust dependencies.

## Design constraints

Rust is the implementation language, and `unsafe` is forbidden at workspace level. Avoid Python application, development, or runtime dependencies. Keep future imported models immutable by default. Model parsers must validate sizes, offsets, integer arithmetic, and resource budgets before allocation or access. Require an independently testable CPU reference before acceleration. Keep structural, behavioral, and computational morphogenesis independently configurable. Quantum-inspired classical computation must not be described as physical quantum computation.

QMF v0 is intentionally experimental and limited to one rank-one matrix. It is not a stable general model format and has no compatibility promise. CRC-32 is only an accidental-corruption check, not authentication. The transformation is a conventional baseline; it is not claimed as novel or optimal.

## Planned boundaries (not implemented)

Safetensors/GGUF model loading, tensor enumeration, model configuration/tokenizer inspection, quantization algorithms, behavior analysis, evolutionary search, generalized QMF, inference, hardware discovery, and desktop UI remain future responsibilities. No placeholder crates are created for them yet. See [ROADMAP.md](ROADMAP.md).
