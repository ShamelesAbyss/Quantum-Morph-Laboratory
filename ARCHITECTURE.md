# Architecture

## Current workspace

```text
crates/
├── qml-core/    Safe dense-matrix primitives and rank-one baseline
├── qml-format/  Narrow experimental QMF v0 reader/writer for one rank-one f64 matrix
├── qml-model/   Bounded, read-only Safetensors metadata parser
└── qml-cli/     CLI help, basic diagnostics, synthetic demo, and inspection commands
```

The workspace remains intentionally small. A future crate should be introduced only when its responsibility is cohesive and independently testable.

## Current data flow

`qml demo-rank1` creates a deterministic synthetic low-rank matrix, applies a largest-pivot rank-one cross approximation, performs matrix-vector products through the factors directly, writes the factors to an experimental `.qmf` file, reloads them, and compares output values. `qml inspect-qmf` validates and summarizes that narrow representation. The demo does not read model files or represent a pretrained model.

`qml inspect-safetensors PATH` opens the file read-only. `qml-model` seeks to determine the file size, reads the eight-byte length prefix and a header capped at 16 MiB, validates the JSON and tensor metadata, and does not read or map the payload. The CLI can emit sorted human-readable output or JSON with `--json`. This operation does not inspect tensor values or identify model architecture. Its rationale and tradeoffs are recorded in [ADR 0001](docs/decisions/0001-bounded-safetensors-inspection.md); the source-backed format research is in [the Safetensors note](research/safetensors-inspection.md).

`qml-core` owns row-major `f64` matrices and rank-one factors. `qml-format` encodes factors using the experimental QMF v0 layout documented in [`research/qmf-experimental-v0.md`](research/qmf-experimental-v0.md). `qml-model` owns safe header parsing and metadata validation; it depends on `serde` and `serde_json`. The `safetensors` 0.8.0 parser and `proptest` are dev-only dependencies used as a differential oracle and for generated invariant tests. The CLI composes local crate APIs.

## Safety and resource constraints

Rust is the implementation language, and `unsafe` is forbidden at workspace level. Python application, development, and runtime dependencies are not part of the project. Imported models remain immutable by default. Model parsers validate sizes, offsets, integer arithmetic, and resource budgets before allocation or access. The Safetensors parser rejects duplicate JSON keys, unknown tensor fields and dtypes, excessive tensor counts/ranks, arithmetic overflow, misaligned sub-byte tensors, invalid ranges, gaps, overlaps, and payload-length disagreement. The parser never allocates according to payload size.

The 16 MiB QML header cap is stricter than upstream's documented limit. The parser accepts only known Safetensors dtypes and validates metadata layout, but it does not validate payload contents. See [the research note](research/safetensors-inspection.md) and [ADR 0001](docs/decisions/0001-bounded-safetensors-inspection.md).

Require an independently testable CPU reference before acceleration. Keep structural, behavioral, and computational morphogenesis independently configurable. Quantum-inspired classical computation must not be described as physical quantum computation.

QMF v0 is intentionally experimental and limited to one rank-one matrix. It is not a stable general model format and has no compatibility promise. CRC-32 is only an accidental-corruption check, not authentication. The transformation is a conventional baseline; it is not claimed as novel or optimal.

## Planned boundaries (not implemented)

Safetensors payload reading, GGUF/model configuration/tokenizer loading, full tensor enumeration over payload values, quantization algorithms, behavior analysis, evolutionary search, generalized QMF, inference, hardware discovery, and desktop UI remain future responsibilities. No placeholder crates are created for them. See [ROADMAP.md](ROADMAP.md).
