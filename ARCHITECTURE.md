# Architecture (initial)

## Current workspace

```text
crates/
├── qml-core/   Safe, dependency-free mathematical primitives and tests
└── qml-cli/    Minimal truthful command-line interface
```

The workspace intentionally begins with two small crates. A future crate should be introduced only when a stable responsibility warrants separation.

## Current data flow

There is not yet a model-loading or transformation data flow. `qml-core` owns a small row-major `f64` matrix abstraction. `qml-cli` currently uses the standard library for argument handling and host diagnostics. No weights are read, written, uploaded, or transformed.

## Design constraints

- Rust is the implementation language; `unsafe` is forbidden at workspace level.
- Avoid Python application, development, or runtime dependencies.
- Keep imported models immutable by default when model IO is introduced.
- Establish bounded parsing, streamed access, and resource budgets before supporting large checkpoints.
- Require an independently testable CPU reference before acceleration.
- Keep structural, behavioral, and computational morphogenesis independently configurable.
- QMF remains experimental until its requirements and binary specification are validated.
- Quantum-inspired classical computation must not be described as physical quantum computation.

## Planned boundaries (not implemented)

Model inspection, transformation algorithms, behavior analysis, evolutionary search, QMF serialization, inference, hardware discovery, and desktop UI are future responsibilities. No placeholder crates are created for them yet. See [ROADMAP.md](ROADMAP.md).
