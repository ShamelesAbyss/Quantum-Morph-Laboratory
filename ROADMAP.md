# Roadmap

Status key: **Done** means present and tested in this repository; **Next** is proposed work; **Research** requires literature review and experimental evidence.

## Foundation

- **Done:** Public repository on `main`; Rust workspace with `qml-core`, `qml-format`, and `qml-cli`.
- **Done:** Basic validated dense matrix representation, CPU matrix-vector product, Frobenius norm, and unit tests.
- **Done:** `qml help`, `qml version`, and basic `qml doctor`; unsupported capabilities are disclosed.
- **Done:** Deterministic largest-pivot rank-one cross approximation, direct factorized matvec, and error measurement on CPU.
- **Done:** Synthetic rank-one CLI demo and bounded experimental QMF v0 serialization/inspection, with round-trip, corruption, and malformed-size tests.
- **Done:** Linux/Windows CI workflow and local formatting, test, and Clippy checks.
- **Next:** Select a repository license with the project owner; currently no license is granted.

## Scientific foundation

- **Done:** First source-backed prior-art batch with 17 topic notes, an index, and a comparative synthesis.
- **Next:** Expand the review to remaining methods named in the directive, including behavioral/interpretability work, distillation, evolutionary search, quantum-inspired optimization, complex-valued networks, and additional compression/file-format methods.
- **Next:** Preserve a source-backed novelty ledger; current entries are explicitly hypotheses and no originality conclusion has been reached.
- **Next:** Establish deterministic test fixtures and baseline measurement conventions.

## First proof of concept

- **Done (synthetic only):** Compare a conventional largest-pivot rank-one approximation on a generated exact rank-one matrix; measure storage and reconstruction/direct-matvec error.
- **Done (experimental only):** Serialize, reload, inspect, and reject malformed/corrupt single-matrix QMF v0 data. This is not a model format.
- **Next:** Load a small, legally redistributable Safetensors tensor fixture with bounded parsing.
- **Next:** Inspect tensor metadata without loading full models to GPU.
- **Next:** Compare the baseline with a defined structural approximation on real tensor data; count all bytes and report reconstruction and direct-operation error.

## Longer term

Q-Scope, Q-Forge, Q-Entangle, Q-Behavior, Q-Genesis, QMF, Q-Core, hardware backends, and desktop UI remain aspirational. They must be introduced only after their requirements, baselines, and validation plans are clear. No target performance or compression ratio is guaranteed.

## Scope caveat

The source directive received so far is incomplete. The master file ends during the QMF section, and the supplied continuation labels itself “Continuation 1 of 2.” This roadmap covers only unambiguous early-stage requirements and must be reconciled with the complete directive when available. The initial literature batch is not comprehensive and establishes no novelty, priority, freedom-to-operate, or non-infringement conclusion.
