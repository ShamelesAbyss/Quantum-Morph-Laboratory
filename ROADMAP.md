# Roadmap

Status key: **Done** means present and tested in this repository; **In progress** means implemented but not yet fully validated; **Next** is proposed work; **Research** requires literature review and experimental evidence. The complete master directive and both continuations are preserved in [`MASTER_DEVELOPMENT_DIRECTIVE.md`](MASTER_DEVELOPMENT_DIRECTIVE.md).

## Foundation

- **Done:** Public repository on `main`; Rust workspace with `qml-core`, `qml-format`, `qml-model`, and `qml-cli`.
- **Done:** Validated dense-matrix representation, CPU matrix-vector product, Frobenius norm, and conventional largest-pivot rank-one baseline.
- **Done:** `qml help`, `qml version`, and basic `qml doctor`; unsupported capabilities are disclosed.
- **Done:** Synthetic rank-one CLI demo and bounded experimental QMF v0 serialization/inspection.
- **Done:** Bounded, read-only Safetensors metadata inspection with text and JSON CLI output; payload bytes are not read.
- **Done:** Linux/Windows CI for formatting, linting, tests, and builds.
- **In progress:** Dependency advisory/license/source policy and its CI job are configured; execution is pending validation for this milestone.
- **Next:** Select a repository license with the project owner; the repository currently grants no project license.

## Scientific foundation

- **Done:** First source-backed prior-art batch with 17 topic notes, an index, and a comparative synthesis.
- **Next:** Expand the review to remaining methods named in the directive, including behavioral/interpretability work, distillation, evolutionary search, quantum-inspired optimization, complex-valued networks, and additional compression/file-format methods.
- **Next:** Preserve a source-backed novelty ledger; current entries are explicitly hypotheses and no originality conclusion has been reached.
- **Next:** Establish deterministic test fixtures and baseline measurement conventions.

## First proof of concept

- **Done (synthetic only):** Compare a conventional largest-pivot rank-one approximation on a generated exact rank-one matrix; measure storage and reconstruction/direct-matvec error.
- **Done (experimental only):** Serialize, reload, inspect, and reject malformed/corrupt single-matrix QMF v0 data. This is not a model format.
- **Done (metadata only):** Inspect bounded Safetensors headers, validate dtype/shape/offset layout and exact file length, and differentially check generated fixtures against upstream Safetensors 0.8.0. No tensor payload values have yet been read or analyzed.
- **Next:** Load a small, legally redistributable Safetensors tensor fixture through a separately bounded payload-reading path; do not commit weights or private datasets.
- **Next:** Establish conventional numerical baselines on real tensor data; count all representation bytes and report reconstruction and direct-operation error.
- **Next:** Define and test a structural approximation against those baselines before making any compression or capability claims.

## Longer term

Q-Scope analysis, Q-Forge, Q-Entangle, Q-Behavior, Q-Genesis, generalized QMF, Q-Core, hardware backends, and desktop UI remain aspirational. Introduce them only after their requirements, baselines, and validation plans are clear. No target performance or compression ratio is guaranteed.

## Scope caveat

The full three-part directive is now available and preserved in the repository. The existing prior-art batch is still incomplete and establishes no novelty, priority, freedom-to-operate, or non-infringement conclusion. QMF v0 remains experimental, narrowly implemented, and unsuitable for important artifacts.
