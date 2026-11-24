# Roadmap

Status key: **Done** means present and tested in this repository; **Next** is proposed work; **Research** requires literature review and experimental evidence.

## Foundation

- **Done:** Public repository on `main`; Rust workspace with `qml-core` and `qml-cli`.
- **Done:** Basic validated dense matrix representation, CPU matrix-vector product, Frobenius norm, and unit tests.
- **Done:** `qml help`, `qml version`, and basic `qml doctor`; unsupported capabilities are disclosed.
- **Next:** Establish continuous integration and verify Linux/Windows build paths.
- **Next:** Select a repository license with the project owner; currently no license is granted.

## Scientific foundation

- **Next:** Complete a sourced prior-art index covering the methods in the master directive; capture authorship, date, canonical source, math, evidence, resources, limitations, licensing, relevance, and overlap.
- **Next:** Establish a novelty ledger and mark every proposal as unreviewed until prior art is compared.
- **Next:** Define deterministic test fixtures and baseline measurement conventions.

## First proof of concept

- **Research:** Load a small, legally redistributable tensor fixture with bounded parsing.
- **Research:** Inspect tensor metadata without loading full models to GPU.
- **Research:** Compare a conventional baseline with a mathematically defined structural approximation.
- **Research:** Measure total bytes including metadata and residuals; report reconstruction and direct-operation error.
- **Research:** Serialize, reload, and test the transformed representation.

## Longer term

Q-Scope, Q-Forge, Q-Entangle, Q-Behavior, Q-Genesis, QMF, Q-Core, hardware backends, and desktop UI remain aspirational. They must be introduced only after their requirements, baselines, and validation plans are clear. No target performance or compression ratio is guaranteed.

## Scope caveat

The source directive received so far is incomplete. The master file ends during the QMF section, and the supplied continuation labels itself “Continuation 1 of 2.” This roadmap covers only unambiguous early-stage requirements and must be reconciled with the complete directive when available.
