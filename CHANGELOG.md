# Changelog

## Unreleased

- Initialized the Rust workspace with `qml-core` and `qml-cli`.
- Added safe dense-matrix primitives and unit tests.
- Added basic `qml help`, `qml version`, and `qml doctor` commands.
- Added Linux/Windows CI and project architecture, roadmap, build, contribution, handoff, and security documentation.
- Completed a first source-backed prior-art batch covering 17 topics, with detailed notes, an index, a comparative synthesis, and an updated novelty ledger.
- Added deterministic largest-pivot rank-one cross approximation, factorized matrix-vector multiplication, and reconstruction-error measurement as conventional CPU baselines.
- Added `qml demo-rank1` for a generated synthetic fixture and `qml inspect-qmf` for an experimental single-matrix QMF v0 container; both report limitations explicitly.
- Added QMF v0 deterministic serialization, bounded parsing, checksum validation, and tests for round trips, corruption, unsupported fields, non-finite values, and excessive declared shapes.
- No pretrained-model inspection, transformation, inference, or real-model benchmark functionality is implemented; no QML novelty or general compression result is claimed.
