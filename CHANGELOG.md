# Changelog

## Unreleased

- Initialized the Rust workspace with `qml-core`, `qml-format`, and `qml-cli`.
- Added safe dense-matrix primitives and unit tests.
- Added basic `qml help`, `qml version`, and `qml doctor` commands.
- Added Linux/Windows CI and project architecture, roadmap, build, contribution, handoff, and security documentation.
- Completed a first source-backed prior-art batch covering 17 topics, with detailed notes, an index, a comparative synthesis, and an updated novelty ledger.
- Added deterministic largest-pivot rank-one cross approximation, factorized matrix-vector multiplication, and reconstruction-error measurement as conventional CPU baselines.
- Added `qml demo-rank1` for a generated synthetic fixture and `qml inspect-qmf` for an experimental single-matrix QMF v0 container; both report limitations explicitly.
- Added QMF v0 deterministic serialization, bounded parsing, checksum validation, and tests for round trips, corruption, unsupported fields, non-finite values, and excessive declared shapes.
- Added `qml-model` with bounded, read-only Safetensors metadata inspection and text/JSON CLI reporting. The parser validates duplicate keys, dtype/shape size arithmetic, offsets, contiguous layouts, header/resource limits, and file length without reading payload bytes.
- Added property-based tests and generated F32/F4 differential fixtures checked by the exact-pinned Safetensors 0.8.0 dev dependency; no model weights or third-party datasets are included.
- Added a source-linked Safetensors research note and an architectural decision record for the header-only parser and its tradeoffs.
- **Milestone I — Q-Scope Foundation:**
  - Extended `qml-core` with full dense linear algebra (`transpose`, `matmul`, `matvec_transposed`, cosine similarity), deterministic SplitMix64 PRNG, exact one-sided Jacobi thin SVD, deterministic Halko–Martinsson–Tropp randomized SVD, spectral metrics (stable rank, Roy–Vetterli effective rank, energy ranks), and Grassmannian subspace affinity.
  - Extended `qml-model` with bounded & constant-memory streaming Safetensors tensor payload access (`SafetensorsFile`, `PayloadBudget`), exact safe-Rust dtype codecs (`F16`, `BF16`, `F8_E4M3`, `F8_E5M2`, `F32`, `F64`, ints, bool), sharded Safetensors index parser, Hugging Face `config.json` parser, and bounded read-only GGUF v2/v3 metadata inspector (`inspect_gguf`).
  - Added `qml-scope` (Q-Scope Neural Model Observatory) with two-pass streaming distributional statistics, sparsity & outlier detection, histogram/entropy estimation, quantization susceptibility proxy, 2D matrix SVD structure analysis, transformer role classification, and cross-layer subspace affinity analysis.
  - Added `qml scope` and `qml inspect-gguf` CLI commands and upgraded `qml doctor` with safe CPU SIMD feature detection.
- **Milestones II & III — Quantization Baselines, RMQ & MAP (`qml-quant`):**
  - Implemented reproducible conventional baselines (`FP16`, `BF16`, `INT8_SYM`, `INT8_ASYM`, `INT4_SYM_G32(Q4_0)`, `INT4_ASYM_G32(Q4_1)`, `INT4_SYM_G128`, `LowRankSvdMatrix`, `SparseResidual`) with exact complete byte accounting (`StorageBreakdown`) and direct compressed `LinearOperator::matvec`.
  - Implemented experimental RMQ Shared-Basis (`U_shared * C_l * V_sharedᵀ + R_l` via deterministic 2D-HOOI) and RMQ Shared-Core (`A_l * S * B_l + R_l`) engines with sparse outlier and INT4 residual options plus head-to-head baseline ablation (`compare_rmq_vs_baselines`), validated on both correlated clusters and uncorrelated full-rank Gaussian controls.
  - Implemented experimental MAP (Morphological Adaptive Precision) allocator combining local sensitivity, outlier crest factor, and bounded downstream spectral error-propagation amplification under a strict global byte budget.
