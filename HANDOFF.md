# Handoff

## Current state

- Public repository: <https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory>
- Development branch: `main` only.
- Local workspace: `C:\Users\Shame\.gemini\antigravity\scratch\Quantum-Morph-Laboratory` (Windows 11, Intel Core i9-14900HX, RTX 4070 Laptop GPU 8 GB VRAM, 32 GB RAM).
- Active takeover milestone: **Milestone IV — QMF v1 Generalized Container & Q-Forge Transformation Engine** (`qml-format` v1 + `qml-forge`).
- The workspace now contains `qml-core`, `qml-format`, `qml-model`, `qml-scope`, `qml-quant`, `qml-forge`, and `qml-cli`; zero new external dependencies were introduced and no Python runtime dependency exists anywhere in the graph.
- No project license has been selected. All workspace crates are marked `publish = false`; do not add a project license without owner approval.

## Implemented capabilities

- `qml-core`: Validated dense-matrix primitives (`transpose`, `matmul`, `matvec_transposed`, element-wise ops, cosine similarity), deterministic SplitMix64 PRNG, deterministic one-sided Jacobi exact thin SVD, deterministic Halko–Martinsson–Tropp randomized SVD, direct factorized matrix-vector multiplication (`U (Σ (Vᵀ x))`), spectral summary metrics (spectral norm, stable rank, Roy–Vetterli effective rank, 90%/95%/99% Frobenius energy ranks, Grassmannian subspace affinity), plus the original largest-pivot rank-one baseline.
- `qml-model`: Bounded read-only Safetensors header inspection (16 MiB cap), bounded & constant-memory streaming Safetensors payload reader (`SafetensorsFile`, `PayloadBudget`), pure safe Rust bit-exact codecs (`F64`, `F32`, `F16`, `BF16`, `F8_E4M3`, `F8_E5M2`, integer/bool), sharded Safetensors index parser (`model.safetensors.index.json`), Hugging Face `config.json` parser, and bounded read-only GGUF v2/v3 metadata/tensor-directory inspector (`inspect_gguf`).
- `qml-scope`: Q-Scope Neural Model Observatory computing two-pass constant-memory streaming statistics (Welford mean/variance, skewness, excess kurtosis, norms, sparsity fractions, 3σ/6σ outliers, crest factor, 32-bin histogram & Shannon entropy, quantization susceptibility proxy), 2D matrix SVD spectrum analysis, transformer layer-role classification, and pairwise cross-layer Grassmannian subspace affinity analysis.
- `qml-quant`: Reproducible conventional baselines (`FP16`, `BF16`, `INT8_SYM`, `INT8_ASYM`, `INT4_SYM_G32(Q4_0)`, `INT4_ASYM_G32(Q4_1)`, `INT4_SYM_G128`, `SVD_FP16`, `SVD_INT8`, `SparseResidual`) with exact byte-for-byte storage accounting (`StorageBreakdown`) and direct compressed `LinearOperator::matvec`; experimental RMQ Shared-Basis (`U_shared C_l V_sharedᵀ + R_l` via deterministic 2D-HOOI) and RMQ Shared-Core (`A_l S B_l + R_l`) engines with sparse/INT4 residuals and baseline ablation comparator; experimental MAP (Morphological Adaptive Precision) sensitivity + downstream spectral propagation mixed-precision allocator.
- `qml-format`: Bounded reader and deterministic writer for experimental QMF v0 (single rank-one `f64`) and QMF v1 (`QmfArchiveV1`, `QmfV1Builder`, `QmfOperator`) supporting multi-tensor containers, shared latent structure tables (for RMQ), dense FP32/FP16/BF16, groupwise INT8/INT4, truncated SVD, sparse COO FP16 residuals, per-blob + whole-file CRC-32 checks, and direct compressed operator execution (`research/qmf-experimental-v1.md`).
- `qml-forge`: Modular Q-Forge transformation pipeline (`forge_safetensors_to_qmf`, `compare_safetensors_and_qmf`) converting immutable Safetensors inputs into verified QMF v1 artifacts with full provenance, complete file-level byte accounting, and direct round-trip matvec verification.
- `qml-cli`: `doctor` (with safe x86 SIMD detection), `demo-rank1`, `inspect-qmf`, `inspect-safetensors`, `inspect-gguf`, and `scope` (text and JSON).

## Validation actually executed on the implementation

All of the following passed locally on Windows 11 with Rust 1.94.0:

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace` — 63 tests passed (15 `qml-core`, 7 `qml-format`, 24 `qml-model` unit + 2 Safetensors oracle, 2 `qml-scope`, 7 `qml-quant`, 1 `qml-forge`, 5 `qml-cli` integration).
- `cargo build --workspace`
- Zero new third-party dependencies added (preserves `cargo-deny` policy compliance; CI `dependency-policy` job verifies on push).

## Research and design findings

The Safetensors file structure, dtype bit sizes, offset rules, and upstream-parser limitation are documented with primary-source URLs in `research/safetensors-inspection.md`. The 16 MiB cap is a QML resource policy and is stricter than upstream's documented maximum. The upstream crate is dev-only; runtime parsing stays within the small `qml-model` responsibility. This is a header/layout check only, not payload verification or complete compatibility certification.

## Known limits and blockers

No Safetensors payload reader, tensor-value analysis, model architecture/configuration reader, GGUF support, quantizer, transformation pipeline, inference runtime, or GUI is implemented. The prior-art review remains incomplete. No QML novelty, model-compression advantage, or real-model result is claimed. QMF v0 is experimental and unsuitable for important artifacts. No pretrained weights or private datasets are in the repository or fixtures.

## Next steps

1. Continue on `main`; verify the working tree and current SHA before editing.
2. Keep the next milestone scoped to a separately bounded Safetensors payload reader for legally appropriate/generated small fixtures, or other user-approved research work. Preserve read-only-by-default behavior, add resource-budget tests, and do not commit model weights.
3. Continue the remaining source-backed prior-art review and keep the novelty ledger explicit about unresolved overlap.
