# Handoff

## Current state

- Public repository: <https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory>
- Development branch: `main` only.
- Local workspace: `C:\Users\Shame\.gemini\antigravity\scratch\Quantum-Morph-Laboratory` (Windows 11, Intel Core i9-14900HX, RTX 4070 Laptop GPU 8 GB VRAM, 32 GB RAM).
- Active takeover milestone: **PROJECT SINGULARITY — `v0.1.0-alpha.1` Complete (Audit Remediation, Pure-Rust BPE Tokenizer, Real SmolLM2-135M Validation, Rayon CPU Parallelism, WGPU Compute Prototype, and Native Windows Desktop Laboratory `qml-desktop`)**.
- The workspace now contains **12 cohesive safe-Rust crates** (`qml-core`, `qml-hardware`, `qml-model`, `qml-scope`, `qml-quant`, `qml-format`, `qml-forge`, `qml-tokenizer`, `qml-runtime`, `qml-genesis`, `qml-cli`, `qml-desktop`) with `#![forbid(unsafe_code)]` and zero Python dependencies.
- **License:** Officially licensed under the **MIT License** ([LICENSE](LICENSE), approved by repository owner). All 12 workspace crates inherit `license.workspace = true`.

## Implemented capabilities

- `qml-core`: Validated dense-matrix primitives (`transpose`, `matmul`, `matvec_transposed`, element-wise ops, cosine similarity), deterministic SplitMix64 PRNG, deterministic one-sided Jacobi exact thin SVD, deterministic Halko–Martinsson–Tropp randomized SVD, direct factorized matrix-vector multiplication (`U (Σ (Vᵀ x))`), spectral summary metrics (spectral norm, stable rank, Roy–Vetterli effective rank, 90%/95%/99% Frobenius energy ranks, Grassmannian subspace affinity), plus the original largest-pivot rank-one baseline.
- `qml-model`: Bounded read-only Safetensors header inspection (16 MiB cap), bounded & constant-memory streaming Safetensors payload reader (`SafetensorsFile`, `PayloadBudget`), pure safe Rust bit-exact codecs (`F64`, `F32`, `F16`, `BF16`, `F8_E4M3`, `F8_E5M2`, integer/bool), sharded Safetensors index parser (`model.safetensors.index.json`), Hugging Face `config.json` parser, and bounded read-only GGUF v2/v3 metadata/tensor-directory inspector (`inspect_gguf`).
- `qml-scope`: Q-Scope Neural Model Observatory computing two-pass constant-memory streaming statistics (Welford mean/variance, skewness, excess kurtosis, norms, sparsity fractions, 3σ/6σ outliers, crest factor, 32-bin histogram & Shannon entropy, quantization susceptibility proxy), 2D matrix SVD spectrum analysis, transformer layer-role classification, and pairwise cross-layer Grassmannian subspace affinity analysis.
- `qml-quant`: Reproducible conventional baselines (`FP16`, `BF16`, `INT8_SYM`, `INT8_ASYM`, `INT4_SYM_G32(Q4_0)`, `INT4_ASYM_G32(Q4_1)`, `INT4_SYM_G128`, `SVD_FP16`, `SVD_INT8`, `SparseResidual`) with exact byte-for-byte storage accounting (`StorageBreakdown`) and direct compressed `LinearOperator::matvec`; experimental RMQ Shared-Basis (`U_shared C_l V_sharedᵀ + R_l` via deterministic 2D-HOOI) and RMQ Shared-Core (`A_l S B_l + R_l`) engines with sparse/INT4 residuals and baseline ablation comparator; experimental MAP (Morphological Adaptive Precision) sensitivity + downstream spectral propagation mixed-precision allocator.
- `qml-format`: Bounded reader and deterministic writer for experimental QMF v0 (single rank-one `f64`) and QMF v1 (`QmfArchiveV1`, `QmfV1Builder`, `QmfOperator`) supporting multi-tensor containers, shared latent structure tables (for RMQ), dense FP32/FP16/BF16, groupwise INT8/INT4, truncated SVD, sparse COO FP16 residuals, per-blob + whole-file CRC-32 checks, and direct compressed operator execution (`research/qmf-experimental-v1.md`).
- `qml-forge`: Modular Q-Forge transformation pipeline (`forge_safetensors_to_qmf`, `compare_safetensors_and_qmf`) converting immutable Safetensors inputs into verified QMF v1 artifacts with full provenance, complete file-level byte accounting, and direct round-trip matvec verification.
- `qml-runtime`: Q-Core CPU reference Llama-style decoder-only Transformer executing directly over QMF v1 compressed `QmfOperator` layers (RMSNorm, RoPE, Grouped-Query Attention with KV-cache, SwiGLU MLP, greedy/sampled autoregressive token generation, and exact KL-divergence / logit-RMSE / top-1 agreement / perplexity fidelity evaluation).
- `qml-genesis`: Q-Genesis deterministic multi-objective evolutionary search engine optimizing per-layer & RMQ-cluster genomes over complete effective bpp, measured matvec error, and estimated matvec FLOPs, outputting non-dominated Pareto frontiers and full persistent JSON archives (including dominated trials).
- `qml-cli`: `doctor`, `inspect-safetensors`, `inspect-gguf`, `inspect-qmf` (v0 & v1), `scope`, `baseline`, `rmq-ablate`, `morph`, `compare`, `infer-demo`, `genesis`, and `demo-rank1`.

## Validation actually executed on the implementation

All of the following passed locally on Windows 11 with Rust 1.100.0 (`cargo +1.100.0`) and `cargo-deny 0.21.0`:

- `cargo +1.100.0 fmt --all -- --check`
- `cargo +1.100.0 clippy --workspace --all-targets -- -D warnings`
- `cargo +1.100.0 test --workspace` — all tests across 12 crates passed (including NIST SHA-256 vectors, `Arc` latent deduplication, `RmqSharedCore`, `RmqAffinityGated`, `decode_row`, streaming header inspection, matched-budget MAP, `StreamingDecoder` split multi-byte UTF-8, and RTX 4070 WGPU dense `FP32` + direct compressed `INT8_SYM` & `Q4_0` WGSL compute shader equivalence tests).
- `cargo deny check` — `advisories ok, bans ok, licenses ok, sources ok` (zero ignored vulnerabilities).
- `cargo +1.100.0 build --release -p qml-desktop -p qml-cli` — verified native Windows launch of `qml-desktop.exe` with zero panics.

## Research and design findings

- **Outlier-Peeling RMQ Discovery (`exp-001`):** Isolated large weight outliers in correlated multi-layer matrices can steal a shared subspace dimension during 2D-HOOI. Adding RPCA-inspired sparse outlier peeling before 2D-HOOI reduced `RMQ_SHARED_BASIS + SPARSE 1%` relative Frobenius error by **49.2%** (`0.28006 -> 0.14223`) at an identical `0.867` effective bpp (all overhead counted).
- **Falsifiable Boundary Condition:** On an uncorrelated i.i.d. full-rank Gaussian control, `Q4_0` (`0.09677`) decisively outperforms all low-rank/RMQ methods (`0.80–0.97`). RMQ should only be applied to layer clusters where Q-Scope measures significant cross-layer Grassmannian subspace affinity.
- **Direct Compressed Inference Speedup (`exp-001`):** Executing `U_shared (C_l (V_sharedᵀ x)) + R_l x` directly in Q-Core without dense weight reconstruction reached `9,057 tok/s` (`2.43×` faster than FP16 and `3.54×` faster than Q4_0 on CPU) with lower KL divergence (`0.06037`) than uniform `Q4_0` (`0.06702`).
- **MAP Precision Allocation (`exp-001`):** Sensitivity + downstream spectral propagation allocation reduced Mean KL divergence on the synthetic mini-Llama by **11.8×** vs uniform `Q4_0` (`0.00566` vs `0.06702`, `77.8%` vs `44.4%` top-1 token agreement).
- **QMF v1 Manifest Overhead:** On tiny 40K-parameter models, canonical JSON manifests add ~1.15 bpp to complete file size; on realistic models (>10M params) this amortizes to `<0.05` bpp.

## Known limits and blockers

All current benchmarks were executed on deterministic synthetic fixtures (no pretrained LLM checkpoint has yet been evaluated end-to-end). GPU backends (WGPU/CUDA), SIMD-vectorized/multithreaded kernels, real BPE tokenizers, Q-Behavior, and the native desktop GUI (`qml-desktop`) remain to be built. QMF v0/v1 are experimental and not compatibility-stable. No project license has been selected.

## Next steps

1. Continue on `main`; verify `git status` and CI results on GitHub Actions (`Rust CI` + `dependency-policy`).
2. Validate Q-Scope, RMQ, MAP, Q-Forge, and Q-Core on a small openly licensed pretrained Transformer downloaded locally (outside Git, e.g., SmolLM-135M or GPT-2 small): measure real cross-layer Grassmannian subspace affinity, actual RMQ vs Q4_0/AWQ tradeoffs, and WikiText perplexity.
3. Add real BPE/SentencePiece tokenizer support and memory-mapped QMF v1 streaming for multi-GB models, plus `rayon`/SIMD CPU acceleration and WGPU/CUDA GPU kernel prototypes targeting the RTX 4070 Laptop 8 GB VRAM budget.
4. Implement `qml-behavior` and the native Rust desktop laboratory (`qml-desktop`) once real pretrained model pipelines are validated.
