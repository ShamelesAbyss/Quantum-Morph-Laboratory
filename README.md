<p align="center">
  <img src="assets/hero.svg" alt="Quantum Morph Laboratory — Don't just compress intelligence. Transform its representation." width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.94%2B%20(2024%20Edition)-7C3AED?style=for-the-badge&logo=rust&logoColor=white" alt="Rust"/>
  <img src="https://img.shields.io/badge/Unsafe%20Code-FORBIDDEN-06B6D4?style=for-the-badge" alt="Zero Unsafe"/>
  <img src="https://img.shields.io/badge/Python%20Deps-ZERO-0EA5E9?style=for-the-badge" alt="Zero Python"/>
  <img src="https://img.shields.io/badge/Crates-9%20Modular-8B5CF6?style=for-the-badge" alt="9 Crates"/>
  <img src="https://img.shields.io/badge/Tests-67%20Passing-10B981?style=for-the-badge" alt="67 Tests"/>
  <img src="https://img.shields.io/badge/Status-Experimental%20Research%20Lab-F59E0B?style=for-the-badge" alt="Experimental"/>
</p>

<h3 align="center"><em>Don't just compress intelligence. Transform its representation.</em></h3>

<p align="center">
  An experimental, local-first Rust neural model research laboratory investigating shared-subspace weight representations, adaptive precision allocation, native compressed inference, and multi-objective evolutionary representation discovery.
</p>

---

> **Scientific Integrity Notice:** Quantum Morph Laboratory (QML) is an active experimental research platform, not production software. "Quantum-inspired" refers strictly to classical mathematics (tensor networks, spectral decompositions, subspace geometry) and makes **no** claim of physical quantum computation or quantum advantage. Current benchmarks run on deterministic synthetic fixtures; pretrained billion-parameter model validation, GPU kernels, and a native desktop UI remain upcoming milestones. Every byte—including scales, zero-points, shared bases, sparse indices, JSON manifests, and checksums—is counted in reported sizes.

---

## Architecture & Representation Pipeline

<p align="center">
  <img src="assets/architecture.svg" alt="Quantum Morph Laboratory Implemented Rust Pipeline Architecture" width="100%"/>
</p>

### Implemented Subsystems (9 Safe Rust Crates)

| Subsystem | Crate | Implemented Capabilities | Maturity |
|---|---|---|---|
| **Core Math** | `qml-core` | Validated dense matrices, `matmul`/`transpose`/`matvec_transposed`, deterministic SplitMix64 PRNG, exact **one-sided Jacobi thin SVD**, Halko–Martinsson–Tropp randomized SVD, spectral metrics (stable rank, Roy–Vetterli effective rank, 90/95/99% energy ranks), Grassmannian subspace affinity | Implemented & tested |
| **Model Ingestion** | `qml-model` | Bounded read-only Safetensors header inspection (16 MiB cap), **constant-memory streaming** payload reader (`PayloadBudget`), exact safe-Rust codecs (`F64`, `F32`, `F16`, `BF16`, `F8_E4M3`, `F8_E5M2`, ints, bool), sharded `index.json` & Hugging Face `config.json` parsers, bounded **GGUF v2/v3** metadata inspector | Implemented & tested |
| **Q-Scope Observatory** | `qml-scope` | Two-pass streaming Welford statistics (mean, std, skewness, excess kurtosis, norms, sparsity, 3σ/6σ outliers, crest factor, 32-bin histogram & Shannon entropy, quantization susceptibility proxy), 2D SVD structure, transformer role classification, cross-layer Grassmannian subspace affinity | Implemented & tested |
| **Baselines, RMQ & MAP** | `qml-quant` | Conventional baselines (`FP16`, `BF16`, `INT8_SYM`, `INT8_ASYM`, `Q4_0`-style `INT4_SYM`, `Q4_1`-style `INT4_ASYM`, `LowRankSvd`, `SparseResidual`) with exact `StorageBreakdown`; **RMQ Shared-Basis** (`U·Cₗ·Vᵀ + Rₗ` via 2D-HOOI with **RPCA-inspired outlier peeling**); **RMQ Shared-Core** (`Aₗ·S·Bₗ + Rₗ`); **MAP** sensitivity + downstream spectral propagation allocator | Experimental & benchmarked |
| **QMF Container** | `qml-format` | Experimental **QMF v0** (rank-one prototype) and **QMF v1** generalized multi-tensor binary container with a **Shared Latent Table** (`U_shared`, `V_sharedᵀ`), heterogeneous per-tensor representations, sparse COO FP16 residuals, canonical JSON manifests, per-blob & whole-file CRC-32 checks, and direct `QmfOperator::matvec` | Experimental (unstable) |
| **Q-Forge Engine** | `qml-forge` | Read-only Safetensors → QMF v1 transformation pipeline supporting FP16, BF16, INT8, INT4, SVD, role-clustered RMQ, and MAP mixed precision with full provenance and round-trip verification (`compare`) | Experimental & tested |
| **Q-Core Runtime** | `qml-runtime` | Native CPU reference decoder-only Llama-style Transformer (RMSNorm, RoPE, Grouped-Query Attention + KV-cache, SwiGLU MLP) executing **directly over QMF v1 compressed operators** with zero dense weight reconstruction, autoregressive token generation, KL divergence & perplexity evaluation | Experimental CPU reference |
| **Q-Genesis** | `qml-genesis` | Multi-objective evolutionary representation search optimizing hybrid per-layer & RMQ-cluster genomes over complete effective BPP, measured matvec error, and estimated matvec FLOPs, maintaining non-dominated **Pareto frontiers** and persistent JSON archives | Experimental & tested |
| **Laboratory CLI** | `qml-cli` | 12 commands (`doctor`, `inspect-safetensors`, `inspect-gguf`, `inspect-qmf`, `scope`, `baseline`, `rmq-ablate`, `morph`, `compare`, `infer-demo`, `genesis`, `demo-rank1`) with text and `--json` output | Implemented & tested |

### Not Yet Implemented (Upcoming Milestones)

- Pretrained open-weight model evaluation campaigns (e.g., SmolLM / TinyLlama local validation without committing weights)
- GPU compute backends (WGPU / Vulkan / CUDA targeting RTX 4070 Laptop 8 GB VRAM) & SIMD-accelerated CPU kernels
- Q-Behavior activation/behavioral interpretability & regression suite
- Native Rust desktop research laboratory UI (`qml-desktop`)

---

## Measured Experimental Benchmarks

All figures below were measured directly from `qml.exe --release` on Windows 11 x86_64 (Intel Core i9-14900HX, Rust 1.94.0, CPU reference) and archived in [`experiments/exp-001-rmq-map-qcore-baseline.toml`](experiments/exp-001-rmq-map-qcore-baseline.toml).

<p align="center">
  <img src="assets/benchmarks.svg" alt="QML Measured RMQ Ablation and Q-Core Direct Inference Benchmarks" width="100%"/>
</p>

### 1. RMQ Ablation — Correlated Subspace vs Uncorrelated Gaussian Control (`8 × 64 × 64`, 32,768 params)

| Method | Total Bytes | Eff. BPP | Correlated Rel. Frob | Correlated MV Err | Gaussian Control Rel. Frob |
|---|---:|---:|---:|---:|---:|
| `INT8_SYM_G64` | 33,952 | 8.289 | 0.00854 | 0.01166 | 0.00592 |
| `INT4_SYM_G32 (Q4_0)` | 18,592 | 4.539 | 0.10226 | 0.13111 | **0.09677** |
| `INDEPENDENT_SVD_INT8` (r=6) | 6,528 | 1.594 | 0.07123 | 0.09110 | 0.83756 |
| `RMQ_SHARED_BASIS` (r=6) | 1,584 | **0.387** | 0.30349 | 0.23179 | 0.96642 |
| `RMQ_SHARED_BASIS + SPARSE 1%` (outlier-peeled) | 3,552 | **0.867** | 0.14223 *(−49.2% via peeling)* | 0.17307 | 0.92428 |
| `RMQ_SHARED_CORE + SPARSE 1%` (`Aₗ·S·Bₗ + Rₗ`) | 8,592 | **2.098** | **0.06703** | **0.08812** | 0.80094 |

**Key falsifiable finding:** Introducing RPCA-inspired outlier peeling before 2D-HOOI cut `RMQ_SHARED_BASIS + SPARSE 1%` error by **49.2%** (from `0.280` to `0.142`) at identical `0.867` effective bpp, while `RMQ_SHARED_CORE + SPARSE 1%` beat both `Q4_0` and independent SVD at `2.098` bpp. Equally importantly, the **uncorrelated i.i.d. Gaussian control** proves RMQ is **not** a universal replacement for block quantization: it gains traction only when Q-Scope detects cross-layer subspace affinity, motivating hybrid **MAP / Q-Genesis** selection.

### 2. Q-Core Native Direct-Compressed Inference (Synthetic 4-Layer Mini-Llama, 40,224 params, 164,864 B source)

| QMF v1 Variant | Complete QMF Bytes | Complete BPP | Mean KL vs FP16 | Logit RMSE | Top-1 Match | Direct CPU Tok/s |
|---|---:|---:|---:|---:|---:|---:|
| `FP16` (Reference) | 87,423 | 17.39 | 0.00000 | 0.00000 | 100.0% | 3,733 |
| `INT8_SYM (g64)` | 51,009 | 10.14 | 0.00018 | 0.01859 | 100.0% | 3,146 |
| `MAP Adaptive (5.5 bpp target)` | 37,668 | 7.49 | **0.00566** *(11.8× < Q4_0)* | 0.10040 | **77.8%** | 2,526 |
| `INT4_SYM (Q4_0 g32)` | 32,339 | 6.43 | 0.06702 | 0.34413 | 44.4% | 2,557 |
| `RMQ Shared-Basis + 1% Sparse` | **26,681** | **5.31** | **0.06037** | **0.33443** | **55.6%** | **9,057 (2.43× FP16)** |

*(Note: Complete QMF BPP includes the ~5.8 KB JSON manifest, adding ~1.15 bpp on a tiny 40K-parameter model; on realistic models this metadata amortizes to < 0.05 bpp.)*

---

## Quick Start — Build, Test & Run

Requires only a stable Rust toolchain (`rustc` / `cargo`). Zero Python, zero cloud accounts, zero mandatory GPU drivers.

```sh
# Format, strict Clippy lint, full test suite (67 tests), and release build
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --release -p qml-cli

# Host diagnostics & SIMD feature detection
cargo run --release -p qml-cli -- doctor

# Inspect Safetensors or GGUF files (read-only, bounded headers, never modifies source)
cargo run --release -p qml-cli -- inspect-safetensors ./path/to/model.safetensors
cargo run --release -p qml-cli -- inspect-gguf ./path/to/model.gguf

# Run Q-Scope Observatory (streaming statistics, outlier crest, SVD spectrum, cross-layer affinity)
cargo run --release -p qml-cli -- scope ./path/to/model.safetensors --svd-rank 16

# Run reproducible quantization baselines & RMQ ablations (correlated + Gaussian control)
cargo run --release -p qml-cli -- baseline --synthetic
cargo run --release -p qml-cli -- rmq-ablate --layers 8 --dim 64 --rank 6

# Transform Safetensors -> QMF v1 (fp16 | bf16 | int8 | int4 | int4-asym | svd | rmq | map)
cargo run --release -p qml-cli -- morph ./model.safetensors --method rmq --rank 8 --output ./out.qmf
cargo run --release -p qml-cli -- inspect-qmf ./out.qmf
cargo run --release -p qml-cli -- compare ./model.safetensors ./out.qmf

# Run native direct-compressed Q-Core transformer inference & Q-Genesis Pareto evolution
cargo run --release -p qml-cli -- infer-demo
cargo run --release -p qml-cli -- genesis --synthetic --population 12 --generations 5
```

See [BUILD.md](BUILD.md) for `cargo-deny` policy checks and CI workflow details.

---

## Research Foundation & Continuity

- **Prior-Art Literature Review (17 topics):** [`research/prior-art-index.md`](research/prior-art-index.md) and [`research/prior-art-synthesis.md`](research/prior-art-synthesis.md)
- **Novelty Ledger:** [`research/novelty-ledger.md`](research/novelty-ledger.md) (explicitly distinguishes established literature, falsifiable hypotheses, and QML experimental findings)
- **Container Specifications:** [`research/qmf-experimental-v0.md`](research/qmf-experimental-v0.md) and [`research/qmf-experimental-v1.md`](research/qmf-experimental-v1.md)
- **Architecture & Decisions:** [`ARCHITECTURE.md`](ARCHITECTURE.md), [`ROADMAP.md`](ROADMAP.md), [`docs/decisions/0001-bounded-safetensors-inspection.md`](docs/decisions/0001-bounded-safetensors-inspection.md)
- **Cross-Agent Handoff:** [`AGENTS.md`](AGENTS.md), [`HANDOFF.md`](HANDOFF.md), [`MASTER_DEVELOPMENT_DIRECTIVE.md`](MASTER_DEVELOPMENT_DIRECTIVE.md)

## Licensing Status

No project license has been selected yet. Public visibility does **not** grant permission to reuse, modify, or redistribute Quantum Morph Laboratory source code; do not select or imply a license without explicit owner approval. Third-party Rust crate dependency licenses are continuously verified by `cargo-deny` in CI (`deny.toml`).
