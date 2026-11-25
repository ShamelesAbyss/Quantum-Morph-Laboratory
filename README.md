<p align="center">
  <img src="assets/hero.svg" alt="Quantum Morph Laboratory — Don't just compress intelligence. Transform its representation." width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.100%2B%20(2024%20Edition)-7C3AED?style=for-the-badge&logo=rust&logoColor=white" alt="Rust"/>
  <img src="https://img.shields.io/badge/Unsafe%20Code-FORBIDDEN-06B6D4?style=for-the-badge" alt="Zero Unsafe"/>
  <img src="https://img.shields.io/badge/Python%20Deps-ZERO-0EA5E9?style=for-the-badge" alt="Zero Python"/>
  <img src="https://img.shields.io/badge/Crates-12%20Modular-8B5CF6?style=for-the-badge" alt="12 Crates"/>
  <img src="https://img.shields.io/badge/Tests-76%20Passing-10B981?style=for-the-badge" alt="76 Tests"/>
  <img src="https://img.shields.io/badge/License-MIT-22D3EE?style=for-the-badge" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Release-v0.1.0--alpha.1%20SINGULARITY-F59E0B?style=for-the-badge" alt="Experimental"/>
</p>

<h3 align="center"><em>Don't just compress intelligence. Transform its representation.</em></h3>

<p align="center">
  An experimental, local-first Rust neural model research laboratory investigating shared-subspace weight representations, adaptive precision allocation, native compressed inference, and multi-objective evolutionary representation discovery.
</p>

---

> **Scientific Integrity Notice:** Quantum Morph Laboratory (QML) is an active experimental research platform (`v0.1.0-alpha.1 — PROJECT SINGULARITY`), not production software. "Quantum-inspired" refers strictly to classical mathematics (tensor networks, spectral decompositions, subspace geometry) and makes **no** claim of physical quantum computation or quantum advantage. QML has been validated end-to-end on both synthetic benchmarks (`exp-001`) and a real openly licensed pretrained 135M-parameter language model (`HuggingFaceTB/SmolLM2-135M-Instruct`, `exp-002`, weights stored locally outside Git). Every byte—including scales, zero-points, shared bases, sparse indices, JSON manifests, and checksums—is counted in reported sizes, and negative controls are preserved alongside positive results.

---

## Architecture & Representation Pipeline

<p align="center">
  <img src="assets/architecture.svg" alt="Quantum Morph Laboratory Implemented Rust Pipeline Architecture" width="100%"/>
</p>

### Implemented Subsystems (12 Safe Rust Crates)

| Subsystem | Crate | Implemented Capabilities | Maturity |
|---|---|---|---|
| **Core Math & SHA-256** | `qml-core` | Validated dense matrices, `matmul`/`transpose`/`matvec_transposed`, SplitMix64 PRNG, exact **one-sided Jacobi thin SVD**, HMT randomized SVD, spectral metrics, Grassmannian subspace affinity, pure-Rust FIPS 180-4 **SHA-256** | Implemented & tested |
| **Hardware & WGPU** | `qml-hardware` | Cross-platform CPU/SIMD/RAM telemetry, WGPU GPU adapter discovery, resource presets, and a validated **WGSL compute shader** `f32` matvec prototype with CPU equivalence test & graceful headless fallback | Experimental prototype |
| **Model Ingestion** | `qml-model` | Bounded read-only Safetensors header inspection (16 MiB cap), **constant-memory streaming** payload reader (`PayloadBudget`), exact safe-Rust codecs (`F64`, `F32`, `F16`, `BF16`, `F8_E4M3`, `F8_E5M2`, ints, bool), sharded `index.json` & Hugging Face `config.json` parsers, bounded **GGUF v2/v3** inspector | Implemented & tested |
| **Q-Scope Observatory** | `qml-scope` | Two-pass streaming Welford statistics (mean, std, skewness, kurtosis, norms, sparsity, 3σ/6σ outliers, crest factor, 32-bin histogram & Shannon entropy, quantization susceptibility proxy), 2D SVD structure, transformer role classification, cross-layer Grassmannian subspace affinity | Implemented & tested |
| **Baselines, RMQ & MAP** | `qml-quant` | Rayon-parallelized conventional baselines (`FP16`, `BF16`, `INT8_SYM`, `INT8_ASYM`, `Q4_0`-style `INT4_SYM`, `Q4_1`-style `INT4_ASYM`, `LowRankSvd`, `SparseResidual`) with exact `StorageBreakdown`; **RMQ Shared-Basis** (`U·Cₗ·Vᵀ + Rₗ` via 2D-HOOI + outlier peeling); **RMQ Shared-Core** (`Aₗ·S·Bₗ + Rₗ`); **MAP** sensitivity + downstream spectral propagation allocator | Experimental & benchmarked |
| **QMF Container** | `qml-format` | Experimental **QMF v0** & multi-GB **QMF v1** (`QmfBudget` default 8 GiB, streaming header-only inspector, `QmfLatentCache` `Arc` deduplication, first-class `RmqSharedCore`, on-demand `decode_row`, Shared Latent Table, sparse COO FP16 residuals, CRC-32) | Experimental (unstable) |
| **Q-Forge Engine** | `qml-forge` | Memory-bounded tensor-by-tensor and cluster-by-cluster Safetensors → QMF v1 transformation pipeline with real SHA-256 provenance and direct round-trip matvec verification (`compare`) | Experimental & tested |
| **Pure-Rust Tokenizer** | `qml-tokenizer` | Pure safe Rust Hugging Face `tokenizer.json` Byte-Level BPE tokenizer (GPT-2/SmolLM/Llama-3/Qwen2 style) with individual-digit pre-tokenization, special tokens, ChatML formatting, and streaming token decode | Implemented & tested |
| **Q-Core Runtime** | `qml-runtime` | Native Rayon-parallelized Llama-style Transformer (RMSNorm, RoPE, GQA + KV-cache, SwiGLU MLP, `tie_word_embeddings`, on-demand compressed embedding lookup via `decode_row`, `Arc` shared-latent cache) executing **directly over QMF v1 compressed operators**, streaming generation, KL divergence & perplexity | Experimental CPU runtime |
| **Q-Genesis** | `qml-genesis` | Multi-objective NSGA-II evolutionary search optimizing hybrid per-layer & RMQ-cluster genomes over complete effective BPP, measured matvec error, and estimated matvec FLOPs | Experimental & tested |
| **Laboratory CLI** | `qml-cli` | 13 commands (`doctor`, `inspect-safetensors`, `inspect-gguf`, `inspect-qmf`, `scope`, `baseline`, `rmq-ablate`, `morph`, `compare`, `infer-demo`, `generate`, `genesis`, `demo-rank1`) with text and `--json` output | Implemented & tested |
| **Native Desktop App** | `qml-desktop` | Native Windows GUI (`eframe`/`egui` + `egui_plot` + `rfd`) with Deep Obsidian / Ultraviolet / Electric Cyan theme and 10 real-engine views: Command Center, Model Library, Q-Scope, Q-Entangle heatmap, Q-Forge Studio, Q-Genesis Pareto Lab, QMF Explorer, Q-Core Inference Chamber, RMQ Benchmarks, and System Observatory | `v0.1.0-alpha.1` |

### Real Pretrained Model Validation (`HuggingFaceTB/SmolLM2-135M-Instruct`, `exp-002`)

Measured directly from `qml.exe --release` on Windows 11 x86_64 (Intel Core i9-14900HX, 32 threads, Apache-2.0 source `model.safetensors` SHA-256 `5af571cb...`, 134,515,008 parameters, 269,060,552 bytes; see [`experiments/exp-002-smollm2-135m.toml`](experiments/exp-002-smollm2-135m.toml)):

| QMF v1 Variant | Complete QMF Bytes | Complete BPP | Mean MV Err | Mean KL vs BF16 | Logit RMSE | Top-1 Match | Direct CPU Tok/s | Generated English Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `BF16` (Reference) | 269,145,551 | 16.007 *(0.007 overhead)* | 0.00000 | 0.00000 | 0.00000 | 100.0% | 6.90 | Coherent SVD explanation |
| `INT8_SYM (g64)` | 138,885,054 | 8.260 *(1.94×)* | 0.00629 | 0.00223 | 0.23987 | 97.7% | 4.29 | Near-identical to BF16 |
| `INT4_SYM (Q4_0 g32)` | 75,847,831 | 4.511 *(3.55×)* | 0.10160 | 0.94767 | 3.99471 | 59.1% | 3.85 | Degraded (`"SINGULAR VARIATION..."`) |
| **`MAP (Quant-Only, matched)`** | **76,010,084** | **4.521 *(3.54×)*** | **0.08891 (−12.5%)** | **0.79210 (−16.4%)** | **3.60194 (−9.8%)** | **61.4%** | 3.22 | **Coherent SVD explanation** |
| `MAP (Unrestricted SVD-8 — Control)` | 75,843,859 | 4.511 | 0.05817 *(misleading)* | 7.66516 *(collapsed)* | 10.19178 | 20.5% | 0.68 | Collapsed (`"imimimim..."`) |

- **Real Cross-Layer Affinity (Q-Scope):** On `SmolLM2-135M`, adjacent `k_proj` layers exhibit strong right-subspace Grassmannian affinity (`layers.17 <-> 18` right=`0.6455`, shared=`0.3464` despite element-wise cosine `-0.0044`), while the 240-pair global mean is `0.0648`—confirming RMQ must be selectively affinity-gated.
- **Preserved Negative Control:** Allowing rank-8 SVD on full-rank `embed_tokens` inside MAP lowered unweighted Frobenius error by 43% while destroying generation (`PPL = 5201`), proving why quantization-only MAP (or embedding-protected menus) is mandatory on real LLMs.

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

## License

Quantum Morph Laboratory is licensed under the **[MIT License](LICENSE)** (Copyright © 2026 ShamelesAbyss). Third-party Rust crate dependency licenses and advisories are continuously verified by `cargo-deny` in CI ([`deny.toml`](deny.toml)). Prebuilt Windows and Linux release binaries with SHA-256 checksums are published on the [GitHub Releases](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases) page (see [`RELEASE_NOTES.md`](RELEASE_NOTES.md)).

---

### In Loving Memory of Ashlan Haley Rapponotti

*June 21, 2003 – November 13, 2026*

For the beautiful woman who listened to my passions, believed in me, and made my world a happier place simply by being part of it.

Every creation, every discovery, and every milestone reached carries a piece of my heart dedicated to you.

**Forever loved. Forever missed. Forever remembered.** 💜
