# Architecture

## Current Workspace (9 Safe Rust Crates)

```text
crates/
├── qml-core/     Validated dense matrices, SplitMix64 PRNG, one-sided Jacobi & HMT randomized SVD, spectral metrics
├── qml-model/    Bounded Safetensors header & streaming payload reader, exact dtype codecs, shard/config/GGUF parsers
├── qml-scope/    Q-Scope Neural Observatory: streaming moments, outliers, histograms, SVD, cross-layer affinity
├── qml-quant/    Conventional baselines (FP16/BF16/INT8/Q4_0/Q4_1/SVD), RMQ Shared-Basis/Shared-Core, MAP allocator
├── qml-format/   Experimental QMF v0 (single rank-one) and QMF v1 (multi-tensor + Shared Latent Table + CRC-32)
├── qml-forge/    Q-Forge Safetensors -> QMF v1 transformation pipeline and direct round-trip verifier
├── qml-runtime/  Q-Core CPU reference Llama-style decoder-only Transformer executing directly over QMF v1 operators
├── qml-genesis/  Q-Genesis multi-objective evolutionary search with Pareto frontier & persistent JSON archives
└── qml-cli/      `qml` command-line laboratory exposing 12 diagnostic, analysis, morph, inference, and evolution commands
```

Every crate enforces `#![forbid(unsafe_code)]` via workspace lints. Zero Python dependencies exist anywhere in the normal, dev, or build dependency graph.

## End-to-End Data Flow

1. **Ingestion (`qml-model`):** Opens `.safetensors` (or `.gguf`) read-only, validates bounded metadata headers (`<= 16 MiB`) with checked arithmetic, and provides streaming chunked decoding (`F16`, `BF16`, `F8_E4M3`, `F8_E5M2`, `F32`, `F64`, integers, bool) under explicit `PayloadBudget` ceilings.
2. **Observatory (`qml-scope`):** Computes two-pass constant-memory streaming statistics (Welford moments, sparsity, 3σ/6σ outliers, crest factor, binned Shannon entropy), 2D matrix singular-value spectra, transformer role classification, and cross-layer Grassmannian subspace affinity.
3. **Quantization, RMQ & MAP (`qml-quant`):** Evaluates conventional baselines with exact `StorageBreakdown` byte accounting, fits RMQ shared-basis clusters (`U_shared · C_l · V_sharedᵀ + R_l` via 2D-HOOI with RPCA-inspired outlier peeling) or shared-core clusters (`A_l · S · B_l + R_l`), and solves global byte-budget mixed-precision allocations in MAP.
4. **Transformation & Containerization (`qml-forge` + `qml-format`):** Transforms immutable Safetensors models into deterministic QMF v1 containers storing shared latent matrices once in `shared_latents`, encoding heterogeneous per-tensor representations and sparse COO FP16 residuals, and verifying per-blob + whole-file CRC-32 checksums.
5. **Direct Native Execution (`qml-runtime`):** Loads `QmfOperator` handles directly from QMF v1 and runs RMSNorm, RoPE, Grouped-Query Attention with KV-cache, and SwiGLU MLP **without reconstructing dense weight matrices**, reporting throughput, KL divergence, top-1 token agreement, and perplexity.
6. **Evolutionary Discovery (`qml-genesis`):** Evolves per-layer and RMQ-cluster genomes over complete effective BPP, measured matvec error, and estimated matvec FLOPs, outputting non-dominated Pareto frontiers and persistent JSON experiment archives.

## Safety and Resource Constraints

- `unsafe` is forbidden at workspace level (`Cargo.toml`).
- Imported model files are opened read-only and never modified.
- All parsers bound header/manifest/file sizes and use checked integer arithmetic before allocation.
- CRC-32 verifies accidental corruption only; it is not cryptographic authentication.
- QMF v0 and v1 remain experimental and not compatibility-stable.

## Planned Future Crates (Not Yet Created)

- Pretrained model tokenizer + real-checkpoint benchmarking harness
- Hardware-accelerated compute backends (SIMD / Rayon / WGPU / CUDA)
- `qml-behavior` (behavioral interpretability & regression evaluation)
- `qml-desktop` (native Rust graphical research laboratory)
