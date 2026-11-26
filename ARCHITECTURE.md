# Architecture

## Current Workspace (13 Safe Rust Crates)

```text
crates/
├── qml-core/      Validated dense matrices, SplitMix64 PRNG, one-sided Jacobi & HMT randomized SVD, spectral metrics, FIPS 180-4 SHA-256
├── qml-hardware/  Cross-platform CPU/SIMD/RAM telemetry, WGPU discovery, ResourcePreset budgets, 6 persistent VRAM WGSL compute kernels
├── qml-model/     Bounded read-only Safetensors & GGUF v2/v3 streaming payload readers, exact dtype codecs, shard & HF config parsers
├── qml-scope/     Q-Scope Neural Observatory: two-pass streaming moments, outliers, histograms, SVD spectra, cross-layer Grassmannian affinity
├── qml-quant/     Rayon baselines (FP16/BF16/INT8/Q4_0/Q4_1/SVD), RMQ Shared-Basis/Shared-Core with outlier peeling, two-pass streaming MAP
├── qml-format/    Experimental QMF v0 & multi-GB QMF v1: out-of-core QmfFileReader, atomic writer, Shared Latent Table, CRC-32 & SHA-256
├── qml-forge/     Q-Forge atomic streaming Safetensors -> QMF v1 pipeline with cooperative cancellation, affinity-gated RMQ & verifier
├── qml-tokenizer/ Pure safe Rust Hugging Face tokenizer.json Byte-Level BPE tokenizer with ChatML formatting & stateful UTF-8 StreamingDecoder
├── qml-runtime/   Q-Core Llama-style Transformer executing directly over QMF v1 on CPU (Rayon) or persistent GPU Hybrid + reversible interventions
├── qml-behavior/  Q-Behavior Mechanistic Observatory: layer hidden-state tracing, cosine/L2 drift, RSA & reversible causal ablation studies
├── qml-genesis/   Q-Genesis role/depth-stratified multi-objective NSGA-II evolutionary search, user profiles & Genome-to-QMF v1 export
├── qml-cli/       `qml` laboratory CLI exposing 14 diagnostic, inspection, morph, inference, behavior, and evolutionary commands
└── qml-desktop/   PROJECT PRISM native WGPU 3D desktop scientific observatory (7 custom WGSL shaders, HDR bloom, 11 real-engine views)
```

Every crate enforces `#![forbid(unsafe_code)]` via workspace lints. Zero Python dependencies exist anywhere in the normal, dev, or build dependency graph.

## End-to-End Data Flow

1. **Ingestion (`qml-model`):** Opens `.safetensors` or `.gguf` read-only, validates bounded metadata headers (`<= 16 MiB`) with checked arithmetic, and provides streaming chunked decoding (`F16`, `BF16`, `F8_E4M3`, `F8_E5M2`, `F32`, `F64`, `Q8_0`, `Q4_0`, `Q4_1`, integers, bool) under explicit `PayloadBudget` ceilings controlled by `ResourcePreset`.
2. **Observatory (`qml-scope`):** Computes two-pass constant-memory streaming statistics (Welford moments, sparsity, 3σ/6σ outliers, crest factor, 32-bin histogram & Shannon entropy), 2D matrix singular-value spectra, transformer role classification, and cross-layer Grassmannian subspace affinity.
3. **Quantization, RMQ & MAP (`qml-quant`):** Evaluates conventional baselines with exact `StorageBreakdown` byte accounting, fits RMQ shared-basis clusters (`U_shared · C_l · V_sharedᵀ + R_l` via 2D-HOOI with RPCA-inspired outlier peeling) or shared-core clusters (`A_l · S · B_l + R_l`), and solves global byte-budget mixed-precision allocations in two-pass streaming MAP.
4. **Transformation & Containerization (`qml-forge` + `qml-format`):** Transforms immutable Safetensors models into deterministic QMF v1 containers via sibling temp files and atomic rename (`write_to_file_atomic`, source-output alias protection), storing shared latent matrices once in `shared_latents`, supporting cooperative `ForgeControl` cancellation, and verifying per-blob + 64 KiB streaming whole-file CRC-32 & SHA-256 integrity.
5. **Direct Native CPU & Persistent GPU Execution (`qml-runtime` + `qml-hardware`):** Opens QMF v1 via out-of-core `QmfFileReader`, keeps compressed weights persistently resident in VRAM across 6 WGSL kernels (`FP32`, `FP16`, `BF16`, `INT8_SYM`, `Q4_0`, `Q4_1`) with `Arc<PersistentGpuMatrix>` deduplication for RMQ latents and reusable scratch buffers, and runs RMSNorm, RoPE, Grouped-Query Attention with KV-cache, and SwiGLU MLP without dense weight reconstruction.
6. **Mechanistic Interpretability (`qml-behavior`):** Captures post-block hidden states across all layers via `forward_token_with_trace` to compute per-layer cosine similarity, relative L2 divergence, and Representational Similarity Analysis (RSA), and executes strictly runtime-only reversible causal interventions (`ScaleResidual`, `AblateAttention`, `AblateMlp`).
7. **Evolutionary Discovery (`qml-genesis`):** Evolves role- and depth-stratified per-layer & RMQ-cluster genomes over complete effective BPP, measured matvec error, and estimated matvec FLOPs, exporting Pareto-optimal winners directly to verified QMF v1 artifacts.
8. **PROJECT PRISM Native 3D Visualization (`qml-desktop::prism`):** Renders live measured laboratory telemetry and schematic architectural geometry through a custom multi-pass WGPU pipeline (`Rgba16Float` HDR target, `Depth32Float` Z-buffer, GPU-instanced crystals, neural filaments, Keplerian particles, separable 9-tap Gaussian selective bloom, ACES filmic tone mapping) with interactive Orbit/Ortho cameras, 3D ray picking, and automatic 2.5D CPU fallback.

## Safety and Resource Constraints

- `unsafe` is forbidden at workspace level (`Cargo.toml`).
- Imported model files are opened read-only and never modified; output writes use atomic staging with alias protection.
- All parsers bound header/manifest/file sizes and use checked integer arithmetic before allocation.
- CRC-32 verifies accidental corruption only; SHA-256 records source provenance and artifact integrity.
- QMF v0 and v1 remain experimental research formats and are not yet compatibility-stable.
