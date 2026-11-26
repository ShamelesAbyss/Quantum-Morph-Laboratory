# Architecture

## Current Workspace (13 Safe Rust Crates)

```text
crates/
├── qml-core/      Validated dense matrices, SplitMix64 PRNG, one-sided Jacobi & HMT randomized SVD, spectral metrics, FIPS 180-4 SHA-256
├── qml-hardware/  CPU/SIMD/RAM telemetry, WGPU discovery, 9 compressed matvec WGSL shaders + 6 Full GPU-Resident Transformer WGSL shaders (`gpu_full.rs`)
├── qml-model/     Bounded read-only single & multi-shard Safetensors (`SafetensorsSource`) + GGUF v2/v3 incl. K-quants (`Q2_K..Q6_K`), HF config parser
├── qml-scope/     Q-Scope Neural Observatory: two-pass streaming moments, outliers, histograms, SVD spectra, cross-layer Grassmannian affinity
├── qml-quant/     Baselines, RMQ Shared-Basis/Core, MAP/MAP-B, Hadamard INT3/INT2/Ternary, RVQ, ECHO, and Activation-Aware Blockwise Cholesky GPTQ (`gptq.rs`)
├── qml-format/    Experimental QMF v0 & multi-GB QMF v1: out-of-core QmfFileReader, atomic writer, Shared Latent Table, sub-3-bit ops, CRC-32 & SHA-256
├── qml-forge/     Q-Forge atomic single/multi-shard streaming pipeline (`forge_path_to_file_atomic`) with calibration GPTQ, cancellation & verifier
├── qml-tokenizer/ Pure safe Rust Hugging Face tokenizer.json Byte-Level BPE tokenizer (SmolLM/Llama-3/Qwen2) with ChatML & stateful UTF-8 StreamingDecoder
├── qml-runtime/   Q-Core Llama/Qwen2 Transformer (`q/k/v` biases, `KvCacheMode::Int8`, calibration collector, `Cpu`, `GpuHybrid`, & single-submission `GpuFull`)
├── qml-behavior/  Q-Behavior Mechanistic Observatory: hidden-state tracing, RSA, causal ablations, `ReferenceSuiteCache` & 4-category Capability Suite
├── qml-genesis/   Q-Genesis Extreme role/depth-stratified multi-objective NSGA-II evolution (incl. sub-3-bit & ECHO genes) + QMF v1 export
├── qml-cli/       `qml` laboratory CLI exposing 15 commands (incl. `--calibrate`, `--backend gpu-full`, sharded `morph` & `resonance-ladder`)
└── qml-desktop/   PROJECT PRISM native WGPU 3D desktop observatory (7 PRISM WGSL shaders, HDR bloom, 11 views, `GpuFull` inference, VRAM Fit Matrix)
```

Every crate enforces `#![forbid(unsafe_code)]` via workspace lints. Zero Python dependencies exist anywhere in the normal, dev, or build dependency graph.

## End-to-End Data Flow

1. **Ingestion (`qml-model`):** Opens single `.safetensors`, multi-shard `model.safetensors.index.json` (`SafetensorsSource`, holding at most one shard handle open at a time), or `.gguf` read-only, validates bounded metadata headers (`<= 16 MiB`), and decodes exact `F16`, `BF16`, `FP8`, `F32`, `F64`, `Q8_0`, `Q4_0`, `Q4_1`, and **GGUF K-quants (`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`)** under explicit `PayloadBudget` ceilings (`4 GiB/tensor`, `32 GiB/total` default).
2. **Observatory (`qml-scope`):** Computes two-pass constant-memory streaming statistics, 2D singular-value spectra, transformer role classification, and cross-layer Grassmannian subspace affinity.
3. **Activation Calibration & Sub-3-Bit Compression (`qml-quant` + `qml-runtime`):** `QCoreModel::collect_calibration_profile` records exact per-layer `input_rms` and `64×64` block-diagonal Hessian covariances (`H_b = (2/N) X_bᵀ X_b`) over disjoint calibration tokens. `qml-quant::gptq` rotates covariances orthogonally (`Q H_b Qᵀ`) to match `HadamardSpec`, fits activation-weighted affine/symmetric grids, and executes Rayon-parallelized Blockwise Cholesky GPTQ (`actorder` + adaptive shrinkage damping) and activation-weighted ECHO recovery with zero extra storage bytes.
4. **Atomic Single & Sharded Transformation (`qml-forge` + `qml-format`):** `forge_path_to_file_atomic` streams single or sharded Safetensors checkpoints into deterministic QMF v1 containers via sibling temp files and atomic rename (`write_to_file_atomic`, source-output alias protection, zero duplicate full-file RAM buffer), with streaming CRC-32 + SHA-256 verification.
5. **Single-Submission Full GPU-Resident Execution (`qml-runtime` + `qml-hardware::gpu_full`):** Opens QMF v1 via out-of-core `QmfFileReader` and executes the complete Llama or Qwen2 transformer (`RMSNorm`, optional Qwen2 `q/k/v` biases, GPU bit-exact `Hadamard FWHT`, `RoPE + KV-Append`, online-softmax `GQA Attention`, `SwiGLU`, residuals, and 9 direct-compressed matvec kernels) inside a **single `wgpu::CommandEncoder` submission per token** (`QCoreBackend::GpuFull`), falling back gracefully to `GpuHybrid` or multi-threaded Rayon `Cpu` (with optional `KvCacheMode::Int8`).
6. **Mechanistic Interpretability & Capability Retention (`qml-behavior`):** Captures layerwise cosine/L2/RSA traces, executes reversible causal interventions, and evaluates compressed models against precomputed `ReferenceSuiteCache` logits (freeing BF16 reference VRAM before loading compressed candidates on 8 GB GPUs).
7. **Evolutionary Discovery & PROJECT PRISM 3D Observatory (`qml-genesis` + `qml-desktop`):** Multi-objective NSGA-II representation discovery paired with the multi-pass WGPU 3D observatory (`Rgba16Float` HDR selective bloom, 3D ray picking, Consumer VRAM Fit Matrix).

## Safety and Resource Constraints

- `unsafe` is forbidden at workspace level (`Cargo.toml`).
- Imported model files are opened read-only and never modified; output writes use atomic staging with alias protection.
- All parsers bound header/manifest/file sizes and use checked integer arithmetic before allocation.
- CRC-32 verifies accidental corruption only; SHA-256 records source provenance and artifact integrity.
- QMF v0 and v1 remain experimental research formats and are not yet compatibility-stable.
