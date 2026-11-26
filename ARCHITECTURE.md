# Architecture

## Current Workspace (13 Safe Rust Crates)

```text
crates/
├── qml-core/      Validated dense matrices, SplitMix64 PRNG, one-sided Jacobi & HMT randomized SVD, spectral metrics, FIPS 180-4 SHA-256
├── qml-hardware/  CPU/SIMD/RAM telemetry, hardware-only WGPU compute, 10 compressed matvec WGSL shaders (incl. `MATVEC_VQ_WGSL`) + 8 `gpu_full.rs` shaders (incl. GPU INT8 KV-cache)
├── qml-model/     Bounded read-only single & multi-shard Safetensors (`SafetensorsSource`, direct row-seek & zero-f64 BF16 readers, 8 GiB/tensor) + GGUF v2/v3 incl. K-quants
├── qml-scope/     Q-Scope Neural Observatory: two-pass streaming moments, outliers, histograms, SVD spectra, cross-layer Grassmannian affinity
├── qml-quant/     Baselines, RMQ Shared-Basis/Core, MAP/MAP-B, Hadamard INT3/INT2/Ternary, RVQ (`fit_vq_calibrated`), ECHO, Blockwise Cholesky GPTQ (`gptq.rs`) & `QCAL1` disk cache
├── qml-format/    Experimental QMF v0 & multi-GB QMF v1: out-of-core QmfFileReader, atomic writer, Shared Latent Table, sub-3-bit ops, CRC-32 & SHA-256
├── qml-forge/     Q-Forge atomic single/multi-shard streaming pipeline (`forge_path_to_file_atomic`) with calibration GPTQ & RVQ, cancellation & verifier
├── qml-tokenizer/ Pure safe Rust Hugging Face tokenizer.json Byte-Level BPE tokenizer (SmolLM/Llama-3/Qwen2) with ChatML & stateful UTF-8 StreamingDecoder
├── qml-runtime/   Q-Core Llama/Qwen2 Transformer (`q/k/v` biases, FP32/FP64 & GPU INT8 KV-cache, `streaming.rs` <450 MiB VRAM 7B–32B calibration, `Cpu`, `GpuHybrid`, `GpuFull`)
├── qml-behavior/  Q-Behavior Mechanistic Observatory: hidden-state tracing, RSA, causal ablations, `ReferenceSuiteCache` (`QREF1` binary cache) & 4-category Capability Suite
├── qml-genesis/   Q-Genesis Extreme role/depth-stratified multi-objective NSGA-II evolution (incl. sub-3-bit & ECHO genes) + QMF v1 export
├── qml-cli/       `qml` laboratory CLI exposing 15 commands (incl. `--calibrate`, `--streaming-ref`, `--kv-cache int8`, `rvq-200/175/150`, sharded `morph` & `resonance-ladder`)
└── qml-desktop/   PROJECT PRISM native WGPU 3D desktop observatory (7 PRISM WGSL shaders, HDR bloom, 11 views, `GpuFull` inference, VRAM Fit Matrix)
```

Every crate enforces `#![forbid(unsafe_code)]` via workspace lints. Zero Python dependencies exist anywhere in the normal, dev, or build dependency graph.

## End-to-End Data Flow

1. **Ingestion (`qml-model`):** Opens single `.safetensors`, multi-shard `model.safetensors.index.json` (`SafetensorsSource`, at most one shard open), or `.gguf` read-only, validates bounded headers (`<= 16 MiB`), supports direct row-seek (`read_tensor_rows_f64`) and zero-`f64` BF16 bit reads (`read_tensor_bf16_bits`), and decodes exact `F16`, `BF16`, `FP8`, `F32`, `F64`, `Q8_0`, `Q4_0`, `Q4_1`, and **GGUF K-quants (`Q2_K..Q6_K`)** under `PayloadBudget` ceilings (`8 GiB/tensor`, `64 GiB/total` default).
2. **Observatory (`qml-scope`):** Computes two-pass constant-memory streaming statistics, 2D singular-value spectra, transformer role classification, and cross-layer Grassmannian subspace affinity.
3. **Layer-by-Layer Streaming Calibration & Sub-3-Bit Compression (`qml-runtime::streaming` + `qml-quant::gptq`):** Streams `7B–32B` BF16 checkpoints **one transformer block at a time** (`< 450 MiB` peak VRAM on 7B) + `<= 32,768`-row chunked `lm_head` to record exact per-layer `input_rms`, `64×64` block-diagonal Hessians `H_b = (2/N) X_bᵀ X_b` (`QCAL1` binary cache), and exact BF16 reference logits (`QREF1` binary cache). `qml-quant::gptq` rotates covariances orthogonally (`Q H_b Qᵀ`), fits activation-weighted grids, and runs Rayon-parallelized Blockwise Cholesky GPTQ (`actorder` + adaptive shrinkage damping), calibrated ECHO, and calibrated RVQ (`fit_vq_calibrated`).
4. **Atomic Single & Sharded Transformation (`qml-forge` + `qml-format`):** `forge_path_to_file_atomic` streams single or sharded Safetensors checkpoints into deterministic QMF v1 containers via sibling temp files and atomic rename (`write_to_file_atomic`, source-output alias protection, zero duplicate full-file RAM buffer), with streaming CRC-32 + SHA-256 verification.
5. **Single-Submission Full GPU-Resident Execution + GPU INT8 KV-Cache (`qml-runtime` + `qml-hardware::gpu_full`):** Executes the complete Llama or Qwen2 transformer (`RMSNorm`, optional Qwen2 `q/k/v` biases, GPU bit-exact uniform-barrier `Hadamard FWHT`, `RoPE + FP32/INT8 KV-Append`, online-softmax `GQA Attention` over FP32 or packed INT8 KV, `SwiGLU`, residuals, and all **10 direct-compressed matvec shaders including `MATVEC_VQ_WGSL`**) inside a **single `wgpu::CommandEncoder` submission per token** (`QCoreBackend::GpuFull`). Declines software CPU WGPU emulators (`DeviceType::Cpu`) so GPU-less machines automatically use fast Rayon CPU.
6. **Mechanistic Interpretability & Capability Retention (`qml-behavior`):** Captures layerwise cosine/L2/RSA traces, executes reversible causal interventions, and evaluates compressed models against disk-cached `ReferenceSuiteCache` (`QREF1`) logits with zero reference VRAM contention.
7. **Evolutionary Discovery & PROJECT PRISM 3D Observatory (`qml-genesis` + `qml-desktop`):** Multi-objective NSGA-II representation discovery paired with the multi-pass WGPU 3D observatory (`Rgba16Float` HDR selective bloom, 3D ray picking, Consumer VRAM Fit Matrix).

## Safety and Resource Constraints

- `unsafe` is forbidden at workspace level (`Cargo.toml`).
- Imported model files are opened read-only and never modified; output writes use atomic staging with alias protection.
- All parsers bound header/manifest/file sizes and use checked integer arithmetic before allocation.
- CRC-32 verifies accidental corruption only; SHA-256 records source provenance and artifact integrity.
- QMF v0 and v1 remain experimental research formats and are not yet compatibility-stable.
