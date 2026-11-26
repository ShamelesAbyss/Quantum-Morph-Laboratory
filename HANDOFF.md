# Handoff

## Current state

- Public repository: <https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory>
- Development branch: `main` only.
- Local workspace: `C:\Users\Shame\.gemini\antigravity\scratch\Quantum-Morph-Laboratory` (Windows 11, Intel Core i9-14900HX, RTX 4070 Laptop GPU 8 GB VRAM, 32 GB RAM, 2560×1440 144 Hz display).
- Local pretrained model cache (NEVER commit to Git): `C:\Users\Shame\.gemini\antigravity\scratch\qml-models-cache` (`smollm2-135m`, `smollm2-360m`, `smollm2-1.7b`, `qwen2.5-1.5b`, `qwen2.5-3b`).
- Latest completed milestone: **`v0.5.0` — PROJECT EVENT HORIZON Wave II** (Activation-Aware Blockwise Cholesky GPTQ + Hadamard Compensation, Full GPU-Resident Single-Submission WGSL Transformer `GpuFull` with `22.1×` speedup, Qwen2 Attention Biases, Multi-Shard Streaming Safetensors, GGUF K-Quants `Q2_K..Q6_K`, `KvCacheMode::Int8`, `ReferenceSuiteCache`, and `exp-005` Live Validation across `360M`, `1.5B`, `1.7B`, and 2-shard `3B` breaking `2.75`, `2.50`, and `2.25 bpp`).
- **Permanent Release Distribution Policy:** All user-facing releases use clean semantic versions (`v0.3.0`, `v0.4.0`, `v0.5.0`, ...) and MUST be published as full, visible public GitHub Releases (`draft: false`, `prerelease: false`, `make_latest: true`) with all 4 compiled assets (`Quantum-Morph-Laboratory-<tag>-Windows-x64.exe`, `Quantum-Morph-Laboratory-<tag>-Windows-x64-Portable.zip`, `Quantum-Morph-Laboratory-<tag>-Linux-x64.tar.gz`, `SHA256SUMS.txt`), full `RELEASE_NOTES.md`, and prominent `README.md` download links. Prereleases are strictly opt-in when explicitly requested by the repository owner. A Git tag, CI build, Actions artifact, or source archive alone is NEVER a completed release.
- The workspace contains **13 cohesive safe-Rust crates** (`qml-core`, `qml-hardware`, `qml-model`, `qml-scope`, `qml-quant`, `qml-format`, `qml-forge`, `qml-tokenizer`, `qml-runtime`, `qml-behavior`, `qml-genesis`, `qml-cli`, `qml-desktop`) with `#![forbid(unsafe_code)]` and zero Python dependencies.
- **License:** MIT ([LICENSE](LICENSE)). All workspace crates inherit `license.workspace = true`.
- **Permanent Memorial Dedication:** The memorial dedication to Ashlan Haley Rapponotti at the end of `README.md` MUST be preserved verbatim across all future updates and releases.

## Implemented capabilities

- `qml-core`: Dense linear algebra, SplitMix64 PRNG, deterministic one-sided Jacobi exact thin SVD, HMT randomized SVD, spectral metrics, Grassmannian subspace affinity, and pure safe-Rust FIPS 180-4 SHA-256.
- `qml-hardware`: Cross-platform CPU/SIMD/RAM telemetry, WGPU GPU adapter discovery with full `adapter.limits()`, enforced `ResourcePreset` budgets, **9 persistent VRAM WGSL matrix kernels** (`FP32`, `FP16`, `BF16`, `INT8_SYM`, `Q4_0`, `Q4_1`, `INT3_ASYM`, `INT2_ASYM`, `TERNARY`), **6 full-transformer GPU-resident WGSL kernels** (`RMSNORM`, `HADAMARD_FWD`, `ROPE_AND_KV_APPEND`, `GQA_ATTN`, `SWIGLU`, `VEC_ADD`) in `gpu_full.rs` executing all layers + `lm_head` in one submission per token, and `GPU_SCRATCH_MAX_ELEMS = 262_144`.
- `qml-model`: Unified single + multi-shard `SafetensorsSource` (`PayloadBudget` up to `4 GiB/tensor`, `32 GiB/total`, max 1 open shard handle) and GGUF v2/v3 readers with exact safe-Rust codecs (`F64`, `F32`, `F16`, `BF16`, `F8_E4M3`, `F8_E5M2`, `Q8_0`, `Q4_0`, `Q4_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, ints, bool) and HF `config.json` parser.
- `qml-scope`: Two-pass streaming Welford statistics, sparsity, 3σ/6σ outliers, crest factor, 32-bin histogram & Shannon entropy, quantization susceptibility proxy, 2D SVD spectrum, transformer role classification, and pairwise cross-layer Grassmannian subspace affinity.
- `qml-quant`: Conventional baselines (`FP16`, `BF16`, `INT8_SYM`, `INT8_ASYM`, `Q4_0`, `Q4_1`, `LowRankSvd`, `SparseResidual`); RMQ Shared-Basis & Shared-Core; two-pass streaming `MAP`; **PROJECT RESONANCE (`resonance.rs`)** (blockwise Randomized FWHT, LS-refitted `Int3AsymGroupMatrix`/`Int2AsymGroupMatrix`, base-3 `TernaryPackedMatrix`, 1/2-stage `VectorQuantMatrix`, **PROJECT ECHO**, variance-additive `MAP-B` `QMR-300..150`); and **Wave II Activation-Aware Blockwise Cholesky GPTQ (`gptq.rs`)** (`LayerCalibrationAccumulator`, `ModelCalibrationProfile`, exact `Q_b H_b Q_bᵀ` Hadamard Hessian rotation, `actorder`, Ledoit–Wolf adaptive shrinkage, calibrated GPTQ for `INT4`, `INT3`, `INT2`, `TERNARY`, and `ECHO`).
- `qml-format`: Experimental QMF v0 & multi-GB QMF v1 with out-of-core lazy `QmfFileReader`, atomic publication (`write_to_file_atomic`) with source-output alias protection, 64 KiB streaming CRC-32 + SHA-256 verification, `Arc` shared-latent deduplication, `RmqSharedCore`, sub-3-bit operators, and on-demand `decode_row`.
- `qml-forge`: Streaming single & multi-shard Safetensors → QMF v1 pipeline with `forge_path_to_file_atomic` (zero duplicate full-file RAM buffer), fast BF16/FP16 streaming, cooperative `ForgeControl` cancellation, calibrated GPTQ + `MAP-B` Pass 1/Pass 2, and direct round-trip verification.
- `qml-tokenizer`: Pure safe-Rust Hugging Face `tokenizer.json` Byte-Level BPE tokenizer with individual-digit pre-tokenization, ChatML formatting, and stateful UTF-8 `StreamingDecoder`.
- `qml-runtime`: Native Llama & Qwen2 (optional `q/k/v` attention biases) Transformer executing directly over QMF v1 compressed operators on CPU (Rayon), `GpuHybrid`, or single-submission `GpuFull`, plus `KvCacheMode::Int8`, `collect_calibration_profile`, `forward_token_with_trace`, and reversible `LayerIntervention` hooks.
- `qml-behavior`: Layer-by-layer hidden activation tracing (`compare_model_behavior`), cosine similarity, relative L2 divergence, RSA, reversible causal ablation studies (`run_intervention_study`), `derive_layer_importance_multipliers`, held-out 4-category **Capability Retention Suite**, and `ReferenceSuiteCache` (allowing BF16 reference models to drop and free VRAM prior to loading compressed candidates).
- `qml-genesis`: **Q-Genesis Extreme** role- and depth-stratified multi-objective NSGA-II evolutionary search, user profiles, persistent JSON archives, and direct Genome-to-QMF v1 export.
- `qml-cli`: 15 laboratory commands (`doctor`, `inspect-safetensors`, `inspect-gguf`, `inspect-qmf`, `scope`, `baseline`, `rmq-ablate`, `morph`, `compare`, `resonance-ladder --calibrate`, `infer-demo`, `generate`, `behavior`, `genesis`, `demo-rank1`) supporting sharded Safetensors and `GpuFull`.
- `qml-desktop`: **PROJECT PRISM** native WGPU 3D scientific observatory (`eframe` WGPU + `wgpu 24`, 7 custom WGSL shaders, `Rgba16Float` HDR selective bloom, ACES filmic tone mapping, interactive Orbit/Ortho cameras & 3D ray picking, automatic 2.5D CPU fallback, 5 persisted graphics presets, 11 real-engine views, `GpuFull` default GPU inference, and the **Consumer VRAM Fit Matrix**).

## Validation actually executed on `v0.5.0`

All of the following passed locally on Windows 11 with Rust `1.100.0` (`cargo +1.100.0`) and `cargo-deny 0.21.0`:

- `cargo +1.100.0 fmt --all -- --check`
- `cargo +1.100.0 clippy --workspace --all-targets -- -D warnings`
- `cargo +1.100.0 test --workspace` — all **88 automated tests** across **13 crates** passed (including all 15 WGSL GPU compute shaders, `GpuFull` vs CPU parity, blockwise Cholesky GPTQ + Hadamard rotation proofs, GGUF K-quants, INT8 KV-cache, and PRISM WGPU rendering pipelines on NVIDIA RTX 4070).
- `cargo +1.100.0 build --release -p qml-desktop -p qml-cli` + smoke-test launch of `qml.exe --version` (`qml 0.5.0`) and `qml-desktop.exe`.
- `cargo deny check` — `advisories ok, bans ok, licenses ok, sources ok` (zero ignored vulnerabilities).
- **Live Real-Model GPU Experiments (`exp-005`, RTX 4070):** Matched-byte GPTQ+Hadamard controls on `SmolLM2-360M` (`22.1×` `GpuFull` speedup; `2.75 bpp` cliff broken), Stage B `SmolLM2-1.7B` (`2.50 bpp` broken) & `Qwen2.5-1.5B` (`2.25 bpp` broken), and Stage C 2-shard `Qwen2.5-3B` (`2.25 bpp`, `829.6 MiB`, `PPLx=2.76×`, `Top-1=55.4%`, `18.6 tok/s`; `experiments/exp-005-event-horizon-wave2-gptq-scaling.toml`).

## Known limits and blockers

- QMF v0/v1 remain experimental research formats and are not yet compatibility-stable across major milestones.
- Uniform scalar `1.73 bpp` ternary still hits a perplexity cliff at `<= 1.7B` even with block-64 GPTQ; non-uniform 2-stage RVQ/AQLM GPU lookup-table kernels and `7B–14B` scaling are required to push below `2.0 bpp`.
- Calibrating and evaluating `7B–32B` BF16 reference models (`15–64 GB`) exceeds 8 GB VRAM; layer-by-layer streaming CPU/hybrid calibration & reference caching is required in Wave III.
- `GpuFull` currently uses FP32 KV storage while `KvCacheMode::Int8` runs on CPU/hybrid; GPU-resident INT8 KV-cache is Wave III work.
- Safe cross-platform WGPU intentionally does not expose GPU temperature, board power draw, or OS-wide VRAM usage; `qml-desktop` explicitly notes this and never fabricates telemetry.

## Next steps (PROJECT EVENT HORIZON Wave III — `v0.6.0`)

1. **Layer-by-Layer Streaming Calibration & Reference Evaluation for `7B–32B` Models:** Stream BF16 safetensors one transformer block at a time during calibration (`H_b` collection) and reference logit computation so `Qwen2.5-7B` (`15.2 GB`), `14B` (`28 GB`), and `27B–32B` (`54–64 GB`) calibrate and forge cleanly on 32 GB RAM + 8 GB VRAM without needing the full BF16 model in VRAM or RAM!
2. **Stage D (`Qwen2.5-7B-Instruct`, 7.6B params) Live 8 GB GPU Breakthrough:** Compress `15.2 GB` BF16 (which CANNOT fit on an 8 GB RTX 4070!) into `~2.1–3.1 GiB` (`2.25–3.25 bpp`) and execute natively on `GpuFull` at interactive speed with full held-out capability evaluation.
3. **GPU-Resident INT8 KV-Cache & Sub-2-Bit (`< 2.0 bpp`) Kernels:** Port `KvCacheMode::Int8` directly into `gpu_full.rs` WGSL shaders, add larger-block (`128`) Hessian GPTQ + GPU-resident 2-stage RVQ / AQLM LUT matvec shaders (`MATVEC_VQ_WGSL`) and mixed-precision `QMR-200 / QMR-175 / QMR-150` to break the `< 2.0 bpp` barrier!
