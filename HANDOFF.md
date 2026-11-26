# Handoff

## Current state

- Public repository: <https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory>
- Development branch: `main` only.
- Local workspace: `C:\Users\Shame\.gemini\antigravity\scratch\Quantum-Morph-Laboratory` (Windows 11, Intel Core i9-14900HX, RTX 4070 Laptop GPU 8 GB VRAM, 32 GB RAM, 2560×1440 144 Hz display).
- Local pretrained model cache (NEVER commit to Git): `C:\Users\Shame\.gemini\antigravity\scratch\qml-models-cache` (`smollm2-135m` and `smollm2-360m`).
- Latest completed milestone: **`v0.4.0` — PROJECT RESONANCE & EVENT HORIZON Wave I** (Complete-Artifact Sub-3-Bit Compression Engine `3.00 -> 1.50 bpp`, 9 Persistent GPU-Resident WGSL Compute Kernels, 4-Category Capability Retention Suite, Q-Genesis Extreme, Consumer VRAM Fit Matrix, `exp-004` Real-Model Validation, and `exp-003` Evidence Audit Reconciliation).
- **Permanent Release Distribution Policy:** All user-facing releases use clean semantic versions (`v0.3.0`, `v0.4.0`, ...) and MUST be published as full, visible public GitHub Releases (`draft: false`, `prerelease: false`, `make_latest: true`) with all 4 compiled assets (`Quantum-Morph-Laboratory-<tag>-Windows-x64.exe`, `Quantum-Morph-Laboratory-<tag>-Windows-x64-Portable.zip`, `Quantum-Morph-Laboratory-<tag>-Linux-x64.tar.gz`, `SHA256SUMS.txt`), full `RELEASE_NOTES.md`, and prominent `README.md` download links. Prereleases are strictly opt-in when explicitly requested by the repository owner. A Git tag, CI build, Actions artifact, or source archive alone is NEVER a completed release.
- The workspace contains **13 cohesive safe-Rust crates** (`qml-core`, `qml-hardware`, `qml-model`, `qml-scope`, `qml-quant`, `qml-format`, `qml-forge`, `qml-tokenizer`, `qml-runtime`, `qml-behavior`, `qml-genesis`, `qml-cli`, `qml-desktop`) with `#![forbid(unsafe_code)]` and zero Python dependencies.
- **License:** MIT ([LICENSE](LICENSE)). All workspace crates inherit `license.workspace = true`.
- **Permanent Memorial Dedication:** The memorial dedication to Ashlan Haley Rapponotti at the end of `README.md` MUST be preserved verbatim across all future updates and releases.

## Implemented capabilities

- `qml-core`: Dense linear algebra, SplitMix64 PRNG, deterministic one-sided Jacobi exact thin SVD, HMT randomized SVD, spectral metrics, Grassmannian subspace affinity, and pure safe-Rust FIPS 180-4 SHA-256.
- `qml-hardware`: Cross-platform CPU/SIMD/RAM telemetry, WGPU GPU adapter discovery, enforced `ResourcePreset` budgets, and **9 persistent VRAM WGSL compute kernels** (`FP32`, `FP16`, `BF16`, `INT8_SYM`, `Q4_0`, `Q4_1`, `INT3_ASYM`, `INT2_ASYM`, `TERNARY`) with fast Walsh–Hadamard input pre-rotation and reusable zero-allocation scratch buffers (`GPU_SCRATCH_MAX_ELEMS = 131,072`).
- `qml-model`: Bounded read-only Safetensors & GGUF v2/v3 streaming payload readers (`PayloadBudget` up to 16 GiB), exact safe-Rust codecs (`F64`, `F32`, `F16`, `BF16`, `F8_E4M3`, `F8_E5M2`, `Q8_0`, `Q4_0`, `Q4_1`, ints, bool), sharded `index.json` & HF `config.json` parsers.
- `qml-scope`: Two-pass streaming Welford statistics, sparsity, 3σ/6σ outliers, crest factor, 32-bin histogram & Shannon entropy, quantization susceptibility proxy, 2D SVD spectrum, transformer role classification, and pairwise cross-layer Grassmannian subspace affinity.
- `qml-quant`: Rayon-parallelized conventional baselines (`FP16`, `BF16`, `INT8_SYM`, `INT8_ASYM`, `Q4_0`, `Q4_1`, `LowRankSvd`, `SparseResidual`); RMQ Shared-Basis (`U·C_l·Vᵀ + R_l` via 2D-HOOI + outlier peeling) & Shared-Core (`A_l·S·B_l + R_l`); two-pass streaming `MAP`; and **PROJECT RESONANCE (`resonance.rs`)** — blockwise Randomized FWHT incoherence, LS-refitted `Int3AsymGroupMatrix`/`Int2AsymGroupMatrix`, base-3 packed `TernaryPackedMatrix` (5 trits/byte, Lloyd–Max), 1/2-stage k-means++ `VectorQuantMatrix` with LUT matvec, **PROJECT ECHO** (`EchoCorrectedMatrix` INT8 low-rank + sparse residual), variance-additive `MAP-B` (`QMR-300..150`), and `estimate_consumer_hardware_fit`.
- `qml-format`: Experimental QMF v0 & multi-GB QMF v1 with out-of-core lazy `QmfFileReader`, atomic publication (`write_to_file_atomic`) with source-output alias protection, 64 KiB streaming CRC-32 + SHA-256 verification, `Arc` shared-latent deduplication, `RmqSharedCore`, sub-3-bit operators (`Int3AsymGroup`, `Int2AsymGroup`, `TernaryPacked`, `VectorQuant`, `EchoCorrected`), and on-demand `decode_row`.
- `qml-forge`: Streaming tensor-by-tensor & cluster-by-cluster Safetensors → QMF v1 pipeline with cooperative `ForgeControl` cancellation, live progress callbacks, `RmqAffinityGated`, sub-3-bit strategies (`QMR-300..150`, `Int3Hadamard`, `Int2Hadamard`, `TernaryHadamard`, `VectorQuant`, `Int2Echo`, `BehaviorProtected`), atomic publication, and direct round-trip matvec verification.
- `qml-tokenizer`: Pure safe-Rust Hugging Face `tokenizer.json` Byte-Level BPE tokenizer with individual-digit pre-tokenization, ChatML formatting, and stateful UTF-8 `StreamingDecoder`.
- `qml-runtime`: Native Llama-style Transformer executing directly over QMF v1 compressed operators (including all sub-3-bit + ECHO variants) on CPU (Rayon) or Persistent GPU Hybrid (`QCoreBackend::GpuHybrid`), plus `forward_token_with_trace` and strictly runtime-only reversible `LayerIntervention` hooks.
- `qml-behavior`: Layer-by-layer hidden activation tracing (`compare_model_behavior`), cosine similarity, relative L2 divergence, Representational Similarity Analysis (RSA), reversible causal ablation studies (`run_intervention_study`), `derive_layer_importance_multipliers`, and held-out 4-category **Capability Retention Suite** (`Reasoning`, `Coding`, `Knowledge`, `InstructionFollowing` + `catastrophic_failure`).
- `qml-genesis`: **Q-Genesis Extreme** role- and depth-stratified multi-objective NSGA-II evolutionary search (including sub-3-bit Hadamard & ECHO layer genes), user profiles (`Balanced`, `Smallest`, `BestQuality`, `Fastest`), persistent JSON archives, and direct Genome-to-QMF v1 export.
- `qml-cli`: 15 laboratory commands (`doctor`, `inspect-safetensors`, `inspect-gguf`, `inspect-qmf`, `scope`, `baseline`, `rmq-ablate`, `morph`, `compare`, `resonance-ladder`, `infer-demo`, `generate`, `behavior`, `genesis`, `demo-rank1`).
- `qml-desktop`: **PROJECT PRISM** native WGPU 3D scientific observatory (`eframe` WGPU + `wgpu 24`, 7 custom WGSL shaders, `Rgba16Float` HDR selective bloom, ACES filmic tone mapping, interactive Orbit/Ortho cameras & 3D ray picking, automatic 2.5D CPU fallback, 5 persisted graphics presets, 11 real-engine views, `QMR-300..150` Q-Forge presets, and the **Consumer VRAM Fit Matrix**).

## Validation actually executed on `v0.4.0`

All of the following passed locally on Windows 11 with Rust `1.100.0` (`cargo +1.100.0`) and `cargo-deny 0.21.0`:

- `cargo +1.100.0 fmt --all -- --check`
- `cargo +1.100.0 clippy --workspace --all-targets -- -D warnings`
- `cargo +1.100.0 test --workspace` — all **84 automated tests** across **13 crates** passed (including all 9 persistent GPU WGSL compute shaders and PRISM WGPU rendering pipelines on NVIDIA RTX 4070).
- `cargo +1.100.0 build --release -p qml-desktop -p qml-cli`
- `cargo deny check` — `advisories ok, bans ok, licenses ok, sources ok` (zero ignored vulnerabilities).
- **Live Real-Model GPU Experiments (`exp-004`, RTX 4070):** Full Sub-3-Bit Ladder and matched-byte Hadamard controls on `SmolLM2-135M-Instruct` and `SmolLM2-360M-Instruct`, exact CPU/GPU `INT3_HADAMARD` parity check, and Layer-31 selective precision protection study (`experiments/exp-004-event-horizon-sub3bit.toml`).

## Known limits and blockers

- QMF v0/v1 remain experimental research formats and are not yet compatibility-stable across major milestones.
- Weight-only sub-3-bit PTQ hits severe perplexity cliffs below `~3.0 bpp` on tiny `135M–360M` models; activation-aware Hessian/GPTQ/AWQ calibration and `1.5B–32B` model testing are needed in Wave II.
- `QCoreBackend::GpuHybrid` keeps persistent compressed linear projections on GPU across 9 WGSL kernels while running lightweight RMSNorm, RoPE, and KV-cache attention on CPU; full GPU-resident attention & INT8 KV-cache kernels remain Wave II optimization work.
- Safe cross-platform WGPU intentionally does not expose GPU temperature, board power draw, or OS-wide VRAM usage; `qml-desktop` explicitly notes this and never fabricates telemetry.

## Next steps (PROJECT EVENT HORIZON Wave II)

1. **Activation-Aware Calibration (GPTQ / AWQ Hessian Compensation):** Collect calibration-token input covariances (`XᵀX`) in `qml-forge` / `qml-quant` to drive error-compensated sub-3-bit quantization and ECHO recovery adapters below `2.50–2.00 bpp`.
2. **Stage B–F Model Scaling (`1.5B–32B`) & Extended Architectures:** Add Qwen2 attention biases, Mistral/Gemma/MoE support, and GGUF K-quant ingestion (`Q2_K`, `Q3_K`, `Q4_K`, `Q6_K`), then run the complete `QMR-300..150` ladder on `1.5B–7B+` pretrained checkpoints where `embed_tokens` drops below `5%` of parameters.
3. **Full GPU-Resident Attention, RMSNorm, RoPE & INT8 KV-Cache:** Eliminate per-token CPU↔GPU synchronization in `qml-runtime` for interactive `7B–32B` sub-3-bit execution on consumer `6–8 GB` VRAM GPUs.
