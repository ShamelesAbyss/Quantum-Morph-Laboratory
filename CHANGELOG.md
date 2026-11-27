# Changelog

## v0.6.3 — PROJECT CONTINUUM

Restore original artwork and structured scientific presentation; permanently enforce native GitHub Releases-only distribution and the exact memorial through owner rules and Rust CI checks. Harden release validation, complete offline packaging, source archive, source-bound integrity manifest, optional configured signing, scoped permissions, non-overwriting publication and independent public verification. No numerical algorithms, Prism renderer or historical measurements change. Architecture admission and independent real-model reference reassessment remain subsequent work.

## v0.6.2 — PROJECT ATLAS

Disk-backed QMF v1 construction shares encoders with the retained memory builder. Q-Forge streams verified artifacts with independent file, builder-memory, disk, tensor and manifest limits, configurable scratch, explicit overwrite, cancellation and repeated source/index protection. Synthetic parity/resource/cleanup tests instrument bounded builder storage. Quantizer/cluster workspace remains a memory bottleneck. No large-model scientific validation is claimed.


## v0.6.1 — PROJECT AEGIS

Safe no-clobber/authorized artifact publication; strict untied output projections; bounded, checksummed and provenance-verified caches; composite numerical fidelity terminology. Legacy caches are preserved and require regeneration. See RELEASE_NOTES.md and HANDOFF.md.



## `v0.6.0` — PROJECT EVENT HORIZON Wave III (2026-11-26)

- **Layer-by-Layer Streaming BF16 Calibration & Exact Reference Evaluator (`qml-runtime::streaming`, `qml-model::payload`):**
  - Added `stream_calibrate_and_reference` streaming single-file and multi-shard Safetensors checkpoints **one transformer layer at a time** (with `read_tensor_rows_f64` distinct-row embedding seek, zero-`f64` `read_tensor_bf16_bits`, and `<= 32,768`-row chunked `lm_head` GPU evaluation), enabling `14.19 GiB` 7B (and `14B–32B`) calibration & exact BF16 reference evaluation in **`444.5 MiB` peak VRAM** on an 8 GB consumer GPU.
  - Raised default `PayloadBudget` ceilings to `8 GiB/tensor` and `64 GiB/total`, and added compact binary disk caching for `ModelCalibrationProfile` (`QCAL1`) and `ReferenceSuiteCache` (`QREF1`) with automatic cache reuse in `qml resonance-ladder`.
- **GPU-Resident Per-Head INT8 KV-Cache (`qml-hardware::gpu_full`, `qml-runtime`, `qml-cli`):**
  - Added `GpuKvMode::Int8` (`ROPE_AND_KV_APPEND_INT8_WGSL` + `GQA_ATTN_INT8_WGSL`) packing 4× signed `i8` codes per `u32` word with per-head `f32` scales inside VRAM (`~3.88×` smaller KV VRAM than FP32 at `head_dim=128`), exposed via `--kv-cache int8` in `qml generate` and `qml resonance-ladder`.
- **10th Persistent GPU WGSL Kernel (`MATVEC_VQ_WGSL`) & Activation-Weighted 2-Stage RVQ (`qml-hardware`, `qml-quant::gptq`, `qml-forge`, `qml-runtime`):**
  - Implemented `GpuMatrixKind::VectorQuant` and `MATVEC_VQ_WGSL`, wired directly into `GpuHybrid` and single-submission `GpuFull` (`< 1e-4` CPU/GPU parity across all 10 compressed matrix formats).
  - Added `fit_vq_calibrated` (Hadamard-rotated activation-weighted 1/2-stage RVQ with joint alternating stage refinement) and `rvq-200` / `rvq-175` / `rvq-150` CLI ladder presets.
- **Stage D `Qwen2.5-7B-Instruct` (`7.62B`, `14.19 GiB` BF16) Live 8 GB GPU Breakthrough (`exp-006`):**
  - Stream-calibrated 4-shard `14.19 GiB` `Qwen2.5-7B-Instruct` in `196.8s` at **`444.5 MiB` peak VRAM**; all `197/197` compressed projections ran natively on `GpuFull` + GPU INT8 KV-cache (`0` CPU fallback).
  - `INT4_Q4_0_GPTQ` (`4.501 bpp`, `3,795.1 MiB` VRAM, `6.8 tok/s`): `PPLx = 0.98×`, `KL = 0.077`, `Top-1 = 90.9%`, `Ret = 95.4%`.
  - `INT3_HADAMARD_GPTQ` (`3.251 bpp`, `2,741.5 MiB` VRAM, `6.8 tok/s`): `PPLx = 1.07×`, `KL = 0.164`, `Top-1 = 86.3%`, `Ret = 89.7%`.
  - **`INT2_HADAMARD_GPTQ` (`2.251 bpp`, `1,898.7 MiB` VRAM — `7.11×` smaller than BF16, `9.7 tok/s`):** **`PPLx = 1.43×`, `KL = 0.854`, `Top-1 = 65.7%`, `Ret = 67.8%`** (`OK` — fluent explanations & Rust code generation in `1.85 GiB` VRAM!).
  - **`QMR-200_GPTQ` (`2.000` exact complete BPP, `1,670.4 MiB` VRAM — `8.00×` smaller than BF16, `9.1 tok/s`):** **`PPLx = 2.13×`, `KL = 1.633`, `Top-1 = 52.6%`, `Ret = 49.8%`** (`OK` — **exact `2.000 bpp` barrier broken!**).
  - Preserved honest sub-2-bit boundary (`QMR-175_GPTQ` `1.750 bpp` & `TERNARY_HADAMARD_GPTQ` `1.727 bpp` cliff at `PPLx 116–205×`) and `SmolLM2-360M` GPU `MATVEC_VQ_WGSL` negative control (`RVQ2_D8_G128` without Cholesky `H⁻¹` compensation: `KL = 9.865` vs `2.811` Cholesky `INT2`).

## `v0.5.1` — PROJECT EVENT HORIZON Wave II (Public Release Fix, 2026-11-26)
- **Strict WGSL Barrier Uniformity (`qml-hardware::gpu_full`):** Made every `workgroupBarrier()` in `HADAMARD_FWD_WGSL` strictly uniform across all 256 threads for clean cross-vendor WGSL/DX12/Vulkan/Metal execution.
- **Hardware-Only WGPU Compute Selection (`qml-hardware`):** `WgpuMatvecEngine::try_new()` now declines software CPU emulators (`wgpu::DeviceType::Cpu`, such as headless Windows WARP `Microsoft Basic Render Driver` or `llvmpipe`), allowing GPU-less systems and headless CI runners to fall back automatically to native multi-threaded Rayon CPU without software driver faults.

## `v0.5.0` — PROJECT EVENT HORIZON Wave II (2026-11-26)

- **Activation-Aware Blockwise Cholesky GPTQ + Hadamard Engine (`qml-quant::gptq`):**
  - Implemented streaming per-layer input activation covariance collection (`LayerCalibrationAccumulator`, `LayerCalibrationStats`, `ModelCalibrationProfile`, `64×64` block-diagonal `H_b = (2/N) X_bᵀ X_b`) and exact orthogonal Hadamard covariance rotation (`H_b' = Q_b H_b Q_bᵀ`).
  - Implemented Rayon-parallelized Blockwise Cholesky GPTQ with descending activation-salience ordering (`actorder`), Ledoit–Wolf-style adaptive sample shrinkage damping, activation-weighted least-squares affine/symmetric grid refitting (`fit_weighted_affine_group`, `fit_weighted_sym_scale`), and activation-weighted **PROJECT ECHO** (`fit_echo_calibrated`) across `INT4_SYM`, `INT3_ASYM`, `INT2_ASYM`, and `TERNARY_PACKED` with **zero extra storage bytes**.
- **Full GPU-Resident Single-Submission WGSL Transformer (`qml-hardware::gpu_full` + `qml-runtime::QCoreBackend::GpuFull`):**
  - Implemented 6 new WGSL compute shaders (`RMSNORM_WGSL`, `HADAMARD_FWD_WGSL` with bit-packed exact signs, `ROPE_AND_KV_APPEND_WGSL`, `GQA_ATTN_WGSL` with numerically stable online softmax, `SWIGLU_WGSL`, `VEC_ADD_WGSL`) chaining all 24–36 transformer layers + `lm_head` into **ONE single `CommandEncoder` submission per token** with zero intermediate CPU readbacks.
  - Raised `GPU_SCRATCH_MAX_ELEMS` to `262,144` (covering 152K Qwen2.5 & 256K Gemma vocabularies) and requested full `adapter.limits()` so multi-GB VRAM tensors upload cleanly.
  - Verified **`22.1×` real end-to-end speedup** (`1.84 -> 40.70 tok/s`, up to `63.3 tok/s` at 2-bit) over `GpuHybrid` on `SmolLM2-360M` (RTX 4070) with identical output and `< 1e-3` logit RMSE parity vs CPU.
- **Qwen2 Architecture, Sharded Multi-GB Streaming, INT8 KV-Cache & GGUF K-Quants (`qml-model`, `qml-forge`, `qml-runtime`, `qml-behavior`, `qml-cli`, `qml-desktop`):**
  - Added Qwen2/Qwen2.5 optional attention biases (`q_proj.bias`, `k_proj.bias`, `v_proj.bias`) across CPU, `GpuHybrid`, and `GpuFull`.
  - Added unified single + multi-shard `SafetensorsSource` (`model.safetensors.index.json`), zero-duplicate-RAM `forge_path_to_file_atomic`, raised default `PayloadBudget` to `4 GiB/tensor` & `32 GiB/total`, and added exact safe-Rust **GGUF K-quant** payload decoding (`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`).
  - Added `KvCacheMode::Int8` (per-token symmetric INT8 + f16 scale KV-cache) and `ReferenceSuiteCache` (freeing 100% of BF16 reference VRAM before loading large compressed candidates).
  - Upgraded `qml resonance-ladder --calibrate`, `qml generate --backend gpu` (`GpuFull`), `qml morph` (sharded support), and `qml-desktop` Q-Core Inference Chamber to `QCoreBackend::GpuFull`.
- **Progressive Multi-Scale Real-Model Evidence (`exp-005`, Stage A `360M` → Stage B `1.5B` & `1.7B` → Stage C 2-Shard `3B`):**
  - `SmolLM2-360M`: GPTQ+Hadamard cut `INT3` KL by `−67.3%` (`1.910 -> 0.624`), `QMR-300` KL by `−61.8%` (`2.803 -> 1.072`), broke the `QMR-275` (`2.75 bpp`) cliff (`PPLx 3.52×`), and cut `INT2` PPLx by `196×` (`2,653.7× -> 13.53×`, `41,068×` vs plain 2-bit) at `63.3 tok/s`.
  - `SmolLM2-1.7B` (`3.42 GB` BF16): `INT3_HADAMARD_GPTQ` hit `83.0%` Top-1 in `663.9 MiB` VRAM, and **`QMR-250_GPTQ` (`2.500 bpp`, `510.5 MiB` VRAM, `22.6 tok/s`)** broke the `2.50 bpp` barrier (`PPLx = 2.79×`, `58.0%` Top-1).
  - `Qwen2.5-1.5B` (`3.09 GB` BF16, Qwen2 biases): `INT3_HADAMARD_GPTQ` achieved **`PPLx = 1.03×`** (`+3%` over BF16!) at `3.253 bpp` (`599.7 MiB` VRAM, `22.8 tok/s`), and **`INT2_HADAMARD_GPTQ` (`2.253 bpp`, `415.7 MiB` VRAM, `31.5 tok/s`)** broke the `2.25 bpp` 2-bit cliff (`PPLx = 4.12×`, `41.7%` Top-1).
  - **Stage C `Qwen2.5-3B` (2-shard `6.17 GB` BF16):** `INT4_Q4_0_GPTQ` hit **`PPLx = 1.01×`** (`91.4%` retention, `1.62 GiB` VRAM), `INT3_HADAMARD_GPTQ` hit **`PPLx = 1.07×`** (`KL = 0.237`, `1.17 GiB` VRAM), and **`INT2_HADAMARD_GPTQ` (`2.253 bpp`, `829.6 MiB` VRAM — `7.43×` smaller, `18.6 tok/s`)** achieved **`PPLx = 2.76×` (`OK`), `KL = 1.606`, `Top-1 = 55.4%`** with coherent English generation.

## `v0.4.0` — PROJECT RESONANCE & EVENT HORIZON Wave I (2026-11-26)

- **Sub-3-Bit Neural Representation Engine (`qml-quant::resonance`):**
  - Added deterministic blockwise Randomized Fast Walsh–Hadamard Transform (`fwht_in_place`, `apply_hadamard_forward`, `apply_hadamard_inverse`, `HadamardSpec`) with zero-overhead SplitMix64 sign vectors for incoherence processing.
  - Added closed-form least-squares alternating affine refit + Gaussian sigma-clipping (`fit_affine_uniform_group`) and Rayon row parallelism for `Int3AsymGroupMatrix` (8 codes / 3 bytes), `Int2AsymGroupMatrix` (4 codes / byte), and `TernaryPackedMatrix` (5 trits / byte base-3 packing, `1.60 bpp` raw, Lloyd–Max threshold).
  - Added `VectorQuantMatrix` (1/2-stage k-means++ Residual Vector Quantization with FP16 codebooks and zero-multiply LUT-accelerated `matvec`) and **PROJECT ECHO (`EchoCorrectedMatrix`)** pairing compact INT8 low-rank SVD adapters + optional sparse FP16 residuals with complete byte accounting.
  - Added the complete-artifact Sub-3-Bit Research Ladder (`ResonanceProfile`: `QMR-300` through `QMR-150`), `BehaviorImportanceMap`, variance-additive scale-invariant `allocate_resonance_map_b` allocator, fast probe-only matvec scoring, and `estimate_consumer_hardware_fit` (`1.7B–32B` VRAM planner).
- **Backward-Compatible QMF v1 Extension (`qml-format`):**
  - Extended `QmfRepresentation`, `QmfV1Builder`, `QmfOperator` (with `decode_row`), and out-of-core `QmfFileReader` to serialize, CRC-32 verify, and execute `Int3AsymGroup`, `Int2AsymGroup`, `TernaryPacked`, `VectorQuant`, and `EchoCorrected`.
- **9 Persistent GPU-Resident WGSL Compute Kernels (`qml-hardware` + `qml-runtime`):**
  - Added 3 direct-compressed sub-3-bit WGSL shaders (`MATVEC_INT3_ASYM_WGSL`, `MATVEC_INT2_ASYM_WGSL`, `MATVEC_TERNARY_WGSL`) with fast input Walsh–Hadamard pre-rotation inside `PersistentGpuMatrix::matvec`, verified to `< 1e-4` against CPU reference on NVIDIA RTX 4070 and exact 3-decimal full-model parity on `SmolLM2-135M`.
- **Q-Forge, Q-Behavior, Q-Genesis Extreme, CLI & Desktop (`qml-forge`, `qml-behavior`, `qml-genesis`, `qml-cli`, `qml-desktop`):**
  - Added sub-3-bit strategies (`Int3Hadamard`, `Int2Hadamard`, `TernaryHadamard`, `VectorQuant`, `Int2Echo`, `ResonanceMapB`, `BehaviorProtected`) to Q-Forge, `qml morph`, and Q-Forge Studio.
  - Added held-out 4-category **Capability Retention Suite** (`Reasoning`, `Coding`, `Knowledge`, `InstructionFollowing` + `catastrophic_failure` flag) and `derive_layer_importance_multipliers` to `qml-behavior`.
  - Added sub-3-bit genes (`Int3Hadamard`, `Int2Hadamard`, `TernaryHadamard`, `Int2Echo`) to `qml-genesis`, `qml resonance-ladder` to `qml-cli`, and the **Consumer VRAM Fit Matrix** to `qml-desktop` System Observatory.
- **Reproducible Real-Model Research (`exp-004`) & Evidence Audit:**
  - Archived full `SmolLM2-135M-Instruct` and `SmolLM2-360M-Instruct` sub-3-bit ladder measurements, matched-byte Hadamard controls (`−37.9%` / `−39.5%` Mean KL and `137.5×` / `209.4×` lower perplexity inflation with zero extra bytes), positive scaling confirmation (`135M -> 360M`), and the Layer-31 selective protection study (`experiments/exp-004-event-horizon-sub3bit.toml`).
  - Reconciled `README.md` `exp-003` metrics verbatim with `experiments/exp-003-smollm2-360m-gpu.toml` and fixed `.github/workflows/release.yml` so `workflow_dispatch` explicitly checks out `inputs.tag`.

## Earlier History (`v0.1.0-alpha.1` → `v0.3.0`)

- Initialized the Rust workspace with `qml-core`, `qml-format`, and `qml-cli`.
- Added safe dense-matrix primitives and unit tests.
- Added basic `qml help`, `qml version`, and `qml doctor` commands.
- Added Linux/Windows CI and project architecture, roadmap, build, contribution, handoff, and security documentation.
- Completed a first source-backed prior-art batch covering 17 topics, with detailed notes, an index, a comparative synthesis, and an updated novelty ledger.
- Added deterministic largest-pivot rank-one cross approximation, factorized matrix-vector multiplication, and reconstruction-error measurement as conventional CPU baselines.
- Added `qml demo-rank1` for a generated synthetic fixture and `qml inspect-qmf` for an experimental single-matrix QMF v0 container; both report limitations explicitly.
- Added QMF v0 deterministic serialization, bounded parsing, checksum validation, and tests for round trips, corruption, unsupported fields, non-finite values, and excessive declared shapes.
- Added `qml-model` with bounded, read-only Safetensors metadata inspection and text/JSON CLI reporting. The parser validates duplicate keys, dtype/shape size arithmetic, offsets, contiguous layouts, header/resource limits, and file length without reading payload bytes.
- Added property-based tests and generated F32/F4 differential fixtures checked by the exact-pinned Safetensors 0.8.0 dev dependency; no model weights or third-party datasets are included.
- Added a source-linked Safetensors research note and an architectural decision record for the header-only parser and its tradeoffs.
- **Milestone I — Q-Scope Foundation:**
  - Extended `qml-core` with full dense linear algebra (`transpose`, `matmul`, `matvec_transposed`, cosine similarity), deterministic SplitMix64 PRNG, exact one-sided Jacobi thin SVD, deterministic Halko–Martinsson–Tropp randomized SVD, spectral metrics (stable rank, Roy–Vetterli effective rank, energy ranks), and Grassmannian subspace affinity.
  - Extended `qml-model` with bounded & constant-memory streaming Safetensors tensor payload access (`SafetensorsFile`, `PayloadBudget`), exact safe-Rust dtype codecs (`F16`, `BF16`, `F8_E4M3`, `F8_E5M2`, `F32`, `F64`, ints, bool), sharded Safetensors index parser, Hugging Face `config.json` parser, and bounded read-only GGUF v2/v3 metadata inspector (`inspect_gguf`).
  - Added `qml-scope` (Q-Scope Neural Model Observatory) with two-pass streaming distributional statistics, sparsity & outlier detection, histogram/entropy estimation, quantization susceptibility proxy, 2D matrix SVD structure analysis, transformer role classification, and cross-layer subspace affinity analysis.
  - Added `qml scope` and `qml inspect-gguf` CLI commands and upgraded `qml doctor` with safe CPU SIMD feature detection.
- **Milestones II & III — Quantization Baselines, RMQ & MAP (`qml-quant`):**
  - Implemented reproducible conventional baselines (`FP16`, `BF16`, `INT8_SYM`, `INT8_ASYM`, `INT4_SYM_G32(Q4_0)`, `INT4_ASYM_G32(Q4_1)`, `INT4_SYM_G128`, `LowRankSvdMatrix`, `SparseResidual`) with exact complete byte accounting (`StorageBreakdown`) and direct compressed `LinearOperator::matvec`.
  - Implemented experimental RMQ Shared-Basis (`U_shared * C_l * V_sharedᵀ + R_l` via deterministic 2D-HOOI) and RMQ Shared-Core (`A_l * S * B_l + R_l`) engines with sparse outlier and INT4 residual options plus head-to-head baseline ablation (`compare_rmq_vs_baselines`), validated on both correlated clusters and uncorrelated full-rank Gaussian controls.
  - Implemented experimental MAP (Morphological Adaptive Precision) allocator combining local sensitivity, outlier crest factor, and bounded downstream spectral error-propagation amplification under a strict global byte budget.
- **Milestone IV — QMF v1 & Q-Forge (`qml-format` v1, `qml-forge`):**
  - Evolved `qml-format` with experimental QMF v1 (`QmfArchiveV1`, `QmfV1Builder`, `QmfOperator`) supporting multi-tensor directories, shared latent tables for RMQ cross-layer bases, FP32/FP16/BF16, groupwise INT8/INT4, truncated SVD, sparse COO FP16 residuals, deterministic JSON manifests, and per-blob + whole-file CRC-32 integrity checks (`research/qmf-experimental-v1.md`).
  - Added `qml-forge` transformation pipeline (`forge_safetensors_to_qmf`, `compare_safetensors_and_qmf`) preserving read-only source files and verifying direct compressed execution after full QMF v1 serialization round trips.
- **Milestone V — Q-Core Native Inference, Q-Genesis & Laboratory CLI (`qml-runtime`, `qml-genesis`, `qml-cli`):**
  - Added `qml-runtime` (Q-Core) native CPU decoder-only Transformer executing directly over QMF v1 compressed operators with RMSNorm, RoPE, Grouped-Query Attention + KV-cache, SwiGLU MLP, autoregressive token generation, and KL-divergence / perplexity evaluation.
  - Added `qml-genesis` (Q-Genesis) multi-objective evolutionary search discovering Pareto-optimal hybrid precision/SVD/RMQ genomes with persistent JSON experiment archives.
  - Integrated `baseline`, `rmq-ablate`, `morph`, `compare`, `infer-demo`, `genesis`, and QMF v0/v1 `inspect-qmf` into `qml-cli`.
- **Milestone VI — Algorithmic Refinements, Reproducible Benchmarks & Visual Identity:**
  - Added RPCA-inspired outlier-peeling refinement to `RmqSharedBasisCluster::fit`, cutting correlated-cluster reconstruction error by 49.2% at identical `0.867` effective bpp.
  - Added NSGA-II-style elite capping (`pop_size / 2` spaced across BPP) to `qml-genesis` so every generation breeds fresh crossover/mutated genomes.
  - Recorded measured release-mode Windows 11 / Intel i9-14900HX experimental data in `experiments/exp-001-rmq-map-qcore-baseline.toml`.
  - Created custom deep-obsidian/ultraviolet/cyan SVG visual identity (`assets/hero.svg`, `assets/architecture.svg`, `assets/benchmarks.svg`) and refreshed `README.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `research/novelty-ledger.md`, and `HANDOFF.md`.
- **Project Singularity — Phase I (Audit Remediation & Core Hardening):**
  - Resolved Rust 1.99+ strict Clippy `clippy::chunks_exact_to_as_chunks` warnings in `qml-model` and `qml-format` using `as_chunks::<N>().0`.
  - Implemented pure safe Rust FIPS 180-4 SHA-256 (`qml_core::sha256`) verified against NIST vectors, and replaced placeholder hashes in `experiments/exp-001-rmq-map-qcore-baseline.toml` and `ForgeReport`/`CompareReport`.
  - Upgraded QMF v1 with configurable multi-GB `QmfBudget` (default 8 GiB), streaming header-only `inspect_qmf_v1_header`, `QmfLatentCache` (`Arc<QuantizedFactor>` runtime deduplication), first-class `RmqSharedCore` serialization/execution, and on-demand `decode_row`.
  - Refactored `qml-forge` to operate strictly tensor-by-tensor (single-tensor methods) and cluster-by-cluster (RMQ methods), bounding peak transformation memory, and added `--method rmq-core` to `qml-cli`.
  - Eliminated dense `f64` embedding table reconstruction in `qml-runtime` (`QCoreModel` now performs on-demand `embed_tokens.decode_row(token_id)` and supports `tie_word_embeddings` + `QmfLatentCache`).
  - Added matched-exact-byte-budget MAP vs uniform Q4_0 unit test in `qml-quant`.
- **Project Singularity — Phases II–IV (Real Pretrained Model Validation, WGPU Compute, Rayon Parallelism & Native Windows Desktop App `v0.1.0-alpha.1`):**
  - Added `qml-tokenizer`: pure safe Rust Hugging Face `tokenizer.json` Byte-Level BPE tokenizer with individual-digit pre-tokenization, special tokens, ChatML formatting, and streaming token decode.
  - Added `qml generate` CLI command and `TransformerConfig::from_hf_config_bytes` / `QCoreModel::generate_with_callback`.
  - Parallelized `qml-quant` compressed `matvec` kernels across CPU threads using `rayon`.
  - Validated the complete end-to-end pipeline on real `HuggingFaceTB/SmolLM2-135M-Instruct` (`experiments/exp-002-smollm2-135m.toml`): Q-Scope discovered strong adjacent-layer `k_proj` right-subspace Grassmannian affinity (`0.6455`), Q-Core generated coherent English text directly from compressed QMF v1 weights, quantization-only MAP at `4.521 bpp` beat uniform `Q4_0` (`4.511 bpp`) by 16.4% KL divergence, and the unrestricted SVD-8 MAP failure mode was honestly preserved as a negative control.
  - Added `qml-hardware`: cross-platform CPU/SIMD/RAM telemetry, WGPU GPU adapter discovery, resource presets, and a validated WGSL compute shader `f32` matvec prototype verified against CPU on NVIDIA RTX 4070 Laptop GPU.
  - Added `qml-desktop` (`v0.1.0-alpha.1`): native Windows desktop research workbench (`eframe`/`egui` + `egui_plot` + `rfd`) with Deep Obsidian / Ultraviolet / Electric Cyan visual identity, crash logging, persistent library state, and 10 real-engine views.
- **Project Prism (`v0.3.0` — Public Release & Distribution Overhaul):**
  - Published **Quantum Morph Laboratory `v0.3.0` | PROJECT PRISM** as a full, visible public GitHub Release (`draft: false`, `prerelease: false`, `make_latest: true`) with standardized standalone Windows x64 executable (`Quantum-Morph-Laboratory-v0.3.0-Windows-x64.exe`), Windows x64 Portable ZIP, Linux x64 tarball, and verified `SHA256SUMS.txt`.
  - Permanently upgraded `.github/workflows/release.yml` to enforce full validation (`fmt`, `clippy --workspace --all-targets -- -D warnings`, `test --workspace`), remove automatic hyphen-based prerelease hiding, verify SHA-256 checksums before publishing, and dynamically title releases from `RELEASE_NOTES.md`.
  - Added prominent top-of-README direct download badges/links (`/releases/latest` + direct asset URLs), comprehensive `RELEASE_NOTES.md`, and permanent release distribution rules in `AGENTS.md` and `HANDOFF.md`.
  - Migrated `qml-desktop` to native **WGPU (`wgpu 24`)** rendering via `eframe::egui_wgpu::CallbackTrait` with automatic 2.5D CPU fallback and **7 original WGSL shaders** (`BACKDROP_WGSL`, `CRYSTAL_WGSL`, `FILAMENT_WGSL`, `PARTICLE_WGSL`, `BLOOM_THRESHOLD_WGSL`, `BLOOM_BLUR_WGSL`, `COMPOSITE_WGSL`) driving an `Rgba16Float` HDR offscreen target, `Depth32Float` depth buffer, half-resolution separable 9-tap Gaussian selective bloom, and ACES filmic tone mapping.
  - Built pure-Rust `OrbitCamera` (orbit, pan, zoom, double-click reset, Perspective/Orthographic toggle, 3D ray-sphere picking, fullscreen 3D mode) and **7 interactive 3D workspaces**: Computational Singularity Command Center, Q-Scope 3D Neural Anatomy (6 visual modes on real measured tensors), Q-Entangle 3D Latent Constellation, Q-Forge Morphogenesis Chamber, Q-Genesis 3D Evolutionary Crucible (with recorded `parent_ids` lineage), Q-Behavior Dual Activation Observatory, and Hardware Observatory Computational Anatomy.
  - Added persisted **PROJECT PRISM Graphics Quality Presets** (`Minimal`, `Balanced`, `High`, `Ultra`, `Cinematic`), live measured FPS, 11 custom procedural vector subsystem icons, collapsible rail (`Ctrl+B`), global keyboard shortcuts (`1–9/0/-`, `F`, `P`, `R`, `Esc`), QMF proportional storage treemap + radial precision ring, Q-Core Zen Mode, procedural 64×64 window icon, `assets/qml-prism-logo.svg`, and 6 visually audited native GPU screenshots (`2560×1440` & `1920×1080` at `157–187 FPS` on RTX 4070) in `assets/screenshots/`.
- **Project Transcendence — Wave III (`v0.2.0-alpha.2` — Q-Behavior Mechanistic Observatory & Scalable Q-Genesis):**
  - Added 13th safe-Rust crate `qml-behavior` (`compare_model_behavior`, `run_intervention_study`, `LayerBehaviorMetrics`, Representational Similarity Analysis) and `QCoreModel::forward_token_with_trace` + reversible `LayerIntervention` (`ScaleResidual`, `AblateAttention`, `AblateMlp`) in `qml-runtime`.
  - Added `qml behavior` CLI command and the 11th real-engine desktop view (`Q-Behavior Observatory`) in `qml-desktop` with interactive `egui_plot` layer cosine/RSA/L2 drift curves and reversible causal intervention controls.
  - Upgraded `qml-genesis` with role- and depth-stratified tensor sampling (`select_stratified_tensor_names`), user preference profiles (`GenesisProfile`: `Balanced`, `Smallest`, `BestQuality`, `Fastest`), and direct Genome-to-QMF v1 export (`export_genome_to_qmf_v1`, `--export-qmf`).
  - Validated `qml behavior` live on `SmolLM2-360M-Instruct` (RTX 4070 Vulkan), discovering a sharp Layer 31 representational cliff under `Q4_0` (`Rel L2` tripling `0.2038 -> 0.6002`, `RSA` dropping `0.9813 -> 0.5697`) and verifying reversible Layer 0 MLP ablation (`KL = 17.23`, `100%` Top-1 flip).
- **Project Transcendence — Wave II (`v0.2.0-alpha.1` — PROJECT ACCELERATION: Persistent GPU-Resident Compressed Q-Core & SmolLM2-360M Validation):**
  - Added direct compressed WGSL compute shaders for `FP16`, `BF16`, and `Q4_1` (interleaved `(scale_f16, min_f16)` 1-instruction `unpack2x16float`) alongside `FP32`, `INT8_SYM`, and `Q4_0` in `qml-hardware`.
  - Implemented `PersistentGpuMatrix` and preallocated reusable scratch buffers (`GPU_SCRATCH_MAX_ELEMS = 131,072`) for zero per-token weight transfers, zero buffer allocations, and zero bind-group allocations.
  - Integrated `QCoreBackend::GpuHybrid` and `QCoreBackendStats` into `qml-runtime`, supporting persistent VRAM execution of all QMF v1 formats including `Arc<PersistentGpuMatrix>` deduplication for RMQ Shared-Basis/Core latents, with `--backend cpu|gpu` in `qml generate` and a GPU Hybrid toggle in `qml-desktop`.
  - Raised default `PayloadBudget` ceilings (`2 GiB` per tensor, `16 GiB` cumulative) and validated end-to-end on real `HuggingFaceTB/SmolLM2-360M-Instruct` (`723.7 MB`, `experiments/exp-003-smollm2-360m-gpu.toml`), achieving up to `20.97 tok/s` (`INT8_SYM` in `355.8 MiB` VRAM, `95.5%` Top-1) and `5.30×` GPU speedup over 32-thread CPU on `MAP 4.52 bpp` (`194.5 MiB` VRAM).
- **Project Transcendence — Wave I (`v0.1.0-alpha.2` — Audit Hardening, Out-of-Core Streaming & GGUF Payload Decoding):**
  - Added out-of-core file-backed `QmfFileReader<R: Read + Seek>` in `qml-format` (zero full-payload RAM buffering, lazy per-tensor seeking with `QmfLatentCache`, and 64 KiB streaming CRC-32 + SHA-256 verification) and wired it into `QCoreModel::from_qmf_reader`, `qml generate`, and `qml-desktop`.
  - Added `QmfArchiveV1::write_to_file_atomic` with canonicalized source-output alias protection (`QmfError::SourceOutputAlias`), sibling temporary file streaming, `sync_all()`, and atomic rename.
  - Refactored `MAP` (`MapLayerProfile`, `profile_layer_for_map`, `allocate_from_profiles`) into a genuinely two-pass streaming pipeline in `qml-quant` and `qml-forge` so decoded dense matrices are never retained together in memory.
  - Added true cooperative cancellation (`ForgeControl`, `ForgeError::Cancelled`) and live per-tensor progress callbacks to `qml-forge`, wired into `qml-desktop` Q-Forge Studio along with real `ResourcePreset` `PayloadBudget` enforcement and strictly opt-in model discovery.
  - Added bounded, read-only GGUF v2/v3 tensor payload decoding (`GgufFile`) in `qml-model` supporting `F32`, `F16`, `BF16`, `Q8_0`, `Q4_0`, and `Q4_1` block layouts.
- **Project Singularity — Post-Audit Hardening & Wave II Enhancements (`v0.1.0-alpha.1`):**
  - Eliminated transitive `RUSTSEC-2026-0194` & `RUSTSEC-2026-0195` (`quick-xml 0.30.0` via `eframe`'s unused `accesskit` -> `atspi` -> `zbus_xml` chain) by using minimal `eframe` features (`default_fonts`, `glow`, `x11`, `wayland`), achieving green `cargo deny check` across advisories, bans, licenses, and sources with zero ignored vulnerabilities.
  - Added direct compressed GPU `INT8_SYM` (`extractBits` + `unpack2x16float`) and `Q4_0 (INT4_SYM)` (nibble unpack + `unpack2x16float`) WGSL compute shaders to `qml-hardware` (zero dense weight reconstruction in VRAM; `3.9×`–`7.1×` VRAM savings vs FP32), verified against CPU on NVIDIA RTX 4070 Laptop GPU.
  - Added stateful UTF-8 `StreamingDecoder` to `qml-tokenizer` and wired it into `qml generate` and `qml-desktop` Q-Core Inference Chamber to prevent replacement glyphs when multi-byte symbols (e.g., `Σ`) span BPE token boundaries.
  - Implemented `TransformStrategy::RmqAffinityGated` in `qml-forge`, `qml morph --method rmq-gated`, and `qml-desktop` Q-Forge Studio: gates RMQ Shared-Basis + sparse residual strictly to clusters exceeding a configurable Grassmannian subspace affinity threshold while routing low-affinity layers to `Q4_0` and embeddings/heads to `INT8_SYM`.
  - Upgraded `qml-desktop` QMF Explorer with an interactive per-representation payload breakdown bar chart, searchable Tensor Directory table, and Shared Latent table; upgraded System Observatory with multi-kernel (`FP32` / `INT8_SYM` / `Q4_0`) GPU self-testing; added native JSON report export to Q-Scope, Q-Forge, and Q-Genesis.
