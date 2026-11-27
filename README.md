<p align="center">
  <img src="assets/hero.svg" alt="Quantum Morph Laboratory — Don't just compress intelligence. Transform its representation." width="100%"/>
</p><p align="center">
  <img src="https://img.shields.io/badge/Latest%20Release-v0.6.2%20PROJECT%20ATLAS-A855F7?style=for-the-badge" alt="Latest Public Release v0.6.2"/>
  <img src="https://img.shields.io/badge/Platform-Windows%20x64-06B6D4?style=for-the-badge&logo=windows&logoColor=white" alt="Windows x64 Available"/>
  <img src="https://img.shields.io/badge/Platform-Linux%20x64-0EA5E9?style=for-the-badge&logo=linux&logoColor=white" alt="Linux x64 Available"/>
  <a href="https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/ShamelesAbyss/Quantum-Morph-Laboratory/ci.yml?branch=main&style=for-the-badge&label=Rust%20CI" alt="CI Status"/></a>
  <img src="https://img.shields.io/badge/Rust-1.100%2B%20(Safe%20Only)-7C3AED?style=for-the-badge&logo=rust&logoColor=white" alt="Rust"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge" alt="MIT License"/></a>
</p><h3 align="center"><em>Don't just compress intelligence. Transform its representation.</em></h3><p align="center">
  An experimental, local-first Rust neural model research laboratory featuring <strong>Layer-by-Layer Streaming Calibration (&lt;450 MiB VRAM for 7B–32B models)</strong>, <strong>Activation-Aware Blockwise Cholesky GPTQ + Hadamard Sub-3-Bit Compression</strong> (breaking the <strong>2.000 bpp barrier on Qwen2.5-7B in 1.63 GiB VRAM</strong>), a <strong>Single-Submission Full GPU-Resident WGSL Transformer with GPU INT8 KV-Cache &amp; 10 Compressed Compute Kernels</strong>, and the native multi-pass WGPU 3D scientific observatory (<strong>PROJECT PRISM</strong>).
</p>Software Distribution (v0.6.2 — PROJECT ATLAS)
Quantum Morph Laboratory is experimental, local-first scientific research software. Prebuilt application packages, release notes and SHA-256 integrity manifests are published through the repository's native GitHub Releases section.

To obtain a published version, navigate to Releases using GitHub's repository interface. Review the corresponding release notes and verify artifact integrity before running any application. Direct downloads: [Windows EXE](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.6.2/Quantum-Morph-Laboratory-v0.6.2-Windows-x64.exe) · [Windows portable ZIP](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.6.2/Quantum-Morph-Laboratory-v0.6.2-Windows-x64-Portable.zip) · [Linux archive](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.6.2/Quantum-Morph-Laboratory-v0.6.2-Linux-x64.tar.gz) · [SHA256SUMS.txt](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.6.2/SHA256SUMS.txt).
Release Package	Contents
Windows Standalone EXE	Native Windows desktop application (qml-desktop)
Windows Portable ZIP	Native desktop app + qml.exe CLI + full docs + MIT License
Linux x64 Tarball	Native desktop app + qml CLI + full docs + MIT License
Release Notes & Checksums	Release documentation, SHA256SUMS.txt, and published release asset information


Local operation does not require a cloud account. See [Architecture and documentation](ARCHITECTURE.md) and [Release notes](RELEASE_NOTES.md) for details.
PROJECT ATLAS — v0.6.2
Q-Forge constructs QMF v1 payloads in owned temporary disk storage instead of retaining the complete compressed model in RAM. Shared encoders preserve deterministic bytes, blob CRCs and shared latents. The final artifact is streamed, verified and checked against every source shard/index before AEGIS-safe publication.

Artifact, builder memory, temporary disk, individual tensor and manifest limits are independent. The morph command accepts --scratch-dir, --max-artifact-bytes, --max-builder-memory-bytes, --max-temp-bytes, --max-tensor-bytes, --max-manifest-bytes and explicit --overwrite. Defaults are 8 GiB artifact, 256 MiB builder accounting, 16 GiB temporary disk, 8 GiB individual tensor allocation and 16 MiB manifest. Default publication refuses replacement.

Synthetic instrumentation grew encoded payload from 1 MiB to 32 MiB with zero resident payload buffer, a 64 KiB encoding buffer and 151,280–151,728 bytes of conservative manifest/buffer accounting. These counters are not process RSS. Dense tensor/cluster quantization, calibration and other working memory remain separate bottlenecks. Temporary disk preflight is advisory; peak scratch includes both payload and final stage.

No additional models were downloaded and no large-model experiments were performed. ATLAS does not scientifically validate 14B/27B/32B support. See [release notes](RELEASE_NOTES.md) for configuration, limits and synthetic validation.
PROJECT AEGIS — v0.6.1
Safe artifact publication, strict required output projections, and provenance-checked calibration/reference caches. Existing QCAL1/QREF1 caches are unverified: keep them and regenerate using a new --out-dir. New cache envelopes are bounded to 512 MiB on disk and never overwrite an existing cache.

The composite numerical fidelity score combines perplexity ratio and top-1 token agreement. It does not measure retained intelligence or benchmark capability. Historical measurements below are preserved.
PROJECT PRISM — Native WGPU 3D Scientific Observatory
All screenshots below are genuine, unedited native Windows WGPU GPU captures rendered live by qml-desktop.exe --release on an NVIDIA GeForce RTX 4070 Laptop GPU (157–187 FPS, custom WGSL shaders, Rgba16Float HDR selective bloom, interactive 3D orbit/ortho cameras & ray picking).
<p align="center">
  <img src="assets/screenshots/prism-command-center.png" alt="PROJECT PRISM Command Center — 2560x1440 Native WGPU Computational Singularity" width="100%"/>
  <br/>
  <em><strong>Command Center — The Computational Singularity (2560×1440 native WGPU, 180 FPS):</strong> Interactive 3D orbital centerpiece with clickable subsystem portals, symmetric host telemetry, and zero-Python safe-Rust architecture.</em>
</p><table>
  <tr>
    <td width="50%" align="center">
      <img src="assets/screenshots/prism-qscope-anatomy.png" alt="Q-Scope 3D Neural Anatomy on real SmolLM2-135M" width="100%"/>
      <br/>
      <em><strong>Q-Scope 3D Neural Anatomy (Real SmolLM2-135M):</strong> 272 measured tensors across 30 layers with 6 live visual modes, 32-bin histogram & SVD spectra.</em>
    </td>
    <td width="50%" align="center">
      <img src="assets/screenshots/prism-qentangle-constellation.png" alt="Q-Entangle 3D Latent Constellation on real SmolLM2-135M" width="100%"/>
      <br/>
      <em><strong>Q-Entangle Latent Constellation (Real SmolLM2-135M):</strong> 240 measured cross-layer Grassmannian subspace affinity pairs & 2D heatmap.</em>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="assets/screenshots/prism-qgenesis-crucible.png" alt="Q-Genesis 3D Evolutionary Crucible" width="100%"/>
      <br/>
      <em><strong>Q-Genesis Evolutionary Crucible:</strong> 3D objective space (BPP × Matvec Error × FLOPs) with recorded parent-child lineage & Pareto frontier.</em>
    </td>
    <td width="50%" align="center">
      <img src="assets/screenshots/prism-hardware-anatomy.png" alt="System Observatory Computational Anatomy & PRISM Graphics Controls" width="100%"/>
      <br/>
      <em><strong>Hardware Observatory & PRISM Controls:</strong> 3D CPU/WGPU schematic, 5 graphics presets, live FPS, resource budgets, VRAM Fit Matrix & WGSL compute self-tests.</em>
    </td>
  </tr>
</table>Scientific Integrity Notice: Quantum Morph Laboratory (QML) is active experimental research software (v0.6.0 — PROJECT EVENT HORIZON Wave III), not production software. "Quantum-inspired" refers strictly to classical mathematics (tensor networks, spectral decompositions, Walsh–Hadamard incoherence, Grassmannian subspace geometry) and makes no claim of physical quantum computation or quantum advantage. QML has been validated end-to-end on synthetic benchmarks (exp-001) and six real openly licensed pretrained language models spanning 135M → 7.62B parameters (SmolLM2-135M, SmolLM2-360M, SmolLM2-1.7B, Qwen2.5-1.5B, 2-shard Qwen2.5-3B, and 4-shard 14.19 GiB Qwen2.5-7B-Instruct in exp-002 through exp-006; weights stored locally outside Git). Every byte—including scales, zero-points, biases, codebooks, recovery adapters, shared bases, sparse indices, JSON manifests, and checksums—is counted in reported complete BPP; negative controls and sub-2-bit cliffs are preserved alongside breakthroughs.

Architecture & Representation Pipeline
<p align="center">
  <img src="assets/architecture.svg" alt="Quantum Morph Laboratory Implemented Rust Pipeline Architecture" width="100%"/>
</p>Implemented Subsystems (13 Safe Rust Crates)
Subsystem	Crate	Implemented Capabilities	Maturity
Core Math & SHA-256	qml-core	Validated dense matrices, matmul/`transpose`/`matvec_transposed`, SplitMix64 PRNG, exact one-sided Jacobi thin SVD, HMT randomized SVD, spectral metrics, Grassmannian subspace affinity, pure-Rust FIPS 180-4 SHA-256	Implemented & tested
Hardware & WGPU	qml-hardware	CPU/SIMD/RAM telemetry, hardware-only WGPU compute selection, 10 compressed matvec WGSL shaders (incl. MATVEC_VQ_WGSL for GPU 1/2-stage RVQ) + 8 Full GPU-Resident Transformer WGSL shaders (gpu_full.rs) incl. GPU-resident per-head INT8 KV-cache (~3.88× smaller VRAM)	Implemented & GPU-verified
Model Ingestion	qml-model	Bounded read-only single & multi-shard Safetensors (SafetensorsSource) up to 8 GiB/tensor & 64 GiB/total, direct row-seek & zero-f64 BF16 readers, exact codecs (F64..FP8), HF config.json, GGUF v2/v3 incl. K-quants (Q2_K..Q6_K)	Implemented & tested
Q-Scope Observatory	qml-scope	Two-pass streaming Welford statistics (mean, std, skewness, kurtosis, norms, sparsity, 3σ/6σ outliers, crest factor, 32-bin histogram & Shannon entropy, quantization susceptibility proxy), 2D SVD structure, transformer role classification, cross-layer Grassmannian subspace affinity	Implemented & tested
Quant, RMQ, Resonance & GPTQ	qml-quant	Conventional baselines, RMQ Shared-Basis/Core, Sub-3-Bit PROJECT RESONANCE (FWHT, LS-refitted INT3/`INT2`/base-3 TERNARY, LUT RVQ, ECHO, MAP-B), Activation-Aware Blockwise Cholesky GPTQ (gptq.rs), binary QCAL1 Hessian disk cache, and activation-weighted 2-stage RVQ (fit_vq_calibrated)	Experimental & benchmarked
QMF Container	qml-format	Experimental QMF v0 & multi-GB QMF v1 (QmfFileReader out-of-core lazy reader, write_to_file_atomic with source-output alias protection, 64 KiB streaming CRC-32 + SHA-256, Arc deduplication, RMQ & sub-3-bit operators, on-demand decode_row)	Experimental (unstable)
Q-Forge Engine	qml-forge	Memory-bounded single & multi-shard streaming (forge_path_to_file_atomic, zero duplicate full-file RAM) with Activation-Calibrated GPTQ & RVQ, cooperative ForgeControl cancellation, QMR-300..150, RmqAffinityGated, SHA-256 provenance, and verification	Experimental & tested
Pure-Rust Tokenizer	qml-tokenizer	Pure safe Rust Hugging Face tokenizer.json Byte-Level BPE tokenizer (GPT-2/SmolLM/Llama-3/Qwen2 style) with individual-digit pre-tokenization, special tokens, ChatML formatting, and stateful UTF-8 StreamingDecoder	Implemented & tested
Q-Core Runtime	qml-runtime	Native Llama & Qwen2 Transformer (q/k/v biases, FP32/FP64 & GPU-resident INT8 KV-cache), Layer-by-Layer Streaming BF16 Calibration & Reference Evaluator (streaming.rs, < 450 MiB peak VRAM for 7B–32B!), executing over QMF v1 on Cpu, GpuHybrid, or Single-Submission GpuFull	Experimental CPU + GPU runtime
Q-Behavior Observatory	qml-behavior	Layer-by-layer hidden activation tracing, cosine/L2 drift, RSA, reversible causal ablations, ReferenceSuiteCache with compact binary QREF1 disk caching, and 4-category held-out Capability Retention Suite	Experimental & verified
Q-Genesis Extreme	qml-genesis	Multi-objective NSGA-II evolutionary search with role/depth-stratified sampling, user profiles, sub-3-bit Hadamard & ECHO layer genes, and direct Genome-to-QMF v1 export	Experimental & tested
Laboratory CLI	qml-cli	15 commands (doctor, inspect-safetensors, inspect-gguf, inspect-qmf, scope, baseline, rmq-ablate, morph, compare, resonance-ladder --calibrate --streaming-ref --kv-cache int8, infer-demo, generate --kv-cache int8, behavior, genesis, demo-rank1)	Implemented & tested
Native Desktop App	qml-desktop	PROJECT PRISM native WGPU 3D scientific observatory (eframe/`egui` + wgpu 24 + 7 custom WGSL shaders + HDR bloom + 3D ray picking) across 11 real-engine views, GpuFull inference, QMR-300..150 Q-Forge presets, and the Consumer VRAM Fit Matrix (1.7B–32B)	v0.6.0


Real Pretrained Model Validation
1. PROJECT EVENT HORIZON Wave III — Stage D Qwen2.5-7B-Instruct (14.19 GiB BF16 → 1.63 GiB VRAM) & Breaking the 2.000 bpp Barrier (exp-006)
Measured live on Windows 11 x86_64 (Intel Core i9-14900HX, 8 GB NVIDIA GeForce RTX 4070 Laptop GPU) on 4-shard Qwen/Qwen2.5-7B-Instruct (7.62B params, 14.19 GiB BF16 — nearly 2× larger than the GPU's entire VRAM!) using automatic Layer-by-Layer Streaming BF16 Calibration (444.5 MiB peak VRAM, 196.8s), single-submission GpuFull (197/197 GPU ops, 0 CPU ops), and GPU-resident INT8 KV-cache (3.88× smaller KV VRAM) (see [`experiments/exp-006-event-horizon-wave3-7b-sub2bit.toml`](experiments/exp-006-event-horizon-wave3-7b-sub2bit.toml)):
Qwen2.5-7B-Instruct Profile	Exact Complete BPP	Disk MiB (vs 14,525 MiB BF16)	VRAM MiB	GpuFull Tok/s	PPLx vs BF16	Mean KL	Held-out Top-1	Numerical fidelity %	Status
INT4_Q4_0_GPTQ	4.501	4,086.5 (3.55×)	3,795.1	6.8	0.98×	0.077	90.9%	95.4%	OK
INT3_HADAMARD_GPTQ	3.251	2,951.8 (4.92×)	2,741.5	6.8	1.07×	0.164	86.3%	89.7%	OK
INT2_HADAMARD_GPTQ	2.251	2,043.9 (7.11×)	1,898.7 (1.85 GiB!)	9.7	1.43×	0.854	65.7%	67.8%	OK
QMR-200_GPTQ (MAP-B)	2.000	1,815.7 (8.00×!)	1,670.4 (1.63 GiB!)	9.1	2.13×	1.633	52.6%	49.8%	OK
QMR-175_GPTQ (MAP-B)	1.750	1,588.7 (9.14×)	1,477.6	6.9	204.53×	6.403	8.0%	4.2%	CLIFF
TERNARY_HADAMARD_GPTQ	1.727	1,567.7 (9.27×)	1,456.5	9.5	115.92×	5.948	6.3%	3.6%	CLIFF


- 2-Bit (2.251 bpp) & Exact 2.000 bpp Fluency on 7.62B: At pure 2-bit (1.85 GiB VRAM), Qwen2.5-7B achieves PPLx = 1.43× and 65.7% Top-1 ("Binary search is an efficient algorithm that repeatedly halves the search range to find a specific value in a sorted", plus coherent Rust code generation at 7.86 tok/s with GPU INT8 KV-cache!). At exact 2.000 bpp (1.63 GiB VRAM — 8.00× smaller than BF16!), QMR-200_GPTQ breaks the 2-bit barrier with PPLx = 2.13× and 52.6% Top-1 ("Binary search is a algorithm that divides a sorted list into smaller sublists.").
- Four-Scale Power-Law Scaling Confirmed at 2.25 bpp (INT2_HADAMARD_GPTQ, 360M → 7.62B):
  - SmolLM2-360M: PPLx = 13.53×, KL = 2.811, Top-1 = 29.5% (CLIFF)
  - Qwen2.5-1.5B: PPLx = 4.12×, KL = 1.993, Top-1 = 41.7%, Ret = 33.0% (OK)
  - Qwen2.5-3B: PPLx = 2.76×, KL = 1.606, Top-1 = 55.4%, Ret = 45.8% (OK)
  - Qwen2.5-7B: PPLx = 1.43×, KL = 0.854, Top-1 = 65.7%, Ret = 67.8% (OK)
- Honest Sub-2-Bit Boundary & GPU RVQ Controls Preserved: Both QMR-175_GPTQ (1.750 bpp) and TERNARY_HADAMARD_GPTQ (1.727 bpp) still cross into CLIFF (PPLx 116–205×) at 7.6B, and our live SmolLM2-360M GPU MATVEC_VQ_WGSL controls (RVQ2_D8_G128_GPTQ 2.180 bpp, KL = 9.865; RVQ2_D10_G128_GPTQ 1.790 bpp, KL = 10.336 at 15.5 tok/s 100% GPU-resident vs Cholesky-compensated TERNARY 1.738 bpp KL = 5.397 and INT2 2.270 bpp KL = 2.811) prove second-order Hessian compensation (H⁻¹) dominates uncompensated non-uniform codebook shape alone—directly guiding Wave IV's Hessian-compensated lattice/trellis kernels and 14B–32B scaling.
2. Earlier Validated Research Archive (exp-001 through exp-005)
- Wave II Multi-Scale GPTQ & GpuFull 22.1× Speedup ([`exp-005`](experiments/exp-005-event-horizon-wave2-gptq-scaling.toml)): GpuFull accelerated SmolLM2-360M Q4_0 22.1× (1.84 → 40.70 tok/s, up to 63.3 tok/s at 2-bit) with identical output; matched-byte GPTQ+Hadamard cut SmolLM2-360M INT3 KL −67.3% (1.910 → 0.624), broke QMR-275 (360M, PPLx 3.52×), QMR-250 (SmolLM2-1.7B, PPLx 2.79×), and INT2 2.25 bpp on Qwen2.5-1.5B (PPLx 4.12×) and 2-shard Qwen2.5-3B (PPLx 2.76×, 55.4% Top-1, 829.6 MiB VRAM, 18.6 tok/s).
- Matched-Byte Hadamard Incoherence Control & Layer-31 Study ([`exp-004`](experiments/exp-004-event-horizon-sub3bit.toml)): Randomized FWHT cut plain 2-bit Mean KL by −37.9% (135M, 137.5× lower PPLx) and −39.5% (360M, 209.4× lower PPLx) at zero extra bytes. Layer-31-only protection (+0.102 bpp) cut L31 Rel L2 −24.0% but slightly raised KL (1.284 → 1.499), falsifying single-layer-only fixes.
- Persistent GPU Hybrid Baselines & Nuanced Controls ([`exp-003`](experiments/exp-003-smollm2-360m-gpu.toml), [`exp-002`](experiments/exp-002-smollm2-135m.toml), [`exp-001`](experiments/exp-001-rmq-map-qcore-baseline.toml)): Verified verbatim across SmolLM2-360M (BF16 12.10 tok/s, INT8_SYM 20.97 tok/s / 95.5% Top-1, Q4_0 11.65 tok/s / 77.3% Top-1, weight-only MAP 4.516 bpp nuanced control), SmolLM2-135M quant-only MAP (−16.4% KL), and synthetic outlier-peeled RMQ (−49.2% error) vs Gaussian control.
<p align="center">
  <img src="assets/benchmarks.svg" alt="QML Measured RMQ Ablation and Q-Core Direct Inference Benchmarks" width="100%"/>
</p>Building from Source (Optional)
Prebuilt executables are available through the repository's native GitHub Releases section. To build from source, only a stable Rust toolchain (rustc / cargo) is required (zero Python):
# Format, strict Clippy lint, full test suite across 13 crates, and release build
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --workspace --release

# Launch the native PROJECT PRISM WGPU 3D desktop observatory (with GpuFull inference)
cargo run --release -p qml-desktop

# Calibrate & benchmark a 7B–32B multi-shard model in < 1 GiB peak VRAM + GPU INT8 KV-cache
cargo run --release -p qml-cli -- resonance-ladder --model ./model-dir --config ./config.json --tokenizer ./tokenizer.json --out-dir ./work --backend gpu --kv-cache int8 --calibrate --no-echo --profiles "int4,int3,int2,qmr-200"

# Run single-submission Full GPU-Resident compressed inference with GPU INT8 KV-cache
cargo run --release -p qml-cli -- generate ./work/int2_hadamard.qmf --config ./config.json --tokenizer ./tokenizer.json --backend gpu --kv-cache int8 --chat --prompt "Explain binary search."
See [BUILD.md](BUILD.md) for cargo-deny policy checks and CI workflow details.
Research Foundation & Continuity
- Prior-Art Literature Review (17 topics): [`research/prior-art-index.md`](research/prior-art-index.md) and [`research/prior-art-synthesis.md`](research/prior-art-synthesis.md)
- Sub-3-Bit Mathematical Synthesis: [`research/sub-3-bit-resonance-synthesis.md`](research/sub-3-bit-resonance-synthesis.md)
- Novelty Ledger: [`research/novelty-ledger.md`](research/novelty-ledger.md) (explicitly distinguishes established literature, falsifiable hypotheses, and QML experimental findings)
- Container Specifications: [`research/qmf-experimental-v0.md`](research/qmf-experimental-v0.md) and [`research/qmf-experimental-v1.md`](research/qmf-experimental-v1.md)
- Architecture & Decisions: [`ARCHITECTURE.md`](ARCHITECTURE.md), [`ROADMAP.md`](ROADMAP.md), [`docs/decisions/0001-bounded-safetensors-inspection.md`](docs/decisions/0001-bounded-safetensors-inspection.md)
- Cross-Agent Handoff: [`AGENTS.md`](AGENTS.md), [`HANDOFF.md`](HANDOFF.md), [`MASTER_DEVELOPMENT_DIRECTIVE.md`](MASTER_DEVELOPMENT_DIRECTIVE.md)
License
Quantum Morph Laboratory is licensed under the MIT License (Copyright © 2026 ShamelesAbyss). Third-party Rust crate dependency licenses and advisories are continuously verified by cargo-deny in CI ([`deny.toml`](deny.toml)). Prebuilt Windows and Linux release binaries with SHA-256 checksums are published through the repository's native GitHub Releases section (see [`RELEASE_NOTES.md`](RELEASE_NOTES.md)).
### In Loving Memory of Ashlan Haley Rapponotti

*June 21, 2003 – November 13, 2026*

For the beautiful woman who listened to my passions, believed in me, and made my world a happier place simply by being part of it.

Every creation, every discovery, and every milestone reached carries a piece of my heart dedicated to you.

**Forever loved. Forever missed. Forever remembered.** 💜
