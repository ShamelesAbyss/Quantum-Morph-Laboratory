# Handoff

## Current state

- Public repository: <https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory>
- Development branch: `main` only.
- Local workspace: `C:\Users\Shame\.gemini\antigravity\scratch\Quantum-Morph-Laboratory` (Windows 11, Intel Core i9-14900HX, RTX 4070 Laptop GPU 8 GB VRAM, 32 GB RAM, 2560×1440 144 Hz display).
- Local pretrained model cache (NEVER commit to Git): `C:\Users\Shame\.gemini\antigravity\scratch\qml-models-cache` (`smollm2-135m` and `smollm2-360m`).
- Latest completed milestone: **PROJECT PRISM (`v0.3.0`) — Public Native Multi-Pass WGPU 3D Scientific Observatory & Corrected Public Release Distribution**.
- **Permanent Release Distribution Policy:** All user-facing releases use clean semantic versions (`v0.3.0`, `v0.3.1`, `v0.4.0`, ...) and MUST be published as full, visible public GitHub Releases (`draft: false`, `prerelease: false`, `make_latest: true`) with all 4 compiled assets (`Quantum-Morph-Laboratory-<tag>-Windows-x64.exe`, `Quantum-Morph-Laboratory-<tag>-Windows-x64-Portable.zip`, `Quantum-Morph-Laboratory-<tag>-Linux-x64.tar.gz`, `SHA256SUMS.txt`), full `RELEASE_NOTES.md`, and prominent `README.md` download links. Prereleases are strictly opt-in when explicitly requested by the repository owner. A Git tag, CI build, Actions artifact, or source archive alone is NEVER a completed release.
- The workspace contains **13 cohesive safe-Rust crates** (`qml-core`, `qml-hardware`, `qml-model`, `qml-scope`, `qml-quant`, `qml-format`, `qml-forge`, `qml-tokenizer`, `qml-runtime`, `qml-behavior`, `qml-genesis`, `qml-cli`, `qml-desktop`) with `#![forbid(unsafe_code)]` and zero Python dependencies.
- **License:** MIT ([LICENSE](LICENSE)). All workspace crates inherit `license.workspace = true`.
- **Permanent Memorial Dedication:** The memorial dedication to Ashlan Haley Rapponotti at the end of `README.md` MUST be preserved verbatim across all future updates and releases.

## Implemented capabilities

- `qml-core`: Dense linear algebra, SplitMix64 PRNG, deterministic one-sided Jacobi exact thin SVD, HMT randomized SVD, spectral metrics, Grassmannian subspace affinity, and pure safe-Rust FIPS 180-4 SHA-256.
- `qml-hardware`: Cross-platform CPU/SIMD/RAM telemetry, WGPU GPU adapter discovery, enforced `ResourcePreset` budgets, and 6 persistent VRAM WGSL compute kernels (`FP32`, `FP16`, `BF16`, `INT8_SYM`, `Q4_0`, `Q4_1`) with reusable zero-allocation scratch buffers (`GPU_SCRATCH_MAX_ELEMS = 131,072`).
- `qml-model`: Bounded read-only Safetensors & GGUF v2/v3 streaming payload readers (`PayloadBudget` up to 16 GiB), exact safe-Rust codecs (`F64`, `F32`, `F16`, `BF16`, `F8_E4M3`, `F8_E5M2`, `Q8_0`, `Q4_0`, `Q4_1`, ints, bool), sharded `index.json` & HF `config.json` parsers.
- `qml-scope`: Two-pass streaming Welford statistics, sparsity, 3σ/6σ outliers, crest factor, 32-bin histogram & Shannon entropy, quantization susceptibility proxy, 2D SVD spectrum, transformer role classification, and pairwise cross-layer Grassmannian subspace affinity.
- `qml-quant`: Rayon-parallelized conventional baselines (`FP16`, `BF16`, `INT8_SYM`, `INT8_ASYM`, `Q4_0`, `Q4_1`, `LowRankSvd`, `SparseResidual`) with exact `StorageBreakdown`; RMQ Shared-Basis (`U·C_l·Vᵀ + R_l` via 2D-HOOI + RPCA-inspired outlier peeling); RMQ Shared-Core (`A_l·S·B_l + R_l`); two-pass streaming MAP allocator.
- `qml-format`: Experimental QMF v0 & multi-GB QMF v1 with out-of-core lazy `QmfFileReader`, atomic publication (`write_to_file_atomic`) with source-output alias protection, 64 KiB streaming CRC-32 + SHA-256 verification, `Arc` shared-latent deduplication, `RmqSharedCore`, and on-demand `decode_row`.
- `qml-forge`: Streaming tensor-by-tensor & cluster-by-cluster Safetensors → QMF v1 pipeline with cooperative `ForgeControl` cancellation, live progress callbacks, `RmqAffinityGated` hybrid routing, atomic publication, and direct round-trip matvec verification.
- `qml-tokenizer`: Pure safe-Rust Hugging Face `tokenizer.json` Byte-Level BPE tokenizer with individual-digit pre-tokenization, ChatML formatting, and stateful UTF-8 `StreamingDecoder`.
- `qml-runtime`: Native Llama-style Transformer executing directly over QMF v1 compressed operators on CPU (Rayon) or Persistent GPU Hybrid (`QCoreBackend::GpuHybrid`), plus `forward_token_with_trace` and strictly runtime-only reversible `LayerIntervention` hooks.
- `qml-behavior`: Layer-by-layer hidden activation tracing (`compare_model_behavior`), cosine similarity, relative L2 divergence, Representational Similarity Analysis (RSA), peak drift layer detection, and reversible causal ablation studies (`run_intervention_study`).
- `qml-genesis`: Role- and depth-stratified multi-objective NSGA-II evolutionary search (`select_stratified_tensor_names`), user profiles (`Balanced`, `Smallest`, `BestQuality`, `Fastest`), persistent JSON archives, and direct Genome-to-QMF v1 export (`export_genome_to_qmf_v1`).
- `qml-cli`: 14 laboratory commands (`doctor`, `inspect-safetensors`, `inspect-gguf`, `inspect-qmf`, `scope`, `baseline`, `rmq-ablate`, `morph`, `compare`, `infer-demo`, `generate`, `behavior`, `genesis`, `demo-rank1`).
- `qml-desktop`: **PROJECT PRISM** native WGPU 3D scientific observatory (`eframe` WGPU + `wgpu 24`, 7 custom WGSL shaders, `Rgba16Float` HDR selective bloom, ACES filmic tone mapping, interactive Orbit/Ortho cameras & 3D ray picking, automatic 2.5D CPU fallback, 5 persisted graphics presets, 11 custom procedural vector subsystem icons, collapsible navigation rail, Q-Core Zen Mode, and 11 real-engine views).

## Validation actually executed on `v0.3.0-alpha.1`

All of the following passed locally on Windows 11 with Rust `1.100.0` (`cargo +1.100.0`) and `cargo-deny 0.21.0`:

- `cargo +1.100.0 fmt --all -- --check`
- `cargo +1.100.0 clippy --workspace --all-targets -- -D warnings`
- `cargo +1.100.0 test --workspace` — all **81 automated tests** across **13 crates** passed (including live WGPU shader & multi-pass pipeline validation on NVIDIA RTX 4070).
- `cargo +1.100.0 build --workspace --release`
- `cargo deny check` — `advisories ok, bans ok, licenses ok, sources ok` (zero ignored vulnerabilities).
- **Live Native GPU Visual Audit:** Launched `target/release/qml-desktop.exe` across 6 workspaces (`2560×1440` Command Center + `1920×1080` real `SmolLM2-135M` Q-Scope 3D Anatomy & Q-Entangle Constellation + live Q-Genesis 3D Crucible + Q-Forge Morphogenesis + Hardware Observatory), verified `157–187 FPS` on RTX 4070, visually inspected every captured PNG in `assets/screenshots/`, and confirmed zero overlaps, zero broken glyphs, and clean scientific provenance labeling.

## Known limits and blockers

- QMF v0/v1 remain experimental research formats and are not yet compatibility-stable across major milestones.
- `QCoreBackend::GpuHybrid` keeps persistent compressed linear projections on GPU while running lightweight RMSNorm, RoPE, and KV-cache attention on CPU; full GPU-resident attention & KV-cache kernels remain future optimization work.
- Safe cross-platform WGPU intentionally does not expose GPU temperature, board power draw, or OS-wide VRAM usage; `qml-desktop` explicitly notes this and never fabricates telemetry.

## Next steps

1. **Full GPU-Resident Attention & KV-Cache:** Eliminate per-token CPU↔GPU synchronization in `qml-runtime` for interactive 1B–7B model throughput.
2. **Large-Scale (1B–3B) Perplexity & Mechanistic Studies:** Evaluate affinity-gated RMQ, Q-Genesis exported QMF v1 models, and Q-Behavior layer-drift interventions (e.g., protecting Layer 31 precision) on 1B–3B pretrained transformers.
3. **Extended Model Architectures:** Support Qwen2/Mistral attention biases and Mixture-of-Experts routing inside `QCoreModel` and `qml-scope`.
