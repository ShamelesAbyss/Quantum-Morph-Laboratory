# Roadmap

Status key: **Done** means implemented, tested, and verified on `main`; **Experimental** means working research implementation with reproducible benchmarks; **Next** is planned future work. The complete master directive is preserved in [`MASTER_DEVELOPMENT_DIRECTIVE.md`](MASTER_DEVELOPMENT_DIRECTIVE.md).

## Completed Foundations & Milestones (`v0.1.0-alpha.1` → `v0.6.0`)

- **Done — Foundation & MIT License:** Public repository on `main`, Rust 2024 workspace (13 safe-Rust crates), `#![forbid(unsafe_code)]`, zero Python dependencies, Windows + Ubuntu CI, `cargo-deny` zero-ignored-advisories policy, MIT License (`LICENSE`), automated public GitHub Releases with SHA-256 checksums, and 17-topic source-backed prior-art review.
- **Done — Milestones I–IV (Q-Scope, Baselines, RMQ, MAP, Out-of-Core QMF v1 & Atomic Q-Forge):** Streaming Safetensors & GGUF v2/v3 readers, exact bit-level codecs, one-sided Jacobi & HMT SVD, Grassmannian cross-layer affinity, conventional baselines, outlier-peeled RMQ Shared-Basis/Core, affinity-gated RMQ, constant-memory two-pass MAP, out-of-core `QmfFileReader`, and atomic Q-Forge with cooperative cancellation.
- **Done — PROJECT ACCELERATION, Q-Behavior & PROJECT PRISM (`v0.2.0-alpha.1` → `v0.3.0`):** Persistent GPU-resident compressed inference (`GpuHybrid`), `qml-behavior` mechanistic tracing/RSA/reversible causal ablations, stratified `qml-genesis` with QMF export, and the **PROJECT PRISM** native multi-pass WGPU 3D scientific observatory (`7` WGSL shaders, `Rgba16Float` HDR bloom, interactive 3D picking across 7 workspaces, `157–187 FPS`).
- **Done (Experimental) — PROJECT RESONANCE & EVENT HORIZON Wave I (`v0.4.0`):** Complete-artifact Sub-3-Bit Ladder (`QMR-300..150`), blockwise Randomized FWHT incoherence (`−37.9%` / `−39.5%` KL, `137.5×` / `209.4×` lower PPLx at zero extra bytes), LS-refitted `INT3`/`INT2`/base-3 `TERNARY`, LUT RVQ, PROJECT ECHO, variance-additive `MAP-B`, 4-category Capability Retention Suite, 9 persistent GPU WGSL matvec kernels, Consumer VRAM Fit Matrix, and `exp-004` Stage A benchmarks.
- **Done (Experimental) — PROJECT EVENT HORIZON Wave II (`v0.5.0`–`v0.5.1`):** Activation-Aware Blockwise Cholesky GPTQ + Hadamard (`qml-quant::gptq`), single-submission `QCoreBackend::GpuFull` (`22.1×` speedup, `40.7–63.3 tok/s`), Qwen2 biases, multi-shard `SafetensorsSource`, GGUF K-quants (`Q2_K..Q6_K`), `ReferenceSuiteCache`, hardware-only WGPU compute selection, and `exp-005` Stage A/B/C scaling (`360M → 1.5B → 1.7B → 2-shard 3B`).
- **Done (Experimental) — PROJECT EVENT HORIZON Wave III (`v0.6.0`):**
  - **Layer-by-Layer Streaming BF16 Calibration & Reference Engine (`qml-runtime::streaming`):** Calibrates and reference-evaluates `14.19 GiB` 4-shard `Qwen2.5-7B-Instruct` in **`444.5 MiB` peak VRAM** (`196.8s`) with compact binary `QCAL1` & `QREF1` disk caching (`PayloadBudget` up to `8 GiB/tensor`, `64 GiB/total`).
  - **GPU-Resident Per-Head INT8 KV-Cache (`gpu_full.rs`):** `ROPE_AND_KV_APPEND_INT8_WGSL` + `GQA_ATTN_INT8_WGSL` delivering `~3.88×` smaller KV VRAM natively in `GpuFull` (`--kv-cache int8`).
  - **10th Persistent GPU WGSL Kernel (`MATVEC_VQ_WGSL`) & `fit_vq_calibrated`:** Direct GPU execution for all 10 compressed formats (`< 1e-4` CPU/GPU parity) + honest uncompensated RVQ negative control (`exp-006`).
  - **Stage D `Qwen2.5-7B-Instruct` (`7.62B`, `14.19 GiB` BF16) Live 8 GB GPU Breakthrough (`exp-006`):** `INT4` (`PPLx 0.98×`, `90.9%` Top-1, `3.71 GiB`), `INT3` (`PPLx 1.07×`, `86.3%` Top-1, `2.68 GiB`), **`INT2` (`2.251 bpp`, `PPLx 1.43×`, `65.7%` Top-1, `1.85 GiB` VRAM, `9.7 tok/s`)**, and **`QMR-200_GPTQ` (`2.000` exact complete bpp, `PPLx 2.13×`, `52.6%` Top-1, `1.63 GiB` VRAM — `8.00×` smaller than BF16, `9.1 tok/s`, breaking the 2.000 bpp barrier!)**.

## PROJECT AEGIS (v0.6.1)

Scoped artifact publication, required output projection correctness, bounded provenance-checked caches and accurate numerical fidelity terminology. Later milestones require separate authorization; no Wave IV work is included.

## PROJECT ATLAS (v0.6.2)

Disk-backed QMF v1 construction and Q-Forge integration with independent resource limits, verified AEGIS-safe publication and synthetic validation only. This establishes construction foundations; 14B/27B/32B experiments, sampled MAP-B, persistent/resumable jobs, new quantizers and architecture support remain separate unauthorized assignments.

## Next Milestones (PROJECT EVENT HORIZON Wave IV & Beyond)

The owner subsequently authorized PROJECT COLOSSUS v0.6.5 using exclusively
Qwen2.5-Coder-14B-Instruct, superseding the earlier 22B–24B selection. This one
experiment reuses existing algorithms and records objective coding-task
correctness separately from numerical fidelity. Only throughput reporting may
change in the application. All experiment data stays under D:\QML-COLOSSUS;
no additional model, new compression algorithm, architecture implementation or
graphical change is authorized. See experiment 007 and docs/PROJECT_COLOSSUS.md.
Stop after its validated, independently verified native public release.
Future items below remain proposals rather than authorization.

PROJECT SENTINEL v0.6.4 is strictly limited to runtime configuration admission, optional attention-bias correctness, synthetic validation and a verified release. The owner requires a stop after SENTINEL. No next milestone, real-model download, architecture implementation or compression/graphics work is authorized by this assignment. Its exact admission subset is documented in docs/RUNTIME_ADMISSION.md; independent real-model inference comparison remains outside this milestone.

CONTINUUM supersedes earlier authorization limits for continued correctness and scientific development, but does not validate the large-model projections below. The v0.6.3 restoration checkpoint establishes permanent distribution compliance and verifiable release infrastructure. The next correctness gates are strict supported-configuration admission, malformed optional-tensor errors and provenance-pinned independent small-model reference validation; see docs/CAPABILITIES.md. Large-model fit and performance remain unverified hypotheses.

- **Next — Stage E (`14B`) & Stage F (`27B–32B`) on 8 GB Consumer GPUs:** Add sampled-row fast `MAP-B` Pass 1 screening (10× faster on `7B–32B`), resumable layer-checkpointed disk streaming in Q-Forge, and benchmark `14B` (`~3.4–4.2 GiB` at `2.00–2.25 bpp`) and `27B–32B` (`~6.8–7.6 GiB` at `1.75–2.00 bpp`) live on an 8 GB RTX 4070!
- **Next — Breaking `< 2.00 bpp` (`1.75–1.50 bpp`) via Hessian-Compensated Lattice/Trellis Quantization:** Combine Wave III's `MATVEC_VQ_WGSL` + `H⁻¹` Cholesky error compensation (QuIP#/QTIP-style E8/trellis + block-128 GPTQ + PV-Tuning coordinate refinement) to push `7B–32B` below `1.75 bpp` cleanly.
- **Next — Extended Architectures & Hierarchical Activation-Calibrated Q-Genesis:** Add Mistral/Gemma/MoE routing, connect `qml-genesis` to cached `QCAL1`/`QREF1` evaluation, and expand long-context & coding/math capability suites.
