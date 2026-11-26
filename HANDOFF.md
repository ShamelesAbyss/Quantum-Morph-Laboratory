# Handoff

## PROJECT AEGIS — v0.6.1 (local validation complete; CI/release pending)

- Scope: artifact publication, required untied output projections, calibration/reference cache safety, terminology and release only. Do not start Wave IV or any subsequent milestone without authorization.
- Work performed on `main` in the Codex workspace; no downloaded models or new live-model experiments. The original v0.6.0 checkout was clean.
- `qml-core::publication` uses `tempfile` exclusive randomized sibling staging, `same-file` filesystem identities, destination symbolic-link rejection, and explicit overwrite authorization. Default publication is no-clobber. Authorized persistence never deletes the destination first; failures preserve prior bytes. Q-Forge checks every backing shard against the destination.
- `qml-format` exposes `write_to_file_atomic_with_overwrite`; its existing API defaults to refusing replacement. File-size and manifest validation remain in force. Original weights remain read-only.
- `qml-runtime` propagates required `lm_head.weight` loading errors in memory and lazy file-backed loading. Streaming uses embedding weights only for explicitly tied models and checks projection dimensions and finite BF16 values.
- `qml-core::cache` supplies versioned SHA-256 envelopes and bounded binary access. QCAL2/QREF2 validate dimensions, counts, checked byte lengths, finite f32/f64 values, duplicate calibration tensors, exact reference sequence/vocabulary dimensions and trailing bytes. Disk cache size is limited to 512 MiB.
- CLI cache identity hashes all backing weights, config, tokenizer, optional reference QMF, calibration/evaluation tokens, backend and reference/calibration modes. QREF2 additionally binds labels/categories/tokens. Existing QCAL1/QREF1 caches are unverified and rejected without modification: regenerate with a new output directory. New cache publication refuses overwrite and reports errors.
- Composite score renamed to `composite_numerical_fidelity_pct`; historical JSON's old field name remains a deserialization alias. Formula and measurements are preserved. README/CLI/release notes explain that it does not establish retained intelligence or benchmark capability.
- Updated workspace version, lockfile, README download links, release notes, architecture, roadmap and changelog. MIT license, history, Prism renderer and permanent memorial dedication are preserved.
- Added regression tests for failed/no-clobber publication, hard links, distinct staging names, authorized replacement, Unix symbolic/dangling links, required projection failures in all three execution paths, corrupted lazy output tensors, non-finite streaming projections, cache round trips, provenance/dataset mismatches, oversized lengths, non-finite values and malformed dimensions. Existing tests were retained.

### Validation state

- Rust 1.100.0 selected with a directory-local rustup override; cargo-deny 0.21.0.
- `cargo fmt --all -- --check`: passed.
- `cargo clippy --workspace --all-targets -- -D warnings`: passed on final v0.6.1 source.
- `cargo test --workspace`: all 95 tests passed on Windows, zero failed/ignored; all doc-tests passed. This includes the final projection/cache/publication regressions and existing desktop/GPU parity tests. Unix-only symbolic-link coverage will run in Linux CI.
- `cargo deny check`: passed on v0.6.1 (`advisories ok, bans ok, licenses ok, sources ok`); existing duplicate-version warnings remain allowed by the unchanged policy.
- `cargo build --workspace --release`: passed (optimized workspace build, 5m 05s). Release CLI smoke check pending below. All five requested commands passed locally. Windows/Linux CI, tag and public release verification remain pending.

### Limits and remaining work

Cache checksums detect corruption rather than authenticate a malicious producer. Hashing source provenance reads every backing weight byte once. Decoded f64 values and collection overhead require additional bounded RAM beyond the 512 MiB file ceiling. Path-based publication assumes output ancestors are user-controlled, not concurrently replaced by a hostile process. No claim of new live-model benchmark validation is made. Baseline commit: `1631b824ce9daa8f256d46c45497f1e804442dd8`. Next: push this validated milestone, verify Windows/Linux CI, publish all four mandatory assets as a latest public non-prerelease, verify download hashes, update this handoff and stop.

## Current state

- Public repository: <https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory>
- Development branch: `main` only.
- Local workspace: `C:\Users\Shame\.gemini\antigravity\scratch\Quantum-Morph-Laboratory` (Windows 11, Intel Core i9-14900HX, RTX 4070 Laptop GPU 8 GB VRAM, 32 GB RAM, 2560×1440 144 Hz display).
- Local pretrained model caches (NEVER commit to Git):
  - `C:\Users\Shame\.gemini\antigravity\scratch\qml-models-cache` (`smollm2-135m`, `smollm2-360m`, `smollm2-1.7b`, `qwen2.5-1.5b`, `qwen2.5-3b`)
  - `D:\qml-models-cache\qwen2.5-7b` (`Qwen/Qwen2.5-7B-Instruct`, 4 shards, `14.19 GiB` BF16)
  - `D:\qml-work\qwen25-7b` (cached `_cal_profile.qcal`, `_ref_logits.qref`, and forged `int4/int3/int2/qmr-200/qmr-175/ternary` QMF v1 artifacts)
- Latest completed milestone: **`v0.6.0` — PROJECT EVENT HORIZON Wave III** (Layer-by-Layer Streaming BF16 Calibration & Reference Engine `< 450 MiB` peak VRAM for `7B–32B`, GPU-Resident Per-Head INT8 KV-Cache `~3.88×` smaller VRAM, 10th Persistent GPU Kernel `MATVEC_VQ_WGSL`, `QCAL1`/`QREF1` binary disk caches, activation-weighted `fit_vq_calibrated`, and `exp-006` Live Stage D **`Qwen2.5-7B-Instruct`** 8 GB GPU Breakthrough breaking **`2.251 bpp` (`1.85 GiB`, `PPLx = 1.43×`, `65.7%` Top-1)** and **exact `2.000 bpp` (`1.63 GiB`, `PPLx = 2.13×`, `52.6%` Top-1, `8.00×` smaller than BF16)**!).
- **Permanent Release Distribution Policy:** All user-facing releases use clean semantic versions (`v0.3.0`, `v0.4.0`, `v0.5.1`, `v0.6.0`, ...) and MUST be published as full, visible public GitHub Releases (`draft: false`, `prerelease: false`, `make_latest: true`) with all 4 compiled assets (`Quantum-Morph-Laboratory-<tag>-Windows-x64.exe`, `Quantum-Morph-Laboratory-<tag>-Windows-x64-Portable.zip`, `Quantum-Morph-Laboratory-<tag>-Linux-x64.tar.gz`, `SHA256SUMS.txt`), full `RELEASE_NOTES.md`, and prominent `README.md` download links. Never delete or overwrite historical tags/releases.
- The workspace contains **13 cohesive safe-Rust crates** with `#![forbid(unsafe_code)]` and zero Python dependencies.
- **License:** MIT ([LICENSE](LICENSE)). All workspace crates inherit `license.workspace = true`.
- **Permanent Memorial Dedication:** The memorial dedication to Ashlan Haley Rapponotti at the end of `README.md` (and `RELEASE_NOTES.md`) MUST be preserved verbatim across all future updates and releases.

## Implemented capabilities

- `qml-core`: Dense linear algebra, SplitMix64 PRNG, deterministic one-sided Jacobi exact thin SVD, HMT randomized SVD, spectral metrics, Grassmannian subspace affinity, and pure safe-Rust FIPS 180-4 SHA-256.
- `qml-hardware`: Cross-platform CPU/SIMD/RAM telemetry, hardware-only WGPU compute selection (skips `DeviceType::Cpu` software emulators), **10 persistent VRAM WGSL matrix kernels** (`FP32`, `FP16`, `BF16`, `INT8_SYM`, `Q4_0`, `Q4_1`, `INT3_ASYM`, `INT2_ASYM`, `TERNARY`, **`VECTOR_QUANT`**), **8 full-transformer GPU-resident WGSL kernels** (`RMSNORM`, uniform-barrier `HADAMARD_FWD`, `ROPE_AND_KV_APPEND` FP32 & INT8, `GQA_ATTN` FP32 & INT8, `SWIGLU`, `VEC_ADD`) in `gpu_full.rs`, and `GPU_SCRATCH_MAX_ELEMS = 262_144`.
- `qml-model`: Unified single + multi-shard `SafetensorsSource` (`PayloadBudget` up to `8 GiB/tensor`, `64 GiB/total`, direct row-seek `read_tensor_rows_f64`, zero-`f64` `read_tensor_bf16_bits`) and GGUF v2/v3 readers with exact safe-Rust codecs (`F64..FP8`, `Q8_0`, `Q4_0`, `Q4_1`, `Q2_K..Q6_K`) and HF `config.json` parser.
- `qml-scope`: Two-pass streaming Welford statistics, outliers, histograms, SVD spectra, role classification, and cross-layer Grassmannian affinity.
- `qml-quant`: Conventional baselines, RMQ Shared-Basis/Core, `MAP`/`MAP-B`, PROJECT RESONANCE (`HadamardSpec`, `Int3`/`Int2`/`Ternary`/`VectorQuant`/`Echo`), Blockwise Cholesky GPTQ (`gptq.rs`), `fit_vq_calibrated`, and compact binary `QCAL1` Hessian disk persistence.
- `qml-format`: Experimental QMF v0 & multi-GB QMF v1 (`QmfFileReader`, `write_to_file_atomic`, streaming CRC-32 + SHA-256, `Arc` shared-latent deduplication, sub-3-bit operators, `decode_row`).
- `qml-forge`: Streaming single & multi-shard Safetensors → QMF v1 pipeline (`forge_path_to_file_atomic`) with calibrated GPTQ & RVQ, cancellation, and verification.
- `qml-tokenizer`: Pure safe-Rust Hugging Face `tokenizer.json` Byte-Level BPE tokenizer with ChatML formatting and stateful UTF-8 `StreamingDecoder`.
- `qml-runtime`: Native Llama & Qwen2 Transformer (`q/k/v` biases, FP32/FP64 & GPU-resident INT8 KV-cache, `Cpu`, `GpuHybrid`, single-submission `GpuFull`), plus **`streaming.rs`** Layer-by-Layer Streaming BF16 Calibration & Reference Evaluator (`< 450 MiB` peak VRAM on 7B).
- `qml-behavior`: Layer-by-layer hidden tracing, RSA, reversible causal ablations, `ReferenceSuiteCache` with binary `QREF1` disk caching, and 4-category held-out Capability Retention Suite.
- `qml-genesis`: Q-Genesis Extreme role/depth-stratified multi-objective NSGA-II search + Genome-to-QMF v1 export.
- `qml-cli`: 15 laboratory commands supporting sharded Safetensors, `--streaming-ref`, `--kv-cache int8`, `--calibrate`, `rvq-200/175/150`, and `GpuFull`.
- `qml-desktop`: PROJECT PRISM native WGPU 3D scientific observatory (`eframe` WGPU + `wgpu 24`, 7 custom WGSL shaders, `Rgba16Float` HDR bloom, 11 real-engine views, `GpuFull` default GPU inference, Consumer VRAM Fit Matrix).

## Validation actually executed on `v0.6.0`

All of the following passed locally on Windows 11 with Rust `1.100.0` (`cargo +1.100.0`) and `cargo-deny 0.21.0`:

- `cargo +1.100.0 fmt --all -- --check`
- `cargo +1.100.0 clippy --workspace --all-targets -- -D warnings`
- `cargo +1.100.0 test --workspace` — all **88 automated tests** across **13 crates** passed (including all 18 WGSL GPU compute shaders, `MATVEC_VQ_WGSL` parity, GPU-resident INT8 KV-cache parity + VRAM reduction, Layer-by-Layer Streaming BF16 calibration parity, Cholesky GPTQ + Hadamard rotation proofs, GGUF K-quants, and PRISM WGPU rendering pipelines on NVIDIA RTX 4070).
- `cargo +1.100.0 build --release -p qml-desktop -p qml-cli` + smoke-test launch of `qml.exe --version` (`qml 0.6.0`) and `qml-desktop.exe`.
- `cargo deny check` — `advisories ok, bans ok, licenses ok, sources ok` (zero ignored vulnerabilities).
- **Live Real-Model GPU Experiments (`exp-006`, RTX 4070):** 4-shard `14.19 GiB` BF16 `Qwen2.5-7B-Instruct` streamed at `444.5 MiB` peak VRAM (`196.8s`) and evaluated on `GpuFull` + GPU INT8 KV-cache across `INT4` (`0.98×` PPLx / `90.9%` Top-1), `INT3` (`1.07×` / `86.3%`), **`INT2` (`2.251 bpp`, `1.85 GiB`, `1.43×` / `65.7%`, `9.7 tok/s`)**, **`QMR-200` (`2.000 bpp`, `1.63 GiB`, `2.13×` / `52.6%`, `9.1 tok/s`)**, plus honest `1.75/1.73 bpp` cliff boundaries and `SmolLM2-360M` GPU `MATVEC_VQ_WGSL` negative control (`experiments/exp-006-event-horizon-wave3-7b-sub2bit.toml`).

## Known limits and blockers

- QMF v0/v1 remain experimental research formats and are not yet compatibility-stable across major milestones.
- Both `QMR-175_GPTQ` (`1.750 bpp`) and `TERNARY_HADAMARD_GPTQ` (`1.727 bpp`) still cliff at 7.6B; uncompensated 2-stage RVQ also cliffs without Cholesky `H⁻¹` compensation. Breaking `< 2.00 bpp` requires Hessian-compensated lattice/trellis quantization (QuIP#/QTIP-style E8 + block-128 GPTQ).
- `MAP-B` Pass 1 currently evaluates 8 full quantizers across all rows (~15 min on 7.6B); sampled-row screening will speed up `14B–32B` allocation ~10×.
- Safe cross-platform WGPU intentionally does not expose GPU temperature, board power draw, or OS-wide VRAM usage.

## Next steps (PROJECT EVENT HORIZON Wave IV — `v0.7.0`)

1. **Stage E (`14B`, e.g., `Qwen2.5-14B-Instruct`) & Stage F (`27B–32B`, e.g., `Qwen2.5-32B-Instruct`) on 8 GB RTX 4070:** Implement sampled-row `MAP-B` Pass 1 screening and resumable layer-checkpointed Q-Forge disk streaming, then run `14B` (`~3.6 GiB` at `2.00 bpp`) and `32B` (`~7.5 GiB` at `1.75–2.00 bpp`) live on 8 GB VRAM!
2. **Hessian-Compensated Lattice/Trellis Quantization (`< 1.75 bpp` Breakthrough):** Pair our GPU-resident `MATVEC_VQ_WGSL` / E8 lattice decoder with blockwise Cholesky `H⁻¹` compensation and PV-Tuning coordinate refinement to push `7B–32B` cleanly into `1.75–1.50 bpp`.
3. **Extended Architectures & Calibrated Q-Genesis:** Add Mistral/Gemma/MoE support and connect `qml-genesis` to cached `QCAL1`/`QREF1` profiles.
