# Handoff

## PROJECT ATLAS — v0.6.2 COMPLETE

Starting checkpoint: 1c80f69477c483750dd1162f8ba68698aa918e5e. Work is exclusively on main. This owner-authorized assignment supersedes the prior AEGIS stop only for bounded-memory artifact construction, associated Q-Forge/CLI configuration, synthetic validation and v0.6.2 publication.

### Implementation

- QmfV1Builder retains its existing in-memory API and shares all encoder implementations with a new disk payload store. new_disk and finish_disk return a DiskArtifact. Scalar conversion streams through a fixed 64 KiB buffer; packed bytes are chunked. No complete compressed model payload is retained by the disk builder.
- The manifest, blob offsets/lengths/CRCs, deterministic ordering, shared latent references and QMF v1 version are preserved. The legacy small-model builder remains available. Existing reader defaults remain unchanged.
- DiskArtifact streams the complete artifact into an AEGIS sibling stage with incremental CRC-32 and SHA-256, reopens it through QmfFileReader, checks the whole-file hash and every blob without loading operators, checks cancellation and all protected source paths again, and uses AEGIS authorized/no-clobber publication.
- ArtifactLimits separates maximum artifact file, builder resident accounting, combined owned temporary file bytes, individual tensor allocation and manifest size. Defaults: 8 GiB / 256 MiB / 16 GiB / 8 GiB / 16 MiB respectively. Explicit reader_budget enables larger file-backed reads without raising default RAM allocations.
- DiskBuildOptions configures scratch storage. Free space is checked before construction, periodically during payload output and before final staging. It is advisory rather than a reservation; actual write failures propagate. NamedTempFile guards exclusively create and remove only owned scratch/stage files.
- Existing forge_path_to_file_atomic now uses disk construction. forge_path_to_file_atomic_with_options adds limits and explicit overwrite authorization. ForgeControl remains authoritative for cancellation in the forge API; standalone builders use DiskBuildOptions.cancel. Decoded tensor allocation is clamped by PayloadBudget and the artifact's tensor limit. Every backing shard and the original index/input file are protected again immediately before publication.
- Morph adds --scratch-dir, --max-artifact-bytes, --max-builder-memory-bytes, --max-temp-bytes, --max-tensor-bytes, --max-manifest-bytes and explicit --overwrite. Existing transformation strategies and calibration branches are unchanged.
- Added QmfError cancellation/resource errors and cooperative file-integrity verification. Checked arithmetic protects lengths, offsets, serialized/temporary totals, parameter counts and allocation conversions. Bounded manifest counting and conservative reservations guard builder metadata; legacy infallible metadata insertion latches any disk-mode resource failure instead of silently publishing.
- Normal dependencies tempfile and sysinfo are already in the workspace dependency graph; sysinfo is requested with only its disk feature for the new preflight responsibility. tempfile/serde_json test dependencies support isolated synthetic fixtures. No dependency versions were upgraded.
- Updated README downloads, release notes, changelog, architecture and roadmap. MIT license, memorial text, Prism renderer/shaders/identity, GPU kernels, compression/research algorithms and historical experiments were preserved.

### Synthetic validation and measured storage

- Byte-identical memory/disk artifacts across 17 transformation strategies, plus calibrated GPTQ parity. Existing file readers and operators reproduce numerical outputs, shared latent caches reuse the same entries, and sizes, offsets, complete bpp and SHA-256 match.
- Regression coverage includes independent resource bounds, checked overflow, physical free-space preflight rejection, logical scratch exhaustion, source allocation rejection before decoding, corrupt scratch CRCs, malformed dimensions, metadata limits, failed publication, cancellation during transformation and before publication, late aliases to a second shard, index protection, temporary file cleanup and unrelated scratch-file preservation. A CLI integration test exercises resource flags and explicit overwrite.
- Instrumented eight-tensor fixture, 1,048,576 encoded payload bytes: resident payload buffer 0 bytes; encoding buffer 65,536 bytes; conservative manifest/buffer accounting 151,280 bytes.
- Same eight tensor entries with longer arrays, 33,554,432 encoded payload bytes: resident payload buffer 0 bytes; encoding buffer 65,536 bytes; conservative manifest/buffer accounting 151,728 bytes.
- These counters measure builder-owned payload/buffers and conservative metadata accounting, not process RSS. Borrowed fixture arrays, input tensors, quantizer workspace, OS page cache and other application allocations are excluded. Fixtures write only modest synthetic data; no pretrained model was downloaded or transformed.
- cargo fmt --all -- --check: passed.
- cargo clippy --workspace --all-targets -- -D warnings: passed.
- cargo test --workspace: all 107 Windows tests passed, zero failures/ignored; doc-tests passed. Linux CI passed all 108 tests, including Unix-only link protection.
- cargo build --workspace --release: passed (45.52 seconds with the existing local dependency cache).
- cargo deny check: advisories/bans/licenses/sources passed under the unchanged policy; existing duplicate-version warnings remain allowed.
- Initial strict Clippy identified a cloned-reference test slice; corrected before final successful validation. Existing tests were not weakened or removed.

### Remaining memory bottlenecks and limitations

The 256 MiB resident limit is conservative builder-owned-state accounting (serialized manifest × 16 plus two 64 KiB I/O buffers), not an OS RSS cap. Existing quantizers still allocate full dense tensors and workspace; RMQ may retain a complete dense cluster. Calibration profiles, reports, source metadata and caller-owned operators require additional memory. Therefore this is bounded-payload-memory artifact construction, not complete end-to-end bounded-memory transformation.

Temporary disk limits count logical bytes of payload scratch plus final stage, so peak overlap is approximately twice the artifact payload plus manifest/header. Existing artifacts and unrelated files are not included in that owned-space budget. Disk preflight cannot reserve space against concurrent processes. Cleanup preserves user scratch directories and unrelated files. As in AEGIS, publication assumes user-controlled ancestors rather than hostile concurrent directory replacement.

Cancellation is checked between numerical operations and during bounded encoding, copying and verification; an individual running quantizer/kernel is not preempted. Some existing convenience readers and CLI compare still materialize full artifacts; file-backed APIs with explicit reader budgets are needed for larger files. Default CLI inference reader budgets remain conservative. Checksums detect corruption, not malicious producer authentication.

No 14B, 27B or 32B support has been scientifically validated. No sampled MAP-B optimization, persistent/resumable scheduling, new quantizers, GPU changes, architecture extensions or new research experiments were undertaken.

### Exact files changed from the ATLAS checkpoint

ARCHITECTURE.md; CHANGELOG.md; Cargo.lock; Cargo.toml; HANDOFF.md; README.md; RELEASE_NOTES.md; ROADMAP.md; crates/qml-cli/Cargo.toml; crates/qml-cli/src/main.rs; crates/qml-cli/tests/atlas_cli.rs; crates/qml-forge/Cargo.toml; crates/qml-forge/src/lib.rs; crates/qml-format/Cargo.toml; crates/qml-format/src/lib.rs; crates/qml-format/src/v1.rs; crates/qml-format/src/v1/disk.rs.
### Commit and CI status

Implementation commit: d426d91f6df53a2906612f65467346f0e4fb393a. Main-branch Windows/Linux/dependency-policy CI passed: https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/actions/runs/45812230204. Annotated tag v0.6.2 points to this validated commit. Release workflow was started at: https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/actions/runs/45812861029.

### Public release verified — November 27, 2026

Git and the GitHub plugin were reauthenticated successfully for ShamelesAbyss. The public release is v0.6.2 | PROJECT ATLAS: https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/tag/v0.6.2. The latest-release endpoint returns v0.6.2; draft=false and prerelease=false. The release workflow sets make_latest=true. All four mandatory assets were downloaded without authentication. All three binary hashes match the public SHA256SUMS.txt.

- Quantum-Morph-Laboratory-v0.6.2-Windows-x64.exe: 13778944 bytes; SHA-256 b488cacd6ded4125dbc6f36e465c9a6ca2ae21a17891f2ef030de45c342f5a7d.
- Quantum-Morph-Laboratory-v0.6.2-Windows-x64-Portable.zip: 9038341 bytes; SHA-256 8ce1e99cde45f5105b4cc4d74a58d4048ce17e5aeeea74be6d06cf89d851dd15.
- Quantum-Morph-Laboratory-v0.6.2-Linux-x64.tar.gz: 11914842 bytes; SHA-256 0adb4fa746517bb305fdfe219b5b28d58da00ea0762e6cb7d41400be9958ce89.

The standalone Windows EXE is byte-identical to the portable desktop binary. The downloaded portable CLI executed as qml 0.6.2, and both portable archives contain the expected applications and MIT license. Previous source CI was observed green on Windows (107 tests), Linux (108 tests) and dependency policy. Historical Actions run records currently return 404; they cannot be re-fetched. Source code and the release tag remain unchanged. All five required local validation commands passed as recorded above.

The access failure after the initial push blocked final verification temporarily; it is resolved. A final documentation-only local commit f831e0a0860720792229f130d0846f932ea25007 and the newer remote README edit 4ed1575fa2577cff4c58982b58dc5d29b23ee068 are joined with a regular merge on main, preserving both histories. The newer README edit is retained, with the mandatory direct download links and exact permanent memorial restored. No source code, mathematical algorithms, protected subsystem or experiment measurements were changed during this reconciliation.

The earlier local completion report and Windows packages remain historical local backups. The public release and checksums are now verified; a refreshed completion report supersedes the earlier blocked status. No ATLAS implementation issues remain beyond the documented resource limitations. No 14B/27B/32B scientific validation is claimed.

**STRICT STOP: PROJECT ATLAS is complete. Await further instructions; do not begin another assignment.**

## PROJECT AEGIS — v0.6.1 COMPLETE

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
- `cargo test --workspace`: all 95 tests passed on Windows, zero failed/ignored; all doc-tests passed. This includes the final projection/cache/publication regressions and existing desktop/GPU parity tests. Linux CI passed all 96 tests, including Unix symbolic/dangling-link coverage.
- `cargo deny check`: passed on v0.6.1 (`advisories ok, bans ok, licenses ok, sources ok`); existing duplicate-version warnings remain allowed by the unchanged policy.
- `cargo build --workspace --release`: passed (optimized workspace build, 5m 05s). Release CLI smoke check passed (`qml 0.6.1`). All five requested commands passed locally.

Initial compile/lint checks during implementation found a missing direct qml-core dependency, a serialization test API mismatch, and Clippy formatting/conditional suggestions; these were corrected before the successful final validation. No existing tests were weakened or removed.

### Limits and remaining work

Cache checksums detect corruption rather than authenticate a malicious producer. Hashing source provenance reads every backing weight byte once. Decoded f64 values and collection overhead require additional bounded RAM beyond the 512 MiB file ceiling. Path-based publication assumes output ancestors are user-controlled, not concurrently replaced by a hostile process. No claim of new live-model benchmark validation is made. Baseline commit: `1631b824ce9daa8f256d46c45497f1e804442dd8`. Implementation commit: `7c669db9e369bcd92ec08f4cb0cc4fefc663680a`, pushed to `main`; annotated release tag `v0.6.1` points to this commit. [Windows/Linux/dependency-policy CI](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/actions/runs/45805000853) passed all jobs. [Release workflow](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/actions/runs/45805910710) passed both platform builds and public publication. Public asset verification is complete below. No scoped implementation issues remain; stop and await authorization for further work.

### Public release verified (November 26, 2026)

- Latest completed milestone: **v0.6.1 | PROJECT AEGIS**. [Public release](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/tag/v0.6.1); GitHub's public `releases/latest` endpoint returns `v0.6.1`. Confirmed `draft: false`, `prerelease: false`; release workflow sets `make_latest: true`.
- Downloaded all four mandatory public release assets without authentication. All three binary package hashes matched the downloaded `SHA256SUMS.txt`. The Windows standalone EXE is byte-identical to the portable archive's `qml-desktop.exe`; the packaged CLI executed successfully as `qml 0.6.1`. The portable ZIP includes both executables, documentation and MIT LICENSE. The Linux archive lists both binaries and LICENSE; the Linux workflow also smoke-tested its CLI.
- Verified Windows standalone EXE: 13,719,040 bytes; SHA-256 `4b75143261d56150ce660d0cac2f5e27029c0d599c28150b177e90a5f43f7a59`.
- Verified Windows portable ZIP: 8,964,903 bytes; SHA-256 `ca1d4e88678e7d2929ab024ad33461f03b2e77a7d91dd6d94827b4eccbd2d906`.
- Verified Linux archive: 11,819,180 bytes; SHA-256 `2a080e0af708743ef7417e0f9da5baee8bae9776c77301d18c2fe55ebd1452fa`.
- Implementation and release tag: `7c669db9e369bcd92ec08f4cb0cc4fefc663680a`. A final documentation-only follow-up records these verification results on `main`; the release tag and repository history remain immutable.
- Permanent memorial text verified verbatim against the baseline README. MIT license, experiment manifests, mathematical algorithms and Project Prism renderer were not modified. No models were downloaded and no later milestone was begun.
- **STRICT STOP: PROJECT AEGIS is complete. Await explicit further instructions.** Future roadmap entries below are historical planning, not authorization.
## Current state

- Public repository: <https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory>
- Development branch: `main` only.
- Local workspace: `C:\Users\Shame\.gemini\antigravity\scratch\Quantum-Morph-Laboratory` (Windows 11, Intel Core i9-14900HX, RTX 4070 Laptop GPU 8 GB VRAM, 32 GB RAM, 2560×1440 144 Hz display).
- Local pretrained model caches (NEVER commit to Git):
  - `C:\Users\Shame\.gemini\antigravity\scratch\qml-models-cache` (`smollm2-135m`, `smollm2-360m`, `smollm2-1.7b`, `qwen2.5-1.5b`, `qwen2.5-3b`)
  - `D:\qml-models-cache\qwen2.5-7b` (`Qwen/Qwen2.5-7B-Instruct`, 4 shards, `14.19 GiB` BF16)
  - `D:\qml-work\qwen25-7b` (cached `_cal_profile.qcal`, `_ref_logits.qref`, and forged `int4/int3/int2/qmr-200/qmr-175/ternary` QMF v1 artifacts)
- Previous completed milestone: **`v0.6.0` — PROJECT EVENT HORIZON Wave III** (Layer-by-Layer Streaming BF16 Calibration & Reference Engine `< 450 MiB` peak VRAM for `7B–32B`, GPU-Resident Per-Head INT8 KV-Cache `~3.88×` smaller VRAM, 10th Persistent GPU Kernel `MATVEC_VQ_WGSL`, `QCAL1`/`QREF1` binary disk caches, activation-weighted `fit_vq_calibrated`, and `exp-006` Live Stage D **`Qwen2.5-7B-Instruct`** 8 GB GPU Breakthrough breaking **`2.251 bpp` (`1.85 GiB`, `PPLx = 1.43×`, `65.7%` Top-1)** and **exact `2.000 bpp` (`1.63 GiB`, `PPLx = 2.13×`, `52.6%` Top-1, `8.00×` smaller than BF16)**!).
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

## Future roadmap — NOT AUTHORIZED by PROJECT AEGIS (Wave IV — `v0.7.0`)

1. **Stage E (`14B`, e.g., `Qwen2.5-14B-Instruct`) & Stage F (`27B–32B`, e.g., `Qwen2.5-32B-Instruct`) on 8 GB RTX 4070:** Implement sampled-row `MAP-B` Pass 1 screening and resumable layer-checkpointed Q-Forge disk streaming, then run `14B` (`~3.6 GiB` at `2.00 bpp`) and `32B` (`~7.5 GiB` at `1.75–2.00 bpp`) live on 8 GB VRAM!
2. **Hessian-Compensated Lattice/Trellis Quantization (`< 1.75 bpp` Breakthrough):** Pair our GPU-resident `MATVEC_VQ_WGSL` / E8 lattice decoder with blockwise Cholesky `H⁻¹` compensation and PV-Tuning coordinate refinement to push `7B–32B` cleanly into `1.75–1.50 bpp`.
3. **Extended Architectures & Calibrated Q-Genesis:** Add Mistral/Gemma/MoE support and connect `qml-genesis` to cached `QCAL1`/`QREF1` profiles.
