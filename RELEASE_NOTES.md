# Quantum Morph Laboratory `v0.2.0-alpha.1` — PROJECT ACCELERATION (Persistent GPU-Resident Compressed Q-Core & SmolLM2-360M Validation)

> *Don't just compress intelligence. Transform its representation.*

**Quantum Morph Laboratory (`v0.2.0-alpha.1`)** integrates **persistent GPU-resident compressed inference** directly into **Q-Core**, executing all 6 compressed matrix formats (`FP32`, `FP16`, `BF16`, `INT8_SYM`, `Q4_0`, `Q4_1`) plus `LowRankSvd` and deduplicated **RMQ Shared-Basis / Shared-Core** latents natively in VRAM with **zero per-token weight transfers** and **zero per-token GPU buffer allocations** — written in **100% safe Rust (`#![forbid(unsafe_code)]`)**, **MIT Licensed**, with **zero Python dependencies**.

---

## Major Upgrades Implemented

### 1. Persistent GPU-Resident Compressed Operators & Reusable Scratch Buffers (`qml-hardware`)
- Added **direct compressed WGSL compute shaders** for **`FP16`** (`unpack2x16float`), **`BF16`** (`bitcast<f32>`), and **`Q4_1 (INT4_ASYM)`** (interleaved `(scale_f16, min_f16)` packed into a single `u32` per group for 1-instruction `unpack2x16float`), joining `FP32`, `INT8_SYM`, and `Q4_0 (INT4_SYM)`.
- Added **`PersistentGpuMatrix`** and preallocated reusable GPU scratch buffers (`GPU_SCRATCH_MAX_ELEMS = 131,072` floats = 512 KiB per buffer, covering vocabularies up to 131K tokens):
  - Weight buffers and `wgpu::BindGroup`s are uploaded and bound **ONCE** at model load time.
  - Each autoregressive token transfers only the tiny activation vector (`cols * 4` bytes, ~4 KiB) with **zero weight transfers**, **zero GPU buffer allocations**, and **zero bind-group allocations** per token.
- Verified exact CPU-vs-GPU equivalence (`max_abs_err < 1e-4`) across all 6 persistent compressed formats on an NVIDIA GeForce RTX 4070 Laptop GPU.

### 2. Integrated GPU Hybrid Q-Core Backend (`qml-runtime`, `qml-cli`, `qml-desktop`)
- Added **`QCoreBackend::GpuHybrid`** (`QCoreModel::from_qmf_reader_with_backend` & `QCoreModel::from_qmf_with_backend`) with automatic graceful fallback to Rayon CPU on headless hosts.
- Deduplicates shared RMQ latents (`U_shared`, `V_sharedᵀ`, `S`) across layers in VRAM via `Arc<PersistentGpuMatrix>`, reporting exact `QCoreBackendStats` (GPU adapter, GPU-resident operator count, CPU fallbacks, compressed VRAM bytes, deduplicated GPU latents).
- Added `--backend cpu|gpu` to **`qml generate`** and a **`GPU Hybrid (Persistent Compressed VRAM)`** toggle with live VRAM telemetry in **`qml-desktop` Q-Core Inference Chamber**.
- Raised default `PayloadBudget` ceilings (`2 GiB` per tensor, `16 GiB` cumulative) to support 360M–3B pretrained models out of the box.

### 3. Real Pretrained Model Scaling & GPU Acceleration Benchmarks (`SmolLM2-360M-Instruct`, `exp-003`)
Measured directly on Windows 11 x86_64 (Intel Core i9-14900HX 32 threads + NVIDIA GeForce RTX 4070 Laptop GPU Vulkan, Apache-2.0 `SmolLM2-360M-Instruct`, `361,821,120` params, `723.67 MB`; archived in `experiments/exp-003-smollm2-360m-gpu.toml`):

| Model & Variant | Complete QMF BPP | Compressed VRAM | CPU (32T) Tok/s | **GPU Hybrid Tok/s** | Speedup | Mean KL vs BF16 | Top-1 Match |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SmolLM2-135M` `MAP 4.52` | 4.521 | **72.30 MiB** | 4.18 | **11.53** | **2.76×** | 0.79210 | 61.4% |
| `SmolLM2-135M` `BF16` | 16.007 | 256.50 MiB | 7.56 | **11.67** | **1.54×** | 0.00000 | 100.0% |
| `SmolLM2-360M` `BF16` | 16.004 | 690.00 MiB | 4.74 | **12.10** | **2.55×** | 0.00000 | 100.0% |
| **`SmolLM2-360M` `INT8_SYM`** | **8.256** | **355.78 MiB** | — | **20.97** | **4.42× vs BF16 CPU** | **0.00706** | **95.5%** |
| `SmolLM2-360M` `Q4_0` | 4.506 | **194.07 MiB** | — | **11.65** | — | 0.32446 | 77.3% |
| `SmolLM2-360M` `MAP 4.52` | 4.516 | **194.49 MiB** | 2.25 | **11.92** | **5.30×** | 0.43276 | 68.2% |

- **Bit-for-Bit Coherent Equivalence:** GPU Hybrid generated **exact identical English text** to Rayon CPU on every tested `SmolLM2-135M` and `SmolLM2-360M` variant while keeping `100%` of linear projection operators (`211/211` on 135M, `225/225` on 360M) compressed in VRAM.
- **Preserved Scientific Nuance (`exp-003`):** On `SmolLM2-360M`, weight-only MAP lowered mean matvec error by `11.2%` vs uniform `Q4_0` at matched budget (`0.09001` vs `0.10134`), yet uniform `Q4_0` achieved lower output KL divergence (`0.32446` vs `0.43276`) because `g128` groups hurt wider-model logits — directly motivating activation-aware **Q-Behavior** sensitivity in `v0.2.0-alpha.2`.

---

## Security, Policy & Quality Verification

- **License:** MIT ([LICENSE](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/blob/main/LICENSE))
- **Rust CI (Ubuntu & Windows, Rust `1.100.0`):** `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and all **78 automated tests** pass cleanly (including all 6 persistent GPU compressed kernels + full GPU Hybrid RMQ Q-Core transformer equivalence).
- **Dependency Policy (`cargo-deny 0.21.0`):** `advisories ok, bans ok, licenses ok, sources ok` — zero ignored advisories.

---

## Quick Start (Downloaded Assets)

1. **Windows:** Download `qml-desktop-v0.2.0-alpha.1-windows-x86_64.exe` (or extract `quantum-morph-laboratory-v0.2.0-alpha.1-windows-x86_64.zip`) and double-click `qml-desktop.exe`.
2. **Linux:** Extract `quantum-morph-laboratory-v0.2.0-alpha.1-linux-x86_64.tar.gz` and run `./qml-desktop` or `./qml --help`.
3. **Verify Integrity:** Compare downloaded binaries against `SHA256SUMS.txt`.
