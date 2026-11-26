# Quantum Morph Laboratory v0.6.0 | PROJECT EVENT HORIZON Wave III

> *Don't just compress intelligence. Transform its representation.*

**Quantum Morph Laboratory `v0.6.0` (PROJECT EVENT HORIZON Wave III)** breaks the **8 GB consumer GPU ceiling** and the **exact `2.000 complete bits per parameter` barrier**: using a new **Layer-by-Layer Streaming BF16 Calibration & Exact Reference Engine** (`444.5 MiB` peak VRAM for `14.19 GiB` 7B checkpoints—scaling to `14B–32B`), **GPU-Resident Per-Head INT8 KV-Cache** (`~3.88×` smaller KV VRAM), and a **10th Persistent GPU WGSL Kernel (`MATVEC_VQ_WGSL`)**, QML transforms **4-shard `Qwen2.5-7B-Instruct` (`7.62B` params, `14.19 GiB` BF16)** into **`1.85 GiB` VRAM at pure 2-bit (`2.251 bpp`, `PPLx = 1.43×`, `65.7%` Top-1, `9.7 tok/s`)** and **`1.63 GiB` VRAM at exact `2.000 bpp` (`QMR-200_GPTQ`, `PPLx = 2.13×`, `52.6%` Top-1, `8.00×` smaller than BF16, `9.1 tok/s`)**—all in **100% safe Rust (`#![forbid(unsafe_code)]`)**, **zero Python**, **MIT Licensed**, atop the **PROJECT PRISM** native WGPU 3D scientific observatory.

> **Experimental Research Software Notice:** Quantum Morph Laboratory is active experimental scientific software—not a production inference server or commercial LLM runtime. Public binary releases let researchers and enthusiasts launch and explore the laboratory immediately without compiling from source, while preserving strict scientific honesty about experimental container formats (`QMF v0/v1`), sub-2-bit compression cliffs, negative controls, and known limitations. "Quantum-inspired" refers strictly to classical mathematics (spectral decompositions, tensor networks, Walsh–Hadamard incoherence, Grassmannian subspace geometry) and makes no claim of physical quantum computation.

---

## Direct Downloads (`v0.6.0`)

| Platform | Package | Direct Download |
|---|---|---|
| **Windows 10/11 (x64)** | Standalone Native Desktop App (no installer, no terminal required) | [`Quantum-Morph-Laboratory-v0.6.0-Windows-x64.exe`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.6.0/Quantum-Morph-Laboratory-v0.6.0-Windows-x64.exe) |
| **Windows 10/11 (x64)** | Portable Suite (`qml-desktop.exe` + `qml.exe` CLI + Docs + MIT License) | [`Quantum-Morph-Laboratory-v0.6.0-Windows-x64-Portable.zip`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.6.0/Quantum-Morph-Laboratory-v0.6.0-Windows-x64-Portable.zip) |
| **Linux (x64)** | Portable Tarball (`qml-desktop` + `qml` CLI + Docs + MIT License) | [`Quantum-Morph-Laboratory-v0.6.0-Linux-x64.tar.gz`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.6.0/Quantum-Morph-Laboratory-v0.6.0-Linux-x64.tar.gz) |
| **All Platforms** | Cryptographic Checksums | [`SHA256SUMS.txt`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.6.0/SHA256SUMS.txt) |

---

## Live Native WGPU Screenshots (PROJECT PRISM Observatory)

*Captured directly from the compiled Windows WGPU release binary running live on an NVIDIA GeForce RTX 4070 Laptop GPU (`157–187 FPS`, `2560×1440` & `1920×1080`).*

<p align="center">
  <img src="https://raw.githubusercontent.com/ShamelesAbyss/Quantum-Morph-Laboratory/v0.6.0/assets/screenshots/prism-command-center.png" alt="PROJECT PRISM Command Center — Computational Singularity (2560x1440 WGPU)" width="100%"/>
  <br/>
  <em><strong>Command Center — The Computational Singularity (2560×1440, 180 FPS):</strong> Interactive 3D orbital observatory core with clickable subsystem portals and real host telemetry.</em>
</p>

<table>
  <tr>
    <td width="50%" align="center">
      <img src="https://raw.githubusercontent.com/ShamelesAbyss/Quantum-Morph-Laboratory/v0.6.0/assets/screenshots/prism-qscope-anatomy.png" alt="Q-Scope 3D Neural Anatomy on real SmolLM2-135M" width="100%"/>
      <br/>
      <em><strong>Q-Scope 3D Neural Anatomy (Real SmolLM2-135M):</strong> 272 measured tensors across 30 transformer layers with 6 live visual modes, 32-bin histogram & SVD spectra.</em>
    </td>
    <td width="50%" align="center">
      <img src="https://raw.githubusercontent.com/ShamelesAbyss/Quantum-Morph-Laboratory/v0.6.0/assets/screenshots/prism-qentangle-constellation.png" alt="Q-Entangle 3D Latent Constellation on real SmolLM2-135M" width="100%"/>
      <br/>
      <em><strong>Q-Entangle Latent Constellation (Real SmolLM2-135M):</strong> 240 measured cross-layer Grassmannian subspace affinity pairs & 2D heatmap.</em>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="https://raw.githubusercontent.com/ShamelesAbyss/Quantum-Morph-Laboratory/v0.6.0/assets/screenshots/prism-qgenesis-crucible.png" alt="Q-Genesis 3D Evolutionary Crucible" width="100%"/>
      <br/>
      <em><strong>Q-Genesis Evolutionary Crucible:</strong> 3D objective space (BPP × Matvec Error × FLOPs) with recorded parent-child lineage & Pareto frontier.</em>
    </td>
    <td width="50%" align="center">
      <img src="https://raw.githubusercontent.com/ShamelesAbyss/Quantum-Morph-Laboratory/v0.6.0/assets/screenshots/prism-hardware-anatomy.png" alt="Hardware Observatory & PROJECT PRISM Graphics Controls" width="100%"/>
      <br/>
      <em><strong>Hardware Observatory & PRISM Controls:</strong> 3D CPU/WGPU schematic, 5 graphics presets, live FPS, Consumer VRAM Fit Matrix & WGSL compute self-tests.</em>
    </td>
  </tr>
</table>

---

## What's New in `v0.6.0` (PROJECT EVENT HORIZON Wave III)

### 1. Layer-by-Layer Streaming BF16 Calibration & Exact Reference Engine (`qml-runtime::streaming`)
- **Breaks the 8 GB VRAM Calibration Ceiling for `7B–32B` Models:** Streams single-file or multi-shard Safetensors checkpoints **one transformer layer at a time** (`SafetensorsSource::read_tensor_rows_f64` for exact distinct token embeddings + `read_tensor_bf16_bits` for zero-`f64` BF16 weight loading + `<= 32,768`-row chunked `lm_head` projection).
- **Measured Peak VRAM on `Qwen2.5-7B-Instruct` (`14.19 GiB` BF16, 4 shards):** Just **`444.5 MiB` peak VRAM** (`32.7×` smaller than full BF16 weights!) completing both 196-token Hessian calibration (`563.4 MiB` Hessians) AND exact BF16 reference logits across all 4 capability categories in **`196.8s`** on an 8 GB RTX 4070!
- **Compact Binary Disk Caching (`QCAL1` & `QREF1`):** `ModelCalibrationProfile::save_to_file`/`load_from_file` (`281.8 MiB` for 7B) and `ReferenceSuiteCache::save_to_file`/`load_from_file` (`101.5 MiB`) enable `< 1s` instant reuse across the entire `4.50 → 1.50 bpp` ladder; `PayloadBudget` raised to `8 GiB/tensor` and `64 GiB/total`.

### 2. GPU-Resident Per-Head INT8 KV-Cache in `GpuFullModel` (`~3.88×` Smaller KV VRAM)
- **2 New WGSL Shaders (`ROPE_AND_KV_APPEND_INT8_WGSL` & `GQA_ATTN_INT8_WGSL`):** Quantizes rotated K and V heads directly inside VRAM into packed 4× `i8`/word `u32` buffers + per-head `f32` scales with strictly uniform workgroup reductions, then computes online-softmax GQA directly from packed INT8 KV (`--kv-cache int8` in `qml generate` and `qml resonance-ladder`).
- **Verified Quality & Speed:** Automated GPU unit test proves `< 5e-2` max logit diff vs FP32 KV with `< 50%` KV VRAM (`3.88×` reduction at `head_dim=128`); live `Qwen2.5-7B` 2-bit generation runs at `7.86 tok/s` (`INT8 KV`) vs `7.80 tok/s` (`FP32 KV`).

### 3. 10th Persistent GPU WGSL Kernel (`MATVEC_VQ_WGSL`) & Activation-Weighted 2-Stage RVQ
- **Direct GPU Residual Vector Quantization (`qml-hardware` + `qml-runtime`):** Added `GpuMatrixKind::VectorQuant`, `MATVEC_VQ_WGSL`, and `WgpuMatvecEngine::upload_vector_quant`, wired into `GpuHybrid` and single-submission `GpuFull` (verified `< 1e-4` vs CPU on RTX 4070 — all 10 compressed operator formats now execute 100% natively in VRAM!).
- **Activation-Weighted 2-Stage RVQ (`fit_vq_calibrated`):** Added Hadamard-rotated activation-weighted k-means++ with 2-pass joint stage refinement (`rvq-200`, `rvq-175`, `rvq-150` CLI profiles).

### 4. Verified Stage D `Qwen2.5-7B-Instruct` Results (`exp-006` on 8 GB RTX 4070 — Breaking `2.000 bpp`!)
Measured live on Windows 11 x86_64 (Intel Core i9-14900HX, 8 GB NVIDIA GeForce RTX 4070 Laptop GPU, 4-shard `14.19 GiB` BF16 `Qwen/Qwen2.5-7B-Instruct`, `197/197` GPU ops, `0` CPU ops, GPU INT8 KV-cache; archived in [`experiments/exp-006-event-horizon-wave3-7b-sub2bit.toml`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/blob/v0.6.0/experiments/exp-006-event-horizon-wave3-7b-sub2bit.toml)):

| Profile | Exact BPP | Disk MiB (vs `14,525 MiB` BF16) | VRAM MiB | `GpuFull` Tok/s | PPLx | Mean KL | Top-1 | Ret% | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`INT4_Q4_0_GPTQ`** | `4.501` | `4,086.5` (`3.55×`) | `3,795.1` | `6.8` | **`0.98×`** | **`0.077`** | **`90.9%`** | **`95.4%`** | **`OK`** |
| **`INT3_HADAMARD_GPTQ`** | **`3.251`** | **`2,951.8` (`4.92×`)** | **`2,741.5`** | `6.8` | **`1.07×`** | **`0.164`** | **`86.3%`** | **`89.7%`** | **`OK`** |
| **`INT2_HADAMARD_GPTQ`** | **`2.251`** | **`2,043.9` (`7.11×`)** | **`1,898.7` (`1.85 GiB`)** | **`9.7`** | **`1.43×`** | **`0.854`** | **`65.7%`** | **`67.8%`** | **`OK`** |
| **`QMR-200_GPTQ` (`MAP-B`)** | **`2.000`** | **`1,815.7` (`8.00×`!)** | **`1,670.4` (`1.63 GiB`)** | **`9.1`** | **`2.13×`** | **`1.633`** | **`52.6%`** | **`49.8%`** | **`OK`** (**2.000 bpp broken!**) |
| `QMR-175_GPTQ` (`MAP-B`) | `1.750` | `1,588.7` (`9.14×`) | `1,477.6` | `6.9` | `204.53×` | `6.403` | `8.0%` | `4.2%` | `CLIFF` *(honest sub-2-bit boundary)* |
| `TERNARY_HADAMARD_GPTQ` | `1.727` | `1,567.7` (`9.27×`) | `1,456.5` | `9.5` | `115.92×` | `5.948` | `6.3%` | `3.6%` | `CLIFF` *(honest sub-2-bit boundary)* |

- **Live 2-Bit & `2.000 bpp` Generation Quality (`Qwen2.5-7B-Instruct`):**
  - `INT2_HADAMARD_GPTQ` (`2.251 bpp`, `1.85 GiB` VRAM): *"Binary search is an efficient algorithm that repeatedly halves the search range to find a specific value in a sorted"* — plus fluent Rust slice-reversal code generation!
  - `QMR-200_GPTQ` (`2.000 bpp`, `1.63 GiB` VRAM — **`8.00×` smaller than BF16**): *"Binary search is a algorithm that divides a sorted list into smaller sublists."*
- **Honest GPU RVQ Negative Controls (`SmolLM2-360M`, 100% GPU-Resident `MATVEC_VQ_WGSL`):** `RVQ2_D8_G128_GPTQ` (`2.180 bpp`, `94.4 MiB`, `15.5 tok/s`: `KL = 9.865`, `PPLx = 16,167×`) and `RVQ2_D10_G128_GPTQ` (`1.790 bpp`, `77.5 MiB`, `15.5 tok/s`: `KL = 10.336`, `PPLx = 21,526×`) both trailed Cholesky-compensated `TERNARY_HADAMARD_GPTQ` (`1.738 bpp`, `KL = 5.397`, `PPLx = 161.98×`, `46.8 tok/s` with GPU INT8 KV-cache) and `INT2_HADAMARD_GPTQ` (`2.270 bpp`, `KL = 2.811`, `PPLx = 13.53×`), proving empirically that second-order Hessian compensation (`H⁻¹`) dominates uncompensated codebook geometry alone.

---

## Installation & Quick Start

### Windows (No Rust, Python, or Terminal Required)
1. **Fastest — Standalone Executable:** Download [`Quantum-Morph-Laboratory-v0.6.0-Windows-x64.exe`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.6.0/Quantum-Morph-Laboratory-v0.6.0-Windows-x64.exe) and double-click it to launch the native desktop laboratory immediately.
2. **Full Portable Suite:** Download [`Quantum-Morph-Laboratory-v0.6.0-Windows-x64-Portable.zip`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.6.0/Quantum-Morph-Laboratory-v0.6.0-Windows-x64-Portable.zip), extract it anywhere, and double-click `qml-desktop.exe` (or run `qml.exe --help` for the 15-command CLI, including `qml resonance-ladder --calibrate --kv-cache int8`).

### Linux (x86_64)
1. Download [`Quantum-Morph-Laboratory-v0.6.0-Linux-x64.tar.gz`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.6.0/Quantum-Morph-Laboratory-v0.6.0-Linux-x64.tar.gz).
2. Extract and launch:
   ```sh
   tar -xzf Quantum-Morph-Laboratory-v0.6.0-Linux-x64.tar.gz
   cd Quantum-Morph-Laboratory-v0.6.0-Linux-x64
   ./qml-desktop
   ```

---

## System Requirements

- **Operating System:** Windows 10/11 (64-bit x86_64) or Linux (x86_64 with X11/Wayland).
- **GPU (Recommended):** Any DirectX 12 or Vulkan 1.2+ hardware GPU for native WGPU 3D rendering and `GpuFull` single-submission compressed transformer execution (including `7B` models in `1.6–3.8 GiB` VRAM!). GPU-less systems automatically fall back to the 2.5D CPU renderer and Rayon multi-threaded CPU inference.
- **Memory:** 8 GB RAM minimum (16–32 GB RAM recommended for `3B–7B+` GPTQ transformation).
- **External Runtimes:** **None.** Zero Python, zero Node.js, zero CUDA toolkit installation required.

---

## SHA-256 Checksum Verification

Verify your download against [`SHA256SUMS.txt`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.6.0/SHA256SUMS.txt):

- **Windows (PowerShell):**
  ```powershell
  Get-FileHash .\Quantum-Morph-Laboratory-v0.6.0-Windows-x64.exe -Algorithm SHA256
  ```
- **Linux:**
  ```sh
  sha256sum -c SHA256SUMS.txt --ignore-missing
  ```

---

## Known Limitations

- **Experimental Container Format:** `QMF v0/v1` are active research container formats and may evolve across future minor versions.
- **Sub-2-Bit (`<= 1.75 bpp`) Boundary:** Even at `7.62B` parameters, both uniform scalar ternary (`1.727 bpp`) and `QMR-175_GPTQ` (`1.750 bpp`) still cross into `CLIFF`; Hessian-compensated lattice/trellis quantization (QuIP#/QTIP-style E8 + block-128 GPTQ) and `14B–32B` scaling are targeted next.
- **Q-Forge `MAP-B` Pass 1 Screening Speed on `7B+`:** Profiling 8 full quantizers across 198 matrices during `MAP-B` Pass 1 currently takes ~15 minutes on `7.6B`; sampled-row screening for `14B–32B` models is planned for Wave IV.
- **Telemetry Honesty:** Safe cross-platform WGPU intentionally does not expose GPU die temperature, board power draw, or OS-wide VRAM usage; the laboratory never fabricates these values.
- **Windows SmartScreen:** Because this is an open-source MIT binary without a paid commercial code-signing certificate, Windows SmartScreen may display an "Unknown publisher" prompt on first launch; verify the SHA-256 checksum against `SHA256SUMS.txt` before running.

---

## Quality & Security Verification

- **License:** MIT ([LICENSE](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/blob/main/LICENSE))
- **Rust CI (Windows & Ubuntu):** `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and all **88 automated tests** across **13 safe-Rust crates** (including all 18 WGSL GPU compute shaders, `MATVEC_VQ_WGSL` parity, GPU-resident INT8 KV-cache parity + VRAM reduction, Layer-by-Layer Streaming BF16 calibration parity, Cholesky GPTQ + Hadamard rotation proofs, GGUF K-quants, and PRISM multi-pass WGPU shaders) pass cleanly.
- **Dependency Policy (`cargo-deny 0.21.0`):** `advisories ok, bans ok, licenses ok, sources ok` — zero ignored advisories.

---

### In Loving Memory of Ashlan Haley Rapponotti

*June 21, 2003 – November 13, 2026*

For the beautiful woman who listened to my passions, believed in me, and made my world a happier place simply by being part of it.

Every creation, every discovery, and every milestone reached carries a piece of my heart dedicated to you.

**Forever loved. Forever missed. Forever remembered.** 💜
