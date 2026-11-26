# Quantum Morph Laboratory v0.5.0 | PROJECT EVENT HORIZON Wave II

> *Don't just compress intelligence. Transform its representation.*

**Quantum Morph Laboratory `v0.5.0` (PROJECT EVENT HORIZON Wave II)** delivers three major breakthroughs toward running capable multi-billion-parameter language models on consumer 6–8 GB GPUs: an **Activation-Aware Blockwise Cholesky GPTQ + Hadamard Compensation Engine** (zero extra storage bytes), a **Full GPU-Resident Single-Submission WGSL Transformer** (`22.1×` real end-to-end token throughput speedup on RTX 4070), and **Multi-Shard Qwen2 / Llama Scaling up to 3B Parameters** (`exp-005`), breaking the **`2.75 bpp`**, **`2.50 bpp`**, and **`2.25 bpp` (`2-bit`)** post-training quantization barriers with coherent language generation—all in **100% safe Rust (`#![forbid(unsafe_code)]`)**, **zero Python**, **MIT Licensed**, atop the **PROJECT PRISM** native WGPU 3D scientific observatory.

> **Experimental Research Software Notice:** Quantum Morph Laboratory is active experimental scientific software—not a production inference server or commercial LLM runtime. Public binary releases let researchers and enthusiasts launch and inspect the laboratory immediately without compiling from source, while preserving strict scientific honesty about experimental container formats (`QMF v0/v1`), sub-2-bit compression cliffs, negative controls, and known limitations. "Quantum-inspired" refers strictly to classical mathematics (spectral decompositions, tensor networks, Walsh–Hadamard incoherence, Grassmannian subspace geometry) and makes no claim of physical quantum computation.

---

## Direct Downloads (`v0.5.0`)

| Platform | Package | Direct Download |
|---|---|---|
| **Windows 10/11 (x64)** | Standalone Native Desktop App (no installer, no terminal required) | [`Quantum-Morph-Laboratory-v0.5.0-Windows-x64.exe`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.5.0/Quantum-Morph-Laboratory-v0.5.0-Windows-x64.exe) |
| **Windows 10/11 (x64)** | Portable Suite (`qml-desktop.exe` + `qml.exe` CLI + Docs + MIT License) | [`Quantum-Morph-Laboratory-v0.5.0-Windows-x64-Portable.zip`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.5.0/Quantum-Morph-Laboratory-v0.5.0-Windows-x64-Portable.zip) |
| **Linux (x64)** | Portable Tarball (`qml-desktop` + `qml` CLI + Docs + MIT License) | [`Quantum-Morph-Laboratory-v0.5.0-Linux-x64.tar.gz`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.5.0/Quantum-Morph-Laboratory-v0.5.0-Linux-x64.tar.gz) |
| **All Platforms** | Cryptographic Checksums | [`SHA256SUMS.txt`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.5.0/SHA256SUMS.txt) |

---

## Live Native WGPU Screenshots (PROJECT PRISM Observatory)

*Captured directly from the compiled Windows WGPU release binary running live on an NVIDIA GeForce RTX 4070 Laptop GPU (`157–187 FPS`, `2560×1440` & `1920×1080`).*

<p align="center">
  <img src="https://raw.githubusercontent.com/ShamelesAbyss/Quantum-Morph-Laboratory/v0.5.0/assets/screenshots/prism-command-center.png" alt="PROJECT PRISM Command Center — Computational Singularity (2560x1440 WGPU)" width="100%"/>
  <br/>
  <em><strong>Command Center — The Computational Singularity (2560×1440, 180 FPS):</strong> Interactive 3D orbital observatory core with clickable subsystem portals and real host telemetry.</em>
</p>

<table>
  <tr>
    <td width="50%" align="center">
      <img src="https://raw.githubusercontent.com/ShamelesAbyss/Quantum-Morph-Laboratory/v0.5.0/assets/screenshots/prism-qscope-anatomy.png" alt="Q-Scope 3D Neural Anatomy on real SmolLM2-135M" width="100%"/>
      <br/>
      <em><strong>Q-Scope 3D Neural Anatomy (Real SmolLM2-135M):</strong> 272 measured tensors across 30 transformer layers with 6 live visual modes, 32-bin histogram & SVD spectra.</em>
    </td>
    <td width="50%" align="center">
      <img src="https://raw.githubusercontent.com/ShamelesAbyss/Quantum-Morph-Laboratory/v0.5.0/assets/screenshots/prism-qentangle-constellation.png" alt="Q-Entangle 3D Latent Constellation on real SmolLM2-135M" width="100%"/>
      <br/>
      <em><strong>Q-Entangle Latent Constellation (Real SmolLM2-135M):</strong> 240 measured cross-layer Grassmannian subspace affinity pairs & 2D heatmap.</em>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="https://raw.githubusercontent.com/ShamelesAbyss/Quantum-Morph-Laboratory/v0.5.0/assets/screenshots/prism-qgenesis-crucible.png" alt="Q-Genesis 3D Evolutionary Crucible" width="100%"/>
      <br/>
      <em><strong>Q-Genesis Evolutionary Crucible:</strong> 3D objective space (BPP × Matvec Error × FLOPs) with recorded parent-child lineage & Pareto frontier.</em>
    </td>
    <td width="50%" align="center">
      <img src="https://raw.githubusercontent.com/ShamelesAbyss/Quantum-Morph-Laboratory/v0.5.0/assets/screenshots/prism-hardware-anatomy.png" alt="Hardware Observatory & PROJECT PRISM Graphics Controls" width="100%"/>
      <br/>
      <em><strong>Hardware Observatory & PRISM Controls:</strong> 3D CPU/WGPU schematic, 5 graphics presets, live FPS, Consumer VRAM Fit Matrix & WGSL compute self-tests.</em>
    </td>
  </tr>
</table>

---

## What's New in `v0.5.0` (PROJECT EVENT HORIZON Wave II)

### 1. Activation-Aware Blockwise Cholesky GPTQ + Hadamard Engine (`qml-quant::gptq`, Zero Extra Bytes)
- **Streaming Input Covariance Calibration (`ModelCalibrationProfile`):** Collects per-layer input channel RMS and `64×64` block-diagonal Hessians `H_b = (2/N) X_bᵀ X_b` from a strictly disjoint calibration corpus (sharing accumulators across QKV and Gate/Up projections).
- **Exact Orthogonal Hadamard Covariance Rotation (`Q_b H_b Q_bᵀ`):** Rotates each `64×64` Hessian block into the exact randomized Walsh–Hadamard incoherence basis so GPTQ error compensation operates directly in the quantized coordinate system (`< 1e-10` quadratic-form preservation).
- **Descending Activation-Salience Ordering (`actorder`) + Ledoit–Wolf Adaptive Shrinkage:** Rayon row-parallel Cholesky GPTQ across **all 5 representations** (`INT4 Q4_0`, `INT3 Hadamard`, `INT2 Hadamard`, `Ternary Hadamard`, and activation-weighted **PROJECT ECHO**), adding **zero extra storage bytes** to QMF v1 artifacts.

### 2. Full GPU-Resident Single-Submission WGSL Transformer (`QCoreBackend::GpuFull` — `22.1×` Speedup)
- **6 New WGSL Compute Shaders (`crates/qml-hardware/src/gpu_full.rs`):** `RMSNORM_WGSL`, `HADAMARD_FWD_WGSL` (1-bit/element packed exact sign vectors), `ROPE_AND_KV_APPEND_WGSL`, `GQA_ATTN_WGSL` (numerically stable online softmax for arbitrary sequence length), `SWIGLU_WGSL`, and `VEC_ADD_WGSL`.
- **Single Command Submission Per Token:** Chains all 24–36 transformer layers, optional Qwen2 attention biases, GPU-resident KV-cache, and `lm_head` into **one `wgpu::CommandEncoder` submission per token**, eliminating per-projection CPU↔GPU sync overhead.
- **High-Vocab & Multi-GB VRAM Support:** Unlocked full adapter limits (`adapter.limits()`) and expanded persistent GPU scratch buffers to `262,144` elements (supporting 152K Qwen2.5 and 256K Gemma vocabularies).
- **Measured Real-Model Throughput (`SmolLM2-360M` `Q4_0`, identical greedy output on RTX 4070):** `GpuHybrid` `1.84 tok/s` (`36,923 ms`) → **`GpuFull` `40.70 tok/s` (`1,671 ms`) — `22.1×` faster!** Reaches up to **`63.3 tok/s`** at 2-bit on `360M`, **`31.5 tok/s`** on `Qwen2.5-1.5B`, and **`18.6 tok/s`** on 2-shard `Qwen2.5-3B`.

### 3. Qwen2 Architecture, Multi-Shard Streaming, INT8 KV-Cache & GGUF K-Quants
- **Qwen2 Attention Biases (`qml-runtime`):** Optional `q_proj.bias`, `k_proj.bias`, and `v_proj.bias` supported identically across CPU, `GpuHybrid`, and `GpuFull`.
- **Unified Single + Multi-Shard Safetensors (`SafetensorsSource`):** Transparently streams single `.safetensors`, sharded `model.safetensors.index.json`, or model directories with at most one open shard handle (`PayloadBudget` up to `4 GiB/tensor`, `32 GiB/total`) and direct atomic file forging (`forge_path_to_file_atomic`) without duplicate full-file RAM copies.
- **GGUF K-Quant Safe-Rust Ingestion (`qml-model::gguf`):** Added exact safe-Rust dequantizers for `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, and `Q6_K`.
- **INT8 KV-Cache (`KvCacheMode::Int8`) & `ReferenceSuiteCache` (`qml-behavior`):** Per-head symmetric INT8 KV-cache compression (`< 5e-2` max diff vs FP64) and cached BF16 reference logits so `ref_model` drops and frees 100% of 8 GB VRAM before loading multi-GB compressed candidates.

### 4. Verified Real-Model Research Results (`exp-005` on RTX 4070 — Breaking `2.75`, `2.50`, and `2.25 bpp`!)
Measured live on Windows 11 x86_64 (Intel Core i9-14900HX, NVIDIA GeForce RTX 4070 Laptop GPU 8 GB VRAM, 4-category strictly held-out Capability Retention Suite; archived in [`experiments/exp-005-event-horizon-wave2-gptq-scaling.toml`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/blob/v0.5.0/experiments/exp-005-event-horizon-wave2-gptq-scaling.toml)):

#### Stage A Matched-Byte Control (`SmolLM2-360M-Instruct`, Zero Extra Bytes vs `v0.4.0` Weight-Only)
| Variant | Complete BPP | Disk MiB | `v0.4.0` Weight-Only PPLx → **`v0.5.0` GPTQ PPLx** | Mean KL | Top-1 Match | `GpuFull` tok/s | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| `INT3_HADAMARD` | `3.270` | `141.4` | `5.63×` → **`1.64×`** (`3.43×` lower) | `0.624` (`−67.3%`) | `68.2%` (`+26.7 pp`) | `41.8` | `OK` |
| `QMR-300` | `3.000` | `129.7` | `13.35×` → **`2.49×`** (`5.36×` lower) | `1.072` (`−61.8%`) | `58.5%` (`+26.7 pp`) | `44.8` | `OK` |
| **`QMR-275`** | **`2.750`** | **`119.0`** | *(cliff)* → **`3.52×`** (**2.75 bpp broken!**) | **`1.457`** | **`44.3%`** | **`50.7`** | **`OK`** |
| `QMR-250` | `2.500` | `108.2` | `393.75×` → **`5.45×`** (`72.2×` lower) | `1.981` (`−69.8%`) | `36.9%` (`7.2×` higher) | `54.5` | `CLIFF` |
| `INT2_HADAMARD` | `2.270` | `98.3` | `2,653.7×` → **`13.53×`** (`41,068×` vs plain 2-bit) | `2.811` | `29.5%` | `63.3` | `CLIFF` |

#### Stage B & Stage C Scaling (`1.5B`, `1.7B`, and 2-Shard `3B` Models on RTX 4070 `GpuFull`)
| Model | Variant | Complete BPP | Disk MiB (vs BF16) | PPLx | Mean KL | Top-1 Match | Cap. Ret. | `GpuFull` tok/s | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SmolLM2-1.7B` | `INT3_HADAMARD_GPTQ` | `3.252` | `663.9` (`5.15×`) | `1.71×` | `0.698` | `83.0%` | `70.7%` | `14.1` | `OK` |
| `SmolLM2-1.7B` | **`QMR-250_GPTQ`** | **`2.500`** | **`510.5` (`6.70×`)** | **`2.79×`** | **`1.327`** | **`58.0%`** | **`46.9%`** | **`22.6`** | **`OK`** (**2.50 bpp broken!**) |
| `Qwen2.5-1.5B` | `INT3_HADAMARD_GPTQ` | `3.253` | `599.7` (`5.15×`) | **`1.03×`** | `0.295` | `77.7%` | `87.2%` | `22.8` | `OK` |
| `Qwen2.5-1.5B` | **`INT2_HADAMARD_GPTQ`** | **`2.253`** | **`415.7` (`7.43×`)** | **`4.12×`** | **`1.993`** | **`41.7%`** | **`33.0%`** | **`31.5`** | **`OK`** (**2.25 bpp broken!**) |
| `Qwen2.5-3B` (2-shard) | `INT4_Q4_0_GPTQ` | `4.502` | `1,657.2` (`3.72×`) | **`1.01×`** | `0.147` | `84.0%` | `91.4%` | `11.2` | `OK` |
| `Qwen2.5-3B` (2-shard) | `INT3_HADAMARD_GPTQ` | `3.253` | `1,197.4` (`5.15×`) | **`1.07×`** | `0.237` | `78.3%` | `85.8%` | `8.7` | `OK` |
| `Qwen2.5-3B` (2-shard) | **`INT2_HADAMARD_GPTQ`** | **`2.253`** | **`829.6` (`7.43×`)** | **`2.76×`** | **`1.606`** | **`55.4%`** | **`45.8%`** | **`18.6`** | **`OK`** (**coherent 2-bit!**) |
| `Qwen2.5-1.5B` | `TERNARY_HADAMARD_GPTQ` | `1.731` | `319.6` (`9.66×`) | `590.16×` | `7.239` | `7.4%` | `0.2%` | `27.6` | `CLIFF` *(honest sub-2-bit boundary)* |

- **Strong Scaling Discovery:** At exact `2.25 bpp` (`INT2_HADAMARD_GPTQ`), doubling parameters from `1.5B → 3B` cuts perplexity inflation (`4.12× → 2.76×`), improves Top-1 agreement (`41.7% → 55.4%`), and lifts Capability Retention (`33.0% → 45.8%`), producing coherent technical explanations at `18.6 tok/s` in just `829.6 MiB`—validating our Wave III trajectory toward `7B–32B` sub-3-bit models on 8 GB GPUs.
- **Honest Sub-2-Bit Boundary:** Uniform scalar `1.73 bpp` ternary still crosses into `CLIFF` at `<= 1.7B` even with block-64 GPTQ—motivating Wave III's non-uniform 2-stage RVQ/AQLM GPU lookup-table kernels, larger Hessian blocks (`128`), and `7B–14B` scaling.

---

## Installation & Quick Start

### Windows (No Rust, Python, or Terminal Required)
1. **Fastest — Standalone Executable:** Download [`Quantum-Morph-Laboratory-v0.5.0-Windows-x64.exe`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.5.0/Quantum-Morph-Laboratory-v0.5.0-Windows-x64.exe) and double-click it to launch the native desktop laboratory immediately.
2. **Full Portable Suite:** Download [`Quantum-Morph-Laboratory-v0.5.0-Windows-x64-Portable.zip`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.5.0/Quantum-Morph-Laboratory-v0.5.0-Windows-x64-Portable.zip), extract it anywhere, and double-click `qml-desktop.exe` (or run `qml.exe --help` for the 15-command CLI, including `qml resonance-ladder --calibrate`).

### Linux (x86_64)
1. Download [`Quantum-Morph-Laboratory-v0.5.0-Linux-x64.tar.gz`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.5.0/Quantum-Morph-Laboratory-v0.5.0-Linux-x64.tar.gz).
2. Extract and launch:
   ```sh
   tar -xzf Quantum-Morph-Laboratory-v0.5.0-Linux-x64.tar.gz
   cd Quantum-Morph-Laboratory-v0.5.0-Linux-x64
   ./qml-desktop
   ```

---

## System Requirements

- **Operating System:** Windows 10/11 (64-bit x86_64) or Linux (x86_64 with X11/Wayland).
- **GPU (Recommended):** Any DirectX 12 or Vulkan 1.2+ capable GPU for native WGPU 3D rendering and `GpuFull` single-submission compressed transformer execution. Systems without a compatible GPU automatically fall back to the 2.5D CPU renderer and Rayon multi-threaded CPU inference.
- **Memory:** 8 GB RAM minimum (16–32 GB RAM recommended for `1.5B–3B+` GPTQ calibration and transformation).
- **External Runtimes:** **None.** Zero Python, zero Node.js, zero CUDA toolkit installation required.

---

## SHA-256 Checksum Verification

Verify your download against [`SHA256SUMS.txt`](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/download/v0.5.0/SHA256SUMS.txt):

- **Windows (PowerShell):**
  ```powershell
  Get-FileHash .\Quantum-Morph-Laboratory-v0.5.0-Windows-x64.exe -Algorithm SHA256
  ```
- **Linux:**
  ```sh
  sha256sum -c SHA256SUMS.txt --ignore-missing
  ```

---

## Known Limitations

- **Experimental Container Format:** `QMF v0/v1` are active research container formats and may evolve across future minor versions.
- **Sub-2-Bit (`<= 1.75 bpp`) Scalar Floor:** Uniform scalar ternary (`1.73 bpp`) still exhibits high perplexity drift on `<= 1.7B` models even with block-64 GPTQ; 2-stage RVQ/AQLM GPU lookup-table kernels and `7B–14B` scaling are targeted in Wave III.
- **Models Larger Than 8 GB BF16 VRAM:** While `Qwen2.5-3B` BF16 (`5.89 GiB`) calibrates directly on `GpuFull`, `7B–32B` BF16 reference models (`15–64 GB`) require streaming CPU/layer-by-layer calibration (planned for Wave III).
- **GPU-Resident KV-Cache Precision:** `KvCacheMode::Int8` currently operates on the CPU/hybrid KV path while `GpuFull` uses FP32 KV storage; GPU-resident INT8 KV-cache quantization is planned for Wave III.
- **Telemetry Honesty:** Safe cross-platform WGPU intentionally does not expose GPU die temperature, board power draw, or OS-wide VRAM usage; the laboratory never fabricates these values.
- **Windows SmartScreen:** Because this is an open-source MIT binary without a paid commercial code-signing certificate, Windows SmartScreen may display an "Unknown publisher" prompt on first launch; verify the SHA-256 checksum against `SHA256SUMS.txt` before running.

---

## Quality & Security Verification

- **License:** MIT ([LICENSE](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/blob/main/LICENSE))
- **Rust CI (Windows & Ubuntu):** `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and all **88 automated tests** across **13 safe-Rust crates** (including all 15 WGSL GPU compute shaders, `GpuFull` vs CPU parity, blockwise Cholesky GPTQ + Hadamard rotation proofs, GGUF K-quant codecs, INT8 KV-cache, and PRISM multi-pass WGPU shaders) pass cleanly.
- **Dependency Policy (`cargo-deny 0.21.0`):** `advisories ok, bans ok, licenses ok, sources ok` — zero ignored advisories.

---

### In Loving Memory of Ashlan Haley Rapponotti

*June 21, 2003 – November 13, 2026*

For the beautiful woman who listened to my passions, believed in me, and made my world a happier place simply by being part of it.

Every creation, every discovery, and every milestone reached carries a piece of my heart dedicated to you.

**Forever loved. Forever missed. Forever remembered.** 💜
