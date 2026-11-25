# Quantum Morph Laboratory `v0.1.0-alpha.1` — PROJECT SINGULARITY

> *Don't just compress intelligence. Transform its representation.*

**Quantum Morph Laboratory (QML)** is now officially **MIT Licensed** and ships its first downloadable native desktop research workbench (`qml-desktop`) alongside the 13-command laboratory CLI (`qml`) — written in **100% safe Rust (`#![forbid(unsafe_code)]`)** with **zero Python dependencies**.

---

## Major Upgrades Implemented

### 1. Native Windows & Linux Desktop Research Workbench (`qml-desktop`)
- Launches directly as a native GUI application (no terminal window required on Windows) with a custom **Deep Obsidian / Ultraviolet / Electric Cyan** scientific visual identity, crash logging (`%LOCALAPPDATA%\QuantumMorphLaboratory\logs\qml-desktop.log`), and persistent read-only model library state.
- **10 Real-Engine Laboratory Views** (zero mocked dashboards — every button invokes live Rust crates):
  - **Command Center:** Hardware summary, quick actions, subsystem maturity matrix, and live telemetry log.
  - **Model Library:** Import Safetensors, GGUF, or QMF artifacts or scan local caches strictly read-only.
  - **Q-Scope Observatory:** Streaming tensor statistics, outlier crest factors, quantization susceptibility, and live `egui_plot` **32-bin weight distribution histograms** & **singular value decay curves** + JSON export.
  - **Q-Entangle:** Interactive **cross-layer Grassmannian subspace affinity heatmap** and ranked entangled layer pairs.
  - **Q-Forge Studio:** Memory-bounded streaming Safetensors → QMF v1 transformation with live SHA-256 provenance, complete byte accounting, and JSON export.
  - **Q-Genesis Lab:** Live 3-objective NSGA-II **Pareto frontier** (`Effective BPP` vs `Matvec Rel Error`) + JSON archive export.
  - **QMF Explorer:** Streaming header audit (no weight loading required), interactive **per-representation byte breakdown bar chart**, searchable **Tensor Directory**, and deduplicated **Shared Latent Structure** table.
  - **Q-Core Inference Chamber:** Load `.qmf` + `config.json` + `tokenizer.json`, stream live decoded tokens with `tok/s` telemetry, and run side-by-side **KL divergence / Logit RMSE / Top-1 agreement / Perplexity** comparisons against a reference QMF.
  - **Benchmarks & RMQ:** Live head-to-head RMQ vs `Q4_0`/`Q4_1`/`INT8` ablations across correlated clusters and uncorrelated i.i.d. Gaussian controls.
  - **System Observatory:** Live CPU, SIMD, RAM, and WGPU GPU adapter telemetry, resource presets, and a **3-kernel live GPU compute self-test** (`FP32`, `INT8_SYM`, `Q4_0`).

### 2. Real Pretrained Language Model Validation (`HuggingFaceTB/SmolLM2-135M-Instruct`, `exp-002`)
- Built **`qml-tokenizer`**: pure-Rust Hugging Face `tokenizer.json` Byte-Level BPE tokenizer with individual-digit pre-tokenization, ChatML support, and a stateful UTF-8 **`StreamingDecoder`** (handling multi-byte symbols like `Σ` cleanly across BPE token boundaries).
- Validated the entire pipeline end-to-end on real **SmolLM2-135M-Instruct** (`134,515,008` parameters, `269.06 MB`, Apache-2.0):
  - **Q-Scope Discovery:** Adjacent `k_proj` layers exhibit strong right-subspace Grassmannian affinity (`layers.17 <-> 18`: `right = 0.6455`, `shared = 0.3464` despite element-wise cosine `-0.0044`), while global 240-pair mean is `0.0648`.
  - **Matched-Budget MAP vs Uniform `Q4_0`:** Quantization-only `MAP` (`4.521` complete QMF bpp) reduced Mean KL divergence vs `BF16` by **16.4%** (`0.79210` vs `0.94767`), Logit RMSE by **9.8%**, and matvec error by **12.5%** vs uniform `Q4_0` (`4.511` bpp), generating coherent English SVD explanations where `Q4_0` degraded.
  - **Preserved Negative Control:** Unrestricted rank-8 SVD on `embed_tokens` lowered unweighted Frobenius error by 43% while collapsing text generation (`Mean KL = 7.665`, `PPL = 5201`), preserved honestly in `experiments/exp-002-smollm2-135m.toml`.

### 3. Direct Compressed GPU WGSL Compute Shaders & Rayon Parallelism (`qml-hardware`, `qml-quant`)
- Parallelized `qml-quant` compressed `matvec` kernels across all CPU cores with `rayon`.
- Added **direct compressed GPU `INT8_SYM` (`extractBits` + `unpack2x16float`)** and **`Q4_0 / INT4_SYM` (4-bit nibble unpack + `unpack2x16float`)** WGSL compute shaders in `qml-hardware` — executing directly on packed GPU storage buffers with **zero dense weight reconstruction in VRAM** (`~3.9×` and `~7.1×` less weight VRAM than `FP32`), verified against CPU on an NVIDIA GeForce RTX 4070 Laptop GPU (`max_abs_err < 1e-4`).

### 4. Selective Affinity-Gated Hybrid RMQ & Multi-GB QMF v1 Hardening
- Added **`TransformStrategy::RmqAffinityGated`** (`qml morph --method rmq-gated` & Q-Forge Studio): gates RMQ Shared-Basis + sparse outlier residual strictly to clusters meeting a configurable Grassmannian subspace affinity threshold while routing uncorrelated layers to `Q4_0` and protecting embeddings/`lm_head` in `INT8_SYM`.
- Upgraded **QMF v1** with configurable multi-GB `QmfBudget` (default 8 GiB), streaming header-only inspection, `Arc` shared-latent deduplication, first-class `RmqSharedCore`, on-demand `decode_row` (eliminating dense `f64` embedding reconstruction in Q-Core), and pure-Rust FIPS 180-4 **SHA-256** provenance.

---

## Security, Policy & Quality Verification

- **License:** MIT ([LICENSE](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/blob/main/LICENSE))
- **Rust CI (Ubuntu & Windows, Rust `1.100.0`):** `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and all **76 automated tests** pass cleanly.
- **Dependency Policy (`cargo-deny 0.21.0`):** `advisories ok, bans ok, licenses ok, sources ok` — zero ignored vulnerabilities (eliminated transitive `RUSTSEC-2026-0194` / `RUSTSEC-2026-0195` by scoping `eframe` features cleanly).

---

## Quick Start (Downloaded Assets)

1. **Windows:** Download `qml-desktop-v0.1.0-alpha.1-windows-x86_64.exe` (or extract `quantum-morph-laboratory-v0.1.0-alpha.1-windows-x86_64.zip`) and double-click `qml-desktop.exe`.
2. **Linux:** Extract `quantum-morph-laboratory-v0.1.0-alpha.1-linux-x86_64.tar.gz` and run `./qml-desktop` or `./qml --help`.
3. **Verify Integrity:** Compare downloaded binaries against `SHA256SUMS.txt`.

> **Experimental Notice:** QMF v0/v1, RMQ, MAP, Q-Genesis, and the WGPU compute kernels are active experimental research prototypes. Imported model weights are always opened strictly read-only.
