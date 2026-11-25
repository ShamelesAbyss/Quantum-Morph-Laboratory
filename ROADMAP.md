# Roadmap

Status key: **Done** means implemented, tested, and pushed on `main`; **Experimental** means working prototype with reproducible synthetic measurements awaiting real pretrained-model validation; **Next** is planned work. The complete master directive is preserved in [`MASTER_DEVELOPMENT_DIRECTIVE.md`](MASTER_DEVELOPMENT_DIRECTIVE.md).

## Completed Foundations & Milestones (I–V)

- **Done — Foundation:** Public repository on `main`, Rust 2024 workspace, zero `unsafe`, zero Python dependencies, Windows + Ubuntu CI, `cargo-deny` policy, 17-topic source-backed prior-art review.
- **Done — Milestone I (Q-Scope Foundation):** Bounded & constant-memory streaming Safetensors payload access, exact bit-level `F16`/`BF16`/`FP8`/`F32`/`F64`/int codecs, sharded index & `config.json` parsers, bounded GGUF v2/v3 metadata inspector, one-sided Jacobi exact thin SVD + Halko–Martinsson–Tropp randomized SVD, and `qml-scope` observatory with cross-layer Grassmannian subspace affinity.
- **Done — Milestone II (Quantization Baselines):** Reproducible `FP16`, `BF16`, `INT8_SYM`, `INT8_ASYM`, `Q4_0`-style `INT4_SYM`, `Q4_1`-style `INT4_ASYM`, `LowRankSvd`, and `SparseResidual` implementations with exact byte-for-byte `StorageBreakdown` accounting and direct compressed `LinearOperator::matvec`.
- **Done (Experimental) — Milestone III (RMQ & MAP Engines):** RMQ Shared-Basis (`U_shared · C_l · V_sharedᵀ + R_l` via 2D-HOOI with RPCA-inspired outlier peeling), RMQ Shared-Core (`A_l · S · B_l + R_l`), falsifiable Gaussian control ablations, and MAP sensitivity + downstream spectral propagation mixed-precision allocator.
- **Done (Experimental) — Milestone IV (QMF v1 & Q-Forge):** Generalized multi-tensor QMF v1 binary container with Shared Latent Table, sparse residuals, canonical JSON manifests, per-blob & whole-file CRC-32 checks, and `qml-forge` Safetensors → QMF v1 transformation & comparison pipeline.
- **Done (Experimental) — Milestone V (Q-Core & Q-Genesis):** CPU reference Llama-style decoder-only Transformer executing directly over QMF v1 compressed operators with KV-cache, token generation, and KL divergence / perplexity evaluation; multi-objective evolutionary Pareto discovery engine (`qml-genesis`); full 12-command laboratory CLI (`qml`); original SVG branding & measured benchmark archive (`exp-001`).

## Next Milestones

- **Next — Real Pretrained Model Validation:** Evaluate Q-Scope, RMQ, MAP, Q-Forge, and Q-Core on a small openly licensed pretrained Transformer (e.g., SmolLM-135M or GPT-2 small downloaded locally outside Git), measuring real cross-layer subspace affinity and WikiText perplexity without committing model weights.
- **Next — Hardware Acceleration & System Observatory:** Extend `qml doctor` with full RAM/VRAM/GPU device discovery; add multithreaded CPU execution (`rayon`) and investigate WGPU/CUDA kernels targeting NVIDIA RTX 4070 Laptop GPU (8 GB VRAM) after preserving CPU reference equivalence tests.
- **Next — Q-Behavior Subsystem:** Activation-space characterization, behavioral regression suites, and transparent representation-intervention experiments separate from structural compression.
- **Next — Native Desktop Laboratory (`qml-desktop`):** Visually distinctive deep-obsidian/ultraviolet/cyan native Rust desktop GUI (`eframe`/`egui` + `wgpu`) integrating Q-Scope, Q-Forge, Q-Entangle, Q-Genesis, QMF Explorer, Q-Core, and System Observatory driven exclusively by real measured data.
- **Next — Project License Selection:** Select an explicit repository license with the project owner.
