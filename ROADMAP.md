# Roadmap

Status key: **Done** means implemented, tested, and pushed on `main`; **Experimental** means working prototype with reproducible synthetic measurements awaiting real pretrained-model validation; **Next** is planned work. The complete master directive is preserved in [`MASTER_DEVELOPMENT_DIRECTIVE.md`](MASTER_DEVELOPMENT_DIRECTIVE.md).

## Completed Foundations & Milestones (I–VI + PROJECT SINGULARITY)

- **Done — Foundation:** Public repository on `main`, Rust 2024 workspace, zero `unsafe`, zero Python dependencies, Windows + Ubuntu CI, `cargo-deny` policy, 17-topic source-backed prior-art review.
- **Done — Milestone I (Q-Scope Foundation):** Bounded & constant-memory streaming Safetensors payload access, exact bit-level `F16`/`BF16`/`FP8`/`F32`/`F64`/int codecs, sharded index & `config.json` parsers, bounded GGUF v2/v3 metadata inspector, one-sided Jacobi exact thin SVD + Halko–Martinsson–Tropp randomized SVD, and `qml-scope` observatory with cross-layer Grassmannian subspace affinity.
- **Done — Milestone II (Quantization Baselines):** Reproducible `FP16`, `BF16`, `INT8_SYM`, `INT8_ASYM`, `Q4_0`-style `INT4_SYM`, `Q4_1`-style `INT4_ASYM`, `LowRankSvd`, and `SparseResidual` implementations with exact byte-for-byte `StorageBreakdown` accounting and Rayon-parallelized direct compressed `LinearOperator::matvec`.
- **Done (Experimental) — Milestone III (RMQ & MAP Engines):** RMQ Shared-Basis (`U_shared · C_l · V_sharedᵀ + R_l` via 2D-HOOI with RPCA-inspired outlier peeling), RMQ Shared-Core (`A_l · S · B_l + R_l`), falsifiable Gaussian control ablations, and MAP sensitivity + downstream spectral propagation mixed-precision allocator.
- **Done (Experimental) — Milestone IV (QMF v1 & Q-Forge):** Multi-GB QMF v1 binary container (`QmfBudget` default 8 GiB, streaming header-only inspector, `Arc` shared-latent deduplication, first-class `RmqSharedCore`, on-demand `decode_row`, SHA-256 provenance, CRC-32) and memory-bounded tensor-by-tensor / cluster-by-cluster `qml-forge` pipeline.
- **Done (Experimental) — Milestone V–VI (Q-Core, Q-Genesis & Real-Model Validation):** Pure-Rust Byte-Level BPE `qml-tokenizer`, `QCoreModel` with zero dense embedding reconstruction & `tie_word_embeddings`, full end-to-end validation on real `HuggingFaceTB/SmolLM2-135M-Instruct` (`exp-002`: coherent English generation, matched-budget MAP beating uniform `Q4_0` by 16.4% KL divergence, and preserved negative control), and Q-Genesis NSGA-II Pareto discovery.
- **Done (Alpha) — Project Singularity (`qml-hardware` & `qml-desktop` `v0.1.0-alpha.1`):** Cross-platform CPU/SIMD/RAM telemetry, WGPU GPU adapter discovery, validated WGSL compute shader `f32` matvec prototype (verified on NVIDIA RTX 4070 Laptop GPU), and the 10-view native Windows desktop research laboratory (`qml-desktop.exe`).

## Next Milestones

- **Next — Full GPU-Accelerated Q-Core Kernels:** Extend the validated WGPU compute matvec prototype into quantized (`Q4_0`/`INT8`/RMQ) GPU kernels for interactive multi-billion-parameter (1B–7B) inference.
- **Next — Selective Affinity-Gated RMQ on Pretrained LLMs:** Apply Q-Scope affinity gating to compress only high-affinity cross-layer clusters (e.g., adjacent `k_proj` layers with `right_affinity > 0.4`) on 1B–3B models with WikiText-2 perplexity benchmarks.
- **Next — Q-Behavior Subsystem:** Activation-space characterization, behavioral regression suites, and transparent representation-intervention experiments separate from structural compression.
- **Next — Project License Selection:** Select an explicit repository license with the project owner.
