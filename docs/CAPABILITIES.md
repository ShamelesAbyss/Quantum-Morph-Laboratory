# Current capability and evidence inventory

This is a source/evidence reassessment for CONTINUUM, not a new pretrained-model benchmark. Historical experiment files remain unchanged.

| Component | Implemented evidence | Boundary |
|---|---|---|
| qml-core | Dense matrices, exact/randomized SVD, deterministic RNG, FIPS SHA-256 tests | Classical numerical foundations; no quantum hardware/state simulation |
| qml-model | Bounded Safetensors payload reads, shard indexes, GGUF v2/v3 inspection and tested scalar/block codecs | Unknown model architecture inspection does not authorize runtime execution |
| qml-scope | Streaming moments, histograms, spectra, tensor roles and subspace comparisons | Some spectral operations still materialize matrices |
| qml-quant | Conventional FP16/BF16/INT8/INT4, RMQ, MAP, Hadamard sub-3-bit, GPTQ, VQ/ECHO and tested direct operators | Approximation/activation error is not general task capability |
| qml-format | QMF v0/v1, shared factors, bounded metadata readers, per-blob/full CRC and streaming SHA-256 | Experimental formats; explicit file/allocation limits |
| qml-forge | Tensor/cluster transformation, calibrated paths, disk-backed QMF payload construction, cancellation and safe publication | Quantizers and RMQ clusters still require dense workspace; not full-process bounded RSS |
| qml-tokenizer | Native Hugging Face byte-level BPE, ChatML formatting and stateful UTF-8 decoding | Not universal tokenizer support or SentencePiece ingestion |
| qml-runtime | Llama-style/Qwen2 mathematical execution, optional Q/K/V biases, CPU/GpuHybrid/GpuFull, KV caches, streaming calibration/reference | HF configuration admission requires stricter architecture/feature checks; present optional bias corruption must not be silently treated as absence |
| qml-hardware | Hardware selection, direct compressed matvec kernels and GPU transformer primitives with parity tests | Device/driver support varies; no OS-wide peak VRAM or power telemetry |
| qml-behavior | Traces, RSA/interventions, numerical reference comparisons and provenance-verified caches | Composite numerical fidelity is not retained intelligence or a task benchmark |
| qml-genesis | Stratified evolutionary representation search and QMF export | Experimental objective proxies; not evidence of a novel superior model |
| qml-cli | Inspection, transformation, generation, comparison and experiment commands | Some convenience paths still materialize complete artifacts |
| qml-desktop | Actual native PROJECT PRISM WGPU observatory, real-engine views and historical committed screenshots | Historical render FPS is not model throughput; no new graphical redesign in this reassessment |

## Evidence classes

- Synthetic mathematical/operator/serialization/resource tests establish their stated contracts and fixtures.
- exp-001 through exp-006 contain the existing experimental record, including pretrained-model measurements. Restoration does not rerun or independently reproduce these measurements.
- exp-006 records Qwen2.5-7B-Instruct, Apache-2.0 provenance, four shard hashes, hardware, calibration/evaluation details and negative sub-2-bit results. Its corpus is small; token agreement and perplexity ratios are numerical fidelity measures.
- GPU weight byte accounting and peak streaming-layer VRAM are not whole-process RAM, total GPU allocation or OS-wide peak measurements.
- ATLAS's 1 MiB/32 MiB fixture measures builder-owned buffers and conservative metadata accounting; it does not establish end-to-end bounded-memory quantization.
- No directly measured 14B, 27B or 32B model fit or task quality follows from consumer-hardware estimates or the current synthetic fixtures.

## Next correctness gates before additional real-model claims

1. Reject unsupported HF architecture/positional/activation configurations rather than silently executing a Llama-like interpretation.
2. Distinguish absent optional tensors from malformed tensors; required tensors must fail explicitly.
3. Use a provenance-pinned, appropriately licensed small model and its actual tokenizer/configuration.
4. Compare native execution with an independent reference implementation under matching weights, prompts and precision; record tolerances and unfavorable results.
5. Measure artifact bytes and scoped memory/load/prefill/generation metrics separately; do not extrapolate large-model fit.

This inventory is based on the current crate source, existing regression tests and experiment manifests. Any subsequent experiment must include exact source/config/tokenizer identities and reproducible commands.

