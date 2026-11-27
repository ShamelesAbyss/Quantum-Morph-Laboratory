# CONTINUUM baseline identity audit — November 27, 2026

This is a read-only provenance audit of the existing owner-provided Qwen2.5-7B-Instruct cache. No weights were downloaded, modified, transformed, committed or newly benchmarked. Historical exp-006 measurements remain unchanged and have not been independently reproduced by this audit.

The upstream repository identifies the model as Apache-2.0 licensed. The four cached shard SHA-256 values and byte lengths match both the historical experiment and the pinned upstream file metadata. Configuration, tokenizer, tokenizer configuration and shard-index files match upstream Git blob identities at revision a09a35458c702b33eeacc393d103063234e8bc28. Their independently calculated local SHA-256 values are recorded below.

Sources: [pinned model repository](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28), [pinned license](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/LICENSE), [historical experiment](../experiments/exp-006-event-horizon-wave3-7b-sub2bit.toml). These are research provenance references, not QML software distribution destinations.

| File | Bytes | SHA-256 |
|---|---:|---|
| model-00001-of-00004.safetensors | 3,945,441,440 | a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7 |
| model-00002-of-00004.safetensors | 3,864,726,352 | f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185 |
| model-00003-of-00004.safetensors | 3,864,726,424 | 8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5 |
| model-00004-of-00004.safetensors | 3,556,377,672 | 1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd |
| config.json | 663 | 7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c |
| tokenizer.json | 7,031,645 | c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 |
| tokenizer_config.json | 7,305 | 5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583 |
| model.safetensors.index.json | 27,752 | 624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028 |

Reproduction: hash each immutable source file with SHA-256; compare shard byte lengths/hashes with exp-006 and the pinned upstream LFS metadata; compare the four non-weight files using git hash-object against upstream blob IDs at the pinned revision. A match establishes file identity with the recorded checkpoint, not runtime correctness or task quality.

Native qml 0.6.3 inspect-safetensors --json passed separately for all four shards: 339 unique BF16 tensors, 15,231,233,024 payload bytes, 38,832 header bytes, and four eight-byte header-length fields, totaling 15,231,271,888 source-file bytes. This agrees with the sum of the four historical shard lengths and the recorded 7,615,616,512 parameters. The historical bf16_total_bytes field instead says 15,230,871,888, a 400,000-byte discrepancy. That historical field is preserved verbatim; subsequent byte-accounting work must use the measured sum and record the discrepancy rather than silently altering the experiment. This metadata audit is not an inference or memory benchmark.

The inspect-safetensors command currently accepts individual Safetensors files, not the shard-index JSON. Passing the index produced an explicit bounded-header error; inspecting its four backing shards succeeded. Sharded source loading in Forge/runtime is a separate API capability.

All 339 inspected names, BF16 dtypes, tensor shapes and shard-index assignments match this checkpoint's Qwen2 configuration: embedding and untied output matrices, final RMSNorm, and 28 blocks containing two norm vectors, seven projection matrices and three q/k/v bias vectors each. This is source metadata consistency, not validation of numerical tensor contents or end-to-end execution.

The actual configuration names Qwen2ForCausalLM and qwen2, uses SiLU, 28 layers, hidden size 3584, intermediate size 18944, 28 attention heads, four KV heads, head dimension 128, vocabulary 152064, RMSNorm epsilon 1e-6 and RoPE theta 1000000. Embeddings are untied. Sliding-window attention is disabled despite a nonzero sliding_window metadata value; admission must inspect the active control rather than rejecting that inactive value alone.

The next correctness milestone must establish supported architecture/activation/positional admission, finite positive numerical constants, checked dimension products and explicit errors for present-but-malformed optional bias tensors in every loader. A genuinely absent optional bias remains different from a corrupt present tensor. Supported existing configurations must retain their execution semantics.

Independent native reference comparison remains pending. The installed Ollama CLI did not provide a functioning reference server in this session, and it was not repaired or replaced as part of repository restoration. No agreement, perplexity, latency, peak RAM/VRAM or model-quality result is inferred from provenance. Small-fixture parity, historical experiments and future independent pretrained-model comparisons remain separate evidence classes.
