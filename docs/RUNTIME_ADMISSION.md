# SENTINEL runtime admission and optional tensor contract

This describes the v0.6.4 supported decoder subset. It is an input-correctness contract, not an additional architecture implementation or a pretrained-model quality result.

## Hugging Face configuration admission

The parser requires a recognized identity: model_type llama or qwen2, or the corresponding LlamaForCausalLM / Qwen2ForCausalLM architecture. When both are present they must agree. An empty architecture list is harmless when model_type supplies identity. Other families, non-causal heads, conflicting identities and unidentified dimension-only configurations fail explicitly.

The existing decoder implements SiLU/SwiGLU and full-head, split-half, unscaled RoPE. SiLU's swish alias is accepted. An explicit head_dim or rotary_dim must equal hidden_size / num_attention_heads. Hidden size must divide evenly into heads; query heads must divide evenly into KV-head groups; all dimensions must be nonzero and the rotary head dimension even.

Legacy rope_scaling and modern rope_parameters are examined. Null/empty/default unscaled settings are admitted, including identity factors and linear factor exactly one. Conflicting type aliases or theta values, non-identity scaling, partial/interleaved rotation, non-unit attention factors, layer-specific RoPE dictionaries and unknown rotary parameters fail explicitly. This does not claim support for scaled Llama-3.1/YaRN/long-context variants merely because their family name is Llama.

Qwen2 sliding-window controls are evaluated for actual layer use. A disabled use_sliding_window flag leaves nominal sliding_window and max_window_layers metadata inactive. Explicit null windows disable windowing; a start layer beyond the model's layers affects no layer. An explicit, correctly sized all-full_attention layer_types list is supported. Any active sliding_attention or other hybrid layer type is rejected. Turning a flag off does not override explicitly active layer_types.

RMSNorm epsilon and RoPE theta must be positive and finite in CPU and GPU precision. The maximum rotary frequency/angle must remain finite in the GPU precision used by the existing kernels. Scalar dimensions must fit the existing u32 shader interface. Checked products protect embedding, projection, sequence, trace, KV-cache and layer-directory byte spans and operator counts against usize/isize overflow before allocation. Runtime layer directories grow only as real layers load, avoiding an enormous initial reservation from an unbacked declared count. These checks establish representability; they are not a RAM reservation or a claim that every admitted model fits available hardware.

Omitted family defaults follow the supported HF configurations: RMSNorm epsilon 1e-6 and theta 10000 for both families; Llama context 2048 and Qwen2 context 32768. Explicit Qwen2.5 checkpoint theta 1000000 remains unchanged when supplied. Missing/null KV-head counts use query-head counts. Explicit supported values are preserved. Earlier generic defaults incorrectly used context 2048 for Qwen2 and epsilon 1e-5 when fields were omitted.

The parser accepts at most 1 MiB of configuration JSON. Root controls and nested rotary fields reject duplicate keys, preventing an active setting from being silently replaced by a later neutral one. Recognized checkpoint identity, initialization, dtype, token-ID and interface/backend metadata do not select new runtime mathematics. Dropout rates are validated but are inactive during inference. Native backend selection remains explicit, and use_cache metadata does not change the mathematical causal decoder. Nonempty custom-code maps, pre-quantized loader configurations, cross-attention, Q/K normalization, ALiBi, soft-capping, output-attention bias layouts or MLP biases are not silently substituted. Standard optional Qwen2 q/k/v biases remain supported. Unclassified fields fail with a named error rather than being guessed harmless.

Primary semantic references: [HF Llama defaults](https://github.com/huggingface/transformers/blob/v4.43.1/src/transformers/models/llama/configuration_llama.py), [HF Qwen2 defaults](https://github.com/huggingface/transformers/blob/v4.43.1/src/transformers/models/qwen2/configuration_qwen2.py), [HF Qwen2 layer-type controls](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/configuration_qwen2.py), and [HF rotary parameter definitions](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py). These references explain the admitted semantics; HF library execution is not a runtime dependency.

## Optional q/k/v attention biases

All three loaders use the same distinction:

- No directory entry: the optional bias remains absent.
- An existing entry: require exactly one dimension, with query length hidden_size and key/value length num_key_value_heads × head_dim.
- For QMF, check the dense vector encoding and exact payload byte length before reading/decoding; malformed lengths cannot force a large irrelevant allocation.
- Decode through the existing bounded reader, propagate read/CRC/representation failures with the tensor name, then require the exact decoded length and finite values. Safetensors metadata already validates its shape/dtype byte lengths.

The contract applies before execution in in-memory QMF, file-backed QMF and streaming BF16 calibration/reference. A failed load is never reinterpreted as absence. It protects CPU, GpuHybrid and GpuFull loading through their shared model construction paths; no backend, shader or quantization mathematics was replaced.

AEGIS's untied output projection remains required. Missing or corrupt lm_head weights still produce errors, while explicitly tied embeddings retain their existing behavior.

## Runtime cache migration

The CLI's existing reference/calibration cache identity now includes a Q-Core runtime-semantics domain and serialized effective validated configuration. Raw configuration bytes alone cannot distinguish corrected omitted-field defaults or earlier ignored bias failures. Existing cache formats, checksum/provenance enforcement and publication algorithms are unchanged. Older cache files remain intact, but their provenance no longer matches SENTINEL; regenerate into a new output directory instead of reusing or deleting historical evidence. This is a runtime correctness migration, not a cache format redesign.

## Evidence and limits

Nineteen focused synthetic regression tests cover supported family defaults and representative metadata layouts; malformed, conflicting and unknown architecture controls; activations; attention geometry; active/inactive window settings; rotary variants; finite constants; dimension and input bounds; unbacked layer-count allocation; runtime identity stability/settings sensitivity; absent/valid biases; wrong ranks/lengths; NaN and infinities in F32/F16/BF16; decoded-length mismatches; CRC/I/O errors and a source truncated after streaming metadata admission. Positive q/k/v cases affect multi-token logits and preserve memory/file CPU equality. Existing required-output-projection and GPU/CPU parity tests are retained.

Synthetic metadata with realistic dimensions allocates no model weights and establishes no large-model fit claim. Tests use owned temporary fixtures, not existing model files. Safetensors has no embedded checksum: a finite, structurally valid changed value cannot be identified as corruption without an independently expected source identity. QMF CRC checks detect accidental corruption, not producer authenticity. No new model download, pretrained-model inference comparison, throughput/memory benchmark or graphical benchmark is part of SENTINEL. Stop after its verified release.
