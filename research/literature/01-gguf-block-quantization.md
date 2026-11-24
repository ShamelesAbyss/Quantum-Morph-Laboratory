# GGUF and conventional block quantization

**Index status:** Established prior art; this is not Quantum Morph Laboratory (QML) work. QML has not independently reproduced the cited methods or benchmark results.

## Identification and bibliographic status

GGUF is an open binary model-file format, not a quantization algorithm or a scholarly publication. The upstream specification describes it as a format for storing models for inference with GGML and compatible executors. It packages tensor data with typed metadata, tensor descriptors, offsets and alignment information; quantized tensor encodings are selected separately by tensor type. The specification does not present a research-paper title, author list, DOI or publication date. Hugging Face's documentation credits Georgi Gerganov with developing GGUF, while the upstream document is a mutable project specification with contributions from multiple project participants. [1] [2]

The specific conventional quantizers in scope here are GGML's legacy block types, especially Q4_0 and Q4_1. They are implemented and defined by the GGML project, not introduced by a GGUF paper. The closely related scholarly precedent for standard affine quantization is Jacob et al., **“Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.”** Authors: Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam and Dmitry Kalenichenko. The preprint was submitted 15 December 2017; the paper appeared in the CVPR 2018 proceedings, pp. 2704–2713. DOI: `10.1109/CVPR.2018.00286`; arXiv: `1712.05877`. This paper provides relevant background on affine integer quantization and quantization-aware training, but it is **not** a GGUF or GGML Q4_0/Q4_1 paper. [5] [6]

## Mathematical mechanism

GGUF specifies storage and interpretation conventions: a file header and metadata key/value pairs precede tensor information, and tensor descriptors identify each tensor's dimensions, type and aligned offset into tensor data. The quantization scheme is represented by each tensor's GGML type and its block encoding; `general.quantization_version` records quantization-format versioning separately from the named quantization type. GGUF does not prescribe a unique quantizer or require all tensors in a file to use the same encoding. [1]

For conventional block quantization, a tensor's values are partitioned into small groups. Each group stores packed low-bit integer codes and one or more shared scale parameters. This lowers weight storage while introducing rounding/clipping error; dequantization reconstructs approximate floating-point values before or during matrix-vector operations. The precise mapping depends on the quantizer variant. [2] [3]

In upstream GGML's Q4_0 reference quantizer, each block contains 32 input values. The implementation selects the signed extremum with greatest absolute magnitude and derives a shared signed step `d = max_signed / -8`; it divides values by `d`, shifts into a 4-bit code range, rounds/clamps, and packs two codes per byte. Each block stores a 16-bit scale and 16 packed bytes, or 18 bytes for 32 values (4.5 effective bits per value including scale overhead). Dequantization uses the stored scale and centered 4-bit codes. Q4_0 is therefore a symmetric, scale-only block encoding, not simply “four bits per weight” in total storage. [2] [3]

Q4_1 is an affine/asymmetric counterpart with 32 values per block. It stores 16-bit `d` and 16-bit `m`, where `d = (max-min)/15` and `m=min`; codes approximate `(x-m)/d`, and reconstruction is `x̂ = d·q + m`. A block is 20 bytes (4-bit codes plus two half-precision parameters), or 5 effective bits per value. This is per-block affine quantization. The formula and layout are directly visible in the upstream reference implementation and type definition. [2] [3]

These formulas describe the legacy Q4_0/Q4_1 reference methods, not every encoding in a `.gguf` file. GGML also defines larger “K” superblocks, importance-aware “I” types, and other formats; e.g., Q4_K uses a 256-value superblock containing eight 32-value groups with quantized subscales/minima. The project README's `Q4_K_M` is a mixed tensor-level preset, not a synonym for the simple uniform Q4_0 encoding. [1] [2] [3]

Jacob et al.'s paper uses a different quantization setting: integer values `q` map to real values by an affine mapping of the form `r = S(q-Z)`, with scales/zero points and integer-only arithmetic; the work co-designs quantization-aware training to retain task accuracy. Its paper-level system uses 8-bit weights and activations, with selected bias values at 32 bits, and focuses on CNN/mobile inference. That published method should not be conflated with post-training, low-bit, weight-only GGML Q4 block serialization. [5] [6]

## Reported results and evidence type

**Upstream implementation-maintainer snapshot, not a peer-reviewed paper or independent reproduction.** The current `llama.cpp` quantization README presents a Llama 3.1 8B comparison. It reports `Q4_K_M` at 4.8944 bits/weight and 4.58 GiB, versus F16 at 16.0005 bits/weight and 14.96 GiB. Its table lists prompt processing at 512 tokens as 821.81 ± 21.44 tokens/s for Q4_K_M versus 923.49 ± 0.53 for F16, and text generation at 128 tokens as 71.93 ± 1.52 versus 29.17 ± 0.04 tokens/s. These are maintainers' documented measurements and size figures; they are **not** experimental results for Q4_0/Q4_1 in isolation. The table does not state hardware, runtime/build revision, backend, benchmark command, dataset/prompts, or the meaning of its ± statistic. Treat the speed figures as non-portable and insufficiently specified for independent reproduction. [4]

That README also says quantization can introduce accuracy loss measured by perplexity and/or KL divergence, and that an importance matrix can mitigate it. It warns that requantizing already-quantized weights can severely reduce quality compared with starting from 16-bit or 32-bit tensors. These are maintainer descriptions and operating guidance, not a controlled quality result for the basic block methods. [4]

**Peer-reviewed adjacent result, not a GGUF result.** Jacob et al. compare floating-point and 8-bit integer-only MobileNet-based systems on ImageNet and COCO, using Qualcomm Snapdragon 835 and 821 CPU cores. For a COCO MobileNet SSD setup on a single Snapdragon 835 core, their table reports 100%-depth-multiplier float versus 8-bit results of 22.1 versus 21.7 mAP and 778 versus 687 ms on LITTLE cores, and 370 versus 272 ms on big cores. At 50% depth multiplier, the paper reports 16.7 versus 16.6 mAP and 270 versus 146 ms on LITTLE, 121 versus 61 ms on big. The paper states up to 50% shorter runtime with minimal accuracy loss in that experiment (−1.8% relative, as stated by the authors). This is a trained 8-bit CNN system with specified task and device context, not evidence that GGUF Q4 block quantization achieves the same tradeoff. [5] [6]

## Computational requirements

GGUF as a file format has no intrinsic compute requirement. Requirements depend on the converter, quantizer, inference runtime, model, backend and requested context/batch size. For the current upstream conversion/quantization tool, the maintainers state that models are fully loaded in memory during quantization and that memory and disk requirements are currently the same. Their Llama 3.1 examples list 32.1 GB original and 4.9 GB Q4_K_M output for 8B, 280.9 GB and 43.1 GB for 70B, and 1,625.1 GB and 249.1 GB for 405B. These are documentation examples, not universal minimums or a Q4_0-specific requirement; the documented example invocation uses eight CPU threads. [4]

The GGUF format's memory-mappable layout is a loading feature and should not be read as a guarantee that quantization itself is out-of-core or low-memory. The upstream quantization README explicitly says the current tool fully loads models. [1] [4]

## Limitations

Blockwise shared scales trade storage for quantization error. A block's extrema determine the quantization range in the simple Q4_0/Q4_1 implementations, so outliers can use a disproportionate part of a block's representable range; low bit depth and rounding can degrade model quality. The upstream tool itself identifies accuracy loss and mentions perplexity/KL evaluation and importance-matrix guidance. Actual quality depends on model weights, variant, calibration/importance data, tensor-wise precision choices and runtime kernel behavior. [2] [4]

GGUF is a container/metadata convention; compatible readers must support the tensor types and model architecture they encounter. Its specification says the supported type enumeration is mirrored from the authoritative GGML header and evolves over time. A filename quantization label alone therefore does not fully specify every tensor's actual type or a benchmark result. [1]

There is no controlled GGUF paper experiment with a universal baseline, test suite or hardware configuration. The Q4_K_M README numbers are materially under-specified for replication and cannot substantiate a general speedup or quality claim. [4]

## Licensing considerations

The GGML and `llama.cpp` repositories' current `LICENSE` files state the MIT License for the software. This is the license for those implementations; it does not grant rights to model weights distributed in GGUF. GGUF metadata has fields to record a model's license and source, but each model's publisher sets the applicable terms. Check the base model's source license and any conversion/redistribution conditions before using or redistributing its weights. [7] [8] [1]

The Jacob et al. CVPR page says copyright and related rights remain with the authors or other rights holders and directs users to comply with those terms. The arXiv record lists its article license as a non-exclusive distribution license. Neither cited record establishes a license for GGUF software, a model's weights, or a separate implementation. No code or weights license for that paper was verified here. [5] [6]

## Reputable implementations

- **GGML / `llama.cpp` (`llama-quantize`)** — upstream reference code and canonical practical source for GGUF and GGML quantization types; the maintained projects publish the format specification, reference encoders and quantization tooling. The repositories state MIT licensing for their software. [1] [2] [4] [7] [8]
- **Hugging Face Hub GGUF support and `@huggingface/gguf` parser** — reputable ecosystem support for inspecting GGUF metadata/tensor information and parsing files. It is not the authoritative quantizer specification. Its documentation links to its package/repository but the license of that parser package was not verified in this review; consult that repository's current license before reuse. [9]

## Relevance to QML and overlap assessment

This is established prior art relevant to QML as a baseline family for neural-weight compression and efficient inference, plus a mature model-file organization format. It suggests useful comparison axes for a local-first Rust research lab—effective bits/value including metadata, reconstruction error, task/perplexity impact, encode/decode cost, throughput, memory use and interoperability. No claim is made that GGUF or conventional block quantization originated in QML.

**Potential overlap:** the established shared idea is local grouping of weights with compact codes and block-level scale/offset metadata; that is generic shared mechanism, not evidence of method identity. Whether QML's RMQ, MAP, QMF or any other named methods overlap more deeply cannot be established from this record because their definitions, papers and internal implementation details were not provided or verified. Any relationship beyond the generic blockwise-quantization analogy is a **QML hypothesis**, not a finding of novelty or prior-art identity.

## Evidence-status labels

- **Established published method / prior art:** GGUF as an established upstream format; GGML Q4_0/Q4_1 as implemented blockwise encodings; affine integer quantization and quantization-aware training as a published method in Jacob et al.
- **Independently reproduced by QML:** No. QML has not reproduced this work.
- **QML hypothesis:** Any deeper conceptual or implementation overlap between Q4 blocks and RMQ/MAP/QMF beyond shared blockwise quantization structure.
- **Unknown:** A formal GGUF research paper/publication date, project-wide individual authorship list, parser-package license, and hardware/test harness behind the README performance table.

## References

[1]: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md "GGUF format specification"
[2]: https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c "GGML reference quantization implementations"
[3]: https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h "GGML quantization block data structures"
[4]: https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md "llama.cpp quantization guide and maintainer-reported comparisons"
[5]: https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html "Jacob et al. CVPR 2018 paper record"
[6]: https://arxiv.org/abs/1712.05877 "Jacob et al. arXiv preprint and record"
[7]: https://github.com/ggml-org/ggml/blob/master/LICENSE "GGML software license"
[8]: https://github.com/ggml-org/llama.cpp/blob/master/LICENSE "llama.cpp software license"
[9]: https://huggingface.co/docs/hub/en/gguf "Hugging Face GGUF documentation"
