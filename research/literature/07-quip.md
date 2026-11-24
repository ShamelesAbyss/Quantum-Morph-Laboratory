# QuIP and QuIP#: prior art for low-bit neural-model quantization

**Classification:** Established published methods. This note records prior art, not work by Quantum Morph Laboratory (QML). QML has not reproduced either method. All experimental results below are attributed to the originating papers unless expressly identified as an implementation-maintainer statement; no independent reproduction was verified for this note.

## Identification

**QuIP** means *Quantization with Incoherence Processing*. The paper is *QuIP: 2-Bit Quantization of Large Language Models With Guarantees*, by Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher De Sa. It first appeared on arXiv on 25 July 2023 (arXiv:2307.13304; DOI 10.48550/arXiv.2307.13304) and was published in *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. The proceedings record lists DOI 10.52202/075280-0196. [1] [2]

**QuIP#** means *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks*, by Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. Its arXiv v1 was submitted 6 February 2024 (arXiv:2402.04396; DOI 10.48550/arXiv.2402.04396); the reviewed v2 is dated 4 June 2024. It appeared in the *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, pages 48630–48656, 2024. [3] [4]

## Mathematical principles

Both methods are post-training, weight-only quantizers. They use a per-layer quadratic proxy loss for weight-rounding error,

`ℓ(Ŵ) = tr((Ŵ − W) H (Ŵ − W)ᵀ) = Eₓ[||(Ŵ − W)x||²]`,

where `H = Eₓ[xxᵀ]` is estimated from calibration activations. Adaptive rounding feeds prior rounding error back into subsequent weight decisions. In QuIP, LDLQ chooses the linear feedback from an LDL factorization of `H`; the paper proves worst- and average-case optimality within its specified class of adaptive linear-feedback rounding procedures (for its stated setting). [1] [5]

QuIP’s second component, “incoherence processing,” applies structured random orthogonal transforms before quantization and their inverse/post-processing afterward. This spreads unusually large weight entries and important Hessian directions across coordinates, while preserving the proxy objective under orthogonal transforms. The original method uses structured Kronecker products of random orthogonal matrices. Its core premise is that adaptive rounding works better when both weights and the proxy Hessian are incoherent with the coordinate basis. [1] [5]

QuIP# retains the proxy-loss/adaptive-rounding framework but updates the processing and codebook. It uses randomized Hadamard transforms (RHTs)—Hadamard transforms with random sign modulation—to achieve incoherence with a fast Walsh–Hadamard transform. For a dimension `n` that is a power of two this is `O(n log n)`; the paper describes structured factorizations for other dimensions and a randomized Fourier-transform fallback where needed. It extends LDLQ to blockwise adaptive rounding (BlockLDLQ) so vectors can be quantized together. Its 8-dimensional E8P codebook is shaped for the approximately ball-shaped distribution expected after incoherence processing; residual vector quantization (RVQ) builds 3- and 4-bit codebooks from lower-rate codebooks. The full QuIP# method also fine-tunes selected unquantized parameters and transform sign vectors on a small development set. [3] [6]

## Paper-reported experimental results

These are author-reported results, not QML results or independent reproductions.

**QuIP.** The experiments quantized OPT-family models up to 66B parameters and Llama 2 70B. For OPT, calibration used 128 random 2,048-token C4 segments; evaluation included WikiText2, PTB, C4, and zero-shot tasks. The authors report quantization on one GPU with up to 48 GB memory, one transformer block at a time. On OPT-30B, the paper’s table shows QuIP at 2 bits with WikiText2 perplexity 6.326 and C4 perplexity 8.937, compared with OPTQ at 2 bits with 123.908 and 70.541 respectively; lower perplexity is better. On Llama 2 70B, 2-bit QuIP reported WikiText2/PTB/C4 perplexities of 11.48/17.40/13.55 and ARC Easy/LAMBADA accuracies of 57.87/65.24, versus the corresponding 16-bit values 9.56/14.04/11.45 and 65.40/72.40. The 2-bit OPTQ comparator on Llama 2 70B had perplexities 71.70/88.19/29.59 and ARC Easy/LAMBADA 42.47/25.77. The authors further report that the 2-bit gap narrows with larger OPT models. These results support the paper’s claim that incoherence processing made 2-bit PTQ viable; they do not imply lossless compression or equivalence to full precision. [5]

**QuIP#.** The main experiments cover Llama 1 and Llama 2 from 7B to 70B, with additional reported tests on Mixtral 8x7B and Falcon 180B. The paper compares against FP16, AWQ, OmniQuant, AQLM, QuIP, and ablations. At context length 2,048, Llama 2 70B at 2 bits has reported WikiText2/C4 perplexities of 4.16/6.12 for full QuIP#; the FP16 baseline is 3.32/5.52, while the paper-reported OmniQuant 2-bit row is 7.81/12.3. At context length 4,096, QuIP# 2-bit Llama 2 70B reports 3.91/5.71, versus FP16 3.12/4.97 and AQLM at 2.07 effective bits, 3.94/5.72. These rows use different conditions and should not be conflated. The paper notes that AWQ/OmniQuant and AQLM comparison numbers were drawn from their respective papers, and that the AQLM paper’s QuIP# numbers used an older QuIP# version. [6]

For throughput, the paper reports QuIP#-2-bit decoding of Llama 2 7B/13B/70B at 170.50/104.83/32.74 tokens/s on an RTX 4090 using FlashAttention’s Llama implementation, reaching 29.60%/33.80%/56.84% of peak memory bandwidth. Its separate Hugging Face Llama implementation table reports 106.3/25.9 tokens/s for Llama 2 7B/70B at 2 bits; the FP16 70B run was out of memory (OOM) in that specific comparison. These are paper benchmarks on the stated GPU and software paths, not general guarantees. The authors also report 2-bit results on Mixtral 8x7B (WikiText2/C4 4.69/8.25 versus FP16 3.45/6.85) and Falcon 180B (4.18/7.06 versus 3.30/6.31), without fine-tuning. [6]

## Computational requirements and limitations

For QuIP, the paper states single-GPU quantization with up to 48 GB of memory and blockwise processing, but those figures describe the authors’ experiments, not a universal minimum requirement. [5]

For QuIP#, the authors say experiments ran on NVIDIA A100 GPUs, except inference timing on an RTX 4090. Hessian generation used 6,144 sequences from RedPajama 1T at the model’s native context length (2,048 for Llama 1, 4,096 for Llama 2). The paper reports under 10 GPU-hours to quantize Llama 2 70B without fine-tuning and around 100 GPU-hours with fine-tuning, excluding Hessian generation. A methods discussion separately estimates the fine-tuning component at around 50 GPU-hours for a 70B model; treat that as a separate approximate estimate rather than add it to or substitute it for the appendix’s full quantization estimate. The paper does not establish a universal minimum GPU memory or CPU-memory requirement. The official repository calls for CUDA kernels and the `fast-hadamard-transform` package; Hessian construction uses an FP64 accumulator by default and may run slower on hardware with weak FP64 throughput. [6] [7]

Method limitations and practical caveats include dependence on calibration activations/Hessian estimates; the proxy loss is not the same as end-task quality; results depend on model, context length, codebook, scale, fine-tuning, evaluation pipeline, and effective bits; and QuIP#’s fast path imposes design tradeoffs. The paper says E8 codebooks become intractable at higher bitrates, motivating RVQ. Its RHT has dimension/factorization constraints and uses an RFFT fallback in some cases, which the authors report performs slightly worse but remains effective. The repository is Llama-script-oriented and says other architectures require adapting the scripts. Its README says it is no longer under active development and notes that an optimized 1-bit matrix-vector kernel needed to make 3-bit inference fast is missing; its simple Hugging Face generation path also has kernel-launch/CUDA-graph caveats. [6] [7]

## Licensing and implementations

**Papers and code are distinct.** The QuIP arXiv record lists the arXiv non-exclusive distribution license; this does not by itself license the implementation or model checkpoints. The official QuIP repository page does not identify a software license, and its root has no verified `LICENSE` file; treat code reuse terms as unknown unless clarified by the authors. QuIP#’s arXiv record lists CC BY 4.0 for the paper. Its official implementation explicitly states GNU GPL v3. [2] [3] [7] [8]

**Weights are separate.** QuIP#’s repository says Llama weights are governed by Meta’s Llama license and Mistral weights by Apache 2.0. These statements do not settle rights for every third-party base model or every quantized checkpoint. The repository links prequantized checkpoints but does not establish a single blanket license for them; check each checkpoint and its base-model terms. No separate license for all QuIP/QuIP# model weights was verified for this note. [7]

- **Official QuIP reference code:** [Cornell-RelaxML/quip](https://github.com/Cornell-RelaxML/quip). Author-linked implementation for QuIP. **License: not stated / not verified**; do not assume permissive rights from repository availability. [8]
- **Official QuIP# reference code and CUDA kernels:** [Cornell-RelaxML/quip-sharp](https://github.com/Cornell-RelaxML/quip-sharp). Quantization, evaluation, and inference implementation. **License: GPL-3.0.** The maintainers say it is no longer actively developed. [7]
- **Third-party QuIP-for-all:** [chu-tianxiang/QuIP-for-all](https://github.com/chu-tianxiang/QuIP-for-all). README describes it as an adaptation of the official QuIP# repository for more architectures; repository reports **GPL-3.0**. The official QuIP# maintainers explicitly say they have not verified this implementation’s correctness, so it is an implementation lead, not independent validation. Its own README claims default calibration can consume 500–750 GB CPU memory; that is the third-party maintainer’s estimate and was not independently checked. [7] [9]

## Relevance to QML and overlap hypotheses

QuIP and QuIP# are directly relevant prior art for neural-model representation and compression because they combine second-order-informed adaptive rounding with transforms intended to redistribute weight and Hessian energy, and QuIP# adds structured vector codebooks and post-quantization fitting. Their published mechanisms establish that incoherence transforms, LDL/BlockLDL adaptive error feedback, and E8-based vector quantization are already known approaches in low-bit LLM PTQ. This note does not attribute them to QML.

Potential overlap with **RMQ, MAP, QMF, or other QML-labeled methods is a QML hypothesis, not an established fact**: the present research assignment supplies the names but not their formal definitions, equations, code, or experiment records. At a broad conceptual level, any QML method involving basis changes, error-feedback rounding, vector/lattice codebooks, or low-bit weight approximation could warrant a feature-by-feature comparison with QuIP/QuIP#. No conclusion about technical equivalence, derivation, or priority overlap with RMQ/MAP/QMF can be made from names alone. Such overlap is **unknown** pending comparison against QML’s own method documents. QML reproduction status is **not independently reproduced by QML**.

## References

[1]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/0df38cd13520747e1e64e5b123a78ef8-Abstract-Conference.html "QuIP NeurIPS 2023 proceedings record"
[2]: https://arxiv.org/abs/2307.13304 "QuIP arXiv record"
[3]: https://arxiv.org/abs/2402.04396 "QuIP# arXiv record"
[4]: https://proceedings.mlr.press/v235/tseng24a.html "QuIP# ICML 2024 proceedings record"
[5]: https://arxiv.org/html/2307.13304v2 "QuIP paper full text, version 2"
[6]: https://arxiv.org/html/2402.04396v2 "QuIP# paper full text, version 2"
[7]: https://github.com/Cornell-RelaxML/quip-sharp "Official QuIP# repository and README"
[8]: https://github.com/Cornell-RelaxML/quip "Official QuIP repository and README"
[9]: https://github.com/chu-tianxiang/QuIP-for-all "Third-party QuIP-for-all repository"
