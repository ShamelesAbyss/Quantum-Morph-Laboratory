# AQLM — Additive Quantization for Language Models

**Prior-art classification:** AQLM is an established published method by Egiazarian et al. (ICML 2024), not work originating in Quantum Morph Laboratory (QML). QML has not reproduced it. The paper extends prior additive-quantization and multi-codebook-quantization research to post-training compression of language-model weights. [1] [2]

## Bibliographic record

The paper is titled **“Extreme Compression of Large Language Models via Additive Quantization.”** Its authors are Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, and Dan Alistarh. It appeared in the *Proceedings of the 41st International Conference on Machine Learning*, Proceedings of Machine Learning Research (PMLR), volume 235, pages 12284–12303, in 2024; the conference ran July 21–27, 2024. The arXiv preprint was submitted January 11, 2024, and the current arXiv version is v4, revised September 11, 2024. [1] [2]

Canonical sources are the [PMLR proceedings record](https://proceedings.mlr.press/v235/egiazarian24a.html), [arXiv record](https://arxiv.org/abs/2401.06118), and [arXiv full text](https://arxiv.org/html/2401.06118v4). Identifier: arXiv:2401.06118; arXiv/DataCite DOI: [10.48550/arXiv.2401.06118](https://doi.org/10.48550/arXiv.2401.06118). The PMLR record does not list a separate proceedings DOI. [1] [2]

## Mathematical mechanism

AQLM represents each group of weights as a sum of codewords selected from multiple learned codebooks. For a group size of *g*, with *M* codebooks each containing 2^*B* candidate vectors, the stored representation selects one *B*-bit index per codebook and sums those *M* vectors. Thus the group uses *M·B* index bits before amortized codebook and other format overhead. [3]

Rather than only minimizing direct weight reconstruction error, AQLM minimizes error in the layer output on calibration inputs. In the paper’s notation, its objective is approximately

`min_(C,b) ||W X − W_hat(C,b) X||²_F`,

where *W* is the original matrix, *X* contains representative input activations, *C* denotes codebooks, and *b* denotes discrete code assignments. [3]

The procedure initializes codebooks through residual K-means. It updates discrete assignments using beam search, formulated as a fully connected discrete Markov-random-field optimization problem, and optimizes codebooks and per-output-unit scales with gradient-based updates (the described implementation uses Adam). It also jointly tunes codebooks across transformer blocks against calibration activations. This is input-aware, multi-codebook additive quantization, not a quantum-computing method. [3]

## Published experimental results

The following are results reported by the paper’s authors, not independent replications. The main Llama 2 comparisons evaluate WikiText-2 and C4 perplexity (lower is better) and average accuracy on five zero-shot tasks. Calibration uses RedPajama-v1; Llama 2 uses 4096-token context. The paper reports average bit rates for quantized weights and excludes floating-point parameters from that rate, so those bit rates are not complete model-file sizes. [3]

At approximately 2 bits per parameter, the paper reports for Llama 2 7B: AQLM at 2.02 bits, WikiText-2 PPL 6.59 and C4 PPL 8.54, compared with QuIP# at 2.02 bits, 8.22 and 11.01. For Llama 2 70B, AQLM at 2.07 bits reports 3.94 and 5.72, compared with QuIP# at 2.01 bits, 4.16 and 6.01. These are the paper’s test-set metrics under its evaluation setup, not guarantees for other models or evaluation pipelines. [3]

At approximately 3 bits, Llama 2 7B AQLM at 3.04 bits reports 5.46 WikiText-2 and 7.08 C4 PPL, compared with GPTQ at 3.00 bits (8.06 and 10.61) and SpQR at 2.98 bits (6.20 and 8.20). For Llama 2 70B, the reported AQLM 3.01-bit scores are 3.36 and 5.17; the paper lists GPTQ (3.00 bits) at 4.40 and 6.26, SpQR (2.98 bits) at 3.85 and 5.63, and QuIP (3.01 bits) at 3.87 and 5.67. [3]

The paper also benchmarks inference on an Nvidia RTX 3090 and an Intel Core i9-13900K. For one 2-bit AQLM configuration, reported matrix-vector-multiplication speedups over FP16 on the RTX 3090 were 1.31×, 1.20×, and 1.20× for Llama 2 7B, 13B, and 70B; a 2×8-bit-codebook setup reached 1.57×, 1.82×, and 3.05×. On an 8-core i9-13900K, reported 2×8-bit-codebook speedups over FP32 were 2.75×, 3.54×, and 3.69×, respectively. These are kernel/layer benchmarks under the paper’s configurations, not general end-to-end speed guarantees. [3]

## Compute, limitations, and licensing

The authors say their experiments used Nvidia A100 or H100 GPUs, with 1–8 GPUs and activation offloading. The inference tests used the RTX 3090 and i9-13900K noted above. The paper does not establish a universal minimum hardware requirement or report one reproducible wall-clock budget for quantizing every model. It characterizes AQLM as more computationally expensive than direct PTQ methods such as RTN or GPTQ because its code representation and optimization are more complex. Results also depend on calibration data and configuration; the paper’s Llama 2 7B ablation shows PPL improving as calibration samples rise from 128 to 4096, with diminishing returns. [3]

The official [AQLM PyTorch repository](https://github.com/vahe1994/AQLM) states that it implements the paper and offers GPU/CPU inference paths. Its checked-in software license is Apache License 2.0. [4] [5] The paper itself is distributed on arXiv under the arXiv perpetual non-exclusive distribution license; that should not be mistaken for an open-source software license or a general CC reuse license. The PMLR record identifies the proceedings work as copyright of the authors and PMLR. Model-weight rights are separate: no blanket AQLM weight license was verified. A quantized checkpoint inherits relevant restrictions from its specific base model and any checkpoint-specific terms, which must be checked on that model’s own card. [1] [2]

## Relevance and possible overlap with QML

**Established published method:** AQLM is prior art for learned additive/product-style multi-codebook representation of neural weights, calibration-input-aware reconstruction, and discrete code assignment with learned codebooks. Its reported targets include the extreme 2–3-bit/parameter regime. It is directly relevant to a Rust research lab studying model representation and compression as a comparison baseline or literature precedent, but the paper does not establish that any QML implementation exists. [1] [3]

**QML hypothesis / unknown:** Potential overlap with QML’s RMQ, MAP, and QMF cannot be established from those project abbreviations alone. AQLM explicitly casts its code-assignment stage as MAP inference in a Markov random field, so a QML method independently defined around MAP/MRF code inference may share a mathematical subproblem; an acronym match alone is not evidence of method identity or derivation. AQLM also uses multiple additive codebooks and residual K-means initialization, which could be conceptually adjacent to a QML residual or multi-codebook quantizer, if RMQ means that. No specific connection to QMF is verified. Compare actual representation, objective, optimization and inference equations before making stronger overlap claims. [3]

**Independent reproduction:** None by QML; QML has not reproduced AQLM. Software-maintainer README metrics and downloadable quantized checkpoints are maintainer claims/artifacts, not independent reproduction, and are not used here as evidence for the paper’s core results. [4]

## References

[1]: https://proceedings.mlr.press/v235/egiazarian24a.html "ICML 2024 PMLR proceedings record for AQLM"
[2]: https://arxiv.org/abs/2401.06118 "arXiv record for AQLM"
[3]: https://arxiv.org/html/2401.06118v4 "AQLM paper full text, version 4"
[4]: https://github.com/vahe1994/AQLM "Official AQLM PyTorch repository"
[5]: https://github.com/vahe1994/AQLM/blob/main/LICENSE "AQLM repository Apache License 2.0"
