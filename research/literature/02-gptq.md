# GPTQ post-training quantization

**Prior-art classification.** GPTQ is an established, published post-training quantization method by Frantar, Ashkboos, Hoefler, and Alistarh. It is prior art, not work by Quantum Morph Laboratory (QML). QML has not independently reproduced GPTQ; the reported measurements below are the paper authors’ results, not QML measurements. [1] [2] [3]

## Bibliographic record

The arXiv paper is titled *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. The ICLR 2023 conference record at IST Austria lists the same work under the title *OPTQ: Accurate post-training quantization for generative pre-trained transformers*; these are title variants for the same paper, not separate methods. The authors are Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh (rendered as Dan-Adrian Alistarh in the ISTA record). [1] [2] [3]

The preprint was first submitted on 31 October 2022 and revised on 22 March 2023. It was published at the 11th International Conference on Learning Representations (ICLR 2023), held 1–5 May 2023. The arXiv identifier is **2210.17323** and its DOI is **10.48550/arXiv.2210.17323**. I found no distinct DOI for the ICLR proceedings record in the publication record inspected. [2] [3]

## Mathematical principles

GPTQ addresses each linear layer using a calibration-input reconstruction objective. With layer weights \(W\) and calibration activations \(X\), it seeks quantized weights \(\widehat W\) minimizing the squared layer-output error, \(\|WX-\widehat W X\|_F^2\). This makes the input covariance, rather than unweighted weight error alone, relevant to how harmful a rounding choice is. [1]

The method builds on Optimal Brain Quantization (OBQ), itself an approximate second-order/Optimal Brain Surgeon-style compensation method. The layerwise quadratic has Hessian \(H=2XX^\top\) over the remaining unquantized coordinates, with diagonal damping used for numerical stability. For a coordinate \(q\), OBQ’s greedy criterion selects the smallest estimated increase in loss, proportional to \((\operatorname{quant}(w_q)-w_q)^2/[H^{-1}]_{qq}\), then compensates the remaining weights using the corresponding inverse-Hessian column. [1]

GPTQ’s scaling change is to use a common fixed column order across weight rows instead of independently choosing the next weight per row. This allows sharing the inverse-Hessian update across rows. The implementation batches updates lazily (paper default block size \(B=128\)) and uses a Cholesky reformulation of inverse-Hessian information to improve numerical stability. The paper’s standard setup uses uniform, asymmetric per-row quantization; groupwise grids and activation-order options are compatible refinements, not the core second-order compensation principle. [1] [4]

## Paper-reported experimental results

The authors implemented GPTQ in PyTorch and evaluated OPT and BLOOM model families. Their stated calibration set was 128 random 2,048-token segments from C4. They processed one Transformer block at a time on a single 80-GB NVIDIA A100. Their primary baseline, round-to-nearest (RTN), used the same asymmetric per-row grid without GPTQ’s error compensation. [1] [4]

For OPT-175B on WikiText-2, perplexity (lower is better) was 8.34 at FP16, 8.37 with 4-bit GPTQ, and 10.54 with 4-bit RTN; 3-bit GPTQ scored 8.68 while 3-bit RTN collapsed to 7.3×10³. Across WikiText-2, PTB, and C4, the paper’s OPT-175B FP16 perplexities were 8.34 / 12.01 / 10.13, versus 8.37 / 12.26 / 10.28 at 4-bit GPTQ and 8.68 / 12.68 / 10.67 at 3-bit GPTQ. The same table reports BLOOM-176B at 8.11 / 14.59 / 11.71 in FP16; 8.21 / 14.75 / 11.81 at 4-bit GPTQ; and 8.64 / 15.57 / 12.27 at 3-bit GPTQ. RTN’s 3-bit WikiText-2 perplexity was 571 for BLOOM-176B. These are the paper’s reported evaluations, not independent reproductions. [1]

The paper reports full-model quantization times on one NVIDIA A100: 4.2 hours for OPT-175B and 3.8 hours for BLOOM-176B. Its inference-speed result is specifically for 3-bit OPT-175B, batch size 1, generating sequences of length 128: average per-token latency was 230 ms FP16 on five A100 80-GB GPUs versus 71 ms quantized on one A100 (3.24×); on A6000 48-GB GPUs, it was 589 ms FP16 on eight GPUs versus 130 ms quantized on two (4.53×). The paper attributes most of this speedup to bespoke kernels reducing memory traffic; it is not a claim that the quantization itself reduces the arithmetic count. [1]

The C4 calibration sample is generic text rather than task-specific data, but the paper’s appendix explicitly notes that C4 evaluation is not fully zero-shot because calibration segments were sampled from the C4 training set. [1]

## Compute and implementation conditions

The authors report executing quantization of even the 175B-scale models on one 80-GB A100, using the 128 × 2,048-token C4 calibration segments and blockwise model loading. These are reported experimental conditions, not a universal minimum hardware requirement or a guarantee for other model architectures, software stacks, or implementations. The author repository lists PyTorch and Transformers dependencies and says its 3-bit CUDA extensions were tested with CUDA 11.4; these are repository-era instructions rather than current universal requirements. [1] [4]

The paper’s stated limitations are that its speedups derive from reduced memory movement and do not reduce the multiplication count; its experiments focus on weight-only quantization and do not quantize activations. It also documents numerical-instability risk in repeated inverse-Hessian updates at scale, motivating damping and the Cholesky reformulation. Accuracy depends on bit width, model, quantization grid/group size, and calibration setup; the OPT-66B model is called out as an outlier. [1]

## Licensing and implementations

The arXiv record displays a CC BY 4.0 license for the preprint. The authors’ reference repository, `IST-DASLab/gptq`, declares Apache-2.0. Those licenses apply to the identified paper and software artifacts, respectively; they do not themselves grant rights to upstream model weights. In the paper’s examples, OPT checkpoint terms are governed by Meta’s OPT-175B license, which grants limited non-commercial research rights and includes use restrictions; the BigScience BLOOM model card identifies the BigScience BLOOM RAIL License v1.0. A GPTQ-converted checkpoint should therefore be assessed under its originating model’s terms as well as any applicable software terms. This is a source summary, not legal advice. [2] [4] [8] [9]

| Implementation | Role and evidence |
|---|---|
| [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq) | Authors’ reference implementation for the ICLR paper; includes quantization scripts and CUDA kernels. Repository license: Apache-2.0. Its README documents the tested software versions and reports author-run experiment conditions. [4] |
| [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel) | A later, maintained GPTQ toolkit; Hugging Face Transformers currently identifies GPT-QModel as its active GPTQ backend and says AutoGPTQ is no longer supported there. Repository LICENSE is Apache-2.0. This is a maintainer-supported implementation, not evidence of a QML reproduction or a replication of the original tables. [5] [6] |
| [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) | Author-group inference kernel for FP16×INT4 and GPTQ-compatible weights, with a GPTQ example; repository license: Apache-2.0. It is chiefly an inference-kernel/runtime project, not a substitute for the paper’s quantization-method description. Its own performance claims and benchmark setup are Marlin-project claims, separate from GPTQ’s paper results. [7] |

No Rust GPTQ implementation was verified in the sources reviewed. That is a search finding, not a claim that none exists.

## Relevance and potential overlap with QML

GPTQ is relevant as prior art for any QML work involving post-training compression of neural weights: it already combines calibration-dependent layer reconstruction, low-bit quantization, and second-order error compensation at large language-model scale. A QML comparison should distinguish that established method from any proposed representation, data structure, objective, or systems implementation. The paper does not establish novelty for QML, and the QML project should not present GPTQ as QML work. [1] [3]

**Potential overlap with RMQ, MAP, and QMF: unknown at method level.** Their definitions or specifications were not available in the task context, so I cannot establish equivalence, derivation, or a meaningful claim-by-claim overlap. As a **QML hypothesis only**, overlap would be plausible if any of those methods also minimize calibration-weighted reconstruction error while adjusting unquantized parameters to compensate for quantization error. Similarity at that abstract level would not establish identity or priority. The well-established direct lineage within the paper is GPTQ’s derivation from OBQ, and its evaluation against RTN; related families explicitly discussed include AdaRound, AdaQuant, BRECQ, ZeroQuant, LLM.int8(), and nuQmm. [1]

## Evidence status and gaps

- **Established published method:** GPTQ/OPTQ as described and evaluated in the cited ICLR 2023 paper. [1] [3]
- **Independently reproduced by QML:** No. QML has not reproduced GPTQ; all numerical results above are paper-reported.
- **QML hypothesis:** The possible high-level overlap described above, conditional on RMQ/MAP/QMF specifications.
- **Unknown:** Specific RMQ/MAP/QMF overlap and whether an applicable Rust port exists. No separate DOI for the conference record was verified.

## References

[1]: https://arxiv.org/html/2210.17323v2 "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers, revised paper"
[2]: https://arxiv.org/abs/2210.17323 "arXiv record, GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
[3]: https://research-explorer.ista.ac.at/record/17378 "ISTA research record for the ICLR 2023 OPTQ paper"
[4]: https://github.com/IST-DASLab/gptq "Authors’ GPTQ reference implementation"
[5]: https://raw.githubusercontent.com/ModelCloud/GPTQModel/main/LICENSE "GPT-QModel license file"
[6]: https://huggingface.co/docs/transformers/en/quantization/gptq "Hugging Face Transformers GPTQ documentation"
[7]: https://github.com/IST-DASLab/marlin "IST-DASLab Marlin inference kernel"
[8]: https://huggingface.co/facebook/opt-1.3b/blob/main/LICENSE.md "Meta OPT-175B license text distributed with OPT model"
[9]: https://huggingface.co/bigscience/bloom "BigScience BLOOM model card and license"
