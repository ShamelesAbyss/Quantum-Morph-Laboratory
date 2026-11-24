# CALDERA model compression — prior art

**Classification:** CALDERA is an established published method and prior art; it is not Quantum Morph Laboratory (QML) work. QML has not independently reproduced it. Any relation to QML proposals below is labelled as a hypothesis or unknown, not as evidence of originality. [1] [2]

## Bibliographic record

The paper is **“Compressing Large Language Models using Low Rank and Low Precision Decomposition,”** by Rajarshi Saha, Naomi Sagan, Varun Srivastava, Andrea J. Goldsmith, and Mert Pilanci. The arXiv preprint was submitted 29 May 2024 and revised to v2 on 3 November 2024. It was accepted to the NeurIPS 2024 main conference, *Advances in Neural Information Processing Systems 37*. The official NeurIPS record lists DOI [10.52202/079017-2823](https://doi.org/10.52202/079017-2823); the arXiv identifier is **2405.18886**, with arXiv/DataCite DOI [10.48550/arXiv.2405.18886](https://doi.org/10.48550/arXiv.2405.18886). [1] [2] [3]

Canonical sources are the [NeurIPS proceedings record and paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a20e8451ffb07ad25282c21945ad4f19-Abstract-Conference.html) and [arXiv record/full text](https://arxiv.org/abs/2405.18886). [1] [2] [3]

## Mathematical mechanism

For each weight matrix \(W\in\mathbb{R}^{n\times d}\), CALDERA constructs a quantized full-shape backbone \(Q\) plus a quantized low-rank product \(LR\), with \(L\in\mathbb{R}^{n\times k}\) and \(R\in\mathbb{R}^{k\times d}\). It approximately minimizes the calibration-activation-weighted reconstruction loss

\[
\min_{Q,L,R}\ \|(Q+LR-W)X^\top\|_F^2,
\]

subject to each component belonging to its chosen low-precision quantization codebook. Here \(X\) contains representative input activations for the layer; this makes the objective target layer-output error on calibration inputs rather than raw weight error alone. [2]

The algorithm initializes the low-rank factors at zero, quantizes the backbone using LDLQ with feedback based on the calibration Hessian \(H=X^\top X/m\), and alternates between updating \(Q\) and factorizing the residual \(W-Q\) with a low-precision low-rank (LPLR) routine. LPLR iteratively refines both factors. The experiments also use randomized Hadamard-transform preprocessing to reduce incoherence and, for the backbone, QuIP#'s E8 lattice quantizer. The method permits distinct bit budgets for \(Q\), \(L\), and \(R\), and the low-rank factors can additionally be fine-tuned/adapted. [2] [4]

The paper proves an upper bound on the calibration-weighted approximation error under stated rank, quantizer-range, and quantization-error assumptions. This is a bound on a matrix approximation objective; it is not a guarantee of downstream task accuracy or of generalization beyond the calibration distribution. [2]

## Published experimental results

The following are author-reported results, not independent reproductions. The main evaluations measure WikiText-2 and C4 test perplexity (lower is better) and zero-shot accuracy on Winogrande, RTE, PiQA, ARC-Easy, and ARC-Challenge (higher is better), using EleutherAI’s Language Model Evaluation Harness. LLaMA-2 context length is 4,096 tokens; LLaMA-3 8B uses 8,192. Calibration uses 256 random RedPajama samples for LLaMA-2 7B and LLaMA-3 8B, and 6,144 RedPajama samples from the QuIP# authors’ calibration set for the other main models. [2]

With LLaMA-2 7B, CALDERA at rank 256 with 2-bit \(Q\) and 4-bit factors reports 2.4 average bits/parameter, WikiText-2/C4 perplexities of **6.19/8.14**, and accuracies of **66.0/60.6/75.6/63.6/34.0%** on the five tasks in the order above. The plain 2-bit QuIP# baseline reports 2.0 bits/parameter, perplexities **8.23/10.8**, and task accuracies **61.7/57.8/69.6/61.2/29.9%**. A QuIP# low-rank decomposition with rank-64 16-bit factors at 2.4 average bits reports perplexities 7.73/10.0; the paper notes QuIP# does not quantize its factors. [2]

For LLaMA-2 70B, rank-256 CALDERA with 4-bit factors reports 2.2 average bits/parameter and perplexities **3.98/5.76**, versus **4.16/6.01** for plain 2-bit QuIP#. CALDERA’s five task accuracies are **77.6/71.5/79.8/79.5/47.4%**, while QuIP# reports **74.2/70.0/78.8/77.9/48.6%**. Thus the CALDERA row improves both perplexities and most listed accuracies, but not ARC-Challenge; results do not show uniform wins on every metric. [2]

For LLaMA-3 8B, rank-256 CALDERA at 2.4 average bits reports WikiText-2/C4 perplexities **8.22/9.56** and five-task accuracies **69.7/65.0/75.1/73.2/40.0%**. Plain 2-bit QuIP# reports **13.8/15.6** perplexities and **63.2/52.7/67.6/57.6/28.2%** accuracies. The paper also includes LLaMA-2 13B and appendix Mistral-7B results. [2]

These are comparisons under the paper’s configurations: \(B_Q=2\), rank and low-rank bit budgets vary, and the QuIP# baselines are either plain 2-bit quantization or a higher-precision-factor low-rank decomposition. The paper reports an unquantized 16-bit reference as well. Its abstract summarizes the result as outperforming existing post-training compression approaches below 2.5 bits/parameter, but the tabulated main comparisons are principally against QuIP# and unquantized models. [1] [2]

## Compute and implementation requirements

The paper reports Hessian computation for most tested models on four NVIDIA A10 GPUs, with LLaMA-2 70B Hessian computation on one H100; quantization on four NVIDIA RTX A6000 GPUs; and evaluation on A6000 GPUs except the LLaMA-2 70B zero-shot runs, which used an H100. It reports approximately **34 GPU-hours** to compress LLaMA-2 7B or Mistral-7B at rank 256 on A10G GPUs, **59 GPU-hours** for LLaMA-2 13B on an A6000, and **90 GPU-hours** for LLaMA-2 70B on an H100. These are the authors’ measured configurations, not universal minimum requirements. [2]

The authors’ implementation is Python/PyTorch and builds on QuIP# and LoftQ. Its README says it was tested with Python 3.10/3.11, CUDA 12.1/12.2, and PyTorch 2.2; its setup additionally depends on QuIP# CUDA dequantization kernels and `fast-hadamard-transform`. It is therefore a GPU-oriented research reference, not a verified native Rust implementation. [4] [5]

## Limitations and licensing

The paper characterizes CALDERA as more compute-intensive than simpler compression methods because of its iterative optimization, despite describing the compression cost as a one-time expense. Its quantized models often approach but do not match the unquantized reference, and the tabled results include task-specific regressions such as LLaMA-2 70B ARC-Challenge. The authors identify layer-specific rank selection and improved fine-tuning as possible further work. [2]

The arXiv HTML identifies the article as CC BY 4.0; that is the paper’s publication license, not a software or model-weight license. [3] The CALDERA GitHub repository’s API reports no recognized repository license and has no root `LICENSE` file in its contents. Its `pyproject.toml` contains an MIT license classifier, but that metadata is not a substitute for a verified license grant. Treat the CALDERA software license as **unverified** and obtain clarification before reuse or redistribution. [4] [5] [6]

No CALDERA model-weight artifacts or checkpoint-specific licenses were verified in the repository’s root contents. Base-model terms remain separate: the paper points to Meta’s LLaMA community licenses for LLaMA experiments. Rights for any particular base or converted checkpoint must be checked against that model’s own license; the paper and code do not grant them. [2] [4]

## Reputable implementation

The [authors’ CALDERA repository](https://github.com/pilancilab/caldera) is the authoritative implementation linked from the paper. Its README describes the \(Q+LR\) method, experimental workflow and scripts, and it includes QuIP# as a submodule. This is a primary-source implementation and maintainer documentation, not evidence of independent reproduction. Its software license remains unverified for the reasons above. [2] [4] [5] [6]

## Relevance to QML and potential overlap

**QML hypothesis:** CALDERA is directly relevant as prior art for neural weight compression that combines a low-bit full-size residual/backbone with low-rank factors, optimizes against calibration activations, and uses heterogeneous precision across components. It is a useful comparison point for QML’s representation/compression research, but no QML implementation or empirical comparison is claimed here. [2]

**RMQ overlap hypothesis:** QML’s ledger describes RMQ as exploring \(W_l\approx A_l S B_l+R_l\), with shared latent structure and layer-specific transforms. CALDERA’s \(W\approx Q+LR\) is structurally adjacent as “low-rank product plus additive remainder,” but the roles differ: CALDERA quantizes the full-shape \(Q\) backbone and low-rank factors, whereas the ledger’s RMQ description emphasizes sharing a latent \(S\) across layers. Whether RMQ shares factors or uses equivalent objectives is unverified; this is conceptual overlap, not method identity. [2] [7]

**MAP overlap hypothesis:** The ledger frames MAP as precision allocation using functional importance and cross-layer error propagation. CALDERA is calibration-aware and varies the rank/bit budget in experiments, but its stated objective is local layer activation-weighted reconstruction and the described study does not establish MAP-style cross-layer propagation or a general importance-driven allocator. Overlap is plausible at the broad precision-versus-error tradeoff level only; exact overlap is unverified. [2] [7]

**QMF and other methods:** QMF is not defined in the reviewed QML materials, so exact overlap is unknown. The paper explicitly builds on QuIP#’s LDLQ quantization and randomized Hadamard preprocessing, refines the earlier LPLR method, and relates the low-rank/adaptation component to LoRA and quantization-aware PEFT methods such as LoftQ/LQ-LoRA. These are established technical relationships reported in the paper, not QML claims. [2]

## Evidence status

- **Established published method:** CALDERA, with paper, publication record, algorithm, theory, and author-reported experiments verified from primary sources. [1] [2] [3]
- **Independently reproduced:** No. QML has not reproduced CALDERA.
- **QML hypothesis:** Relevance to QML compression; possible conceptual overlap with RMQ and MAP as described in QML’s ledger.
- **Unknown:** Exact RMQ/MAP equivalence; all QMF overlap; CALDERA repository’s legally operative software license; and any rights for individual base-model or derived checkpoint weights.

## References

[1]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/a20e8451ffb07ad25282c21945ad4f19-Abstract-Conference.html "NeurIPS 2024 proceedings record for CALDERA"
[2]: https://proceedings.neurips.cc/paper_files/paper/2024/file/a20e8451ffb07ad25282c21945ad4f19-Paper-Conference.pdf "CALDERA full paper in NeurIPS 2024 proceedings"
[3]: https://arxiv.org/abs/2405.18886 "arXiv record and version history for CALDERA"
[4]: https://github.com/pilancilab/caldera "Authors’ CALDERA implementation repository"
[5]: https://raw.githubusercontent.com/pilancilab/caldera/main/pyproject.toml "CALDERA package metadata and dependencies"
[6]: https://api.github.com/repos/pilancilab/caldera "GitHub repository metadata and license field for CALDERA"
[7]: ../novelty-ledger.md "QML novelty ledger defining RMQ and MAP working hypotheses"
