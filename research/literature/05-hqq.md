# Half-Quadratic Quantization (HQQ)

**Prior-art status:** HQQ is an established, author-published quantization method predating Quantum Morph Laboratory (QML). It is not QML work. QML has not independently reproduced it, and this note makes no novelty claim for QML.

## Bibliographic record

**Title:** *Half-Quadratic Quantization of Large Machine Learning Models*. **Authors:** Hicham Badri and Appu Shaji, affiliated in the original article with Mobius Labs. The authors’ citation metadata dates the article to November 2023; no day is specified. Dropbox Tech republished it on October 22, 2025 and explicitly identifies it as a Mobius team article originally published in 2023. The primary source is an author technical article/project blog, not a formal conference proceedings paper or an arXiv/DOI paper. No HQQ DOI or arXiv identifier was found; the canonical source is the [original HQQ article](https://dropbox.github.io/hqq_blog/). [1] [2]

The project’s official implementation is the [HQQ GitHub repository](https://github.com/dropbox/hqq), which identifies itself as the official implementation and carries an Apache-2.0 license. [3]

## Mathematical mechanism

HQQ is calibration-free, weight-only, groupwise affine quantization. Given weight values `W`, scale `s`, and zero-point `z`, the article defines quantization as `Q(W) = round(W/s + z)` and dequantization as `s(Q(W) − z)`. It seeks quantization parameters that reduce a sparsity-promoting `l_p` penalty on the weight reconstruction error, using `p < 1` to model a heavy-tailed error distribution. The article formulates the objective over scale and zero-point, then fixes the scale while iteratively optimizing the zero-point. [1]

The half-quadratic (HQ) solver introduces a temporary auxiliary error variable `W_e` and a quadratic penalty weighted by positive parameter `β`. It alternates two tractable subproblems: a generalized soft-thresholding/proximal update for `W_e`, and a closed-form groupwise-mean update for `z`; `β` is increased by a factor `κ`. The authors state that these updates use closed-form solutions without gradients and can run in inference mode using half precision. The temporary `W_e` is an optimization variable, not a separately stored residual tensor in the resulting quantized representation. [1]

The article’s reported Llama-2 setting is `p=0.7`, `β=1`, `κ=1.01`, 20 iterations with early stopping. It says zero-points are separately quantized to 8 bits, ungrouped and unoptimized. The article notes limited experimentation with solver hyperparameters, so the listed settings are not shown to be optimal. [1]

## Author-reported experimental results

These are the authors’ reported results, not independent replications. For Llama-2, the article reports WikiText-2 perplexity (lower is better) and runtime GPU memory in GB for running the quantized model (with additional prediction memory depending on sequence length). Its comparison includes FP16, bitsandbytes (BNB), GPTQ via AutoGPTQ, AWQ via AutoAWQ, and HQQ, with differing group sizes. [1]

Examples from the reported table:

- At 4 bits, HQQ group-size 64 reports Llama-2-7B PPL 5.30 / 4.6 GB, 13B PPL 4.70 / 8.2 GB, and 70B PPL 3.19 / 37.52 GB. The listed GPTQ group-size-64 results are 5.38 / 5.0 GB, 4.73 / 9.1 GB, and 3.23 / 41.13 GB, respectively. AWQ group-size 64 reports 5.28 / 4.6 GB, 4.70 / 8.5 GB, and 3.20 / 37.08 GB. These are author-reported comparisons under the article’s stated evaluation, not a guarantee of superiority across settings. [1]
- At 2 bits, HQQ group-size 16 reports Llama-2-7B PPL 7.30 / 4.1 GB, 13B 6.36 / 6.9 GB, and 70B 4.12 / 30.27 GB. The FP16 13B entry is 4.63 / 25.6 GB; the authors describe HQQ-quantized 70B as having lower PPL than full-precision 13B at comparable memory use. The article also shows that group-size choices matter: HQQ group-size 32 at 2 bits gives 70B PPL 4.82 / 24.2 GB, while its 7B PPL is 15.61. [1]
- For OpenCLIP vision transformers evaluated on ImageNet, the article reports zero-shot and linear-probe top-1/top-5 accuracy. For example, HQQ 4-bit ViT-B-32 has zero-shot top-1 0.639 versus BNB 4-bit 0.608; HQQ 3-bit ViT-H-14 has zero-shot top-1 0.755 versus FP16 ViT-L-14 0.731; HQQ 2-bit ViT-H-14 has 0.716 versus FP16 ViT-B-32 0.664. These are the authors’ reported model/dataset comparisons; the different model sizes in the last two comparisons should not be read as controlled same-model quantization comparisons. [1]

For speed, the authors report that HQQ processes Llama-2-70B in under five minutes and is more than 50× faster than GPTQ. They also report more than 100× speed-up over an AdamW/autograd baseline for quantizing Llama-2-7B. They state that HQQ’s quantization runs on the GPU in half precision, with CPU use for transfer after the solver finishes, and caution that GPTQ/AWQ processing time varies substantially by machine. The article does not identify the exact machine, GPU, software versions, or timing protocol for these speed claims; they are therefore maintainer/author claims, not independently verified measurements here. [1] [2]

## Compute, limitations, and license

The author article says quantization is GPU-based and half-precision; it does not provide a complete machine specification or minimum VRAM requirement for the reported quantization runs. Its GPU memory table is for running the quantized models, not a stated minimum device specification. Do not infer a specific GPU requirement from the speed claims. [1]

The official HQQ documentation in Transformers describes support for CUDA and XPU, 1/2/3/4/8-bit configurations, and warns that CPU/disk offload is not supported in that integration. It also documents implementation constraints: faster torchao/BitBLAS backends require `axis=1`, and dynamic layer-tag configurations cannot target a single decoder layer. These are integration/backend constraints and may change by version; they are not general mathematical limitations of the HQQ objective. [4]

The paper/article’s empirical scope is principally Llama-2 on WikiText-2 and OpenCLIP/ViT on ImageNet. The method minimizes weight reconstruction error rather than directly minimizing activation/output error; results on a particular model or downstream task cannot be generalized without evaluation. The article acknowledges limited hyperparameter exploration, and its own 2-bit results show substantial degradation in some cases (for example, Llama-2-7B HQQ group-size 32 has PPL 15.61). No independent QML reproduction has been performed. [1]

The HQQ implementation repository is Apache-2.0 licensed. The author article has no separate article/software license statement in the sources reviewed; no distinct HQQ license for model weights was established. HQQ is a quantizer, not a grant to redistribute source checkpoints: users must verify the original model’s license and terms for any quantized or redistributed weights, and must check the licenses of optional runtime/backend dependencies separately. This note does not infer that model weights inherit the HQQ code license. [1] [3]

## QML relevance and overlap

HQQ is directly relevant prior art for any QML proposal involving neural-weight compression, low-bit scalar quantization, groupwise scale/zero-point selection, or a robust reconstruction objective. The method was published by its authors in 2023, before QML. QML has no weight-loading or transformation pipeline and has not reproduced HQQ; its results therefore remain author-reported literature results, not QML results. [1] [3]

**RMQ — Relational Morph Quantization.** The current QML novelty ledger describes RMQ as a proposal to represent layer weights with shared latent structures, layer-specific transforms, and residual information, e.g. `W_l ≈ A_l × S × B_l + R_l`. HQQ shares the broad goal of compressing weights and can be a relevant scalar-quantization baseline or component. Its reported mechanism is groupwise affine quantization and iterative fitting of quantization parameters; it does not establish a shared cross-layer latent matrix-factor representation or store the auxiliary HQ error variable as an explicit residual. If an RMQ design adds groupwise low-bit quantization, scale/zero-point fitting, or stored residual correction, those parts would overlap with known quantization/residual-compression techniques and need separate comparisons. This is a scope comparison, not a claim that RMQ is distinct or novel. [1] [3] The RMQ description is from QML’s [novelty ledger](../novelty-ledger.md).

**MAP — Morphological Adaptive Precision.** QML’s ledger proposes precision allocation guided by measured functional importance and cross-layer error propagation. HQQ’s article compares multiple fixed bit-width/group-size configurations but does not propose an importance- or cross-layer-error-based bit-allocation algorithm. The official implementation supports dynamic per-layer-type configuration, including different bit widths, so software-level heterogeneous precision is an existing capability and should not be treated as new by itself. A QML allocation rule could overlap with the broader established literature on mixed-precision and sensitivity-guided quantization; HQQ’s specific HQ solver is not evidence of such a sensitivity-allocation method. [1] [3] [4] The MAP description is from QML’s [novelty ledger](../novelty-ledger.md).

**QMF — QML Model Format.** The available architecture note says QMF remains experimental and its requirements and binary specification are not validated. HQQ’s quantized weights and their scales/zero-points are data that a format could encode, but the cited HQQ sources do not define QMF or establish overlap with a QML serialization format. Any format-level overlap is therefore unknown until QMF’s specification exists. [3] [4] See the project’s [architecture note](../../ARCHITECTURE.md).

**Other known methods.** HQQ and GPTQ/AWQ share the goal of low-bit weight-only model compression. The original HQQ article directly benchmarks against GPTQ and AWQ, but the approaches differ: GPTQ is described in its primary paper as one-shot quantization using approximate second-order information, while AWQ uses offline activation statistics to identify salient channels and scale them; HQQ’s author article instead describes a calibration-free weight-reconstruction objective with a half-quadratic solver. These distinctions make GPTQ/AWQ useful prior-art comparators, not evidence that HQQ or any QML proposal is original. [1] [5] [6]

## References

[1]: https://dropbox.github.io/hqq_blog/ "Half-Quadratic Quantization of Large Machine Learning Models"
[2]: https://dropbox.tech/machine-learning/halfquadratic-quantization-of-large-machine-learning-models "Half-Quadratic Quantization of large machine learning models"
[3]: https://github.com/dropbox/hqq "Official HQQ implementation"
[4]: https://huggingface.co/docs/transformers/en/quantization/hqq "HQQ quantization in Transformers documentation"
[5]: https://arxiv.org/abs/2210.17323 "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
[6]: https://arxiv.org/abs/2306.00978 "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
