# SmoothQuant — prior-art record

**Classification:** established published method and prior art. SmoothQuant is not QML work. QML has not reproduced it; the results below are attributed to the paper’s authors or to implementation maintainers as indicated.

## Bibliographic record

**Title:** “SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.” **Authors:** Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. The first two authors are marked as equal contributors. The work appeared in the *Proceedings of the 40th International Conference on Machine Learning* (PMLR, vol. 202, pp. 38087–38099, 2023; conference dates July 23–29, 2023). The arXiv record gives the first submission date as November 18, 2022, and identifies the paper as ICML 2023. [1] [2]

**Canonical sources and identifiers:** PMLR proceedings record: https://proceedings.mlr.press/v202/xiao23c.html. arXiv:2211.10438, DOI https://doi.org/10.48550/arXiv.2211.10438. The DOI is the arXiv-issued DOI; the PMLR record does not list a separate proceedings DOI. [1] [2]

## Method and mathematical mechanism

SmoothQuant is a training-free post-training quantization (PTQ) transformation intended to make weight-and-activation INT8 quantization (W8A8) practical for large transformer models. The authors’ premise is that activation outliers make activations harder to quantize than weights, and that outlier channels are relatively persistent across tokens. Rather than quantizing activations per channel—which they describe as difficult to map efficiently to common INT8 GEMM kernels—they transfer some of the channel-wise scale variation into the weights. [3]

For a linear map \(Y=XW\), let \(s\) be a positive per-input-channel scale vector. The transformation is

\[
\hat X = X\operatorname{diag}(s)^{-1},\qquad
\hat W = \operatorname{diag}(s)W,\qquad
Y=\hat X\hat W.
\]

The floating-point linear map is therefore unchanged before quantization. For input channel \(j\), the paper selects

\[
s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}},
\]

where channel activation ranges are estimated from calibration data and \(\alpha\) controls how much quantization difficulty is shifted from activations to weights. The paper reports that \(\alpha=0.5\) is a useful general setting for OPT and BLOOM, while GLM-130B benefits from a larger value; it also emphasizes that the appropriate setting depends on the model. Scales can be folded into preceding operations offline when the architecture permits, avoiding an extra runtime scaling kernel. The authors use INT8 for compute-heavy linear and attention batched-matrix-multiply operations, while retaining FP16 for lighter element-wise operations such as LayerNorm and Softmax. [3]

The paper’s deployment variants trade quantization granularity and runtime scale computation for efficiency: O1 uses per-token dynamic activation scaling; O2 uses per-tensor dynamic scaling; O3 uses per-tensor static scaling. The paper’s main setting quantizes weights and activations to INT8; it does not claim that all operations in a transformer are integer-only. [3]

## Results reported by the paper

These are the paper authors’ own experiments, not independent replications and not QML results. For OPT-175B, the paper evaluates seven zero-shot benchmarks and WikiText perplexity. FP16 scores 66.9% average accuracy and 10.99 perplexity; SmoothQuant-O3 scores 66.8% and 11.17, respectively. In the same comparison, naive W8A8 and ZeroQuant give 35.5% and 35.8% average accuracy, and LLM.int8() gives 66.7%. Results use 512 random Pile validation sentences for calibration; the authors choose \(\alpha=0.5\) for OPT/BLOOM and \(0.75\) for GLM-130B after a quick grid search on a subset of Pile validation data. [3]

For OPT-175B, the paper reports SmoothQuant-O1/O2/O3 average accuracy of 66.5%/66.4%/66.8% against FP16’s 66.9%, with WikiText perplexities 11.11/11.14/11.17 against FP16’s 10.99. These averages cover LAMBADA, HellaSwag, PIQA, WinoGrande, OpenBookQA, RTE, and COPA; they are paper-reported benchmark outcomes, not a general guarantee of accuracy preservation. [3]

For performance, the authors measure on NVIDIA A100 80GB systems. In their PyTorch implementation, SmoothQuant-O3 reaches up to 1.51× speedup and 1.96× lower peak memory than FP16 for OPT models; the 1.51× result is for OPT-30B at sequence length 256. In FasterTransformer, they report up to 1.56× speedup on OPT-13B/30B and nearly halved memory. For large OPT models, they report similar or faster latency with half the GPU count: OPT-66B on one GPU rather than two, and OPT-175B on four rather than eight. [3]

For MT-NLG 530B, the authors report W8A8 average accuracy of 73.1%, matching the FP16 average of 73.1% across LAMBADA, HellaSwag, PIQA, and WinoGrande. Their serving comparison uses eight A100 80GB GPUs for INT8 versus 16 for FP16; latency is similar rather than uniformly faster (for sequence lengths 128/256/512/1024: INT8 253/434/839/1689 ms versus FP16 232/451/838/1707 ms). This is a reported configuration and result, not a minimum hardware requirement for applying the transformation. [3]

The author repository later reports perplexity for its simulated W8A8 evaluation on Llama-2, Llama-3, Falcon, Mistral, and Mixtral models, alongside FP16 comparison values. For example, it lists Llama-2-7B at 5.474 FP16 versus 5.515 SmoothQuant W8A8, and Mixtral-8x7B at 3.842 versus 3.893. These are project-maintainer-provided results and should be distinguished from the original paper’s results and from independent reproduction. [4]

## Computational requirements and limitations

The method itself is an offline channel rescaling plus quantization; the paper does not specify a universal minimum GPU, calibration-set size, or compute budget for using it. Its paper benchmarks used NVIDIA A100 80GB GPUs, CUTLASS INT8 GEMM kernels, PyTorch/Hugging Face and FasterTransformer; multi-GPU tests use tensor parallelism. The reported MT-NLG 530B experiment required eight A100 80GB GPUs for the INT8 run. The author repository documents an example software environment of Python 3.8, PyTorch 1.12.1 with CUDA 11.3, and Transformers 4.36.0, and says its PyTorch INT8 inference module uses CUTLASS through `torch-int`; treat these as that repository’s example setup, not universal requirements. [3] [4]

The authors identify calibration and alpha selection as relevant choices: they use 512 random Pile sentences for reported calibration and tune alpha on a subset of Pile. Model and architecture differences affect quantization difficulty; the paper reports static-quantization accuracy degradation when calibration statistics differ from evaluation activations, including a 0.8 percentage-point average drop for BLOOM-176B’s O3 setting and about a one-point drop for GLM-130B’s O3 setting versus FP16 in their respective evaluations. [3] [4]

Performance depends on kernels, hardware, and serving stack. The paper’s strong speed and memory claims are specific to its tested A100/CUTLASS configurations; per-channel activation quantization was specifically avoided because the authors found it ill-suited to the efficient GEMM path they target. Static scaling saves runtime scale calculation but relies on representative calibration. SmoothQuant is specifically a W8A8 approach and does not by itself establish benefits for arbitrary bit widths, non-transformer architectures, or workloads. The paper gives no formal guarantee of preserved downstream accuracy for unseen models or calibration distributions. [3]

## Licensing and implementations

The arXiv record marks the article as CC BY 4.0. The proceedings page states “Copyright © the authors and PMLR”; do not infer that the proceedings-hosted version has exactly the same license terms from the arXiv license alone. The author implementation repository is marked MIT and its license file grants permission subject to the MIT notice and disclaimer. [1] [2] [5]

The quantized checkpoints are not covered by the SmoothQuant code license merely because the author repository links them. The repository points to model-specific converted OPT checkpoints on Hugging Face, but this review did not verify every checkpoint’s current model-card terms or the underlying base-model license. Treat weights, base-model terms, and any dataset terms as separate and model-specific; there is no verified blanket “SmoothQuant weights” license here. Dependencies and backend code likewise retain their own license terms. [4] [5]

Reputable implementations verified from their own documentation:

- **MIT HAN Lab SmoothQuant** — https://github.com/mit-han-lab/smoothquant — **MIT**. Author-maintained reference code; includes smoothing/calibration and conversion examples, INT8 PyTorch path using CUTLASS/`torch-int`, evaluation scripts, and links to FasterTransformer. [4] [5]
- **NVIDIA TensorRT-LLM** — https://nvidia.github.io/TensorRT-LLM/reference/precision.html — **Apache-2.0** for the repository’s project code, with third-party components carrying their own listed terms. NVIDIA’s documentation describes SmoothQuant W8A8 preprocessing and examples, and presents a model support matrix that varies by release; this is vendor-maintainer documentation, not a new independent validation of the paper’s results. [6] [7]
- **Intel Neural Compressor** — https://github.com/intel/neural-compressor/blob/main/docs/source/PT_SmoothQuant.md — **Apache-2.0** repository. Its current documentation exposes a `SmoothQuantConfig` workflow and alpha setting for PyTorch/IPEX; it is a maintained framework integration, not evidence that QML has independently reproduced the original benchmark. [8] [9]

## QML relevance and potential overlap

**QML relevance:** SmoothQuant is relevant prior art for a local-first Rust research lab that investigates model representations and compression because it is a concrete, mathematically equivalent reparameterization followed by PTQ, with explicit calibration and quantization steps. It supplies a published comparator for claims involving activation-outlier mitigation through channel rescaling and W8A8. Its implementation and performance results are ecosystem-specific; the paper does not establish a Rust implementation or QML integration. [1] [3]

**Potential overlap:** The paper discusses earlier weight equalization and channel-splitting methods, which target quantization error in weights, as well as Outlier Suppression, ZeroQuant, and LLM.int8() as related approaches for transformer/LLM quantization. SmoothQuant’s distinctive described operation is a diagonal per-channel rescaling that moves part of the activation quantization burden to weights while preserving the linear map before rounding. Any conceptual relationship to QML’s RMQ, MAP, QMF, or other internal methods cannot be determined from the supplied task context because those method definitions are not available in this literature record. A method-by-method equivalence or novelty assessment is therefore **unknown**, not a claim of overlap or originality. This record makes no claim that SmoothQuant is original to QML. [3]

## Evidence status and gaps

- **Established published method:** Yes — ICML 2023 proceedings and the arXiv record establish publication, authorship, date, and method description. [1] [2]
- **Independently reproduced by QML:** No — QML has not reproduced it. All numerical claims above are attributed either to the paper authors or, where explicitly identified, to software maintainers. [3] [4]
- **QML hypothesis:** The potential mapping to QML methods is not assessed as a hypothesis here; it remains unknown until those methods are defined and compared.
- **Gaps:** No independent replication was evaluated in this task. The current licensing of each converted model checkpoint and its base model was not audited. A separate proceedings DOI was not listed on the PMLR record. QML’s RMQ/MAP/QMF definitions were not supplied, so their possible overlap cannot be resolved. Minimum resource requirements beyond the stated tested configurations were not established by the paper.

## References

[1]: https://proceedings.mlr.press/v202/xiao23c.html "SmoothQuant ICML 2023 proceedings record"
[2]: https://arxiv.org/abs/2211.10438 "SmoothQuant arXiv record"
[3]: https://arxiv.org/html/2211.10438v6 "SmoothQuant full paper text"
[4]: https://github.com/mit-han-lab/smoothquant "MIT HAN Lab SmoothQuant implementation"
[5]: https://github.com/mit-han-lab/smoothquant/blob/main/LICENSE "MIT HAN Lab SmoothQuant software license"
[6]: https://nvidia.github.io/TensorRT-LLM/reference/precision.html "TensorRT-LLM numerical precision and SmoothQuant documentation"
[7]: https://github.com/NVIDIA/TensorRT-LLM/blob/main/LICENSE "TensorRT-LLM license"
[8]: https://github.com/intel/neural-compressor/blob/main/docs/source/PT_SmoothQuant.md "Intel Neural Compressor SmoothQuant documentation"
[9]: https://github.com/intel/neural-compressor "Intel Neural Compressor repository and license metadata"
