# AWQ — Activation-aware Weight Quantization (prior art)

**Index ID:** LIT-03-AWQ
**Evidence status:** **Established published method.** QML has not independently reproduced or implemented AWQ. This entry records prior art, not QML work, and makes no originality claim.

## Bibliographic record

The published title is **“AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration.”** Authors: Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. The work appeared in *Proceedings of Machine Learning and Systems*, volume 6 (MLSys 2024), pages 87–100. Its preprint was submitted to arXiv on 1 June 2023 as arXiv:2306.00978. The arXiv title differs slightly (“…for LLM Compression and Acceleration”); the proceedings record is the publication-level citation. The arXiv page lists DOI 10.48550/arXiv.2306.00978, which is an arXiv-issued DOI; I did not verify a separate proceedings DOI. [1] [2]

## Mathematical principles

AWQ is post-training, weight-only quantization. It does not protect salient weights by leaving them in floating point in its final representation. Instead it observes activation magnitudes to identify salient input channels, then rescales the corresponding weight channels before quantization and applies the inverse scale to the matching activations. The equivalent transformation aims to lower quantization error on important channels while keeping a uniform low-bit weight representation. [1]

For a linear layer with weights \(W\), inputs \(X\), and diagonal channel scale \(D_s=\operatorname{diag}(s)\), the paper’s calibration objective is to choose \(s\) to minimize the output mismatch after quantizing the scaled weights:

\[
\min_s \; \lVert Q(WD_s)D_s^{-1}X-WX\rVert.
\]

Here, \(X\) consists of cached input features from a small calibration set; the inverse scale can usually be folded into the preceding operator. Rather than optimize an unconstrained scale with gradients, AWQ measures average activation magnitude per input channel, forms \(s=s_X^\alpha\), and grid-searches a single exponent \(\alpha\) in [0,1]. The paper uses 20 grid points and additionally clips weights to reduce quantization MSE. Its experiments focus on grouped INT3/INT4 weight quantization, generally with group size 128. [1]

The paper motivates scaling with a rounding-error argument: under the simplifying assumption that the group quantization step \(\Delta\) is approximately unchanged, multiplying a salient weight by \(s>1\) while dividing its activation by \(s\) reduces that weight’s relative rounding error by roughly \(1/s\). But scaling can also alter \(\Delta\), increasing errors for nonsalient weights. The authors illustrate this tradeoff on OPT-6.7B: with 1% salient channels, WikiText perplexity improved from 23.54 at \(s=1\) to 11.92 at \(s=2\), then worsened to 12.36 at \(s=4\). This is the paper’s motivation, not a universal error bound. [1]

## Published results and conditions

The following are **results reported by the paper’s authors**, not an independent reproduction. The core language-modeling comparison uses WikiText-2 perplexity, lower being better; the principal baselines are round-to-nearest (RTN), GPTQ, and GPTQ with reordering (GPTQ-R). Most cited comparisons use group size 128 and INT3 or INT4. [1]

For example, at INT3-g128 the paper reports Llama-2-7B WikiText-2 perplexity of 6.24 for AWQ, versus 6.66 for RTN, 6.43 for GPTQ, and 6.42 for GPTQ-R; the FP16 reference is 5.47. For LLaMA-7B, the corresponding reported values are 6.35 (AWQ), 7.01 (RTN), 8.81 (GPTQ), and 5.68 (FP16). These are particular model/configuration rows, not a claim that AWQ wins every model or benchmark. At INT4-g128, the paper also reports AWQ at 5.60 for Llama-2-7B versus 5.73 RTN and 5.69 GPTQ, with FP16 at 5.47. [1]

For domain tasks, the authors report INT4-g128 CodeLlama-7B-Instruct MBPP pass@1 of 40.64 for AWQ, 37.51 RTN, and 31.97 GPTQ; the FP16 reference is 38.53. For Llama-2-7B on GSM8K they report 13.57 for AWQ, 11.07 RTN, 12.13 GPTQ, and 13.87 FP16. These results are benchmark-specific and do not establish general equivalence to FP16. [1]

TinyChat is a separate inference system delivered alongside AWQ. It combines quantized linear kernels with platform-specific packing and kernel fusion, so the speed claims are system-level results and should not be attributed to the AWQ scaling algorithm alone. The paper reports batch-size-1 runs with a four-token prompt and 200 generated tokens, using median latency. For VILA-7B W4A16 throughput, it reports 81.6 / 58.5 / 11.5 for the FP16 implementation and 155.3 / 168.1 / 35.6 for TinyChat on A100 / RTX 4090 / Jetson Orin, respectively; the table’s throughput figures are tokens/s. The authors report VILA-13B as 48.5 / OOM / 6.1 FP16 and 102.1 / 99.0 / 17.5 W4A16 on those devices. This shows both hardware dependence and that one evaluated FP16 configuration did not fit on the RTX 4090. [1]

The paper also reports a 10× smaller calibration-set comparison for OPT-6.7B INT3-g128 (16 sequences for AWQ versus 192 for GPTQ), and a distribution-shift test using PubMed abstracts and Enron email subsets: when calibration and evaluation distributions differed, reported perplexity increase was 0.5–0.6 for AWQ versus 2.3–4.9 for GPTQ. Those are authors’ specific test results, not a guarantee against calibration mismatch. [1]

## Computational requirements

AWQ’s offline search needs representative calibration inputs to collect per-channel activation statistics and evaluate candidate scales; the paper specifies a small Pile-derived calibration set, group size 128 in the main comparisons, and a 20-point \(\alpha\) search. It avoids backpropagation and layer reconstruction. The paper does not establish a general minimum CPU/GPU, RAM/VRAM, quantization time, or peak-memory requirement for running this search, so those quantities are **not reported/verified here**. [1]

The original `llm-awq` repository provides a concrete Python 3.10 installation path, PyTorch code, CUDA kernels, and FlashAttention installation instructions; edge-device setup may require platform-specific PyTorch binaries and a different Python version. These are implementation-specific setup requirements, not intrinsic mathematical requirements of AWQ. Its measured runtime examples use accelerators including A100, RTX 4090, RTX 4070 laptop, and Jetson Orin; the paper also describes Raspberry Pi 4B tests. No universal hardware requirement follows from those examples. [3] [1]

## Limitations and licensing

AWQ is principally a **weight-only** method: activations are used to guide scales but are not thereby quantized by AWQ. Accuracy and speed depend on bit width, group size, calibration data, model, and deployment kernels. At more aggressive quantization the paper documents degradation and a scaling tradeoff; its speed results depend on TinyChat’s kernel/fusion engineering and differ by platform. Therefore memory reduction does not imply a fixed end-to-end speedup. The authors evaluate selected LLaMA/OPT families, instruction-tuned models, coding/math tasks, and visual-language models; benchmark scope should not be read as exhaustive. [1]

The `llm-awq` software repository has an MIT license. That license covers the repository software, not automatically the AWQ paper, third-party dependencies, or quantized model weights. The AutoAWQ repository also declares MIT, but its own README says AutoAWQ is deprecated and no longer maintained; the README points to vLLM’s adoption. Treat it as a useful historical/reputable implementation, not the preferred maintained path. [3] [4] [5] [6]

The arXiv record links an arXiv perpetual non-exclusive distribution license. I did not verify a separate permissive license for the paper text or a distinct proceedings DOI; do not infer one from the software license. Quantized checkpoints remain subject to the source model’s license and any dataset/dependency terms. The official model-zoo page does not establish a blanket license for all underlying models in the sources reviewed, so each checkpoint’s terms must be checked separately. [2] [3]

## Reputable implementations

- **MIT HAN Lab `llm-awq`** — [repository](https://github.com/mit-han-lab/llm-awq). The authors’ implementation includes AWQ search, model examples, quantized weight loading, CUDA kernels, and TinyChat. Repository license: MIT. [3] [4]
- **AutoAWQ** — [repository](https://github.com/casper-hansen/AutoAWQ). A widely used AWQ implementation with an MIT license, but explicitly deprecated/no longer maintained according to its own README. Hugging Face Transformers documents loading AutoAWQ- and `llm-awq`-quantized models; the documentation also warns that AutoAWQ can downgrade Transformers and records a particular fused/unfused benchmark, which is maintainer documentation rather than independent AWQ validation. [5] [6] [7]

## Relevance and potential overlap with QML

**QML relevance:** AWQ is direct prior art for sensitivity/saliency-guided neural weight quantization. In particular, using activations to determine channel importance and modifying channel scales to reduce low-bit quantization error overlaps conceptually with proposals that allocate representation precision according to functional importance. It is not a low-rank/shared-latent factorization method, and AWQ’s published contribution should be credited to its authors. QML has no AWQ implementation or reproduction; the repository remains an early scaffold. [1] [QML README](../../README.md)

**RMQ:** The QML novelty ledger describes RMQ as shared latent structures, layer-specific transforms, and residual information (proposed `W_l ≈ A_l × S × B_l + R_l`). That representation/factorization direction is not AWQ’s channel rescaling plus groupwise low-bit quantization. There may be an engineering interaction if a future RMQ representation is quantized, but whether AWQ scales transfer to such factors is an untested QML hypothesis, not an established result. [QML novelty ledger](../novelty-ledger.md)

**MAP:** The ledger describes MAP as precision allocation from functional importance and cross-layer error propagation. AWQ has clear conceptual overlap in activation-informed saliency, but its standard method searches per-channel rescaling while quantizing weights at a chosen uniform bit width; it does not implement MAP’s proposed cross-layer error propagation or adaptive bit allocation. Any combination or differentiation remains a QML hypothesis pending formal definitions and experiments. [QML novelty ledger](../novelty-ledger.md)

**QMF:** Current QML documentation calls QMF experimental and says its requirements and binary specification remain unvalidated; the supplied project materials do not provide a sufficiently precise QMF mathematical definition for a detailed comparison. AWQ is a quantization method, but a stronger overlap assessment is **unknown** until QMF is defined. [QML architecture](../../ARCHITECTURE.md) [QML roadmap](../../ROADMAP.md)

**Other known methods:** The paper identifies GPTQ as a close weight-quantization baseline, distinguishes AWQ’s activation-derived scaling from GPTQ’s second-order error compensation/reconstruction approach, and reports combining AWQ with GPTQ for INT2-g64 on OPT. This establishes compatibility in that reported experiment, not a general guarantee. SmoothQuant is another relevant neighbor: AWQ’s paper situates it in W8A8 activation/weight quantization, while AWQ focuses on low-bit weight-only quantization and activation-informed scaling. Both use equivalent scaling transformations to manage quantization difficulty, so conceptual overlap exists even though their target quantization settings and formulations differ. [1]

## Gaps

No QML reproduction, implementation, or head-to-head experiment exists. Exact compute/time/memory requirements for AWQ calibration are not established by the reviewed paper or repository overview. A separate proceedings DOI and a blanket license for AWQ quantized weights were not verified. The source model’s license must be checked per checkpoint. QMF’s intended math is currently too underspecified in QML’s own documentation for a meaningful method-level overlap conclusion.

## References

[1]: https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration"
[2]: https://arxiv.org/abs/2306.00978 "AWQ arXiv record 2306.00978"
[3]: https://github.com/mit-han-lab/llm-awq "MIT HAN Lab llm-awq repository"
[4]: https://raw.githubusercontent.com/mit-han-lab/llm-awq/main/LICENSE "llm-awq MIT License"
[5]: https://github.com/casper-hansen/AutoAWQ "AutoAWQ repository and maintenance notice"
[6]: https://raw.githubusercontent.com/casper-hansen/AutoAWQ/main/LICENSE "AutoAWQ MIT License"
[7]: https://huggingface.co/docs/transformers/en/quantization/awq "Hugging Face Transformers AWQ documentation"
