# BitNet and ternary neural networks — prior-art record

**This is prior art, not QML work.** BitNet and BitNet b1.58 were introduced in Microsoft Research-authored papers in 2023 and 2024; QML did not originate these methods and has not reproduced them. The local QML ledger describes RMQ and MAP as untested proposals, and its QMF specification is unavailable/incomplete, so overlap statements below are bounded by those documented descriptions. [1] [3] [11]

## Record fields

- **ID:** 08
- **Topic:** BitNet and ternary neural networks
- **Research title(s):** “BitNet: Scaling 1-bit Transformers for Large Language Models” (original binary-weight method); “The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits” (BitNet b1.58 ternary extension); later model report, “BitNet b1.58 2B4T Technical Report.” [1] [3] [8]
- **Authors:** Original BitNet: Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, Furu Wei. BitNet b1.58: Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei. 2B4T report: Shuming Ma, Hongyu Wang, Shaohan Huang, Xingxing Zhang, Ying Hu, Ting Song, Yan Xia, Furu Wei. [1] [3] [8]
- **Publication dates:** Original BitNet submitted to arXiv on 17 October 2023; BitNet b1.58 on 27 February 2024; 2B4T technical report v1 on 16 April 2025 and v2 revised 25 April 2025. These records identify the works as arXiv preprints/technical reports and label them “work in progress”; a peer-reviewed proceedings/journal publication was not verified in the cited records. [1] [3] [8]
- **Canonical sources and identifiers:** Original BitNet: arXiv:2310.11453, DOI 10.48550/arXiv.2310.11453. BitNet b1.58: arXiv:2402.17764, DOI 10.48550/arXiv.2402.17764. Later model: arXiv:2504.12285, DOI 10.48550/arXiv.2504.12285. [1] [3] [8]

## Mathematical principles

The 2023 BitNet architecture replaces Transformer linear layers with BitLinear and trains binary weights from scratch. It centers each weight matrix by its mean and applies a sign function, giving weights in \(\{-1,+1\}\). Its activation path uses absmax quantization (8-bit in the reported experiments), a scaling factor based on the absolute maximum, and LayerNorm/SubLN before quantization to stabilize activation variance. Since the sign and clipping operations are nondifferentiable, training uses a straight-through estimator; latent learnable weights, gradients, and optimizer states are retained at high precision. Thus “1-bit” describes the inference weight representation, not the full training state or every tensor in the model. [2]

BitNet b1.58 adds zero to the weight alphabet: each quantized weight is in \(\{-1,0,+1\}\). For a matrix \(W\), the paper defines \(\gamma=(1/nm)\sum_{ij}|W_{ij}|\) and quantizes as \(\widetilde W=\mathrm{RoundClip}(W/(\gamma+\epsilon),-1,1)\). The three-symbol alphabet corresponds to \(\log_2(3)\approx1.585\) bits of information per symbol, hence “1.58-bit”; it does not by itself imply that every implementation stores each weight in exactly 1.58 physical bits. The paper trains this architecture from scratch with 1.58-bit weights and 8-bit activations. The zero value can suppress/filter features, and ternary-weight matrix products can be implemented with additions/subtractions and skips rather than general multiplications, subject to scaling and other operations. [4] [5]

## Reported results and conditions

These are **author-reported experiments**, not independent QML replications.

- **Original binary BitNet:** Authors trained autoregressive models from 125M to 30B parameters on an English corpus assembled from The Pile, Common Crawl snapshots, RealNews, and CC-Stories. They compared their models with FP16 Transformer baselines trained on the same datasets/settings and with post-training quantization approaches including Absmax, SmoothQuant, GPTQ, and QuIP. For the 6.7B comparison, the paper reports BitNet W1A8 perplexity 17.07 and average zero-shot downstream score 55.9; the FP16 Transformer values are 15.19 and 57.8. The paper therefore reports competitive low-bit performance, not blanket superiority over FP16. Energy claims use an arithmetic-energy model at 7nm/45nm process assumptions rather than a direct measurement of a deployed processor. [2]
- **BitNet b1.58 controlled comparison:** The authors say they pretrained b1.58 and their reproduced FP16 LLaMA baselines on RedPajama for 100 billion tokens, with model sizes from 700M to 3.9B. At 3B, b1.58 reports validation perplexity 9.91 versus 10.04 for LLaMA, average task score 50.2 versus 49.7, 2.22 GB versus 7.89 GB memory, and 1.87 ms versus 5.07 ms per output token. The efficiency timing/memory used FasterTransformer and a 2-bit Ladder kernel for b1.58; the paper does not identify the exact GPU for these Table 1 measurements in the source passage reviewed. For a separate 70B throughput test, the authors specify two 80GB A100 GPUs, pipeline parallelism, sequence length 512, and maximum batch size under the memory limit: 2,977 versus 333 tokens/s (8.9×), with batch sizes 176 versus 16. The paper’s 71.4× matrix-multiplication energy-saving figure is based on an energy model, not measured end-to-end chip power. [4]
- **Later 2B4T model:** The 2025 technical report and official model card describe a model trained on 4 trillion tokens and compare instruction-tuned checkpoints, not a strictly matched training-token/model-size experiment. The report’s table gives BitNet b1.58 2B 0.4 GB non-embedding memory, 29 ms CPU time-per-output-token, and estimated 0.028 J energy, versus varying figures across LLaMA 3.2 1B, Gemma-3 1B, Qwen2.5 1.5B, SmolLM2 1.7B, and MiniCPM 2B. Its benchmark results vary by task; the overall average shown is 54.19, below Qwen2.5’s 55.23 but above some other listed baselines. The paper also compares Qwen2.5 1.5B BF16/GPTQ/AWQ variants and reports 0.4 GB non-embedding memory and an average 55.01 versus averages 55.72, 52.15, and 51.17, respectively. Authors describe re-running benchmarks with a public evaluation pipeline. Energy is explicitly estimated, and the comparisons involve models with different sizes and pretraining-token counts (the table reports 1.1T–18T for several baselines, versus 4T for BitNet). [8] [9]
- **Inference implementation report:** The Microsoft-authored bitnet.cpp report benchmarks an Apple M2 Ultra/64GB and Intel Core i7-13700H/64GB. It reports decoding speedups against llama.cpp of 1.37×–5.07× on M2 and 2.37×–6.17× on Intel in its unlimited-thread tests; two-thread speedups are separately reported. It reports J/token energy savings on selected model sizes and exact-output matches against an FP32 reference for the tested kernel/model prompts. The authors explicitly state that the 700M model used for an accuracy test and an 8B demo model were not trained or released by Microsoft. These are implementation authors’ results, not third-party replications. [5]

## Computational requirements

The original paper reports training scales up to 30B parameters and supplies training hyperparameters, but the cited source does not verify a specific training hardware count, total FLOPs, wall-clock duration, or minimum GPU. Large-scale training is not “1-bit memory” training: high-precision latent weights, gradients, and optimizer states are used. The 2B4T model was trained on 4T tokens, but its training compute/hardware budget was not verified in the cited sources. [2] [8]

For local inference, low-bit representations alone do not guarantee speedups: specialized kernels matter. The official bitnet.cpp repository currently documents CPU and GPU inference support, and lists Python ≥3.10, CMake ≥3.22, and Clang ≥18 for its build environment. Its benchmark report’s CPU systems and model-size limits are stated above; on the tested M2 Ultra it reports 70B and 100B runs, while the tested Intel machine supports up to 13B in that configuration. These are measured test configurations, not universal minimum requirements. The official model card cautions that the standard Transformers path does not realize the optimized inference speed/energy benefits and directs users to bitnet.cpp for those gains. [5] [6] [9]

## Known limitations and caveats

BitNet and b1.58 are quantization-aware training architectures, not generic post-training compressors for arbitrary pretrained weights; the papers train the low-bit models from scratch. Training state and non-linear/embedding components retain higher precision, and b1.58 still uses 8-bit activations in the described scheme. Realized storage and speed depend on packing formats, scale/metadata overhead, and kernels; the theoretical 1.585 bits/symbol is not a guarantee of end-to-end model size or speed. [2] [4] [5]

The evidence comprises author-run experiments and maintainer-reported engineering benchmarks. Baselines, model sizes, training data/tokens, evaluation settings, hardware, and kernel choices differ across studies; estimated energy should not be conflated with measured power. The 2025 model card identifies limited non-English/underrepresented-domain support, possible inaccurate or harmful outputs, elevated defects on election-critical questions, and a research/development-only disclaimer. The model card recommends long-sequence adaptation for specialized contexts beyond its stated 4096-token pretraining length. [4] [5] [8] [9]

## Licensing considerations

The arXiv records show a non-exclusive distribution license for the paper manuscripts; that is not a software or model-weights reuse license. The official Microsoft bitnet.cpp repository declares the MIT License. The Microsoft BitNet b1.58 2B4T model card declares MIT for weights and code. Other model releases, dependencies, submodules, datasets, and derivative artifacts may have separate terms; their licenses were not established here and should be checked independently before use or redistribution. No official training implementation for the original papers was verified in the sources reviewed. [1] [3] [6] [7] [9]

## Reputable implementations

- **Microsoft `bitnet.cpp` (official inference framework):** <https://github.com/microsoft/BitNet>. MIT per the official repository. It contains optimized kernels and documents supported formats/models; it is an inference framework, not evidence that QML reproduced training or results. The repository’s performance numbers are maintainer claims and should be read alongside the conditions in its technical report. [5] [6] [7]
- **Microsoft BitNet b1.58 2B4T model:** <https://huggingface.co/microsoft/bitnet-b1.58-2B-4T>. The official model card declares MIT for model weights and code and points to packed inference, BF16 master-weight, and GGUF variants. Its results are author/model-maintainer reports, not independent QML measurements. [8] [9]

## Relevance to QML and potential overlap

BitNet is directly relevant prior art for neural-model representation/compression: it establishes binary-weight and ternary-weight Transformer alternatives, quantization-aware training, activation quantization, packed inference formats, and specialized integer/addition-based kernels. QML’s stated local-first Rust research-lab scope could intersect at the level of importing/inspecting representations, measuring full artifact overhead, benchmarking model quality and inference, or investigating a distinct representation. No such QML method or reproduction is established here. [10]

- **RMQ:** QML describes RMQ as shared latent structures plus layer-specific transforms and residual information (for example, \(W_l\approx A_l S B_l+R_l\)). BitNet’s ternarization is a direct overlap in the broad category of weight representation/compression, but the cited BitNet method does not use that shared cross-layer factorization/residual construction. It is a relevant comparison baseline, not evidence that RMQ is original or equivalent. [2] [4] [11]
- **MAP:** QML describes MAP as precision allocation based on measured functional importance and cross-layer error propagation. BitNet uses a fixed low-bit weight/activation scheme in the cited methods; no comparable sensitivity-driven, adaptive per-layer precision allocation is part of the described BitNet method. Broad compression/precision overlap exists, but not identity of the mechanism. [4] [11]
- **QMF:** QML’s available documents mention QMF serialization but report that the directive is incomplete and provide no complete binary specification. A QMF that stores ternary weights could overlap at the encoding/container level; the precise overlap cannot be assessed without the missing specification. [10] [11]

## Evidence status

**Established published method** (published as identifiable arXiv preprints/technical reports; venue publication not verified in these records). **Not independently reproduced by QML.** No QML implementation, experiment, or independent replication is asserted. The possible connection to RMQ/MAP/QMF is a **QML hypothesis/overlap assessment**, not a claim of novelty. [1] [3] [8] [10] [11]

## Gaps

The reviewed primary sources do not establish an exact compute budget, training hardware, or training duration for the 2023 BitNet models or the 2025 2B4T training run. The b1.58 paper passage reviewed does not state the GPU used for its smaller-scale latency/memory table. Benchmark result details are not independent replications. The legal status of paper text, the datasets, third-party dependencies/submodules, and non-Microsoft models is distinct from the explicit MIT declarations for the Microsoft inference repository and official 2B4T model card and has not been separately audited. QMF semantics remain unavailable in the current QML project documents.

## References

[1]: https://arxiv.org/abs/2310.11453 "BitNet: Scaling 1-bit Transformers for Large Language Models"
[2]: https://arxiv.org/html/2310.11453v1 "BitNet: Scaling 1-bit Transformers for Large Language Models, full text"
[3]: https://arxiv.org/abs/2402.17764 "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
[4]: https://arxiv.org/html/2402.17764v1 "The Era of 1-bit LLMs, full text"
[5]: https://arxiv.org/html/2410.16144v2 "1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs"
[6]: https://github.com/microsoft/BitNet "Microsoft BitNet official inference framework repository"
[7]: https://raw.githubusercontent.com/microsoft/BitNet/main/LICENSE "Microsoft BitNet repository MIT License"
[8]: https://arxiv.org/abs/2504.12285 "BitNet b1.58 2B4T Technical Report"
[9]: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T "Microsoft BitNet b1.58 2B4T model card"
[10]: ../../README.md "Quantum Morph Laboratory README"
[11]: ../novelty-ledger.md "Quantum Morph Laboratory research novelty ledger"
[12]: ../../ARCHITECTURE.md "Quantum Morph Laboratory architecture notes"
