# CompactifAI model compression

**Prior-art record.** CompactifAI is an established published method by external authors, not work by Quantum Morph Laboratory (QML). QML has not reproduced it. Its relevance to QML is as prior art in tensor-network compression and model representation.

## Bibliographic record

The paper is **“CompactifAI: Extreme Compression of Large Language Models using Quantum-Inspired Tensor Networks.”** The arXiv v1 was submitted on 25 January 2024 and v2 was revised on 13 May 2024. The work was published in the *Proceedings of the 33rd European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2025)*, pp. 531–537; the meeting was held 23–25 April 2025. The proceedings record is DOI [10.14428/esann/2025.ES2025-8](https://doi.org/10.14428/esann/2025.ES2025-8). The exact day of publication is not stated in the sources reviewed; April 2025 is the verified proceedings date. [1] [2]

The arXiv record lists Andrei Tomut, Saeed S. Jahromi, Abhijoy Sarkar, Uygar Kurt, Sukhbinder Singh, Faysal Ishtiaq, César Muñoz, Prabdeep Singh Bajaj, Ali Elborady, Gianni del Bimbo, Mehrazin Alizadeh, David Montero, Pablo Martín-Ramiro, Muhammad Ibrahim, Oussama Tahiri Alaoui, John Malcolm, Samuel Mugel, and Román Orús. The ESANN proceedings author line also includes Borja Aizpurua, for 19 authors total. This note uses the conference-paper author list as the published version and records the author-list difference rather than silently merging versions. [1] [2]

## Mathematical principles and mechanism

CompactifAI tensorizes selected trained weight matrices in self-attention and multilayer-perceptron (MLP) layers. It reshapes a matrix into a higher-order tensor with grouped input/output indices, then performs sequential singular-value decompositions to express it as a matrix product operator (MPO). At each decomposition, it retains the largest singular components up to a chosen bond dimension, χ. Lower χ discards more inter-index correlations and stores fewer trainable parameters. In the paper’s illustrative 216×216 matrix example, the resulting MPO parameter count is given as `2 × 36χ + 36χ²`. More generally, MPO storage grows polynomially with the number of factors for fixed bond dimension, instead of storing the full dense matrix. [2]

The method is post-training compression: a sensitivity profile helps choose which layers to tensorize and how aggressively; then the selected weights are replaced by MPOs and the compressed network is briefly retrained, a phase the authors call “healing.” The paper explains that independently truncating layers is not globally optimal because other layers’ effects are not incorporated into each local decomposition. It uses layer profiling and healing to mitigate that issue. The framework combines with quantization; it is not a claim of quantum-hardware execution. [2]

## What was measured

The published benchmark compressed Meta Llama 2 7B. The paper compares the original float-32 model (table: 27.1 GB, 7B parameters) with 8-bit and 4-bit quantized versions and two CompactifAI variants. The paper labels these “88%” and “93%” compressed. Its table reports 4.1 GB and 2.1 GB respectively, both with 2.1B parameters; the 88% model uses float-16 tensorized layers, and the 93% model combines float-16 tensorized layers with int-4 quantization in untensorized layers. Both CompactifAI variants were applied to an FP16 version; the 93% version adds 4-bit quantization to the untensorized layers of the healed 88% model. The baselines therefore differ in both representation and precision, and the compression labels should not be read as a pure tensorization-only comparison. The paper does not clearly reconcile the “88%” label with the table’s 27.1-to-4.1-GB original-versus-compressed figures, so the label and raw table values are retained without inferring a common denominator. [2]

Reported benchmark scores (accuracy, percent) were: MMLU, 46.41 original / 45.32 88% / 44.16 93%; HellaSwag, 80.55 / 77.87 / 76.54; BoolQ, 79.76 / 77.90 / 76.77; TriviaQA, 19.03 / 18.33 / 18.10; and GSM8K, 23.05 / 22.58 / 17.74. These were measured using the Language Model Evaluation Harness after healing. The reported differences are not uniformly 2–3 percentage points: for the 93% model, the absolute drop versus original ranges from 0.93 points on TriviaQA to 5.31 points on GSM8K. The paper’s headline “2–3%” characterization should therefore be treated as a broad summary, not a per-task bound. [2]

The authors report healing on generic chat datasets including UltraChat, Alpaca, and OpenHermes for less than one epoch on one AWS EC2 instance with eight NVIDIA A10G GPUs using distributed training. They report about 50% faster training (roughly 2×) versus the original and purely quantized baselines on the same amount of MMLU data used for healing, and over 25% faster inference than the original under their distributed evaluation. These are author-reported measurements in that setup, not independent reproduction or a general hardware guarantee. The paper attributes training gains to reduced CPU–GPU transfer time from fewer parameters. It notes that the 4-bit quantized baseline was 13% slower at inference than the original. [2]

The authors’ layer-sensitivity analysis uses MMLU accuracy. In Llama 2 7B, early blocks were more sensitive to compression than middle and later blocks; they advise care below 50% compression for early layers and excluding each block’s last MLP layer from tensorization. Context-aware SVD is described as ongoing work, not as a measured result in this benchmark. [2]

## Compute, limitations, and reproducibility

The reported healing run used eight A10G GPUs on one AWS EC2 instance. The paper does not identify the EC2 instance type, GPU memory, CPU/RAM, runtime or cost for decomposition, peak compression memory, full training schedule, inference batch/context settings, or a portable minimum-hardware requirement. Those requirements are **not reported** in the reviewed primary source. The hardware information is a description of the authors’ experiment, not a minimum requirement. [2]

The evidence is a single-model Llama 2 7B evaluation on five tasks, with dataset-based healing before scoring. The reported metrics support the outcomes in that setup but do not establish equivalence across model families, domains, compression targets, or independent implementations. The paper itself notes layer-local truncations may not be globally optimal and that training from scratch with MPO weights remains untested for comparable accuracy. It says data are available upon reasonable request. The available sources reviewed here do not provide an independent reproduction of the paper’s method or results. QML has not reproduced it. [2]

## Licensing and implementations

The arXiv record provides the paper under an arXiv perpetual non-exclusive distribution license. That is not, by itself, an open-source license for the method’s software or model weights. No CompactifAI weight files or downloadable artifacts for the paper’s Llama 2 experiment were verified in the cited sources. Meta’s Llama 2 weights require accepting Meta’s Llama license and use policy; permission to use those base weights does not grant rights to any separately produced CompactifAI weights. [1] [5] [6]

The official [CompactifAI GitHub organization](https://github.com/CompactifAI) links a repository described as official, but the inspected [CompactifAI repository](https://github.com/CompactifAI/CompactifAI) is an API integration/examples repository: its README documents remote model IDs and API requests. It does not provide the paper’s MPO/SVD compression code or the reported experimental weights. The inspected repository page lists only a README at top level and no license file; no software license for the repository was verified. The API is credential-based; model/API access terms are distinct from a public software or weights license. Official API documentation describes models delivered through the service but does not state a downloadable-weight license on the catalog page. [3] [4]

No reputable public, independently verifiable implementation of the paper’s core MPO compression procedure was found in the sources reviewed. The official API repository is included as a related vendor resource, not represented as an implementation of the paper’s algorithm. A third-party .NET API client appeared in search, but it is an API wrapper, not evidence of an implementation of the research method, and is not listed as a core-method implementation.

## Relevance to QML and overlap

**Established prior art:** CompactifAI demonstrates post-training neural-weight compression via sequential-SVD MPO factorization, rank/bond-dimension truncation, layer-sensitivity profiling, brief recovery training, and optional quantization. Any QML claim involving these same mathematical operations, or a close reimplementation of their use on transformer attention/MLP matrices, should be compared against this publication. This note makes no claim about novelty or ownership by QML. [2]

**QML hypothesis / overlap assessment:** The task names RMQ, MAP, and QMF but supplies no definitions or source records for them. Their specific overlap with CompactifAI is therefore unknown. At a general level, any QML component that compresses weight matrices through tensor-network factorization, truncates singular components or bond ranks, or profiles layer sensitivity may share conceptual or implementation territory with CompactifAI and with earlier tensorized-neural-network work cited by the paper. Whether the overlap is mathematical, algorithmic, or merely terminological cannot be determined without the QML method definitions. Quantization is an explicitly combined technique in the paper, while low-rank approximation and pruning are discussed as neighboring compression families; that does not establish identity between those methods and CompactifAI. [2]

## Evidence status

**Established published method:** the paper and its ESANN 2025 proceedings record establish CompactifAI as a published MPO/tensor-network LLM-compression method. **Independent reproduction:** not verified in this review; QML has not reproduced the method. **QML hypothesis:** potential overlap with RMQ/MAP/QMF is unassessed absent their definitions. **Unknown:** core-method implementation license, weights license for any CompactifAI artifact, and undisclosed compute requirements. [1] [2] [3]

## References

[1]: https://arxiv.org/abs/2401.14109 "arXiv record: CompactifAI: Extreme Compression of Large Language Models using Quantum-Inspired Tensor Networks"
[2]: https://www.esann.org/sites/default/files/proceedings/2025/ES2025-8.pdf "ESANN 2025 proceedings paper: CompactifAI"
[3]: https://github.com/CompactifAI/CompactifAI "Official CompactifAI GitHub repository"
[4]: https://docs.compactif.ai/models/ "CompactifAI API model catalog"
[5]: https://github.com/meta-llama/llama "Meta Llama 2 official repository and model licensing information"
[6]: https://info.arxiv.org/help/license/index.html "arXiv license information"
