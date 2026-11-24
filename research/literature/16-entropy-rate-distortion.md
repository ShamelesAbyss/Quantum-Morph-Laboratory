# Entropy coding and rate–distortion optimization for neural compression

**Prior-art classification: established published method; this is prior art, not QML work.** QML has not independently reproduced these methods. The phrase covers a research line rather than one single algorithm. Its central precedent is end-to-end learned transform coding with a rate–distortion objective; the 2018 scale-hyperprior paper adds a learned side-information entropy model, and Minnen, Ballé, and Toderici combine that hierarchy with autoregressive context. [1] [2] [3]

## Bibliographic record

The foundational paper is **“End-to-end Optimized Image Compression”** by Johannes Ballé, Valero Laparra, and Eero P. Simoncelli. It appeared at ICLR 2017; its arXiv preprint was submitted November 5, 2016, and revised March 3, 2017. The arXiv identifier is **1611.01704** and its arXiv/DataCite DOI is **10.48550/arXiv.1611.01704**. [1]

The directly relevant hyperprior paper is **“Variational Image Compression with a Scale Hyperprior”** by Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, and Nick Johnston. It appeared in the ICLR 2018 conference track, April 30–May 3, 2018. Its arXiv preprint was submitted February 1, 2018, and revised May 1, 2018. Identifier: **arXiv:1802.01436**; arXiv/DataCite DOI: **10.48550/arXiv.1802.01436**. [2]

A consequential extension is **“Joint Autoregressive and Hierarchical Priors for Learned Image Compression”** by David Minnen, Johannes Ballé, and George Toderici. It appeared in *Advances in Neural Information Processing Systems 31* (NeurIPS 2018), pages 10794–10803; the conference was December 3–8, 2018. Its arXiv preprint was submitted September 8, 2018. Identifier: **arXiv:1809.02736**; arXiv/DataCite DOI: **10.48550/arXiv.1809.02736**. No separate proceedings DOI was identified in the consulted NeurIPS record. [3] [4]

## Mathematical mechanism

A neural analysis transform maps an input image `x` to a latent tensor `y`; quantization gives discrete symbols `ŷ`, and a synthesis transform maps them back to a reconstruction `x̂`. The discrete latents are losslessly encoded using a shared entropy model, typically through arithmetic/range coding. The rate term is the model cross-entropy, approximately `R = E[-log₂ p(ŷ)]`, while distortion `D(x, x̂)` measures reconstruction error. Training minimizes the Lagrangian `R + λD`, where `λ` selects the compression–quality tradeoff. The 2016 work made end-to-end optimization of this transform-coding objective practical by substituting additive uniform noise for nondifferentiable quantization during training, then using actual quantization for compression. [1] [2] [3]

The scale hyperprior introduces another learned latent `z = hₐ(|y|)`. Its quantized version `ẑ` is separately entropy-coded as side information. A hyper-synthesis transform uses `ẑ` to predict location/scale parameters for conditional distributions over the main symbols `ŷ`; the decoder first recovers `ẑ`, reconstructs those parameters, and then decodes `ŷ`. This lets the code adapt its entropy model to the individual image, paying for side bits when they reduce the expected main-latent code length enough. [2] The paper models the main-latent likelihood using a Gaussian convolved with a unit-width uniform distribution, corresponding to quantized symbols and its noisy training relaxation. [2]

Minnen et al. add a causal context model over already-decoded main latents. Their conditional entropy model factors as `p(ŷ | ẑ) = ∏ᵢ p(ŷᵢ | ẑ, ŷ<ᵢ)`, with a masked convolution predicting context features and a network combining those features with the hyperprior to predict Gaussian means and scales. This combines global/spatial information carried by the hyperprior with local causal predictions. Both streams are included in the rate objective; arithmetic coding requires the decoder to reproduce the same conditional probabilities. [3]

## Reported experimental results

The following are **results reported by the papers’ authors**, not independent reproductions by QML. The scale-hyperprior paper evaluates rate–distortion performance on the Kodak image set using PSNR and MS-SSIM and compares to conventional codecs and then-published neural methods. It reports that the hyperprior model trained for squared error surpassed published ANN-based methods in PSNR and came close to HEVC/BPG performance; for MS-SSIM, the model trained on that metric outperformed the compared conventional codecs and prior learned method. The paper emphasizes that training and evaluation distortion metrics must be matched. [2]

For computation, the scale-hyperprior paper reports average encode/decode times in milliseconds for its Python/TensorFlow implementation, with `N=128` or `192` filters, on Kodak and Tecnick datasets. Reported ranges are 331.54–576.34 ms for CPU Kodak encode/decode, 1003.73–1971.85 ms for CPU Tecnick, 242.12–338.09 ms for GPU Kodak, and 491.88–1018.35 ms for GPU Tecnick. These timings were not optimized; the paper says only arithmetic coding used a custom C++ operator and the hyperprior increased runtime by 20%–50% over the factorized-prior model on average. The paper does not identify the CPU/GPU models in this reported table, so it is not a transferable hardware benchmark. [2]

Minnen et al. report on Kodak rate–distortion curves using PSNR and MS-SSIM and state average rate reductions of **15.8%** relative to the previous learned hierarchical baseline, **59.8%** relative to JPEG, **more than 35%** relative to JPEG2000 and WebP in the arXiv abstract (the proceedings abstract states 35% relative to JPEG2000), and **8.4%** relative to BPG. The paper claims the combined model was the first learned method to outperform BPG on both PSNR and MS-SSIM. These are author-reported comparisons under the paper’s image-quality and aggregation setup, not general guarantees for other data or settings. [3] [4]

A material qualification is that Minnen et al. explicitly say they did **not** implement a full decoder for the autoregressive variants when reporting rate–distortion performance. For those models they compared Shannon entropies and empirically checked that they were within a fraction of a percent of actual arithmetic-coded bitstream size. Thus the quoted relative savings are paper-reported rate estimates, not evidence of a full high-performance shipping codec or fast autoregressive decoding. [3]

## Computational requirements and limitations

The models require learned transforms, entropy-model parameter networks, and an entropy coder whose symbol probabilities are synchronized between encoder and decoder. A trained model and its entropy-coding tables/format are part of a usable codec; model parameters and coding metadata must not be omitted when accounting for deployable storage. The cited image-compression papers do not establish direct suitability or a compute budget for compressing neural-network weights. [1] [2] [3]

The principal limitation of the autoregressive extension is serial decoding: each latent prediction depends on previously decoded symbols, limiting parallelism and adding substantial computational cost. The NeurIPS paper discusses lightweight restricted-context variants as a speed–rate tradeoff and notes modest rate increases for such restrictions. The authors also avoided a full autoregressive decoder in their reported comparisons, as noted above. [3] The scale-hyperprior paper’s timing study was explicitly not optimized, and the aggregation methods for PSNR and MS-SSIM differ; it cautions that comparing rate–distortion curves aggregated differently can mislead. [2]

Performance depends on the learned model, training corpus, distortion metric, and `λ`; the methods optimize lossy **image** reconstruction, not arbitrary model-parameter objectives. Neither paper’s results alone establish compression of arbitrary neural models, preservation of downstream model behavior, or a universal compute/storage advantage. These are limitations on transferring the method to QML, not claims that the published image-compression results are invalid. [1] [2] [3]

## Licensing and implementations

The papers are scholarly publications, not software licenses. The arXiv records show the arXiv perpetual non-exclusive distribution license, which should not be treated as a blanket open-source code or weights license. Model checkpoints/weights may have terms separate from the paper and repository; no blanket weights license was verified for the cited implementations. Check each artifact and its upstream dependencies before redistribution or commercial reuse. [1] [2] [4]

The official **TensorFlow Compression** repository includes an implementation of Ballé et al.’s scale-hyperprior model and entropy/range-coding tools. The repository identifies its software as Apache-2.0. Its model script describes itself as educational and warns that reproducing exact paper results may require hyperparameter tuning. The repository README states that TensorFlow Compression entered maintenance mode in February 2024 and that package compatibility is constrained by TensorFlow versions; this is a current maintainer statement, not a paper result. The license/terms of any separately distributed pretrained checkpoint were not established here. [5] [6]

**CompressAI** is a reputable PyTorch library and evaluation platform with implementations named `bmshj2018-hyperprior`, `mbt2018-mean`, and `mbt2018` for the scale-hyperprior and Minnen et al. model families. Its repository declares **BSD 3-Clause Clear** for the software. Its model-zoo documentation describes pretrained model use and training/evaluation setup, but a distinct license for individual pretrained weights was not verified. CompressAI is a third-party implementation/model zoo; its existence is not proof that the original paper’s exact result was independently reproduced. [7] [8]

## Relevance to QML and possible overlap

**Established published method:** the literature establishes end-to-end rate–distortion optimization, entropy coding of quantized neural latents, learned conditional entropy models, and—in later work—hyperprior side information and autoregressive latent context. These are prior art, not QML work, and should be treated as known techniques when assessing any QML compression proposal. QML has not reproduced these papers. [1] [2] [3]

**QML hypothesis / potential overlap:** the project ledger describes RMQ as a proposed shared latent structure with layer-specific transforms and residual information (`Wₗ ≈ Aₗ × S × Bₗ + Rₗ`). That is not the same representation as the papers’ image latents, but entropy-modeling and coding of a residual or side stream could be conceptually relevant if QML stores such streams. The ledger describes MAP as an untested proposal for precision allocation based on functional importance and cross-layer error propagation. The `R + λD` tradeoff is conceptually adjacent to precision allocation under a distortion budget, but the neural image papers’ distortion term is image reconstruction error, not downstream functional sensitivity. Those similarities are hypotheses for comparison, not evidence of method identity or derivation. [2] [3]

**QMF overlap unknown:** the current roadmap lists QMF as aspirational without a mathematical definition, and the supplied master directive is noted as incomplete; there is not enough defined QMF mechanism to assess overlap. Other established neighboring fields include transform coding, variational autoencoders, learned entropy models, quantization, and arithmetic/range coding. More specific mapping requires QMF’s formal objective, representation, and coding procedure. [1] [2] [3]

## References

[1]: https://arxiv.org/abs/1611.01704 "End-to-end Optimized Image Compression, arXiv record"
[2]: https://arxiv.org/html/1802.01436v2 "Variational Image Compression with a Scale Hyperprior, full paper"
[3]: https://proceedings.neurips.cc/paper_files/paper/2018/file/53edebc543333dfbf7c5933af792c9c4-Paper.pdf "Joint Autoregressive and Hierarchical Priors for Learned Image Compression, NeurIPS 2018 paper"
[4]: https://arxiv.org/abs/1809.02736 "Joint Autoregressive and Hierarchical Priors, arXiv record"
[5]: https://github.com/tensorflow/compression "Official TensorFlow Compression repository and README"
[6]: https://github.com/tensorflow/compression/blob/master/models/bmshj2018.py "TensorFlow Compression scale-hyperprior model implementation"
[7]: https://github.com/InterDigitalInc/CompressAI "CompressAI repository and license declaration"
[8]: https://interdigitalinc.github.io/CompressAI/zoo.html "CompressAI model-zoo documentation"
