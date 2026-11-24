# Tensor Train decomposition and Matrix Product States

**Index ID:** 11
**Scope:** Tensor Train (TT) decomposition / Matrix Product States (MPS), including one representative downstream neural-network compression application.
**Status:** This is established prior art. It is not QML work, and QML has not reproduced it.

## Foundational publication and dates

The foundational numerical-analysis paper is Ivan V. Oseledets, “Tensor-Train Decomposition,” *SIAM Journal on Scientific Computing*, 33(5), 2295–2317, DOI [10.1137/090752286](https://doi.org/10.1137/090752286). SIAM lists the issue as January 2011; the article was published online on 22 September 2011. It was submitted on 10 March 2009 and accepted on 19 June 2011. No arXiv identifier for this paper was verified; the DOI and SIAM record are the canonical citation. [1] [2]

“Tensor train” and “matrix product state” describe the same chain-shaped factorization, with the names used in different literatures and conventions. The MPS form predates the 2011 TT article in quantum many-body physics; Oseledets’s paper explicitly identifies the TT format with a linear tensor network/MPS and presents a constructive numerical framework for its ranks, rounding, and operations. Thus 2011 is the publication date of this foundational TT treatment, not the origin date of MPS. [2] [3] [4]

## Mathematical mechanism

For an order-`d` tensor with mode sizes `n_1,...,n_d`, TT represents each entry as a product of small three-way cores contracted along shared bond indices:

\[
A(i_1,\ldots,i_d)=\sum_{\alpha_1,\ldots,\alpha_{d-1}}
G_1(1,i_1,\alpha_1)G_2(\alpha_1,i_2,\alpha_2)\cdots G_d(\alpha_{d-1},i_d,1).
\]

The boundary ranks are one; each intermediate bond size \(r_k\) is the TT rank (MPS bond dimension). A tensor with ranks \((r_1,\ldots,r_{d-1})\) stores \(\sum_k r_{k-1}n_kr_k\) values rather than \(\prod_k n_k\), with \(r_0=r_d=1\). For equal mode size \(n\) and bounded interior rank \(r\), this is on the order of \(dnr^2\). The exact minimal rank at cut \(k\) equals the matrix rank of the unfolding that groups modes 1 through \(k\) against the remaining modes. The rank profile therefore depends on the chosen mode ordering and the unfolding spectra. [2] [3]

TT-SVD reshapes the input successively into matrices, performs an SVD at each cut, retains the left singular factors as cores, and passes the residual factor to the next step. Truncating singular values controls approximation error. If \(\epsilon_k\) is the Frobenius norm of the discarded singular-value tail at cut \(k\), then the TT-SVD error is bounded by \(\|A-B\|_F\leq(\sum_{k=1}^{d-1}\epsilon_k^2)^{1/2}\). The resulting approximation is quasi-optimal within a factor \(\sqrt{d-1}\) of the best TT approximation with those rank bounds. For a tensor already represented as TT, TT-rounding uses orthogonalization and a sequence of small SVDs to reduce ranks without materializing the full tensor. [2] [4]

In quantum physics, the physical/site index corresponds to a local state basis and the bond index carries correlations across a bipartition; the unfolding singular values are Schmidt coefficients. In neural-network compression, a dense weight matrix is reshaped into a higher-order tensor and replaced by a TT-matrix/TT-layer; its cores are optimized or fitted in place of the dense matrix. Reshaping is a design choice, not a consequence of TT alone. [3] [5] [6]

## Reported experimental results

**Foundational TT paper (author-reported, not an independent reproduction).** Oseledets reports a Laplace-like operator compression test in Table 3.1. For mode size \(n=2\), TT-rounding times for dimensions \(d=4,8,16,32,64,128\) are respectively \(0.0008, 0.0016, 0.0038, 0.010, 0.051, 0.44\) seconds. For \(n=1024\), times for \(d=4,8,16,32\) are \(0.0023, 0.022, 0.24, 2.3\) seconds; the \(d=64,128\) cases ran out of memory. The operator has TT ranks 2 and the reported approximation error is around machine precision. This test concerns an exactly low-rank structured operator, not a neural model. [2]

The paper also applies TT arithmetic to a 19-dimensional discretized operator eigenproblem, with one-dimensional grid sizes \(n=8,16,32,64\), maximum matrix mode size \(n^2=4096\), operator TT ranks no higher than 10, shifted iteration ranks no higher than 11, and solution TT ranks no higher than 4. The computed minimum-eigenvalue estimates stabilize at approximately \(2.41\times10^3, 2.51\times10^3, 2.56\times10^3, 2.58\times10^3\); relative model-error estimates against the finest grid are reported as \(6\times10^{-2},2.7\times10^{-2},7\times10^{-3}\) (no value for the finest-grid comparison). Average time per iteration, including matrix-vector multiplication and rounding at \(\epsilon=10^{-6}\), is \(0.030,0.036,0.050,0.062\) seconds. The power method used roughly 1,000–2,000 iterations. Importantly, the paper says this was a simplified test of TT rounding and matrix-vector multiplication, not the best eigensolver; the authors state that verifying the computed eigenvalue is actually the smallest eigenvalue remained difficult and that comparison with full tensor solves was under investigation. These values are paper-reported, not QML results or independently reproduced results. [2]

**Neural-network application (separate later prior art).** Novikov et al., “Tensorizing Neural Networks,” submitted to arXiv on 22 September 2015 and revised 20 December 2015, applied TT to fully connected layer weights. For a two-layer MNIST network with 1,024 hidden units, they report test errors of 1.6% with 12,602 total parameters at TT rank 8 and 1.9% with 7,698 parameters at rank 6. Their stated HashedNet comparison on the same architecture uses 12,720 parameters and reports 2.79% test error. These are the paper’s results and baseline, not independent replication. [5] [6]

For ImageNet ILSVRC-2012, the authors replaced the first fully connected layer of VGG-16 with a TT layer (rank 2 in the cited table) and report a 194,622-fold reduction in parameters for that matrix, with VGG-16 top-5 validation error of 11.5%, versus 11.2% for the dense baseline; the whole-network compression factor was 3.9 because the other layers remained. Replacing an additional fully connected layer raised whole-network compression to 7.4, with top-5 error 12.3%. The same paper reports one TT-layer inference timing/memory comparison for a 25,088×4,096 layer at TT rank 4: CPU timings were 16.1 ms/1 image and 97.2 ms/100 images for dense, versus 1.2 ms/1 and 94.7 ms/100 for TT; GPU timings were 2.7/33 ms versus 1.9/12.9 ms, respectively; per-image forward-pass memory was reported as 392 MB dense and 0.766 MB TT. These are experiment-specific results, not universal speed or memory guarantees. [6]

The Tensorizing Neural Networks paper states its implementation used a MATLAB extension of MatConvNet and the TT-Toolbox. It reports a quad-core Intel Core i5-4460, 16 GB RAM, and one NVIDIA GeForce GTX 980 GPU; this describes that study’s environment, not a minimum requirement for TT. [6]

## Computational requirements and limitations

The compact-storage claim holds when the data admit modest TT ranks. The exact rank can be as large as the smaller side of each unfolding, which grows exponentially near the middle for generic tensors. TT-SVD on an explicitly supplied dense tensor still requires that full tensor to be stored/read and can have exponential input-size cost; the Tensor Network reference explicitly cautions that TT-SVD is not suitable when the full tensor cannot fit in memory. Cross/interpolation and structured-operator methods can avoid dense construction, but are additional algorithms with their own assumptions and are not implied by the decomposition itself. [2] [4]

Truncation is approximate, so rank caps or tolerances introduce error. A low-rank representation may fail to exist at the desired tolerance, and operations such as additions or products can grow ranks and require recompression. Bond ranks, numerical conditioning, tensorization/ordering, and workload all affect storage, accuracy, and runtime. A smaller parameter count does not by itself guarantee faster training or inference; contractions may have substantial overhead. TT is a chain topology and can be less suitable than a tree or other network topology when the useful correlations do not align with the selected ordering. [2] [4] [6]

The 2011 source provides operation counts and example timings but does not identify the machine used for those timing tables. No single hardware or memory requirement for the general method was verified. QML has not run or reproduced either the foundational examples or the neural-network experiments. [2] [6]

## Licensing and implementation notes

The SIAM record provides a permissions workflow, and the author-hosted full text carries SIAM copyright and a notice against unauthorized reproduction. No open license for the paper was verified. The 2015 arXiv record links to arXiv’s perpetual non-exclusive distribution license; that grants arXiv distribution rights, not a blanket license for third-party reuse of paper text, code, or model checkpoints. The cited NN paper does not establish a separate license for pretrained weights. The TensorNet repository links pretrained MNIST model and data files externally, but a license for those files was not verified. Treat paper, code, dependencies, datasets, and weights as separately licensed assets. [1] [2] [5] [13] [14]

Reputable implementations located in official repositories/documentation include:

- **ttpy** — Python TT toolbox associated with the method’s author; the repository states MIT. The repository describes TT-SVD, rounding, algebra, and solvers. [7]
- **TensorLy** — Python tensor-learning library; its API documents `tensorly.decomposition.tensor_train` as recursive-SVD TT decomposition. The repository license is BSD 3-Clause. [8] [9]
- **T3F** — TensorFlow TT/MPS library by Novikov and colleagues; its paper describes GPU execution, batch operations, autodifferentiation, and Riemannian optimization, and states the library is MIT licensed. The paper includes implementation benchmarks against TTpy, but those are software-paper benchmarks, not independent replication of the foundational TT theory or QML results. [10] [11]
- **ITensorMPS.jl** — Julia MPS/MPO library based on ITensor; the official repository states Apache-2.0. It targets finite MPS/MPO calculations and quantum many-body workflows. [12]
- **TensorNet**, the original neural-compression authors’ MATLAB and Theano/Lasagne code, is publicly available and references the 2015 paper. The repository page reviewed did not state a license; licensing of its code, linked pretrained models, and data is therefore **not verified**. Its setup describes older MatConvNet/TT-Toolbox dependencies. [13]

These software licenses apply to the respective code repositories only; they do not change the article, dataset, dependency, or weight terms. This note is literature research, not legal advice.

## Relevance to Quantum Morph Laboratory

TT/MPS is directly relevant as established prior art for representing or compressing a high-dimensional tensor, operator, or neural weight tensor as a sequence of low-rank cores. It supplies a mathematically characterized baseline: explicit rank controls compactness, singular-value truncation controls a Frobenius approximation error, and low-rank matrix operations can be performed through core contractions. The 2015 neural-network paper demonstrates that TT weight layers were already explored for neural-model compression, including MNIST and VGG/ImageNet results. QML should index this as prior art, not as a QML method or result. [2] [5] [6]

**Potential overlap with RMQ/MAP/QMF or other known methods:** No specifications or definitions for RMQ, MAP, or QMF were provided in this task, so method-specific overlap cannot be verified. At a general level, TT overlaps any approach that tensorizes model parameters and compresses them through low-rank factors or bottlenecks. It is specifically a chain factorization with SVD-defined unfolding ranks; it is not synonymous with CP/canonical decomposition, Tucker, or hierarchical Tucker, which have different factorization structures and rank notions. Whether any QML method is mathematically equivalent, a special case, or only shares a compression objective is **unknown pending their formal definitions**. This comparison is a QML hypothesis, not a finding that QML developed or reproduced TT/MPS. [2]

## Evidence status

- **Established published method:** Yes — TT decomposition is established prior art with a peer-reviewed 2011 SIAM paper and a formal construction/error analysis; MPS is the corresponding established tensor-network terminology/form used in physics. [1] [2] [3]
- **Independently reproduced by QML:** No. QML has not reproduced the method or cited experiments.
- **QML hypothesis:** Potential relevance as a reference baseline and broad structural overlap with tensorized low-rank compression; specific overlap is unverified.
- **Unknown:** Specific relationship to RMQ/MAP/QMF; whether any historical paper weights have reusable licenses; hardware for the 2011 timing tables.

## References

[1]: https://epubs.siam.org/doi/10.1137/090752286 "Tensor-Train Decomposition, SIAM publisher record"
[2]: https://users.math.msu.edu/users/iwenmark/Teaching/CMSE890/TENSOR_oseledets2011.pdf "Tensor-Train Decomposition, author-hosted full text"
[3]: https://tensornetwork.org/mps/ "Matrix Product State / Tensor Train"
[4]: https://tensornetwork.org/mps/algorithms/ttsvd/ "TT-SVD Algorithm"
[5]: https://arxiv.org/abs/1509.06569 "Tensorizing Neural Networks, arXiv record"
[6]: https://arxiv.org/html/1509.06569v2 "Tensorizing Neural Networks, full text"
[7]: https://github.com/oseledets/ttpy "ttpy Python Tensor Train toolbox"
[8]: https://tensorly.org/stable/modules/generated/tensorly.decomposition.tensor_train.html "TensorLy tensor_train API"
[9]: https://raw.githubusercontent.com/tensorly/tensorly/main/LICENSE.txt "TensorLy BSD 3-Clause license"
[10]: https://github.com/Bihaqo/t3f "T3F Tensor Train decomposition on TensorFlow"
[11]: https://arxiv.org/html/1801.01928v2 "Tensor Train Decomposition on TensorFlow, T3F"
[12]: https://github.com/ITensor/ITensorMPS.jl "ITensorMPS.jl repository"
[13]: https://github.com/Bihaqo/TensorNet "TensorNet authors’ neural TT-layer implementation"
[14]: https://arxiv.org/licenses/nonexclusive-distrib/1.0/ "arXiv non-exclusive license to distribute"
