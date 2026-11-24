# Tensor Ring decomposition — prior art

**Classification:** established published method; this is prior art, not Quantum Morph Laboratory work. QML has not independently reproduced it, as specified for this research task. Any relevance or overlap with QML-specific methods below is a hypothesis, not an established QML result.

## Bibliographic record

The primary work is Qibin Zhao, Guoxu Zhou, Shengli Xie, Liqing Zhang, and Andrzej Cichocki, “Tensor Ring Decomposition.” The canonical record is arXiv:1606.05535, first submitted 17 June 2016. The arXiv record supplies DOI 10.48550/arXiv.1606.05535. The author’s publication page lists it as an arXiv preprint (2016); no journal or conference publication for this work is identified by those records. [1] [2] [3]

## Mathematical mechanism

For a tensor \(\mathcal{T}\) of order \(d\), with mode sizes \(n_k\), TR represents each entry as a trace of a cyclic product of matrix slices from third-order cores \(\mathcal{Z}_k\in\mathbb{R}^{r_k\times n_k\times r_{k+1}}\), where \(r_{d+1}=r_1\):

\[
T(i_1,\ldots,i_d)=\operatorname{Tr}\left[Z_1(i_1)Z_2(i_2)\cdots Z_d(i_d)\right].
\]

Equivalently, the shared bond indices are summed around a closed cycle. The trace removes the need for the first/last bond ranks to be one, unlike the open-ended Tensor Train (TT) representation. The authors describe TR as invariant to circular shifts of mode ordering and, when opened at a bond, as a sum of \(r_1\) TT representations sharing the interior cores. The paper also derives relationships to classical CP and Tucker decompositions. Storage is \(\sum_k r_k n_k r_{k+1}\); with equal mode size \(n\) and rank \(r\), this is \(O(dnr^2)\). [1]

The paper gives four optimization approaches: sequential-SVD TR-SVD, TR-ALS with ranks supplied by the user, and rank-adaptive ALS variants TR-ALSAR and TR-BALS. TensorLy’s documented `tensor_ring` API is an implementation using recursive SVD; it is not a claim of independent experimental reproduction of the paper’s results. [1] [4]

## What the original paper actually reports

The paper evaluates synthetic signals and two real datasets, not neural-network compression. For synthetic data, three highly oscillatory functions were tensorized as 10th-order tensors of shape \(4^{10}\); a noisy version of one function at SNR 60 dB was also tested. With an error tolerance of \(10^{-3}\), the paper compares relative error, average rank, parameter count, and execution time for TT-SVD and the proposed TR algorithms. It reports that TR-SVD and TT-SVD had comparable error/rank/storage and execution time on the clean functions, while TR-SVD used fewer parameters on the noisy signal. TR-BALS was reported as more accurate/compact/faster than TR-ALSAR in those tests. These are the authors’ results under their specified implementation and setup, not independent replications. [1]

For COIL-100, 7,200 color object images were downsampled to 32×32×3 and arranged as a 32×32×3×7,200 tensor. A 1-nearest-neighbor classifier used a core as features; train ratios were 50% and 10%, and results were averaged over ten random splits. TR-SVD’s best listed accuracies were 99.19% at 50% training and 89.89% at 10% training (both at relative error about 0.3). The listed best TT-SVD accuracies were 99.05% and 89.11%; CP-ALS listed 97.56% and 83.38%. The paper also reports smaller TR ranks at comparable reconstruction errors. [1]

For KTH action videos, six actions from 25 subjects and four scenarios were downsampled to 20×20×32 and organized as 20×20×32×600. Classification used 5×5-fold cross-validation. Best listed accuracies were 87.7% for TR-SVD (relative error about 0.29), 87.3% for TR-ALS, 87.0% for TR-BALS, 84.8% for TT-SVD, and 80.8% for CP-ALS. The paper explicitly cautions that these results might not be state of the art on KTH and emphasizes comparison among decomposition frameworks. [1]

All reported computations used a Windows workstation with a 3.33 GHz Intel Xeon CPU, 64 GB memory, and MATLAB R2012b. This is the paper’s experimental environment, not a general minimum requirement or a measured resource profile for TensorLy or QML. [1]

## Limitations, licensing, and implementation status

The paper identifies practical tradeoffs among its algorithms. TR-SVD is described as efficient and stable, but its rank profile tends to be smaller at the ends and larger toward the middle, which may not fit a given tensor well. TR-ALS requires manually fixed ranks. The adaptive ALS algorithms cost more and may converge slowly; in the KTH experiment, TR-ALSAR and TR-BALS did not approximate the data at arbitrary requested error bounds. These experiments concern tensor approximation and feature extraction, not a neural-model compression pipeline. [1]

The arXiv page records a non-exclusive distribution license granted to arXiv; that is not, by itself, an open-content reuse license for arbitrary purposes. I found no paper-specific software license or pretrained neural weights associated with this work in the checked primary record and author publication listing. The paper does not report released pretrained neural weights. [2] [3] [6] [7]

A reputable implementation is TensorLy’s `tensor_ring` decomposition. Its API documentation cites Zhao et al. and describes recursive-SVD decomposition of an input tensor into third-order factors. The TensorLy repository’s license file states BSD 3-Clause. This software license applies to TensorLy; it should not be conflated with the paper’s rights or any unlocated third-party implementation. TensorLy is a Python library, so it is a reference implementation rather than a native Rust implementation. [4] [5]

## Relevance to QML and possible overlap

**QML hypothesis:** TR is relevant as prior art for representing a large tensor through connected low-dimensional cores, and the paper’s compression/feature-extraction results show a general tensor-factorization approach. That is not evidence that TR compresses neural models well: the original experiments are on synthetic tensors, images, and videos, with no neural-network weight tensors or neural-model accuracy/latency evaluation. [1]

**Potential overlap:** the clearest established overlap is with TT / matrix-product-state tensor networks: the paper explicitly treats TT as a special case of TR and TR as a sum of TT forms. It also relates TR to CP and Tucker. For RMQ, MAP, QMF, or other QML-local names, no definitions or source material were provided in this task, so any equivalence or implementation overlap is **unknown**. If those methods use cyclic low-rank tensor cores or equivalent trace contractions, similarity would be a hypothesis requiring comparison of their actual equations and code, not evidence of QML originality. [1]

## References

[1]: https://ar5iv.labs.arxiv.org/html/1606.05535 "Tensor Ring Decomposition full text"
[2]: https://arxiv.org/abs/1606.05535 "Tensor Ring Decomposition arXiv record"
[3]: https://qibinzhao.github.io/publications/003-publications/ "Qibin Zhao publication listing"
[4]: https://tensorly.org/stable/modules/generated/tensorly.decomposition.tensor_ring.html "TensorLy tensor_ring API"
[5]: https://raw.githubusercontent.com/tensorly/tensorly/main/LICENSE.txt "TensorLy BSD 3-Clause license"
[6]: https://arxiv.org/licenses/nonexclusive-distrib/1.0/ "arXiv non-exclusive distribution license"
[7]: https://arxiv.org/abs/1606.05535 "arXiv record and rights information"
