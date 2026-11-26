# Project Resonance — Sub-3-Bit Neural Representation Synthesis

> **Status:** Active research synthesis (`v0.4.0 — PROJECT RESONANCE`). Strictly classical mathematics (orthogonality, vector/lattice quantization, low-rank residual recovery, activation-weighted distortion). No claim of physical quantum computation.

## 1. The Consumer Hardware Feasibility Equation (27B–32B in 6–8 GB VRAM)

For an open pretrained transformer of $N$ parameters to execute interactively inside a consumer GPU VRAM budget $B_{\text{VRAM}}$ (e.g., 8.0 GiB on an NVIDIA RTX 4070 Laptop GPU, leaving ~1.0–1.5 GiB for OS/display compositor, KV-cache, and compute scratch), the **complete** effective bits per parameter $b_{\text{eff}}$ (counting every scale, zero-point, codebook, index, recovery factor, alignment padding, and manifest byte) must satisfy:

$$\text{Weights}_{\text{GiB}} = \frac{N \cdot b_{\text{eff}}}{8 \cdot 2^{30}}$$

| Model Scale | 4.51 bpp (`Q4_0`) | `QMR-300` (3.00 bpp) | `QMR-250` (2.50 bpp) | `QMR-200` (2.00 bpp) | `QMR-175` (1.75 bpp) | `QMR-150` (1.50 bpp) | Target Consumer Hardware |
|---|---:|---:|---:|---:|---:|---:|---|
| `1.7B` | 0.89 GiB | 0.59 GiB | 0.50 GiB | 0.40 GiB | 0.35 GiB | 0.30 GiB | 4 GB VRAM / CPU |
| `7B` | 3.68 GiB | 2.44 GiB | 2.04 GiB | 1.63 GiB | 1.43 GiB | 1.22 GiB | 4–6 GB VRAM |
| `14B` | 7.35 GiB | 4.89 GiB | 4.07 GiB | 3.26 GiB | 2.85 GiB | 2.44 GiB | 6–8 GB VRAM |
| **`27B`** | 14.18 GiB | 9.43 GiB | 7.86 GiB | **6.29 GiB** | **5.50 GiB** | **4.71 GiB** | **8 GB VRAM (<= 2.00 bpp)** |
| **`32B`** | 16.80 GiB | 11.18 GiB | 9.31 GiB | 7.45 GiB | **6.52 GiB** | **5.59 GiB** | **8 GB VRAM (<= 1.75 bpp)** |

*(With quantized INT8 KV-cache, a 32B GQA model at 4096 context adds only ~0.25–0.50 GiB of KV-cache VRAM, making `QMR-175` / `QMR-150` genuinely executable inside 8 GB dedicated VRAM if intelligence retention can be preserved.)*

---

## 2. Prior-Art Foundations & Mathematical Adaptation in QML

| Research Work | Core Mathematical Mechanism | QML Pure-Rust Adaptation (`v0.4.0`) |
|---|---|---|
| **QuIP#** (Tseng et al., ICML 2024, arXiv:2402.04396) | Blockwise **Randomized Hadamard Transform (RHT)** $W' = W \cdot \text{diag}(s) H_b$ spreads weight outliers across coordinates in $O(n \log n)$ time so distributions become near-Gaussian before 2-bit/lattice quantization. | Implemented exact deterministic `fwht_block_in_place` + SplitMix64 Rademacher signs (zero sign storage overhead). At inference, input $x' = H_b(s \odot x)$ is rotated in $O(k \log b)$ before direct compressed matvec. |
| **AQLM** (Egiazarian et al., ICML 2024, arXiv:2401.06118) & **RVQ** | Multi-codebook **Additive / Residual Vector Quantization** maps $d$-dimensional weight subvectors to sums of learned codebook centroids, achieving fractional rates (`1.0–2.5 bpp`) superior to scalar rounding. | Implemented `VectorQuantMatrix` (1-stage VQ & 2-stage RVQ, $d \in \{4, 8\}$, $K \in \{16..256\}$) with deterministic k-means++ initialization, per-group FP16 norm scales, and **LUT-accelerated matvec** (`dot(codebook[k], x_block)`). |
| **BitNet b1.58** (Ma et al., 2024, arXiv:2402.17764) | Ternary weights $\{-1, 0, +1\}$ carry $\log_2(3) \approx 1.585$ bits of entropy. | Implemented `TernaryPackedMatrix` packing **5 trits into 1 byte** (`3^5 = 243 <= 256`, `1.60 bpp` raw payload) with optimal analytical threshold + per-group FP16 scale + optional Hadamard rotation (PTQ regime; honestly distinguishes PTQ from QAT). |
| **AWQ** (Lin et al., MLSys 2024, arXiv:2306.00978) & **GPTQ** (Frantar et al., ICLR 2023) | Salient activation channels dominate output distortion; protecting top channels or weighting error by activation magnitude preserves reasoning at low bits. | Implemented activation-weighted channel scaling & calibration-guided quantization + **MAP-B** behavior-aware layer importance weighting directly coupled to `qml-behavior`. |
| **PV-Tuning** (Malinovskii et al., NeurIPS 2024, arXiv:2405.14852) & **LoRA/Sparse Error Recovery** | Compact post-quantization corrective parameters recover structured residual distortion $E = W - \hat{W}_0$ without full-model retraining. | Implemented **PROJECT ECHO (`EchoCorrectedMatrix`)**: compact INT8 low-rank SVD adapter ($U_e V_e^\top$) + optional sparse FP16 outlier residual attached to any sub-3-bit base operator, with **100% of recovery bytes counted** in complete artifact BPP. |

---

## 3. The QML Resonance Family (`QMR-300` → `QMR-150`)

Each `QMR-*` profile specifies a **complete serialized artifact BPP ceiling**, NOT a uniform per-tensor bit width:

- **`QMR-300` (≤ 3.00 bpp):** Protects sensitive embeddings, early/late layers, and behavior-critical projections in `INT4`/`INT3+Hadamard` while placing tolerant MLP layers in `Int2+Hadamard+Echo` / `Int3+Hadamard`.
- **`QMR-275` (≤ 2.75 bpp):** Mixed `Int3+Hadamard` + `Int2+Hadamard+Echo` + `Int2+Hadamard`.
- **`QMR-250` (≤ 2.50 bpp):** `Int2+Hadamard` (`g64/g128`) core with `Int3+Hadamard` & `Echo` reserved for high-importance layers (Layer 0, final block, `v_proj`/`o_proj`).
- **`QMR-225` (≤ 2.25 bpp):** `Int2+Hadamard (g128)` + `RVQ-d8` + `Ternary+Hadamard` + targeted `Echo` recovery.
- **`QMR-200` (≤ 2.00 bpp):** Hybrid `Ternary+Hadamard` + `RVQ-d8` + `Int2+Hadamard` on critical blocks.
- **`QMR-175` (≤ 1.75 bpp):** Dominant `Ternary+Hadamard (g128)` + `VQ-d8` with selective `Int2+Hadamard` protection.
- **`QMR-150` (≤ 1.50 bpp):** Extreme sub-2-bit frontier combining `VQ-d8 (1.25 bpp)` + `Ternary+Hadamard (1.72 bpp)` + targeted Echo; rigorously audited for capability cliff boundaries.

### Scientific Honesty & Small-Model Embedding Floor
On small models (e.g., `SmolLM2-135M`, where `embed_tokens` alone is 21.0% of all parameters), keeping embeddings in `INT8` (`8.25 bpp`) consumes `1.73 bpp` globally before a single transformer layer is stored. `MAP-B` therefore evaluates `embed_tokens` inside the same global byte budget with elevated behavioral sensitivity weight so every `QMR` profile genuinely meets its declared complete on-disk BPP budget while documenting exact per-role bit allocation.
