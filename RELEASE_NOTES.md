# Quantum Morph Laboratory `v0.2.0-alpha.2` — PROJECT TRANSCENDENCE (Q-Behavior Mechanistic Observatory & Scalable Q-Genesis)

> *Don't just compress intelligence. Transform its representation.*

**Quantum Morph Laboratory (`v0.2.0-alpha.2`)** expands the workspace to **13 safe-Rust crates (`#![forbid(unsafe_code)]`)** and **11 real-engine desktop views** by launching **Q-Behavior** (layer-by-layer hidden activation tracing, Representational Similarity Analysis, and strictly reversible causal layer interventions) alongside **role-stratified Q-Genesis** with user optimization profiles and direct **Genome-to-QMF v1 export** — **MIT Licensed**, **zero Python dependencies**.

---

## Major Upgrades Implemented

### 1. New 13th Crate: `qml-behavior` — Q-Behavior Mechanistic & Activation Observatory
- Extended **`QCoreModel`** (`qml-runtime`) with **`forward_token_with_trace`** and strictly runtime-only, reversible **`LayerIntervention`** hooks (`ScaleResidual`, `AblateAttention`, `AblateMlp`) — never modifying stored QMF weights.
- Implemented **`compare_model_behavior`** (`qml-behavior`): captures post-block hidden states across every transformer layer for both reference and transformed models, computing per-layer **Mean/Min Cosine Similarity**, **Mean/Max Relative L2 Activation Divergence**, **Representational Similarity Analysis (RSA)** (Pearson correlation of pairwise token similarity geometry), **Peak Drift Layer**, and output **Logit KL Divergence**.
- Implemented **`run_intervention_study`**: measures exact causal KL divergence, top-1 prediction flip rate, and downstream layer activation shifts under reversible layer interventions.
- Added **`qml behavior`** CLI command (`--ablate-mlp`, `--ablate-attn`, `--backend cpu|gpu`, `--json`) and the **11th Real-Engine Desktop Tab (`Q-Behavior Observatory`)** in `qml-desktop` with interactive `egui_plot` layer drift & RSA curves + JSON export.

### 2. Live Mechanistic Discovery on Real `SmolLM2-360M-Instruct` (RTX 4070 GPU)
Running `qml behavior` (`BF16` vs `Q4_0`, 32 layers, GPU Hybrid on RTX 4070) immediately uncovered a concrete mechanistic bottleneck:
- **Layers 00–30:** High representational stability (`Cosine = 0.975–0.995`, `Rel L2 = 0.10–0.20`, `RSA = 0.86–0.998`).
- **Layer 31 (Final Transformer Block):** Sharp representational cliff — **`Rel L2` triples (`0.2038 → 0.6002`)**, **`Cosine` drops (`0.9764 → 0.9064`, min `0.5741`)**, and **`RSA` drops (`0.9813 → 0.5697`)**, pinpointing the exact layer where 4-bit quantization inflicts maximum representational distortion.
- **Reversible Causal Ablation (`AblateMlp { layer: 0 }`):** Measured `Mean Logit KL = 17.23315` and `100.0%` Top-1 flip rate, confirming Layer 0 MLP's causal criticality without mutating any weight file.

### 3. Scalable Whole-Model Q-Genesis & Direct Genome-to-QMF v1 Export (`qml-genesis`, `qml-cli`, `qml-desktop`)
- Replaced the old 6-matrix prefix cap with **`select_stratified_tensor_names`**: samples evenly across **all transformer roles** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) and **early/middle/late network depth** (default `21` stratified matrices).
- Added **`GenesisProfile`** (`Balanced`, `Smallest`, `BestQuality`, `Fastest`) and **`GenesisArchive::select_best(profile)`** (`--profile` CLI flag).
- Added **`export_genome_to_qmf_v1`** (`--export-qmf OUT.qmf`): directly materializes any Pareto-optimal evolved genome into a verified, atomically published **QMF v1** artifact (protecting `embed_tokens`/`lm_head` in `INT8_SYM` and propagating role-consensus genes across full models).

---

## Security, Policy & Quality Verification

- **License:** MIT ([LICENSE](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/blob/main/LICENSE))
- **Rust CI (Ubuntu & Windows, Rust `1.100.0`):** `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and all **79 automated tests** across **13 crates** pass cleanly.
- **Dependency Policy (`cargo-deny 0.21.0`):** `advisories ok, bans ok, licenses ok, sources ok` — zero ignored advisories.

---

## Quick Start (Downloaded Assets)

1. **Windows:** Download `qml-desktop-v0.2.0-alpha.2-windows-x86_64.exe` (or extract `quantum-morph-laboratory-v0.2.0-alpha.2-windows-x86_64.zip`) and double-click `qml-desktop.exe`.
2. **Linux:** Extract `quantum-morph-laboratory-v0.2.0-alpha.2-linux-x86_64.tar.gz` and run `./qml-desktop` or `./qml --help`.
3. **Verify Integrity:** Compare downloaded binaries against `SHA256SUMS.txt`.
