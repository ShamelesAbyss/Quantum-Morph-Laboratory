# Quantum Morph Laboratory `v0.3.0-alpha.1` — PROJECT PRISM (The Architecture of Intelligence, Made Visible)

> *Don't just compress intelligence. Transform its representation.*

**Quantum Morph Laboratory (`v0.3.0-alpha.1` — PROJECT PRISM)** transforms `qml-desktop` into a GPU-rendered, multi-pass native **WGPU** 3D scientific observatory — giving all **13 safe-Rust crates (`#![forbid(unsafe_code)]`)** and **11 real-engine laboratory workspaces** a living, physically coherent visual identity while preserving every validated mathematical, inference, and experimental capability (**MIT Licensed**, **zero Python dependencies**, **zero ignored advisories**).

---

## Major Visual & Architectural Upgrades Implemented

### 1. Native Multi-Pass WGPU 3D Scientific Renderer (`crates/qml-desktop/src/prism/`)
- Migrated `qml-desktop` from OpenGL (`glow`) to a dedicated native **WGPU (`wgpu 24`, DX12 / Vulkan / Metal)** rendering pipeline via `eframe::egui_wgpu::CallbackTrait`, with graceful automatic 2.5D CPU fallback on headless/minimal systems.
- Authored **7 original WGSL shaders** (`prism/shaders.rs`), validated directly by automated unit tests on live WGPU hardware:
  - **`BACKDROP_WGSL`**: Deep obsidian & ultraviolet procedural space with subtle polar/Cartesian interference reticles, Singularity gravitational-lensing accretion horizon, and circular Gaussian pinpoint starfield.
  - **`CRYSTAL_WGSL`**: Depth-aware (`Depth32Float`) GPU-instanced 3D crystalline meshes (Octahedra, Prism Slabs, UV-Spheres, Torus Rings) with Fresnel iridescence, dual studio lighting, and HDR emission.
  - **`FILAMENT_WGSL`**: Camera-aligned luminous 3D neural pathway ribbons with traveling spectral energy pulses.
  - **`PARTICLE_WGSL`**: GPU-driven Keplerian orbital particle halos modulated by live compute activity.
  - **`BLOOM_THRESHOLD_WGSL`**, **`BLOOM_BLUR_WGSL`**, and **`COMPOSITE_WGSL`**: Full **`Rgba16Float` HDR** offscreen render target + half-resolution 2-pass separable 9-tap Gaussian selective bloom + ACES filmic tone mapping.
- Implemented a pure-Rust interactive **`OrbitCamera`** (`prism/camera.rs`) supporting drag-to-orbit, Shift/right-drag pan, scroll zoom, double-click reset, Perspective / Orthographic projection toggle (`P`), fullscreen 3D expansion (`F`), and **3D ray-sphere picking** with screen-projected scientific callouts and provenance badges.

### 2. Seven Real-Data & Schematic 3D Scientific Workspaces (`prism/scenes.rs`, `views.rs`)
- **Command Center — The Computational Singularity:** Interactive 3D orbital observatory core (explicitly labeled `ARTISTIC LABORATORY IDENTITY`) where every orbiting laboratory subsystem node is 3D-clickable to jump directly into its workspace, running at **155–187 FPS** on NVIDIA RTX 4070 (`2560×1440` native WGPU verified).
- **Q-Scope — 3D Neural Anatomy:** Real layered 3D transformer tower mapping actual measured `ModelScopeReport` tensors into architectural crystals across **6 live scientific inspection modes** (`Architectural Anatomy`, `Spectral Structure`, `Quantization Sensitivity`, `Parameter Distribution`, `Numerical Outliers (6σ)`, `Compression Eligibility`) with click-to-inspect histograms & SVD spectra.
- **Q-Entangle — The Latent Constellation:** Spatial 3D helical Grassmannian subspace affinity constellation (explicitly distinguishing classical principal-angle subspace geometry from physical quantum entanglement) paired with an Obsidian → Ultraviolet → Cyan affinity heatmap and clickable layer-pair inspection.
- **Q-Forge — The Morphogenesis Chamber:** Split 3D transformation chamber depicting immutable read-only dense weight slabs morphing across a central shared-latent manifold into compact QMF v1 blocks, coupled to live transformation progress and exact byte accounting.
- **Q-Genesis — The Evolutionary Crucible:** Interactive 3D multi-objective Pareto evolution space (`Effective BPP × Matvec Rel Error × Estimated FLOPs`) where organism geometry encodes dominant representation family (`Octahedron = RMQ`, `Sphere = SVD`, `Slab = Quant`) and filaments strictly trace actual recorded `parent_ids` lineage, with generation scrubbing and Profile Winner highlighting.
- **Q-Behavior — Dual Activation Observatory:** Parallel 3D Reference vs Transformed transformer columns bridged by actual measured cosine similarity, relative L2 drift, and RSA telemetry (labeled explicitly as classical activation science).
- **System Observatory — Computational Anatomy & Graphics Presets:** 3D CPU/WGPU silicon topology schematic paired with persisted **PROJECT PRISM Graphics Quality Presets** (`Minimal`, `Balanced`, `High`, `Ultra`, `Cinematic`), resolution scale, particle density, bloom intensity, animation intensity, reduced motion toggle, FPS cap (`30/60/120/144`), and live measured frame rate — strictly honoring telemetry honesty (never fabricating GPU temperature, power draw, or OS-wide VRAM usage).

### 3. Global Laboratory UX, Custom Vector Branding & Visual Polish
- Procedural 64×64 RGBA application icon, custom animated orbital PRISM emblem (`assets/qml-prism-logo.svg`), and **11 original procedural vector subsystem icons** in a collapsible navigation rail (`Ctrl+B`).
- Breadcrumb header bar with live measured FPS (`PRISM WGPU • FPS`), global keyboard shortcuts (`1–9/0/-`, `Ctrl+B`, `F`, `P`, `R`, `Esc`), proportional QMF storage treemap & precision ring in QMF Explorer, and distraction-free **Zen Mode** + ambient compute pulse strip in Q-Core Inference.
- Visual validation performed directly on native Windows WGPU (`2560×1440` and `1920×1080`) with real `SmolLM2-135M-Instruct` Q-Scope/Q-Entangle measurements and live Q-Genesis evolution; all 6 genuine GPU screenshots are archived in `assets/screenshots/`.

---

## Security, Policy & Quality Verification

- **License:** MIT ([LICENSE](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/blob/main/LICENSE))
- **Rust CI (Ubuntu & Windows, Rust `1.100.0`):** `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and all **81 automated tests** across **13 crates** (including live WGPU shader & pipeline compilation validation) pass cleanly.
- **Dependency Policy (`cargo-deny 0.21.0`):** `advisories ok, bans ok, licenses ok, sources ok` — zero ignored advisories.

---

## Quick Start (Downloaded Assets)

1. **Windows:** Download `qml-desktop-v0.3.0-alpha.1-windows-x86_64.exe` (or extract `quantum-morph-laboratory-v0.3.0-alpha.1-windows-x86_64.zip`) and double-click `qml-desktop.exe`.
2. **Linux:** Extract `quantum-morph-laboratory-v0.3.0-alpha.1-linux-x86_64.tar.gz` and run `./qml-desktop` or `./qml --help`.
3. **Verify Integrity:** Compare downloaded binaries against `SHA256SUMS.txt`.
