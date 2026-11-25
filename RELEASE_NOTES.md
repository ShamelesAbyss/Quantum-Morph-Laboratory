# Quantum Morph Laboratory `v0.1.0-alpha.2` — PROJECT TRANSCENDENCE (Wave I: Out-of-Core Streaming, Atomic Safety & GGUF Payload Decoding)

> *Don't just compress intelligence. Transform its representation.*

**Quantum Morph Laboratory (`v0.1.0-alpha.2`)** resolves all remaining independent-audit stability and memory-scaling deficiencies from `v0.1.0-alpha.1` while expanding real-model format support — written in **100% safe Rust (`#![forbid(unsafe_code)]`)**, **MIT Licensed**, with **zero Python dependencies**.

---

## Major Upgrades Implemented

### 1. Out-of-Core File-Backed QMF v1 Streaming Reader (`qml-format`, `qml-runtime`, `qml-cli`, `qml-desktop`)
- Added **`QmfFileReader<R: Read + Seek>`**: opens multi-GB QMF v1 files while holding **zero payload bytes in RAM** (only the JSON manifest), lazily seeking and verifying individual tensor blobs on demand with `QmfLatentCache` `Arc` deduplication.
- Added **`QmfFileReader::verify_integrity_streaming()`**: validates whole-file CRC-32 and computes exact FIPS 180-4 SHA-256 in constant 64 KiB chunks (`O(1)` memory) without cloning full archives.
- Switched **`QCoreModel::from_qmf_reader`**, **`qml generate`** (including `--compare-qmf`), and **`qml-desktop` Q-Core Inference Chamber** to stream tensors directly from disk via `QmfFileReader`.

### 2. Atomic QMF Publication & Source-Output Alias Protection (`qml-format`, `qml-cli`, `qml-desktop`)
- Added **`QmfArchiveV1::write_to_file_atomic(output_path, source_path, budget)`**: canonicalizes paths to reject any attempt to overwrite an immutable source model (`QmfError::SourceOutputAlias`), streams CRC-32 + SHA-256 into a sibling temporary file, calls `sync_all()`, and atomically replaces the destination path so interrupted or failed runs never leave partial `.qmf` artifacts.

### 3. Two-Pass Streaming MAP & True Cooperative Q-Forge Cancellation (`qml-quant`, `qml-forge`, `qml-desktop`)
- Refactored **MAP (Morphological Adaptive Precision)** into a genuinely **two-pass streaming pipeline** (`MapLayerProfile`, `profile_layer_for_map`, `allocate_from_profiles`) — profiling one matrix at a time and dropping decoded dense weights immediately before encoding only each tensor's selected tier.
- Added **`ForgeControl`** (`Arc<AtomicBool>` cancellation + live per-tensor progress callback) and `ForgeError::Cancelled`, wired across single-tensor, two-pass MAP, and RMQ cluster loops so clicking **Cancel** in `qml-desktop` genuinely aborts the background transformation worker.
- Wired **`ResourcePreset`** (`Conservative` / `Balanced` / `Performance`) directly into `PayloadBudget::max_total_bytes` so host memory presets enforce real computation bounds.

### 4. Bounded GGUF v2/v3 Quantized Payload Decoding (`qml-model`)
- Implemented **`GgufFile<R: Read + Seek>`** for bounded, read-only tensor payload decoding of real GGUF models supporting **`F32`**, **`F16`**, **`BF16`**, **`Q8_0`**, **`Q4_0`**, and **`Q4_1`** GGML block layouts with `general.alignment` offset calculation and strict `PayloadBudget` enforcement.

### 5. Explicit Opt-In Model Discovery & Documentation Alignment (`qml-desktop`, `HANDOFF.md`, `README.md`)
- Removed automatic first-launch directory scanning in `qml-desktop`: model discovery is now **strictly user-initiated** via explicit import or folder scan actions.
- Updated `HANDOFF.md`, `CHANGELOG.md`, and `README.md` to eliminate obsolete pre-Singularity caveats while preserving full scientific honesty and the Permanent Memorial Dedication.

---

## Security, Policy & Quality Verification

- **License:** MIT ([LICENSE](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/blob/main/LICENSE))
- **Rust CI (Ubuntu & Windows, Rust `1.100.0`):** `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and all **79 automated tests** pass cleanly.
- **Dependency Policy (`cargo-deny 0.21.0`):** `advisories ok, bans ok, licenses ok, sources ok` — zero ignored advisories.

---

## Quick Start (Downloaded Assets)

1. **Windows:** Download `qml-desktop-v0.1.0-alpha.2-windows-x86_64.exe` (or extract `quantum-morph-laboratory-v0.1.0-alpha.2-windows-x86_64.zip`) and double-click `qml-desktop.exe`.
2. **Linux:** Extract `quantum-morph-laboratory-v0.1.0-alpha.2-linux-x86_64.tar.gz` and run `./qml-desktop` or `./qml --help`.
3. **Verify Integrity:** Compare downloaded binaries against `SHA256SUMS.txt`.

> **Experimental Notice:** QMF v0/v1, RMQ, MAP, Q-Genesis, and the WGPU compute kernels are active experimental research prototypes. Imported model weights are always opened strictly read-only.
