# QMF v1 experimental binary layout

**Status:** Experimental generalized multi-tensor container with shared latent structure support. Not yet compatibility-stable; breaking revisions remain permitted until formal stabilization. CRC-32 provides accidental-corruption detection only and does not provide cryptographic authentication.

## Binary Layout

All integers and IEEE-754 floating-point values are little-endian.

| Offset | Length | Field |
|---:|---:|---|
| 0 | 4 | Magic bytes `QMF\0` |
| 4 | 2 | Format version (`1`) |
| 6 | 2 | Flags (`0x0001` = little-endian; all other bits must be zero) |
| 8 | 8 | `manifest_len` (`u64`, bounded `<= 16 MiB`) |
| 16 | 8 | `payload_len` (`u64`, bounded by total file ceiling) |
| 24 | 8 | Reserved (`u64`, must be zero) |
| 32 | `manifest_len` | Canonical deterministic UTF-8 JSON `QmfManifest` |
| `32 + manifest_len` | `payload_len` | Contiguous binary blobs referenced by `(offset, len, crc32)` |
| End − 4 | 4 | Whole-file CRC-32/ISO-HDLC over all preceding bytes |

## Supported Representations

- **Shared Latent Table (`shared_latents`):** Stores shared matrices (`U_shared`, `V_shared`, or shared cores) once as `fp32`, `fp16`, or `int8_sym` blocks, referenced by ID from multiple layers.
- **Per-Tensor Representations (`tensors`):**
  - `dense_f32`, `dense_f16`, `dense_bf16`
  - `int8_sym_group` (f16 scales + i8 codes)
  - `int4_sym_group` (Q4_0-style f16 scales + packed 4-bit nibbles)
  - `int4_asym_group` (Q4_1-style f16 scales + f16 mins + packed 4-bit nibbles)
  - `low_rank_svd` (quantized left/right factors)
  - `rmq_shared_basis` (references `u_latent_id` and `vt_latent_id`, stores compact per-layer quantized core `C_l` and optional sparse COO FP16 residual `R_l`)

## Safety Invariants

Parsers enforce strict bounds (`manifest_len <= 16 MiB`, total size `<= 256 MiB`), checked integer arithmetic, exact file length match, zero reserved bits, valid UTF-8 manifest JSON, duplicate tensor/latent rejection, valid shared latent ID resolution, finite floating-point checks, per-blob CRC-32 verification, and whole-file CRC-32 verification.
