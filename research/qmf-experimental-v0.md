# QMF v0 experimental binary layout

**Status:** unstable prototype for a single rank-one matrix representation. It is not a model container, release format, or compatibility promise. Do not distribute important artifacts in this format. The implementation is in `crates/qml-format`; this note must be versioned with any incompatible change.

## Encoded layout

All integer and floating-point fields are little-endian. The fixed 24-byte header is followed by the left factor (`rows` IEEE-754 binary64 values), the right factor (`cols` binary64 values), and a four-byte CRC-32/ISO-HDLC checksum over the header and payload.

| Offset | Length | Field |
|---:|---:|---|
| 0 | 4 | Magic bytes `QMF\0` |
| 4 | 2 | Format version (`0`) |
| 6 | 1 | Representation identifier (`1` = rank-one `f64` matrix) |
| 7 | 1 | Reserved; must be zero |
| 8 | 8 | Row count (`u64`) |
| 16 | 8 | Column count (`u64`) |
| 24 | `8 × rows` | Left factor values |
| `24 + 8 × rows` | `8 × cols` | Right factor values |
| End − 4 | 4 | CRC-32 of every preceding byte |

The represented matrix is `left × rightᵀ`; dense reconstruction is not required for matrix-vector multiplication. v0 stores no original model, tokenizer, architecture metadata, residuals, or experiment lineage.

## Parser and integrity rules

The implementation rejects unknown versions and representation identifiers, nonzero reserved fields, zero or oversized dimensions, integer-size overflow, non-exact file lengths, non-finite factor values, payloads above the configured 256 MiB ceiling, and checksum mismatch. The current maximum dimension per axis is 16,777,216, subject also to the total file-size ceiling. CRC-32 detects accidental corruption only; it is not cryptographic authentication and does not protect against malicious edits.

Serialization is deterministic for identical factor values. No forward/backward compatibility policy is established. A future format may replace this layout entirely. Do not claim QMF is a general model format until tensor directories, model metadata, tokenizer references, integrity/security design, versioning policy, and compatibility tests exist.

## Scope of the demonstration

`qml demo-rank1` generates a small synthetic rank-one matrix, applies a deterministic largest-pivot cross approximation, writes and reloads this representation, compares matrix-vector outputs, and reports storage/error values for that fixture. The pivot approximation is a baseline and is not guaranteed to minimize Frobenius error for arbitrary matrices. The demo is not a pretrained model, real-model compression result, QML innovation claim, or general benchmark.
