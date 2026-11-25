# Safetensors metadata-inspection research note

**Retrieved:** 2026-11-24. **Status:** implementation design input, not a security audit or complete format review.

## Findings from primary sources

The upstream Safetensors repository describes a file as an eight-byte little-endian header-length field, a UTF-8 JSON header, and a raw tensor-data buffer. Each tensor entry identifies a dtype, shape, and `[BEGIN, END]` data offsets; offsets are relative to the tensor buffer and the end is exclusive. The optional `__metadata__` value is a string-to-string map. Upstream documents duplicate names as disallowed, requires the tensor data buffer to be fully indexed without holes, and documents little-endian, row-major data. Its README reports a 100 MiB maximum header size. [1]

The upstream Rust parser has `SafeTensors::read_metadata`, but it accepts a complete file byte slice, parses the header, validates tensor metadata, then verifies that the total data-buffer length matches the file slice. It therefore does not provide a header-only path for very large files without first mapping or reading the entire file. The implementation also uses checked arithmetic for offsets and shape-derived element sizes. [2]

Safetensors 0.8.0 (Apache-2.0) exposes dtype bit widths, including sub-byte F4/F6 representations. Its metadata validation requires the tensor byte ranges, ordered by offsets, to cover the data buffer contiguously; it checks shape/bit-size arithmetic, byte alignment, and that each offset span equals the calculated tensor size. [3] [4]

## Implementation implications

The planned QML inspector reads only the eight-byte length prefix and bounded JSON header from an already-open read-only file. It uses the file's reported length to validate the declared tensor-data region, and does not allocate, map, or read tensor payload bytes. The initial QML header limit will be stricter than upstream's 100 MiB cap and will be disclosed in CLI help and documentation.

The parser must use checked integer arithmetic, reject malformed or duplicate JSON keys, validate supported dtypes and shape-derived bit sizes, reject unaligned tensor sizes, verify exact file length and contiguous non-overlapping offsets, and retain the original file unchanged. Unknown dtypes must be rejected rather than guessed. A generated tiny fixture will be validated against the pinned upstream Rust Safetensors implementation in tests. This does not prove full compatibility with every historical or future writer, and the inspector does not validate numerical values inside tensor payloads.

## References

[1]: https://github.com/huggingface/safetensors/blob/main/README.md "Safetensors upstream README and file-format description"
[2]: https://docs.rs/safetensors/latest/src/safetensors/tensor.rs.html "Safetensors latest Rust parser source"
[3]: https://raw.githubusercontent.com/huggingface/safetensors/v0.8.0/safetensors/src/tensor.rs "Safetensors 0.8.0 Rust tensor and metadata validation source"
[4]: https://crates.io/crates/safetensors "Safetensors crate release and Apache-2.0 license metadata"
