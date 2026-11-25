//! Bounded, read-only model format inspection and tensor payload access.
//!
//! Supports:
//! - Safetensors metadata-only inspection (bounded 16 MiB header cap, never reads payloads)
//! - Safetensors bounded & streaming tensor payload access with safe dtype conversion
//! - Sharded Safetensors index (`model.safetensors.index.json`) parsing
//! - Hugging Face `config.json` architecture metadata parsing
//! - Bounded, read-only GGUF v2/v3 metadata & tensor-directory inspection

pub mod config;
pub mod dtype;
pub mod gguf;
pub mod payload;

pub use config::{MAX_CONFIG_BYTES, ModelConfig};
pub use dtype::{
    DType, bf16_to_f32, decode_chunk_to_f64, f8_e4m3_to_f32, f8_e5m2_to_f32, f16_to_f32,
    f32_to_bf16, f32_to_f16,
};
pub use gguf::{
    GGUF_MAGIC, GgufError, GgufMetadata, GgufTensorInfo, GgufValueSummary, ggml_type_name,
    inspect_gguf,
};
pub use payload::{
    DEFAULT_MAX_TENSOR_BYTES, DEFAULT_MAX_TOTAL_BYTES, DEFAULT_STREAM_CHUNK_ELEMENTS,
    PayloadBudget, PayloadError, SafetensorsFile, SafetensorsShardIndex, encode_safetensors,
    parse_shard_index, tensor_element_count,
};

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Read, Seek, SeekFrom};

/// Maximum header allocation accepted by the default inspector (16 MiB).
pub const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;
/// Maximum number of tensor entries accepted after parsing the bounded header.
pub const MAX_TENSORS: usize = 100_000;
/// Maximum number of dimensions accepted for any tensor.
pub const MAX_TENSOR_RANK: usize = 64;

/// Validated, deterministic summary of a Safetensors file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SafetensorsMetadata {
    /// Total file size, including the prefix, header, and tensor payload.
    pub file_bytes: u64,
    /// Declared JSON header size, including any trailing JSON whitespace.
    pub header_bytes: u64,
    /// Tensor payload size. Payload bytes were not read.
    pub data_bytes: u64,
    /// Tensor entries sorted by name for deterministic output.
    pub tensors: Vec<TensorMetadata>,
    /// Optional file metadata; absent metadata is reported as an empty map.
    pub metadata: BTreeMap<String, String>,
}

/// Validated metadata for one tensor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TensorMetadata {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Half-open byte range within the data buffer.
    pub data_offsets: [u64; 2],
    /// Computed tensor payload size in bytes.
    pub data_bytes: u64,
}

/// Errors produced while inspecting an untrusted Safetensors file.
#[derive(Debug)]
pub enum InspectError {
    Io(io::Error),
    FileTooSmall,
    HeaderTooLarge {
        declared: u64,
        limit: u64,
    },
    InvalidHeaderLimit {
        requested: u64,
        hard_limit: u64,
    },
    HeaderLengthOverflow,
    AllocationFailed,
    InvalidJson(String),
    RankTooLarge {
        tensor: String,
        rank: usize,
        limit: usize,
    },
    InvalidTensor {
        tensor: String,
        reason: String,
    },
    InvalidDataLayout(String),
}

impl fmt::Display for InspectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::FileTooSmall => {
                write!(formatter, "file is shorter than the 8-byte header prefix")
            }
            Self::HeaderTooLarge { declared, limit } => write!(
                formatter,
                "declared header is {declared} bytes; configured limit is {limit} bytes"
            ),
            Self::InvalidHeaderLimit {
                requested,
                hard_limit,
            } => write!(
                formatter,
                "header limit {requested} exceeds the hard limit of {hard_limit} bytes"
            ),
            Self::HeaderLengthOverflow => {
                write!(formatter, "header length does not fit this platform")
            }
            Self::AllocationFailed => {
                write!(formatter, "could not reserve memory for bounded header")
            }
            Self::InvalidJson(message) => {
                write!(formatter, "invalid Safetensors JSON header: {message}")
            }
            Self::RankTooLarge {
                tensor,
                rank,
                limit,
            } => write!(
                formatter,
                "tensor {tensor:?} has rank {rank}; configured limit is {limit}"
            ),
            Self::InvalidTensor { tensor, reason } => {
                write!(formatter, "invalid tensor {tensor:?}: {reason}")
            }
            Self::InvalidDataLayout(message) => {
                write!(formatter, "invalid tensor data layout: {message}")
            }
        }
    }
}

impl std::error::Error for InspectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for InspectError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Inspect a seekable Safetensors file using the default 16 MiB header cap.
///
/// The reader is sought to the beginning and then to the end to determine file
/// size. Only the prefix and declared header are read; tensor data is never
/// accessed. The input is treated as untrusted.
pub fn inspect_safetensors<R: Read + Seek>(reader: R) -> Result<SafetensorsMetadata, InspectError> {
    inspect_safetensors_with_header_limit(reader, MAX_HEADER_BYTES)
}

/// Inspect a Safetensors file with a stricter caller-selected header cap.
///
/// `max_header_bytes` may be smaller than [`MAX_HEADER_BYTES`], but not larger.
pub fn inspect_safetensors_with_header_limit<R: Read + Seek>(
    mut reader: R,
    max_header_bytes: u64,
) -> Result<SafetensorsMetadata, InspectError> {
    if max_header_bytes > MAX_HEADER_BYTES {
        return Err(InspectError::InvalidHeaderLimit {
            requested: max_header_bytes,
            hard_limit: MAX_HEADER_BYTES,
        });
    }

    let file_bytes = reader.seek(SeekFrom::End(0))?;
    if file_bytes < 8 {
        return Err(InspectError::FileTooSmall);
    }
    reader.seek(SeekFrom::Start(0))?;

    let mut length_prefix = [0_u8; 8];
    reader.read_exact(&mut length_prefix)?;
    let header_bytes = u64::from_le_bytes(length_prefix);
    if header_bytes > max_header_bytes {
        return Err(InspectError::HeaderTooLarge {
            declared: header_bytes,
            limit: max_header_bytes,
        });
    }
    let header_len =
        usize::try_from(header_bytes).map_err(|_| InspectError::HeaderLengthOverflow)?;
    let header_end = 8_u64
        .checked_add(header_bytes)
        .ok_or(InspectError::HeaderLengthOverflow)?;
    if header_end > file_bytes {
        return Err(InspectError::InvalidDataLayout(format!(
            "header ends at byte {header_end}, beyond file length {file_bytes}"
        )));
    }

    let mut header = Vec::new();
    header
        .try_reserve_exact(header_len)
        .map_err(|_| InspectError::AllocationFailed)?;
    header.resize(header_len, 0);
    reader.read_exact(&mut header)?;

    let raw: RawHeader = serde_json::from_slice(&header)
        .map_err(|error| InspectError::InvalidJson(error.to_string()))?;

    let data_bytes = file_bytes - header_end;
    let mut tensors = Vec::with_capacity(raw.tensors.len());
    for (name, raw_tensor) in raw.tensors {
        if raw_tensor.shape.len() > MAX_TENSOR_RANK {
            return Err(InspectError::RankTooLarge {
                tensor: name,
                rank: raw_tensor.shape.len(),
                limit: MAX_TENSOR_RANK,
            });
        }
        let bits_per_element =
            dtype_bits(&raw_tensor.dtype).ok_or_else(|| InspectError::InvalidTensor {
                tensor: name.clone(),
                reason: format!("unsupported dtype {:?}", raw_tensor.dtype),
            })?;
        let element_count = raw_tensor
            .shape
            .iter()
            .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
            .ok_or_else(|| InspectError::InvalidTensor {
                tensor: name.clone(),
                reason: "shape element-count arithmetic overflow".to_owned(),
            })?;
        let bit_count = element_count.checked_mul(bits_per_element).ok_or_else(|| {
            InspectError::InvalidTensor {
                tensor: name.clone(),
                reason: "shape and dtype bit-size arithmetic overflow".to_owned(),
            }
        })?;
        if bit_count % 8 != 0 {
            return Err(InspectError::InvalidTensor {
                tensor: name,
                reason: format!("tensor size is not byte-aligned ({bit_count} bits)"),
            });
        }
        let expected_bytes = bit_count / 8;
        let [start, end] = raw_tensor.data_offsets;
        let offset_bytes = end
            .checked_sub(start)
            .ok_or_else(|| InspectError::InvalidTensor {
                tensor: name.clone(),
                reason: "data_offsets end precedes start".to_owned(),
            })?;
        if offset_bytes != expected_bytes {
            return Err(InspectError::InvalidTensor {
                tensor: name,
                reason: format!(
                    "offset span is {offset_bytes} bytes but shape and dtype require {expected_bytes}"
                ),
            });
        }
        if end > data_bytes {
            return Err(InspectError::InvalidTensor {
                tensor: name,
                reason: format!("data offset {end} exceeds payload size {data_bytes}"),
            });
        }
        tensors.push(TensorMetadata {
            name,
            dtype: raw_tensor.dtype,
            shape: raw_tensor.shape,
            data_offsets: [start, end],
            data_bytes: expected_bytes,
        });
    }

    tensors.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then(left.data_offsets.cmp(&right.data_offsets))
    });
    let mut layout: Vec<&TensorMetadata> = tensors.iter().collect();
    layout.sort_by(|left, right| {
        left.data_offsets
            .cmp(&right.data_offsets)
            .then(left.name.cmp(&right.name))
    });
    let mut cursor = 0_u64;
    for tensor in layout {
        let [start, end] = tensor.data_offsets;
        if start != cursor {
            return Err(InspectError::InvalidDataLayout(format!(
                "tensor {:?} starts at {start}, expected contiguous offset {cursor}",
                tensor.name
            )));
        }
        cursor = end;
    }
    if cursor != data_bytes {
        return Err(InspectError::InvalidDataLayout(format!(
            "tensor ranges cover {cursor} bytes but file contains {data_bytes} payload bytes"
        )));
    }

    Ok(SafetensorsMetadata {
        file_bytes,
        header_bytes,
        data_bytes,
        tensors,
        metadata: raw.metadata,
    })
}

#[derive(Debug)]
struct RawHeader {
    tensors: BTreeMap<String, RawTensor>,
    metadata: BTreeMap<String, String>,
}

impl<'de> Deserialize<'de> for RawHeader {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(RawHeaderVisitor)
    }
}

struct RawHeaderVisitor;

impl<'de> Visitor<'de> for RawHeaderVisitor {
    type Value = RawHeader;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON object containing tensor entries and optional __metadata__")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        let mut saw_metadata = false;
        while let Some(key) = map.next_key::<String>()? {
            if key == "__metadata__" {
                if saw_metadata {
                    return Err(de::Error::custom("duplicate __metadata__ key"));
                }
                saw_metadata = true;
                metadata = map.next_value::<UniqueStringMap>()?.0;
            } else {
                if tensors.contains_key(&key) {
                    return Err(de::Error::custom(format!("duplicate tensor name {key:?}")));
                }
                if tensors.len() >= MAX_TENSORS {
                    return Err(de::Error::custom(format!(
                        "tensor count exceeds configured limit {MAX_TENSORS}"
                    )));
                }
                let value = map.next_value::<RawTensor>()?;
                tensors.insert(key, value);
            }
        }
        Ok(RawHeader { tensors, metadata })
    }
}

#[derive(Debug)]
struct RawTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

impl<'de> Deserialize<'de> for RawTensor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            dtype: String,
            shape: Vec<u64>,
            data_offsets: [u64; 2],
        }
        let fields = Fields::deserialize(deserializer)?;
        Ok(Self {
            dtype: fields.dtype,
            shape: fields.shape,
            data_offsets: fields.data_offsets,
        })
    }
}

#[derive(Debug)]
struct UniqueStringMap(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for UniqueStringMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(UniqueStringMapVisitor)
    }
}

struct UniqueStringMapVisitor;

impl<'de> Visitor<'de> for UniqueStringMapVisitor {
    type Value = UniqueStringMap;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON object with unique string keys and string values")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = BTreeMap::new();
        while let Some((key, value)) = map.next_entry::<String, String>()? {
            if values.insert(key.clone(), value).is_some() {
                return Err(de::Error::custom(format!("duplicate metadata key {key:?}")));
            }
        }
        Ok(UniqueStringMap(values))
    }
}

fn dtype_bits(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "F4" => 4,
        "F6_E2M3" | "F6_E3M2" => 6,
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" | "F8_E8M0" | "F8_E4M3FNUZ"
        | "F8_E5M2FNUZ" => 8,
        "I16" | "U16" | "F16" | "BF16" => 16,
        "I32" | "U32" | "F32" => 32,
        "I64" | "U64" | "F64" | "C64" => 64,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::io::Cursor;

    fn file_with_header(header: &str, payload: &[u8]) -> Vec<u8> {
        let padded_len = header.len().div_ceil(8) * 8;
        let mut bytes = Vec::with_capacity(8 + padded_len + payload.len());
        bytes.extend_from_slice(&(padded_len as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(8 + padded_len, b' ');
        bytes.extend_from_slice(payload);
        bytes
    }

    fn inspect(header: &str, payload: &[u8]) -> Result<SafetensorsMetadata, InspectError> {
        inspect_safetensors(Cursor::new(file_with_header(header, payload)))
    }

    #[test]
    fn parses_scalar_zero_sized_and_metadata_entries() {
        let header = r#"{"scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"empty":{"dtype":"F16","shape":[0,3],"data_offsets":[4,4]},"__metadata__":{"format":"synthetic-test"}}"#;
        let result = inspect(header, &[0; 4]).expect("valid synthetic header");
        assert_eq!(result.data_bytes, 4);
        assert_eq!(result.tensors[0].name, "empty");
        assert_eq!(result.tensors[0].data_bytes, 0);
        assert_eq!(result.tensors[1].shape, Vec::<u64>::new());
        assert_eq!(result.metadata["format"], "synthetic-test");
    }

    #[test]
    fn sorts_tensor_output_deterministically() {
        let header = r#"{"z":{"dtype":"U8","shape":[1],"data_offsets":[1,2]},"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let first = inspect(header, &[8, 9]).unwrap();
        let second = inspect(header, &[8, 9]).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first
                .tensors
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>(),
            ["a", "z"]
        );
    }

    #[test]
    fn supports_sub_byte_dtype_when_total_tensor_is_byte_aligned() {
        let result = inspect(
            r#"{"packed":{"dtype":"F4","shape":[2],"data_offsets":[0,1]}}"#,
            &[0x12],
        )
        .unwrap();
        assert_eq!(result.tensors[0].data_bytes, 1);
    }

    #[test]
    fn rejects_unaligned_sub_byte_shape() {
        let error = inspect(
            r#"{"packed":{"dtype":"F4","shape":[1],"data_offsets":[0,1]}}"#,
            &[0],
        )
        .unwrap_err();
        assert!(error.to_string().contains("not byte-aligned"));
    }

    #[test]
    fn rejects_duplicate_tensor_and_metadata_keys() {
        let duplicate_tensor = r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        assert!(
            inspect(duplicate_tensor, &[0])
                .unwrap_err()
                .to_string()
                .contains("duplicate tensor")
        );
        let duplicate_metadata = r#"{"__metadata__":{"x":"a","x":"b"}}"#;
        assert!(
            inspect(duplicate_metadata, &[])
                .unwrap_err()
                .to_string()
                .contains("duplicate metadata")
        );
    }

    #[test]
    fn rejects_duplicate_fields_unknown_fields_and_unknown_dtypes() {
        let duplicate_field =
            r#"{"x":{"dtype":"U8","dtype":"I8","shape":[1],"data_offsets":[0,1]}}"#;
        assert!(inspect(duplicate_field, &[0]).is_err());
        let unknown_field = r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"extra":true}}"#;
        assert!(inspect(unknown_field, &[0]).is_err());
        let unknown_dtype = r#"{"x":{"dtype":"FUTURE","shape":[1],"data_offsets":[0,1]}}"#;
        assert!(
            inspect(unknown_dtype, &[0])
                .unwrap_err()
                .to_string()
                .contains("unsupported dtype")
        );
    }

    #[test]
    fn rejects_overflow_bad_offsets_holes_and_trailing_payload() {
        let overflow =
            r#"{"x":{"dtype":"F64","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#;
        assert!(
            inspect(overflow, &[])
                .unwrap_err()
                .to_string()
                .contains("overflow")
        );
        let reversed = r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[1,0]}}"#;
        assert!(inspect(reversed, &[0]).is_err());
        let hole = r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#;
        assert!(
            inspect(hole, &[0, 0])
                .unwrap_err()
                .to_string()
                .contains("contiguous")
        );
        let extra = r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        assert!(
            inspect(extra, &[0, 0])
                .unwrap_err()
                .to_string()
                .contains("cover 1 bytes")
        );
    }

    struct HeaderOnlyReader {
        inner: Cursor<Vec<u8>>,
        readable_until: u64,
    }

    impl Read for HeaderOnlyReader {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            let position = self.inner.position();
            if position >= self.readable_until {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "payload read attempted",
                ));
            }
            let available = usize::try_from(self.readable_until - position).unwrap_or(usize::MAX);
            let read_len = buffer.len().min(available);
            self.inner.read(&mut buffer[..read_len])
        }
    }

    impl Seek for HeaderOnlyReader {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.inner.seek(position)
        }
    }

    #[test]
    fn never_reads_payload_bytes() {
        let file = file_with_header(
            r#"{"x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#,
            &[1, 2, 3, 4],
        );
        let header_len = u64::from_le_bytes(file[..8].try_into().unwrap());
        let readable_until = 8 + header_len;
        let reader = HeaderOnlyReader {
            inner: Cursor::new(file),
            readable_until,
        };
        let parsed = inspect_safetensors(reader).expect("parser must stop after bounded header");
        assert_eq!(parsed.data_bytes, 4);
    }

    #[test]
    fn enforces_header_budget_before_reading_header() {
        let bytes = file_with_header("{}", &[]);
        let error = inspect_safetensors_with_header_limit(Cursor::new(bytes), 1).unwrap_err();
        assert!(matches!(error, InspectError::HeaderTooLarge { .. }));
        let error = inspect_safetensors_with_header_limit(
            Cursor::new(file_with_header("{}", &[])),
            MAX_HEADER_BYTES + 1,
        )
        .unwrap_err();
        assert!(matches!(error, InspectError::InvalidHeaderLimit { .. }));
    }

    #[test]
    fn rejects_short_prefix_and_truncated_header() {
        assert!(matches!(
            inspect_safetensors(Cursor::new(vec![0; 7])),
            Err(InspectError::FileTooSmall)
        ));
        let mut truncated = 4_u64.to_le_bytes().to_vec();
        truncated.extend_from_slice(b"{}");
        assert!(inspect_safetensors(Cursor::new(truncated)).is_err());
    }

    #[test]
    fn accepts_upstream_supported_dtype_widths() {
        let cases = [
            ("BOOL", 8),
            ("F4", 4),
            ("F6_E2M3", 6),
            ("F6_E3M2", 6),
            ("U8", 8),
            ("I8", 8),
            ("F8_E5M2", 8),
            ("F8_E4M3", 8),
            ("F8_E8M0", 8),
            ("F8_E4M3FNUZ", 8),
            ("F8_E5M2FNUZ", 8),
            ("I16", 16),
            ("U16", 16),
            ("F16", 16),
            ("BF16", 16),
            ("I32", 32),
            ("U32", 32),
            ("F32", 32),
            ("C64", 64),
            ("F64", 64),
            ("I64", 64),
            ("U64", 64),
        ];
        for (dtype, bits) in cases {
            assert_eq!(dtype_bits(dtype), Some(bits), "dtype {dtype}");
        }
    }

    #[test]
    fn rejects_tensor_rank_above_resource_limit() {
        let shape = vec![1; MAX_TENSOR_RANK + 1];
        let header =
            format!("{{\"x\":{{\"dtype\":\"U8\",\"shape\":{shape:?},\"data_offsets\":[0,1]}}}}");
        let error = inspect(&header, &[0]).unwrap_err();
        assert!(
            matches!(error, InspectError::RankTooLarge { rank, limit, .. }
            if rank == MAX_TENSOR_RANK + 1 && limit == MAX_TENSOR_RANK)
        );
    }

    proptest! {
        #[test]
        fn parses_valid_generated_u8_shapes(shape in prop::collection::vec(0_u64..8, 0..=5)) {
            let count = shape.iter().product::<u64>();
            let header = format!(
                "{{\"x\":{{\"dtype\":\"U8\",\"shape\":{shape:?},\"data_offsets\":[0,{count}]}}}}"
            );
            let payload = vec![0_u8; usize::try_from(count).unwrap()];
            let parsed = inspect(&header, &payload).expect("generated valid U8 tensor");
            prop_assert_eq!(&parsed.tensors[0].shape, &shape);
            prop_assert_eq!(parsed.data_bytes, count);
        }
    }
}
