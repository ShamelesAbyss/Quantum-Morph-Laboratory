//! Bounded, read-only GGUF v2/v3 metadata and tensor-directory inspector.
//!
//! Reads the GGUF header, typed KV metadata, and tensor descriptors without
//! loading tensor payload data. Enforces strict bounds on counts, string
//! lengths, and array lengths to protect against malformed/hostile inputs.

use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Read, Seek, SeekFrom};

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
pub const MAX_GGUF_TENSORS: u64 = 100_000;
pub const MAX_GGUF_KV_PAIRS: u64 = 65_536;
pub const MAX_GGUF_STRING_BYTES: u64 = 1024 * 1024;
pub const MAX_GGUF_ARRAY_ELEMENTS: u64 = 1_048_576;
pub const MAX_GGUF_DIMS: u32 = 8;

/// Validated metadata summary of a GGUF file (payloads not read).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GgufMetadata {
    pub file_bytes: u64,
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub metadata: BTreeMap<String, GgufValueSummary>,
    pub tensors: Vec<GgufTensorInfo>,
}

/// Compact serializable summary of a GGUF metadata value.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum GgufValueSummary {
    Uint(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    ArraySummary { element_type: String, len: u64 },
}

/// Validated GGUF tensor descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type_id: u32,
    pub ggml_type: String,
    pub offset: u64,
}

#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    InvalidMagic,
    UnsupportedVersion(u32),
    LimitExceeded(String),
    InvalidValueType(u32),
    InvalidUtf8,
    DuplicateKey(String),
    InvalidLayout(String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "GGUF I/O error: {e}"),
            Self::InvalidMagic => write!(f, "invalid GGUF magic bytes (expected 'GGUF')"),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported GGUF version {v} (supported: 2, 3)")
            }
            Self::LimitExceeded(msg) => write!(f, "GGUF resource limit exceeded: {msg}"),
            Self::InvalidValueType(t) => write!(f, "unknown GGUF metadata value type {t}"),
            Self::InvalidUtf8 => write!(f, "invalid UTF-8 string in GGUF header"),
            Self::DuplicateKey(k) => write!(f, "duplicate GGUF key {k:?}"),
            Self::InvalidLayout(msg) => write!(f, "invalid GGUF layout: {msg}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<io::Error> for GgufError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

/// Inspect a seekable GGUF v2/v3 file's metadata and tensor directory without reading payloads.
pub fn inspect_gguf<R: Read + Seek>(mut reader: R) -> Result<GgufMetadata, GgufError> {
    let file_bytes = reader.seek(SeekFrom::End(0))?;
    if file_bytes < 24 {
        return Err(GgufError::InvalidLayout(
            "file shorter than 24-byte GGUF header".to_owned(),
        ));
    }
    reader.seek(SeekFrom::Start(0))?;

    let mut magic = [0_u8; 4];
    reader.read_exact(&mut magic)?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic);
    }

    let version = read_u32(&mut reader)?;
    if version != 2 && version != 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = read_u64(&mut reader)?;
    let kv_count = read_u64(&mut reader)?;
    if tensor_count > MAX_GGUF_TENSORS {
        return Err(GgufError::LimitExceeded(format!(
            "tensor_count {tensor_count} exceeds limit {MAX_GGUF_TENSORS}"
        )));
    }
    if kv_count > MAX_GGUF_KV_PAIRS {
        return Err(GgufError::LimitExceeded(format!(
            "kv_count {kv_count} exceeds limit {MAX_GGUF_KV_PAIRS}"
        )));
    }

    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut reader)?;
        let value_type = read_u32(&mut reader)?;
        let value = read_gguf_value(&mut reader, value_type, 0)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(GgufError::DuplicateKey(key));
        }
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    let mut seen_names = std::collections::BTreeSet::new();
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut reader)?;
        if !seen_names.insert(name.clone()) {
            return Err(GgufError::DuplicateKey(name));
        }
        let n_dims = read_u32(&mut reader)?;
        if n_dims > MAX_GGUF_DIMS {
            return Err(GgufError::LimitExceeded(format!(
                "tensor {name:?} has {n_dims} dimensions (limit {MAX_GGUF_DIMS})"
            )));
        }
        let mut dimensions = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dimensions.push(read_u64(&mut reader)?);
        }
        let ggml_type_id = read_u32(&mut reader)?;
        let offset = read_u64(&mut reader)?;
        tensors.push(GgufTensorInfo {
            name,
            dimensions,
            ggml_type_id,
            ggml_type: ggml_type_name(ggml_type_id).to_owned(),
            offset,
        });
    }

    // Verify header ended within file bounds and offsets do not exceed file size.
    let header_end = reader.stream_position()?;
    if header_end > file_bytes {
        return Err(GgufError::InvalidLayout(
            "GGUF header extends past end of file".to_owned(),
        ));
    }
    let data_bytes = file_bytes - header_end;
    for tensor in &tensors {
        if tensor.offset > data_bytes {
            return Err(GgufError::InvalidLayout(format!(
                "tensor {:?} offset {} exceeds remaining file payload {} bytes",
                tensor.name, tensor.offset, data_bytes
            )));
        }
    }

    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(GgufMetadata {
        file_bytes,
        version,
        tensor_count,
        kv_count,
        metadata,
        tensors,
    })
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8, GgufError> {
    let mut b = [0_u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16, GgufError> {
    let mut b = [0_u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, GgufError> {
    let mut b = [0_u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, GgufError> {
    let mut b = [0_u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_gguf_string<R: Read>(r: &mut R) -> Result<String, GgufError> {
    let len = read_u64(r)?;
    if len > MAX_GGUF_STRING_BYTES {
        return Err(GgufError::LimitExceeded(format!(
            "string length {len} exceeds limit {MAX_GGUF_STRING_BYTES}"
        )));
    }
    let len_usize = usize::try_from(len)
        .map_err(|_| GgufError::LimitExceeded("string length overflow".into()))?;
    let mut buf = vec![0_u8; len_usize];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| GgufError::InvalidUtf8)
}

fn skip_gguf_value<R: Read + Seek>(
    r: &mut R,
    value_type: u32,
    depth: u32,
) -> Result<(), GgufError> {
    let _ = read_gguf_value(r, value_type, depth)?;
    Ok(())
}

fn read_gguf_value<R: Read + Seek>(
    r: &mut R,
    value_type: u32,
    depth: u32,
) -> Result<GgufValueSummary, GgufError> {
    if depth > 4 {
        return Err(GgufError::LimitExceeded(
            "nested GGUF array depth exceeds limit".to_owned(),
        ));
    }
    Ok(match value_type {
        0 => GgufValueSummary::Uint(u64::from(read_u8(r)?)), // UINT8
        1 => GgufValueSummary::Int(i64::from(read_u8(r)? as i8)), // INT8
        2 => GgufValueSummary::Uint(u64::from(read_u16(r)?)), // UINT16
        3 => GgufValueSummary::Int(i64::from(read_u16(r)? as i16)), // INT16
        4 => GgufValueSummary::Uint(u64::from(read_u32(r)?)), // UINT32
        5 => GgufValueSummary::Int(i64::from(read_u32(r)? as i32)), // INT32
        6 => GgufValueSummary::Float(f64::from(f32::from_bits(read_u32(r)?))), // FLOAT32
        7 => GgufValueSummary::Bool(read_u8(r)? != 0),       // BOOL
        8 => GgufValueSummary::String(read_gguf_string(r)?), // STRING
        9 => {
            // ARRAY
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            if count > MAX_GGUF_ARRAY_ELEMENTS {
                return Err(GgufError::LimitExceeded(format!(
                    "array length {count} exceeds limit {MAX_GGUF_ARRAY_ELEMENTS}"
                )));
            }
            if let Some(fixed_bytes) = fixed_scalar_bytes(elem_type) {
                let skip = count.checked_mul(fixed_bytes).ok_or_else(|| {
                    GgufError::LimitExceeded("array byte-skip overflow".to_owned())
                })?;
                r.seek(SeekFrom::Current(i64::try_from(skip).map_err(|_| {
                    GgufError::LimitExceeded("array seek overflow".to_owned())
                })?))?;
            } else {
                for _ in 0..count {
                    skip_gguf_value(r, elem_type, depth + 1)?;
                }
            }
            GgufValueSummary::ArraySummary {
                element_type: gguf_value_type_name(elem_type).to_owned(),
                len: count,
            }
        }
        10 => GgufValueSummary::Uint(read_u64(r)?), // UINT64
        11 => GgufValueSummary::Int(read_u64(r)? as i64), // INT64
        12 => GgufValueSummary::Float(f64::from_bits(read_u64(r)?)), // FLOAT64
        other => return Err(GgufError::InvalidValueType(other)),
    })
}

fn fixed_scalar_bytes(value_type: u32) -> Option<u64> {
    Some(match value_type {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => return None,
    })
}

fn gguf_value_type_name(value_type: u32) -> &'static str {
    match value_type {
        0 => "UINT8",
        1 => "INT8",
        2 => "UINT16",
        3 => "INT16",
        4 => "UINT32",
        5 => "INT32",
        6 => "FLOAT32",
        7 => "BOOL",
        8 => "STRING",
        9 => "ARRAY",
        10 => "UINT64",
        11 => "INT64",
        12 => "FLOAT64",
        _ => "UNKNOWN",
    }
}

/// Canonical GGML tensor type names from the GGUF/GGML specification.
pub fn ggml_type_name(id: u32) -> &'static str {
    match id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        _ => "UNKNOWN_GGML_TYPE",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn write_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn synthetic_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&3_u32.to_le_bytes()); // version 3
        buf.extend_from_slice(&1_u64.to_le_bytes()); // 1 tensor
        buf.extend_from_slice(&2_u64.to_le_bytes()); // 2 KV pairs
        // KV 1: general.architecture = "llama"
        write_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&8_u32.to_le_bytes()); // STRING
        write_str(&mut buf, "llama");
        // KV 2: llama.block_count = 4
        write_str(&mut buf, "llama.block_count");
        buf.extend_from_slice(&4_u32.to_le_bytes()); // UINT32
        buf.extend_from_slice(&4_u32.to_le_bytes());
        // Tensor 1: blk.0.attn_q.weight [64, 64] Q4_0 offset 0
        write_str(&mut buf, "blk.0.attn_q.weight");
        buf.extend_from_slice(&2_u32.to_le_bytes());
        buf.extend_from_slice(&64_u64.to_le_bytes());
        buf.extend_from_slice(&64_u64.to_le_bytes());
        buf.extend_from_slice(&2_u32.to_le_bytes()); // Q4_0
        buf.extend_from_slice(&0_u64.to_le_bytes()); // offset 0
        // Dummy payload bytes
        buf.extend_from_slice(&[0_u8; 64]);
        buf
    }

    #[test]
    fn inspects_valid_synthetic_gguf_v3() {
        let bytes = synthetic_gguf();
        let meta = inspect_gguf(Cursor::new(bytes)).unwrap();
        assert_eq!(meta.version, 3);
        assert_eq!(meta.tensor_count, 1);
        assert_eq!(meta.kv_count, 2);
        assert_eq!(
            meta.metadata["general.architecture"],
            GgufValueSummary::String("llama".to_owned())
        );
        assert_eq!(meta.tensors[0].name, "blk.0.attn_q.weight");
        assert_eq!(meta.tensors[0].ggml_type, "Q4_0");
        assert_eq!(meta.tensors[0].dimensions, vec![64, 64]);
    }

    #[test]
    fn rejects_invalid_magic_and_version() {
        let mut bad = synthetic_gguf();
        bad[0] = b'X';
        assert!(matches!(
            inspect_gguf(Cursor::new(bad)),
            Err(GgufError::InvalidMagic)
        ));

        let mut bad_ver = synthetic_gguf();
        bad_ver[4..8].copy_from_slice(&99_u32.to_le_bytes());
        assert!(matches!(
            inspect_gguf(Cursor::new(bad_ver)),
            Err(GgufError::UnsupportedVersion(99))
        ));
    }
}
