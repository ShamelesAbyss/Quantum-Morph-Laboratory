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

/// Returns `(block_elements, block_bytes)` for GGUF tensor types supported by [`GgufFile`] decoding.
pub fn ggml_supported_block_layout(ggml_type_id: u32) -> Option<(u64, u64)> {
    match ggml_type_id {
        0 => Some((1, 4)),   // F32
        1 => Some((1, 2)),   // F16
        2 => Some((32, 18)), // Q4_0: 2B f16 scale + 16B nibbles
        3 => Some((32, 20)), // Q4_1: 2B f16 scale + 2B f16 min + 16B nibbles
        8 => Some((32, 34)), // Q8_0: 2B f16 scale + 32B i8 codes
        30 => Some((1, 2)),  // BF16
        _ => None,
    }
}

/// Bounded, read-only GGUF v2/v3 tensor payload reader supporting `F32`, `F16`, `BF16`,
/// `Q8_0`, `Q4_0`, and `Q4_1` tensors. Never modifies the source file.
pub struct GgufFile<R: Read + Seek> {
    reader: R,
    metadata: GgufMetadata,
    data_section_offset: u64,
    cumulative_decoded_bytes: u64,
}

impl<R: Read + Seek> GgufFile<R> {
    /// Open a GGUF v2/v3 stream read-only, parsing metadata and computing the aligned
    /// tensor data section offset without loading any tensor payload bytes.
    pub fn open(mut reader: R) -> Result<Self, GgufError> {
        let metadata = inspect_gguf(&mut reader)?;
        let header_end = reader.stream_position()?;
        let alignment = match metadata.metadata.get("general.alignment") {
            Some(GgufValueSummary::Uint(a)) if *a > 0 && a.is_power_of_two() => *a,
            _ => 32,
        };
        let rem = header_end % alignment;
        let data_section_offset = if rem == 0 {
            header_end
        } else {
            header_end
                .checked_add(alignment - rem)
                .ok_or_else(|| GgufError::InvalidLayout("GGUF alignment overflow".to_owned()))?
        };
        if data_section_offset > metadata.file_bytes {
            return Err(GgufError::InvalidLayout(
                "aligned GGUF tensor data section extends past EOF".to_owned(),
            ));
        }
        Ok(Self {
            reader,
            metadata,
            data_section_offset,
            cumulative_decoded_bytes: 0,
        })
    }

    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    pub fn data_section_offset(&self) -> u64 {
        self.data_section_offset
    }

    /// Decode a named GGUF tensor (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_0`, or `Q4_1`) into `Vec<f64>`
    /// enforcing an explicit [`crate::PayloadBudget`].
    pub fn read_tensor_f64(
        &mut self,
        name: &str,
        budget: &crate::PayloadBudget,
    ) -> Result<Vec<f64>, GgufError> {
        let info = self
            .metadata
            .tensors
            .iter()
            .find(|t| t.name == name)
            .cloned()
            .ok_or_else(|| GgufError::InvalidLayout(format!("GGUF tensor {name:?} not found")))?;
        let (block_elems, block_bytes) =
            ggml_supported_block_layout(info.ggml_type_id).ok_or_else(|| {
                GgufError::InvalidLayout(format!(
                    "GGUF tensor {name:?} uses unsupported ggml type {} (supported: F32, F16, BF16, Q8_0, Q4_0, Q4_1)",
                    info.ggml_type
                ))
            })?;

        let mut total_elems: u64 = 1;
        for &d in &info.dimensions {
            total_elems = total_elems.checked_mul(d).ok_or_else(|| {
                GgufError::InvalidLayout("GGUF tensor element overflow".to_owned())
            })?;
        }
        if !total_elems.is_multiple_of(block_elems) {
            return Err(GgufError::InvalidLayout(format!(
                "GGUF tensor {name:?} element count {total_elems} not divisible by block size {block_elems}"
            )));
        }
        let num_blocks = total_elems / block_elems;
        let raw_bytes = num_blocks
            .checked_mul(block_bytes)
            .ok_or_else(|| GgufError::InvalidLayout("GGUF raw byte size overflow".to_owned()))?;
        let decoded_bytes = total_elems
            .checked_mul(8)
            .ok_or_else(|| GgufError::InvalidLayout("GGUF decoded size overflow".to_owned()))?;

        if raw_bytes > budget.max_tensor_bytes || decoded_bytes > budget.max_tensor_bytes {
            return Err(GgufError::LimitExceeded(format!(
                "GGUF tensor {name:?} exceeds per-tensor budget ({} bytes)",
                budget.max_tensor_bytes
            )));
        }
        let next_cum = self
            .cumulative_decoded_bytes
            .checked_add(decoded_bytes)
            .ok_or_else(|| GgufError::LimitExceeded("cumulative budget overflow".to_owned()))?;
        if next_cum > budget.max_total_bytes {
            return Err(GgufError::LimitExceeded(format!(
                "GGUF cumulative decoded bytes {next_cum} exceeds budget {}",
                budget.max_total_bytes
            )));
        }

        let start = self
            .data_section_offset
            .checked_add(info.offset)
            .ok_or_else(|| GgufError::InvalidLayout("GGUF tensor offset overflow".to_owned()))?;
        let end = start
            .checked_add(raw_bytes)
            .ok_or_else(|| GgufError::InvalidLayout("GGUF tensor end overflow".to_owned()))?;
        if end > self.metadata.file_bytes {
            return Err(GgufError::InvalidLayout(format!(
                "GGUF tensor {name:?} range {start}..{end} exceeds file size {}",
                self.metadata.file_bytes
            )));
        }

        self.reader.seek(SeekFrom::Start(start))?;
        let mut raw = vec![0_u8; raw_bytes as usize];
        self.reader.read_exact(&mut raw)?;

        let mut out = Vec::with_capacity(total_elems as usize);
        match info.ggml_type_id {
            0 => {
                // F32
                for chunk in raw.as_chunks::<4>().0 {
                    out.push(f64::from(f32::from_le_bytes(*chunk)));
                }
            }
            1 => {
                // F16
                for chunk in raw.as_chunks::<2>().0 {
                    let bits = u16::from_le_bytes(*chunk);
                    out.push(f64::from(crate::dtype::f16_to_f32(bits)));
                }
            }
            30 => {
                // BF16
                for chunk in raw.as_chunks::<2>().0 {
                    let bits = u16::from_le_bytes(*chunk);
                    out.push(f64::from(crate::dtype::bf16_to_f32(bits)));
                }
            }
            8 => {
                // Q8_0: 2B f16 scale + 32B i8
                for blk in raw.as_chunks::<34>().0 {
                    let d = f64::from(crate::dtype::f16_to_f32(u16::from_le_bytes([
                        blk[0], blk[1],
                    ])));
                    for &b in &blk[2..34] {
                        out.push(f64::from(b as i8) * d);
                    }
                }
            }
            2 => {
                // Q4_0: 2B f16 scale + 16B nibbles (low nibbles = 0..16, high nibbles = 16..32)
                for blk in raw.as_chunks::<18>().0 {
                    let d = f64::from(crate::dtype::f16_to_f32(u16::from_le_bytes([
                        blk[0], blk[1],
                    ])));
                    let qs = &blk[2..18];
                    for &byte in qs {
                        let lo = i32::from(byte & 0x0f) - 8;
                        out.push(f64::from(lo) * d);
                    }
                    for &byte in qs {
                        let hi = i32::from((byte >> 4) & 0x0f) - 8;
                        out.push(f64::from(hi) * d);
                    }
                }
            }
            3 => {
                // Q4_1: 2B f16 scale + 2B f16 min + 16B nibbles (low = 0..16, high = 16..32)
                for blk in raw.as_chunks::<20>().0 {
                    let d = f64::from(crate::dtype::f16_to_f32(u16::from_le_bytes([
                        blk[0], blk[1],
                    ])));
                    let m = f64::from(crate::dtype::f16_to_f32(u16::from_le_bytes([
                        blk[2], blk[3],
                    ])));
                    let qs = &blk[4..20];
                    for &byte in qs {
                        let lo = f64::from(byte & 0x0f);
                        out.push(lo * d + m);
                    }
                    for &byte in qs {
                        let hi = f64::from((byte >> 4) & 0x0f);
                        out.push(hi * d + m);
                    }
                }
            }
            _ => unreachable!("filtered by ggml_supported_block_layout"),
        }

        self.cumulative_decoded_bytes = next_cum;
        Ok(out)
    }

    /// Decode a 2D GGUF tensor into a [`qml_core::DenseMatrix`].
    /// Note: GGUF stores dimensions in GGML order `[ne0 (cols), ne1 (rows)]`.
    pub fn read_tensor_matrix(
        &mut self,
        name: &str,
        budget: &crate::PayloadBudget,
    ) -> Result<qml_core::DenseMatrix, GgufError> {
        let info = self
            .metadata
            .tensors
            .iter()
            .find(|t| t.name == name)
            .cloned()
            .ok_or_else(|| GgufError::InvalidLayout(format!("GGUF tensor {name:?} not found")))?;
        if info.dimensions.len() != 2 {
            return Err(GgufError::InvalidLayout(format!(
                "GGUF tensor {name:?} has {} dimensions, expected 2",
                info.dimensions.len()
            )));
        }
        // GGML dimension order: ne[0] = inner contiguous columns, ne[1] = outer rows.
        let cols = info.dimensions[0] as usize;
        let rows = info.dimensions[1] as usize;
        let vals = self.read_tensor_f64(name, budget)?;
        qml_core::DenseMatrix::from_flat(rows, cols, vals)
            .map_err(|e| GgufError::InvalidLayout(e.to_string()))
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

    #[test]
    fn decodes_f32_f16_bf16_q8_0_q4_0_and_q4_1_gguf_payloads_exactly() {
        // Build a valid 32-byte-aligned GGUF v3 file containing 4 tensors (F32, Q8_0, Q4_0, Q4_1).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&3_u32.to_le_bytes()); // v3
        buf.extend_from_slice(&4_u64.to_le_bytes()); // 4 tensors
        buf.extend_from_slice(&0_u64.to_le_bytes()); // 0 KV pairs (default alignment = 32)

        // Tensor 0: t_f32 [4] F32 at offset 0 (16 bytes)
        write_str(&mut buf, "t_f32");
        buf.extend_from_slice(&1_u32.to_le_bytes());
        buf.extend_from_slice(&4_u64.to_le_bytes());
        buf.extend_from_slice(&0_u32.to_le_bytes()); // F32
        buf.extend_from_slice(&0_u64.to_le_bytes());

        // Tensor 1: t_q8 [32] Q8_0 at offset 32 (34 bytes)
        write_str(&mut buf, "t_q8");
        buf.extend_from_slice(&1_u32.to_le_bytes());
        buf.extend_from_slice(&32_u64.to_le_bytes());
        buf.extend_from_slice(&8_u32.to_le_bytes()); // Q8_0
        buf.extend_from_slice(&32_u64.to_le_bytes());

        // Tensor 2: t_q4_0 [32] Q4_0 at offset 96 (18 bytes)
        write_str(&mut buf, "t_q4_0");
        buf.extend_from_slice(&1_u32.to_le_bytes());
        buf.extend_from_slice(&32_u64.to_le_bytes());
        buf.extend_from_slice(&2_u32.to_le_bytes()); // Q4_0
        buf.extend_from_slice(&96_u64.to_le_bytes());

        // Tensor 3: t_q4_1 [32] Q4_1 at offset 128 (20 bytes)
        write_str(&mut buf, "t_q4_1");
        buf.extend_from_slice(&1_u32.to_le_bytes());
        buf.extend_from_slice(&32_u64.to_le_bytes());
        buf.extend_from_slice(&3_u32.to_le_bytes()); // Q4_1
        buf.extend_from_slice(&128_u64.to_le_bytes());

        // Pad header to 32-byte alignment
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        let data_start = buf.len();
        buf.resize(data_start + 160, 0);

        // Fill F32: [1.5, -2.25, 0.0, 4.0]
        for (i, &v) in [1.5_f32, -2.25, 0.0, 4.0].iter().enumerate() {
            buf[data_start + i * 4..data_start + (i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        // Fill Q8_0: scale = 0.5 (f16), codes = 0..32
        let scale_half = crate::dtype::f32_to_f16(0.5_f32).to_le_bytes();
        buf[data_start + 32..data_start + 34].copy_from_slice(&scale_half);
        for i in 0..32_u8 {
            buf[data_start + 34 + i as usize] = i;
        }
        // Fill Q4_0: scale = 0.25 (f16), nibbles: lo=10 (+2), hi=6 (-2)
        let scale_q = crate::dtype::f32_to_f16(0.25_f32).to_le_bytes();
        buf[data_start + 96..data_start + 98].copy_from_slice(&scale_q);
        for i in 0..16 {
            buf[data_start + 98 + i] = (6 << 4) | 10;
        }
        // Fill Q4_1: scale = 0.5 (f16), min = -1.0 (f16), nibbles: lo=2 (-> 0.0), hi=6 (-> 2.0)
        let min_neg1 = crate::dtype::f32_to_f16(-1.0_f32).to_le_bytes();
        buf[data_start + 128..data_start + 130].copy_from_slice(&scale_half);
        buf[data_start + 130..data_start + 132].copy_from_slice(&min_neg1);
        for i in 0..16 {
            buf[data_start + 132 + i] = (6 << 4) | 2;
        }

        let budget = crate::PayloadBudget::default();
        let mut gf = GgufFile::open(Cursor::new(buf)).unwrap();
        let v_f32 = gf.read_tensor_f64("t_f32", &budget).unwrap();
        assert_eq!(v_f32, vec![1.5, -2.25, 0.0, 4.0]);

        let v_q8 = gf.read_tensor_f64("t_q8", &budget).unwrap();
        assert_eq!(v_q8.len(), 32);
        assert!((v_q8[0] - 0.0).abs() < 1e-6);
        assert!((v_q8[10] - 5.0).abs() < 1e-6);

        let v_q4_0 = gf.read_tensor_f64("t_q4_0", &budget).unwrap();
        assert_eq!(v_q4_0.len(), 32);
        assert!((v_q4_0[0] - 0.5).abs() < 1e-6); // (10 - 8) * 0.25
        assert!((v_q4_0[16] - (-0.5)).abs() < 1e-6); // (6 - 8) * 0.25

        let v_q4_1 = gf.read_tensor_f64("t_q4_1", &budget).unwrap();
        assert_eq!(v_q4_1.len(), 32);
        assert!((v_q4_1[0] - 0.0).abs() < 1e-6); // 2 * 0.5 - 1.0
        assert!((v_q4_1[16] - 2.0).abs() < 1e-6); // 6 * 0.5 - 1.0
    }
}
