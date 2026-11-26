//! Bounded, read-only Safetensors tensor payload reading and shard-index parsing.
//!
//! Preserves read-only access to imported model files and enforces explicit
//! memory budgets before allocating for any tensor payload. Supports streaming
//! chunked access so arbitrarily large tensors can be analyzed in bounded RAM.

use crate::dtype::{DType, decode_chunk_to_f64};
use crate::{
    InspectError, MAX_HEADER_BYTES, SafetensorsMetadata, TensorMetadata,
    inspect_safetensors_with_header_limit,
};
use qml_core::{DenseMatrix, MatrixError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Read, Seek, SeekFrom};

/// Hard safety ceiling on a single shard index JSON file (16 MiB).
pub const MAX_SHARD_INDEX_BYTES: u64 = 16 * 1024 * 1024;
/// Default per-tensor allocation ceiling (8 GiB in decoded f64 — covers 7B–32B 152K-256K vocab embeddings).
pub const DEFAULT_MAX_TENSOR_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// Default cumulative payload budget for multi-tensor reads (64 GiB — covers 32B BF16 checkpoints).
pub const DEFAULT_MAX_TOTAL_BYTES: u64 = 64 * 1024 * 1024 * 1024;
/// Default element count per streaming chunk (65,536 elements = 512 KiB in f64).
pub const DEFAULT_STREAM_CHUNK_ELEMENTS: usize = 65_536;

/// Explicit memory and header bounds for Safetensors payload access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadBudget {
    pub max_header_bytes: u64,
    pub max_tensor_bytes: u64,
    pub max_total_bytes: u64,
    pub stream_chunk_elements: usize,
}

impl Default for PayloadBudget {
    fn default() -> Self {
        Self {
            max_header_bytes: MAX_HEADER_BYTES,
            max_tensor_bytes: DEFAULT_MAX_TENSOR_BYTES,
            max_total_bytes: DEFAULT_MAX_TOTAL_BYTES,
            stream_chunk_elements: DEFAULT_STREAM_CHUNK_ELEMENTS,
        }
    }
}

/// Errors produced during bounded tensor payload access.
#[derive(Debug)]
pub enum PayloadError {
    Inspect(InspectError),
    Io(io::Error),
    TensorNotFound(String),
    TensorExceedsBudget {
        tensor: String,
        required_bytes: u64,
        limit_bytes: u64,
    },
    TotalBudgetExceeded {
        cumulative_bytes: u64,
        limit_bytes: u64,
    },
    OffsetOverflow {
        tensor: String,
    },
    AllocationFailed {
        tensor: String,
        bytes: usize,
    },
    UnsupportedDecode {
        tensor: String,
        reason: String,
    },
    NotTwoDimensional {
        tensor: String,
        shape: Vec<u64>,
    },
    Matrix(MatrixError),
    InvalidChunkSize,
    InvalidShardIndex(String),
}

impl fmt::Display for PayloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inspect(e) => write!(f, "Safetensors metadata error: {e}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::TensorNotFound(name) => write!(f, "tensor {name:?} not found"),
            Self::TensorExceedsBudget {
                tensor,
                required_bytes,
                limit_bytes,
            } => write!(
                f,
                "tensor {tensor:?} requires {required_bytes} bytes, exceeding per-tensor budget {limit_bytes} bytes"
            ),
            Self::TotalBudgetExceeded {
                cumulative_bytes,
                limit_bytes,
            } => write!(
                f,
                "cumulative read of {cumulative_bytes} bytes exceeds configured total budget {limit_bytes} bytes"
            ),
            Self::OffsetOverflow { tensor } => {
                write!(f, "file offset overflow for tensor {tensor:?}")
            }
            Self::AllocationFailed { tensor, bytes } => {
                write!(f, "failed to allocate {bytes} bytes for tensor {tensor:?}")
            }
            Self::UnsupportedDecode { tensor, reason } => {
                write!(f, "cannot decode tensor {tensor:?}: {reason}")
            }
            Self::NotTwoDimensional { tensor, shape } => {
                write!(
                    f,
                    "tensor {tensor:?} has shape {shape:?}, expected 2D matrix"
                )
            }
            Self::Matrix(e) => write!(f, "matrix construction error: {e}"),
            Self::InvalidChunkSize => write!(f, "streaming chunk size must be non-zero"),
            Self::InvalidShardIndex(msg) => write!(f, "invalid sharded Safetensors index: {msg}"),
        }
    }
}

impl std::error::Error for PayloadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Inspect(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::Matrix(e) => Some(e),
            _ => None,
        }
    }
}

impl From<InspectError> for PayloadError {
    fn from(value: InspectError) -> Self {
        Self::Inspect(value)
    }
}

impl From<io::Error> for PayloadError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<MatrixError> for PayloadError {
    fn from(value: MatrixError) -> Self {
        Self::Matrix(value)
    }
}

/// Bounded, read-only handle over a validated Safetensors stream.
pub struct SafetensorsFile<R: Read + Seek> {
    reader: R,
    metadata: SafetensorsMetadata,
    data_start: u64,
    bytes_read_so_far: u64,
}

impl<R: Read + Seek> SafetensorsFile<R> {
    /// Validate header metadata and construct a bounded payload reader without reading payloads.
    pub fn open(mut reader: R, budget: &PayloadBudget) -> Result<Self, PayloadError> {
        let metadata = inspect_safetensors_with_header_limit(&mut reader, budget.max_header_bytes)?;
        let data_start = 8_u64.checked_add(metadata.header_bytes).ok_or_else(|| {
            PayloadError::OffsetOverflow {
                tensor: "<header>".to_owned(),
            }
        })?;
        Ok(Self {
            reader,
            metadata,
            data_start,
            bytes_read_so_far: 0,
        })
    }

    pub fn metadata(&self) -> &SafetensorsMetadata {
        &self.metadata
    }

    pub fn bytes_read_so_far(&self) -> u64 {
        self.bytes_read_so_far
    }

    pub fn tensor_metadata(&self, name: &str) -> Option<&TensorMetadata> {
        self.metadata
            .tensors
            .binary_search_by(|t| t.name.as_str().cmp(name))
            .ok()
            .map(|idx| &self.metadata.tensors[idx])
    }

    /// Read raw little-endian payload bytes for `name`, enforcing both per-tensor and total budgets.
    pub fn read_tensor_bytes(
        &mut self,
        name: &str,
        budget: &PayloadBudget,
    ) -> Result<Vec<u8>, PayloadError> {
        let info = self
            .tensor_metadata(name)
            .ok_or_else(|| PayloadError::TensorNotFound(name.to_owned()))?
            .clone();

        if info.data_bytes > budget.max_tensor_bytes {
            return Err(PayloadError::TensorExceedsBudget {
                tensor: info.name,
                required_bytes: info.data_bytes,
                limit_bytes: budget.max_tensor_bytes,
            });
        }
        let next_total = self.bytes_read_so_far.checked_add(info.data_bytes).ok_or(
            PayloadError::TotalBudgetExceeded {
                cumulative_bytes: u64::MAX,
                limit_bytes: budget.max_total_bytes,
            },
        )?;
        if next_total > budget.max_total_bytes {
            return Err(PayloadError::TotalBudgetExceeded {
                cumulative_bytes: next_total,
                limit_bytes: budget.max_total_bytes,
            });
        }

        let file_offset = self
            .data_start
            .checked_add(info.data_offsets[0])
            .ok_or_else(|| PayloadError::OffsetOverflow {
                tensor: info.name.clone(),
            })?;
        let len = usize::try_from(info.data_bytes).map_err(|_| PayloadError::OffsetOverflow {
            tensor: info.name.clone(),
        })?;

        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(len)
            .map_err(|_| PayloadError::AllocationFailed {
                tensor: info.name.clone(),
                bytes: len,
            })?;
        buffer.resize(len, 0);

        self.reader.seek(SeekFrom::Start(file_offset))?;
        self.reader.read_exact(&mut buffer)?;
        self.bytes_read_so_far = next_total;
        Ok(buffer)
    }

    /// Read and decode a full tensor to `Vec<f64>`, checking decoded `f64` memory against `budget`.
    pub fn read_tensor_f64(
        &mut self,
        name: &str,
        budget: &PayloadBudget,
    ) -> Result<Vec<f64>, PayloadError> {
        let info = self
            .tensor_metadata(name)
            .ok_or_else(|| PayloadError::TensorNotFound(name.to_owned()))?
            .clone();
        let dtype = DType::parse(&info.dtype).ok_or_else(|| PayloadError::UnsupportedDecode {
            tensor: info.name.clone(),
            reason: format!("unknown dtype {:?}", info.dtype),
        })?;
        if !dtype.supports_f64_decode() {
            return Err(PayloadError::UnsupportedDecode {
                tensor: info.name,
                reason: format!("dtype {dtype} does not support scalar f64 decoding"),
            });
        }
        let element_count = tensor_element_count(&info)?;
        let decoded_bytes = element_count
            .checked_mul(std::mem::size_of::<f64>() as u64)
            .ok_or_else(|| PayloadError::OffsetOverflow {
                tensor: info.name.clone(),
            })?;
        if decoded_bytes > budget.max_tensor_bytes {
            return Err(PayloadError::TensorExceedsBudget {
                tensor: info.name,
                required_bytes: decoded_bytes,
                limit_bytes: budget.max_tensor_bytes,
            });
        }

        let elem_count_usize =
            usize::try_from(element_count).map_err(|_| PayloadError::OffsetOverflow {
                tensor: info.name.clone(),
            })?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(elem_count_usize)
            .map_err(|_| PayloadError::AllocationFailed {
                tensor: info.name.clone(),
                bytes: elem_count_usize.saturating_mul(std::mem::size_of::<f64>()),
            })?;

        let raw = self.read_tensor_bytes(name, budget)?;
        decode_chunk_to_f64(dtype, &raw, &mut values).map_err(|reason| {
            PayloadError::UnsupportedDecode {
                tensor: info.name,
                reason,
            }
        })?;
        Ok(values)
    }

    /// Read a 2D tensor as a validated finite `DenseMatrix`.
    pub fn read_tensor_matrix(
        &mut self,
        name: &str,
        budget: &PayloadBudget,
    ) -> Result<DenseMatrix, PayloadError> {
        let info = self
            .tensor_metadata(name)
            .ok_or_else(|| PayloadError::TensorNotFound(name.to_owned()))?
            .clone();
        if info.shape.len() != 2 {
            return Err(PayloadError::NotTwoDimensional {
                tensor: info.name,
                shape: info.shape,
            });
        }
        let rows = usize::try_from(info.shape[0]).map_err(|_| PayloadError::OffsetOverflow {
            tensor: info.name.clone(),
        })?;
        let cols = usize::try_from(info.shape[1]).map_err(|_| PayloadError::OffsetOverflow {
            tensor: info.name.clone(),
        })?;
        let values = self.read_tensor_f64(name, budget)?;
        Ok(DenseMatrix::from_flat(rows, cols, values)?)
    }

    /// Read specific rows of a 2D tensor directly by seeking to each row's byte offset,
    /// avoiding loading multi-GB embedding tables into RAM.
    pub fn read_tensor_rows_f64(
        &mut self,
        name: &str,
        row_indices: &[usize],
    ) -> Result<BTreeMap<usize, Vec<f64>>, PayloadError> {
        let info = self
            .tensor_metadata(name)
            .ok_or_else(|| PayloadError::TensorNotFound(name.to_owned()))?
            .clone();
        if info.shape.len() != 2 {
            return Err(PayloadError::NotTwoDimensional {
                tensor: info.name,
                shape: info.shape,
            });
        }
        let rows = usize::try_from(info.shape[0]).map_err(|_| PayloadError::OffsetOverflow {
            tensor: info.name.clone(),
        })?;
        let cols = usize::try_from(info.shape[1]).map_err(|_| PayloadError::OffsetOverflow {
            tensor: info.name.clone(),
        })?;
        let dtype = DType::parse(&info.dtype).ok_or_else(|| PayloadError::UnsupportedDecode {
            tensor: info.name.clone(),
            reason: format!("unknown dtype {:?}", info.dtype),
        })?;
        let elem_bytes = dtype
            .byte_width()
            .ok_or_else(|| PayloadError::UnsupportedDecode {
                tensor: info.name.clone(),
                reason: format!("dtype {dtype} is sub-byte"),
            })?;
        let base_offset = self
            .data_start
            .checked_add(info.data_offsets[0])
            .ok_or_else(|| PayloadError::OffsetOverflow {
                tensor: info.name.clone(),
            })?;
        let row_bytes =
            cols.checked_mul(elem_bytes)
                .ok_or_else(|| PayloadError::OffsetOverflow {
                    tensor: info.name.clone(),
                })?;
        let mut raw_row = vec![0u8; row_bytes];
        let mut out = BTreeMap::new();
        for &r in row_indices {
            if r >= rows || out.contains_key(&r) {
                continue;
            }
            let off = base_offset
                .checked_add((r as u64).saturating_mul(row_bytes as u64))
                .ok_or_else(|| PayloadError::OffsetOverflow {
                    tensor: info.name.clone(),
                })?;
            self.reader.seek(SeekFrom::Start(off))?;
            self.reader.read_exact(&mut raw_row)?;
            let mut decoded = Vec::with_capacity(cols);
            decode_chunk_to_f64(dtype, &raw_row, &mut decoded).map_err(|reason| {
                PayloadError::UnsupportedDecode {
                    tensor: info.name.clone(),
                    reason,
                }
            })?;
            out.insert(r, decoded);
        }
        Ok(out)
    }

    /// Read a 2D tensor directly into compact BF16 `u16` bits without allocating an `f64` matrix
    /// (4x less RAM than `read_tensor_matrix`; zero-copy bit copy for native BF16 checkpoints).
    pub fn read_tensor_bf16_bits(
        &mut self,
        name: &str,
        budget: &PayloadBudget,
    ) -> Result<(usize, usize, Vec<u16>), PayloadError> {
        let info = self
            .tensor_metadata(name)
            .ok_or_else(|| PayloadError::TensorNotFound(name.to_owned()))?
            .clone();
        if info.shape.len() != 2 {
            return Err(PayloadError::NotTwoDimensional {
                tensor: info.name,
                shape: info.shape,
            });
        }
        let rows = usize::try_from(info.shape[0]).map_err(|_| PayloadError::OffsetOverflow {
            tensor: info.name.clone(),
        })?;
        let cols = usize::try_from(info.shape[1]).map_err(|_| PayloadError::OffsetOverflow {
            tensor: info.name.clone(),
        })?;
        let dtype = DType::parse(&info.dtype).ok_or_else(|| PayloadError::UnsupportedDecode {
            tensor: info.name.clone(),
            reason: format!("unknown dtype {:?}", info.dtype),
        })?;
        let raw = self.read_tensor_bytes(name, budget)?;
        let bits: Vec<u16> = match dtype {
            DType::BF16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| u16::from_le_bytes(c))
                .collect(),
            DType::F16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| {
                    let v = crate::dtype::f16_to_f32(u16::from_le_bytes(c));
                    crate::dtype::f32_to_bf16(v)
                })
                .collect(),
            DType::F32 => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| {
                    let v = f32::from_le_bytes(c);
                    crate::dtype::f32_to_bf16(v)
                })
                .collect(),
            _ => {
                let mut f64s = Vec::with_capacity(rows * cols);
                decode_chunk_to_f64(dtype, &raw, &mut f64s).map_err(|reason| {
                    PayloadError::UnsupportedDecode {
                        tensor: info.name,
                        reason,
                    }
                })?;
                f64s.into_iter()
                    .map(|v| crate::dtype::f32_to_bf16(v as f32))
                    .collect()
            }
        };
        Ok((rows, cols, bits))
    }

    /// Stream a tensor in fixed-size chunks decoded to `f64`, using constant memory.
    ///
    /// Does not accumulate the full tensor in memory and does not count against
    /// `max_tensor_bytes`, enabling full-model statistical inspection of multi-GB tensors
    /// in < 1 MiB of working RAM.
    pub fn stream_tensor_f64<F>(
        &mut self,
        name: &str,
        chunk_elements: usize,
        mut on_chunk: F,
    ) -> Result<u64, PayloadError>
    where
        F: FnMut(&[f64]),
    {
        if chunk_elements == 0 {
            return Err(PayloadError::InvalidChunkSize);
        }
        let info = self
            .tensor_metadata(name)
            .ok_or_else(|| PayloadError::TensorNotFound(name.to_owned()))?
            .clone();
        let dtype = DType::parse(&info.dtype).ok_or_else(|| PayloadError::UnsupportedDecode {
            tensor: info.name.clone(),
            reason: format!("unknown dtype {:?}", info.dtype),
        })?;
        if !dtype.supports_f64_decode() {
            return Err(PayloadError::UnsupportedDecode {
                tensor: info.name,
                reason: format!("dtype {dtype} does not support scalar f64 decoding"),
            });
        }
        let elem_bytes = dtype
            .byte_width()
            .ok_or_else(|| PayloadError::UnsupportedDecode {
                tensor: info.name.clone(),
                reason: format!("dtype {dtype} is sub-byte"),
            })?;
        let file_offset = self
            .data_start
            .checked_add(info.data_offsets[0])
            .ok_or_else(|| PayloadError::OffsetOverflow {
                tensor: info.name.clone(),
            })?;
        self.reader.seek(SeekFrom::Start(file_offset))?;

        let total_elements = tensor_element_count(&info)?;
        let capped_chunk = chunk_elements.min(1_048_576);
        let chunk_bytes = capped_chunk
            .checked_mul(elem_bytes)
            .ok_or(PayloadError::InvalidChunkSize)?;
        let mut raw_buf = vec![0_u8; chunk_bytes];
        let mut decoded_buf = Vec::with_capacity(capped_chunk);

        let mut remaining = total_elements;
        while remaining > 0 {
            let take_elems = (remaining.min(capped_chunk as u64)) as usize;
            let take_bytes = take_elems * elem_bytes;
            self.reader.read_exact(&mut raw_buf[..take_bytes])?;
            decode_chunk_to_f64(dtype, &raw_buf[..take_bytes], &mut decoded_buf).map_err(
                |reason| PayloadError::UnsupportedDecode {
                    tensor: info.name.clone(),
                    reason,
                },
            )?;
            on_chunk(&decoded_buf);
            remaining -= take_elems as u64;
        }
        Ok(total_elements)
    }
}

/// Compute element count for a validated tensor's shape.
pub fn tensor_element_count(info: &TensorMetadata) -> Result<u64, PayloadError> {
    info.shape
        .iter()
        .try_fold(1_u64, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| PayloadError::OffsetOverflow {
            tensor: info.name.clone(),
        })
}

/// Validated metadata from a Hugging Face sharded Safetensors index (`model.safetensors.index.json`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SafetensorsShardIndex {
    pub total_size: Option<u64>,
    pub shard_files: Vec<String>,
    pub weight_map: BTreeMap<String, String>,
}

/// Parse a bounded `model.safetensors.index.json` safely, rejecting path traversal shard names.
pub fn parse_shard_index<R: Read>(reader: R) -> Result<SafetensorsShardIndex, PayloadError> {
    let mut bytes = Vec::new();
    reader
        .take(MAX_SHARD_INDEX_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_SHARD_INDEX_BYTES {
        return Err(PayloadError::InvalidShardIndex(format!(
            "shard index exceeds {MAX_SHARD_INDEX_BYTES} byte limit"
        )));
    }

    #[derive(Deserialize)]
    struct RawIndex {
        #[serde(default)]
        metadata: Option<RawIndexMetadata>,
        weight_map: BTreeMap<String, String>,
    }

    #[derive(Deserialize)]
    struct RawIndexMetadata {
        #[serde(default)]
        total_size: Option<u64>,
    }

    let raw: RawIndex = serde_json::from_slice(&bytes)
        .map_err(|e| PayloadError::InvalidShardIndex(e.to_string()))?;

    let mut shard_set = std::collections::BTreeSet::new();
    for (tensor, shard) in &raw.weight_map {
        if shard.is_empty()
            || shard.contains("..")
            || shard.starts_with('/')
            || shard.starts_with('\\')
            || shard.contains(':')
        {
            return Err(PayloadError::InvalidShardIndex(format!(
                "unsafe shard filename {shard:?} referenced by tensor {tensor:?}"
            )));
        }
        shard_set.insert(shard.clone());
    }

    Ok(SafetensorsShardIndex {
        total_size: raw.metadata.and_then(|m| m.total_size),
        shard_files: shard_set.into_iter().collect(),
        weight_map: raw.weight_map,
    })
}

/// Unified read-only Safetensors source supporting BOTH single `.safetensors` files AND
/// multi-shard Hugging Face `model.safetensors.index.json` directories with bounded memory
/// (keeps at most ONE shard file open at any time so 3B–32B checkpoints stream safely).
pub struct SafetensorsSource {
    base_dir: std::path::PathBuf,
    single_file: Option<std::path::PathBuf>,
    tensor_to_shard: BTreeMap<String, String>,
    metadata: SafetensorsMetadata,
    cached_shard: Option<(String, SafetensorsFile<std::fs::File>)>,
}

impl SafetensorsSource {
    /// Open either a single `.safetensors` file OR a `model.safetensors.index.json` shard index
    /// (or a directory containing `model.safetensors.index.json` / `model.safetensors`).
    pub fn open(path: &std::path::Path, budget: &PayloadBudget) -> Result<Self, PayloadError> {
        let resolved = if path.is_dir() {
            let idx = path.join("model.safetensors.index.json");
            if idx.exists() {
                idx
            } else {
                path.join("model.safetensors")
            }
        } else {
            path.to_path_buf()
        };

        let is_index = resolved
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".index.json"));

        if is_index {
            let base_dir = resolved
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .to_path_buf();
            let idx_file = std::fs::File::open(&resolved)?;
            let shard_index = parse_shard_index(idx_file)?;
            let mut total_file_bytes = 0_u64;
            let mut total_header_bytes = 0_u64;
            let mut total_data_bytes = 0_u64;
            let mut all_tensors = Vec::new();
            let mut combined_meta = BTreeMap::new();
            let mut tensor_to_shard = BTreeMap::new();

            for shard_name in &shard_index.shard_files {
                let shard_path = base_dir.join(shard_name);
                let f = std::fs::File::open(&shard_path)?;
                let st = SafetensorsFile::open(f, budget)?;
                let m = st.metadata();
                total_file_bytes = total_file_bytes.saturating_add(m.file_bytes);
                total_header_bytes = total_header_bytes.saturating_add(m.header_bytes);
                total_data_bytes = total_data_bytes.saturating_add(m.data_bytes);
                for (k, v) in &m.metadata {
                    combined_meta.entry(k.clone()).or_insert_with(|| v.clone());
                }
                for t in &m.tensors {
                    tensor_to_shard.insert(t.name.clone(), shard_name.clone());
                    all_tensors.push(t.clone());
                }
            }
            all_tensors.sort_by(|a, b| a.name.cmp(&b.name));
            let metadata = SafetensorsMetadata {
                file_bytes: total_file_bytes,
                header_bytes: total_header_bytes,
                data_bytes: total_data_bytes,
                tensors: all_tensors,
                metadata: combined_meta,
            };
            Ok(Self {
                base_dir,
                single_file: None,
                tensor_to_shard,
                metadata,
                cached_shard: None,
            })
        } else {
            let f = std::fs::File::open(&resolved)?;
            let st = SafetensorsFile::open(f, budget)?;
            let metadata = st.metadata().clone();
            let base_dir = resolved
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .to_path_buf();
            Ok(Self {
                base_dir,
                single_file: Some(resolved.clone()),
                tensor_to_shard: BTreeMap::new(),
                metadata,
                cached_shard: Some((resolved.to_string_lossy().into_owned(), st)),
            })
        }
    }

    pub fn metadata(&self) -> &SafetensorsMetadata {
        &self.metadata
    }

    /// All backing weight paths for provenance and source protection.
    pub fn source_paths(&self) -> Vec<std::path::PathBuf> {
        if let Some(path) = &self.single_file {
            return vec![path.clone()];
        }
        self.tensor_to_shard
            .values()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|name| self.base_dir.join(name))
            .collect()
    }

    pub fn is_sharded(&self) -> bool {
        self.single_file.is_none()
    }

    pub fn shard_count(&self) -> usize {
        if self.single_file.is_some() {
            1
        } else {
            self.tensor_to_shard
                .values()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        }
    }

    fn shard_for_tensor(
        &mut self,
        name: &str,
        budget: &PayloadBudget,
    ) -> Result<&mut SafetensorsFile<std::fs::File>, PayloadError> {
        let shard_key = if let Some(single) = &self.single_file {
            single.to_string_lossy().into_owned()
        } else {
            self.tensor_to_shard
                .get(name)
                .cloned()
                .ok_or_else(|| PayloadError::TensorNotFound(name.to_owned()))?
        };

        let needs_open = match &self.cached_shard {
            Some((cur, _)) => cur != &shard_key,
            None => true,
        };
        if needs_open {
            // Drop previous shard handle before opening the new one (bounded RAM & file descriptors)
            self.cached_shard = None;
            let path = if self.single_file.is_some() {
                std::path::PathBuf::from(&shard_key)
            } else {
                self.base_dir.join(&shard_key)
            };
            let f = std::fs::File::open(&path)?;
            let st = SafetensorsFile::open(f, budget)?;
            self.cached_shard = Some((shard_key, st));
        }
        Ok(&mut self.cached_shard.as_mut().expect("opened").1)
    }

    pub fn read_tensor_f64(
        &mut self,
        name: &str,
        budget: &PayloadBudget,
    ) -> Result<Vec<f64>, PayloadError> {
        let st = self.shard_for_tensor(name, budget)?;
        st.read_tensor_f64(name, budget)
    }

    pub fn read_tensor_matrix(
        &mut self,
        name: &str,
        budget: &PayloadBudget,
    ) -> Result<DenseMatrix, PayloadError> {
        let st = self.shard_for_tensor(name, budget)?;
        st.read_tensor_matrix(name, budget)
    }

    pub fn read_tensor_rows_f64(
        &mut self,
        name: &str,
        row_indices: &[usize],
        budget: &PayloadBudget,
    ) -> Result<BTreeMap<usize, Vec<f64>>, PayloadError> {
        let st = self.shard_for_tensor(name, budget)?;
        st.read_tensor_rows_f64(name, row_indices)
    }

    pub fn read_tensor_bf16_bits(
        &mut self,
        name: &str,
        budget: &PayloadBudget,
    ) -> Result<(usize, usize, Vec<u16>), PayloadError> {
        let st = self.shard_for_tensor(name, budget)?;
        st.read_tensor_bf16_bits(name, budget)
    }
}

/// Write a deterministic synthetic Safetensors buffer from named `(shape, dtype, raw_bytes)` entries.
///
/// Useful for reproducible tests, benchmarks, and synthetic model fixtures without external weights.
pub fn encode_safetensors(
    tensors: &[(&str, DType, &[u64], &[u8])],
    metadata: &BTreeMap<String, String>,
) -> Result<Vec<u8>, String> {
    let mut header_map = serde_json::Map::new();
    if !metadata.is_empty() {
        let mut meta_obj = serde_json::Map::new();
        for (k, v) in metadata {
            meta_obj.insert(k.clone(), serde_json::Value::String(v.clone()));
        }
        header_map.insert(
            "__metadata__".to_owned(),
            serde_json::Value::Object(meta_obj),
        );
    }

    let mut payload = Vec::new();
    for &(name, dtype, shape, raw_bytes) in tensors {
        let start = payload.len() as u64;
        payload.extend_from_slice(raw_bytes);
        let end = payload.len() as u64;

        let mut entry = serde_json::Map::new();
        entry.insert(
            "dtype".to_owned(),
            serde_json::Value::String(dtype.as_str().to_owned()),
        );
        entry.insert(
            "shape".to_owned(),
            serde_json::Value::Array(shape.iter().map(|&d| serde_json::Value::from(d)).collect()),
        );
        entry.insert(
            "data_offsets".to_owned(),
            serde_json::Value::Array(vec![
                serde_json::Value::from(start),
                serde_json::Value::from(end),
            ]),
        );
        header_map.insert(name.to_owned(), serde_json::Value::Object(entry));
    }

    let header_json =
        serde_json::to_string(&serde_json::Value::Object(header_map)).map_err(|e| e.to_string())?;
    let padded_len = header_json.len().div_ceil(8) * 8;
    let mut out = Vec::with_capacity(8 + padded_len + payload.len());
    out.extend_from_slice(&(padded_len as u64).to_le_bytes());
    out.extend_from_slice(header_json.as_bytes());
    out.resize(8 + padded_len, b' ');
    out.extend_from_slice(&payload);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::{f32_to_bf16, f32_to_f16};
    use std::io::Cursor;

    fn sample_multi_dtype_file() -> Vec<u8> {
        let f32_vals: Vec<u8> = [1.5_f32, -2.25, 3.0, 0.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f16_vals: Vec<u8> = [1.0_f32, -0.5, 2.0, 4.0]
            .iter()
            .flat_map(|v| f32_to_f16(*v).to_le_bytes())
            .collect();
        let bf16_vals: Vec<u8> = [0.25_f32, -3.5]
            .iter()
            .flat_map(|v| f32_to_bf16(*v).to_le_bytes())
            .collect();
        let mut meta = BTreeMap::new();
        meta.insert("fixture".to_owned(), "synthetic".to_owned());
        encode_safetensors(
            &[
                ("w_f32", DType::F32, &[2, 2], &f32_vals),
                ("w_f16", DType::F16, &[2, 2], &f16_vals),
                ("b_bf16", DType::BF16, &[2], &bf16_vals),
            ],
            &meta,
        )
        .unwrap()
    }

    #[test]
    fn reads_and_decodes_multiple_dtypes_and_matrices() {
        let bytes = sample_multi_dtype_file();
        let budget = PayloadBudget::default();
        let mut file = SafetensorsFile::open(Cursor::new(bytes), &budget).unwrap();

        let mat_f32 = file.read_tensor_matrix("w_f32", &budget).unwrap();
        assert_eq!(mat_f32.rows(), 2);
        assert_eq!(mat_f32.cols(), 2);
        assert_eq!(mat_f32.as_slice(), &[1.5, -2.25, 3.0, 0.5]);

        let mat_f16 = file.read_tensor_matrix("w_f16", &budget).unwrap();
        assert_eq!(mat_f16.as_slice(), &[1.0, -0.5, 2.0, 4.0]);

        let vec_bf16 = file.read_tensor_f64("b_bf16", &budget).unwrap();
        assert_eq!(vec_bf16, vec![0.25, -3.5]);
    }

    #[test]
    fn streams_tensors_in_tiny_chunks_accurately() {
        let bytes = sample_multi_dtype_file();
        let budget = PayloadBudget::default();
        let mut file = SafetensorsFile::open(Cursor::new(bytes), &budget).unwrap();

        let mut collected = Vec::new();
        let mut chunks_seen = 0;
        let count = file
            .stream_tensor_f64("w_f32", 2, |chunk| {
                chunks_seen += 1;
                collected.extend_from_slice(chunk);
            })
            .unwrap();
        assert_eq!(count, 4);
        assert_eq!(chunks_seen, 2);
        assert_eq!(collected, vec![1.5, -2.25, 3.0, 0.5]);
    }

    #[test]
    fn enforces_per_tensor_and_cumulative_memory_budgets() {
        let bytes = sample_multi_dtype_file();
        let tight = PayloadBudget {
            max_tensor_bytes: 16, // w_f32 needs 32 bytes when decoded to f64
            max_total_bytes: 20,
            ..PayloadBudget::default()
        };
        let mut file = SafetensorsFile::open(Cursor::new(bytes), &tight).unwrap();
        assert!(matches!(
            file.read_tensor_f64("w_f32", &tight),
            Err(PayloadError::TensorExceedsBudget { .. })
        ));
        // Raw read of w_f32 is 16 bytes (fits), then reading w_f16 (8 bytes) exceeds total 20.
        assert_eq!(file.read_tensor_bytes("w_f32", &tight).unwrap().len(), 16);
        assert!(matches!(
            file.read_tensor_bytes("w_f16", &tight),
            Err(PayloadError::TotalBudgetExceeded { .. })
        ));
    }

    #[test]
    fn shard_index_parses_valid_and_rejects_path_traversal() {
        let valid = br#"{"metadata":{"total_size":1024},"weight_map":{"a":"model-00001.safetensors","b":"model-00002.safetensors"}}"#;
        let parsed = parse_shard_index(Cursor::new(valid)).unwrap();
        assert_eq!(parsed.total_size, Some(1024));
        assert_eq!(
            parsed.shard_files,
            vec!["model-00001.safetensors", "model-00002.safetensors"]
        );

        let evil = br#"{"weight_map":{"a":"../secret.safetensors"}}"#;
        assert!(matches!(
            parse_shard_index(Cursor::new(evil)),
            Err(PayloadError::InvalidShardIndex(_))
        ));
    }
}
