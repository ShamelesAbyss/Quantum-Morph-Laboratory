//! Experimental QMF v1 generalized neural representation container.
//!
//! Supports multiple tensors, shared latent dictionaries (for RMQ), dense
//! FP32/FP16/BF16, groupwise INT8/INT4 (`Q4_0`/`Q4_1`), truncated SVD factors,
//! RMQ shared-basis & shared-core layers, sparse top-k outlier residuals,
//! model architecture/license provenance, and bounded verified parsing.
//!
//! **Status:** Experimental format; not compatibility-stable. CRC-32 checks
//! detect accidental corruption only and do not provide cryptographic authentication.

mod disk;
pub use disk::{ArtifactLimits, BuilderMemoryStats, DiskArtifact, DiskBuildOptions};
use disk::{COPY_BUFFER_BYTES, DiskPayload, ManifestCounter, PayloadStore};

use crate::{Crc32, QMF_CHECKSUM_BYTES, QMF_MAGIC, QmfError, crc32};
use qml_core::{DenseMatrix, MatrixError, Sha256};
use qml_quant::baselines::{QuantizedFactor, SparseResidual};
use qml_quant::{
    Bf16Matrix, EchoBaseOperator, EchoCorrectedMatrix, FactorPrecision, Fp16Matrix, HadamardSpec,
    Int2AsymGroupMatrix, Int3AsymGroupMatrix, Int4AsymGroupMatrix, Int4SymGroupMatrix,
    Int8SymMatrix, LinearOperator, LowRankSvdMatrix, RmqSharedBasisCluster, RmqSharedCoreCluster,
    StorageBreakdown, TernaryPackedMatrix, VectorQuantMatrix,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

pub const QMF_V1_VERSION: u16 = 1;
pub const QMF_V1_HEADER_BYTES: usize = 32;
pub const QMF_V1_MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
/// Default QMF v1 file-size budget (8 GiB) replacing the legacy 256 MiB prototype ceiling.
pub const DEFAULT_QMF_V1_MAX_FILE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
pub const QMF_V1_FLAG_LITTLE_ENDIAN: u16 = 0x0001;

/// Explicit resource budget for bounded QMF v1 serialization, parsing, and streaming access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QmfBudget {
    pub max_manifest_bytes: usize,
    pub max_file_bytes: u64,
}

impl Default for QmfBudget {
    fn default() -> Self {
        Self {
            max_manifest_bytes: QMF_V1_MAX_MANIFEST_BYTES,
            max_file_bytes: DEFAULT_QMF_V1_MAX_FILE_BYTES,
        }
    }
}

/// Deduplicating cache of decoded shared latent factors (`Arc<QuantizedFactor>`) so
/// multiple layer operators genuinely share a single decoded copy in memory.
#[derive(Debug, Default, Clone)]
pub struct QmfLatentCache {
    decoded: BTreeMap<String, Arc<QuantizedFactor>>,
}

impl QmfLatentCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.decoded.len()
    }

    pub fn is_empty(&self) -> bool {
        self.decoded.is_empty()
    }

    pub fn get(&self, id: &str) -> Option<&Arc<QuantizedFactor>> {
        self.decoded.get(id)
    }

    pub fn insert(&mut self, id: String, factor: Arc<QuantizedFactor>) {
        self.decoded.insert(id, factor);
    }
}

/// Slice descriptor referencing a contiguous byte span inside the QMF v1 payload section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobRef {
    pub offset: u64,
    pub len: u64,
    pub crc32: u32,
}

/// Encoding for a shared latent matrix (e.g., RMQ `U_shared`, `V_shared`, or `S_core`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SharedLatentEncoding {
    Fp32 {
        data: BlobRef,
    },
    Fp16 {
        data: BlobRef,
    },
    Int8Sym {
        group_size: usize,
        scales: BlobRef,
        codes: BlobRef,
    },
}

/// Entry in the QMF v1 shared latent table, referenced by one or more layers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedLatentEntry {
    pub id: String,
    pub rows: usize,
    pub cols: usize,
    pub encoding: SharedLatentEncoding,
}

/// Descriptor for an optional sparse COO FP16 outlier residual attached to a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseResidualDesc {
    pub nnz: usize,
    pub indices: BlobRef,
    pub values_f16: BlobRef,
}

/// Supported per-tensor representations inside QMF v1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum QmfRepresentation {
    DenseF32 {
        data: BlobRef,
    },
    DenseF16 {
        data: BlobRef,
    },
    DenseBf16 {
        data: BlobRef,
    },
    Int8SymGroup {
        group_size: usize,
        scales: BlobRef,
        codes: BlobRef,
    },
    Int4SymGroup {
        group_size: usize,
        scales: BlobRef,
        nibbles: BlobRef,
    },
    Int4AsymGroup {
        group_size: usize,
        scales: BlobRef,
        mins: BlobRef,
        nibbles: BlobRef,
    },
    LowRankSvd {
        rank: usize,
        left: SharedLatentEncoding,
        right: SharedLatentEncoding,
    },
    RmqSharedBasis {
        u_latent_id: String,
        vt_latent_id: String,
        left_rank: usize,
        right_rank: usize,
        core: SharedLatentEncoding,
        sparse_residual: Option<SparseResidualDesc>,
    },
    RmqSharedCore {
        s_latent_id: String,
        rank: usize,
        left: SharedLatentEncoding,
        right: SharedLatentEncoding,
        sparse_residual: Option<SparseResidualDesc>,
    },
    /// PROJECT RESONANCE: 3-bit asymmetric groupwise quantization with optional Hadamard rotation.
    Int3AsymGroup {
        group_size: usize,
        hadamard_block: usize,
        hadamard_seed: u64,
        scales: BlobRef,
        mins: BlobRef,
        packed: BlobRef,
    },
    /// PROJECT RESONANCE: 2-bit asymmetric groupwise quantization with optional Hadamard rotation.
    Int2AsymGroup {
        group_size: usize,
        hadamard_block: usize,
        hadamard_seed: u64,
        scales: BlobRef,
        mins: BlobRef,
        packed: BlobRef,
    },
    /// PROJECT RESONANCE: Packed base-3 ternary (5 trits/byte, 1.60 bpp raw) + optional Hadamard.
    TernaryPacked {
        group_size: usize,
        hadamard_block: usize,
        hadamard_seed: u64,
        scales: BlobRef,
        packed: BlobRef,
    },
    /// PROJECT RESONANCE: Learned 1- or 2-stage Residual Vector Quantization + optional Hadamard.
    VectorQuant {
        subvec_dim: usize,
        codebook_size: usize,
        num_stages: usize,
        group_size: usize,
        hadamard_block: usize,
        hadamard_seed: u64,
        codebooks_f16: BlobRef,
        scales: BlobRef,
        codes: BlobRef,
    },
    /// PROJECT ECHO: Compact low-rank INT8 adapter and/or sparse FP16 outlier recovery wrapped
    /// around any sub-3-bit base representation.
    EchoCorrected {
        base: Box<QmfRepresentation>,
        low_rank_r: usize,
        low_rank_left: Option<SharedLatentEncoding>,
        low_rank_right: Option<SharedLatentEncoding>,
        sparse_residual: Option<SparseResidualDesc>,
    },
}

/// Directory entry for one tensor in QMF v1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QmfTensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub representation: QmfRepresentation,
}

/// Complete manifest embedded at the start of a QMF v1 container.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QmfManifest {
    pub format_status: String,
    pub model_id: String,
    pub source_format: String,
    pub source_license: String,
    pub architecture: String,
    pub transformation: String,
    pub random_seed: u64,
    pub extra_metadata: BTreeMap<String, String>,
    pub shared_latents: Vec<SharedLatentEntry>,
    pub tensors: Vec<QmfTensorEntry>,
}

/// In-memory validated QMF v1 container supporting deterministic serialization and direct execution.
#[derive(Debug, Clone, PartialEq)]
pub struct QmfArchiveV1 {
    manifest: QmfManifest,
    payload: Vec<u8>,
}

/// Builder that packs tensors and shared latent structures into a deterministic QMF v1 archive.
pub struct QmfV1Builder {
    manifest: QmfManifest,
    payload: PayloadStore,
    latent_ids: BTreeSet<String>,
    tensor_names: BTreeSet<String>,
    metadata_error: Option<String>,
}

impl QmfV1Builder {
    pub fn new(
        model_id: impl Into<String>,
        source_format: impl Into<String>,
        source_license: impl Into<String>,
        architecture: impl Into<String>,
        transformation: impl Into<String>,
        random_seed: u64,
    ) -> Self {
        Self {
            manifest: QmfManifest {
                format_status: "experimental-qmf-v1".to_owned(),
                model_id: model_id.into(),
                source_format: source_format.into(),
                source_license: source_license.into(),
                architecture: architecture.into(),
                transformation: transformation.into(),
                random_seed,
                extra_metadata: BTreeMap::new(),
                shared_latents: Vec::new(),
                tensors: Vec::new(),
            },
            payload: PayloadStore::Memory(Vec::new()),
            latent_ids: BTreeSet::new(),
            tensor_names: BTreeSet::new(),
            metadata_error: None,
        }
    }

    pub fn insert_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        if let Err(error) = self.try_insert_metadata(key, value) {
            self.metadata_error = Some(error.to_string());
        }
    }

    /// Fallible metadata insertion for resource-constrained construction.
    pub fn try_insert_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<(), QmfError> {
        let key = key.into();
        let value = value.into();
        self.guard_name(&key)?;
        self.guard_name(&value)?;
        self.manifest.extra_metadata.insert(key, value);
        self.checkpoint()
    }

    pub fn new_disk(
        model_id: impl Into<String>,
        source_format: impl Into<String>,
        source_license: impl Into<String>,
        architecture: impl Into<String>,
        transformation: impl Into<String>,
        random_seed: u64,
        options: DiskBuildOptions,
    ) -> Result<Self, QmfError> {
        disk::cancelled(&options)?;
        options.limits.check(
            "resident encoding buffer",
            COPY_BUFFER_BYTES as u64,
            options.limits.max_resident_bytes,
        )?;
        std::fs::create_dir_all(&options.scratch_directory)?;
        disk::disk_preflight(&options.scratch_directory, 4096)?;
        let file = tempfile::Builder::new()
            .prefix(".qml-payload-")
            .tempfile_in(&options.scratch_directory)?;
        let mut builder = Self::new(
            model_id,
            source_format,
            source_license,
            architecture,
            transformation,
            random_seed,
        );
        builder.payload = PayloadStore::Disk(Box::new(DiskPayload {
            file,
            options,
            bytes: 0,
            stats: BuilderMemoryStats::default(),
        }));
        builder.checkpoint()?;
        Ok(builder)
    }
    fn guard_name(&self, name: &str) -> Result<(), QmfError> {
        if let PayloadStore::Disk(d) = &self.payload {
            let mut count = ManifestCounter {
                bytes: 0,
                limit: d.options.limits.max_manifest_bytes as u64,
            };
            serde_json::to_writer(&mut count, &self.manifest)
                .map_err(|e| QmfError::InvalidManifest(e.to_string()))?;
            let incoming = (name.len() as u64)
                .checked_mul(6)
                .and_then(|n| n.checked_add(2048))
                .ok_or(QmfError::PayloadTooLarge)?;
            let reserved = count
                .bytes
                .checked_add(incoming)
                .ok_or(QmfError::PayloadTooLarge)?;
            d.options.limits.check(
                "manifest reservation",
                reserved,
                d.options.limits.max_manifest_bytes as u64,
            )?;
            let resident = reserved
                .checked_mul(16)
                .and_then(|n| n.checked_add((2 * COPY_BUFFER_BYTES) as u64))
                .ok_or(QmfError::PayloadTooLarge)?;
            d.options.limits.check(
                "builder resident reservation",
                resident,
                d.options.limits.max_resident_bytes,
            )?;
            d.options.limits.check(
                "manifest name",
                name.len() as u64,
                d.options.limits.max_manifest_bytes as u64,
            )?;
        }
        Ok(())
    }
    /// Conservative manifest accounting (serialized bytes × 16), including strings, nodes and capacities.
    fn checkpoint(&mut self) -> Result<(), QmfError> {
        if let Some(error) = &self.metadata_error {
            return Err(QmfError::InvalidManifest(error.clone()));
        }
        if let PayloadStore::Disk(d) = &mut self.payload {
            disk::cancelled(&d.options)?;
            let mut count = ManifestCounter {
                bytes: 0,
                limit: d.options.limits.max_manifest_bytes as u64,
            };
            serde_json::to_writer(&mut count, &self.manifest)
                .map_err(|e| QmfError::InvalidManifest(e.to_string()))?;
            let resident = count
                .bytes
                .checked_mul(16)
                .and_then(|n| n.checked_add((2 * COPY_BUFFER_BYTES) as u64))
                .ok_or(QmfError::PayloadTooLarge)?;
            d.options.limits.check(
                "builder resident accounting",
                resident,
                d.options.limits.max_resident_bytes,
            )?;
            d.options.limits.total_bytes(count.bytes, d.bytes)?;
            d.stats.manifest_accounted_bytes = resident;
        }
        Ok(())
    }
    pub fn memory_stats(&self) -> Option<&BuilderMemoryStats> {
        match &self.payload {
            PayloadStore::Disk(d) => Some(&d.stats),
            _ => None,
        }
    }
    pub fn finish_disk(mut self) -> Result<DiskArtifact, QmfError> {
        self.checkpoint()?;
        self.manifest.shared_latents.sort_by(|a, b| a.id.cmp(&b.id));
        self.manifest.tensors.sort_by(|a, b| a.name.cmp(&b.name));
        match self.payload {
            PayloadStore::Disk(payload) => Ok(DiskArtifact::new(self.manifest, payload)),
            _ => Err(QmfError::InvalidManifest(
                "memory builder requires build".into(),
            )),
        }
    }
    fn append_blob(&mut self, bytes: &[u8]) -> Result<BlobRef, QmfError> {
        self.checkpoint()?;
        if let PayloadStore::Disk(d) = &self.payload {
            d.options.limits.check(
                "individual encoded blob",
                bytes.len() as u64,
                d.options.limits.max_tensor_bytes,
            )?;
        }
        let offset = self.payload.len();
        let mut crc = Crc32::new();
        for chunk in bytes.chunks(COPY_BUFFER_BYTES) {
            self.payload.append(chunk)?;
            crc.update(chunk);
        }
        Ok(BlobRef {
            offset,
            len: bytes.len() as u64,
            crc32: crc.finish(),
        })
    }
    fn append_words<const N: usize>(
        &mut self,
        words: impl ExactSizeIterator<Item = [u8; N]>,
    ) -> Result<BlobRef, QmfError> {
        self.checkpoint()?;
        let bytes = words
            .len()
            .checked_mul(N)
            .ok_or(QmfError::PayloadTooLarge)?;
        if let PayloadStore::Disk(d) = &mut self.payload {
            d.options.limits.check(
                "individual encoded blob",
                bytes as u64,
                d.options.limits.max_tensor_bytes,
            )?;
            d.stats.peak_encoding_buffer_bytes =
                d.stats.peak_encoding_buffer_bytes.max(COPY_BUFFER_BYTES);
        }
        let offset = self.payload.len();
        let mut crc = Crc32::new();
        let mut buffer = [0u8; COPY_BUFFER_BYTES];
        let mut filled = 0;
        for word in words {
            if N > buffer.len() {
                return Err(QmfError::PayloadTooLarge);
            }
            if filled + N > buffer.len() {
                self.payload.append(&buffer[..filled])?;
                crc.update(&buffer[..filled]);
                filled = 0;
            }
            buffer[filled..filled + N].copy_from_slice(&word);
            filled += N;
        }
        if filled > 0 {
            self.payload.append(&buffer[..filled])?;
            crc.update(&buffer[..filled]);
        }
        Ok(BlobRef {
            offset,
            len: bytes as u64,
            crc32: crc.finish(),
        })
    }

    fn encode_factor(
        &mut self,
        factor: &QuantizedFactor,
    ) -> Result<SharedLatentEncoding, QmfError> {
        Ok(match factor {
            QuantizedFactor::Fp32(mat) => {
                let data =
                    self.append_words(mat.as_slice().iter().map(|&v| (v as f32).to_le_bytes()))?;
                SharedLatentEncoding::Fp32 { data }
            }
            QuantizedFactor::Fp16(mat) => {
                let data = self.encode_u16_slice(mat.raw_bits())?;
                SharedLatentEncoding::Fp16 { data }
            }
            QuantizedFactor::Int8Sym(mat) => {
                let scales = self.encode_u16_slice(mat.scales_f16())?;
                let codes = self.append_words(mat.codes().iter().map(|&c| [c as u8]))?;

                SharedLatentEncoding::Int8Sym {
                    group_size: mat.group_size(),
                    scales,
                    codes,
                }
            }
        })
    }

    fn encode_sparse(
        &mut self,
        sp: &SparseResidual,
    ) -> Result<Option<SparseResidualDesc>, QmfError> {
        if sp.nnz() == 0 {
            return Ok(None);
        }
        let indices = self.append_words(sp.indices().iter().map(|i| i.to_le_bytes()))?;
        let values_f16 = self.encode_u16_slice(sp.values_f16())?;
        Ok(Some(SparseResidualDesc {
            nnz: sp.nnz(),
            indices,
            values_f16,
        }))
    }

    fn check_unique_tensor(&mut self, name: &str) -> Result<(), QmfError> {
        self.guard_name(name)?;
        self.checkpoint()?;
        if !self.tensor_names.insert(name.to_owned()) {
            return Err(QmfError::InvalidManifest(format!(
                "duplicate tensor name {name:?}"
            )));
        }
        Ok(())
    }

    pub fn add_dense_f32(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        values: &[f32],
    ) -> Result<(), QmfError> {
        let name = name.into();
        self.check_unique_tensor(&name)?;
        for &v in values {
            if !v.is_finite() {
                return Err(QmfError::Matrix(MatrixError::NonFiniteValue { index: 0 }));
            }
        }
        let data = self.append_words(values.iter().map(|v| v.to_le_bytes()))?;

        self.manifest.tensors.push(QmfTensorEntry {
            name,
            shape,
            representation: QmfRepresentation::DenseF32 { data },
        });
        Ok(())
    }

    pub fn add_fp16(&mut self, name: impl Into<String>, mat: &Fp16Matrix) -> Result<(), QmfError> {
        let name = name.into();
        self.check_unique_tensor(&name)?;
        let data = self.encode_u16_slice(mat.raw_bits())?;

        self.manifest.tensors.push(QmfTensorEntry {
            name,
            shape: vec![mat.rows(), mat.cols()],
            representation: QmfRepresentation::DenseF16 { data },
        });
        Ok(())
    }

    pub fn add_bf16(&mut self, name: impl Into<String>, mat: &Bf16Matrix) -> Result<(), QmfError> {
        let name = name.into();
        self.check_unique_tensor(&name)?;
        let data = self.encode_u16_slice(mat.raw_bits())?;

        self.manifest.tensors.push(QmfTensorEntry {
            name,
            shape: vec![mat.rows(), mat.cols()],
            representation: QmfRepresentation::DenseBf16 { data },
        });
        Ok(())
    }

    pub fn add_int8_sym(
        &mut self,
        name: impl Into<String>,
        mat: &Int8SymMatrix,
    ) -> Result<(), QmfError> {
        let name = name.into();
        self.check_unique_tensor(&name)?;
        let scales = self.encode_u16_slice(mat.scales_f16())?;
        let codes = self.append_words(mat.codes().iter().map(|&c| [c as u8]))?;

        self.manifest.tensors.push(QmfTensorEntry {
            name,
            shape: vec![mat.rows(), mat.cols()],
            representation: QmfRepresentation::Int8SymGroup {
                group_size: mat.group_size(),
                scales,
                codes,
            },
        });
        Ok(())
    }

    pub fn add_int4_sym(
        &mut self,
        name: impl Into<String>,
        mat: &Int4SymGroupMatrix,
    ) -> Result<(), QmfError> {
        let name = name.into();
        self.check_unique_tensor(&name)?;
        let scales = self.encode_u16_slice(mat.scales_f16())?;

        let nibbles = self.append_blob(mat.packed_nibbles())?;
        self.manifest.tensors.push(QmfTensorEntry {
            name,
            shape: vec![mat.rows(), mat.cols()],
            representation: QmfRepresentation::Int4SymGroup {
                group_size: mat.group_size(),
                scales,
                nibbles,
            },
        });
        Ok(())
    }

    pub fn add_int4_asym(
        &mut self,
        name: impl Into<String>,
        mat: &Int4AsymGroupMatrix,
    ) -> Result<(), QmfError> {
        let name = name.into();
        self.check_unique_tensor(&name)?;
        let scales = self.encode_u16_slice(mat.scales_f16())?;
        let mins = self.encode_u16_slice(mat.mins_f16())?;

        let nibbles = self.append_blob(mat.packed_nibbles())?;
        self.manifest.tensors.push(QmfTensorEntry {
            name,
            shape: vec![mat.rows(), mat.cols()],
            representation: QmfRepresentation::Int4AsymGroup {
                group_size: mat.group_size(),
                scales,
                mins,
                nibbles,
            },
        });
        Ok(())
    }

    pub fn add_low_rank_svd(
        &mut self,
        name: impl Into<String>,
        mat: &LowRankSvdMatrix,
    ) -> Result<(), QmfError> {
        let name = name.into();
        self.check_unique_tensor(&name)?;
        let left = self.encode_factor(mat.left())?;
        let right = self.encode_factor(mat.right())?;
        self.manifest.tensors.push(QmfTensorEntry {
            name,
            shape: vec![mat.rows(), mat.cols()],
            representation: QmfRepresentation::LowRankSvd {
                rank: mat.rank(),
                left,
                right,
            },
        });
        Ok(())
    }

    /// Add an entire RMQ shared-basis cluster, storing `U_shared` and `V_sharedᵀ` ONCE in the
    /// shared latent table and referencing them across every member layer.
    pub fn add_rmq_shared_basis_cluster(
        &mut self,
        cluster_id: &str,
        layer_names: &[String],
        cluster: &RmqSharedBasisCluster,
    ) -> Result<(), QmfError> {
        self.guard_name(cluster_id)?;
        if layer_names.len() != cluster.num_layers() {
            return Err(QmfError::InvalidManifest(
                "layer_names length does not match RMQ cluster layer count".to_owned(),
            ));
        }
        let u_id = format!("{cluster_id}.u_shared");
        let vt_id = format!("{cluster_id}.vt_shared");
        if !self.latent_ids.insert(u_id.clone()) || !self.latent_ids.insert(vt_id.clone()) {
            return Err(QmfError::InvalidManifest(format!(
                "duplicate shared latent cluster {cluster_id:?}"
            )));
        }

        let u_enc = self.encode_factor(&cluster.u_shared)?;
        let vt_enc = self.encode_factor(&cluster.vt_shared)?;
        self.manifest.shared_latents.push(SharedLatentEntry {
            id: u_id.clone(),
            rows: cluster.rows,
            cols: cluster.left_rank,
            encoding: u_enc,
        });
        self.manifest.shared_latents.push(SharedLatentEntry {
            id: vt_id.clone(),
            rows: cluster.right_rank,
            cols: cluster.cols,
            encoding: vt_enc,
        });

        for (idx, name) in layer_names.iter().enumerate() {
            self.check_unique_tensor(name)?;
            let core_enc = self.encode_factor(&cluster.cores[idx])?;
            let sparse_residual = match &cluster.residuals[idx] {
                qml_quant::rmq::RmqResidual::None => None,
                qml_quant::rmq::RmqResidual::Sparse(sp) => self.encode_sparse(sp)?,
                qml_quant::rmq::RmqResidual::Int4Sym(_) => {
                    return Err(QmfError::InvalidManifest(
                        "INT4 residual serialization in RMQ v1 not yet enabled".to_owned(),
                    ));
                }
            };
            self.manifest.tensors.push(QmfTensorEntry {
                name: name.clone(),
                shape: vec![cluster.rows, cluster.cols],
                representation: QmfRepresentation::RmqSharedBasis {
                    u_latent_id: u_id.clone(),
                    vt_latent_id: vt_id.clone(),
                    left_rank: cluster.left_rank,
                    right_rank: cluster.right_rank,
                    core: core_enc,
                    sparse_residual,
                },
            });
        }
        Ok(())
    }

    /// Add an entire RMQ shared-core cluster (`W_l ≈ A_l * S * B_l + R_l`), storing `S` ONCE
    /// in the shared latent table and referencing it across every member layer.
    pub fn add_rmq_shared_core_cluster(
        &mut self,
        cluster_id: &str,
        layer_names: &[String],
        cluster: &RmqSharedCoreCluster,
    ) -> Result<(), QmfError> {
        self.guard_name(cluster_id)?;
        if layer_names.len() != cluster.left_transforms.len() {
            return Err(QmfError::InvalidManifest(
                "layer_names length does not match RMQ shared-core cluster count".to_owned(),
            ));
        }
        let s_id = format!("{cluster_id}.s_core");
        if !self.latent_ids.insert(s_id.clone()) {
            return Err(QmfError::InvalidManifest(format!(
                "duplicate shared latent cluster {cluster_id:?}"
            )));
        }
        let s_enc = self.encode_factor(&cluster.shared_core)?;
        self.manifest.shared_latents.push(SharedLatentEntry {
            id: s_id.clone(),
            rows: cluster.rank,
            cols: cluster.rank,
            encoding: s_enc,
        });

        for (idx, name) in layer_names.iter().enumerate() {
            self.check_unique_tensor(name)?;
            let left_enc = self.encode_factor(&cluster.left_transforms[idx])?;
            let right_enc = self.encode_factor(&cluster.right_transforms[idx])?;
            let sparse_residual = match &cluster.residuals[idx] {
                qml_quant::rmq::RmqResidual::None => None,
                qml_quant::rmq::RmqResidual::Sparse(sp) => self.encode_sparse(sp)?,
                qml_quant::rmq::RmqResidual::Int4Sym(_) => {
                    return Err(QmfError::InvalidManifest(
                        "INT4 residual serialization in RMQ v1 not yet enabled".to_owned(),
                    ));
                }
            };
            self.manifest.tensors.push(QmfTensorEntry {
                name: name.clone(),
                shape: vec![cluster.rows, cluster.cols],
                representation: QmfRepresentation::RmqSharedCore {
                    s_latent_id: s_id.clone(),
                    rank: cluster.rank,
                    left: left_enc,
                    right: right_enc,
                    sparse_residual,
                },
            });
        }
        Ok(())
    }

    fn encode_u16_slice(&mut self, values: &[u16]) -> Result<BlobRef, QmfError> {
        self.append_words(values.iter().map(|v| v.to_le_bytes()))
    }

    fn encode_echo_base(&mut self, base: &EchoBaseOperator) -> Result<QmfRepresentation, QmfError> {
        Ok(match base {
            EchoBaseOperator::Int3Asym(mat) => QmfRepresentation::Int3AsymGroup {
                group_size: mat.group_size(),
                hadamard_block: mat.hadamard().block_size,
                hadamard_seed: mat.hadamard().seed,
                scales: self.encode_u16_slice(mat.scales_f16())?,
                mins: self.encode_u16_slice(mat.mins_f16())?,
                packed: self.append_blob(mat.packed())?,
            },
            EchoBaseOperator::Int2Asym(mat) => QmfRepresentation::Int2AsymGroup {
                group_size: mat.group_size(),
                hadamard_block: mat.hadamard().block_size,
                hadamard_seed: mat.hadamard().seed,
                scales: self.encode_u16_slice(mat.scales_f16())?,
                mins: self.encode_u16_slice(mat.mins_f16())?,
                packed: self.append_blob(mat.packed())?,
            },
            EchoBaseOperator::Ternary(mat) => QmfRepresentation::TernaryPacked {
                group_size: mat.group_size(),
                hadamard_block: mat.hadamard().block_size,
                hadamard_seed: mat.hadamard().seed,
                scales: self.encode_u16_slice(mat.scales_f16())?,
                packed: self.append_blob(mat.packed())?,
            },
            EchoBaseOperator::VectorQuant(mat) => QmfRepresentation::VectorQuant {
                subvec_dim: mat.subvec_dim(),
                codebook_size: mat.codebook_size(),
                num_stages: mat.num_stages(),
                group_size: mat.group_size(),
                hadamard_block: mat.hadamard().block_size,
                hadamard_seed: mat.hadamard().seed,
                codebooks_f16: self.encode_u16_slice(mat.codebooks_f16())?,
                scales: self.encode_u16_slice(mat.scales_f16())?,
                codes: self.append_blob(mat.codes())?,
            },
        })
    }

    pub fn add_int3_asym(
        &mut self,
        name: impl Into<String>,
        mat: &Int3AsymGroupMatrix,
    ) -> Result<(), QmfError> {
        let name = name.into();
        self.check_unique_tensor(&name)?;
        let scales = self.encode_u16_slice(mat.scales_f16())?;
        let mins = self.encode_u16_slice(mat.mins_f16())?;
        let packed = self.append_blob(mat.packed())?;
        self.manifest.tensors.push(QmfTensorEntry {
            name,
            shape: vec![mat.rows(), mat.cols()],
            representation: QmfRepresentation::Int3AsymGroup {
                group_size: mat.group_size(),
                hadamard_block: mat.hadamard().block_size,
                hadamard_seed: mat.hadamard().seed,
                scales,
                mins,
                packed,
            },
        });
        Ok(())
    }

    pub fn add_int2_asym(
        &mut self,
        name: impl Into<String>,
        mat: &Int2AsymGroupMatrix,
    ) -> Result<(), QmfError> {
        let name = name.into();
        self.check_unique_tensor(&name)?;
        let scales = self.encode_u16_slice(mat.scales_f16())?;
        let mins = self.encode_u16_slice(mat.mins_f16())?;
        let packed = self.append_blob(mat.packed())?;
        self.manifest.tensors.push(QmfTensorEntry {
            name,
            shape: vec![mat.rows(), mat.cols()],
            representation: QmfRepresentation::Int2AsymGroup {
                group_size: mat.group_size(),
                hadamard_block: mat.hadamard().block_size,
                hadamard_seed: mat.hadamard().seed,
                scales,
                mins,
                packed,
            },
        });
        Ok(())
    }

    pub fn add_ternary_packed(
        &mut self,
        name: impl Into<String>,
        mat: &TernaryPackedMatrix,
    ) -> Result<(), QmfError> {
        let name = name.into();
        self.check_unique_tensor(&name)?;
        let scales = self.encode_u16_slice(mat.scales_f16())?;
        let packed = self.append_blob(mat.packed())?;
        self.manifest.tensors.push(QmfTensorEntry {
            name,
            shape: vec![mat.rows(), mat.cols()],
            representation: QmfRepresentation::TernaryPacked {
                group_size: mat.group_size(),
                hadamard_block: mat.hadamard().block_size,
                hadamard_seed: mat.hadamard().seed,
                scales,
                packed,
            },
        });
        Ok(())
    }

    pub fn add_vector_quant(
        &mut self,
        name: impl Into<String>,
        mat: &VectorQuantMatrix,
    ) -> Result<(), QmfError> {
        let name = name.into();
        self.check_unique_tensor(&name)?;
        let codebooks_f16 = self.encode_u16_slice(mat.codebooks_f16())?;
        let scales = self.encode_u16_slice(mat.scales_f16())?;
        let codes = self.append_blob(mat.codes())?;
        self.manifest.tensors.push(QmfTensorEntry {
            name,
            shape: vec![mat.rows(), mat.cols()],
            representation: QmfRepresentation::VectorQuant {
                subvec_dim: mat.subvec_dim(),
                codebook_size: mat.codebook_size(),
                num_stages: mat.num_stages(),
                group_size: mat.group_size(),
                hadamard_block: mat.hadamard().block_size,
                hadamard_seed: mat.hadamard().seed,
                codebooks_f16,
                scales,
                codes,
            },
        });
        Ok(())
    }

    pub fn add_echo_corrected(
        &mut self,
        name: impl Into<String>,
        mat: &EchoCorrectedMatrix,
    ) -> Result<(), QmfError> {
        let name = name.into();
        self.check_unique_tensor(&name)?;
        let base_rep = self.encode_echo_base(mat.base())?;
        let (low_rank_r, low_rank_left, low_rank_right) = if let Some(lr) = mat.low_rank_adapter() {
            (
                lr.rank(),
                Some(self.encode_factor(lr.left())?),
                Some(self.encode_factor(lr.right())?),
            )
        } else {
            (0, None, None)
        };
        let sparse_residual = match mat.sparse_residual() {
            Some(sp) => self.encode_sparse(sp)?,
            None => None,
        };
        self.manifest.tensors.push(QmfTensorEntry {
            name,
            shape: vec![mat.rows(), mat.cols()],
            representation: QmfRepresentation::EchoCorrected {
                base: Box::new(base_rep),
                low_rank_r,
                low_rank_left,
                low_rank_right,
                sparse_residual,
            },
        });
        Ok(())
    }

    pub fn build(mut self) -> Result<QmfArchiveV1, QmfError> {
        self.manifest.shared_latents.sort_by(|a, b| a.id.cmp(&b.id));
        self.manifest.tensors.sort_by(|a, b| a.name.cmp(&b.name));
        let archive = QmfArchiveV1 {
            manifest: self.manifest,
            payload: match self.payload {
                PayloadStore::Memory(v) => v,
                PayloadStore::Disk(_) => {
                    return Err(QmfError::InvalidManifest(
                        "disk builder requires finish_disk".into(),
                    ));
                }
            },
        };
        archive.validate_internal()?;
        Ok(archive)
    }
}

/// Executable linear operator loaded directly from a validated QMF v1 archive (no dense rebuild).
///
/// Shared latent factors (`u_shared`, `vt_shared`, `shared_core`) are held via `Arc` so
/// multiple layers referencing the same latent table entry genuinely share one decoded copy in RAM.
#[derive(Debug, Clone)]
pub enum QmfOperator {
    DenseF32(DenseMatrix),
    DenseF16(Fp16Matrix),
    DenseBf16(Bf16Matrix),
    Int8Sym(Int8SymMatrix),
    Int4Sym(Int4SymGroupMatrix),
    Int4Asym(Int4AsymGroupMatrix),
    LowRankSvd(LowRankSvdMatrix),
    RmqSharedBasis {
        rows: usize,
        cols: usize,
        u_shared: Arc<QuantizedFactor>,
        core: QuantizedFactor,
        vt_shared: Arc<QuantizedFactor>,
        sparse_residual: Option<SparseResidual>,
    },
    RmqSharedCore {
        rows: usize,
        cols: usize,
        left: QuantizedFactor,
        shared_core: Arc<QuantizedFactor>,
        right: QuantizedFactor,
        sparse_residual: Option<SparseResidual>,
    },
    Int3Asym(Int3AsymGroupMatrix),
    Int2Asym(Int2AsymGroupMatrix),
    TernaryPacked(TernaryPackedMatrix),
    VectorQuant(VectorQuantMatrix),
    EchoCorrected(EchoCorrectedMatrix),
}

impl QmfOperator {
    /// Decode a single row on demand without reconstructing the full dense matrix.
    ///
    /// Used by Q-Core for compressed token embedding lookup (`O(hidden_size)` memory).
    pub fn decode_row(&self, row: usize) -> Result<Vec<f64>, MatrixError> {
        if row >= self.rows() {
            return Err(MatrixError::DimensionMismatch {
                expected: self.rows(),
                actual: row,
            });
        }
        match self {
            Self::DenseF32(m) => {
                let start = row * m.cols();
                Ok(m.as_slice()[start..start + m.cols()].to_vec())
            }
            Self::DenseF16(m) => m.decode_row(row),
            Self::DenseBf16(m) => m.decode_row(row),
            Self::Int8Sym(m) => m.decode_row(row),
            Self::Int4Sym(m) => m.decode_row(row),
            Self::Int4Asym(m) => m.decode_row(row),
            Self::Int3Asym(m) => m.decode_row(row),
            Self::Int2Asym(m) => m.decode_row(row),
            Self::TernaryPacked(m) => m.decode_row(row),
            Self::VectorQuant(m) => m.decode_row(row),
            Self::EchoCorrected(m) => m.decode_row(row),
            Self::LowRankSvd(_) | Self::RmqSharedBasis { .. } | Self::RmqSharedCore { .. } => {
                let mut basis = vec![0.0_f64; self.cols()];
                let mut out = Vec::with_capacity(self.cols());
                for c in 0..self.cols() {
                    basis[c] = 1.0;
                    let col_vec = self.matvec(&basis)?;
                    out.push(col_vec[row]);
                    basis[c] = 0.0;
                }
                Ok(out)
            }
        }
    }
}

impl LinearOperator for QmfOperator {
    fn method_name(&self) -> &'static str {
        match self {
            Self::DenseF32(_) => "QMF_DENSE_F32",
            Self::DenseF16(_) => "QMF_FP16",
            Self::DenseBf16(_) => "QMF_BF16",
            Self::Int8Sym(_) => "QMF_INT8_SYM",
            Self::Int4Sym(_) => "QMF_INT4_SYM",
            Self::Int4Asym(_) => "QMF_INT4_ASYM",
            Self::Int3Asym(m) => m.method_name(),
            Self::Int2Asym(m) => m.method_name(),
            Self::TernaryPacked(m) => m.method_name(),
            Self::VectorQuant(m) => m.method_name(),
            Self::EchoCorrected(m) => m.method_name(),
            Self::LowRankSvd(_) => "QMF_SVD",
            Self::RmqSharedBasis {
                sparse_residual: Some(_),
                ..
            } => "QMF_RMQ_SHARED_BASIS+SPARSE",
            Self::RmqSharedBasis {
                sparse_residual: None,
                ..
            } => "QMF_RMQ_SHARED_BASIS",
            Self::RmqSharedCore {
                sparse_residual: Some(_),
                ..
            } => "QMF_RMQ_SHARED_CORE+SPARSE",
            Self::RmqSharedCore {
                sparse_residual: None,
                ..
            } => "QMF_RMQ_SHARED_CORE",
        }
    }

    fn rows(&self) -> usize {
        match self {
            Self::DenseF32(m) => m.rows(),
            Self::DenseF16(m) => m.rows(),
            Self::DenseBf16(m) => m.rows(),
            Self::Int8Sym(m) => m.rows(),
            Self::Int4Sym(m) => m.rows(),
            Self::Int4Asym(m) => m.rows(),
            Self::Int3Asym(m) => m.rows(),
            Self::Int2Asym(m) => m.rows(),
            Self::TernaryPacked(m) => m.rows(),
            Self::VectorQuant(m) => m.rows(),
            Self::EchoCorrected(m) => m.rows(),
            Self::LowRankSvd(m) => m.rows(),
            Self::RmqSharedBasis { rows, .. } | Self::RmqSharedCore { rows, .. } => *rows,
        }
    }

    fn cols(&self) -> usize {
        match self {
            Self::DenseF32(m) => m.cols(),
            Self::DenseF16(m) => m.cols(),
            Self::DenseBf16(m) => m.cols(),
            Self::Int8Sym(m) => m.cols(),
            Self::Int4Sym(m) => m.cols(),
            Self::Int4Asym(m) => m.cols(),
            Self::Int3Asym(m) => m.cols(),
            Self::Int2Asym(m) => m.cols(),
            Self::TernaryPacked(m) => m.cols(),
            Self::VectorQuant(m) => m.cols(),
            Self::EchoCorrected(m) => m.cols(),
            Self::LowRankSvd(m) => m.cols(),
            Self::RmqSharedBasis { cols, .. } | Self::RmqSharedCore { cols, .. } => *cols,
        }
    }

    fn matvec(&self, vector: &[f64]) -> Result<Vec<f64>, MatrixError> {
        match self {
            Self::DenseF32(m) => m.matvec(vector),
            Self::DenseF16(m) => m.matvec(vector),
            Self::DenseBf16(m) => m.matvec(vector),
            Self::Int8Sym(m) => m.matvec(vector),
            Self::Int4Sym(m) => m.matvec(vector),
            Self::Int4Asym(m) => m.matvec(vector),
            Self::Int3Asym(m) => m.matvec(vector),
            Self::Int2Asym(m) => m.matvec(vector),
            Self::TernaryPacked(m) => m.matvec(vector),
            Self::VectorQuant(m) => m.matvec(vector),
            Self::EchoCorrected(m) => m.matvec(vector),
            Self::LowRankSvd(m) => m.matvec(vector),
            Self::RmqSharedBasis {
                u_shared,
                core,
                vt_shared,
                sparse_residual,
                ..
            } => {
                let vtx = vt_shared.matvec(vector)?;
                let c_vtx = core.matvec(&vtx)?;
                let mut y = u_shared.matvec(&c_vtx)?;
                if let Some(sp) = sparse_residual {
                    sp.add_matvec_into(vector, &mut y)?;
                }
                Ok(y)
            }
            Self::RmqSharedCore {
                left,
                shared_core,
                right,
                sparse_residual,
                ..
            } => {
                let bx = right.matvec(vector)?;
                let s_bx = shared_core.matvec(&bx)?;
                let mut y = left.matvec(&s_bx)?;
                if let Some(sp) = sparse_residual {
                    sp.add_matvec_into(vector, &mut y)?;
                }
                Ok(y)
            }
        }
    }

    fn reconstruct_dense(&self) -> Result<DenseMatrix, MatrixError> {
        match self {
            Self::DenseF32(m) => Ok(m.clone()),
            Self::DenseF16(m) => m.reconstruct_dense(),
            Self::DenseBf16(m) => m.reconstruct_dense(),
            Self::Int8Sym(m) => m.reconstruct_dense(),
            Self::Int4Sym(m) => m.reconstruct_dense(),
            Self::Int4Asym(m) => m.reconstruct_dense(),
            Self::Int3Asym(m) => m.reconstruct_dense(),
            Self::Int2Asym(m) => m.reconstruct_dense(),
            Self::TernaryPacked(m) => m.reconstruct_dense(),
            Self::VectorQuant(m) => m.reconstruct_dense(),
            Self::EchoCorrected(m) => m.reconstruct_dense(),
            Self::LowRankSvd(m) => m.reconstruct_dense(),
            Self::RmqSharedBasis {
                u_shared,
                core,
                vt_shared,
                sparse_residual,
                ..
            } => {
                let u = u_shared.reconstruct_dense()?;
                let c = core.reconstruct_dense()?;
                let vt = vt_shared.reconstruct_dense()?;
                let base = u.matmul(&c)?.matmul(&vt)?;
                if let Some(sp) = sparse_residual {
                    let mut data = base.as_slice().to_vec();
                    sp.add_dense_into(&mut data);
                    DenseMatrix::from_flat(base.rows(), base.cols(), data)
                } else {
                    Ok(base)
                }
            }
            Self::RmqSharedCore {
                left,
                shared_core,
                right,
                sparse_residual,
                ..
            } => {
                let a = left.reconstruct_dense()?;
                let s = shared_core.reconstruct_dense()?;
                let b = right.reconstruct_dense()?;
                let base = a.matmul(&s)?.matmul(&b)?;
                if let Some(sp) = sparse_residual {
                    let mut data = base.as_slice().to_vec();
                    sp.add_dense_into(&mut data);
                    DenseMatrix::from_flat(base.rows(), base.cols(), data)
                } else {
                    Ok(base)
                }
            }
        }
    }

    fn storage_breakdown(&self) -> StorageBreakdown {
        match self {
            Self::DenseF32(m) => StorageBreakdown {
                payload_bytes: m.rows() * m.cols() * 4,
                scale_metadata_bytes: 0,
                structure_bytes: 0,
                header_bytes: 16,
            },
            Self::DenseF16(m) => m.storage_breakdown(),
            Self::DenseBf16(m) => m.storage_breakdown(),
            Self::Int8Sym(m) => m.storage_breakdown(),
            Self::Int4Sym(m) => m.storage_breakdown(),
            Self::Int4Asym(m) => m.storage_breakdown(),
            Self::Int3Asym(m) => m.storage_breakdown(),
            Self::Int2Asym(m) => m.storage_breakdown(),
            Self::TernaryPacked(m) => m.storage_breakdown(),
            Self::VectorQuant(m) => m.storage_breakdown(),
            Self::EchoCorrected(m) => m.storage_breakdown(),
            Self::LowRankSvd(m) => m.storage_breakdown(),
            Self::RmqSharedBasis {
                core,
                sparse_residual,
                ..
            } => {
                let (cp, cs) = core.raw_storage_bytes();
                let sp_bytes = sparse_residual
                    .as_ref()
                    .map_or(0, SparseResidual::entry_bytes);
                StorageBreakdown {
                    payload_bytes: 0,
                    scale_metadata_bytes: cs,
                    structure_bytes: cp + sp_bytes,
                    header_bytes: 24,
                }
            }
            Self::RmqSharedCore {
                left,
                right,
                sparse_residual,
                ..
            } => {
                let (lp, ls) = left.raw_storage_bytes();
                let (rp, rs) = right.raw_storage_bytes();
                let sp_bytes = sparse_residual
                    .as_ref()
                    .map_or(0, SparseResidual::entry_bytes);
                StorageBreakdown {
                    payload_bytes: 0,
                    scale_metadata_bytes: ls + rs,
                    structure_bytes: lp + rp + sp_bytes,
                    header_bytes: 24,
                }
            }
        }
    }
}

impl QmfArchiveV1 {
    pub fn manifest(&self) -> &QmfManifest {
        &self.manifest
    }

    pub fn payload_bytes(&self) -> usize {
        self.payload.len()
    }

    /// Write deterministic QMF v1 bytes (header + JSON manifest + binary payload + CRC-32)
    /// under [`QmfBudget::default`] (up to 8 GiB).
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<usize, QmfError> {
        self.write_to_with_budget(writer, &QmfBudget::default())
    }

    /// Write deterministic QMF v1 bytes enforcing an explicit [`QmfBudget`].
    pub fn write_to_with_budget<W: Write>(
        &self,
        writer: &mut W,
        budget: &QmfBudget,
    ) -> Result<usize, QmfError> {
        self.validate_internal()?;
        let manifest_bytes = serde_json::to_vec(&self.manifest)
            .map_err(|e| QmfError::InvalidManifest(e.to_string()))?;
        if manifest_bytes.len() > budget.max_manifest_bytes {
            return Err(QmfError::PayloadTooLarge);
        }
        let total_u64 = (QMF_V1_HEADER_BYTES as u64)
            .checked_add(manifest_bytes.len() as u64)
            .and_then(|v| v.checked_add(self.payload.len() as u64))
            .and_then(|v| v.checked_add(QMF_CHECKSUM_BYTES as u64))
            .ok_or(QmfError::PayloadTooLarge)?;
        if total_u64 > budget.max_file_bytes {
            return Err(QmfError::PayloadTooLarge);
        }
        let total = usize::try_from(total_u64).map_err(|_| QmfError::PayloadTooLarge)?;

        let mut header = [0_u8; QMF_V1_HEADER_BYTES];
        header[0..4].copy_from_slice(&QMF_MAGIC);
        header[4..6].copy_from_slice(&QMF_V1_VERSION.to_le_bytes());
        header[6..8].copy_from_slice(&QMF_V1_FLAG_LITTLE_ENDIAN.to_le_bytes());
        header[8..16].copy_from_slice(&(manifest_bytes.len() as u64).to_le_bytes());
        header[16..24].copy_from_slice(&(self.payload.len() as u64).to_le_bytes());
        header[24..32].copy_from_slice(&0_u64.to_le_bytes());

        let mut checksum = Crc32::new();
        writer.write_all(&header)?;
        checksum.update(&header);
        writer.write_all(&manifest_bytes)?;
        checksum.update(&manifest_bytes);
        writer.write_all(&self.payload)?;
        checksum.update(&self.payload);
        writer.write_all(&checksum.finish().to_le_bytes())?;
        Ok(total)
    }

    /// Atomically publish a QMF v1 file via a sibling temporary file, computing CRC-32
    /// and FIPS 180-4 SHA-256 in a single streaming pass without duplicating the payload
    /// in RAM, and refusing to overwrite `source_path` if it aliases `output_path`.
    pub fn write_to_file_atomic(
        &self,
        output_path: &std::path::Path,
        source_path: Option<&std::path::Path>,
        budget: &QmfBudget,
    ) -> Result<(usize, String), QmfError> {
        self.write_to_file_atomic_with_overwrite(output_path, source_path, budget, false)
    }

    /// Replace an existing artifact only when `overwrite` is explicitly authorized.
    pub fn write_to_file_atomic_with_overwrite(
        &self,
        output_path: &std::path::Path,
        source_path: Option<&std::path::Path>,
        budget: &QmfBudget,
        overwrite: bool,
    ) -> Result<(usize, String), QmfError> {
        // 1. Protect source model files from accidental overwrite / path aliasing.
        if let Some(src) = source_path
            && paths_alias(src, output_path)
        {
            return Err(QmfError::SourceOutputAlias(format!(
                "output path {} aliases immutable source {}",
                output_path.display(),
                src.display()
            )));
        }

        self.validate_internal()?;
        let manifest_bytes = serde_json::to_vec(&self.manifest)
            .map_err(|e| QmfError::InvalidManifest(e.to_string()))?;
        if manifest_bytes.len() > budget.max_manifest_bytes {
            return Err(QmfError::PayloadTooLarge);
        }
        let total_u64 = (QMF_V1_HEADER_BYTES as u64)
            .checked_add(manifest_bytes.len() as u64)
            .and_then(|v| v.checked_add(self.payload.len() as u64))
            .and_then(|v| v.checked_add(QMF_CHECKSUM_BYTES as u64))
            .ok_or(QmfError::PayloadTooLarge)?;
        if total_u64 > budget.max_file_bytes {
            return Err(QmfError::PayloadTooLarge);
        }
        let total = usize::try_from(total_u64).map_err(|_| QmfError::PayloadTooLarge)?;

        if let Some(parent) = output_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }

        qml_core::publication::validate_destination(output_path, source_path)?;
        let mut temp = qml_core::publication::stage(output_path)?;
        let write_result = (|| -> Result<String, QmfError> {
            let mut writer = std::io::BufWriter::new(temp.as_file_mut());

            let mut header = [0_u8; QMF_V1_HEADER_BYTES];
            header[0..4].copy_from_slice(&QMF_MAGIC);
            header[4..6].copy_from_slice(&QMF_V1_VERSION.to_le_bytes());
            header[6..8].copy_from_slice(&QMF_V1_FLAG_LITTLE_ENDIAN.to_le_bytes());
            header[8..16].copy_from_slice(&(manifest_bytes.len() as u64).to_le_bytes());
            header[16..24].copy_from_slice(&(self.payload.len() as u64).to_le_bytes());
            header[24..32].copy_from_slice(&0_u64.to_le_bytes());

            let mut checksum = Crc32::new();
            let mut sha = Sha256::new();

            writer.write_all(&header)?;
            checksum.update(&header);
            sha.update(&header);

            writer.write_all(&manifest_bytes)?;
            checksum.update(&manifest_bytes);
            sha.update(&manifest_bytes);

            writer.write_all(&self.payload)?;
            checksum.update(&self.payload);
            sha.update(&self.payload);

            let crc_bytes = checksum.finish().to_le_bytes();
            writer.write_all(&crc_bytes)?;
            sha.update(&crc_bytes);

            writer.flush()?;
            writer.get_ref().sync_all()?;
            Ok(sha.finalize_hex())
        })();

        let sha_hex = write_result?;
        qml_core::publication::publish(temp, output_path, source_path, overwrite)?;
        Ok((total, sha_hex))
    }

    /// Bounded read and full integrity verification of a QMF v1 stream using [`QmfBudget::default`].
    pub fn read_from<R: Read>(reader: R) -> Result<Self, QmfError> {
        Self::read_from_with_budget(reader, &QmfBudget::default())
    }

    /// Bounded read and full integrity verification enforcing an explicit [`QmfBudget`].
    pub fn read_from_with_budget<R: Read>(reader: R, budget: &QmfBudget) -> Result<Self, QmfError> {
        let limit = budget.max_file_bytes.saturating_add(1);
        let mut bytes = Vec::new();
        reader.take(limit).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > budget.max_file_bytes {
            return Err(QmfError::PayloadTooLarge);
        }
        Self::from_bytes_with_budget(&bytes, budget)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, QmfError> {
        Self::from_bytes_with_budget(bytes, &QmfBudget::default())
    }

    pub fn from_bytes_with_budget(bytes: &[u8], budget: &QmfBudget) -> Result<Self, QmfError> {
        if bytes.len() < QMF_V1_HEADER_BYTES + QMF_CHECKSUM_BYTES {
            return Err(QmfError::LengthMismatch {
                expected: QMF_V1_HEADER_BYTES + QMF_CHECKSUM_BYTES,
                actual: bytes.len(),
            });
        }
        if bytes[0..4] != QMF_MAGIC {
            return Err(QmfError::InvalidMagic);
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != QMF_V1_VERSION {
            return Err(QmfError::UnsupportedVersion(version));
        }
        let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
        if flags != QMF_V1_FLAG_LITTLE_ENDIAN {
            return Err(QmfError::NonZeroReservedField);
        }
        let manifest_len_u64 = u64::from_le_bytes(bytes[8..16].try_into().expect("8-byte slice"));
        let payload_len_u64 = u64::from_le_bytes(bytes[16..24].try_into().expect("8-byte slice"));
        let reserved = u64::from_le_bytes(bytes[24..32].try_into().expect("8-byte slice"));
        if reserved != 0 {
            return Err(QmfError::NonZeroReservedField);
        }
        let manifest_len =
            usize::try_from(manifest_len_u64).map_err(|_| QmfError::PayloadTooLarge)?;
        if manifest_len > budget.max_manifest_bytes {
            return Err(QmfError::PayloadTooLarge);
        }
        let expected_total_u64 = (QMF_V1_HEADER_BYTES as u64)
            .checked_add(manifest_len_u64)
            .and_then(|v| v.checked_add(payload_len_u64))
            .and_then(|v| v.checked_add(QMF_CHECKSUM_BYTES as u64))
            .ok_or(QmfError::PayloadTooLarge)?;
        if expected_total_u64 > budget.max_file_bytes {
            return Err(QmfError::PayloadTooLarge);
        }
        let expected_total =
            usize::try_from(expected_total_u64).map_err(|_| QmfError::PayloadTooLarge)?;
        if bytes.len() != expected_total {
            return Err(QmfError::LengthMismatch {
                expected: expected_total,
                actual: bytes.len(),
            });
        }

        let checksum_start = bytes.len() - QMF_CHECKSUM_BYTES;
        let stored_crc =
            u32::from_le_bytes(bytes[checksum_start..].try_into().expect("4-byte crc"));
        if crc32(&bytes[..checksum_start]) != stored_crc {
            return Err(QmfError::ChecksumMismatch);
        }

        let manifest_slice = &bytes[QMF_V1_HEADER_BYTES..QMF_V1_HEADER_BYTES + manifest_len];
        let manifest: QmfManifest = serde_json::from_slice(manifest_slice)
            .map_err(|e| QmfError::InvalidManifest(e.to_string()))?;
        let payload = bytes[QMF_V1_HEADER_BYTES + manifest_len..checksum_start].to_vec();

        let archive = Self { manifest, payload };
        archive.validate_internal()?;
        Ok(archive)
    }

    fn blob_slice(&self, blob: BlobRef) -> Result<&[u8], QmfError> {
        let start = usize::try_from(blob.offset).map_err(|_| QmfError::PayloadTooLarge)?;
        let len = usize::try_from(blob.len).map_err(|_| QmfError::PayloadTooLarge)?;
        let end = start.checked_add(len).ok_or(QmfError::PayloadTooLarge)?;
        if end > self.payload.len() {
            return Err(QmfError::InvalidManifest(format!(
                "blob range {start}..{end} exceeds payload length {}",
                self.payload.len()
            )));
        }
        let slice = &self.payload[start..end];
        if crc32(slice) != blob.crc32 {
            return Err(QmfError::ChecksumMismatch);
        }
        Ok(slice)
    }

    fn validate_internal(&self) -> Result<(), QmfError> {
        let mut latent_ids = BTreeSet::new();
        for latent in &self.manifest.shared_latents {
            if !latent_ids.insert(latent.id.as_str()) {
                return Err(QmfError::InvalidManifest(format!(
                    "duplicate shared latent id {:?}",
                    latent.id
                )));
            }
            self.validate_latent_encoding(&latent.encoding)?;
        }
        let mut tensor_names = BTreeSet::new();
        for tensor in &self.manifest.tensors {
            if !tensor_names.insert(tensor.name.as_str()) {
                return Err(QmfError::InvalidManifest(format!(
                    "duplicate tensor name {:?}",
                    tensor.name
                )));
            }
            match &tensor.representation {
                QmfRepresentation::DenseF32 { data }
                | QmfRepresentation::DenseF16 { data }
                | QmfRepresentation::DenseBf16 { data } => {
                    let _ = self.blob_slice(*data)?;
                }
                QmfRepresentation::Int8SymGroup { scales, codes, .. } => {
                    let _ = self.blob_slice(*scales)?;
                    let _ = self.blob_slice(*codes)?;
                }
                QmfRepresentation::Int4SymGroup {
                    scales, nibbles, ..
                } => {
                    let _ = self.blob_slice(*scales)?;
                    let _ = self.blob_slice(*nibbles)?;
                }
                QmfRepresentation::Int4AsymGroup {
                    scales,
                    mins,
                    nibbles,
                    ..
                } => {
                    let _ = self.blob_slice(*scales)?;
                    let _ = self.blob_slice(*mins)?;
                    let _ = self.blob_slice(*nibbles)?;
                }
                QmfRepresentation::LowRankSvd { left, right, .. } => {
                    self.validate_latent_encoding(left)?;
                    self.validate_latent_encoding(right)?;
                }
                QmfRepresentation::RmqSharedBasis {
                    u_latent_id,
                    vt_latent_id,
                    core,
                    sparse_residual,
                    ..
                } => {
                    if !latent_ids.contains(u_latent_id.as_str())
                        || !latent_ids.contains(vt_latent_id.as_str())
                    {
                        return Err(QmfError::InvalidManifest(format!(
                            "tensor {:?} references missing shared latent id",
                            tensor.name
                        )));
                    }
                    self.validate_latent_encoding(core)?;
                    if let Some(sp) = sparse_residual {
                        let _ = self.blob_slice(sp.indices)?;
                        let _ = self.blob_slice(sp.values_f16)?;
                    }
                }
                QmfRepresentation::RmqSharedCore {
                    s_latent_id,
                    left,
                    right,
                    sparse_residual,
                    ..
                } => {
                    if !latent_ids.contains(s_latent_id.as_str()) {
                        return Err(QmfError::InvalidManifest(format!(
                            "tensor {:?} references missing shared core id {s_latent_id:?}",
                            tensor.name
                        )));
                    }
                    self.validate_latent_encoding(left)?;
                    self.validate_latent_encoding(right)?;
                    if let Some(sp) = sparse_residual {
                        let _ = self.blob_slice(sp.indices)?;
                        let _ = self.blob_slice(sp.values_f16)?;
                    }
                }
                rep => self.validate_resonance_rep(rep)?,
            }
        }
        Ok(())
    }

    fn validate_resonance_rep(&self, rep: &QmfRepresentation) -> Result<(), QmfError> {
        match rep {
            QmfRepresentation::Int3AsymGroup {
                scales,
                mins,
                packed,
                ..
            }
            | QmfRepresentation::Int2AsymGroup {
                scales,
                mins,
                packed,
                ..
            } => {
                let _ = self.blob_slice(*scales)?;
                let _ = self.blob_slice(*mins)?;
                let _ = self.blob_slice(*packed)?;
            }
            QmfRepresentation::TernaryPacked { scales, packed, .. } => {
                let _ = self.blob_slice(*scales)?;
                let _ = self.blob_slice(*packed)?;
            }
            QmfRepresentation::VectorQuant {
                codebooks_f16,
                scales,
                codes,
                ..
            } => {
                let _ = self.blob_slice(*codebooks_f16)?;
                let _ = self.blob_slice(*scales)?;
                let _ = self.blob_slice(*codes)?;
            }
            QmfRepresentation::EchoCorrected {
                base,
                low_rank_left,
                low_rank_right,
                sparse_residual,
                ..
            } => {
                self.validate_resonance_rep(base)?;
                if let Some(l) = low_rank_left {
                    self.validate_latent_encoding(l)?;
                }
                if let Some(r) = low_rank_right {
                    self.validate_latent_encoding(r)?;
                }
                if let Some(sp) = sparse_residual {
                    let _ = self.blob_slice(sp.indices)?;
                    let _ = self.blob_slice(sp.values_f16)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn validate_latent_encoding(&self, enc: &SharedLatentEncoding) -> Result<(), QmfError> {
        match enc {
            SharedLatentEncoding::Fp32 { data } | SharedLatentEncoding::Fp16 { data } => {
                let _ = self.blob_slice(*data)?;
            }
            SharedLatentEncoding::Int8Sym { scales, codes, .. } => {
                let _ = self.blob_slice(*scales)?;
                let _ = self.blob_slice(*codes)?;
            }
        }
        Ok(())
    }

    fn decode_factor(
        &self,
        rows: usize,
        cols: usize,
        enc: &SharedLatentEncoding,
    ) -> Result<QuantizedFactor, QmfError> {
        match enc {
            SharedLatentEncoding::Fp32 { data } => {
                let raw = self.blob_slice(*data)?;
                let vals = decode_f32_vec(raw)?;
                Ok(QuantizedFactor::Fp32(DenseMatrix::from_flat(
                    rows, cols, vals,
                )?))
            }
            SharedLatentEncoding::Fp16 { data } => {
                let raw = self.blob_slice(*data)?;
                let bits = decode_u16_vec(raw)?;
                Ok(QuantizedFactor::Fp16(Fp16Matrix::from_raw(
                    rows, cols, bits,
                )?))
            }
            SharedLatentEncoding::Int8Sym {
                group_size,
                scales,
                codes,
            } => {
                let scales_vec = decode_u16_vec(self.blob_slice(*scales)?)?;
                let codes_vec: Vec<i8> =
                    self.blob_slice(*codes)?.iter().map(|&b| b as i8).collect();
                Ok(QuantizedFactor::Int8Sym(Int8SymMatrix::from_parts(
                    rows,
                    cols,
                    *group_size,
                    scales_vec,
                    codes_vec,
                )?))
            }
        }
    }

    /// Decode a 1D vector tensor (e.g., LayerNorm weights or bias) to `Vec<f64>`.
    pub fn load_vector_f64(&self, name: &str) -> Result<Vec<f64>, QmfError> {
        let entry = self
            .manifest
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| QmfError::InvalidManifest(format!("tensor {name:?} not found")))?;
        match &entry.representation {
            QmfRepresentation::DenseF32 { data } => decode_f32_vec(self.blob_slice(*data)?),
            QmfRepresentation::DenseF16 { data } => Ok(decode_u16_vec(self.blob_slice(*data)?)?
                .into_iter()
                .map(|b| f64::from(qml_model::f16_to_f32(b)))
                .collect()),
            QmfRepresentation::DenseBf16 { data } => Ok(decode_u16_vec(self.blob_slice(*data)?)?
                .into_iter()
                .map(|b| f64::from(qml_model::bf16_to_f32(b)))
                .collect()),
            _ => Err(QmfError::InvalidManifest(format!(
                "tensor {name:?} is not a dense vector representation"
            ))),
        }
    }

    /// Load a 2D tensor directly as an executable [`QmfOperator`] without dense reconstruction.
    pub fn load_operator(&self, name: &str) -> Result<QmfOperator, QmfError> {
        let mut cache = QmfLatentCache::new();
        self.load_operator_cached(name, &mut cache)
    }

    /// Load a 2D tensor as an executable [`QmfOperator`], deduplicating shared latent structures
    /// across layers via [`QmfLatentCache`] (`Arc<QuantizedFactor>`).
    pub fn load_operator_cached(
        &self,
        name: &str,
        latent_cache: &mut QmfLatentCache,
    ) -> Result<QmfOperator, QmfError> {
        let entry = self
            .manifest
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| QmfError::InvalidManifest(format!("tensor {name:?} not found")))?;
        if entry.shape.len() != 2 {
            return Err(QmfError::InvalidDimensions);
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];

        let mut get_or_decode_latent = |id: &str| -> Result<Arc<QuantizedFactor>, QmfError> {
            if let Some(existing) = latent_cache.get(id) {
                return Ok(Arc::clone(existing));
            }
            let latent_entry = self
                .manifest
                .shared_latents
                .iter()
                .find(|l| l.id == id)
                .ok_or_else(|| QmfError::InvalidManifest(format!("missing latent {id:?}")))?;
            let decoded = Arc::new(self.decode_factor(
                latent_entry.rows,
                latent_entry.cols,
                &latent_entry.encoding,
            )?);
            latent_cache.insert(id.to_owned(), Arc::clone(&decoded));
            Ok(decoded)
        };

        match &entry.representation {
            QmfRepresentation::DenseF32 { data } => {
                let vals = decode_f32_vec(self.blob_slice(*data)?)?;
                Ok(QmfOperator::DenseF32(DenseMatrix::from_flat(
                    rows, cols, vals,
                )?))
            }
            QmfRepresentation::DenseF16 { data } => {
                let bits = decode_u16_vec(self.blob_slice(*data)?)?;
                Ok(QmfOperator::DenseF16(Fp16Matrix::from_raw(
                    rows, cols, bits,
                )?))
            }
            QmfRepresentation::DenseBf16 { data } => {
                let bits = decode_u16_vec(self.blob_slice(*data)?)?;
                Ok(QmfOperator::DenseBf16(Bf16Matrix::from_raw(
                    rows, cols, bits,
                )?))
            }
            QmfRepresentation::Int8SymGroup {
                group_size,
                scales,
                codes,
            } => {
                let s = decode_u16_vec(self.blob_slice(*scales)?)?;
                let c: Vec<i8> = self.blob_slice(*codes)?.iter().map(|&b| b as i8).collect();
                Ok(QmfOperator::Int8Sym(Int8SymMatrix::from_parts(
                    rows,
                    cols,
                    *group_size,
                    s,
                    c,
                )?))
            }
            QmfRepresentation::Int4SymGroup {
                group_size,
                scales,
                nibbles,
            } => {
                let s = decode_u16_vec(self.blob_slice(*scales)?)?;
                let n = self.blob_slice(*nibbles)?.to_vec();
                Ok(QmfOperator::Int4Sym(Int4SymGroupMatrix::from_parts(
                    rows,
                    cols,
                    *group_size,
                    s,
                    n,
                )?))
            }
            QmfRepresentation::Int4AsymGroup {
                group_size,
                scales,
                mins,
                nibbles,
            } => {
                let s = decode_u16_vec(self.blob_slice(*scales)?)?;
                let m = decode_u16_vec(self.blob_slice(*mins)?)?;
                let n = self.blob_slice(*nibbles)?.to_vec();
                Ok(QmfOperator::Int4Asym(Int4AsymGroupMatrix::from_parts(
                    rows,
                    cols,
                    *group_size,
                    s,
                    m,
                    n,
                )?))
            }
            QmfRepresentation::LowRankSvd { rank, left, right } => {
                let l = self.decode_factor(rows, *rank, left)?;
                let r = self.decode_factor(*rank, cols, right)?;
                Ok(QmfOperator::LowRankSvd(LowRankSvdMatrix::from_factors(
                    l, r,
                )?))
            }
            QmfRepresentation::RmqSharedBasis {
                u_latent_id,
                vt_latent_id,
                left_rank,
                right_rank,
                core,
                sparse_residual,
            } => {
                let u_shared = get_or_decode_latent(u_latent_id)?;
                let vt_shared = get_or_decode_latent(vt_latent_id)?;
                let core_factor = self.decode_factor(*left_rank, *right_rank, core)?;
                let sparse = match sparse_residual {
                    Some(desc) => {
                        let idx = decode_u32_vec(self.blob_slice(desc.indices)?)?;
                        let vals = decode_u16_vec(self.blob_slice(desc.values_f16)?)?;
                        Some(SparseResidual::from_parts(rows, cols, idx, vals)?)
                    }
                    None => None,
                };
                Ok(QmfOperator::RmqSharedBasis {
                    rows,
                    cols,
                    u_shared,
                    core: core_factor,
                    vt_shared,
                    sparse_residual: sparse,
                })
            }
            QmfRepresentation::RmqSharedCore {
                s_latent_id,
                rank,
                left,
                right,
                sparse_residual,
            } => {
                let shared_core = get_or_decode_latent(s_latent_id)?;
                let left_factor = self.decode_factor(rows, *rank, left)?;
                let right_factor = self.decode_factor(*rank, cols, right)?;
                let sparse = match sparse_residual {
                    Some(desc) => {
                        let idx = decode_u32_vec(self.blob_slice(desc.indices)?)?;
                        let vals = decode_u16_vec(self.blob_slice(desc.values_f16)?)?;
                        Some(SparseResidual::from_parts(rows, cols, idx, vals)?)
                    }
                    None => None,
                };
                Ok(QmfOperator::RmqSharedCore {
                    rows,
                    cols,
                    left: left_factor,
                    shared_core,
                    right: right_factor,
                    sparse_residual: sparse,
                })
            }
            QmfRepresentation::Int3AsymGroup {
                group_size,
                hadamard_block,
                hadamard_seed,
                scales,
                mins,
                packed,
            } => Ok(QmfOperator::Int3Asym(Int3AsymGroupMatrix::from_raw(
                rows,
                cols,
                *group_size,
                HadamardSpec::new(*hadamard_block, *hadamard_seed),
                decode_u16_vec(self.blob_slice(*scales)?)?,
                decode_u16_vec(self.blob_slice(*mins)?)?,
                self.blob_slice(*packed)?.to_vec(),
            )?)),
            QmfRepresentation::Int2AsymGroup {
                group_size,
                hadamard_block,
                hadamard_seed,
                scales,
                mins,
                packed,
            } => Ok(QmfOperator::Int2Asym(Int2AsymGroupMatrix::from_raw(
                rows,
                cols,
                *group_size,
                HadamardSpec::new(*hadamard_block, *hadamard_seed),
                decode_u16_vec(self.blob_slice(*scales)?)?,
                decode_u16_vec(self.blob_slice(*mins)?)?,
                self.blob_slice(*packed)?.to_vec(),
            )?)),
            QmfRepresentation::TernaryPacked {
                group_size,
                hadamard_block,
                hadamard_seed,
                scales,
                packed,
            } => Ok(QmfOperator::TernaryPacked(TernaryPackedMatrix::from_raw(
                rows,
                cols,
                *group_size,
                HadamardSpec::new(*hadamard_block, *hadamard_seed),
                decode_u16_vec(self.blob_slice(*scales)?)?,
                self.blob_slice(*packed)?.to_vec(),
            )?)),
            QmfRepresentation::VectorQuant {
                subvec_dim,
                codebook_size,
                num_stages,
                group_size,
                hadamard_block,
                hadamard_seed,
                codebooks_f16,
                scales,
                codes,
            } => Ok(QmfOperator::VectorQuant(VectorQuantMatrix::from_raw(
                rows,
                cols,
                *subvec_dim,
                *codebook_size,
                *num_stages,
                *group_size,
                HadamardSpec::new(*hadamard_block, *hadamard_seed),
                decode_u16_vec(self.blob_slice(*codebooks_f16)?)?,
                decode_u16_vec(self.blob_slice(*scales)?)?,
                self.blob_slice(*codes)?.to_vec(),
            )?)),
            QmfRepresentation::EchoCorrected {
                base,
                low_rank_r,
                low_rank_left,
                low_rank_right,
                sparse_residual,
            } => {
                let base_op = self.decode_echo_base(rows, cols, base)?;
                let lr = match (low_rank_left, low_rank_right) {
                    (Some(l), Some(r)) if *low_rank_r > 0 => {
                        let lf = self.decode_factor(rows, *low_rank_r, l)?;
                        let rf = self.decode_factor(*low_rank_r, cols, r)?;
                        Some(LowRankSvdMatrix::from_factors(lf, rf)?)
                    }
                    _ => None,
                };
                let sparse = match sparse_residual {
                    Some(desc) => Some(SparseResidual::from_parts(
                        rows,
                        cols,
                        decode_u32_vec(self.blob_slice(desc.indices)?)?,
                        decode_u16_vec(self.blob_slice(desc.values_f16)?)?,
                    )?),
                    None => None,
                };
                Ok(QmfOperator::EchoCorrected(EchoCorrectedMatrix::from_parts(
                    base_op, lr, sparse,
                )))
            }
        }
    }

    fn decode_echo_base(
        &self,
        rows: usize,
        cols: usize,
        rep: &QmfRepresentation,
    ) -> Result<EchoBaseOperator, QmfError> {
        match rep {
            QmfRepresentation::Int3AsymGroup {
                group_size,
                hadamard_block,
                hadamard_seed,
                scales,
                mins,
                packed,
            } => Ok(EchoBaseOperator::Int3Asym(Int3AsymGroupMatrix::from_raw(
                rows,
                cols,
                *group_size,
                HadamardSpec::new(*hadamard_block, *hadamard_seed),
                decode_u16_vec(self.blob_slice(*scales)?)?,
                decode_u16_vec(self.blob_slice(*mins)?)?,
                self.blob_slice(*packed)?.to_vec(),
            )?)),
            QmfRepresentation::Int2AsymGroup {
                group_size,
                hadamard_block,
                hadamard_seed,
                scales,
                mins,
                packed,
            } => Ok(EchoBaseOperator::Int2Asym(Int2AsymGroupMatrix::from_raw(
                rows,
                cols,
                *group_size,
                HadamardSpec::new(*hadamard_block, *hadamard_seed),
                decode_u16_vec(self.blob_slice(*scales)?)?,
                decode_u16_vec(self.blob_slice(*mins)?)?,
                self.blob_slice(*packed)?.to_vec(),
            )?)),
            QmfRepresentation::TernaryPacked {
                group_size,
                hadamard_block,
                hadamard_seed,
                scales,
                packed,
            } => Ok(EchoBaseOperator::Ternary(TernaryPackedMatrix::from_raw(
                rows,
                cols,
                *group_size,
                HadamardSpec::new(*hadamard_block, *hadamard_seed),
                decode_u16_vec(self.blob_slice(*scales)?)?,
                self.blob_slice(*packed)?.to_vec(),
            )?)),
            QmfRepresentation::VectorQuant {
                subvec_dim,
                codebook_size,
                num_stages,
                group_size,
                hadamard_block,
                hadamard_seed,
                codebooks_f16,
                scales,
                codes,
            } => Ok(EchoBaseOperator::VectorQuant(VectorQuantMatrix::from_raw(
                rows,
                cols,
                *subvec_dim,
                *codebook_size,
                *num_stages,
                *group_size,
                HadamardSpec::new(*hadamard_block, *hadamard_seed),
                decode_u16_vec(self.blob_slice(*codebooks_f16)?)?,
                decode_u16_vec(self.blob_slice(*scales)?)?,
                self.blob_slice(*codes)?.to_vec(),
            )?)),
            _ => Err(QmfError::InvalidManifest(
                "unsupported base representation inside EchoCorrected".to_owned(),
            )),
        }
    }
}

fn paths_alias(a: &std::path::Path, b: &std::path::Path) -> bool {
    if a == b {
        return true;
    }
    if let (Ok(ca), Ok(cb)) = (std::fs::canonicalize(a), std::fs::canonicalize(b))
        && ca == cb
    {
        return true;
    }
    // Check parent canonicalization + case-insensitive filename on Windows when `b` does not exist yet.
    if let (Some(fa), Some(fb)) = (a.file_name(), b.file_name()) {
        let same_name = if cfg!(windows) {
            fa.to_string_lossy()
                .eq_ignore_ascii_case(&fb.to_string_lossy())
        } else {
            fa == fb
        };
        if same_name
            && let (Some(pa), Some(pb)) = (a.parent(), b.parent())
            && let (Ok(cpa), Ok(cpb)) = (std::fs::canonicalize(pa), std::fs::canonicalize(pb))
            && cpa == cpb
        {
            return true;
        }
    }
    false
}

/// File-backed, out-of-core QMF v1 reader that holds only the JSON manifest in memory
/// and lazily seeks/decodes individual tensors on demand without loading the full payload.
pub struct QmfFileReader<R: Read + Seek> {
    reader: R,
    manifest: QmfManifest,
    manifest_bytes: u64,
    payload_bytes: u64,
    payload_base_offset: u64,
    file_bytes: u64,
}

impl<R: Read + Seek> QmfFileReader<R> {
    /// Open a seekable QMF v1 stream using only header + manifest memory.
    pub fn open(mut reader: R, budget: &QmfBudget) -> Result<Self, QmfError> {
        let (manifest, file_bytes, payload_bytes) = inspect_qmf_v1_header(&mut reader, budget)?;
        let payload_base_offset = file_bytes
            .checked_sub(QMF_CHECKSUM_BYTES as u64)
            .and_then(|v| v.checked_sub(payload_bytes))
            .ok_or(QmfError::PayloadTooLarge)?;
        let manifest_bytes = payload_base_offset
            .checked_sub(QMF_V1_HEADER_BYTES as u64)
            .ok_or(QmfError::PayloadTooLarge)?;
        Ok(Self {
            reader,
            manifest,
            manifest_bytes,
            payload_bytes,
            payload_base_offset,
            file_bytes,
        })
    }

    pub fn manifest(&self) -> &QmfManifest {
        &self.manifest
    }

    pub fn manifest_bytes(&self) -> u64 {
        self.manifest_bytes
    }

    pub fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    /// Stream the entire file in 64 KiB chunks to verify the trailing whole-file CRC-32
    /// and return its exact FIPS 180-4 SHA-256 hex digest in O(1) RAM.
    pub fn verify_integrity_streaming(&mut self) -> Result<String, QmfError> {
        self.verify_integrity_with_cancel(|| Ok(()))
    }

    /// Cooperative verification with the existing reader budget.
    pub fn verify_integrity_with_cancel(
        &mut self,
        mut check: impl FnMut() -> Result<(), QmfError>,
    ) -> Result<String, QmfError> {
        self.reader.seek(SeekFrom::Start(0))?;
        let data_len = self
            .file_bytes
            .checked_sub(QMF_CHECKSUM_BYTES as u64)
            .ok_or(QmfError::ChecksumMismatch)?;
        let mut crc = Crc32::new();
        let mut sha = Sha256::new();
        let mut buf = [0_u8; 64 * 1024];
        let mut remaining = data_len;
        while remaining > 0 {
            check()?;
            let to_read = remaining.min(buf.len() as u64) as usize;
            self.reader.read_exact(&mut buf[..to_read])?;
            crc.update(&buf[..to_read]);
            sha.update(&buf[..to_read]);
            remaining -= to_read as u64;
        }
        let mut trailer = [0_u8; QMF_CHECKSUM_BYTES];
        self.reader.read_exact(&mut trailer)?;
        sha.update(&trailer);
        let expected_crc = u32::from_le_bytes(trailer);
        if crc.finish() != expected_crc {
            return Err(QmfError::ChecksumMismatch);
        }
        Ok(sha.finalize_hex())
    }

    fn read_blob(&mut self, blob: BlobRef) -> Result<Vec<u8>, QmfError> {
        let end = blob
            .offset
            .checked_add(blob.len)
            .ok_or(QmfError::PayloadTooLarge)?;
        if end > self.payload_bytes {
            return Err(QmfError::InvalidManifest(
                "blob range exceeds QMF payload bounds".to_owned(),
            ));
        }
        let len = usize::try_from(blob.len).map_err(|_| QmfError::PayloadTooLarge)?;
        let abs_offset = self
            .payload_base_offset
            .checked_add(blob.offset)
            .ok_or(QmfError::PayloadTooLarge)?;
        self.reader.seek(SeekFrom::Start(abs_offset))?;
        let mut buf = vec![0_u8; len];
        self.reader.read_exact(&mut buf)?;
        if crc32(&buf) != blob.crc32 {
            return Err(QmfError::ChecksumMismatch);
        }
        Ok(buf)
    }

    fn decode_factor(
        &mut self,
        rows: usize,
        cols: usize,
        enc: &SharedLatentEncoding,
    ) -> Result<QuantizedFactor, QmfError> {
        match enc {
            SharedLatentEncoding::Fp32 { data } => {
                let raw = self.read_blob(*data)?;
                let vals = decode_f32_vec(&raw)?;
                Ok(QuantizedFactor::Fp32(DenseMatrix::from_flat(
                    rows, cols, vals,
                )?))
            }
            SharedLatentEncoding::Fp16 { data } => {
                let raw = self.read_blob(*data)?;
                let bits = decode_u16_vec(&raw)?;
                Ok(QuantizedFactor::Fp16(Fp16Matrix::from_raw(
                    rows, cols, bits,
                )?))
            }
            SharedLatentEncoding::Int8Sym {
                group_size,
                scales,
                codes,
            } => {
                let scales_vec = decode_u16_vec(&self.read_blob(*scales)?)?;
                let codes_vec: Vec<i8> = self
                    .read_blob(*codes)?
                    .into_iter()
                    .map(|b| b as i8)
                    .collect();
                Ok(QuantizedFactor::Int8Sym(Int8SymMatrix::from_parts(
                    rows,
                    cols,
                    *group_size,
                    scales_vec,
                    codes_vec,
                )?))
            }
        }
    }

    /// Lazily load a 1D vector tensor directly from disk without buffering other tensors.
    pub fn load_vector_f64(&mut self, name: &str) -> Result<Vec<f64>, QmfError> {
        let entry = self
            .manifest
            .tensors
            .iter()
            .find(|t| t.name == name)
            .cloned()
            .ok_or_else(|| QmfError::InvalidManifest(format!("tensor {name:?} not found")))?;
        match &entry.representation {
            QmfRepresentation::DenseF32 { data } => decode_f32_vec(&self.read_blob(*data)?),
            QmfRepresentation::DenseF16 { data } => Ok(decode_u16_vec(&self.read_blob(*data)?)?
                .into_iter()
                .map(|b| f64::from(qml_model::f16_to_f32(b)))
                .collect()),
            QmfRepresentation::DenseBf16 { data } => Ok(decode_u16_vec(&self.read_blob(*data)?)?
                .into_iter()
                .map(|b| f64::from(qml_model::bf16_to_f32(b)))
                .collect()),
            _ => Err(QmfError::InvalidManifest(format!(
                "tensor {name:?} is not a dense vector representation"
            ))),
        }
    }

    /// Lazily load a single 2D tensor as an executable [`QmfOperator`] directly from the file,
    /// deduplicating shared latents via [`QmfLatentCache`].
    pub fn load_operator_cached(
        &mut self,
        name: &str,
        latent_cache: &mut QmfLatentCache,
    ) -> Result<QmfOperator, QmfError> {
        let entry = self
            .manifest
            .tensors
            .iter()
            .find(|t| t.name == name)
            .cloned()
            .ok_or_else(|| QmfError::InvalidManifest(format!("tensor {name:?} not found")))?;
        if entry.shape.len() != 2 {
            return Err(QmfError::InvalidDimensions);
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];

        match &entry.representation {
            QmfRepresentation::DenseF32 { data } => {
                let vals = decode_f32_vec(&self.read_blob(*data)?)?;
                Ok(QmfOperator::DenseF32(DenseMatrix::from_flat(
                    rows, cols, vals,
                )?))
            }
            QmfRepresentation::DenseF16 { data } => {
                let bits = decode_u16_vec(&self.read_blob(*data)?)?;
                Ok(QmfOperator::DenseF16(Fp16Matrix::from_raw(
                    rows, cols, bits,
                )?))
            }
            QmfRepresentation::DenseBf16 { data } => {
                let bits = decode_u16_vec(&self.read_blob(*data)?)?;
                Ok(QmfOperator::DenseBf16(Bf16Matrix::from_raw(
                    rows, cols, bits,
                )?))
            }
            QmfRepresentation::Int8SymGroup {
                group_size,
                scales,
                codes,
            } => {
                let scales_vec = decode_u16_vec(&self.read_blob(*scales)?)?;
                let codes_vec: Vec<i8> = self
                    .read_blob(*codes)?
                    .into_iter()
                    .map(|b| b as i8)
                    .collect();
                Ok(QmfOperator::Int8Sym(Int8SymMatrix::from_parts(
                    rows,
                    cols,
                    *group_size,
                    scales_vec,
                    codes_vec,
                )?))
            }
            QmfRepresentation::Int4SymGroup {
                group_size,
                scales,
                nibbles,
            } => {
                let scales_vec = decode_u16_vec(&self.read_blob(*scales)?)?;
                let nibbles_vec = self.read_blob(*nibbles)?;
                Ok(QmfOperator::Int4Sym(Int4SymGroupMatrix::from_parts(
                    rows,
                    cols,
                    *group_size,
                    scales_vec,
                    nibbles_vec,
                )?))
            }
            QmfRepresentation::Int4AsymGroup {
                group_size,
                scales,
                mins,
                nibbles,
            } => {
                let scales_vec = decode_u16_vec(&self.read_blob(*scales)?)?;
                let mins_vec = decode_u16_vec(&self.read_blob(*mins)?)?;
                let nibbles_vec = self.read_blob(*nibbles)?;
                Ok(QmfOperator::Int4Asym(Int4AsymGroupMatrix::from_parts(
                    rows,
                    cols,
                    *group_size,
                    scales_vec,
                    mins_vec,
                    nibbles_vec,
                )?))
            }
            QmfRepresentation::LowRankSvd { rank, left, right } => {
                let l = self.decode_factor(rows, *rank, left)?;
                let r = self.decode_factor(*rank, cols, right)?;
                Ok(QmfOperator::LowRankSvd(LowRankSvdMatrix::from_factors(
                    l, r,
                )?))
            }
            QmfRepresentation::RmqSharedBasis {
                u_latent_id,
                vt_latent_id,
                left_rank,
                right_rank,
                core,
                sparse_residual,
            } => {
                let u_shared = self.load_or_cache_latent(u_latent_id, latent_cache)?;
                let vt_shared = self.load_or_cache_latent(vt_latent_id, latent_cache)?;
                let core_factor = self.decode_factor(*left_rank, *right_rank, core)?;
                let sparse = match sparse_residual {
                    Some(desc) => {
                        let idx = decode_u32_vec(&self.read_blob(desc.indices)?)?;
                        let vals = decode_u16_vec(&self.read_blob(desc.values_f16)?)?;
                        Some(SparseResidual::from_parts(rows, cols, idx, vals)?)
                    }
                    None => None,
                };
                Ok(QmfOperator::RmqSharedBasis {
                    rows,
                    cols,
                    u_shared,
                    core: core_factor,
                    vt_shared,
                    sparse_residual: sparse,
                })
            }
            QmfRepresentation::RmqSharedCore {
                s_latent_id,
                rank,
                left,
                right,
                sparse_residual,
            } => {
                let shared_core = self.load_or_cache_latent(s_latent_id, latent_cache)?;
                let left_factor = self.decode_factor(rows, *rank, left)?;
                let right_factor = self.decode_factor(*rank, cols, right)?;
                let sparse = match sparse_residual {
                    Some(desc) => {
                        let idx = decode_u32_vec(&self.read_blob(desc.indices)?)?;
                        let vals = decode_u16_vec(&self.read_blob(desc.values_f16)?)?;
                        Some(SparseResidual::from_parts(rows, cols, idx, vals)?)
                    }
                    None => None,
                };
                Ok(QmfOperator::RmqSharedCore {
                    rows,
                    cols,
                    left: left_factor,
                    shared_core,
                    right: right_factor,
                    sparse_residual: sparse,
                })
            }
            QmfRepresentation::Int3AsymGroup {
                group_size,
                hadamard_block,
                hadamard_seed,
                scales,
                mins,
                packed,
            } => Ok(QmfOperator::Int3Asym(Int3AsymGroupMatrix::from_raw(
                rows,
                cols,
                *group_size,
                HadamardSpec::new(*hadamard_block, *hadamard_seed),
                decode_u16_vec(&self.read_blob(*scales)?)?,
                decode_u16_vec(&self.read_blob(*mins)?)?,
                self.read_blob(*packed)?,
            )?)),
            QmfRepresentation::Int2AsymGroup {
                group_size,
                hadamard_block,
                hadamard_seed,
                scales,
                mins,
                packed,
            } => Ok(QmfOperator::Int2Asym(Int2AsymGroupMatrix::from_raw(
                rows,
                cols,
                *group_size,
                HadamardSpec::new(*hadamard_block, *hadamard_seed),
                decode_u16_vec(&self.read_blob(*scales)?)?,
                decode_u16_vec(&self.read_blob(*mins)?)?,
                self.read_blob(*packed)?,
            )?)),
            QmfRepresentation::TernaryPacked {
                group_size,
                hadamard_block,
                hadamard_seed,
                scales,
                packed,
            } => Ok(QmfOperator::TernaryPacked(TernaryPackedMatrix::from_raw(
                rows,
                cols,
                *group_size,
                HadamardSpec::new(*hadamard_block, *hadamard_seed),
                decode_u16_vec(&self.read_blob(*scales)?)?,
                self.read_blob(*packed)?,
            )?)),
            QmfRepresentation::VectorQuant {
                subvec_dim,
                codebook_size,
                num_stages,
                group_size,
                hadamard_block,
                hadamard_seed,
                codebooks_f16,
                scales,
                codes,
            } => Ok(QmfOperator::VectorQuant(VectorQuantMatrix::from_raw(
                rows,
                cols,
                *subvec_dim,
                *codebook_size,
                *num_stages,
                *group_size,
                HadamardSpec::new(*hadamard_block, *hadamard_seed),
                decode_u16_vec(&self.read_blob(*codebooks_f16)?)?,
                decode_u16_vec(&self.read_blob(*scales)?)?,
                self.read_blob(*codes)?,
            )?)),
            QmfRepresentation::EchoCorrected {
                base,
                low_rank_r,
                low_rank_left,
                low_rank_right,
                sparse_residual,
            } => {
                let base_op = self.read_echo_base(rows, cols, base)?;
                let lr = match (low_rank_left, low_rank_right) {
                    (Some(l), Some(r)) if *low_rank_r > 0 => {
                        let lf = self.decode_factor(rows, *low_rank_r, l)?;
                        let rf = self.decode_factor(*low_rank_r, cols, r)?;
                        Some(LowRankSvdMatrix::from_factors(lf, rf)?)
                    }
                    _ => None,
                };
                let sparse = match sparse_residual {
                    Some(desc) => Some(SparseResidual::from_parts(
                        rows,
                        cols,
                        decode_u32_vec(&self.read_blob(desc.indices)?)?,
                        decode_u16_vec(&self.read_blob(desc.values_f16)?)?,
                    )?),
                    None => None,
                };
                Ok(QmfOperator::EchoCorrected(EchoCorrectedMatrix::from_parts(
                    base_op, lr, sparse,
                )))
            }
        }
    }

    fn read_echo_base(
        &mut self,
        rows: usize,
        cols: usize,
        rep: &QmfRepresentation,
    ) -> Result<EchoBaseOperator, QmfError> {
        match rep {
            QmfRepresentation::Int3AsymGroup {
                group_size,
                hadamard_block,
                hadamard_seed,
                scales,
                mins,
                packed,
            } => Ok(EchoBaseOperator::Int3Asym(Int3AsymGroupMatrix::from_raw(
                rows,
                cols,
                *group_size,
                HadamardSpec::new(*hadamard_block, *hadamard_seed),
                decode_u16_vec(&self.read_blob(*scales)?)?,
                decode_u16_vec(&self.read_blob(*mins)?)?,
                self.read_blob(*packed)?,
            )?)),
            QmfRepresentation::Int2AsymGroup {
                group_size,
                hadamard_block,
                hadamard_seed,
                scales,
                mins,
                packed,
            } => Ok(EchoBaseOperator::Int2Asym(Int2AsymGroupMatrix::from_raw(
                rows,
                cols,
                *group_size,
                HadamardSpec::new(*hadamard_block, *hadamard_seed),
                decode_u16_vec(&self.read_blob(*scales)?)?,
                decode_u16_vec(&self.read_blob(*mins)?)?,
                self.read_blob(*packed)?,
            )?)),
            QmfRepresentation::TernaryPacked {
                group_size,
                hadamard_block,
                hadamard_seed,
                scales,
                packed,
            } => Ok(EchoBaseOperator::Ternary(TernaryPackedMatrix::from_raw(
                rows,
                cols,
                *group_size,
                HadamardSpec::new(*hadamard_block, *hadamard_seed),
                decode_u16_vec(&self.read_blob(*scales)?)?,
                self.read_blob(*packed)?,
            )?)),
            QmfRepresentation::VectorQuant {
                subvec_dim,
                codebook_size,
                num_stages,
                group_size,
                hadamard_block,
                hadamard_seed,
                codebooks_f16,
                scales,
                codes,
            } => Ok(EchoBaseOperator::VectorQuant(VectorQuantMatrix::from_raw(
                rows,
                cols,
                *subvec_dim,
                *codebook_size,
                *num_stages,
                *group_size,
                HadamardSpec::new(*hadamard_block, *hadamard_seed),
                decode_u16_vec(&self.read_blob(*codebooks_f16)?)?,
                decode_u16_vec(&self.read_blob(*scales)?)?,
                self.read_blob(*codes)?,
            )?)),
            _ => Err(QmfError::InvalidManifest(
                "unsupported base representation inside EchoCorrected".to_owned(),
            )),
        }
    }

    fn load_or_cache_latent(
        &mut self,
        id: &str,
        latent_cache: &mut QmfLatentCache,
    ) -> Result<Arc<QuantizedFactor>, QmfError> {
        if let Some(existing) = latent_cache.get(id) {
            return Ok(Arc::clone(existing));
        }
        let latent_entry = self
            .manifest
            .shared_latents
            .iter()
            .find(|l| l.id == id)
            .cloned()
            .ok_or_else(|| QmfError::InvalidManifest(format!("missing latent {id:?}")))?;
        let decoded = Arc::new(self.decode_factor(
            latent_entry.rows,
            latent_entry.cols,
            &latent_entry.encoding,
        )?);
        latent_cache.insert(id.to_owned(), Arc::clone(&decoded));
        Ok(decoded)
    }
}

/// Streaming, bounded header-only inspector for multi-GB QMF v1 files.
pub fn inspect_qmf_v1_header<R: Read + Seek>(
    mut reader: R,
    budget: &QmfBudget,
) -> Result<(QmfManifest, u64, u64), QmfError> {
    let file_bytes = reader.seek(SeekFrom::End(0))?;
    if file_bytes < (QMF_V1_HEADER_BYTES + QMF_CHECKSUM_BYTES) as u64 {
        return Err(QmfError::LengthMismatch {
            expected: QMF_V1_HEADER_BYTES + QMF_CHECKSUM_BYTES,
            actual: file_bytes as usize,
        });
    }
    if file_bytes > budget.max_file_bytes {
        return Err(QmfError::PayloadTooLarge);
    }
    reader.seek(SeekFrom::Start(0))?;
    let mut header = [0_u8; QMF_V1_HEADER_BYTES];
    reader.read_exact(&mut header)?;
    if header[0..4] != QMF_MAGIC {
        return Err(QmfError::InvalidMagic);
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    if version != QMF_V1_VERSION {
        return Err(QmfError::UnsupportedVersion(version));
    }
    let flags = u16::from_le_bytes([header[6], header[7]]);
    if flags != QMF_V1_FLAG_LITTLE_ENDIAN {
        return Err(QmfError::NonZeroReservedField);
    }
    let manifest_len_u64 = u64::from_le_bytes(header[8..16].try_into().expect("8-byte slice"));
    let payload_len_u64 = u64::from_le_bytes(header[16..24].try_into().expect("8-byte slice"));
    let reserved = u64::from_le_bytes(header[24..32].try_into().expect("8-byte slice"));
    if reserved != 0 {
        return Err(QmfError::NonZeroReservedField);
    }
    let manifest_len = usize::try_from(manifest_len_u64).map_err(|_| QmfError::PayloadTooLarge)?;
    if manifest_len > budget.max_manifest_bytes {
        return Err(QmfError::PayloadTooLarge);
    }
    let expected = (QMF_V1_HEADER_BYTES as u64)
        .checked_add(manifest_len_u64)
        .and_then(|v| v.checked_add(payload_len_u64))
        .and_then(|v| v.checked_add(QMF_CHECKSUM_BYTES as u64))
        .ok_or(QmfError::PayloadTooLarge)?;
    if expected != file_bytes {
        return Err(QmfError::LengthMismatch {
            expected: expected as usize,
            actual: file_bytes as usize,
        });
    }
    let mut manifest_bytes = vec![0_u8; manifest_len];
    reader.read_exact(&mut manifest_bytes)?;
    let manifest: QmfManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| QmfError::InvalidManifest(e.to_string()))?;
    Ok((manifest, file_bytes, payload_len_u64))
}

fn decode_u16_vec(raw: &[u8]) -> Result<Vec<u16>, QmfError> {
    if !raw.len().is_multiple_of(2) {
        return Err(QmfError::InvalidManifest(
            "u16 blob length is not a multiple of 2".to_owned(),
        ));
    }
    Ok(raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&c| u16::from_le_bytes(c))
        .collect())
}

fn decode_u32_vec(raw: &[u8]) -> Result<Vec<u32>, QmfError> {
    if !raw.len().is_multiple_of(4) {
        return Err(QmfError::InvalidManifest(
            "u32 blob length is not a multiple of 4".to_owned(),
        ));
    }
    Ok(raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| u32::from_le_bytes(c))
        .collect())
}

fn decode_f32_vec(raw: &[u8]) -> Result<Vec<f64>, QmfError> {
    if !raw.len().is_multiple_of(4) {
        return Err(QmfError::InvalidManifest(
            "f32 blob length is not a multiple of 4".to_owned(),
        ));
    }
    let mut out = Vec::with_capacity(raw.len() / 4);
    for (idx, &c) in raw.as_chunks::<4>().0.iter().enumerate() {
        let v = f64::from(f32::from_le_bytes(c));
        if !v.is_finite() {
            return Err(QmfError::Matrix(MatrixError::NonFiniteValue { index: idx }));
        }
        out.push(v);
    }
    Ok(out)
}

/// Detect the QMF version (`0` or `1`) from the first 6 bytes.
pub fn peek_qmf_version(bytes: &[u8]) -> Result<u16, QmfError> {
    if bytes.len() < 6 {
        return Err(QmfError::LengthMismatch {
            expected: 6,
            actual: bytes.len(),
        });
    }
    if bytes[0..4] != QMF_MAGIC {
        return Err(QmfError::InvalidMagic);
    }
    Ok(u16::from_le_bytes([bytes[4], bytes[5]]))
}

/// Helper to expose `FactorPrecision` in QMF v1 consumers.
pub fn factor_precision_label(p: FactorPrecision) -> &'static str {
    match p {
        FactorPrecision::Fp32 => "fp32",
        FactorPrecision::Fp16 => "fp16",
        FactorPrecision::Int8Sym => "int8_sym",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qml_core::DeterministicRng;
    use qml_quant::{RmqResidualKind, RmqSharedBasisConfig};

    #[test]
    fn qmf_v1_round_trips_heterogeneous_representations_and_shared_rmq_cluster() {
        let mut rng = DeterministicRng::new(2026);
        let m0 =
            DenseMatrix::from_flat(16, 16, (0..256).map(|_| rng.next_normal()).collect()).unwrap();
        let m1 =
            DenseMatrix::from_flat(16, 16, (0..256).map(|_| rng.next_normal()).collect()).unwrap();

        let cluster = RmqSharedBasisCluster::fit(
            &[&m0, &m1],
            RmqSharedBasisConfig {
                left_rank: 4,
                right_rank: 4,
                hooi_iters: 2,
                basis_precision: FactorPrecision::Int8Sym,
                core_precision: FactorPrecision::Fp16,
                residual_kind: RmqResidualKind::SparseTopFraction { fraction: 0.02 },
                seed: 7,
            },
        )
        .unwrap();
        let q4 = Int4SymGroupMatrix::quantize(&m0, 32).unwrap();
        let svd = LowRankSvdMatrix::from_matrix(&m1, 4, FactorPrecision::Fp16, 9).unwrap();

        let mut builder = QmfV1Builder::new(
            "synthetic-test",
            "safetensors",
            "synthetic-generated",
            "mini_transformer",
            "rmq+mixed",
            42,
        );
        builder
            .add_rmq_shared_basis_cluster(
                "attn_q",
                &["l0.q".to_owned(), "l1.q".to_owned()],
                &cluster,
            )
            .unwrap();
        builder.add_int4_sym("l0.up", &q4).unwrap();
        builder.add_low_rank_svd("l1.up", &svd).unwrap();
        builder
            .add_dense_f32("norm.weight", vec![16], &[1.0_f32; 16])
            .unwrap();

        let archive = builder.build().unwrap();
        let mut first = Vec::new();
        let mut second = Vec::new();
        archive.write_to(&mut first).unwrap();
        archive.write_to(&mut second).unwrap();
        assert_eq!(first, second, "QMF v1 serialization must be deterministic");

        let reloaded = QmfArchiveV1::from_bytes(&first).unwrap();
        assert_eq!(reloaded.manifest().shared_latents.len(), 2);
        assert_eq!(reloaded.manifest().tensors.len(), 5);

        let x: Vec<f64> = (0..16).map(|i| (i as f64 - 8.0) / 8.0).collect();
        let op_rmq = reloaded.load_operator("l0.q").unwrap();
        let expected_y = cluster.layer(0).unwrap().matvec(&x).unwrap();
        let actual_y = op_rmq.matvec(&x).unwrap();
        for (a, b) in expected_y.iter().zip(actual_y) {
            assert!((a - b).abs() < 1e-12);
        }

        let op_q4 = reloaded.load_operator("l0.up").unwrap();
        assert_eq!(op_q4.matvec(&x).unwrap(), q4.matvec(&x).unwrap());

        let norm = reloaded.load_vector_f64("norm.weight").unwrap();
        assert_eq!(norm, vec![1.0; 16]);

        // Flip a byte in payload -> must fail checksum!
        let mut corrupt = first.clone();
        let mid = corrupt.len() - 10;
        corrupt[mid] ^= 0x80;
        assert!(matches!(
            QmfArchiveV1::from_bytes(&corrupt),
            Err(QmfError::ChecksumMismatch)
        ));

        // Verify Arc deduplication via QmfLatentCache across both RMQ layers!
        let mut cache = QmfLatentCache::new();
        let op0 = reloaded.load_operator_cached("l0.q", &mut cache).unwrap();
        let op1 = reloaded.load_operator_cached("l1.q", &mut cache).unwrap();
        assert_eq!(cache.len(), 2);
        if let (
            QmfOperator::RmqSharedBasis {
                u_shared: u0,
                vt_shared: vt0,
                ..
            },
            QmfOperator::RmqSharedBasis {
                u_shared: u1,
                vt_shared: vt1,
                ..
            },
        ) = (&op0, &op1)
        {
            assert!(Arc::ptr_eq(u0, u1), "u_shared must be deduplicated via Arc");
            assert!(
                Arc::ptr_eq(vt0, vt1),
                "vt_shared must be deduplicated via Arc"
            );
        } else {
            panic!("expected RmqSharedBasis operators");
        }

        // Verify compressed single-row decoding matches dense reconstruction!
        let row3 = op_q4.decode_row(3).unwrap();
        let dense_q4 = op_q4.reconstruct_dense().unwrap();
        for (c, &val) in row3.iter().enumerate() {
            assert!((val - dense_q4.get(3, c).unwrap()).abs() < 1e-12);
        }
    }

    #[test]
    fn qmf_v1_round_trips_shared_core_rmq_and_header_inspection() {
        let mut rng = DeterministicRng::new(9001);
        let m0 =
            DenseMatrix::from_flat(16, 16, (0..256).map(|_| rng.next_normal()).collect()).unwrap();
        let m1 =
            DenseMatrix::from_flat(16, 16, (0..256).map(|_| rng.next_normal()).collect()).unwrap();
        let cluster = RmqSharedCoreCluster::fit(
            &[&m0, &m1],
            4,
            FactorPrecision::Int8Sym,
            RmqResidualKind::SparseTopFraction { fraction: 0.02 },
            19,
        )
        .unwrap();

        let mut builder = QmfV1Builder::new(
            "shared-core-test",
            "safetensors",
            "synthetic",
            "test",
            "rmq_shared_core",
            19,
        );
        builder
            .add_rmq_shared_core_cluster(
                "attn_v",
                &["l0.v".to_owned(), "l1.v".to_owned()],
                &cluster,
            )
            .unwrap();
        let archive = builder.build().unwrap();
        let mut buf = Vec::new();
        archive.write_to(&mut buf).unwrap();

        // Streaming header-only inspection without loading payload.
        let (hdr_manifest, file_len, payload_len) =
            inspect_qmf_v1_header(std::io::Cursor::new(&buf), &QmfBudget::default()).unwrap();
        assert_eq!(file_len as usize, buf.len());
        assert_eq!(payload_len as usize, archive.payload_bytes());
        assert_eq!(hdr_manifest.shared_latents.len(), 1);

        let reloaded = QmfArchiveV1::from_bytes(&buf).unwrap();
        let mut cache = QmfLatentCache::new();
        let op0 = reloaded.load_operator_cached("l0.v", &mut cache).unwrap();
        let op1 = reloaded.load_operator_cached("l1.v", &mut cache).unwrap();
        if let (
            QmfOperator::RmqSharedCore {
                shared_core: s0, ..
            },
            QmfOperator::RmqSharedCore {
                shared_core: s1, ..
            },
        ) = (&op0, &op1)
        {
            assert!(
                Arc::ptr_eq(s0, s1),
                "shared_core must be deduplicated via Arc"
            );
        } else {
            panic!("expected RmqSharedCore operators");
        }
        let x: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
        let expected = cluster.layer(0).unwrap().matvec(&x).unwrap();
        let actual = op0.matvec(&x).unwrap();
        for (a, b) in expected.iter().zip(actual) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn file_backed_qmf_reader_streams_and_verifies_without_full_payload_buffer() {
        let m = DenseMatrix::from_flat(8, 8, (0..64).map(|i| (i as f64 * 0.17).sin()).collect())
            .unwrap();
        let q = Int8SymMatrix::quantize(&m, 4).unwrap();
        let mut builder =
            QmfV1Builder::new("streaming-test", "MIT", "synthetic", "test", "int8", 7);
        builder.add_int8_sym("layer.0.w", &q).unwrap();
        builder
            .add_dense_f32("norm.w", vec![4], &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        let archive = builder.build().unwrap();

        let dir = std::env::temp_dir().join(format!(
            "qml_format_atomic_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let out_path = dir.join("model.qmf");
        let src_path = dir.join("source.safetensors");
        std::fs::write(&src_path, b"dummy").unwrap();

        // Source-output alias must be rejected.
        let alias_err = archive
            .write_to_file_atomic(&src_path, Some(&src_path), &QmfBudget::default())
            .unwrap_err();
        assert!(matches!(alias_err, QmfError::SourceOutputAlias(_)));

        // Atomic publish succeeds and returns exact SHA-256.
        let (published_bytes, published_sha) = archive
            .write_to_file_atomic(&out_path, Some(&src_path), &QmfBudget::default())
            .unwrap();
        assert!(published_bytes > 0);
        let original = std::fs::read(&out_path).unwrap();
        assert!(
            archive
                .write_to_file_atomic(&out_path, Some(&src_path), &QmfBudget::default())
                .is_err()
        );
        assert_eq!(std::fs::read(&out_path).unwrap(), original);
        assert!(
            archive
                .write_to_file_atomic_with_overwrite(
                    &out_path,
                    Some(&src_path),
                    &QmfBudget {
                        max_manifest_bytes: 0,
                        max_file_bytes: 0
                    },
                    true
                )
                .is_err()
        );
        assert_eq!(std::fs::read(&out_path).unwrap(), original);

        // File-backed reader loads operators and verifies CRC-32 + SHA-256 in streaming mode.
        let f = std::fs::File::open(&out_path).unwrap();
        let mut reader = QmfFileReader::open(f, &QmfBudget::default()).unwrap();
        let verified_sha = reader.verify_integrity_streaming().unwrap();
        assert_eq!(verified_sha, published_sha);

        let norm = reader.load_vector_f64("norm.w").unwrap();
        assert_eq!(norm, vec![1.0, 2.0, 3.0, 4.0]);

        let mut cache = QmfLatentCache::new();
        let op = reader
            .load_operator_cached("layer.0.w", &mut cache)
            .unwrap();
        let x: Vec<f64> = (0..8).map(|i| i as f64 * 0.25).collect();
        let y_expected = q.matvec(&x).unwrap();
        let y_actual = op.matvec(&x).unwrap();
        for (a, b) in y_expected.iter().zip(y_actual) {
            assert!((a - b).abs() < 1e-12);
        }

        // Flip a byte inside the binary payload (before the 4-byte CRC trailer)
        // and confirm streaming integrity check catches silent payload corruption.
        let mut bytes = std::fs::read(&out_path).unwrap();
        let corrupt_idx = bytes.len() - 8;
        bytes[corrupt_idx] ^= 0xFF;
        std::fs::write(&out_path, &bytes).unwrap();
        let f_bad = std::fs::File::open(&out_path).unwrap();
        let mut bad_reader = QmfFileReader::open(f_bad, &QmfBudget::default()).unwrap();
        assert!(matches!(
            bad_reader.verify_integrity_streaming(),
            Err(QmfError::ChecksumMismatch)
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
