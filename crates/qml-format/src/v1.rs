//! Experimental QMF v1 generalized neural representation container.
//!
//! Supports multiple tensors, shared latent dictionaries (for RMQ), dense
//! FP32/FP16/BF16, groupwise INT8/INT4 (`Q4_0`/`Q4_1`), truncated SVD factors,
//! RMQ shared-basis & shared-core layers, sparse top-k outlier residuals,
//! model architecture/license provenance, and bounded verified parsing.
//!
//! **Status:** Experimental format; not compatibility-stable. CRC-32 checks
//! detect accidental corruption only and do not provide cryptographic authentication.

use crate::{Crc32, MAX_QMF_FILE_BYTES, QMF_CHECKSUM_BYTES, QMF_MAGIC, QmfError, crc32};
use qml_core::{DenseMatrix, MatrixError};
use qml_quant::baselines::{QuantizedFactor, SparseResidual};
use qml_quant::{
    Bf16Matrix, FactorPrecision, Fp16Matrix, Int4AsymGroupMatrix, Int4SymGroupMatrix,
    Int8SymMatrix, LinearOperator, LowRankSvdMatrix, RmqSharedBasisCluster, StorageBreakdown,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};

pub const QMF_V1_VERSION: u16 = 1;
pub const QMF_V1_HEADER_BYTES: usize = 32;
pub const QMF_V1_MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
pub const QMF_V1_FLAG_LITTLE_ENDIAN: u16 = 0x0001;

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
    payload: Vec<u8>,
    latent_ids: BTreeSet<String>,
    tensor_names: BTreeSet<String>,
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
            payload: Vec::new(),
            latent_ids: BTreeSet::new(),
            tensor_names: BTreeSet::new(),
        }
    }

    pub fn insert_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.manifest
            .extra_metadata
            .insert(key.into(), value.into());
    }

    fn append_blob(&mut self, bytes: &[u8]) -> BlobRef {
        let offset = self.payload.len() as u64;
        self.payload.extend_from_slice(bytes);
        BlobRef {
            offset,
            len: bytes.len() as u64,
            crc32: crc32(bytes),
        }
    }

    fn encode_factor(&mut self, factor: &QuantizedFactor) -> SharedLatentEncoding {
        match factor {
            QuantizedFactor::Fp32(mat) => {
                let mut raw = Vec::with_capacity(mat.as_slice().len() * 4);
                for &v in mat.as_slice() {
                    raw.extend_from_slice(&(v as f32).to_le_bytes());
                }
                SharedLatentEncoding::Fp32 {
                    data: self.append_blob(&raw),
                }
            }
            QuantizedFactor::Fp16(mat) => {
                let mut raw = Vec::with_capacity(mat.raw_bits().len() * 2);
                for &bits in mat.raw_bits() {
                    raw.extend_from_slice(&bits.to_le_bytes());
                }
                SharedLatentEncoding::Fp16 {
                    data: self.append_blob(&raw),
                }
            }
            QuantizedFactor::Int8Sym(mat) => {
                let mut scales_raw = Vec::with_capacity(mat.scales_f16().len() * 2);
                for &s in mat.scales_f16() {
                    scales_raw.extend_from_slice(&s.to_le_bytes());
                }
                let codes_raw: Vec<u8> = mat.codes().iter().map(|&c| c as u8).collect();
                let scales = self.append_blob(&scales_raw);
                let codes = self.append_blob(&codes_raw);
                SharedLatentEncoding::Int8Sym {
                    group_size: mat.group_size(),
                    scales,
                    codes,
                }
            }
        }
    }

    fn encode_sparse(&mut self, sp: &SparseResidual) -> Option<SparseResidualDesc> {
        if sp.nnz() == 0 {
            return None;
        }
        let mut idx_raw = Vec::with_capacity(sp.nnz() * 4);
        for &i in sp.indices() {
            idx_raw.extend_from_slice(&i.to_le_bytes());
        }
        let mut val_raw = Vec::with_capacity(sp.nnz() * 2);
        for &v in sp.values_f16() {
            val_raw.extend_from_slice(&v.to_le_bytes());
        }
        let indices = self.append_blob(&idx_raw);
        let values_f16 = self.append_blob(&val_raw);
        Some(SparseResidualDesc {
            nnz: sp.nnz(),
            indices,
            values_f16,
        })
    }

    fn check_unique_tensor(&mut self, name: &str) -> Result<(), QmfError> {
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
        let mut raw = Vec::with_capacity(values.len() * 4);
        for &v in values {
            if !v.is_finite() {
                return Err(QmfError::Matrix(MatrixError::NonFiniteValue { index: 0 }));
            }
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let data = self.append_blob(&raw);
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
        let mut raw = Vec::with_capacity(mat.raw_bits().len() * 2);
        for &b in mat.raw_bits() {
            raw.extend_from_slice(&b.to_le_bytes());
        }
        let data = self.append_blob(&raw);
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
        let mut raw = Vec::with_capacity(mat.raw_bits().len() * 2);
        for &b in mat.raw_bits() {
            raw.extend_from_slice(&b.to_le_bytes());
        }
        let data = self.append_blob(&raw);
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
        let mut scales_raw = Vec::with_capacity(mat.scales_f16().len() * 2);
        for &s in mat.scales_f16() {
            scales_raw.extend_from_slice(&s.to_le_bytes());
        }
        let codes_raw: Vec<u8> = mat.codes().iter().map(|&c| c as u8).collect();
        let scales = self.append_blob(&scales_raw);
        let codes = self.append_blob(&codes_raw);
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
        let mut scales_raw = Vec::with_capacity(mat.scales_f16().len() * 2);
        for &s in mat.scales_f16() {
            scales_raw.extend_from_slice(&s.to_le_bytes());
        }
        let scales = self.append_blob(&scales_raw);
        let nibbles = self.append_blob(mat.packed_nibbles());
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
        let mut scales_raw = Vec::with_capacity(mat.scales_f16().len() * 2);
        for &s in mat.scales_f16() {
            scales_raw.extend_from_slice(&s.to_le_bytes());
        }
        let mut mins_raw = Vec::with_capacity(mat.mins_f16().len() * 2);
        for &m in mat.mins_f16() {
            mins_raw.extend_from_slice(&m.to_le_bytes());
        }
        let scales = self.append_blob(&scales_raw);
        let mins = self.append_blob(&mins_raw);
        let nibbles = self.append_blob(mat.packed_nibbles());
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
        let left = self.encode_factor(mat.left());
        let right = self.encode_factor(mat.right());
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

        let u_enc = self.encode_factor(&cluster.u_shared);
        let vt_enc = self.encode_factor(&cluster.vt_shared);
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
            let core_enc = self.encode_factor(&cluster.cores[idx]);
            let sparse_residual = match &cluster.residuals[idx] {
                qml_quant::rmq::RmqResidual::None => None,
                qml_quant::rmq::RmqResidual::Sparse(sp) => self.encode_sparse(sp),
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

    pub fn build(mut self) -> Result<QmfArchiveV1, QmfError> {
        self.manifest.shared_latents.sort_by(|a, b| a.id.cmp(&b.id));
        self.manifest.tensors.sort_by(|a, b| a.name.cmp(&b.name));
        let archive = QmfArchiveV1 {
            manifest: self.manifest,
            payload: self.payload,
        };
        archive.validate_internal()?;
        Ok(archive)
    }
}

/// Executable linear operator loaded directly from a validated QMF v1 archive (no dense rebuild).
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
        u_shared: QuantizedFactor,
        core: QuantizedFactor,
        vt_shared: QuantizedFactor,
        sparse_residual: Option<SparseResidual>,
    },
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
            Self::LowRankSvd(_) => "QMF_SVD",
            Self::RmqSharedBasis {
                sparse_residual: Some(_),
                ..
            } => "QMF_RMQ_SHARED_BASIS+SPARSE",
            Self::RmqSharedBasis {
                sparse_residual: None,
                ..
            } => "QMF_RMQ_SHARED_BASIS",
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
            Self::LowRankSvd(m) => m.rows(),
            Self::RmqSharedBasis { rows, .. } => *rows,
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
            Self::LowRankSvd(m) => m.cols(),
            Self::RmqSharedBasis { cols, .. } => *cols,
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

    /// Write deterministic QMF v1 bytes (header + JSON manifest + binary payload + CRC-32).
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<usize, QmfError> {
        self.validate_internal()?;
        let manifest_bytes = serde_json::to_vec(&self.manifest)
            .map_err(|e| QmfError::InvalidManifest(e.to_string()))?;
        if manifest_bytes.len() > QMF_V1_MAX_MANIFEST_BYTES {
            return Err(QmfError::PayloadTooLarge);
        }
        let total = QMF_V1_HEADER_BYTES
            .checked_add(manifest_bytes.len())
            .and_then(|v| v.checked_add(self.payload.len()))
            .and_then(|v| v.checked_add(QMF_CHECKSUM_BYTES))
            .ok_or(QmfError::PayloadTooLarge)?;
        if total > MAX_QMF_FILE_BYTES {
            return Err(QmfError::PayloadTooLarge);
        }

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

    /// Bounded read and full integrity verification of a QMF v1 stream.
    pub fn read_from<R: Read>(reader: R) -> Result<Self, QmfError> {
        let limit = (MAX_QMF_FILE_BYTES as u64) + 1;
        let mut bytes = Vec::new();
        reader.take(limit).read_to_end(&mut bytes)?;
        if bytes.len() > MAX_QMF_FILE_BYTES {
            return Err(QmfError::PayloadTooLarge);
        }
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, QmfError> {
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
        if manifest_len > QMF_V1_MAX_MANIFEST_BYTES {
            return Err(QmfError::PayloadTooLarge);
        }
        let payload_len =
            usize::try_from(payload_len_u64).map_err(|_| QmfError::PayloadTooLarge)?;
        let expected_total = QMF_V1_HEADER_BYTES
            .checked_add(manifest_len)
            .and_then(|v| v.checked_add(payload_len))
            .and_then(|v| v.checked_add(QMF_CHECKSUM_BYTES))
            .ok_or(QmfError::PayloadTooLarge)?;
        if expected_total > MAX_QMF_FILE_BYTES {
            return Err(QmfError::PayloadTooLarge);
        }
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
            }
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
                let u_entry = self
                    .manifest
                    .shared_latents
                    .iter()
                    .find(|l| l.id == *u_latent_id)
                    .ok_or_else(|| {
                        QmfError::InvalidManifest(format!("missing latent {u_latent_id:?}"))
                    })?;
                let vt_entry = self
                    .manifest
                    .shared_latents
                    .iter()
                    .find(|l| l.id == *vt_latent_id)
                    .ok_or_else(|| {
                        QmfError::InvalidManifest(format!("missing latent {vt_latent_id:?}"))
                    })?;
                let u_shared = self.decode_factor(u_entry.rows, u_entry.cols, &u_entry.encoding)?;
                let vt_shared =
                    self.decode_factor(vt_entry.rows, vt_entry.cols, &vt_entry.encoding)?;
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
        }
    }
}

fn decode_u16_vec(raw: &[u8]) -> Result<Vec<u16>, QmfError> {
    if !raw.len().is_multiple_of(2) {
        return Err(QmfError::InvalidManifest(
            "u16 blob length is not a multiple of 2".to_owned(),
        ));
    }
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn decode_u32_vec(raw: &[u8]) -> Result<Vec<u32>, QmfError> {
    if !raw.len().is_multiple_of(4) {
        return Err(QmfError::InvalidManifest(
            "u32 blob length is not a multiple of 4".to_owned(),
        ));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn decode_f32_vec(raw: &[u8]) -> Result<Vec<f64>, QmfError> {
    if !raw.len().is_multiple_of(4) {
        return Err(QmfError::InvalidManifest(
            "f32 blob length is not a multiple of 4".to_owned(),
        ));
    }
    let mut out = Vec::with_capacity(raw.len() / 4);
    for (idx, c) in raw.chunks_exact(4).enumerate() {
        let v = f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
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
    }
}
