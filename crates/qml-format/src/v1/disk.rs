//! Disk-backed QMF v1 payload staging and independently configured construction limits.
use super::*;
use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};
use tempfile::NamedTempFile;

pub const COPY_BUFFER_BYTES: usize = 64 * 1024;

/// Builder-owned working memory is separate from borrowed operators and quantizer workspace.
#[derive(Debug, Clone)]
pub struct ArtifactLimits {
    pub max_artifact_bytes: u64,
    pub max_resident_bytes: u64,
    pub max_temporary_bytes: u64,
    pub max_tensor_bytes: u64,
    pub max_manifest_bytes: usize,
}
impl Default for ArtifactLimits {
    fn default() -> Self {
        Self {
            max_artifact_bytes: DEFAULT_QMF_V1_MAX_FILE_BYTES,
            max_resident_bytes: 256 * 1024 * 1024,
            max_temporary_bytes: 16 * 1024 * 1024 * 1024,
            max_tensor_bytes: 8 * 1024 * 1024 * 1024,
            max_manifest_bytes: QMF_V1_MAX_MANIFEST_BYTES,
        }
    }
}
impl ArtifactLimits {
    pub fn reader_budget(&self) -> QmfBudget {
        QmfBudget {
            max_manifest_bytes: self.max_manifest_bytes,
            max_file_bytes: self.max_artifact_bytes,
        }
    }
    pub fn check(&self, resource: &'static str, required: u64, limit: u64) -> Result<(), QmfError> {
        if required > limit {
            return Err(QmfError::ResourceLimit {
                resource,
                required,
                limit,
            });
        }
        Ok(())
    }
    pub fn total_bytes(&self, manifest: u64, payload: u64) -> Result<u64, QmfError> {
        let total = manifest
            .checked_add(payload)
            .and_then(|n| n.checked_add((QMF_V1_HEADER_BYTES + QMF_CHECKSUM_BYTES) as u64))
            .ok_or(QmfError::PayloadTooLarge)?;
        self.check("artifact file", total, self.max_artifact_bytes)?;
        Ok(total)
    }
}
#[derive(Clone)]
pub struct DiskBuildOptions {
    pub limits: ArtifactLimits,
    pub scratch_directory: PathBuf,
    pub cancel: Arc<AtomicBool>,
}
impl Default for DiskBuildOptions {
    fn default() -> Self {
        Self {
            limits: ArtifactLimits::default(),
            scratch_directory: std::env::temp_dir(),
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }
}
#[derive(Debug, Clone, Default)]
pub struct BuilderMemoryStats {
    pub payload_bytes: u64,
    pub resident_payload_bytes: usize,
    pub peak_encoding_buffer_bytes: usize,
    pub manifest_accounted_bytes: u64,
}
pub(super) struct DiskPayload {
    pub file: NamedTempFile,
    pub options: DiskBuildOptions,
    pub bytes: u64,
    pub stats: BuilderMemoryStats,
}
pub(super) enum PayloadStore {
    Memory(Vec<u8>),
    Disk(Box<DiskPayload>),
}
impl PayloadStore {
    pub fn len(&self) -> u64 {
        match self {
            Self::Memory(v) => v.len() as u64,
            Self::Disk(d) => d.bytes,
        }
    }
    pub fn append(&mut self, bytes: &[u8]) -> Result<(), QmfError> {
        match self {
            Self::Memory(v) => {
                v.try_reserve(bytes.len())
                    .map_err(|_| QmfError::PayloadTooLarge)?;
                v.extend_from_slice(bytes);
            }
            Self::Disk(d) => {
                cancelled(&d.options)?;
                let next = d
                    .bytes
                    .checked_add(bytes.len() as u64)
                    .ok_or(QmfError::PayloadTooLarge)?;
                d.options.limits.check(
                    "scratch payload",
                    next,
                    d.options.limits.max_temporary_bytes,
                )?;
                d.options.limits.total_bytes(0, next)?;
                if d.bytes == 0 || d.bytes / (64 * 1024 * 1024) != next / (64 * 1024 * 1024) {
                    disk_preflight(
                        d.file.path().parent().ok_or(QmfError::InvalidDimensions)?,
                        bytes.len() as u64,
                    )?;
                }
                d.file.write_all(bytes)?;
                d.bytes = next;
                d.stats.payload_bytes = next;
            }
        }
        Ok(())
    }
}
pub fn cancelled(options: &DiskBuildOptions) -> Result<(), QmfError> {
    if options.cancel.load(Ordering::Relaxed) {
        return Err(QmfError::Cancelled);
    }
    Ok(())
}
/// Advisory check only: concurrent writers can still exhaust a volume; writes remain fallible.
pub fn disk_preflight(directory: &Path, required: u64) -> Result<(), QmfError> {
    let directory = std::fs::canonicalize(directory)?;
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let disk = disks
        .iter()
        .filter_map(|disk| {
            std::fs::canonicalize(disk.mount_point())
                .ok()
                .filter(|mount| directory.starts_with(mount))
                .map(|mount| (mount.components().count(), disk))
        })
        .max_by_key(|(depth, _)| *depth)
        .map(|(_, disk)| disk)
        .ok_or_else(|| {
            QmfError::Io(std::io::Error::other(
                "cannot determine available temporary disk space",
            ))
        })?;
    if required > disk.available_space() {
        return Err(QmfError::ResourceLimit {
            resource: "available disk space",
            required,
            limit: disk.available_space(),
        });
    }
    Ok(())
}

/// Counts serialized manifest bytes without allocating another manifest-sized buffer.
pub(super) struct ManifestCounter {
    pub bytes: u64,
    pub limit: u64,
}
impl Write for ManifestCounter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let next = self
            .bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| std::io::Error::other("manifest size overflow"))?;
        if next > self.limit {
            return Err(std::io::Error::other("manifest limit exhausted"));
        }
        self.bytes = next;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub struct DiskArtifact {
    manifest: QmfManifest,
    payload: Box<DiskPayload>,
}
impl DiskArtifact {
    pub(super) fn new(manifest: QmfManifest, payload: Box<DiskPayload>) -> Self {
        Self { manifest, payload }
    }
    pub fn manifest(&self) -> &QmfManifest {
        &self.manifest
    }
    pub fn memory_stats(&self) -> &BuilderMemoryStats {
        &self.payload.stats
    }
    pub fn serialized_bytes(&self) -> Result<u64, QmfError> {
        let mut count = ManifestCounter {
            bytes: 0,
            limit: self.payload.options.limits.max_manifest_bytes as u64,
        };
        serde_json::to_writer(&mut count, &self.manifest)
            .map_err(|e| QmfError::InvalidManifest(e.to_string()))?;
        self.payload
            .options
            .limits
            .total_bytes(count.bytes, self.payload.bytes)
    }
    /// Verifies the complete staged artifact and every blob before rechecking source protection.
    pub fn publish(
        mut self,
        output: &Path,
        sources: &[PathBuf],
        overwrite: bool,
    ) -> Result<(usize, String), QmfError> {
        cancelled(&self.payload.options)?;
        for source in sources {
            qml_core::publication::validate_destination(output, Some(source))?;
        }
        let total = self.serialized_bytes()?;
        let total_usize = usize::try_from(total).map_err(|_| QmfError::PayloadTooLarge)?;
        let temporary = self
            .payload
            .bytes
            .checked_add(total)
            .ok_or(QmfError::PayloadTooLarge)?;
        self.payload.options.limits.check(
            "combined payload and staged artifact",
            temporary,
            self.payload.options.limits.max_temporary_bytes,
        )?;
        let mut stage = qml_core::publication::stage(output)?;
        disk_preflight(
            stage.path().parent().ok_or(QmfError::InvalidDimensions)?,
            total,
        )?;
        let manifest_len = total
            .checked_sub(self.payload.bytes)
            .and_then(|n| n.checked_sub((QMF_V1_HEADER_BYTES + QMF_CHECKSUM_BYTES) as u64))
            .ok_or(QmfError::PayloadTooLarge)?;
        let mut header = [0u8; QMF_V1_HEADER_BYTES];
        header[..4].copy_from_slice(&QMF_MAGIC);
        header[4..6].copy_from_slice(&QMF_V1_VERSION.to_le_bytes());
        header[6..8].copy_from_slice(&QMF_V1_FLAG_LITTLE_ENDIAN.to_le_bytes());
        header[8..16].copy_from_slice(&manifest_len.to_le_bytes());
        header[16..24].copy_from_slice(&self.payload.bytes.to_le_bytes());
        // Both hashing and JSON serialization stream directly to the final sibling stage.
        let mut writer = ChecksummedWriter {
            file: stage.as_file_mut(),
            crc: Crc32::new(),
            sha: Sha256::new(),
        };
        writer.write_all(&header)?;
        serde_json::to_writer(&mut writer, &self.manifest)
            .map_err(|e| QmfError::InvalidManifest(e.to_string()))?;
        self.payload.file.seek(SeekFrom::Start(0))?;
        let mut buffer = [0u8; COPY_BUFFER_BYTES];
        let mut remaining = self.payload.bytes;
        while remaining > 0 {
            cancelled(&self.payload.options)?;
            let count = usize::try_from(remaining.min(buffer.len() as u64))
                .map_err(|_| QmfError::PayloadTooLarge)?;
            self.payload.file.read_exact(&mut buffer[..count])?;
            writer.write_all(&buffer[..count])?;
            remaining -= count as u64;
        }
        let checksum = writer.crc.finish().to_le_bytes();
        writer.file.write_all(&checksum)?;
        writer.sha.update(&checksum);
        let expected_sha = writer.sha.finalize_hex();
        stage.as_file().sync_all()?;
        let mut reader = QmfFileReader::open(
            stage.as_file_mut(),
            &self.payload.options.limits.reader_budget(),
        )?;
        let verified = reader.verify_integrity_with_cancel(|| cancelled(&self.payload.options))?;
        if verified != expected_sha {
            return Err(QmfError::ChecksumMismatch);
        }
        // No operator decoding and no tensor-sized allocations during blob verification.
        verify_manifest_blobs(&self.manifest, &mut reader, &self.payload.options)?;
        cancelled(&self.payload.options)?;
        for source in sources {
            qml_core::publication::validate_destination(output, Some(source))?;
        }
        cancelled(&self.payload.options)?;
        qml_core::publication::publish(stage, output, None, overwrite)?;
        Ok((total_usize, verified))
    }
}
struct ChecksummedWriter<'a> {
    file: &'a mut File,
    crc: Crc32,
    sha: Sha256,
}
impl Write for ChecksummedWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let n = self.file.write(bytes)?;
        self.crc.update(&bytes[..n]);
        self.sha.update(&bytes[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

fn visit_factor(
    enc: &SharedLatentEncoding,
    f: &mut impl FnMut(BlobRef) -> Result<(), QmfError>,
) -> Result<(), QmfError> {
    match enc {
        SharedLatentEncoding::Fp32 { data } | SharedLatentEncoding::Fp16 { data } => f(*data),
        SharedLatentEncoding::Int8Sym { scales, codes, .. } => {
            f(*scales)?;
            f(*codes)
        }
    }
}
fn visit_sparse(
    sp: &Option<SparseResidualDesc>,
    f: &mut impl FnMut(BlobRef) -> Result<(), QmfError>,
) -> Result<(), QmfError> {
    if let Some(sp) = sp {
        f(sp.indices)?;
        f(sp.values_f16)?;
    }
    Ok(())
}
fn visit_rep(
    rep: &QmfRepresentation,
    f: &mut impl FnMut(BlobRef) -> Result<(), QmfError>,
) -> Result<(), QmfError> {
    match rep {
        QmfRepresentation::DenseF32 { data }
        | QmfRepresentation::DenseF16 { data }
        | QmfRepresentation::DenseBf16 { data } => f(*data)?,
        QmfRepresentation::Int8SymGroup { scales, codes, .. } => {
            f(*scales)?;
            f(*codes)?;
        }
        QmfRepresentation::Int4SymGroup {
            scales, nibbles, ..
        } => {
            f(*scales)?;
            f(*nibbles)?;
        }
        QmfRepresentation::Int4AsymGroup {
            scales,
            mins,
            nibbles,
            ..
        } => {
            f(*scales)?;
            f(*mins)?;
            f(*nibbles)?;
        }
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
            f(*scales)?;
            f(*mins)?;
            f(*packed)?;
        }
        QmfRepresentation::TernaryPacked { scales, packed, .. } => {
            f(*scales)?;
            f(*packed)?;
        }
        QmfRepresentation::VectorQuant {
            codebooks_f16,
            scales,
            codes,
            ..
        } => {
            f(*codebooks_f16)?;
            f(*scales)?;
            f(*codes)?;
        }
        QmfRepresentation::LowRankSvd { left, right, .. } => {
            visit_factor(left, f)?;
            visit_factor(right, f)?;
        }
        QmfRepresentation::RmqSharedBasis {
            core,
            sparse_residual,
            ..
        } => {
            visit_factor(core, f)?;
            visit_sparse(sparse_residual, f)?;
        }
        QmfRepresentation::RmqSharedCore {
            left,
            right,
            sparse_residual,
            ..
        } => {
            visit_factor(left, f)?;
            visit_factor(right, f)?;
            visit_sparse(sparse_residual, f)?;
        }
        QmfRepresentation::EchoCorrected {
            base,
            low_rank_left,
            low_rank_right,
            sparse_residual,
            ..
        } => {
            visit_rep(base, f)?;
            for factor in [low_rank_left, low_rank_right].into_iter().flatten() {
                visit_factor(factor, f)?;
            }
            visit_sparse(sparse_residual, f)?;
        }
    }
    Ok(())
}
fn verify_manifest_blobs<R: Read + Seek>(
    manifest: &QmfManifest,
    reader: &mut QmfFileReader<R>,
    options: &DiskBuildOptions,
) -> Result<(), QmfError> {
    let mut buffer = [0u8; COPY_BUFFER_BYTES];
    let mut verify = |blob: BlobRef| -> Result<(), QmfError> {
        let end = blob
            .offset
            .checked_add(blob.len)
            .ok_or(QmfError::PayloadTooLarge)?;
        if end > reader.payload_bytes {
            return Err(QmfError::InvalidManifest("blob exceeds payload".into()));
        }
        let offset = reader
            .payload_base_offset
            .checked_add(blob.offset)
            .ok_or(QmfError::PayloadTooLarge)?;
        reader.reader.seek(SeekFrom::Start(offset))?;
        let mut remaining = blob.len;
        let mut crc = Crc32::new();
        while remaining > 0 {
            cancelled(options)?;
            let n = remaining.min(buffer.len() as u64) as usize;
            reader.reader.read_exact(&mut buffer[..n])?;
            crc.update(&buffer[..n]);
            remaining -= n as u64;
        }
        if crc.finish() != blob.crc32 {
            return Err(QmfError::ChecksumMismatch);
        }
        Ok(())
    };
    let ids: BTreeSet<_> = manifest
        .shared_latents
        .iter()
        .map(|l| l.id.as_str())
        .collect();
    for latent in &manifest.shared_latents {
        visit_factor(&latent.encoding, &mut verify)?;
    }
    for tensor in &manifest.tensors {
        let elements = tensor
            .shape
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(d as u64))
            .ok_or(QmfError::PayloadTooLarge)?;
        if elements == 0 {
            return Err(QmfError::InvalidDimensions);
        }
        match &tensor.representation {
            QmfRepresentation::DenseF32 { data } => {
                if elements.checked_mul(4) != Some(data.len) {
                    return Err(QmfError::InvalidDimensions);
                }
            }
            QmfRepresentation::DenseF16 { data } | QmfRepresentation::DenseBf16 { data } => {
                if elements.checked_mul(2) != Some(data.len) {
                    return Err(QmfError::InvalidDimensions);
                }
            }
            QmfRepresentation::RmqSharedBasis {
                u_latent_id,
                vt_latent_id,
                ..
            } if !ids.contains(u_latent_id.as_str()) || !ids.contains(vt_latent_id.as_str()) => {
                return Err(QmfError::InvalidManifest("missing shared latent".into()));
            }
            QmfRepresentation::RmqSharedCore { s_latent_id, .. }
                if !ids.contains(s_latent_id.as_str()) =>
            {
                return Err(QmfError::InvalidManifest("missing shared latent".into()));
            }
            _ => {}
        }
        visit_rep(&tensor.representation, &mut verify)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use qml_quant::{RmqResidualKind, RmqSharedBasisConfig};
    fn options(dir: &Path) -> DiskBuildOptions {
        DiskBuildOptions {
            scratch_directory: dir.into(),
            ..Default::default()
        }
    }
    fn builder(options: DiskBuildOptions) -> QmfV1Builder {
        QmfV1Builder::new_disk("synthetic", "safetensors", "MIT", "test", "all", 7, options)
            .unwrap()
    }
    fn populate(b: &mut QmfV1Builder) {
        let m = DenseMatrix::from_flat(16, 16, (0..256).map(|i| (i as f64 * 0.17).sin()).collect())
            .unwrap();
        b.add_dense_f32("norm", vec![16], &[1.0; 16]).unwrap();
        b.add_fp16("f16", &Fp16Matrix::quantize(&m)).unwrap();
        b.add_bf16("bf16", &Bf16Matrix::quantize(&m)).unwrap();
        b.add_int8_sym("int8", &Int8SymMatrix::quantize(&m, 16).unwrap())
            .unwrap();
        b.add_int4_sym("int4", &Int4SymGroupMatrix::quantize(&m, 16).unwrap())
            .unwrap();
        b.add_int4_asym("asym4", &Int4AsymGroupMatrix::quantize(&m, 16).unwrap())
            .unwrap();
        b.add_low_rank_svd(
            "svd",
            &LowRankSvdMatrix::from_matrix(&m, 4, FactorPrecision::Fp16, 7).unwrap(),
        )
        .unwrap();
        let had = HadamardSpec::new(16, 7);
        b.add_int3_asym("int3", &Int3AsymGroupMatrix::quantize(&m, 16, had).unwrap())
            .unwrap();
        b.add_int2_asym("int2", &Int2AsymGroupMatrix::quantize(&m, 16, had).unwrap())
            .unwrap();
        b.add_ternary_packed(
            "ternary",
            &TernaryPackedMatrix::quantize(&m, 16, had).unwrap(),
        )
        .unwrap();
        let cluster = RmqSharedBasisCluster::fit(
            &[&m, &m],
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
        b.add_rmq_shared_basis_cluster("basis", &["basis0".into(), "basis1".into()], &cluster)
            .unwrap();
        let cluster = RmqSharedCoreCluster::fit(
            &[&m, &m],
            4,
            FactorPrecision::Int8Sym,
            RmqResidualKind::SparseTopFraction { fraction: 0.02 },
            7,
        )
        .unwrap();
        b.add_rmq_shared_core_cluster("core", &["core0".into(), "core1".into()], &cluster)
            .unwrap();
    }
    #[test]
    fn disk_and_memory_are_byte_identical_and_reader_compatible() {
        let dir = tempfile::tempdir().unwrap();
        let mut memory = QmfV1Builder::new("synthetic", "safetensors", "MIT", "test", "all", 7);
        let mut disk = builder(options(dir.path()));
        populate(&mut memory);
        populate(&mut disk);
        let archive = memory.build().unwrap();
        let mut expected = Vec::new();
        archive.write_to(&mut expected).unwrap();
        let disk = disk.finish_disk().unwrap();
        assert_eq!(disk.serialized_bytes().unwrap(), expected.len() as u64);
        let out = dir.path().join("artifact.qmf");
        let (size, sha) = disk.publish(&out, &[], false).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), expected);
        assert_eq!(size, expected.len());
        assert_eq!(sha, qml_core::sha256_hex(&expected));
        let mut reader =
            QmfFileReader::open(File::open(out).unwrap(), &QmfBudget::default()).unwrap();
        assert_eq!(reader.manifest(), archive.manifest());
        assert_eq!(reader.verify_integrity_streaming().unwrap(), sha);
        let mut cache = QmfLatentCache::new();
        let x = vec![0.1; 16];
        for tensor in &archive.manifest().tensors {
            if tensor.shape.len() == 2 {
                assert_eq!(
                    reader
                        .load_operator_cached(&tensor.name, &mut cache)
                        .unwrap()
                        .matvec(&x)
                        .unwrap(),
                    archive
                        .load_operator(&tensor.name)
                        .unwrap()
                        .matvec(&x)
                        .unwrap()
                );
            }
        }
        assert_eq!(cache.len(), 3); // two basis latents and one shared core
    }
    #[test]
    fn bounded_payload_memory_instrumentation() {
        let dir = tempfile::tempdir().unwrap();
        for elements in [32_768usize, 1_048_576] {
            let values = vec![0.25f32; elements]; // Borrowed caller allocation, deliberately outside builder accounting.
            let mut opts = options(dir.path());
            opts.limits.max_resident_bytes = 1024 * 1024;
            opts.limits.max_artifact_bytes = 64 * 1024 * 1024 * 1024;
            let mut b = builder(opts);
            for tensor in 0..8 {
                b.add_dense_f32(format!("tensor{tensor}"), vec![elements], &values)
                    .unwrap();
            }
            let disk = b.finish_disk().unwrap();
            let stats = disk.memory_stats().clone();
            assert_eq!(stats.payload_bytes, (elements * 4 * 8) as u64);
            assert_eq!(stats.resident_payload_bytes, 0);
            assert_eq!(stats.peak_encoding_buffer_bytes, COPY_BUFFER_BYTES);
            assert!(stats.manifest_accounted_bytes < 1024 * 1024);
            eprintln!(
                "ATLAS payload={} resident_payload={} encoding_buffer={} manifest_accounted={}",
                stats.payload_bytes,
                stats.resident_payload_bytes,
                stats.peak_encoding_buffer_bytes,
                stats.manifest_accounted_bytes
            );
            disk.publish(&dir.path().join(format!("{elements}.qmf")), &[], false)
                .unwrap();
        }
    }
    #[test]
    fn independent_limits_overflow_preflight_and_owned_cleanup() {
        let dir = tempfile::tempdir().unwrap();
        let scratch = dir.path().join("scratch");
        std::fs::create_dir(&scratch).unwrap();
        std::fs::write(scratch.join("unrelated"), b"keep").unwrap();
        assert!(ArtifactLimits::default().total_bytes(u64::MAX, 1).is_err());
        assert!(disk_preflight(&scratch, u64::MAX).is_err());
        for which in 0..5 {
            let mut opts = options(&scratch);
            match which {
                0 => opts.limits.max_resident_bytes = 1,
                1 => opts.limits.max_manifest_bytes = 1,
                2 => opts.limits.max_artifact_bytes = 128,
                3 => opts.limits.max_temporary_bytes = 8,
                _ => opts.limits.max_tensor_bytes = 8,
            }
            let result =
                QmfV1Builder::new_disk("synthetic", "safetensors", "MIT", "test", "all", 7, opts)
                    .and_then(|mut b| {
                        b.add_dense_f32("x", vec![64], &[1.0; 64])?;
                        b.finish_disk()
                    });
            assert!(result.is_err());
            assert_eq!(std::fs::read_dir(&scratch).unwrap().count(), 1);
        }
        assert_eq!(std::fs::read(scratch.join("unrelated")).unwrap(), b"keep");
    }
    #[test]
    fn corrupted_scratch_failed_publication_and_cancel_preserve_outputs() {
        let dir = tempfile::tempdir().unwrap();
        let scratch = dir.path().join("scratch");
        std::fs::create_dir(&scratch).unwrap();
        let out = dir.path().join("existing.qmf");
        std::fs::write(&out, b"old valid artifact").unwrap();
        let mut b = builder(options(&scratch));
        b.add_dense_f32("x", vec![4], &[1.0; 4]).unwrap();
        let mut artifact = b.finish_disk().unwrap();
        artifact.payload.file.seek(SeekFrom::Start(0)).unwrap();
        artifact.payload.file.write_all(&[0xff]).unwrap();
        assert!(matches!(
            artifact.publish(&out, &[], true),
            Err(QmfError::ChecksumMismatch)
        ));
        assert_eq!(std::fs::read(&out).unwrap(), b"old valid artifact");
        for cancelled_before in [false, true] {
            let opts = options(&scratch);
            let flag = Arc::clone(&opts.cancel);
            let mut b = builder(opts);
            b.add_dense_f32("x", vec![4], &[1.0; 4]).unwrap();
            let a = b.finish_disk().unwrap();
            flag.store(cancelled_before, Ordering::Relaxed);
            assert!(a.publish(&out, &[], false).is_err());
            assert_eq!(std::fs::read(&out).unwrap(), b"old valid artifact");
            assert_eq!(std::fs::read_dir(&scratch).unwrap().count(), 0);
        }
        let mut opts = options(&scratch);
        opts.limits.max_temporary_bytes = 128;
        let mut b = builder(opts);
        b.add_dense_f32("x", vec![4], &[1.0; 4]).unwrap();
        assert!(b.finish_disk().unwrap().publish(&out, &[], true).is_err());
        assert_eq!(std::fs::read(&out).unwrap(), b"old valid artifact");
    }
    #[test]
    fn malformed_shape_and_source_alias_are_rejected_before_publication() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("source");
        std::fs::write(&src, b"immutable").unwrap();
        let out = dir.path().join("out");
        let mut b = builder(options(dir.path()));
        b.add_dense_f32("bad", vec![usize::MAX, 2], &[1.0]).unwrap();
        assert!(b.finish_disk().unwrap().publish(&out, &[], false).is_err());
        assert!(!out.exists());
        let mut b = builder(options(dir.path()));
        b.add_dense_f32("x", vec![1], &[1.0]).unwrap();
        std::fs::hard_link(&src, &out).unwrap();
        assert!(
            b.finish_disk()
                .unwrap()
                .publish(&out, std::slice::from_ref(&src), true)
                .is_err()
        );
        assert_eq!(std::fs::read(src).unwrap(), b"immutable");
    }

    #[test]
    fn metadata_limit_failure_is_not_silently_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let mut opts = options(dir.path());
        opts.limits.max_manifest_bytes = 4096;
        let mut b = builder(opts);
        b.insert_metadata("oversized", "x".repeat(8192));
        assert!(b.finish_disk().is_err());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}
