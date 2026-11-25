//! Experimental QMF containers (`v0` single rank-one prototype and `v1`
//! generalized multi-tensor container with shared latent tables).
//!
//! Both versions remain experimental and not compatibility-stable. CRC-32
//! verifies accidental corruption only and is not cryptographic authentication.

pub mod v1;

pub use v1::{
    BlobRef, QMF_V1_FLAG_LITTLE_ENDIAN, QMF_V1_HEADER_BYTES, QMF_V1_MAX_MANIFEST_BYTES,
    QMF_V1_VERSION, QmfArchiveV1, QmfManifest, QmfOperator, QmfRepresentation, QmfTensorEntry,
    QmfV1Builder, SharedLatentEncoding, SharedLatentEntry, SparseResidualDesc, peek_qmf_version,
};

use qml_core::{MatrixError, RankOneFactors};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io::{Read, Write};

pub const QMF_VERSION: u16 = 0;
pub const QMF_MAGIC: [u8; 4] = *b"QMF\0";
pub const QMF_HEADER_BYTES: usize = 24;
pub const QMF_CHECKSUM_BYTES: usize = 4;
pub const MAX_QMF_DIMENSION: usize = 16_777_216;
pub const MAX_QMF_FILE_BYTES: usize = 256 * 1024 * 1024;
const REPRESENTATION_RANK_ONE_F64: u8 = 1;

#[derive(Debug)]
pub enum QmfError {
    Io(std::io::Error),
    InvalidMagic,
    UnsupportedVersion(u16),
    UnsupportedRepresentation(u8),
    NonZeroReservedField,
    InvalidDimensions,
    PayloadTooLarge,
    LengthMismatch { expected: usize, actual: usize },
    ChecksumMismatch,
    Matrix(MatrixError),
    InvalidManifest(String),
}

impl Display for QmfError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "QMF I/O error: {error}"),
            Self::InvalidMagic => write!(f, "invalid QMF magic bytes"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported QMF version {version}")
            }
            Self::UnsupportedRepresentation(id) => {
                write!(f, "unsupported QMF representation id {id}")
            }
            Self::NonZeroReservedField => write!(f, "reserved QMF header field must be zero"),
            Self::InvalidDimensions => write!(f, "invalid or unsupported QMF dimensions"),
            Self::PayloadTooLarge => write!(f, "QMF payload exceeds the configured file limit"),
            Self::LengthMismatch { expected, actual } => write!(
                f,
                "QMF length mismatch: expected {expected} bytes, received {actual}"
            ),
            Self::ChecksumMismatch => write!(f, "QMF CRC-32 checksum mismatch"),
            Self::Matrix(error) => write!(f, "invalid QMF matrix factors: {error}"),
            Self::InvalidManifest(msg) => write!(f, "invalid QMF v1 manifest: {msg}"),
        }
    }
}

impl Error for QmfError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for QmfError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<MatrixError> for QmfError {
    fn from(value: MatrixError) -> Self {
        Self::Matrix(value)
    }
}

/// A single experimental rank-one matrix in QMF version 0.
#[derive(Debug, Clone, PartialEq)]
pub struct QmfRankOne {
    factors: RankOneFactors,
}

impl QmfRankOne {
    pub fn new(factors: RankOneFactors) -> Self {
        Self { factors }
    }

    pub fn version(&self) -> u16 {
        QMF_VERSION
    }

    pub fn factors(&self) -> &RankOneFactors {
        &self.factors
    }

    /// Exact total container size, including header and checksum.
    pub fn serialized_len(&self) -> Result<usize, QmfError> {
        let elements = self
            .factors
            .rows()
            .checked_add(self.factors.cols())
            .ok_or(QmfError::PayloadTooLarge)?;
        let payload = elements
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(QmfError::PayloadTooLarge)?;
        let total = QMF_HEADER_BYTES
            .checked_add(payload)
            .and_then(|value| value.checked_add(QMF_CHECKSUM_BYTES))
            .ok_or(QmfError::PayloadTooLarge)?;
        if total > MAX_QMF_FILE_BYTES {
            return Err(QmfError::PayloadTooLarge);
        }
        Ok(total)
    }

    /// Write deterministic v0 bytes. Values are streamed without building a dense matrix.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<(), QmfError> {
        let total_len = self.serialized_len()?;
        validate_dimensions(self.factors.rows(), self.factors.cols())?;
        let mut header = [0_u8; QMF_HEADER_BYTES];
        header[0..4].copy_from_slice(&QMF_MAGIC);
        header[4..6].copy_from_slice(&QMF_VERSION.to_le_bytes());
        header[6] = REPRESENTATION_RANK_ONE_F64;
        header[7] = 0;
        header[8..16].copy_from_slice(&(self.factors.rows() as u64).to_le_bytes());
        header[16..24].copy_from_slice(&(self.factors.cols() as u64).to_le_bytes());

        let mut checksum = Crc32::new();
        writer.write_all(&header)?;
        checksum.update(&header);
        for value in self.factors.left().iter().chain(self.factors.right()) {
            let encoded = value.to_le_bytes();
            writer.write_all(&encoded)?;
            checksum.update(&encoded);
        }
        writer.write_all(&checksum.finish().to_le_bytes())?;
        debug_assert_eq!(
            total_len,
            QMF_HEADER_BYTES + (self.factors.parameter_count() * 8) + 4
        );
        Ok(())
    }

    /// Read a QMF v0 file from a stream, rejecting oversized and malformed inputs.
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
        if bytes.len() < QMF_HEADER_BYTES + QMF_CHECKSUM_BYTES {
            return Err(QmfError::LengthMismatch {
                expected: QMF_HEADER_BYTES + QMF_CHECKSUM_BYTES,
                actual: bytes.len(),
            });
        }
        if bytes[0..4] != QMF_MAGIC {
            return Err(QmfError::InvalidMagic);
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != QMF_VERSION {
            return Err(QmfError::UnsupportedVersion(version));
        }
        if bytes[6] != REPRESENTATION_RANK_ONE_F64 {
            return Err(QmfError::UnsupportedRepresentation(bytes[6]));
        }
        if bytes[7] != 0 {
            return Err(QmfError::NonZeroReservedField);
        }
        let rows_u64 = u64::from_le_bytes(bytes[8..16].try_into().expect("fixed header slice"));
        let cols_u64 = u64::from_le_bytes(bytes[16..24].try_into().expect("fixed header slice"));
        let rows = usize::try_from(rows_u64).map_err(|_| QmfError::InvalidDimensions)?;
        let cols = usize::try_from(cols_u64).map_err(|_| QmfError::InvalidDimensions)?;
        validate_dimensions(rows, cols)?;

        let elements = rows.checked_add(cols).ok_or(QmfError::PayloadTooLarge)?;
        let payload_bytes = elements
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(QmfError::PayloadTooLarge)?;
        let expected_len = QMF_HEADER_BYTES
            .checked_add(payload_bytes)
            .and_then(|value| value.checked_add(QMF_CHECKSUM_BYTES))
            .ok_or(QmfError::PayloadTooLarge)?;
        if expected_len > MAX_QMF_FILE_BYTES {
            return Err(QmfError::PayloadTooLarge);
        }
        if bytes.len() != expected_len {
            return Err(QmfError::LengthMismatch {
                expected: expected_len,
                actual: bytes.len(),
            });
        }

        let checksum_start = bytes.len() - QMF_CHECKSUM_BYTES;
        let stored_checksum = u32::from_le_bytes(
            bytes[checksum_start..]
                .try_into()
                .expect("fixed checksum slice"),
        );
        if crc32(&bytes[..checksum_start]) != stored_checksum {
            return Err(QmfError::ChecksumMismatch);
        }

        let payload = &bytes[QMF_HEADER_BYTES..checksum_start];
        let mut values = Vec::with_capacity(elements);
        let (encoded_values, _) = payload.as_chunks::<8>();
        for (index, encoded) in encoded_values.iter().enumerate() {
            let value = f64::from_le_bytes(*encoded);
            if !value.is_finite() {
                return Err(QmfError::Matrix(MatrixError::NonFiniteValue { index }));
            }
            values.push(value);
        }
        let right = values.split_off(rows);
        let left = values;
        Ok(Self::new(RankOneFactors::new(left, right)?))
    }
}

fn validate_dimensions(rows: usize, cols: usize) -> Result<(), QmfError> {
    if rows == 0 || cols == 0 || rows > MAX_QMF_DIMENSION || cols > MAX_QMF_DIMENSION {
        return Err(QmfError::InvalidDimensions);
    }
    let elements = rows.checked_add(cols).ok_or(QmfError::PayloadTooLarge)?;
    if elements
        .checked_mul(std::mem::size_of::<f64>())
        .and_then(|payload| payload.checked_add(QMF_HEADER_BYTES + QMF_CHECKSUM_BYTES))
        .is_none_or(|total| total > MAX_QMF_FILE_BYTES)
    {
        return Err(QmfError::PayloadTooLarge);
    }
    Ok(())
}

pub(crate) struct Crc32(u32);

impl Crc32 {
    pub(crate) fn new() -> Self {
        Self(!0_u32)
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u32::from(*byte);
            for _ in 0..8 {
                let mask = 0_u32.wrapping_sub(self.0 & 1);
                self.0 = (self.0 >> 1) ^ (0xedb8_8320 & mask);
            }
        }
    }

    pub(crate) fn finish(self) -> u32 {
        !self.0
    }
}

pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    let mut checksum = Crc32::new();
    checksum.update(bytes);
    checksum.finish()
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_QMF_DIMENSION, QMF_HEADER_BYTES, QMF_MAGIC, QMF_VERSION, QmfError, QmfRankOne,
    };
    use qml_core::{MatrixError, RankOneFactors};

    fn example() -> QmfRankOne {
        QmfRankOne::new(RankOneFactors::new(vec![1.0, -2.0], vec![0.5, 3.0, 4.0]).unwrap())
    }

    #[test]
    fn deterministic_round_trip_preserves_factors_and_size() {
        let value = example();
        let mut first = Vec::new();
        let mut second = Vec::new();
        value.write_to(&mut first).unwrap();
        value.write_to(&mut second).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), value.serialized_len().unwrap());
        let reloaded = QmfRankOne::from_bytes(&first).unwrap();
        assert_eq!(reloaded, value);
    }

    #[test]
    fn rejects_corruption_and_unknown_version() {
        let mut bytes = Vec::new();
        example().write_to(&mut bytes).unwrap();
        bytes[QMF_HEADER_BYTES] ^= 0x01;
        assert!(matches!(
            QmfRankOne::from_bytes(&bytes),
            Err(QmfError::ChecksumMismatch)
        ));

        let mut bytes = Vec::new();
        example().write_to(&mut bytes).unwrap();
        bytes[4] = 1;
        assert!(matches!(
            QmfRankOne::from_bytes(&bytes),
            Err(QmfError::UnsupportedVersion(1))
        ));
    }

    #[test]
    fn rejects_trailing_truncated_and_bad_magic_data() {
        let mut bytes = Vec::new();
        example().write_to(&mut bytes).unwrap();
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(matches!(
            QmfRankOne::from_bytes(&trailing),
            Err(QmfError::LengthMismatch { .. })
        ));
        assert!(matches!(
            QmfRankOne::from_bytes(&bytes[..10]),
            Err(QmfError::LengthMismatch { .. })
        ));
        bytes[0] = b'X';
        assert!(matches!(
            QmfRankOne::from_bytes(&bytes),
            Err(QmfError::InvalidMagic)
        ));
    }

    #[test]
    fn rejects_unsupported_fields_and_oversized_declared_shapes() {
        let mut bytes = Vec::new();
        example().write_to(&mut bytes).unwrap();

        let mut unsupported_representation = bytes.clone();
        unsupported_representation[6] = 2;
        assert!(matches!(
            QmfRankOne::from_bytes(&unsupported_representation),
            Err(QmfError::UnsupportedRepresentation(2))
        ));

        let mut reserved = bytes;
        reserved[7] = 1;
        assert!(matches!(
            QmfRankOne::from_bytes(&reserved),
            Err(QmfError::NonZeroReservedField)
        ));

        let mut excessive_axis = minimal_header(MAX_QMF_DIMENSION as u64 + 1, 1);
        assert!(matches!(
            QmfRankOne::from_bytes(&excessive_axis),
            Err(QmfError::InvalidDimensions)
        ));

        excessive_axis = minimal_header(MAX_QMF_DIMENSION as u64, MAX_QMF_DIMENSION as u64);
        assert!(matches!(
            QmfRankOne::from_bytes(&excessive_axis),
            Err(QmfError::PayloadTooLarge)
        ));
    }

    #[test]
    fn rejects_non_finite_factor_payload_even_with_valid_checksum() {
        let mut bytes = Vec::new();
        example().write_to(&mut bytes).unwrap();
        bytes[QMF_HEADER_BYTES..QMF_HEADER_BYTES + 8].copy_from_slice(&f64::NAN.to_le_bytes());
        let checksum_start = bytes.len() - 4;
        let checksum = super::crc32(&bytes[..checksum_start]);
        bytes[checksum_start..].copy_from_slice(&checksum.to_le_bytes());

        assert!(matches!(
            QmfRankOne::from_bytes(&bytes),
            Err(QmfError::Matrix(MatrixError::NonFiniteValue { index: 0 }))
        ));
    }

    fn minimal_header(rows: u64, cols: u64) -> Vec<u8> {
        let mut bytes = vec![0_u8; QMF_HEADER_BYTES + 4];
        bytes[0..4].copy_from_slice(&QMF_MAGIC);
        bytes[4..6].copy_from_slice(&QMF_VERSION.to_le_bytes());
        bytes[6] = 1;
        bytes[8..16].copy_from_slice(&rows.to_le_bytes());
        bytes[16..24].copy_from_slice(&cols.to_le_bytes());
        bytes
    }

    #[test]
    fn crc32_matches_standard_check_value() {
        assert_eq!(super::crc32(b"123456789"), 0xcbf4_3926);
    }
}
