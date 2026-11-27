//! Optional means not present in the directory, never failed decoding.
use crate::RuntimeError;
use qml_format::{
    QmfFileReader,
    v1::{QmfRepresentation, QmfTensorEntry},
};
use std::io::{Read, Seek};

pub(super) fn validate_qmf(
    entry: Option<&QmfTensorEntry>,
    name: &str,
    expected: usize,
) -> Result<(), RuntimeError> {
    let Some(entry) = entry else { return Ok(()) };
    validate_shape(name, &entry.shape, expected)?;
    let (data, width) = match &entry.representation {
        QmfRepresentation::DenseF32 { data } => (data, 4_u64),
        QmfRepresentation::DenseF16 { data } | QmfRepresentation::DenseBf16 { data } => {
            (data, 2_u64)
        }
        _ => {
            return Err(RuntimeError::InvalidConfig(format!(
                "optional attention bias {name} is not a supported dense vector representation"
            )));
        }
    };
    let bytes = u64::try_from(expected)
        .ok()
        .and_then(|n| n.checked_mul(width))
        .ok_or_else(|| {
            RuntimeError::InvalidConfig(format!(
                "optional attention bias {name} byte length overflow"
            ))
        })?;
    if data.len != bytes {
        return Err(RuntimeError::InvalidConfig(format!(
            "optional attention bias {name} has payload byte length {}; expected {bytes}",
            data.len
        )));
    }
    Ok(())
}

fn validate_shape(name: &str, shape: &[usize], expected: usize) -> Result<(), RuntimeError> {
    if shape != [expected] {
        return Err(RuntimeError::InvalidConfig(format!(
            "optional attention bias {name} has shape {shape:?}; expected [{expected}]"
        )));
    }
    Ok(())
}

pub(super) fn load(
    name: &str,
    shape: Option<&[usize]>,
    expected: usize,
    decode: impl FnOnce() -> Result<Vec<f64>, RuntimeError>,
) -> Result<Option<Vec<f64>>, RuntimeError> {
    let Some(shape) = shape else { return Ok(None) };
    validate_shape(name, shape, expected)?;
    let values = decode().map_err(|error| {
        RuntimeError::InvalidConfig(format!(
            "optional attention bias {name} failed to load: {error}"
        ))
    })?;
    if values.len() != expected || values.iter().any(|v| !v.is_finite()) {
        return Err(RuntimeError::InvalidConfig(format!(
            "optional attention bias {name} must contain {expected} finite values"
        )));
    }
    Ok(Some(values))
}

pub(super) fn from_reader<R: Read + Seek>(
    reader: &mut QmfFileReader<R>,
    name: &str,
    expected: usize,
) -> Result<Option<Vec<f64>>, RuntimeError> {
    let entry = reader
        .manifest()
        .tensors
        .iter()
        .find(|tensor| tensor.name == name);
    validate_qmf(entry, name, expected)?;
    let shape = entry.map(|tensor| tensor.shape.clone());
    load(name, shape.as_deref(), expected, || {
        Ok(reader.load_vector_f64(name)?)
    })
}
