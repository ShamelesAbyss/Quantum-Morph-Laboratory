//! Optional means not present in the directory, never failed decoding.
use crate::RuntimeError;
use qml_format::QmfFileReader;
use std::io::{Read, Seek};

pub(super) fn load(
    name: &str,
    shape: Option<&[usize]>,
    expected: usize,
    decode: impl FnOnce() -> Result<Vec<f64>, RuntimeError>,
) -> Result<Option<Vec<f64>>, RuntimeError> {
    let Some(shape) = shape else { return Ok(None) };
    if shape != [expected] {
        return Err(RuntimeError::InvalidConfig(format!(
            "optional attention bias {name} has shape {shape:?}; expected [{expected}]"
        )));
    }
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
    let shape = reader
        .manifest()
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .map(|tensor| tensor.shape.clone());
    load(name, shape.as_deref(), expected, || {
        Ok(reader.load_vector_f64(name)?)
    })
}
