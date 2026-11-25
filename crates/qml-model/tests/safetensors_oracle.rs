use qml_model::inspect_safetensors;
use safetensors::SafeTensors;
use std::io::Cursor;

fn synthetic_file(header_json: &str, payload: &[u8]) -> Vec<u8> {
    let header_len = header_json.len().div_ceil(8) * 8;
    let mut bytes = Vec::with_capacity(8 + header_len + payload.len());
    bytes.extend_from_slice(&(header_len as u64).to_le_bytes());
    bytes.extend_from_slice(header_json.as_bytes());
    bytes.resize(8 + header_len, b' ');
    bytes.extend_from_slice(payload);
    bytes
}

#[test]
fn synthetic_f32_tensor_matches_upstream_metadata() {
    let header = r#"{"weights":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"__metadata__":{"source":"generated-test"}}"#;
    let file = synthetic_file(header, &[0; 8]);
    let qml = inspect_safetensors(Cursor::new(&file)).expect("QML metadata parse");
    let upstream = SafeTensors::deserialize(&file).expect("upstream oracle parse");
    let upstream_names = upstream.names();
    assert_eq!(upstream_names, ["weights"]);
    let view = upstream.tensor("weights").expect("known tensor");
    assert_eq!(qml.tensors.len(), 1);
    assert_eq!(qml.tensors[0].name, "weights");
    assert_eq!(qml.tensors[0].dtype, view.dtype().to_string());
    assert_eq!(
        qml.tensors[0].shape,
        view.shape()
            .iter()
            .map(|dim| *dim as u64)
            .collect::<Vec<_>>()
    );
    assert_eq!(qml.tensors[0].data_offsets, [0, 8]);
    assert_eq!(qml.metadata["source"], "generated-test");
}

#[test]
fn sub_byte_fixture_matches_upstream_oracle() {
    let header = r#"{"packed":{"dtype":"F4","shape":[2],"data_offsets":[0,1]}}"#;
    let file = synthetic_file(header, &[0x12]);
    let qml = inspect_safetensors(Cursor::new(&file)).expect("QML accepts aligned F4 tensor");
    let upstream = SafeTensors::deserialize(&file).expect("upstream accepts aligned F4 tensor");
    assert_eq!(
        upstream.tensor("packed").unwrap().dtype().to_string(),
        qml.tensors[0].dtype
    );
    assert_eq!(
        upstream.tensor("packed").unwrap().data().len(),
        qml.tensors[0].data_bytes as usize
    );
}
