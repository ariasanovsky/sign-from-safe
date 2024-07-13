#[cfg(feature = "candle")]
#[test]
fn load_consistent_with_candle() {
    use crate::safetensors::load_matrix_f32;
    use candle_core::{Device, Tensor};
    use safetensors::{serialize, SafeTensors};
    let a = Tensor::arange(0f32, 6f32, &Device::Cpu)
        .unwrap()
        .reshape((2, 3))
        .unwrap()
        .to_dtype(candle_core::DType::BF16)
        .unwrap();
    let b = serialize(core::iter::once(("test".to_string(), &a)), &None).unwrap();
    let tensors = SafeTensors::deserialize(b.as_slice()).unwrap();
    assert!(tensors.len() == 1);
    let a_view = tensors.tensor("test").unwrap();
    let a_mat = load_matrix_f32(&a_view).unwrap();
    let a_vecs: Vec<Vec<half::bf16>> = a.to_vec2().unwrap();
    for i in 0..2 {
        for j in 0..3 {
            assert!(half::bf16::from_f32(a_mat[(i, j)]) == a_vecs[i][j]);
        }
    }
}
