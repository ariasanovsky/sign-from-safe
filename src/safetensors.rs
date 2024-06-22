use cuts_v2::MatRef;
use equator::assert;
use safetensors::{tensor::TensorView, Dtype, View};

pub fn load_matrix_f32(view: TensorView) -> Option<faer::Mat<f32>> {
    let &[nrows, ncols] = view.shape() else {
        return None
    };
    assert!(all(
        view.dtype() == Dtype::BF16,
        view.data().len() == nrows * ncols * 2,
    ));
    let data: &[u16] = bytemuck::cast_slice(view.data());
    let data: &[half::bf16] = unsafe { core::mem::transmute(data) };
    Some(faer::Mat::from_fn(nrows, ncols, |i, j| {
        let ij = j + ncols * i;
        let x = data[ij];
        x.to_f32()
    }))
}

pub fn load_matrix_f64(view: TensorView) -> Option<faer::Mat<f64>> {
    let &[nrows, ncols] = view.shape() else {
        return None
    };
    assert!(all(
        view.dtype() == Dtype::BF16,
        view.data().len() == nrows * ncols * 2,
    ));
    let data: &[u16] = bytemuck::cast_slice(view.data());
    let data: &[half::bf16] = unsafe { core::mem::transmute(data) };
    Some(faer::Mat::from_fn(nrows, ncols, |i, j| {
        let ij = j + ncols * i;
        let x = data[ij];
        x.to_f64()
    }))
}
