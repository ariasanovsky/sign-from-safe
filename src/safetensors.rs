use bytemuck::Pod;
use equator::assert;
use safetensors::{tensor::TensorView, Dtype};

pub fn load_matrix_f32(view: &TensorView) -> Option<faer::Mat<f32>> {
    let &[nrows, ncols] = view.shape() else {
        return None;
    };
    assert!(all(
        view.dtype() == Dtype::BF16,
        view.data().len() == nrows * ncols * 2,
    ));
    let data: &[u16] = bytemuck::cast_slice(view.data());
    let data =
        unsafe { core::slice::from_raw_parts(data.as_ptr() as *const half::bf16, data.len()) };
    Some(faer::Mat::from_fn(nrows, ncols, |i, j| {
        let ij = j + ncols * i;
        let x = data[ij];
        x.to_f32()
    }))
}

pub fn store_f32_matrix_to_bf16(mat: faer::MatRef<'_, f32>) -> Vec<u8> {
    let (nrows, ncols) = mat.shape();
    let mut data = vec![0u8; 2 * nrows * ncols];
    for j in 0..ncols {
        for i in 0..nrows {
            let ij = j + ncols * i;
            let data: &mut [u8; 2] = (&mut data[ij..][..2]).try_into().unwrap();
            *data = unsafe {
                core::mem::transmute::<half::bf16, [u8; 2]>(half::bf16::from_f32(mat[(i, j)]))
            };
        }
    }
    data
}

pub fn load_slice<T: Pod>(view: TensorView) -> Option<&[T]> {
    if view.shape().len() != 1 {
        return None;
    };
    Some(bytemuck::cast_slice(view.data()))
}

pub fn load_matrix_f64(view: &TensorView) -> Option<faer::Mat<f64>> {
    let &[nrows, ncols] = view.shape() else {
        return None;
    };
    assert!(all(
        view.dtype() == Dtype::BF16,
        view.data().len() == nrows * ncols * 2,
    ));
    let data: &[u16] = bytemuck::cast_slice(view.data());
    let data =
        unsafe { core::slice::from_raw_parts(data.as_ptr() as *const half::bf16, data.len()) };
    Some(faer::Mat::from_fn(nrows, ncols, |i, j| {
        let ij = j + ncols * i;
        let x = data[ij];
        x.to_f64()
    }))
}
