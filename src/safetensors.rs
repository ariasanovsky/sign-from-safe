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
            let data: &mut [u8; 2] = (&mut data[2 * ij..][..2]).try_into().unwrap();
            *data = unsafe {
                core::mem::transmute::<half::bf16, [u8; 2]>(half::bf16::from_f32(mat[(i, j)]))
            };
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use faer::{stats::StandardNormalMat, Mat, MatRef};
    use half::bf16;
    use rand::{prelude::Distribution, rngs::StdRng, SeedableRng};

    use super::store_f32_matrix_to_bf16;

    #[test]
    fn stores_bf16_bytes() {
        let rng = &mut StdRng::seed_from_u64(0);
        let dist = StandardNormalMat { nrows: 2, ncols: 3 };
        let mut mat: Mat<f32> = dist.sample(rng);
        for col in mat.col_iter_mut() {
            for x in col.iter_mut() {
                let y = bf16::from_f32(*x);
                *x =  y.to_f32();
            }
        }
        let bytes = store_f32_matrix_to_bf16(mat.as_ref());
        let nums = bytes.as_slice().chunks(2)
            .map(|bytes| {
                let bytes: &[u8; 2] = bytes.try_into().unwrap();
                let pod: &u16 = bytemuck::cast_ref(bytes);
                let num = bf16::from_bits(*pod);
                num.to_f32()
            }).collect::<Vec<_>>();
        let new_mat: MatRef<f32> = faer::mat::from_row_major_slice(nums.as_slice(), 2, 3);
        assert_eq!(mat.as_ref(), new_mat);
    }
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
