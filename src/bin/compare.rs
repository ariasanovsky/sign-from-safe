use std::collections::HashSet;

use equator::assert;
use clap::Parser;
use signtensors::{safetensors::load_matrix_f32, MappedSafetensors};

#[derive(Debug, Parser)]
#[command(name = "SCT")]
#[command(about = "Compares new matrices to old", long_about = None)]
struct Args {
    /// Input directory containing old `safetensors`
    #[arg(short = 'o')]
    old_input: std::path::PathBuf,
    /// Input directory containing new `safetensors`
    #[arg(short = 'n')]
    new_input: std::path::PathBuf,
}

fn main() -> eyre::Result<()> {
    let Args {
        old_input,
        new_input,
    } = Args::try_parse()?;
    let old_buffers = MappedSafetensors::new(old_input);
    let old_safetensors = old_buffers.deserialize();
    let old_tensors = old_safetensors.tensors();
    let new_buffers = MappedSafetensors::new(new_input);
    let new_safetensors = new_buffers.deserialize();
    let mut new_tensors = new_safetensors.tensors();
    for (stem, (header, metadata, tensors)) in old_tensors {
        let (new_header, new_metadata, mut new_tensors) = new_tensors.remove(&stem).unwrap();
        println!("stem: {stem}");
        assert!(all(
            header == new_header,
            metadata.metadata() == new_metadata.metadata(),
            metadata.tensors().keys().collect::<HashSet<_>>() == new_metadata.tensors().keys().collect::<HashSet<_>>(),
        ));
        for (name, view) in tensors {
            let new_view = new_tensors.remove(&name).unwrap();
            let mat = load_matrix_f32(&view);
            let new_mat = load_matrix_f32(&new_view);
            match mat {
                Some(mat) => {
                    println!("{name}");
                    let norm = mat.norm_l2();
                    let new_mat = new_mat.unwrap();
                    let new_norm = new_mat.norm_l2();
                    if !norm.is_finite() || !new_norm.is_finite() {
                        println!("old defects: {:?}\nnew defects: {:?}", count_defects(mat.as_ref()), count_defects(new_mat.as_ref()))
                    } else {
                        let diff = (mat - new_mat).norm_l2();
                        println!("error: {}/{} = {}", diff, norm, diff / norm);
                    }
                },
                None => assert!(new_mat.is_none()),
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
struct Defects {
    nans: usize,
    infs: usize,
    norm: f32,
}

fn count_defects(mat: faer::MatRef<f32>) -> Defects {
    let mut nans = 0;
    let mut infs = 0;
    let mut norm = 0.0;
    for c in mat.col_iter() {
        for &x in c.iter() {
            match x {
                x if x.is_nan() => nans += 1,
                x if x.is_infinite() => infs += 1,
                x => norm += x * x,
            }
        }
    }
    Defects { nans, infs, norm: norm.sqrt() }
}