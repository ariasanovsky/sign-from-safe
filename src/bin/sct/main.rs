use std::{borrow::Cow, collections::HashMap, fs::File, path::PathBuf, sync::Mutex};

use chrono::Utc;
use clap::Parser;
use cuts_v2::{inplace_sct::CutHelper, sct::{Sct, SctMut, SctRef}};
use faer::{dyn_stack::{GlobalPodBuffer, PodStack, StackReq}, linalg::temp_mat_req};
use memmap2::MmapOptions;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use reborrow::{Reborrow, ReborrowMut};
use safetensors::{serialize_to_file, Dtype, SafeTensors};
use equator::assert;
use linya::{Bar, Progress};
use signtensors::safetensors::{load_matrix_f32, load_matrix_f64};

#[derive(Debug, Parser)]
#[command(name = "SCT")]
// #[command(version = "0.1.0")]
#[command(about = "Approximates matrices with signed cuts", long_about = None)]
struct Args {
    /// Input directory containing `safetensors`
    #[arg(short = 'i')]
    input: std::path::PathBuf,
    /// Output directory for new `safetensors`
    #[arg(short = 'o')]
    output: std::path::PathBuf,
    /// The number of rows in our target matrices
    #[arg(short = 'm')]
    nrows: usize,
    /// The number of columns in our target matrices
    #[arg(short = 'n')]
    ncols: usize,
    /// The width
    #[arg(short = 'w')]
    width: usize,
    /// The number of blocks to use in delayed matmul
    #[arg(short = 'b')]
    block_size: usize,
    /// The number of tensors to process in parallel
    #[arg(short = 't')]
    threads: Option<usize>,
    // /// The number of threads to use in mat{vec/mat} (default: to 0)
    // #[arg(short = 'P')]
    // par: Option<usize>,
}

fn main() -> eyre::Result<()> {
    let Args { input, output, nrows, ncols, width, block_size, threads } = Args::try_parse()?;
    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?
    }
    let buffers = input
        .as_path()
        .read_dir()?
        .filter_map(|entry| {
            let file = entry.unwrap().path();
            match file.extension() {
                Some(ext) if ext.eq("safetensors") => {
                    let stem = file.file_stem().unwrap().to_os_string();
                    let file = File::open(file).unwrap();
                    let buffer = unsafe { MmapOptions::new().map(&file) }.unwrap();
                    Some((stem, buffer))
                },
                _ => None,
            }
        }).collect::<Vec<_>>();
    let safetensors = buffers.iter().map(|(stem, buffer)| {
        let tensors = SafeTensors::deserialize(buffer).unwrap();
        (stem, tensors)
    }).collect::<Vec<_>>();
    let tensors = safetensors.iter().flat_map(|(stem, tensors)| {
        // tensors.tensors().into_iter().filter_map(move |(name, view)| {
        //     if view.shape() == &[nrows, ncols] {
        //         let data = view.data();
        //         assert!(all(
        //             view.dtype() == Dtype::BF16,
        //             data.len() == nrows * ncols * 2,
        //         ));
        //         let data: &[i16] = bytemuck::cast_slice(data);
        //         let data: &[half::bf16] = unsafe { core::mem::transmute(data) };
        //         Some((stem.as_os_str(), name, data))
        //     } else {
        //         None
        //     }
        // })
        tensors.tensors().into_iter().map(move |(name, view)| {
            (stem.to_string_lossy(), name, view)
        })
    }).collect::<Vec<_>>();
    let progress = Mutex::new(Progress::new());
    let now = Utc::now().to_rfc3339();
    std::fs::create_dir_all(&output).unwrap();
    let outfiles = tensors.into_iter().filter_map(|(stem, name, view)| {
        if view.shape() != &[nrows, ncols] {
            return None
        }
        let mat = load_matrix_f32(view)?;
        Some((stem, name, mat))
    }).take(1).map(|(stem, name, mat)| {
        let ref mut rng = rand::thread_rng();
        let outfile = format!("{now}.{stem}.{name}.width.{width}.safetensors");
        let outpath = std::env::temp_dir().join(&outfile);
        // panic!("{outpath:?}");
        let label: Cow<_> = if let Some(t) = rayon::current_thread_index() {
            format!("[{t}] {name}").into()
        } else {
            name.into()
        };
        let bar: Bar = progress.lock().unwrap().bar(width, label);
        let checkpoint = (width / 100).max(1);
        let mut sct_block: Sct = Sct::new(nrows, ncols, block_size);
        let mut sct_full: Sct = Sct::new(nrows, ncols, width);
        let mut how_full = 0usize;
        let mut mem = GlobalPodBuffer::new(
            StackReq::new::<u64>(Ord::max(nrows, ncols))
                .and(temp_mat_req::<f32>(block_size, 1).unwrap()),
        );
        let mut stack = PodStack::new(&mut mem);
        let mut remainder_norm = mat.squared_norm_l2();
        let mut two_remainder: faer::Mat<f32> = faer::scale(2.0f32) * &mat;
        let mut two_remainder_transposed = faer::scale(2.0f32) * mat.transpose();
        let mut helper = CutHelper::new(two_remainder.as_ref(), two_remainder_transposed.as_ref());
        for w in 0..width {
            if how_full == block_size {
                {
                    let SctRef { s, c, t } = sct_block.as_ref();
                    let two_remainder = cuts_v2::MatMut::from_faer(two_remainder.as_mut());
                    cuts_v2::bitmagic::matmul::mat_tmat_f32(
                        two_remainder,
                        s.rb(),
                        t.rb(),
                        c.as_slice(),
                    );
                }
                two_remainder_transposed
                    .as_mut()
                    .copy_from(two_remainder.transpose());
                remainder_norm = two_remainder.squared_norm_l2() / 4.0;

                let SctRef { s, c, t } = sct_block.as_ref();
                for k in 0..block_size {
                    sct_full.s[nrows.div_ceil(64) * sct_full.how_full..][..nrows.div_ceil(64)].copy_from_slice(s.rb().storage().col_as_slice(k));
                    sct_full.t[ncols.div_ceil(64) * sct_full.how_full..][..ncols.div_ceil(64)].copy_from_slice(t.rb().storage().col_as_slice(k));
                    sct_full.c[sct_full.how_full] = -c[k] / 2.0;
                    sct_full.how_full += 1;
                }
                how_full = 0;
            }
            how_full += 1;
            let SctMut { mut s, c, mut t } = sct_block.as_mut();
            dbg!(two_remainder.shape(), two_remainder_transposed.shape());
            let cut = helper.cut_mat(
                two_remainder.as_ref(),
                two_remainder_transposed.as_ref(),
                s.rb_mut().split_at_col_mut(how_full).0,
                c.get_mut(..how_full),
                t.rb_mut().split_at_col_mut(how_full).0,
                rng,
                usize::MAX,
                stack.rb_mut(),
            );
            remainder_norm -= (cut * cut) / (nrows * ncols) as f32;
            match w {
                _ if w == width - 1 => {
                    {
                        let SctMut { mut s, c, mut t } = sct_block.as_mut();
                        let two_remainder = cuts_v2::MatMut::from_faer(two_remainder.as_mut());
                        cuts_v2::bitmagic::matmul::mat_tmat_f32(
                            two_remainder,
                            s.rb_mut().split_at_col_mut(how_full).0.rb(),
                            t.rb_mut().split_at_col_mut(how_full).0.rb(),
                            &c.as_slice()[..how_full],
                        );
                    }
                    two_remainder_transposed
                        .as_mut()
                        .copy_from(two_remainder.transpose());
                    let SctRef { s, c, t } = sct_block.as_ref();
                    for k in 0..how_full {
                        sct_full.s[nrows.div_ceil(64) * sct_full.how_full..][..nrows.div_ceil(64)].copy_from_slice(s.rb().storage().col_as_slice(k));
                        sct_full.t[ncols.div_ceil(64) * sct_full.how_full..][..ncols.div_ceil(64)].copy_from_slice(t.rb().storage().col_as_slice(k));
                        sct_full.c[sct_full.how_full] = -c[k] / 2.0;
                        sct_full.how_full += 1;
                        dbg!(sct_full.how_full);
                    }
                    progress.lock().unwrap().set_and_draw(&bar, w + 1);
                    serialize_to_file(sct_full.views(), &None, &outpath).unwrap();
                }
                _ if w % checkpoint == 0 => {
                    progress.lock().unwrap().set_and_draw(&bar, w + 1);
                }
                _ => {},
            }
        }
        return (stem, outpath)
        
    }).collect::<Vec<_>>();
    let mut mapped_files: HashMap<Cow<str>, Vec<PathBuf>> = Default::default();
    for (stem, outfile) in outfiles {
        mapped_files
            .entry(stem)
            .or_default()
            .push(outfile);
    }
    Ok(())
}
