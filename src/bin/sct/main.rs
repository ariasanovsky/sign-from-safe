use clap::Parser;
use cuts_v2::{
    inplace_sct::CutHelper,
    sct::{Sct, SctMut, SctRef},
};
use faer::{
    dyn_stack::{GlobalPodBuffer, PodStack, StackReq},
    linalg::temp_mat_req,
};
use itertools::Itertools;
use linya::{Bar, Progress};
use memmap2::MmapOptions;
use rand::prelude::*;
use rayon::prelude::*;
use reborrow::{Reborrow, ReborrowMut};
use safetensors::{serialize_to_file, SafeTensors};
use signtensors::safetensors::load_matrix_f32;
use std::{
    collections::HashMap,
    fs::File,
    hash::{DefaultHasher, Hasher},
    sync::Mutex,
};

#[derive(Debug, Parser)]
#[command(name = "SCT")]
#[command(about = "Approximates matrices with signed cuts", long_about = None)]
struct Args {
    /// Input directory containing `safetensors`
    #[arg(short = 'i')]
    input: std::path::PathBuf,
    /// Output directory for new `safetensors`
    #[arg(short = 'o')]
    output: std::path::PathBuf,
    /// The width
    #[arg(short = 'c')]
    compression_rate: f64,
    /// The number of blocks to use in delayed matmul
    #[arg(short = 'b')]
    block_size: usize,
    /// The number of tensors to process in parallel
    #[arg(short = 't')]
    threads: Option<usize>,
}

fn main() -> eyre::Result<()> {
    let Args {
        input,
        output,
        compression_rate,
        block_size,
        threads,
    } = Args::try_parse()?;
    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(Ord::min(threads, num_cpus::get()))
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
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    let safetensors = buffers
        .iter()
        .map(|(stem, buffer)| {
            let tensors = SafeTensors::deserialize(buffer).unwrap();
            (stem, tensors)
        })
        .collect::<Vec<_>>();
    let tensors = safetensors
        .iter()
        .flat_map(|(stem, tensors)| {
            tensors
                .tensors()
                .into_iter()
                .map(move |(name, view)| (stem.to_string_lossy(), name, view))
        })
        .collect::<Vec<_>>();
    let progress = Mutex::new(Progress::new());

    let tmp = tempdir::TempDir::new_in("./target", "safetensors")?;
    let (outfiles, mut err): (Vec<_>, Vec<_>) = tensors
        .into_iter()
        .filter(|(_, _, view)| view.shape().len() == 2)
        .sorted_by_key(|(_, _, view)| std::cmp::Reverse(view.shape().iter().product::<usize>()))
        .chunk_by(|(_, _, view)| view.shape().iter().product::<usize>())
        .into_iter()
        .flat_map(|(_, chunk)| {
            chunk
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(stem, name, view)| {
                    let state = &mut DefaultHasher::new();
                    state.write(name.as_bytes());
                    let rng = &mut StdRng::seed_from_u64(state.finish());

                    let mat = load_matrix_f32(&view).unwrap();
                    let (nrows, ncols) = mat.shape();

                    // bf16 size = nrows * ncols * 2
                    // sct size  = (nrows.next_multiple_of(64) / 8 * width) + (ncols.next_multiple_of(64) / 8 * width) + 4 * width
                    //           = width * (4 + nrows.next_multiple_of(64) / 8 + ncols.next_multiple_of(64) / 8)
                    // we want
                    // sct_size ~ bf16_size * compression_rate
                    // width ~ bf16_size / (4 + nrows.next_multiple_of(64) / 8 + ncols.next_multiple_of(64) / 8) * compression_rate
                    let width = (((nrows * ncols * 2) as f64
                        / (4 + nrows.next_multiple_of(64) / 8 + ncols.next_multiple_of(64) / 8)
                            as f64)
                        * compression_rate) as usize;
                    let width = Ord::max(8, width);

                    let outfile = format!("{stem}.{name}.safetensors");
                    let outpath = tmp.path().join(&outfile);
                    let thread_id = rayon::current_thread_index().unwrap_or(0);
                    let label_prefix = format!("[{thread_id}]",);
                    let label = format!("{label_prefix}[{nrows}×{ncols}] {name}");
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
                    let init_norm = mat.squared_norm_l2();
                    let mut remainder_norm = init_norm;
                    let mut two_remainder: faer::Mat<f32> = faer::scale(2.0f32) * &mat;
                    let mut two_remainder_transposed = faer::scale(2.0f32) * mat.transpose();
                    let mut helper =
                        CutHelper::new(two_remainder.as_ref(), two_remainder_transposed.as_ref());
                    for w in 0..width {
                        if how_full == block_size {
                            {
                                let SctRef { s, c, t } = sct_block.as_ref();
                                let two_remainder =
                                    cuts_v2::MatMut::from_faer(two_remainder.as_mut());
                                cuts_v2::bitmagic::matmul::mat_tmat_f32(
                                    two_remainder,
                                    s.rb(),
                                    t.rb(),
                                    c,
                                );
                            }
                            two_remainder_transposed
                                .as_mut()
                                .copy_from(two_remainder.transpose());
                            remainder_norm = two_remainder.squared_norm_l2() / 4.0;

                            let SctRef { s, c, t } = sct_block.as_ref();
                            for k in 0..block_size {
                                sct_full.s[nrows.div_ceil(64) * sct_full.how_full..]
                                    [..nrows.div_ceil(64)]
                                    .copy_from_slice(s.rb().storage().col_as_slice(k));
                                sct_full.t[ncols.div_ceil(64) * sct_full.how_full..]
                                    [..ncols.div_ceil(64)]
                                    .copy_from_slice(t.rb().storage().col_as_slice(k));
                                sct_full.c[sct_full.how_full] = -c[k] / 2.0;
                                sct_full.how_full += 1;
                            }
                            how_full = 0;
                        }
                        how_full += 1;
                        let SctMut { mut s, c, mut t } = sct_block.as_mut();
                        let cut = helper.cut_mat(
                            two_remainder.as_ref(),
                            two_remainder_transposed.as_ref(),
                            s.rb_mut().split_at_col_mut(how_full).0,
                            faer::col::from_slice_mut(&mut c[..how_full]),
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
                                    let two_remainder =
                                        cuts_v2::MatMut::from_faer(two_remainder.as_mut());
                                    cuts_v2::bitmagic::matmul::mat_tmat_f32(
                                        two_remainder,
                                        s.rb_mut().split_at_col_mut(how_full).0.rb(),
                                        t.rb_mut().split_at_col_mut(how_full).0.rb(),
                                        &c[..how_full],
                                    );
                                }
                                two_remainder_transposed
                                    .as_mut()
                                    .copy_from(two_remainder.transpose());
                                let SctRef { s, c, t } = sct_block.as_ref();
                                for k in 0..how_full {
                                    sct_full.s[nrows.div_ceil(64) * sct_full.how_full..]
                                        [..nrows.div_ceil(64)]
                                        .copy_from_slice(s.rb().storage().col_as_slice(k));
                                    sct_full.t[ncols.div_ceil(64) * sct_full.how_full..]
                                        [..ncols.div_ceil(64)]
                                        .copy_from_slice(t.rb().storage().col_as_slice(k));
                                    sct_full.c[sct_full.how_full] = -c[k] / 2.0;
                                    sct_full.how_full += 1;
                                }
                                progress.lock().unwrap().set_and_draw(&bar, w + 1);
                                serialize_to_file(sct_full.views(), &None, &outpath).unwrap();
                            }
                            _ if w % checkpoint == 0 => {
                                progress.lock().unwrap().set_and_draw(&bar, w + 1);
                            }
                            _ => {}
                        }
                    }
                    (
                        format!("{stem}"),
                        (name, outpath, f32::sqrt(remainder_norm / init_norm)),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|(stem, (name, outpath, err))| {
            let file = File::open(outpath).unwrap();
            let buffer = unsafe { MmapOptions::new().map(&file) }.unwrap();
            ((stem, (name, buffer)), err)
        })
        .unzip();
    err.sort_unstable_by(f32::total_cmp);
    let max_err = err.last().copied().unwrap_or(0.0);
    println!("max error: {max_err}");

    let outfiles = outfiles
        .iter()
        .map(|(stem, (name, mmap))| (&**stem, (&**name, SafeTensors::deserialize(mmap).unwrap())))
        .collect::<Vec<_>>();

    let mut mapped_files: HashMap<&str, Vec<(&str, &SafeTensors)>> = HashMap::new();
    for (stem, (name, outfile)) in &outfiles {
        mapped_files.entry(stem).or_default().push((name, outfile));
    }

    std::fs::create_dir_all(&output)?;
    for (&stem, outfile) in &mapped_files {
        serialize_to_file(
            outfile.iter().flat_map(|(name, tensors)| {
                tensors
                    .tensors()
                    .into_iter()
                    .map(move |(prefix, tensor)| (format!("{prefix}.{name}"), tensor))
            }),
            &None,
            &output.join(&format!("{stem}.safetensors")),
        )?;
    }

    Ok(())
}
