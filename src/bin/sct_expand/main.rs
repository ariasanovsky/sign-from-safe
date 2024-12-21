use std::collections::HashMap;

use clap::Parser;
use faer::Mat;
use safetensors::{serialize_to_file, tensor::TensorView, Dtype, View};
use sign_from_safe::{
    safetensors::{load_matrix_f32, load_slice, store_f32_matrix_to_bf16},
    MappedSafetensors,
};
use signtensors::{sct::SctRef, SignMatRef};

#[derive(Debug, Parser)]
#[command(name = "SCT")]
#[command(about = "Approximates matrices with signed cuts", long_about = None)]
struct Args {
    /// Input directory containing `safetensors`
    #[arg(short = 'i')]
    old_input: std::path::PathBuf,
    /// Input directory containing `safetensors`
    #[arg(short = 'n')]
    new_input: std::path::PathBuf,
    /// Output directory for new `safetensors`
    #[arg(short = 'o')]
    output: std::path::PathBuf,
    /// The width
    #[arg(short = 'c')]
    compression_rate: f64,
}

fn main() -> eyre::Result<()> {
    let Args {
        old_input,
        new_input,
        output,
        compression_rate,
    } = Args::try_parse()?;

    std::fs::create_dir_all(&output)?;

    // let old_buffers = old_input
    //     .as_path()
    //     .read_dir()?
    //     .filter_map(|entry| {
    //         let file = entry.unwrap().path();
    //         match file.extension() {
    //             Some(ext) if ext.eq("safetensors") => {
    //                 let stem = file.file_stem().unwrap().to_os_string();
    //                 let file = File::open(file).unwrap();
    //                 let buffer = unsafe { MmapOptions::new().map(&file) }.unwrap();
    //                 Some((stem, buffer))
    //             }
    //             _ => None,
    //         }
    //     })
    //     .collect::<Vec<_>>();
    // let old_safetensors = old_buffers
    //     .iter()
    //     .map(|(stem, buffer)| {
    //         let tensors = SafeTensors::deserialize(buffer).unwrap();
    //         (stem, SafeTensors::read_metadata(buffer).unwrap(), tensors)
    //     })
    //     .collect::<Vec<_>>();
    // let old_tensors = old_safetensors
    //     .iter()
    //     .map(|(stem, metadata, tensors)| {
    //         (
    //             stem.to_string_lossy().to_string(),
    //             (
    //                 metadata,
    //                 tensors.tensors().into_iter().collect::<HashMap<_, _>>(),
    //             ),
    //         )
    //     })
    //     .collect::<HashMap<_, _>>();
    let old_buffers = MappedSafetensors::new(old_input);
    let old_safetensors = old_buffers.deserialize();
    let old_tensors = old_safetensors.tensors();
    // let new_buffers = new_input
    //     .as_path()
    //     .read_dir()?
    //     .filter_map(|entry| {
    //         let file = entry.unwrap().path();
    //         match file.extension() {
    //             Some(ext) if ext.eq("safetensors") => {
    //                 let stem = file.file_stem().unwrap().to_os_string();
    //                 let file = File::open(file).unwrap();
    //                 let buffer = unsafe { MmapOptions::new().map(&file) }.unwrap();
    //                 Some((stem, buffer))
    //             }
    //             _ => None,
    //         }
    //     })
    //     .collect::<Vec<_>>();
    // let new_safetensors = new_buffers
    //     .iter()
    //     .map(|(stem, buffer)| {
    //         let tensors = SafeTensors::deserialize(buffer).unwrap();
    //         (stem, tensors)
    //     })
    //     .collect::<Vec<_>>();
    // let mut new_tensors = new_safetensors
    //     .iter()
    //     .map(|(stem, tensors)| {
    //         (
    //             stem.to_string_lossy().to_string(),
    //             tensors.tensors().into_iter().collect::<HashMap<_, _>>(),
    //         )
    //     })
    //     .collect::<HashMap<_, _>>();
    let new_buffers = MappedSafetensors::new(new_input);
    let new_safetensors = new_buffers.deserialize();
    let new_tensors = new_safetensors.tensors();
    let mut new_tensors = new_tensors
        .into_iter()
        .map(|(stem, (_, _, tensors))| {
            (stem.clone(), tensors.into_iter().collect::<HashMap<_, _>>())
        })
        .collect::<HashMap<_, _>>();
    for (stem, (_, metadata, old_tensors)) in old_tensors {
        let mut new_tensors = new_tensors.remove(&stem).unwrap();
        let iter = old_tensors.into_iter().map(|(name, original)| {
            struct SctTensor<'a> {
                old: TensorView<'a>,
                sct: Option<SctRef<'a>>,
            }

            let tensor = if new_tensors.contains_key(&format!("s.{name}")) {
                impl View for SctTensor<'_> {
                    fn dtype(&self) -> Dtype {
                        Dtype::BF16
                    }

                    fn shape(&self) -> &[usize] {
                        self.old.shape()
                    }

                    fn data(&self) -> std::borrow::Cow<[u8]> {
                        if let Some(SctRef { s, c, t }) = self.sct {
                            let original = load_matrix_f32(&self.old).unwrap();
                            let (nrows, ncols) = original.shape();
                            let mut new = Mat::<f32>::zeros(nrows, ncols);
                            signtensors::bitmagic::matmul::mat_tmat_f32(
                                signtensors::MatMut::from_faer(new.as_mut()),
                                s,
                                t,
                                c,
                            );
                            let remainder_norm = (&new - &original).norm_l2();
                            let init_norm = &original.norm_l2();
                            dbg!(remainder_norm / init_norm);

                            store_f32_matrix_to_bf16(new.as_ref()).into()
                        } else {
                            self.old.data().into()
                        }
                    }

                    fn data_len(&self) -> usize {
                        self.old.data_len()
                    }
                }

                let &[nrows, ncols] = original.shape() else {
                    panic!()
                };

                let s =
                    load_slice::<u64>(new_tensors.remove(&format!("s.{name}")).unwrap()).unwrap();
                let t =
                    load_slice::<u64>(new_tensors.remove(&format!("t.{name}")).unwrap()).unwrap();
                let c =
                    load_slice::<f32>(new_tensors.remove(&format!("c.{name}")).unwrap()).unwrap();

                let width = (((nrows * ncols * 2) as f64
                    / (4 + nrows.next_multiple_of(64) / 8 + ncols.next_multiple_of(64) / 8) as f64)
                    * compression_rate) as usize;
                let width = Ord::max(8, width);
                assert!(width <= c.len());

                let s = SignMatRef::from_storage(
                    signtensors::MatRef::from_col_major_slice(
                        s,
                        nrows.div_ceil(64),
                        width,
                        nrows.div_ceil(64),
                    ),
                    nrows,
                );
                let t = SignMatRef::from_storage(
                    signtensors::MatRef::from_col_major_slice(
                        t,
                        ncols.div_ceil(64),
                        width,
                        ncols.div_ceil(64),
                    ),
                    ncols,
                );
                let c = &c[..width];

                SctTensor {
                    old: original,
                    sct: Some(SctRef { s, c, t }),
                }
            } else {
                SctTensor {
                    old: original,
                    sct: None,
                }
            };
            (name, tensor)
        });

        serialize_to_file(
            iter,
            metadata.metadata(),
            &output.join(&format!("{stem}.safetensors")),
        )
        .unwrap();
    }

    Ok(())
}
