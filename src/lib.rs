use std::{collections::HashMap, ffi::OsString, fs::File, path::Path};

use memmap2::{Mmap, MmapOptions};
use ::safetensors::{tensor::{Metadata, TensorView}, SafeTensors};

pub mod safetensors;
#[cfg(test)]
mod test;

pub struct MappedSafetensors(Vec<(OsString, Mmap)>);

impl MappedSafetensors {
    pub fn new<P: AsRef<Path>>(p: P) -> Self {
        Self(p
            .as_ref()
            .read_dir()
            .unwrap()
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
            }).collect())
    }

    pub fn deserialize(&self) -> DeserializedSafetensors {
        DeserializedSafetensors(self.0
            .iter()
            .map(|(stem, buffer)| {
                let tensors = SafeTensors::deserialize(buffer).unwrap();
                let (header_size, metadata) = SafeTensors::read_metadata(buffer).unwrap();
                (stem, header_size, metadata, tensors)
            }).collect())
    }
}

pub struct DeserializedSafetensors<'a>(Vec<(&'a OsString, usize, Metadata, SafeTensors<'a>)>);

impl DeserializedSafetensors<'_> {
    pub fn tensors(&self) -> DeserializedTensors {
        self.0
            .iter()
            .map(|(stem, header_size, metadata, tensors)| {
                (
                    stem.to_string_lossy().to_string(),
                    (
                        *header_size,
                        metadata,
                        tensors.tensors().into_iter().collect::<HashMap<_, _>>(),
                    ),
                )
            })
            .collect::<HashMap<_, _>>()
        
    }
}

pub type DeserializedTensors<'a> = HashMap<String, (usize, &'a Metadata, HashMap<String, TensorView<'a>>)>;
