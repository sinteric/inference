//! Shared loading for the vision and audio towers: tensors under a prefix
//! are read from the same safetensors file as the text tower, packed at the
//! model's precision, and kept in their own packed cache (`<text cache>.<tag>`)
//! so a text-only load never reads or packs them.

use crate::cpu::kernels::{bf16_bits_to_f32, for_ranges, rows_mut};
use crate::cpu::onednn::{
    apply_cache, file_stamp, pinned_dnnl, CacheMeta, CacheOptions, CacheWriter, OnednnCpu,
    PackedWeight, RawTensor,
};
use crate::cpu::runtime::{CacheWrite, WeightSource};
use crate::{check_files, check_shape, Result, ResultContext as Context};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::load::pack_bf16;
use super::Precision;
use crate::tensor_file::TensorFile;

/// Borrowed BF16 views of the safetensors file while a tower packs.
pub(crate) struct Views<'m> {
    file: &'m TensorFile,
}

impl<'m> Views<'m> {
    pub fn bf16(&self, name: &str, shape: &[usize]) -> Result<&'m [u16]> {
        self.file.view(name, shape)
    }

    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        Ok(self
            .bf16(name, shape)?
            .iter()
            .map(|&b| bf16_bits_to_f32(b))
            .collect())
    }

    pub fn release(&self, name: &str) {
        self.file.release(name);
    }
}

/// Collects packed matrices and small f32 tensors, mirroring both into the
/// tower's packed cache when one is being written.
pub(crate) struct Packer<'b> {
    backend: &'b OnednnCpu,
    precision: Precision,
    cache: Option<CacheWriter>,
    linears: HashMap<String, PackedWeight>,
    raw: Vec<RawTensor>,
}

impl Packer<'_> {
    /// Pack `[n, k]` BF16 rows (`y = x @ W^T`).
    pub fn pack(&mut self, key: &str, n: usize, k: usize, src: &[u16]) -> Result<()> {
        check_files!(src.len() == n * k, "pack {key}: {} != {n}x{k}", src.len());
        let w = pack_bf16(
            self.backend,
            self.precision,
            key,
            n,
            k,
            src,
            &mut self.cache,
        )?;
        self.linears.insert(key.to_string(), w);
        Ok(())
    }

    pub fn raw(&mut self, name: &str, data: Vec<f32>) {
        self.raw.push(RawTensor {
            name: name.to_string(),
            data,
        });
    }
}

pub(crate) struct TowerParts {
    pub linears: HashMap<String, PackedWeight>,
    pub raw: HashMap<String, Vec<f32>>,
    pub source: WeightSource,
}

impl TowerParts {
    pub fn linear(&self, name: &str) -> Result<&PackedWeight> {
        self.linears
            .get(name)
            .with_context(|| format!("EG2 tower linear {name}"))
    }

    pub fn raw(&self, name: &str, len: usize) -> Result<&[f32]> {
        let v = self
            .raw
            .get(name)
            .with_context(|| format!("EG2 tower tensor {name}"))?;
        check_files!(
            v.len() == len,
            "EG2 tower tensor {name}: {} values, expected {len}",
            v.len()
        );
        Ok(v)
    }

    /// `x @ W^T` for `[rows, k]` activations at the tower's precision.
    pub fn matmul(
        &self,
        hw: &OnednnCpu,
        precision: Precision,
        x: &[f32],
        name: &str,
    ) -> Result<Vec<f32>> {
        let w = self.linear(name)?;
        check_shape!(x.len() % w.k() == 0, "EG2 tower linear {name}: input width");
        let mut y = hw.buffer(crate::checked_size(
            "EG2 tower activations",
            &[x.len() / w.k(), w.n()],
        )?)?;
        if precision == Precision::Fp32 {
            hw.linear(x, w, &mut y, false)?;
        } else {
            hw.linear_bf16(x, w, &mut y, false)?;
        }
        Ok(y)
    }

    pub fn check_linear(&self, name: &str, n: usize, k: usize) -> Result<()> {
        let w = self.linear(name)?;
        check_files!(
            w.n() == n && w.k() == k,
            "EG2 tower linear {name}: [{}, {}], expected [{n}, {k}]",
            w.n(),
            w.k()
        );
        Ok(())
    }
}

/// Where the tower identifies its packed cache and source weights.
pub(crate) struct TowerSource<'a> {
    pub st_path: &'a Path,
    pub text_cache: &'a Path,
    pub sha: &'a str,
    pub cache: &'a CacheOptions,
}

fn tower_cache(text_cache: &Path, tag: &str) -> PathBuf {
    let mut name = text_cache.as_os_str().to_os_string();
    name.push(format!(".{tag}"));
    name.into()
}

/// Restore the tower from its packed cache, or run `build` over the
/// safetensors views (writing the cache as it packs).
pub(crate) fn load_tower(
    src: &TowerSource<'_>,
    backend: &OnednnCpu,
    precision: Precision,
    tag: &str,
    build: impl FnOnce(&Views<'_>, &mut Packer<'_>) -> Result<()>,
) -> Result<TowerParts> {
    let cache_file = tower_cache(src.text_cache, tag);
    if src.cache.enabled && !src.cache.verify_weights {
        let stamp = file_stamp(src.st_path).ok();
        let mut linears = HashMap::new();
        let applied = apply_cache(
            &cache_file,
            src.sha,
            stamp,
            pinned_dnnl(),
            backend.cpu_isa(),
            |name, n, k, bytes, layout| {
                let weight = if precision == Precision::Fp32 {
                    backend.restore_packed_bf16_f32_checked(n, k, bytes, layout)?
                } else {
                    backend.restore_packed_bf16_checked(n, k, bytes, layout)?
                };
                linears.insert(name.to_string(), weight);
                Ok(())
            },
        );
        if let Ok(Some(raw)) = applied {
            return Ok(TowerParts {
                linears,
                raw: raw.into_iter().map(|t| (t.name, t.data)).collect(),
                source: WeightSource::Cache { path: cache_file },
            });
        }
    }
    let stamp = file_stamp(src.st_path).with_context(|| src.st_path.display().to_string())?;
    let file = TensorFile::open(src.st_path)?;
    let cache = if src.cache.enabled {
        Some(CacheWriter::start(
            &cache_file,
            &CacheMeta {
                source_sha256: src.sha.to_string(),
                source_size: stamp.size,
                source_mtime_secs: stamp.mtime_secs,
                dnnl: pinned_dnnl(),
                cpu_isa: backend.cpu_isa(),
            },
        ))
    } else {
        None
    };
    let mut packer = Packer {
        backend,
        precision,
        cache,
        linears: HashMap::new(),
        raw: Vec::new(),
    };
    build(&Views { file: &file }, &mut packer)?;
    let Packer {
        cache,
        linears,
        raw,
        ..
    } = packer;
    let cache = match cache {
        Some(w) => w.close(&raw),
        None => CacheWrite::Off,
    };
    Ok(TowerParts {
        linears,
        raw: raw.into_iter().map(|t| (t.name, t.data)).collect(),
        source: WeightSource::Safetensors { cache },
    })
}

/// Run `f(row, out_row)` over the `width`-wide rows of `out` in parallel.
pub(crate) fn par_rows(
    threads: usize,
    out: &mut [f32],
    width: usize,
    grain: usize,
    f: impl Fn(usize, &mut [f32]) + Sync,
) {
    let rows = out.len() / width;
    let p = out.as_mut_ptr() as usize;
    for_ranges(threads, rows, grain, |lo, hi| {
        // SAFETY: workers get disjoint row ranges of `out`.
        let dst = unsafe { rows_mut(p as *mut f32, lo, hi, width) };
        for (r, row) in (lo..hi).zip(dst.chunks_exact_mut(width)) {
            f(r, row);
        }
    });
}
