//! Shared loading for the vision and audio towers: tensors under a prefix
//! are read from the same safetensors file as the text tower, packed at the
//! model's precision, and kept in their own packed cache (`<text cache>.<tag>`)
//! so a text-only load never reads or packs them.

use rt_backend::{Hardware, Packed};
use rt_core::{check_shape, Result, ResultContext as Context};
use rt_kernels::{bf16_bits_to_f32, for_ranges, rows_mut};
use rt_onednn::{
    apply_cache, cache_write_failed, file_stamp, pinned_dnnl, CacheMeta, CacheOptions, CacheWriter,
    RawTensor,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::{pack_bf16, Mapping, Precision};

/// Borrowed BF16 views of the safetensors file while a tower packs.
pub(crate) struct Views<'m> {
    map: &'m Mapping,
}

impl<'m> Views<'m> {
    pub fn shape(&self, name: &str) -> Result<&'m [usize]> {
        Ok(&self
            .map
            .header
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?
            .shape)
    }

    pub fn bf16(&self, name: &str, shape: &[usize]) -> Result<&'m [u16]> {
        let got = self.shape(name)?;
        check_shape!(
            got == shape,
            "EG2 tensor {name}: shape {got:?}, expected {shape:?}"
        );
        self.map.view_bf16(name)
    }

    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        Ok(self
            .bf16(name, shape)?
            .iter()
            .map(|&b| bf16_bits_to_f32(b))
            .collect())
    }

    pub fn release(&self, name: &str) {
        self.map.advise_unused(name);
    }
}

/// Collects packed matrices and small f32 tensors, mirroring both into the
/// tower's packed cache when one is being written.
pub(crate) struct Packer<'b, B: Hardware> {
    backend: &'b B,
    precision: Precision,
    cache: Option<CacheWriter>,
    linears: HashMap<String, B::Packed>,
    raw: Vec<RawTensor>,
}

impl<B: Hardware> Packer<'_, B> {
    /// Pack `[n, k]` BF16 rows (`y = x @ W^T`).
    pub fn pack(&mut self, key: &str, n: usize, k: usize, src: &[u16]) -> Result<()> {
        check_shape!(src.len() == n * k, "pack {key}: {} != {n}x{k}", src.len());
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

pub(crate) struct TowerParts<B: Hardware> {
    pub linears: HashMap<String, B::Packed>,
    pub raw: HashMap<String, Vec<f32>>,
    pub source: &'static str,
}

impl<B: Hardware> TowerParts<B> {
    pub fn linear(&self, name: &str) -> Result<&B::Packed> {
        self.linears
            .get(name)
            .with_context(|| format!("EG2 tower linear {name}"))
    }

    pub fn raw(&self, name: &str, len: usize) -> Result<&[f32]> {
        let v = self
            .raw
            .get(name)
            .with_context(|| format!("EG2 tower tensor {name}"))?;
        check_shape!(
            v.len() == len,
            "EG2 tower tensor {name}: {} values, expected {len}",
            v.len()
        );
        Ok(v)
    }

    /// `x @ W^T` for `[rows, k]` activations at the tower's precision.
    pub fn matmul(&self, hw: &B, precision: Precision, x: &[f32], name: &str) -> Result<Vec<f32>> {
        let w = self.linear(name)?;
        check_shape!(x.len() % w.k() == 0, "EG2 tower linear {name}: input width");
        let mut y = hw.buffer(rt_core::checked_size(
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
        check_shape!(
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
pub(crate) fn load_tower<B: Hardware>(
    src: &TowerSource<'_>,
    backend: &B,
    precision: Precision,
    tag: &str,
    build: impl FnOnce(&Views<'_>, &mut Packer<'_, B>) -> Result<()>,
) -> Result<TowerParts<B>> {
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
                source: "packed-cache",
            });
        }
    }
    let stamp = file_stamp(src.st_path).with_context(|| src.st_path.display().to_string())?;
    let map = Mapping::load(src.st_path)?;
    let cache = if src.cache.enabled {
        CacheWriter::try_create(
            &cache_file,
            &CacheMeta {
                source_sha256: src.sha.to_string(),
                source_size: stamp.size,
                source_mtime_secs: stamp.mtime_secs,
                dnnl: pinned_dnnl(),
                cpu_isa: backend.cpu_isa(),
            },
        )
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
    build(&Views { map: &map }, &mut packer)?;
    let Packer {
        cache,
        linears,
        raw,
        ..
    } = packer;
    if let Some(w) = cache {
        if let Err(e) = w.finish(&raw) {
            cache_write_failed(&cache_file, &e);
        }
    }
    Ok(TowerParts {
        linears,
        raw: raw.into_iter().map(|t| (t.name, t.data)).collect(),
        source: "safetensors",
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
