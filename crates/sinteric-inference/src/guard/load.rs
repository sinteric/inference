//! Loading the encoder: the packed-weight cache in front of the mapped
//! safetensors file, and the layers assembled from both.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::cpu::kernels::f32_slice_to_bf16;
use crate::cpu::onednn::{
    apply_cache, cache_path_with, cache_write_failed, pinned_dnnl, CacheMeta, CacheOptions,
    CacheWriter, FileStamp, OnednnCpu, PackedEntry, PackedWeight, RawTensor,
};
use crate::guard::config::EncoderConfig;
use crate::guard::encoder::{Layer, Precision};
use crate::tensor_file::TensorFile;
use crate::{check_shape, Result, ResultContext as Context};

/// Packed matrices, by cache key.
fn key(layer: usize, part: &str) -> String {
    format!("layers.{layer}.{part}")
}

fn pack_weight(
    backend: &OnednnCpu,
    precision: Precision,
    name: &str,
    n: usize,
    k: usize,
    src: &[f32],
    cache: &mut Option<CacheWriter>,
) -> Result<PackedWeight> {
    let w = if precision == Precision::Bf16 {
        let mut bits = crate::try_vec::<u16>(src.len())?;
        f32_slice_to_bf16(src, &mut bits)?;
        backend.pack_bf16(n, k, &bits)?
    } else {
        backend.pack(n, k, src)?
    };
    let failed = if let Some(c) = cache.as_mut() {
        let mut bytes = crate::try_vec::<u8>(w.bytes())?;
        let wrote = backend.unpack_into(&w, &mut bytes).is_ok()
            && c.write_packed(PackedEntry {
                name: name.to_string(),
                n,
                k,
                bytes,
                layout: w.layout_tag(),
            })
            .is_ok();
        if !wrote {
            cache_write_failed(c.dest(), "write interrupted");
        }
        !wrote
    } else {
        false
    };
    if failed {
        *cache = None;
    }
    Ok(w)
}

/// Every tensor of the encoder and its injection head that is not packed:
/// the norms, `head.*` and the classifier. Other tensors in the file (heads a
/// fine-tune added) are left to [`TensorFile::view`].
pub(super) fn small_tensor_names(layers: usize) -> Vec<String> {
    let mut names = vec![
        "model.embeddings.norm.weight".to_string(),
        "model.final_norm.weight".into(),
        "head.dense.weight".into(),
        "head.norm.weight".into(),
        "classifier.weight".into(),
        "classifier.bias".into(),
    ];
    for i in 0..layers {
        if i > 0 {
            names.push(format!("model.layers.{i}.attn_norm.weight"));
        }
        names.push(format!("model.layers.{i}.mlp_norm.weight"));
    }
    names
}

/// The packed-weight cache file for `st_path`. BF16 packs a different blocked
/// layout, so each precision has its own file.
pub(super) fn cache_file_for(
    st_path: &Path,
    cpu_isa: u64,
    cache: &CacheOptions,
    precision: Precision,
) -> PathBuf {
    let cache_file = cache_path_with(st_path, cpu_isa, cache.file.as_deref());
    if precision != Precision::Bf16 {
        return cache_file;
    }
    let mut name = cache_file.into_os_string();
    name.push(".bf16");
    name.into()
}

/// What the packed-weight cache records about its source, so a changed file,
/// oneDNN or CPU is a miss.
pub(super) fn cache_meta(sha256: String, stamp: FileStamp, cpu_isa: u64) -> CacheMeta {
    CacheMeta {
        source_sha256: sha256,
        source_size: stamp.size,
        source_mtime_secs: stamp.mtime_secs,
        dnnl: pinned_dnnl(),
        cpu_isa,
    }
}

/// The small tensors from the packed-weight cache, with every packed linear
/// restored into `linears`; `None` on a miss, with `linears` left empty.
pub(super) fn read_cache(
    backend: &OnednnCpu,
    precision: Precision,
    cache_file: &Path,
    sha256: &str,
    stamp: FileStamp,
    linears: &mut HashMap<String, PackedWeight>,
) -> Option<Vec<RawTensor>> {
    let applied = apply_cache(
        cache_file,
        sha256,
        Some(stamp),
        pinned_dnnl(),
        backend.cpu_isa(),
        |name, n, k, bytes, layout| {
            let w = if precision == Precision::Bf16 {
                backend.restore_packed_bf16_checked(n, k, bytes, layout)?
            } else {
                backend.restore_packed_checked(n, k, bytes, layout)?
            };
            linears.insert(name.to_string(), w);
            Ok(())
        },
    );
    if let Ok(Some(tensors)) = applied {
        return Some(tensors);
    }
    linears.clear();
    None
}

/// Pack every linear of the safetensors file into `linears`, writing each to
/// the cache through `writer` (dropped on a failed write), and copy out the
/// small tensors.
pub(super) fn pack_from_file(
    backend: &OnednnCpu,
    precision: Precision,
    cfg: &EncoderConfig,
    file: &TensorFile,
    mut writer: Option<CacheWriter>,
    linears: &mut HashMap<String, PackedWeight>,
) -> Result<(Vec<RawTensor>, Option<CacheWriter>)> {
    let (h, inter) = (cfg.hidden, cfg.intermediate);
    // Pack rows `range` of the `[total, k]` tensor under `name`.
    let mut pack = |name: String,
                    tensor: &str,
                    total: usize,
                    k: usize,
                    range: std::ops::Range<usize>|
     -> Result<()> {
        let all = file.view::<f32>(tensor, &[total, k])?;
        let src = &all[range.start * k..range.end * k];
        let w = pack_weight(backend, precision, &name, range.len(), k, src, &mut writer)?;
        linears.insert(name, w);
        Ok(())
    };
    for i in 0..cfg.layers {
        let base = format!("model.layers.{i}");
        let wqkv = format!("{base}.attn.Wqkv.weight");
        pack(key(i, "wqkv"), &wqkv, 3 * h, h, 0..3 * h)?;
        pack(key(i, "wo"), &format!("{base}.attn.Wo.weight"), h, h, 0..h)?;
        let wi = format!("{base}.mlp.Wi.weight");
        pack(key(i, "wi_left"), &wi, 2 * inter, h, 0..inter)?;
        pack(key(i, "wi_right"), &wi, 2 * inter, h, inter..2 * inter)?;
        let wo = format!("{base}.mlp.Wo.weight");
        pack(key(i, "mlp_wo"), &wo, h, inter, 0..h)?;
        for part in ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"] {
            file.release(&format!("{base}.{part}.weight"));
        }
    }
    let mut raw = Vec::new();
    for name in small_tensor_names(cfg.layers) {
        let shape = file.index().shape(&name)?;
        raw.push(RawTensor {
            data: file.view::<f32>(&name, shape)?.to_vec(),
            name,
        });
    }
    Ok((raw, writer))
}

/// The norm `name`, `width` wide, taken out of `small`.
pub(super) fn take_norm(
    small: &mut HashMap<String, Vec<f32>>,
    name: &str,
    width: usize,
) -> Result<Vec<f32>> {
    let v = small
        .remove(name)
        .with_context(|| format!("missing tensor {name}"))?;
    check_shape!(
        v.len() == width,
        "norm {name}: width {}, expected {width}",
        v.len()
    );
    Ok(v)
}

/// The encoder layers, taken out of the packed `linears` and the norms in
/// `small`, each checked against the config's shapes.
pub(super) fn take_layers(
    cfg: &EncoderConfig,
    linears: &mut HashMap<String, PackedWeight>,
    small: &mut HashMap<String, Vec<f32>>,
) -> Result<Vec<Layer>> {
    let (h, inter) = (cfg.hidden, cfg.intermediate);
    let mut layers = Vec::with_capacity(cfg.layers);
    for i in 0..cfg.layers {
        let base = format!("model.layers.{i}");
        let mut take = |part: &str, n: usize, k: usize| -> Result<PackedWeight> {
            let w = linears
                .remove(&key(i, part))
                .with_context(|| format!("missing packed weight {}", key(i, part)))?;
            check_shape!(
                w.n() == n && w.k() == k,
                "packed {}: [{}, {}], expected [{n}, {k}]",
                key(i, part),
                w.n(),
                w.k()
            );
            Ok(w)
        };
        layers.push(Layer {
            attn_norm: if i == 0 {
                None
            } else {
                Some(take_norm(small, &format!("{base}.attn_norm.weight"), h)?)
            },
            wqkv: take("wqkv", 3 * h, h)?,
            wo: take("wo", h, h)?,
            mlp_norm: take_norm(small, &format!("{base}.mlp_norm.weight"), h)?,
            wi_left: take("wi_left", inter, h)?,
            wi_right: take("wi_right", inter, h)?,
            mlp_wo: take("mlp_wo", h, inter)?,
        });
    }
    Ok(layers)
}
