//! EmbeddingGemma 2 text tower over [`rt_backend::Hardware`].
//!
//! [`Precision::Bf16`] (default): matmuls are BF16×BF16 with FP32
//! accumulation; RMSNorm, softmax, RoPE, the residual stream, pooling and L2
//! stay in FP32. [`Precision::Fp32`] runs its own graph ([`fp32`]) with f32
//! activations into every matmul. Token embeddings are a zero-copy mmap of
//! the safetensors BF16 table.

use memmap2::{Mmap, MmapOptions};
use rt_backend::{Hardware, Packed};
use rt_core::{check_shape, Result, ResultContext as Context};
use rt_kernels::{
    add_rmsnorm, bf16_bits_to_f32, f32_slice_to_bf16_nthreads, for_ranges, for_row_tiles,
    rmsnorm_rope_to_bf16, rmsnorm_to_bf16, row_mut, row_ref, scale_inplace, DefaultRope,
    SQRT_512_BF16,
};
use rt_onednn::{
    apply_cache, cache_path_with, cache_write_failed, file_stamp, pinned_dnnl, CacheMeta,
    CacheOptions, CacheWriter, PackedEntry, RawTensor,
};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::config::{
    head_dim, is_global, n_kv, ATTN_SCALE, EMBED_DIM, EMB_NAME, GLOBAL_THETA, HIDDEN, LOCAL_THETA,
    MAX_SEQ, N_LAYERS, N_QUERY_HEADS, PLE_WIDTH, RMS_EPS, SLIDING_WINDOW, VOCAB,
};
use crate::frontend::Batch;

mod audio;
mod fp32;
mod tower;
mod vision;

use tower::TowerSource;

pub use rt_runtime::Precision;

/// Precisions EG2 implements; the first is the default. Weights are the BF16
/// checkpoint either way: [`Precision::Fp32`] widens them exactly (BF16
/// storage), so only activations differ.
pub const PRECISIONS: &[Precision] = &[Precision::Bf16, Precision::Fp32];

/// Stats label of a precision EG2 runs.
fn precision_label(p: Precision) -> &'static str {
    if p == Precision::Fp32 {
        "FP32"
    } else {
        "BF16-matmul/FP32-stream"
    }
}

fn page_size() -> usize {
    let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if p > 0 {
        p as usize
    } else {
        4096
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
struct Info {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

struct Mapping {
    data: Mmap,
    offset: usize,
    header: HashMap<String, Info>,
}

impl Mapping {
    fn load(path: &Path) -> Result<Self> {
        let f = File::open(path).map_err(|e| rt_core::Error::from(e).at(path))?;
        let data = unsafe { MmapOptions::new().map(&f) }?;
        check_shape!(data.len() >= 8, "short safetensors");
        let len = u64::from_le_bytes(data[..8].try_into()?) as usize;
        let offset = 8usize.checked_add(len).context("header size")?;
        check_shape!(
            offset <= data.len() && len <= 128 * 1024 * 1024,
            "{}: invalid safetensors header length {len}, file length {}",
            path.display(),
            data.len()
        );
        let value: Value = serde_json::from_slice(&data[8..offset])
            .map_err(|e| rt_core::Error::from(e).at(path))?;
        let mut header: HashMap<String, Info> = HashMap::new();
        for (n, v) in value.as_object().context("tensor header")? {
            if n != "__metadata__" {
                header.insert(n.clone(), serde_json::from_value(v.clone())?);
            }
        }
        for (name, info) in &header {
            let width = match info.dtype.as_str() {
                "BF16" | "F16" => 2,
                "F32" | "I32" | "U32" => 4,
                "F64" | "I64" | "U64" => 8,
                "I8" | "U8" | "BOOL" => 1,
                dtype => {
                    return Err(rt_core::Error::Shape(format!(
                        "tensor {name}: unsupported dtype {dtype}"
                    ))
                    .at(path))
                }
            };
            let count =
                rt_core::checked_size("tensor shape", &info.shape).map_err(|e| e.at(path))?;
            let bytes =
                rt_core::checked_size("tensor bytes", &[count, width]).map_err(|e| e.at(path))?;
            let [a, b] = info.data_offsets;
            check_shape!(
                a <= b && b <= data.len() - offset && b - a == bytes,
                "{}: tensor {name} shape {:?}, offsets {:?} outside data length {}",
                path.display(),
                info.shape,
                info.data_offsets,
                data.len() - offset
            );
        }
        Ok(Self {
            data,
            offset,
            header,
        })
    }

    fn view_bf16(&self, n: &str) -> Result<&[u16]> {
        let i = self
            .header
            .get(n)
            .with_context(|| format!("missing tensor {n}"))?;
        check_shape!(i.dtype == "BF16", "BF16 required: {n} ({})", i.dtype);
        let count = i
            .shape
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .context("tensor size overflow")?;
        let [a, b] = i.data_offsets;
        let start = self.offset.checked_add(a).context("tensor offset")?;
        check_shape!(
            a <= b
                && b <= self.data.len() - self.offset
                && b - a == count.checked_mul(2).context("tensor bytes")?,
            "tensor bounds {n}"
        );
        check_shape!(
            (self.data.as_ptr() as usize + start) % 2 == 0 && cfg!(target_endian = "little"),
            "BF16 mapped alignment/endian"
        );
        Ok(unsafe { std::slice::from_raw_parts(self.data.as_ptr().add(start).cast(), count) })
    }

    fn advise_unused(&self, n: &str) {
        let Some(i) = self.header.get(n) else {
            return;
        };
        let start = self.offset.saturating_add(i.data_offsets[0]);
        let len = i.data_offsets[1].saturating_sub(i.data_offsets[0]);
        if start >= self.data.len() || len == 0 {
            return;
        }
        let len = len.min(self.data.len() - start);
        let page = page_size();
        let base = self.data.as_ptr() as usize;
        let abs = base + start;
        let aligned = abs.saturating_add(page - 1) & !(page - 1);
        let end = (abs + len) & !(page - 1);
        if end > aligned {
            unsafe {
                libc::madvise(
                    aligned as *mut libc::c_void,
                    end - aligned,
                    libc::MADV_DONTNEED,
                );
            }
        }
    }

    fn embedding_span(&self) -> Result<(usize, usize)> {
        let i = self
            .header
            .get(EMB_NAME)
            .with_context(|| format!("missing tensor {EMB_NAME}"))?;
        check_shape!(
            i.dtype == "BF16" && i.shape == vec![VOCAB, HIDDEN],
            "embedding shape"
        );
        let [a, b] = i.data_offsets;
        let start = self.offset.checked_add(a).context("embedding offset")?;
        let bytes = VOCAB
            .checked_mul(HIDDEN)
            .and_then(|n| n.checked_mul(2))
            .context("embedding bytes")?;
        check_shape!(
            b.saturating_sub(a) == bytes
                && bytes <= self.data.len().saturating_sub(start)
                && start <= self.data.len()
                && a <= b,
            "embedding bounds"
        );
        Ok((start, VOCAB * HIDDEN))
    }

    fn into_embeddings(self) -> Result<EmbeddingTable> {
        let (start, count) = self.embedding_span()?;
        check_shape!(
            (self.data.as_ptr() as usize + start) % 2 == 0 && cfg!(target_endian = "little"),
            "BF16 mapped alignment/endian"
        );
        Ok(EmbeddingTable {
            map: self.data,
            start,
            count,
        })
    }
}

/// Zero-copy BF16 token table. Only gathered rows become RSS.
struct EmbeddingTable {
    map: Mmap,
    start: usize,
    count: usize,
}

impl EmbeddingTable {
    fn from_safetensors(path: &Path) -> Result<Self> {
        Self::open(path, EMB_NAME, &[VOCAB, HIDDEN])
    }

    /// Zero-copy map of one BF16 tensor of `shape`.
    fn open(path: &Path, name: &str, shape: &[usize]) -> Result<Self> {
        let mut f = File::open(path).map_err(|e| rt_core::Error::from(e).at(path))?;
        let file_len = f.metadata()?.len() as usize;
        let mut lenb = [0u8; 8];
        f.read_exact(&mut lenb)?;
        let hlen = u64::from_le_bytes(lenb) as usize;
        check_shape!(
            hlen <= file_len.saturating_sub(8),
            "{}: safetensors header length {hlen} exceeds file {file_len}",
            path.display()
        );
        check_shape!(
            hlen <= 128 * 1024 * 1024,
            "{}: safetensors header exceeds 128 MiB",
            path.display()
        );
        let mut header_bytes = rt_core::try_vec::<u8>(hlen)?;
        f.read_exact(&mut header_bytes)?;
        let value: Value = serde_json::from_slice(&header_bytes)?;
        let info: Info = serde_json::from_value(
            value
                .get(name)
                .cloned()
                .with_context(|| format!("{name} missing from safetensors header"))?,
        )?;
        check_shape!(
            info.dtype == "BF16" && info.shape == shape,
            "{name}: {} {:?}, expected BF16 {shape:?}",
            info.dtype,
            info.shape
        );
        let count = rt_core::checked_size("mapped tensor", shape)?;
        let [a, b] = info.data_offsets;
        let data_start = 8usize
            .checked_add(hlen)
            .and_then(|o| o.checked_add(a))
            .context("tensor offset")?;
        let nbytes = count.checked_mul(2).context("tensor bytes")?;
        check_shape!(
            b.saturating_sub(a) == nbytes
                && nbytes <= file_len.saturating_sub(data_start)
                && data_start <= file_len
                && a <= b,
            "{name} bounds"
        );
        let page = page_size();
        let aligned = data_start & !(page - 1);
        let delta = data_start - aligned;
        let map = unsafe {
            MmapOptions::new()
                .offset(aligned as u64)
                .len(delta + nbytes)
                .map(&f)?
        };
        check_shape!(
            (map.as_ptr() as usize + delta) % 2 == 0 && cfg!(target_endian = "little"),
            "BF16 mapped alignment/endian"
        );
        Ok(Self {
            map,
            start: delta,
            count,
        })
    }

    fn as_slice(&self) -> &[u16] {
        unsafe { std::slice::from_raw_parts(self.map.as_ptr().add(self.start).cast(), self.count) }
    }
}

struct Linear<B: Hardware> {
    weight: B::Packed,
}

/// Soft tokens replacing placeholder rows of one sequence: `(first
/// position, [n, HIDDEN] rows)`, in sequence order.
pub(crate) type Splice = Vec<(usize, Vec<f32>)>;

pub struct Model<B: Hardware> {
    embeddings: EmbeddingTable,
    vision: Option<vision::VisionTower<B>>,
    audio: Option<audio::AudioTower<B>>,
    backend: Arc<B>,
    linears: HashMap<String, Linear<B>>,
    norms: HashMap<String, Vec<f32>>,
    scalars: Vec<f32>,
    local_rope: DefaultRope,
    global_rope: DefaultRope,
    tile: usize,
    max: usize,
    precision: Precision,
    load_source: &'static str,
    load_profile: BTreeMap<String, f64>,
}

fn expected_weight_pin(manifest: Option<&Path>) -> Result<(String, u64)> {
    let m = match manifest {
        Some(path) => rt_assets::Manifest::load(path)?,
        None => rt_assets::Manifest::parse(crate::ASSET_MANIFEST)?,
    };
    let f = m
        .file("model.safetensors")
        .ok_or_else(|| rt_core::Error::Asset("model.safetensors not in manifest".into()))?;
    Ok((f.sha256.clone(), f.size))
}

fn skip_tensor(name: &str) -> bool {
    name.starts_with("vision_tower.")
        || name.starts_with("audio_tower.")
        || name.starts_with("embed_vision.")
        || name.starts_with("embed_audio.")
}

/// `q_proj`/`k_proj`/`v_proj` → (layer, 0/1/2) so they pack as one matmul.
fn qkv_part(name: &str) -> Option<(usize, usize)> {
    let rest = name.strip_prefix("language_model.layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let idx = match rest {
        "self_attn.q_proj.weight" => 0,
        "self_attn.k_proj.weight" => 1,
        "self_attn.v_proj.weight" => 2,
        _ => return None,
    };
    Some((layer, idx))
}

fn pack_bf16<B: Hardware>(
    backend: &B,
    precision: Precision,
    key: &str,
    n: usize,
    k: usize,
    src: &[u16],
    cache: &mut Option<CacheWriter>,
) -> Result<B::Packed> {
    let w = if precision == Precision::Fp32 {
        backend.pack_bf16_f32(n, k, src)?
    } else {
        backend.pack_bf16(n, k, src)?
    };
    let failed = if let Some(c) = cache.as_mut() {
        let mut bytes = rt_core::try_vec::<u8>(w.bytes())?;
        let wrote = backend.unpack_into(&w, &mut bytes).is_ok()
            && c.write_packed(PackedEntry {
                name: key.to_string(),
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

impl<B: Hardware> Model<B> {
    pub fn load(path: &Path, backend: Arc<B>, tile: usize) -> Result<Self> {
        Self::load_with(path, backend, tile, &CacheOptions::default())
    }

    pub fn load_with(
        path: &Path,
        backend: Arc<B>,
        tile: usize,
        cache: &CacheOptions,
    ) -> Result<Self> {
        Self::load_with_precision(path, backend, tile, cache, Precision::Bf16)
    }

    pub fn load_with_precision(
        path: &Path,
        backend: Arc<B>,
        tile: usize,
        cache: &CacheOptions,
        precision: Precision,
    ) -> Result<Self> {
        Self::load_impl(path, backend, tile, cache, precision).map_err(|e| e.at(path))
    }

    fn load_impl(
        path: &Path,
        backend: Arc<B>,
        tile: usize,
        cache: &CacheOptions,
        precision: Precision,
    ) -> Result<Self> {
        check_shape!(tile > 0, "tile");
        let precision = Precision::resolve(Some(precision), "EmbeddingGemma 2", PRECISIONS)?;
        let st_path = path.join("model.safetensors");
        let (sha, pinned_size) = expected_weight_pin(cache.manifest.as_deref())?;
        let mut cache_file = cache_path_with(&st_path, backend.cpu_isa(), cache.file.as_deref());
        if precision == Precision::Fp32 {
            // FP32 packs a different blocked layout; keep both caches.
            let mut name = cache_file.into_os_string();
            name.push(".fp32");
            cache_file = name.into();
        }
        let mut load_profile = BTreeMap::new();

        if cache.enabled && !cache.verify_weights {
            let t = Instant::now();
            let stamp = file_stamp(&st_path).ok();
            let mut linears = HashMap::new();
            let applied = apply_cache(
                &cache_file,
                &sha,
                stamp,
                pinned_dnnl(),
                backend.cpu_isa(),
                |name, n, k, bytes, layout| {
                    let weight = if precision == Precision::Fp32 {
                        backend.restore_packed_bf16_f32_checked(n, k, bytes, layout)?
                    } else {
                        backend.restore_packed_bf16_checked(n, k, bytes, layout)?
                    };
                    linears.insert(name.to_string(), Linear { weight });
                    Ok(())
                },
            );
            load_profile.insert("cache_read_s".into(), t.elapsed().as_secs_f64());
            if let Ok(Some(raw)) = applied {
                if let (Ok(parts), Ok(embeddings)) = (
                    assemble_raw(raw),
                    EmbeddingTable::from_safetensors(&st_path),
                ) {
                    return Self::finish(
                        backend,
                        linears,
                        parts,
                        embeddings,
                        tile,
                        precision,
                        "packed-cache",
                        load_profile,
                    );
                }
            }
        }

        let stamp = file_stamp(&st_path).with_context(|| st_path.display().to_string())?;
        check_shape!(
            stamp.size == pinned_size,
            "safetensors size {} != pinned {pinned_size}",
            stamp.size
        );
        if cache.verify_weights {
            let t = Instant::now();
            rt_assets::verify_sha256_mmap(&st_path, &sha)?;
            load_profile.insert("verify_s".into(), t.elapsed().as_secs_f64());
        }

        let t = Instant::now();
        let mapping = Mapping::load(&st_path)?;
        load_profile.insert("mmap_s".into(), t.elapsed().as_secs_f64());
        let t = Instant::now();
        let mut linears = HashMap::new();
        let mut raw_out: Vec<RawTensor> = Vec::new();
        let mut cache_w = if cache.enabled {
            CacheWriter::try_create(
                &cache_file,
                &CacheMeta {
                    source_sha256: sha,
                    source_size: stamp.size,
                    source_mtime_secs: stamp.mtime_secs,
                    dnnl: pinned_dnnl(),
                    cpu_isa: backend.cpu_isa(),
                },
            )
        } else {
            None
        };
        let mut names = mapping.header.keys().cloned().collect::<Vec<_>>();
        names.sort();
        let mut qkv: HashMap<usize, [Option<(usize, usize, Vec<u16>)>; 3]> = HashMap::new();
        for n in names {
            if n == EMB_NAME || skip_tensor(&n) || !n.starts_with("language_model.") {
                continue;
            }
            let shape = &mapping.header[&n].shape;
            if n.ends_with(".weight") && shape.len() == 2 {
                let bits = mapping.view_bf16(&n)?;
                if let Some((layer, part)) = qkv_part(&n) {
                    let slot = qkv.entry(layer).or_insert([None, None, None]);
                    slot[part] = Some((shape[0], shape[1], bits.to_vec()));
                    mapping.advise_unused(&n);
                    continue;
                }
                let key = n
                    .strip_suffix(".weight")
                    .context("weight tensor suffix")?
                    .to_string();
                let weight = pack_bf16(
                    &*backend,
                    precision,
                    &key,
                    shape[0],
                    shape[1],
                    bits,
                    &mut cache_w,
                )?;
                linears.insert(key, Linear { weight });
                mapping.advise_unused(&n);
            } else if shape.len() == 1 {
                let bits = mapping.view_bf16(&n)?;
                let mut data = vec![0f32; bits.len()];
                for (d, &b) in data.iter_mut().zip(bits) {
                    *d = bf16_bits_to_f32(b);
                }
                raw_out.push(RawTensor {
                    name: n.clone(),
                    data,
                });
                mapping.advise_unused(&n);
            } else {
                rt_core::invalid_shape!("unexpected text-tower tensor {n} shape {shape:?}");
            }
        }
        for (layer, parts) in qkv {
            let Some((qn, qk, q)) = parts[0].as_ref() else {
                rt_core::invalid_shape!("missing q_proj for layer {layer}");
            };
            let Some((kn, kk, k)) = parts[1].as_ref() else {
                rt_core::invalid_shape!("missing k_proj for layer {layer}");
            };
            let Some((vn, vk, v)) = parts[2].as_ref() else {
                rt_core::invalid_shape!("missing v_proj for layer {layer}");
            };
            check_shape!(*qk == *kk && *kk == *vk, "qkv input width layer {layer}");
            let mut cat = Vec::with_capacity(q.len() + k.len() + v.len());
            cat.extend_from_slice(q);
            cat.extend_from_slice(k);
            cat.extend_from_slice(v);
            let key = format!("language_model.layers.{layer}.self_attn.qkv_proj");
            let weight = pack_bf16(
                &*backend,
                precision,
                &key,
                qn + kn + vn,
                *qk,
                &cat,
                &mut cache_w,
            )?;
            linears.insert(key, Linear { weight });
        }
        load_profile.insert("pack_s".into(), t.elapsed().as_secs_f64());
        let embeddings = mapping.into_embeddings()?;
        if let Some(w) = cache_w {
            let t = Instant::now();
            if let Err(e) = w.finish(&raw_out) {
                cache_write_failed(&cache_file, &e);
            }
            load_profile.insert("cache_write_s".into(), t.elapsed().as_secs_f64());
        }
        let parts = assemble_raw(raw_out)?;
        Self::finish(
            backend,
            linears,
            parts,
            embeddings,
            tile,
            precision,
            "safetensors",
            load_profile,
        )
    }

    fn finish(
        backend: Arc<B>,
        linears: HashMap<String, Linear<B>>,
        parts: Assembled,
        embeddings: EmbeddingTable,
        tile: usize,
        precision: Precision,
        load_source: &'static str,
        load_profile: BTreeMap<String, f64>,
    ) -> Result<Self> {
        validate_weights(&linears, &parts.norms)?;
        check_shape!(parts.scalars.len() == N_LAYERS, "layer_scalar count");
        Ok(Self {
            embeddings,
            vision: None,
            audio: None,
            backend,
            linears,
            norms: parts.norms,
            scalars: parts.scalars,
            local_rope: DefaultRope::new(head_dim(0), MAX_SEQ, LOCAL_THETA)?,
            global_rope: DefaultRope::new(head_dim(5), MAX_SEQ, GLOBAL_THETA)?,
            tile,
            max: MAX_SEQ,
            precision,
            load_source,
            load_profile,
        })
    }

    fn linear(&self, x: &[f32], name: &str) -> Result<Vec<f32>> {
        let l = self
            .linears
            .get(name)
            .with_context(|| format!("linear {name}"))?;
        check_shape!(x.len() % l.weight.k() == 0, "linear matrix {name}");
        let mut y = self.backend.buffer(rt_core::checked_size(
            "activation elements",
            &[x.len() / l.weight.k(), l.weight.n()],
        )?)?;
        self.backend.linear_bf16(x, &l.weight, &mut y, false)?;
        Ok(y)
    }

    fn linear_src(&self, x: &[u16], name: &str) -> Result<Vec<f32>> {
        let l = self
            .linears
            .get(name)
            .with_context(|| format!("linear {name}"))?;
        check_shape!(x.len() % l.weight.k() == 0, "linear matrix {name}");
        let mut y = self.backend.buffer(rt_core::checked_size(
            "activation elements",
            &[x.len() / l.weight.k(), l.weight.n()],
        )?)?;
        self.backend.linear_bf16_src(x, &l.weight, &mut y, false)?;
        Ok(y)
    }

    fn to_bf16(&self, x: &[f32]) -> Result<Vec<u16>> {
        let mut y = self.backend.buffer_u16(x.len())?;
        f32_slice_to_bf16_nthreads(x, &mut y, self.backend.threads())?;
        Ok(y)
    }

    fn rms(&self, x: &[f32], name: &str, width: usize) -> Result<Vec<f32>> {
        let scale = self
            .norms
            .get(name)
            .with_context(|| format!("norm {name}"))?;
        check_shape!(x.len() % width == 0 && scale.len() == width, "rms {name}");
        self.backend
            .rmsnorm(x.len() / width, width, x, Some(scale), RMS_EPS)
    }

    fn rms_bf16(&self, x: &[f32], name: &str, width: usize) -> Result<Vec<u16>> {
        let scale = self
            .norms
            .get(name)
            .with_context(|| format!("norm {name}"))?;
        check_shape!(x.len() % width == 0 && scale.len() == width, "rms {name}");
        let mut y = self.backend.buffer_u16(x.len())?;
        let t = Instant::now();
        rmsnorm_to_bf16(
            x,
            width,
            Some(scale),
            RMS_EPS,
            &mut y,
            self.backend.threads(),
        )?;
        self.backend
            .add_category_s("glue", t.elapsed().as_secs_f64());
        Ok(y)
    }

    /// Residual rows rounded to the AMX tile / 4-tile block (shape+ISA
    /// only, including `M<16→16`). Pad rows are zeroed so glue and GEMM
    /// never see stale pool data.
    fn work_rows(&self, live: usize) -> usize {
        self.backend.residual_rows(live)
    }

    /// Scaled token embeddings in `m_pad >= b * s` rows; pad rows are zero.
    /// `splices` (empty, or one per sequence) overwrite placeholder rows
    /// with unscaled soft tokens.
    fn embed(
        &self,
        batch: &Batch,
        b: usize,
        s: usize,
        m_pad: usize,
        splices: &[Splice],
    ) -> Result<Vec<f32>> {
        check_shape!(self.embeddings.count == VOCAB * HIDDEN, "embedding shape");
        let ntok = b * s;
        check_shape!(m_pad >= ntok, "embedding rows");
        let mut x = self.backend.buffer(rt_core::checked_size(
            "activation elements",
            &[m_pad, HIDDEN],
        )?)?;
        let x_p = x.as_mut_ptr() as usize;
        let ids: Vec<u32> = batch.input_ids.iter().flatten().copied().collect();
        let table = self.embeddings.as_slice().as_ptr() as usize;
        for_row_tiles(
            self.backend.threads(),
            ntok,
            self.backend.residual_row_tile(ntok),
            8,
            |lo, hi| {
                for t in lo..hi {
                    let id = ids[t] as usize;
                    let src = unsafe { row_ref(table as *const u16, id, HIDDEN) };
                    let dst = unsafe { row_mut(x_p as *mut f32, t, HIDDEN) };
                    embed_row(src, dst, SQRT_512_BF16);
                }
            },
        )?;
        if m_pad > ntok {
            x[ntok * HIDDEN..].fill(0.0);
        }
        check_shape!(splices.is_empty() || splices.len() == b, "splice rows");
        for (bi, splice) in splices.iter().enumerate() {
            for (pos, rows) in splice {
                let n = rows.len() / HIDDEN;
                check_shape!(
                    rows.len() % HIDDEN == 0 && pos + n <= s,
                    "soft tokens {pos}+{n} outside sequence of {s}"
                );
                x[(bi * s + pos) * HIDDEN..][..rows.len()].copy_from_slice(rows);
            }
        }
        Ok(x)
    }

    fn to_heads(&self, src: &[f32], b: usize, s: usize, nh: usize, d: usize) -> Result<Vec<f32>> {
        self.to_heads_strided(src, b, s, nh, d, nh * d, 0)
    }

    fn to_heads_strided(
        &self,
        src: &[f32],
        b: usize,
        s: usize,
        nh: usize,
        d: usize,
        stride: usize,
        off: usize,
    ) -> Result<Vec<f32>> {
        let mut out = self.backend.buffer(rt_core::checked_size(
            "activation elements",
            &[b, nh, s, d],
        )?)?;
        let ntok = b * s;
        let src_p = src.as_ptr() as usize;
        let dst_p = out.as_mut_ptr() as usize;
        for_ranges(self.backend.threads(), ntok, 8, |lo, hi| {
            for idx in lo..hi {
                let bi = idx / s;
                let p = idx % s;
                for h in 0..nh {
                    let so = (bi * s + p) * stride + off + h * d;
                    let dst = ((bi * nh + h) * s + p) * d;
                    let src = unsafe { row_ref(src_p as *const f32, so / d, d) };
                    let dst = unsafe { row_mut(dst_p as *mut f32, dst / d, d) };
                    dst.copy_from_slice(src);
                }
            }
        });
        Ok(out)
    }

    fn split_qkv_heads(
        &self,
        fused: &[f32],
        b: usize,
        s: usize,
        nq: usize,
        nkv: usize,
        d: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let qn = nq * d;
        let kn = nkv * d;
        let stride = qn + kn + kn;
        check_shape!(
            fused.len() >= b * s * stride && fused.len().is_multiple_of(stride),
            "fused qkv width"
        );
        // Projection rows include the residual pad; keys and queries must
        // contain only live tokens so pad rows cannot enter attention.
        let fused = &fused[..b * s * stride];
        let mut q = self.backend.buffer(rt_core::checked_size(
            "activation elements",
            &[b, nq, s, d],
        )?)?;
        let mut k = self.backend.buffer(rt_core::checked_size(
            "activation elements",
            &[b, nkv, s, d],
        )?)?;
        let mut v = self.backend.buffer(rt_core::checked_size(
            "activation elements",
            &[b, nkv, s, d],
        )?)?;
        let ntok = b * s;
        let src_p = fused.as_ptr() as usize;
        let q_p = q.as_mut_ptr() as usize;
        let k_p = k.as_mut_ptr() as usize;
        let v_p = v.as_mut_ptr() as usize;
        for_ranges(self.backend.threads(), ntok, 8, |lo, hi| {
            for idx in lo..hi {
                let bi = idx / s;
                let p = idx % s;
                let base = (bi * s + p) * stride;
                for h in 0..nq {
                    let so = base + h * d;
                    let dst = ((bi * nq + h) * s + p) * d;
                    let src = unsafe { row_ref(src_p as *const f32, so / d, d) };
                    let dst = unsafe { row_mut(q_p as *mut f32, dst / d, d) };
                    dst.copy_from_slice(src);
                }
                for h in 0..nkv {
                    let ks = base + qn + h * d;
                    let vs = base + qn + kn + h * d;
                    let dst = ((bi * nkv + h) * s + p) * d;
                    let ksrc = unsafe { row_ref(src_p as *const f32, ks / d, d) };
                    let vsrc = unsafe { row_ref(src_p as *const f32, vs / d, d) };
                    let kdst = unsafe { row_mut(k_p as *mut f32, dst / d, d) };
                    let vdst = unsafe { row_mut(v_p as *mut f32, dst / d, d) };
                    kdst.copy_from_slice(ksrc);
                    vdst.copy_from_slice(vsrc);
                }
            }
        });
        Ok((q, k, v))
    }

    pub fn forward(&self, batch: &Batch) -> Result<Vec<Vec<f32>>> {
        let b = batch.input_ids.len();
        batch.validate(self.max)?;
        check_shape!(
            batch.attention_mask.len() == b
                && batch
                    .input_ids
                    .iter()
                    .zip(&batch.attention_mask)
                    .all(|(ids, m)| !ids.is_empty() && ids.len() == m.len()),
            "attention_mask length"
        );
        if dense_uniform(batch) {
            return self.forward_dense(batch, &[]);
        }
        let rows = compact_rows(batch)?;
        check_shape!(
            rows.iter().all(|(_, ids)| ids.len() <= self.max),
            "sequence length"
        );
        let rectangular = batch
            .input_ids
            .iter()
            .all(|ids| ids.len() == batch.input_ids[0].len());
        if !worth_split(&rows) && rectangular {
            // Short unique lengths: N serial B1 forwards lose to one padded batch.
            return self.forward_dense(batch, &[]);
        }
        self.forward_grouped_rows(rows)
    }

    /// Sort by live length and run equal-length groups with no pad tokens.
    fn forward_grouped_rows(&self, mut rows: Vec<(usize, Vec<u32>)>) -> Result<Vec<Vec<f32>>> {
        let b = rows.len();
        rows.sort_by_key(|(_, ids)| std::cmp::Reverse(ids.len()));
        let mut out = vec![Vec::new(); b];
        let mut i = 0;
        while i < rows.len() {
            let len = rows[i].1.len();
            let mut j = i + 1;
            while j < rows.len() && rows[j].1.len() == len {
                j += 1;
            }
            let sub = Batch {
                input_ids: rows[i..j].iter().map(|(_, ids)| ids.clone()).collect(),
                attention_mask: vec![vec![1u32; len]; j - i],
            };
            let got = self.forward_dense(&sub, &[])?;
            for (k, (orig, _)) in rows[i..j].iter().enumerate() {
                out[*orig] = got[k].clone();
            }
            i = j;
        }
        Ok(out)
    }

    fn forward_dense(&self, batch: &Batch, splices: &[Splice]) -> Result<Vec<Vec<f32>>> {
        let b = batch.input_ids.len();
        let s = batch.input_ids[0].len();
        check_shape!(s > 0 && s <= self.max, "sequence length");
        check_shape!(
            batch.attention_mask.len() == b
                && batch
                    .input_ids
                    .iter()
                    .zip(&batch.attention_mask)
                    .all(|(ids, m)| ids.len() == s && m.len() == s),
            "dense batch"
        );
        if self.precision == Precision::Fp32 {
            return self.forward_dense_fp32(batch, b, s, splices);
        }
        self.backend.begin_shape(b, s);
        let tokens = rt_core::checked_size("EG2 tokens", &[b, s])?;
        rt_core::checked_size(
            "EG2 workspace",
            &[
                tokens
                    .checked_add(64)
                    .ok_or(rt_core::Error::SizeOverflow("padded rows"))?,
                N_LAYERS,
                PLE_WIDTH,
                4,
            ],
        )?;
        let m_pad = self.work_rows(tokens);
        // These are the existing reservation counts; sum before allocating any.
        let row_bytes = HIDDEN * 4 * 4
            + 2048 * 3 * 4
            + 3072 * 2 * 4
            + PLE_WIDTH * 3 * 4
            + N_LAYERS * PLE_WIDTH * 2 * 4
            + HIDDEN * 3 * 2
            + HIDDEN * 2 * 4;
        let planned = rt_core::checked_size("EG2 reserved activations", &[m_pad, row_bytes])?
            .checked_add(rt_core::checked_size(
                "EG2 reserved BF16 heads",
                &[tokens, N_QUERY_HEADS, 512, 5, 2],
            )?)
            .ok_or(rt_core::Error::SizeOverflow("EG2 workspace"))?;
        self.backend.validate_workspace(planned)?;
        self.backend.reserve(
            rt_core::checked_size("activation elements", &[m_pad, HIDDEN])?,
            4,
        )?;
        self.backend.reserve(
            rt_core::checked_size("activation elements", &[m_pad, 2048])?,
            3,
        )?;
        self.backend.reserve(
            rt_core::checked_size("activation elements", &[m_pad, 3072])?,
            2,
        )?;
        self.backend.reserve(
            rt_core::checked_size("activation elements", &[m_pad, PLE_WIDTH])?,
            3,
        )?;
        self.backend.reserve(
            rt_core::checked_size("activation elements", &[m_pad, N_LAYERS, PLE_WIDTH])?,
            2,
        )?;
        self.backend.reserve_u16(
            rt_core::checked_size("activation elements", &[m_pad, HIDDEN])?,
            3,
        )?;
        self.backend.reserve_u16(
            rt_core::checked_size("activation elements", &[tokens, N_QUERY_HEADS, 512])?,
            3,
        )?;
        self.backend.reserve_u16(
            rt_core::checked_size("activation elements", &[tokens, N_QUERY_HEADS, 512])?,
            2,
        )?;
        self.backend.reserve(
            rt_core::checked_size("activation elements", &[m_pad, HIDDEN])?,
            2,
        )?;

        let mut x = self.embed(batch, b, s, m_pad, splices)?;
        let mut ple_raw = self.linear(&x, "language_model.ple.per_layer_model_projection")?;
        let scale = (HIDDEN as f32).powf(-0.5);
        scale_inplace(&mut ple_raw, scale, self.backend.threads());
        let ple = self.rms(
            &ple_raw,
            "language_model.ple.per_layer_projection_norm.weight",
            PLE_WIDTH,
        )?;
        self.backend.recycle(ple_raw);

        let padding: Vec<Vec<usize>> = batch
            .attention_mask
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter_map(|(i, &v)| (v == 0).then_some(i))
                    .collect()
            })
            .collect();

        for layer in 0..N_LAYERS {
            let base = format!("language_model.layers.{layer}");
            let global = is_global(layer);
            let d = head_dim(layer);
            let nkv = n_kv(layer);
            let nq = N_QUERY_HEADS;

            let residual = x;
            let h_bf =
                self.rms_bf16(&residual, &format!("{base}.input_layernorm.weight"), HIDDEN)?;
            let qkv_name = format!("{base}.self_attn.qkv_proj");
            let (q, k, v) = if self.linears.contains_key(&qkv_name) {
                let fused = self.linear_src(&h_bf, &qkv_name)?;
                let qkv = self.split_qkv_heads(&fused, b, s, nq, nkv, d)?;
                self.backend.recycle(fused);
                qkv
            } else {
                let q_flat = self.linear_src(&h_bf, &format!("{base}.self_attn.q_proj"))?;
                let k_flat = self.linear_src(&h_bf, &format!("{base}.self_attn.k_proj"))?;
                let v_flat = self.linear_src(&h_bf, &format!("{base}.self_attn.v_proj"))?;
                let q = self.to_heads(&q_flat, b, s, nq, d)?;
                let k = self.to_heads(&k_flat, b, s, nkv, d)?;
                let v = self.to_heads(&v_flat, b, s, nkv, d)?;
                self.backend.recycle(q_flat);
                self.backend.recycle(k_flat);
                self.backend.recycle(v_flat);
                (q, k, v)
            };
            self.backend.recycle_u16(h_bf);
            let q_scale = self
                .norms
                .get(&format!("{base}.self_attn.q_norm.weight"))
                .with_context(|| format!("{base}.self_attn.q_norm.weight"))?;
            let k_scale = self
                .norms
                .get(&format!("{base}.self_attn.k_norm.weight"))
                .with_context(|| format!("{base}.self_attn.k_norm.weight"))?;
            let rope = if global {
                &self.global_rope
            } else {
                &self.local_rope
            };
            let mut q_bf = self.backend.buffer_u16(q.len())?;
            let mut k_bf = self.backend.buffer_u16(k.len())?;
            let mut v_bf = self.backend.buffer_u16(v.len())?;
            let threads = self.backend.threads();
            // Row-major [b,nh,s,d] so each pass streams contiguous head rows.
            let t_glue = Instant::now();
            rmsnorm_rope_to_bf16(
                &q,
                &mut q_bf,
                b,
                s,
                nq,
                d,
                Some(q_scale),
                RMS_EPS,
                Some((&rope.cos, &rope.sin)),
                threads,
            )?;
            rmsnorm_rope_to_bf16(
                &k,
                &mut k_bf,
                b,
                s,
                nkv,
                d,
                Some(k_scale),
                RMS_EPS,
                Some((&rope.cos, &rope.sin)),
                threads,
            )?;
            rmsnorm_to_bf16(&v, d, None, RMS_EPS, &mut v_bf, threads)?;
            self.backend
                .add_category_s("glue", t_glue.elapsed().as_secs_f64());
            self.backend.recycle(q);
            self.backend.recycle(k);
            self.backend.recycle(v);

            let attn = self.backend.attention_gqa_src(
                &q_bf,
                &k_bf,
                &v_bf,
                b,
                s,
                nq,
                nkv,
                d,
                self.tile,
                !global,
                SLIDING_WINDOW,
                ATTN_SCALE,
                &padding,
            )?;
            self.backend.recycle_u16(q_bf);
            self.backend.recycle_u16(k_bf);
            self.backend.recycle_u16(v_bf);
            check_shape!(attn.len() == m_pad * nq * d, "attn residual rows");

            let o = self.linear(&attn, &format!("{base}.self_attn.o_proj"))?;
            self.backend.recycle(attn);
            let post_attn = self
                .norms
                .get(&format!("{base}.post_attention_layernorm.weight"))
                .with_context(|| format!("{base}.post_attention_layernorm.weight"))?;
            let mut h = residual;
            check_shape!(h.len() == o.len(), "attn residual");
            let t_glue = Instant::now();
            add_rmsnorm(
                &mut h,
                &o,
                HIDDEN,
                Some(post_attn),
                1.0,
                RMS_EPS,
                self.backend.threads(),
            )?;
            self.backend
                .add_category_s("glue", t_glue.elapsed().as_secs_f64());
            self.backend.recycle(o);

            let residual = h;
            let pre_bf = self.rms_bf16(
                &residual,
                &format!("{base}.pre_feedforward_layernorm.weight"),
                HIDDEN,
            )?;
            let up = self.linear_src(&pre_bf, &format!("{base}.mlp.up_proj"))?;
            let gate_w = &self
                .linears
                .get(&format!("{base}.mlp.gate_proj"))
                .with_context(|| format!("{base}.mlp.gate_proj"))?
                .weight;
            let gated = self.backend.gated_linear_tanh_src(&pre_bf, gate_w, &up)?;
            self.backend.recycle_u16(pre_bf);
            self.backend.recycle(up);
            let down = self.linear(&gated, &format!("{base}.mlp.down_proj"))?;
            self.backend.recycle(gated);
            let post_ff = self
                .norms
                .get(&format!("{base}.post_feedforward_layernorm.weight"))
                .with_context(|| format!("{base}.post_feedforward_layernorm.weight"))?;
            let mut h = residual;
            let t_glue = Instant::now();
            add_rmsnorm(
                &mut h,
                &down,
                HIDDEN,
                Some(post_ff),
                1.0,
                RMS_EPS,
                self.backend.threads(),
            )?;
            self.backend
                .add_category_s("glue", t_glue.elapsed().as_secs_f64());
            self.backend.recycle(down);

            let residual = h;
            let res_bf = self.to_bf16(&residual)?;
            let gate =
                self.linear_src(&res_bf, &format!("{base}.ple_block.per_layer_input_gate"))?;
            self.backend.recycle_u16(res_bf);
            let mut mixed = self.backend.buffer(gate.len())?;
            let ple_off = layer * PLE_WIDTH;
            let threads = self.backend.threads();
            let gate_p = gate.as_ptr() as usize;
            let ple_p = ple.as_ptr() as usize;
            let mix_p = mixed.as_mut_ptr() as usize;
            let mix_rows = gate.len() / PLE_WIDTH;
            for_row_tiles(
                threads,
                mix_rows,
                self.backend.residual_row_tile(tokens),
                8,
                |lo, hi| {
                    for ti in lo..hi {
                        let src = unsafe { row_ref(gate_p as *const f32, ti, PLE_WIDTH) };
                        let p = unsafe {
                            row_ref(
                                (ple_p as *const f32).add(ti * (N_LAYERS * PLE_WIDTH) + ple_off),
                                0,
                                PLE_WIDTH,
                            )
                        };
                        let row = unsafe { row_mut(mix_p as *mut f32, ti, PLE_WIDTH) };
                        unsafe { rt_kernels::gelu_tanh_mul_unchecked(src, p, row) };
                    }
                },
            )?;
            self.backend.recycle(gate);
            let proj = self.linear(&mixed, &format!("{base}.ple_block.per_layer_projection"))?;
            self.backend.recycle(mixed);
            let ple_norm = self
                .norms
                .get(&format!(
                    "{base}.ple_block.post_per_layer_input_norm.weight"
                ))
                .with_context(|| format!("{base}.ple_block.post_per_layer_input_norm.weight"))?;
            x = residual;
            let t_glue = Instant::now();
            add_rmsnorm(
                &mut x,
                &proj,
                HIDDEN,
                Some(ple_norm),
                self.scalars[layer],
                RMS_EPS,
                self.backend.threads(),
            )?;
            self.backend
                .add_category_s("glue", t_glue.elapsed().as_secs_f64());
            self.backend.recycle(proj);
        }
        self.backend.recycle(ple);

        let x_bf = self.rms_bf16(&x, "language_model.norm.weight", HIDDEN)?;
        self.backend.recycle(x);
        let proj = self.linear_src(&x_bf, "language_model.embedding_projection")?;
        self.backend.recycle_u16(x_bf);
        // Pool only the live prefix, using the original attention mask.
        let pooled = mean_pool(&proj[..tokens * EMBED_DIM], batch, b, s, EMBED_DIM);
        self.backend.recycle(proj);
        Ok(l2_rows(pooled, EMBED_DIM))
    }

    /// Load the image and/or audio towers next to the text tower. Each keeps
    /// its own packed cache beside the text one, so text-only loads never
    /// read or pack them.
    pub fn load_media(
        &mut self,
        path: &Path,
        cache: &CacheOptions,
        image: bool,
        audio: bool,
    ) -> Result<()> {
        if !image && !audio {
            return Ok(());
        }
        let st_path = path.join("model.safetensors");
        let (sha, _) = expected_weight_pin(cache.manifest.as_deref())?;
        let text_cache = cache_path_with(&st_path, self.backend.cpu_isa(), cache.file.as_deref());
        let src = TowerSource {
            st_path: &st_path,
            text_cache: &text_cache,
            sha: &sha,
            cache,
        };
        if image && self.vision.is_none() {
            let t = Instant::now();
            let tower = vision::VisionTower::load(&src, &*self.backend, self.precision)
                .map_err(|e| e.at(path))?;
            self.load_profile
                .insert("vision_load_s".into(), t.elapsed().as_secs_f64());
            self.vision = Some(tower);
        }
        if audio && self.audio.is_none() {
            let t = Instant::now();
            let tower = audio::AudioTower::load(&src, &*self.backend, self.precision)
                .map_err(|e| e.at(path))?;
            self.load_profile
                .insert("audio_load_s".into(), t.elapsed().as_secs_f64());
            self.audio = Some(tower);
        }
        Ok(())
    }

    pub fn threads(&self) -> usize {
        self.backend.threads()
    }

    pub fn has_image(&self) -> bool {
        self.vision.is_some()
    }

    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }

    /// Soft tokens `[img.soft_tokens(), HIDDEN]` of one image.
    pub(crate) fn image_soft_tokens(&self, img: &crate::media::ImagePatches) -> Result<Vec<f32>> {
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| rt_core::Error::Unsupported {
                what: "EG2 image input",
                reason: "load with Config::image = true".into(),
            })?;
        tower.forward(&*self.backend, self.precision, self.tile, img)
    }

    /// Soft tokens `[f.soft_tokens(), HIDDEN]` of one audio clip.
    pub(crate) fn audio_soft_tokens(&self, f: &crate::media::AudioFeatures) -> Result<Vec<f32>> {
        let tower = self
            .audio
            .as_ref()
            .ok_or_else(|| rt_core::Error::Unsupported {
                what: "EG2 audio input",
                reason: "load with Config::audio = true".into(),
            })?;
        tower.forward(&*self.backend, self.precision, f)
    }

    /// Embed sequences whose placeholder rows are replaced by soft tokens.
    /// Equal lengths share one dense forward; outputs keep input order.
    pub(crate) fn forward_spliced(&self, rows: Vec<(Vec<u32>, Splice)>) -> Result<Vec<Vec<f32>>> {
        check_shape!(
            rows.iter()
                .all(|(ids, _)| !ids.is_empty() && ids.len() <= self.max),
            "sequence length"
        );
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(rows[i].0.len()));
        let mut rows: Vec<Option<(Vec<u32>, Splice)>> = rows.into_iter().map(Some).collect();
        let mut out = vec![Vec::new(); rows.len()];
        let mut i = 0;
        while i < order.len() {
            let len = rows[order[i]].as_ref().map_or(0, |r| r.0.len());
            let mut j = i + 1;
            while j < order.len() && rows[order[j]].as_ref().map_or(0, |r| r.0.len()) == len {
                j += 1;
            }
            let mut ids = Vec::with_capacity(j - i);
            let mut splices = Vec::with_capacity(j - i);
            for &k in &order[i..j] {
                let (row, splice) = rows[k].take().context("spliced row")?;
                ids.push(row);
                splices.push(splice);
            }
            let sub = Batch {
                attention_mask: vec![vec![1u32; len]; ids.len()],
                input_ids: ids,
            };
            sub.validate(self.max)?;
            let got = self.forward_dense(&sub, &splices)?;
            for (&k, emb) in order[i..j].iter().zip(got) {
                out[k] = emb;
            }
            i = j;
        }
        Ok(out)
    }

    /// Run one discarded 16-token forward through the normal execution path.
    ///
    /// Populates execution caches and faults working pages without changing
    /// weights or subsequent results. Uses the normal execution path and
    /// returns its typed errors. No warmup runs implicitly when loading a Model.
    pub fn warmup(&self) -> Result<()> {
        let batch = Batch {
            input_ids: vec![vec![2u32; 16]],
            attention_mask: vec![vec![1u32; 16]],
        };
        self.forward(&batch)?;
        Ok(())
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    pub fn stats(&self) -> Value {
        let mut stats = self.backend.stats();
        stats["load_source"] = serde_json::json!(self.load_source);
        stats["load_profile_s"] = serde_json::json!(self.load_profile);
        stats["precision"] = serde_json::json!(precision_label(self.precision));
        stats["modalities"] = serde_json::json!({
            "image": self.vision.as_ref().map(|t| t.load_source()),
            "audio": self.audio.as_ref().map(|t| t.load_source()),
        });
        stats
    }
}

struct Assembled {
    norms: HashMap<String, Vec<f32>>,
    scalars: Vec<f32>,
}

fn assemble_raw(raw: Vec<RawTensor>) -> Result<Assembled> {
    let mut map: HashMap<String, Vec<f32>> = raw.into_iter().map(|t| (t.name, t.data)).collect();
    let mut scalars = vec![1.0f32; N_LAYERS];
    for i in 0..N_LAYERS {
        let key = format!("language_model.layers.{i}.layer_scalar");
        {
            let v = map
                .remove(&key)
                .with_context(|| format!("missing tensor {key}"))?;
            check_shape!(v.len() == 1, "layer_scalar rank {key}");
            scalars[i] = v[0];
        }
    }
    Ok(Assembled {
        norms: map,
        scalars,
    })
}

fn embed_row(src: &[u16], dst: &mut [f32], scale: f32) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bf16") {
            unsafe {
                embed_row_avx512(src, dst, scale);
            }
            return;
        }
    }
    for (d, &bits) in dst.iter_mut().zip(src) {
        *d = bf16_bits_to_f32(bits) * scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn embed_row_avx512(src: &[u16], dst: &mut [f32], scale: f32) {
    use std::arch::x86_64::*;
    let n = src.len();
    let s = _mm512_set1_ps(scale);
    let mut i = 0;
    while i + 16 <= n {
        let packed = _mm256_loadu_si256(src.as_ptr().add(i).cast());
        let f = _mm512_cvtpbh_ps(std::mem::transmute::<__m256i, __m256bh>(packed));
        _mm512_storeu_ps(dst.as_mut_ptr().add(i), _mm512_mul_ps(f, s));
        i += 16;
    }
    while i < n {
        *dst.get_unchecked_mut(i) = bf16_bits_to_f32(*src.get_unchecked(i)) * scale;
        i += 1;
    }
}

fn dense_uniform(batch: &Batch) -> bool {
    let s = batch.input_ids[0].len();
    batch.input_ids.iter().all(|ids| ids.len() == s)
        && batch
            .attention_mask
            .iter()
            .all(|m| m.len() == s && m.iter().all(|&v| v != 0))
}

fn compact_rows(batch: &Batch) -> Result<Vec<(usize, Vec<u32>)>> {
    let mut rows = Vec::with_capacity(batch.input_ids.len());
    for (i, (ids, mask)) in batch
        .input_ids
        .iter()
        .zip(&batch.attention_mask)
        .enumerate()
    {
        let live: Vec<u32> = ids
            .iter()
            .zip(mask)
            .filter_map(|(&id, &m)| (m != 0).then_some(id))
            .collect();
        check_shape!(!live.is_empty(), "empty sequence");
        rows.push((i, live));
    }
    Ok(rows)
}

/// Split mixed lengths only when pad waste is large and groups are not all tiny B1s.
fn worth_split(rows: &[(usize, Vec<u32>)]) -> bool {
    if rows.len() <= 1 {
        return false;
    }
    let max = rows.iter().map(|(_, r)| r.len()).max().unwrap_or(0);
    let live: usize = rows.iter().map(|(_, r)| r.len()).sum();
    let padded = rows.len().saturating_mul(max);
    if live.saturating_mul(5) / 4 >= padded {
        return false;
    }
    let mut counts = HashMap::new();
    for (_, r) in rows {
        *counts.entry(r.len()).or_insert(0usize) += 1;
    }
    if counts.len() == 1 {
        return true;
    }
    let all_singletons = counts.values().all(|&n| n == 1);
    !(all_singletons && max <= 256)
}

fn mean_pool(x: &[f32], batch: &Batch, b: usize, s: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * dim];
    for bi in 0..b {
        let mut count = 0.0f32;
        for p in 0..s {
            if batch.attention_mask[bi][p] == 0 {
                continue;
            }
            count += 1.0;
            let src = (bi * s + p) * dim;
            let dst = bi * dim;
            for j in 0..dim {
                out[dst + j] += x[src + j];
            }
        }
        let inv = 1.0 / count.max(1.0);
        for j in 0..dim {
            out[bi * dim + j] *= inv;
        }
    }
    out
}

fn l2_rows(mut x: Vec<f32>, dim: usize) -> Vec<Vec<f32>> {
    let mut rows = Vec::with_capacity(x.len() / dim);
    for row in x.chunks_exact_mut(dim) {
        let sum = row.iter().map(|v| v * v).sum::<f32>();
        let inv = 1.0 / sum.sqrt().max(1e-12);
        for v in row.iter_mut() {
            *v *= inv;
        }
        rows.push(row.to_vec());
    }
    rows
}

pub fn mrl_truncate(emb: &[f32], dim: usize) -> Vec<f32> {
    let d = dim.min(emb.len());
    let mut v = emb[..d].to_vec();
    let sum = v.iter().map(|x| x * x).sum::<f32>();
    let inv = 1.0 / sum.sqrt().max(1e-12);
    for x in &mut v {
        *x *= inv;
    }
    v
}

fn validate_weights<B: Hardware>(
    linears: &HashMap<String, Linear<B>>,
    norms: &HashMap<String, Vec<f32>>,
) -> Result<()> {
    let linear = |name: &str, n: usize, k: usize| -> Result<()> {
        let w = &linears
            .get(name)
            .with_context(|| format!("missing EG2 tensor {name}"))?
            .weight;
        check_shape!(
            w.n() == n && w.k() == k,
            "EG2 tensor {name}: shape [{}, {}], expected [{n}, {k}]",
            w.n(),
            w.k()
        );
        Ok(())
    };
    let norm = |name: &str, width: usize| -> Result<()> {
        let v = norms
            .get(name)
            .with_context(|| format!("missing EG2 tensor {name}"))?;
        check_shape!(
            v.len() == width,
            "EG2 tensor {name}: width {}, expected {width}",
            v.len()
        );
        Ok(())
    };
    linear(
        "language_model.ple.per_layer_model_projection",
        N_LAYERS * PLE_WIDTH,
        HIDDEN,
    )?;
    linear("language_model.embedding_projection", EMBED_DIM, HIDDEN)?;
    norm(
        "language_model.ple.per_layer_projection_norm.weight",
        PLE_WIDTH,
    )?;
    norm("language_model.norm.weight", HIDDEN)?;
    for layer in 0..N_LAYERS {
        let base = format!("language_model.layers.{layer}");
        let d = head_dim(layer);
        let nq = N_QUERY_HEADS;
        let nkv = n_kv(layer);
        let fused = format!("{base}.self_attn.qkv_proj");
        if linears.contains_key(&fused) {
            linear(&fused, (nq + 2 * nkv) * d, HIDDEN)?;
        } else {
            linear(&format!("{base}.self_attn.q_proj"), nq * d, HIDDEN)?;
            for part in ["k", "v"] {
                linear(&format!("{base}.self_attn.{part}_proj"), nkv * d, HIDDEN)?;
            }
        }
        linear(&format!("{base}.self_attn.o_proj"), HIDDEN, nq * d)?;
        for part in ["up", "gate"] {
            linear(
                &format!("{base}.mlp.{part}_proj"),
                crate::config::INTERMEDIATE,
                HIDDEN,
            )?;
        }
        linear(
            &format!("{base}.mlp.down_proj"),
            HIDDEN,
            crate::config::INTERMEDIATE,
        )?;
        linear(
            &format!("{base}.ple_block.per_layer_input_gate"),
            PLE_WIDTH,
            HIDDEN,
        )?;
        linear(
            &format!("{base}.ple_block.per_layer_projection"),
            HIDDEN,
            PLE_WIDTH,
        )?;
        for name in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "ple_block.post_per_layer_input_norm",
        ] {
            norm(&format!("{base}.{name}.weight"), HIDDEN)?;
        }
        for part in ["q", "k"] {
            norm(&format!("{base}.self_attn.{part}_norm.weight"), d)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod malformed_file_tests {
    use super::*;
    use std::path::PathBuf;
    fn test_path(name: &str) -> PathBuf {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/round1-input-tests")
            .join(format!("{}-{}", "eg2", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }
    #[test]
    fn embedded_pin_and_explicit_manifest_errors_need_no_model_assets() {
        let embedded = rt_assets::Manifest::parse(crate::ASSET_MANIFEST).unwrap();
        let file = embedded.file("model.safetensors").unwrap();
        assert_eq!(
            expected_weight_pin(None).unwrap(),
            (file.sha256.clone(), file.size)
        );
        let missing = test_path("round2-missing-manifest.toml");
        let _ = std::fs::remove_file(&missing);
        assert!(
            matches!(expected_weight_pin(Some(&missing)), Err(rt_core::Error::File { path, .. }) if path == missing)
        );
        let explicit = test_path("round2-explicit-manifest.toml");
        std::fs::write(
            &explicit,
            crate::ASSET_MANIFEST.replace(&file.sha256, &"a".repeat(64)),
        )
        .unwrap();
        assert_eq!(
            expected_weight_pin(Some(&explicit)).unwrap(),
            ("a".repeat(64), file.size)
        );
        std::fs::write(&explicit, "malformed manifest").unwrap();
        assert!(
            matches!(expected_weight_pin(Some(&explicit)), Err(rt_core::Error::File { path, .. }) if path == explicit)
        );
        std::fs::remove_file(explicit).unwrap();
    }

    fn tensor_file(header: Value, body: &[u8]) -> Vec<u8> {
        let mut header = serde_json::to_vec(&header).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(body);
        bytes
    }
    #[test]
    fn truncated_and_invalid_headers_return_errors() {
        let p = test_path("truncated.safetensors");
        for bytes in [
            vec![],
            vec![0; 7],
            u64::MAX.to_le_bytes().to_vec(),
            [8u64.to_le_bytes().as_slice(), b"[broken]"].concat(),
        ] {
            std::fs::write(&p, bytes).unwrap();
            assert!(Mapping::load(&p).is_err());
        }
        for header in [
            serde_json::json!({"x":{"dtype":"F32","shape":[usize::MAX,2],"data_offsets":[0,0]}}),
            serde_json::json!({"x":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}),
            serde_json::json!({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}),
        ] {
            std::fs::write(&p, tensor_file(header, &[])).unwrap();
            assert!(Mapping::load(&p).is_err());
        }
        std::fs::remove_file(p).unwrap();
    }
    #[test]
    fn wrong_dtype_missing_tensor_and_embedding_shape_return_errors() {
        let p = test_path("shape.safetensors");
        let header = serde_json::json!({"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}});
        std::fs::write(&p, tensor_file(header, &[0])).unwrap();
        let mapping = Mapping::load(&p).unwrap();
        assert!(mapping.view_bf16("x").is_err());
        assert!(mapping.view_bf16("missing").is_err());
        assert!(mapping.embedding_span().is_err());
        drop(mapping);
        std::fs::remove_file(p).unwrap();
    }
    #[test]
    fn missing_projection_weights_are_rejected_without_assets() {
        let linears: HashMap<String, Linear<rt_onednn::OnednnCpu>> = HashMap::new();
        assert!(validate_weights(&linears, &HashMap::new()).is_err());
    }
}
