//! Direct safetensors Vela model. Data/control/indexing are Rust; operators
//! go through [`rt_backend::Hardware`] (oneDNN-CPU today).
use memmap2::{Mmap, MmapOptions};
use rt_backend::{Hardware, Packed};
use rt_core::{check_shape, Result, ResultContext as Context};
use rt_kernels::{add_row_bias, for_ranges, layout_qkv_yarn};
use rt_onednn::{
    apply_cache, cache_path_with, cache_write_failed, file_stamp, pinned_dnnl, CacheMeta,
    CacheOptions, CacheWriter, PackedEntry, RawTensor,
};
use rt_text::{Inputs, Logits};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const EMB_NAME: &str = "encoder.embeddings.tok_embeddings.weight";

fn page_size() -> usize {
    let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if p > 0 {
        p as usize
    } else {
        4096
    }
}

#[derive(Deserialize)]
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
    fn view(&self, n: &str) -> Result<&[f32]> {
        let i = self
            .header
            .get(n)
            .with_context(|| format!("missing tensor {n}"))?;
        check_shape!(i.dtype == "F32", "FP32 required: {n}");
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
                && b - a == count.checked_mul(4).context("tensor bytes")?,
            "tensor bounds {n}"
        );
        check_shape!(
            (self.data.as_ptr() as usize + start) % 4 == 0 && cfg!(target_endian = "little"),
            "FP32 mapped alignment/endian"
        );
        Ok(unsafe { std::slice::from_raw_parts(self.data.as_ptr().add(start).cast(), count) })
    }

    /// Best-effort: drop whole pages of a consumed tensor so the 1.2 GiB
    /// mapping does not stay resident alongside the packed oneDNN copy.
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

    fn embedding_span(&self, vocab: usize, h: usize) -> Result<(usize, usize)> {
        let i = self
            .header
            .get(EMB_NAME)
            .with_context(|| format!("missing tensor {EMB_NAME}"))?;
        check_shape!(
            i.dtype == "F32" && i.shape == vec![vocab, h],
            "embedding shape"
        );
        let [a, b] = i.data_offsets;
        let start = self.offset.checked_add(a).context("embedding offset")?;
        let bytes = vocab
            .checked_mul(h)
            .and_then(|n| n.checked_mul(4))
            .context("embedding bytes")?;
        check_shape!(
            b.saturating_sub(a) == bytes
                && bytes <= self.data.len().saturating_sub(start)
                && start <= self.data.len()
                && a <= b,
            "embedding bounds"
        );
        Ok((start, vocab * h))
    }

    fn into_embeddings(self, vocab: usize, h: usize) -> Result<EmbeddingTable> {
        let (start, count) = self.embedding_span(vocab, h)?;
        check_shape!(
            (self.data.as_ptr() as usize + start) % 4 == 0 && cfg!(target_endian = "little"),
            "FP32 mapped alignment/endian"
        );
        Ok(EmbeddingTable {
            map: self.data,
            start,
            count,
        })
    }
}

/// Zero-copy token table: a view of mmapped safetensors. Only gathered rows
/// become RSS. The 256000×768 table is not copied to the heap or the cache.
struct EmbeddingTable {
    map: Mmap,
    start: usize,
    count: usize,
}

impl EmbeddingTable {
    fn from_safetensors(path: &Path, vocab: usize, h: usize) -> Result<Self> {
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
                .get(EMB_NAME)
                .cloned()
                .context("embeddings missing from safetensors header")?,
        )?;
        check_shape!(
            info.dtype == "F32" && info.shape == vec![vocab, h],
            "embedding shape"
        );
        let [a, b] = info.data_offsets;
        let data_start = 8usize
            .checked_add(hlen)
            .and_then(|o| o.checked_add(a))
            .context("embedding offset")?;
        let nbytes = vocab
            .checked_mul(h)
            .and_then(|n| n.checked_mul(4))
            .context("embedding bytes")?;
        check_shape!(
            b.saturating_sub(a) == nbytes
                && nbytes <= file_len.saturating_sub(data_start)
                && data_start <= file_len
                && a <= b,
            "embedding bounds"
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
            (map.as_ptr() as usize + delta) % 4 == 0 && cfg!(target_endian = "little"),
            "FP32 mapped alignment/endian"
        );
        Ok(Self {
            map,
            start: delta,
            count: vocab * h,
        })
    }

    fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.map.as_ptr().add(self.start).cast(), self.count) }
    }
}

struct Linear<B: Hardware> {
    weight: B::Packed,
    bias: Option<Vec<f32>>,
}
struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}
struct Rope {
    cos: Vec<f32>,
    sin: Vec<f32>,
}

pub struct Model<B: Hardware> {
    embeddings: EmbeddingTable,
    backend: Arc<B>,
    cfg: Value,
    enc: Value,
    linears: HashMap<String, Linear<B>>,
    norms: HashMap<String, Norm>,
    global: Rope,
    local: Rope,
    h: usize,
    nh: usize,
    d: usize,
    inter: usize,
    proj: usize,
    layers: usize,
    max: usize,
    vocab: usize,
    tile: usize,
    tau: f32,
    tau_span: f32,
    eps: f32,
    profiling: bool,
    profile: Mutex<BTreeMap<String, f64>>,
    load_source: &'static str,
    load_profile: BTreeMap<String, f64>,
}

fn number(v: &Value, k: &str) -> Result<usize> {
    v[k].as_u64()
        .map(|v| v as usize)
        .with_context(|| format!("config {k}"))
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

impl<B: Hardware> Model<B> {
    pub fn load(path: &Path, backend: Arc<B>, tile: usize) -> Result<Self> {
        Self::load_with(path, backend, tile, &CacheOptions::default(), false)
    }

    pub fn load_with(
        path: &Path,
        backend: Arc<B>,
        tile: usize,
        cache: &CacheOptions,
        profiling: bool,
    ) -> Result<Self> {
        Self::load_impl(path, backend, tile, cache, profiling).map_err(|e| e.at(path))
    }

    fn load_impl(
        path: &Path,
        backend: Arc<B>,
        tile: usize,
        cache: &CacheOptions,
        profiling: bool,
    ) -> Result<Self> {
        let cfg = rt_core::read_json(path.join("config.json"))?;
        let enc = cfg["encoder_config"].clone();
        let h = number(&enc, "hidden_size")?;
        let nh = number(&enc, "num_attention_heads")?;
        check_shape!(
            nh > 0 && h % nh == 0 && tile > 0 && cfg["readout"] == "both",
            "architecture/readout"
        );
        let d = h / nh;
        check_shape!(d >= 2 && d % 2 == 0, "RoPE width {d}");
        let inter = number(&enc, "intermediate_size")?;
        let proj = number(&cfg, "proj_dim")?;
        let layers = number(&enc, "num_hidden_layers")?;
        let max = number(&cfg, "max_length")?;
        let vocab = number(&enc, "vocab_size")?;
        for (name, dim) in [
            ("hidden_size", h),
            ("intermediate_size", inter),
            ("proj_dim", proj),
            ("layers", layers),
            ("max_length", max),
            ("vocab_size", vocab),
        ] {
            check_shape!(dim > 0, "Vela config {name} must be positive");
        }
        rt_core::checked_size("Vela largest projection", &[h, inter, 2, 4])?;
        rt_core::checked_size("Vela RoPE table", &[max, d, 4])?;
        let eps = enc["layer_norm_eps"].as_f64().context("norm epsilon")? as f32;
        let st_path = path.join("model.safetensors");
        let (sha, pinned_size) = expected_weight_pin(cache.manifest.as_deref())?;
        let cache_file = cache_path_with(&st_path, backend.cpu_isa(), cache.file.as_deref());
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
                    let weight = backend.restore_packed_checked(n, k, bytes, layout)?;
                    linears.insert(name.to_string(), Linear { weight, bias: None });
                    Ok(())
                },
            );
            load_profile.insert("cache_read_s".into(), t.elapsed().as_secs_f64());
            if let Ok(Some(raw)) = applied {
                if let (Ok(parts), Ok(embeddings)) = (
                    assemble_raw(raw, &mut linears, h),
                    EmbeddingTable::from_safetensors(&st_path, vocab, h),
                ) {
                    return Self::finish(
                        backend,
                        cfg,
                        enc,
                        linears,
                        parts,
                        embeddings,
                        h,
                        nh,
                        d,
                        inter,
                        proj,
                        layers,
                        max,
                        vocab,
                        tile,
                        eps,
                        "packed-cache",
                        load_profile,
                        profiling,
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
        for n in names {
            if n == EMB_NAME {
                continue;
            }
            if n == "log_tau" || n == "log_tau_span" || n.ends_with(".bias") {
                raw_out.push(RawTensor {
                    name: n.clone(),
                    data: mapping.view(&n)?.to_vec(),
                });
                mapping.advise_unused(&n);
                continue;
            }
            if !n.ends_with(".weight") {
                continue;
            }
            let shape = &mapping.header[&n].shape;
            let base = n.strip_suffix(".weight").context("weight tensor suffix")?;
            let bn = format!("{base}.bias");
            if shape.len() == 2 {
                if n.ends_with(".mlp.Wi.weight") && !mapping.header.contains_key(&bn) {
                    check_shape!(shape == &vec![2 * inter, h], "GeGLU projection shape");
                    let raw = mapping.view(&n)?;
                    for (split, values) in
                        [("left", &raw[..inter * h]), ("right", &raw[inter * h..])]
                    {
                        let key = format!("{base}.{split}");
                        let weight = pack_weight(&*backend, &key, inter, h, values, &mut cache_w)?;
                        linears.insert(key, Linear { weight, bias: None });
                    }
                    mapping.advise_unused(&n);
                    continue;
                }
                let weight = pack_weight(
                    &*backend,
                    base,
                    shape[0],
                    shape[1],
                    mapping.view(&n)?,
                    &mut cache_w,
                )?;
                linears.insert(base.to_string(), Linear { weight, bias: None });
                mapping.advise_unused(&n);
            } else if shape.len() == 1 {
                check_shape!(shape[0] == h, "normalization width");
                raw_out.push(RawTensor {
                    name: n.clone(),
                    data: mapping.view(&n)?.to_vec(),
                });
                mapping.advise_unused(&n);
            }
        }
        load_profile.insert("pack_s".into(), t.elapsed().as_secs_f64());
        let embeddings = mapping.into_embeddings(vocab, h)?;
        if let Some(w) = cache_w {
            let t = Instant::now();
            if let Err(e) = w.finish(&raw_out) {
                cache_write_failed(&cache_file, &e);
            }
            load_profile.insert("cache_write_s".into(), t.elapsed().as_secs_f64());
        }
        let parts = assemble_raw(raw_out, &mut linears, h)?;
        Self::finish(
            backend,
            cfg,
            enc,
            linears,
            parts,
            embeddings,
            h,
            nh,
            d,
            inter,
            proj,
            layers,
            max,
            vocab,
            tile,
            eps,
            "safetensors",
            load_profile,
            profiling,
        )
    }

    fn record(&self, name: &str, t: Option<Instant>) {
        if let Some(t) = t {
            *self
                .profile
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entry(name.into())
                .or_default() += t.elapsed().as_secs_f64();
        }
    }
    fn timer(&self) -> Option<Instant> {
        self.profiling.then(Instant::now)
    }
    fn rope(enc: &Value, max: usize, d: usize, theta: f64) -> Result<Rope> {
        let r = &enc["rope_scaling"];
        check_shape!(r["rope_type"] == "yarn", "RoPE type");
        let factor = r["factor"].as_f64().context("RoPE factor")?;
        let orig = r["original_max_position_embeddings"]
            .as_f64()
            .context("RoPE original positions")?;
        check_shape!(
            factor.is_finite()
                && factor > 0.
                && orig.is_finite()
                && orig > 0.
                && theta.is_finite()
                && theta > 1.,
            "invalid Vela RoPE parameters"
        );
        let corr =
            |n: f64| (d as f64) * (orig / (n * 2. * std::f64::consts::PI)).ln() / (2. * theta.ln());
        let mut lo = corr(r["beta_fast"].as_f64().context("RoPE beta_fast")?);
        let mut hi = corr(r["beta_slow"].as_f64().context("RoPE beta_slow")?);
        if r["truncate"].as_bool().unwrap_or(false) {
            lo = lo.floor();
            hi = hi.ceil();
        }
        lo = lo.max(0.);
        hi = hi.min((d - 1) as f64);
        if lo == hi {
            hi += 0.001;
        }
        let amp = (1. + 0.1 * factor.ln()) as f32;
        let mut inv = vec![0.; d / 2];
        for (i, v) in inv.iter_mut().enumerate() {
            let freq = (theta as f32).powf((2 * i) as f32 / d as f32);
            let e = 1. - ((i as f32 - lo as f32) / (hi - lo) as f32).clamp(0., 1.);
            *v = (1. / (factor as f32 * freq)) * (1. - e) + (1. / freq) * e;
        }
        let mut cos =
            rt_core::try_vec::<f32>(rt_core::checked_size("Vela RoPE table", &[max, d / 2])?)?;
        let mut sin = cos.clone();
        for p in 0..max {
            for i in 0..d / 2 {
                let phase = p as f32 * inv[i];
                cos[p * d / 2 + i] = phase.cos() * amp;
                sin[p * d / 2 + i] = phase.sin() * amp;
            }
        }
        Ok(Rope { cos, sin })
    }
    fn norm(&self, x: &[f32], name: &str) -> Result<Vec<f32>> {
        let n = self
            .norms
            .get(name)
            .with_context(|| format!("norm {name}"))?;
        check_shape!(x.len() % self.h == 0, "norm matrix");
        self.backend
            .layernorm(x.len() / self.h, self.h, x, &n.weight, &n.bias, self.eps)
    }
    fn linear(&self, x: &[f32], name: &str) -> Result<Vec<f32>> {
        let l = self
            .linears
            .get(name)
            .with_context(|| format!("linear {name}"))?;
        check_shape!(x.len() % l.weight.k() == 0, "linear matrix");
        let mut y = self.backend.buffer(rt_core::checked_size(
            "activation elements",
            &[x.len() / l.weight.k(), l.weight.n()],
        )?)?;
        self.backend.linear(x, &l.weight, &mut y, false)?;
        if let Some(bias) = &l.bias {
            add_row_bias(&mut y, bias, self.backend.threads())?;
        }
        Ok(y)
    }
    fn linear_norm(&self, x: &[f32], norm: &str, linear: &str) -> Result<Vec<f32>> {
        let z = self.norm(x, norm)?;
        let result = self.linear(&z, linear);
        self.backend.recycle(z);
        result
    }
    fn linear_add(&self, x: &[f32], name: &str, y: &mut [f32]) -> Result<()> {
        let l = self
            .linears
            .get(name)
            .with_context(|| format!("linear {name}"))?;
        self.backend.linear(x, &l.weight, y, true)?;
        if let Some(bias) = &l.bias {
            add_row_bias(y, bias, self.backend.threads())?;
        }
        Ok(())
    }
    fn validate(&self, r: &Inputs) -> Result<(usize, usize)> {
        r.validate(self.max, self.vocab)
    }
    fn embedding(&self, r: &Inputs, b: usize, s: usize) -> Result<Vec<f32>> {
        check_shape!(
            self.embeddings.count == self.vocab * self.h,
            "embedding shape"
        );
        let ntok = b * s;
        let mut x = self.backend.buffer(rt_core::checked_size(
            "activation elements",
            &[ntok, self.h],
        )?)?;
        let ids: Vec<u32> = r.input_ids.iter().flatten().copied().collect();
        let table = self.embeddings.as_slice();
        let h = self.h;
        let x_p = x.as_mut_ptr() as usize;
        let t_p = table.as_ptr() as usize;
        let t_len = table.len();
        for_ranges(self.backend.threads(), ntok, 8, |lo, hi| {
            let x = unsafe { std::slice::from_raw_parts_mut(x_p as *mut f32, ntok * h) };
            let table = unsafe { std::slice::from_raw_parts(t_p as *const f32, t_len) };
            for t in lo..hi {
                let id = ids[t] as usize;
                x[t * h..(t + 1) * h].copy_from_slice(&table[id * h..(id + 1) * h]);
            }
        });
        Ok(x)
    }
    fn qkv(
        &self,
        z: &[f32],
        b: usize,
        s: usize,
        local: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let timing = self.timer();
        let rope = if local { &self.local } else { &self.global };
        let mut q = self.backend.buffer(rt_core::checked_size(
            "activation elements",
            &[b, s, self.h],
        )?)?;
        let mut k = self.backend.buffer(q.len())?;
        let mut v = self.backend.buffer(q.len())?;
        let scale = (self.d as f32).sqrt().recip();
        layout_qkv_yarn(
            z,
            &mut q,
            &mut k,
            &mut v,
            b,
            s,
            self.nh,
            self.d,
            &rope.cos,
            &rope.sin,
            scale,
            self.backend.threads(),
        )?;
        self.record("qkv_rope_layout", timing);
        Ok((q, k, v))
    }

    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        r: &Inputs,
        b: usize,
        s: usize,
        local: bool,
    ) -> Result<Vec<f32>> {
        let timing = self.timer();
        let radius = number(&self.enc, "local_attention")? / 2;
        let padding = r
            .attention_mask
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter_map(|(i, &v)| (v == 0).then_some(i))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let out = self.backend.attention(
            q, k, v, b, s, self.nh, self.d, self.tile, local, radius, &padding,
        )?;
        self.record("attention_with_primitives", timing);
        Ok(out)
    }

    fn pool(&self, x: &[f32], r: &Inputs, b: usize, s: usize) -> Vec<f32> {
        let timing = self.timer();
        let mut out = vec![0.; r.q_index.len() * self.h];
        for bi in 0..b {
            let mut events: BTreeMap<usize, Vec<(usize, bool)>> = BTreeMap::new();
            for (i, q) in r.q_index.iter().enumerate() {
                if q[0] == bi {
                    events.entry(q[2]).or_default().push((i, false));
                    events.entry(q[3]).or_default().push((i, true));
                }
            }
            let Some(&last) = events.keys().next_back() else {
                continue;
            };
            let mut prefix = vec![0f64; self.h];
            for p in 0..=last {
                if let Some(ev) = events.get(&p) {
                    for &(i, end) in ev {
                        let count = (r.q_index[i][3] - r.q_index[i][2]).max(1) as f32;
                        for j in 0..self.h {
                            let rounded = prefix[j] as f32;
                            out[i * self.h + j] = if end {
                                (rounded - out[i * self.h + j]) / count
                            } else {
                                rounded
                            };
                        }
                    }
                }
                if p < last {
                    for j in 0..self.h {
                        prefix[j] += x[(bi * s + p) * self.h + j] as f64;
                    }
                }
            }
        }
        self.record("prefix_pool", timing);
        out
    }
    fn gather<const N: usize>(&self, x: &[f32], indices: &[[usize; N]], s: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(indices.len() * self.h);
        for a in indices {
            let i = (a[0] * s + a[1]) * self.h;
            v.extend_from_slice(&x[i..i + self.h]);
        }
        v
    }
    fn unit(&self, mut x: Vec<f32>) -> Vec<f32> {
        for row in x.chunks_exact_mut(self.proj) {
            let sum = row.iter().map(|v| v * v).sum::<f32>();
            let denom = sum.sqrt().max(1e-12);
            for v in row {
                *v /= denom;
            }
        }
        x
    }
    pub fn forward(&self, r: &Inputs) -> Result<Logits> {
        let all_timing = self.timer();
        let (b, s) = self.validate(r)?;
        self.backend.begin_shape(b, s);
        let tokens = rt_core::checked_size("Vela tokens", &[b, s])?;
        rt_core::checked_size("Vela activations", &[tokens, self.h.max(self.inter), 4])?;
        let row_bytes = rt_core::checked_size("Vela reserved hidden", &[self.h, 10, 4])?
            .checked_add(rt_core::checked_size(
                "Vela reserved intermediate",
                &[self.inter, 3, 4],
            )?)
            .ok_or(rt_core::Error::SizeOverflow("Vela workspace"))?;
        self.backend.validate_workspace(rt_core::checked_size(
            "Vela reserved activations",
            &[tokens, row_bytes],
        )?)?;
        self.backend.reserve(
            rt_core::checked_size("activation elements", &[tokens, self.h])?,
            4,
        )?;
        self.backend.reserve(
            rt_core::checked_size("activation elements", &[tokens, 3, self.h])?,
            2,
        )?;
        self.backend.reserve(
            rt_core::checked_size("activation elements", &[tokens, self.inter])?,
            3,
        )?;
        let emb = self.embedding(r, b, s)?;
        let mut x = self.norm(&emb, "encoder.embeddings.norm")?;
        self.backend.recycle(emb);
        let every = number(&self.enc, "global_attn_every_n_layers")?;
        check_shape!(every > 0, "global attention cadence");
        for i in 0..self.layers {
            let base = format!("encoder.layers.{i}");
            let local = i % every != 0;
            let qkv = if i == 0 {
                self.linear(&x, &format!("{base}.attn.Wqkv"))?
            } else {
                self.linear_norm(
                    &x,
                    &format!("{base}.attn_norm"),
                    &format!("{base}.attn.Wqkv"),
                )?
            };
            let (q, k, v) = self.qkv(&qkv, b, s, local)?;
            self.backend.recycle(qkv);
            let a = self.attention(&q, &k, &v, r, b, s, local)?;
            self.backend.recycle(q);
            self.backend.recycle(k);
            self.backend.recycle(v);
            self.linear_add(&a, &format!("{base}.attn.Wo"), &mut x)?;
            self.backend.recycle(a);
            let geglu = if let Some(left) = self.linears.get(&format!("{base}.mlp.Wi.left")) {
                let norm = self.norm(&x, &format!("{base}.mlp_norm"))?;
                let right = self.linear(&norm, &format!("{base}.mlp.Wi.right"))?;
                let result = self.backend.gated_linear(&norm, &left.weight, &right)?;
                self.backend.recycle(norm);
                self.backend.recycle(right);
                result
            } else {
                let z =
                    self.linear_norm(&x, &format!("{base}.mlp_norm"), &format!("{base}.mlp.Wi"))?;
                check_shape!(z.len() == b * s * 2 * self.inter, "GeGLU width");
                let geglu_timing = self.timer();
                let mut left = self.backend.buffer(rt_core::checked_size(
                    "activation elements",
                    &[b, s, self.inter],
                )?)?;
                let mut right = self.backend.buffer(left.len())?;
                for ((row, l), r) in z
                    .chunks_exact(2 * self.inter)
                    .zip(left.chunks_exact_mut(self.inter))
                    .zip(right.chunks_exact_mut(self.inter))
                {
                    l.copy_from_slice(&row[..self.inter]);
                    r.copy_from_slice(&row[self.inter..]);
                }
                self.backend.recycle(z);
                let gelu = self.backend.gelu_erf(b * s, self.inter, &left)?;
                self.backend.recycle(left);
                let geglu = self.backend.mul(b * s, self.inter, &gelu, &right)?;
                self.backend.recycle(gelu);
                self.backend.recycle(right);
                self.record("geglu_split_with_primitives", geglu_timing);
                geglu
            };
            self.linear_add(&geglu, &format!("{base}.mlp.Wo"), &mut x)?;
            self.backend.recycle(geglu);
        }
        let normalized = self.norm(&x, "encoder.final_norm")?;
        self.backend.recycle(x);
        x = normalized;
        let ho = self.gather(&x, &r.opt_index, s);
        let hq = self.gather(&x, &r.q_index, s);
        let pooled = self.pool(&x, r, b, s);
        let pooled = self.unit(self.linear(&self.norm(&pooled, "norm")?, "w_p")?);
        let uq = self.linear(&self.norm(&hq, "norm")?, "w_q")?;
        let mut u = self.linear(&self.norm(&ho, "norm")?, "w_o")?;
        for (i, o) in r.opt_index.iter().enumerate() {
            for j in 0..self.proj {
                u[i * self.proj + j] += uq[o[2] * self.proj + j];
            }
        }
        let u = self.unit(u);
        let cls = self.linear(&ho, "cls_mlp.0")?;
        let width = self
            .linears
            .get("cls_mlp.0")
            .context("cls_mlp.0")?
            .weight
            .n();
        let cls = self.backend.relu(r.opt_index.len(), width, &cls)?;
        let cls = self.linear(&cls, "cls_mlp.3")?;
        let mut opt_logits = vec![0.; r.opt_index.len()];
        for (i, o) in r.opt_index.iter().enumerate() {
            let dot = (0..self.proj)
                .map(|j| u[i * self.proj + j] * pooled[o[2] * self.proj + j])
                .sum::<f32>();
            opt_logits[i] = dot / self.tau + cls[i];
        }
        let t = self.unit(self.linear(
            &self.norm(&self.gather(&x, &r.unit_index, s), "norm")?,
            "w_t",
        )?);
        let e = self.unit(self.linear(
            &self.norm(&self.gather(&x, &r.ent_index, s), "norm")?,
            "w_e",
        )?);
        let mut sp = self.backend.mm(
            1,
            r.unit_index.len(),
            r.ent_index.len(),
            self.proj,
            0,
            self.proj,
            0,
            self.proj,
            true,
            &t,
            &e,
        )?;
        for v in &mut sp {
            *v /= self.tau_span;
        }
        check_shape!(
            opt_logits.iter().chain(&sp).all(|v| v.is_finite()),
            "nonfinite logits"
        );
        let span_logits = sp
            .chunks_exact(r.ent_index.len())
            .map(|v| v.to_vec())
            .collect();
        self.record("forward", all_timing);
        Ok(Logits {
            opt_logits,
            span_logits,
        })
    }
    pub fn stats(&self) -> Value {
        let mut stats = self.backend.stats();
        stats["rust_profile_s"] =
            serde_json::json!(*self.profile.lock().unwrap_or_else(|e| e.into_inner()));
        stats["load_source"] = serde_json::json!(self.load_source);
        stats["load_profile_s"] = serde_json::json!(self.load_profile);
        stats
    }
    pub fn config(&self) -> &Value {
        &self.cfg
    }
}

struct Assembled {
    norms: HashMap<String, Norm>,
    tau: f32,
    tau_span: f32,
}

fn assemble_raw<B: Hardware>(
    raw: Vec<RawTensor>,
    linears: &mut HashMap<String, Linear<B>>,
    h: usize,
) -> Result<Assembled> {
    let mut map: HashMap<String, Vec<f32>> = raw.into_iter().map(|t| (t.name, t.data)).collect();
    let tau = map
        .get("log_tau")
        .and_then(|v| v.first())
        .copied()
        .context("log_tau")?
        .exp();
    let tau_span = map
        .get("log_tau_span")
        .and_then(|v| v.first())
        .copied()
        .context("log_tau_span")?
        .exp();
    check_shape!(
        tau.is_finite() && tau > 0. && tau_span.is_finite() && tau_span > 0.,
        "temperature"
    );
    for (name, linear) in linears.iter_mut() {
        let bn = format!("{name}.bias");
        if let Some(b) = map.remove(&bn) {
            check_shape!(b.len() == linear.weight.n(), "linear bias width");
            linear.bias = Some(b);
        }
    }
    let keys: Vec<String> = map.keys().cloned().collect();
    let mut norms = HashMap::new();
    for n in keys {
        let Some(base) = n.strip_suffix(".weight") else {
            continue;
        };
        let Some(weight) = map.remove(&n) else {
            continue;
        };
        if weight.len() != h {
            continue;
        }
        let bias = map
            .remove(&format!("{base}.bias"))
            .unwrap_or_else(|| vec![0.; h]);
        check_shape!(bias.len() == h, "norm bias width");
        norms.insert(base.to_string(), Norm { weight, bias });
    }
    Ok(Assembled {
        norms,
        tau,
        tau_span,
    })
}

fn pack_weight<B: Hardware>(
    backend: &B,
    key: &str,
    n: usize,
    k: usize,
    src: &[f32],
    cache: &mut Option<CacheWriter>,
) -> Result<B::Packed> {
    let w = backend.pack(n, k, src)?;
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
    fn finish(
        backend: Arc<B>,
        cfg: Value,
        enc: Value,
        linears: HashMap<String, Linear<B>>,
        parts: Assembled,
        embeddings: EmbeddingTable,
        h: usize,
        nh: usize,
        d: usize,
        inter: usize,
        proj: usize,
        layers: usize,
        max: usize,
        vocab: usize,
        tile: usize,
        eps: f32,
        load_source: &'static str,
        load_profile: BTreeMap<String, f64>,
        profiling: bool,
    ) -> Result<Self> {
        validate_weights(&linears, &parts.norms, h, inter, proj, layers)?;
        backend.validate_workspace(rt_core::checked_size("Vela RoPE tables", &[max, d, 8])?)?;
        let global = Self::rope(
            &enc,
            max,
            d,
            enc["global_rope_theta"].as_f64().context("global theta")?,
        )?;
        let local = Self::rope(
            &enc,
            max,
            d,
            enc["local_rope_theta"].as_f64().context("local theta")?,
        )?;
        Ok(Self {
            embeddings,
            backend,
            cfg,
            enc,
            linears,
            norms: parts.norms,
            global,
            local,
            h,
            nh,
            d,
            inter,
            proj,
            layers,
            max,
            vocab,
            tile,
            tau: parts.tau,
            tau_span: parts.tau_span,
            eps,
            profiling,
            profile: Mutex::new(BTreeMap::new()),
            load_source,
            load_profile,
        })
    }
}

fn validate_weights<B: Hardware>(
    linears: &HashMap<String, Linear<B>>,
    norms: &HashMap<String, Norm>,
    h: usize,
    inter: usize,
    proj: usize,
    layers: usize,
) -> Result<()> {
    let linear = |name: &str, n: usize, k: usize| -> Result<()> {
        let w = &linears
            .get(name)
            .with_context(|| format!("missing Vela tensor {name}"))?
            .weight;
        check_shape!(
            w.n() == n && w.k() == k,
            "Vela tensor {name}: shape [{}, {}], expected [{n}, {k}]",
            w.n(),
            w.k()
        );
        Ok(())
    };
    let norm = |name: &str| -> Result<()> {
        let v = norms
            .get(name)
            .with_context(|| format!("missing Vela tensor {name}"))?;
        check_shape!(
            v.weight.len() == h && v.bias.len() == h,
            "Vela norm {name}: width {}, expected {h}",
            v.weight.len()
        );
        Ok(())
    };
    norm("encoder.embeddings.norm")?;
    norm("encoder.final_norm")?;
    norm("norm")?;
    for name in ["w_p", "w_q", "w_o", "w_t", "w_e"] {
        linear(name, proj, h)?;
    }
    linear("cls_mlp.0", 2 * h, h)?;
    linear("cls_mlp.3", 1, 2 * h)?;
    for layer in 0..layers {
        let base = format!("encoder.layers.{layer}");
        linear(&format!("{base}.attn.Wqkv"), 3 * h, h)?;
        linear(&format!("{base}.attn.Wo"), h, h)?;
        if layer > 0 {
            norm(&format!("{base}.attn_norm"))?;
        }
        norm(&format!("{base}.mlp_norm"))?;
        if linears.contains_key(&format!("{base}.mlp.Wi.left")) {
            for part in ["left", "right"] {
                linear(&format!("{base}.mlp.Wi.{part}"), inter, h)?;
            }
        } else {
            linear(&format!("{base}.mlp.Wi"), 2 * inter, h)?;
        }
        linear(&format!("{base}.mlp.Wo"), h, inter)?;
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
            .join(format!("{}-{}", "vela", std::process::id()));
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
        assert!(mapping.view("x").is_err());
        assert!(mapping.view("missing").is_err());
        assert!(mapping.embedding_span(4, 2).is_err());
        drop(mapping);
        std::fs::remove_file(p).unwrap();
    }
    #[test]
    fn missing_projection_weights_are_rejected_without_assets() {
        let linears: HashMap<String, Linear<rt_onednn::OnednnCpu>> = HashMap::new();
        assert!(validate_weights(&linears, &HashMap::new(), 2, 2, 2, 1).is_err());
    }
}
