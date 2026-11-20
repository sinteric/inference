//! Loading the text tower: the packed-weight cache when it matches the
//! checkpoint, otherwise every text-tower matrix packed from the mapped
//! safetensors file (Q, K and V as one matmul), the cache written as they
//! pack, and the norms and per-layer scalars checked against the config.

use super::{Linear, Model, Precision};
use crate::cpu::kernels::{bf16_bits_to_f32, DefaultRope};
use crate::cpu::onednn::{
    apply_cache, cache_path_with, file_stamp, pinned_dnnl, CacheMeta, CacheOptions, CacheWriter,
    OnednnCpu, PackedWeight, RawTensor,
};
use crate::cpu::runtime::{CacheWrite, WeightSource};
use crate::embeddinggemma2::config::{
    head_dim, n_kv, EMBED_DIM, EMB_NAME, GLOBAL_THETA, HIDDEN, LOCAL_THETA, MAX_SEQ, N_LAYERS,
    N_QUERY_HEADS, PLE_WIDTH, VOCAB,
};
use crate::tensor_file::{Mapped, TensorFile};
use crate::{check_files, check_shape, Error, Result, ResultContext as Context};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// The zero-copy BF16 token table of the checkpoint at `path`.
fn embedding_table(path: &Path) -> Result<Mapped<u16>> {
    TensorFile::open(path)?.into_view(EMB_NAME, &[VOCAB, HIDDEN])
}

pub(super) fn expected_weight_pin(manifest: Option<&Path>) -> Result<(String, u64)> {
    let m = match manifest {
        Some(path) => crate::assets::Manifest::load(path)?,
        None => crate::assets::Manifest::parse(crate::embeddinggemma2::ASSET_MANIFEST)?,
    };
    let f = m
        .file("model.safetensors")
        .ok_or_else(|| crate::Error::model_files("model.safetensors not in manifest"))?;
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

pub(super) fn pack_bf16(
    backend: &OnednnCpu,
    precision: Precision,
    key: &str,
    n: usize,
    k: usize,
    src: &[u16],
    cache: &mut Option<CacheWriter>,
) -> Result<PackedWeight> {
    let w = if precision == Precision::Fp32 {
        backend.pack_bf16_f32(n, k, src)?
    } else {
        backend.pack_bf16(n, k, src)?
    };
    if let Some(c) = cache.as_mut() {
        c.record(backend, &w, key, n, k)?;
    }
    Ok(w)
}

struct Assembled {
    norms: HashMap<String, Vec<f32>>,
    scalars: Vec<f32>,
}

fn assemble_raw(raw: Vec<RawTensor>) -> Result<Assembled> {
    let mut map: HashMap<String, Vec<f32>> = raw.into_iter().map(|t| (t.name, t.data)).collect();
    let mut scalars = vec![1.0f32; N_LAYERS];
    for (i, scalar) in scalars.iter_mut().enumerate() {
        let key = format!("language_model.layers.{i}.layer_scalar");
        {
            let v = map
                .remove(&key)
                .with_context(|| format!("missing tensor {key}"))?;
            check_files!(v.len() == 1, "layer_scalar rank {key}");
            *scalar = v[0];
        }
    }
    Ok(Assembled {
        norms: map,
        scalars,
    })
}

fn validate_weights(
    linears: &HashMap<String, Linear>,
    norms: &HashMap<String, Vec<f32>>,
) -> Result<()> {
    let linear = |name: &str, n: usize, k: usize| -> Result<()> {
        let w = &linears
            .get(name)
            .with_context(|| format!("missing EG2 tensor {name}"))?
            .weight;
        check_files!(
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
        check_files!(
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
                crate::embeddinggemma2::config::INTERMEDIATE,
                HIDDEN,
            )?;
        }
        linear(
            &format!("{base}.mlp.down_proj"),
            HIDDEN,
            crate::embeddinggemma2::config::INTERMEDIATE,
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

impl Model {
    #[cfg(test)]
    pub fn load(path: &Path, backend: Arc<OnednnCpu>, tile: usize) -> Result<Self> {
        Self::load_with(path, backend, tile, &CacheOptions::default())
    }

    #[cfg(test)]
    pub fn load_with(
        path: &Path,
        backend: Arc<OnednnCpu>,
        tile: usize,
        cache: &CacheOptions,
    ) -> Result<Self> {
        Self::load_with_precision(path, backend, tile, cache, Precision::Bf16)
    }

    pub fn load_with_precision(
        path: &Path,
        backend: Arc<OnednnCpu>,
        tile: usize,
        cache: &CacheOptions,
        precision: Precision,
    ) -> Result<Self> {
        Self::load_impl(path, backend, tile, cache, precision).map_err(|e| e.at(path))
    }

    fn load_impl(
        path: &Path,
        backend: Arc<OnednnCpu>,
        tile: usize,
        cache: &CacheOptions,
        precision: Precision,
    ) -> Result<Self> {
        check_shape!(tile > 0, "tile");
        let st_path = path.join("model.safetensors");
        let (sha, pinned_size) = expected_weight_pin(cache.manifest.as_deref())?;
        let mut cache_file = cache_path_with(&st_path, backend.cpu_isa(), cache.file.as_deref());
        if precision == Precision::Fp32 {
            // FP32 packs a different blocked layout; keep both caches.
            let mut name = cache_file.into_os_string();
            name.push(".fp32");
            cache_file = name.into();
        }
        let mut load = Load {
            backend,
            tile,
            precision,
            st_path,
            cache_file,
            sha,
            profile: BTreeMap::new(),
        };
        if cache.enabled && !cache.verify_weights {
            if let Some((linears, parts, embeddings)) = load.read_cache() {
                let source = WeightSource::Cache {
                    path: load.cache_file.clone(),
                };
                return load.finish(linears, parts, embeddings, source);
            }
        }
        load.pack_checkpoint(cache, pinned_size)
    }
}

/// One load of the text tower: what it reads, how it packs, and how long
/// each stage took.
struct Load {
    backend: Arc<OnednnCpu>,
    tile: usize,
    precision: Precision,
    st_path: std::path::PathBuf,
    cache_file: std::path::PathBuf,
    sha: String,
    profile: BTreeMap<String, f64>,
}

/// Packed linears by name, as the model looks them up.
type Linears = HashMap<String, Linear>;

impl Load {
    /// The packed weights from the cache when it matches the checkpoint; any
    /// failure falls back to packing from the checkpoint.
    fn read_cache(&mut self) -> Option<(Linears, Assembled, Mapped<u16>)> {
        let t = Instant::now();
        let stamp = file_stamp(&self.st_path).ok();
        let mut linears = HashMap::new();
        let (backend, precision) = (&self.backend, self.precision);
        let applied = apply_cache(
            &self.cache_file,
            &self.sha,
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
        self.profile
            .insert("cache_read_s".into(), t.elapsed().as_secs_f64());
        let raw = applied.ok()??;
        let (Ok(parts), Ok(embeddings)) = (assemble_raw(raw), embedding_table(&self.st_path))
        else {
            return None;
        };
        Some((linears, parts, embeddings))
    }

    /// Every text-tower matrix packed from the checkpoint, the cache written
    /// as they pack when `cache` asks for it.
    fn pack_checkpoint(mut self, cache: &CacheOptions, pinned_size: u64) -> Result<Model> {
        let st_path = self.st_path.clone();
        let stamp = file_stamp(&st_path).with_context(|| st_path.display().to_string())?;
        check_files!(
            stamp.size == pinned_size,
            "safetensors size {} != pinned {pinned_size}",
            stamp.size
        );
        if cache.verify_weights {
            let t = Instant::now();
            crate::assets::verify_sha256_mmap(&st_path, &self.sha)?;
            self.profile
                .insert("verify_s".into(), t.elapsed().as_secs_f64());
        }

        let t = Instant::now();
        let file = TensorFile::open(&st_path)?;
        self.profile
            .insert("mmap_s".into(), t.elapsed().as_secs_f64());
        let t = Instant::now();
        let mut cache_w = if cache.enabled {
            Some(CacheWriter::start(
                &self.cache_file,
                &CacheMeta {
                    source_sha256: self.sha.clone(),
                    source_size: stamp.size,
                    source_mtime_secs: stamp.mtime_secs,
                    dnnl: pinned_dnnl(),
                    cpu_isa: self.backend.cpu_isa(),
                },
            ))
        } else {
            None
        };
        let (linears, raw_out) = self.pack_all(&file, &mut cache_w)?;
        self.profile
            .insert("pack_s".into(), t.elapsed().as_secs_f64());
        let embeddings = file.into_view(EMB_NAME, &[VOCAB, HIDDEN])?;
        let cache_write = match cache_w {
            Some(w) => {
                let t = Instant::now();
                let written = w.close(&raw_out);
                self.profile
                    .insert("cache_write_s".into(), t.elapsed().as_secs_f64());
                written
            }
            None => CacheWrite::Off,
        };
        let parts = assemble_raw(raw_out)?;
        self.finish(
            linears,
            parts,
            embeddings,
            WeightSource::Safetensors { cache: cache_write },
        )
    }

    /// Pack every text-tower matrix of `file` (Q, K and V of a layer as one)
    /// and widen its vectors to FP32.
    fn pack_all(
        &self,
        file: &TensorFile,
        cache_w: &mut Option<CacheWriter>,
    ) -> Result<(Linears, Vec<RawTensor>)> {
        let mut linears = HashMap::new();
        let mut raw_out: Vec<RawTensor> = Vec::new();
        let names = file.index().names().map(str::to_owned).collect::<Vec<_>>();
        let mut qkv: HashMap<usize, [Part; 3]> = HashMap::new();
        for n in names {
            if n == EMB_NAME || skip_tensor(&n) || !n.starts_with("language_model.") {
                continue;
            }
            let shape = file.index().shape(&n)?;
            if n.ends_with(".weight") && shape.len() == 2 {
                let bits = file.view::<u16>(&n, shape)?;
                if let Some((layer, part)) = qkv_part(&n) {
                    let slot = qkv.entry(layer).or_insert([None, None, None]);
                    slot[part] = Some((shape[0], shape[1], bits.to_vec()));
                    file.release(&n);
                    continue;
                }
                let key = n
                    .strip_suffix(".weight")
                    .context("weight tensor suffix")?
                    .to_string();
                let weight = pack_bf16(
                    &self.backend,
                    self.precision,
                    &key,
                    shape[0],
                    shape[1],
                    bits,
                    cache_w,
                )?;
                linears.insert(key, Linear { weight });
                file.release(&n);
            } else if shape.len() == 1 {
                let bits = file.view::<u16>(&n, shape)?;
                let mut data = vec![0f32; bits.len()];
                for (d, &b) in data.iter_mut().zip(bits) {
                    *d = bf16_bits_to_f32(b);
                }
                raw_out.push(RawTensor {
                    name: n.clone(),
                    data,
                });
                file.release(&n);
            } else {
                return Err(Error::model_files(format!(
                    "unexpected text-tower tensor {n} shape {shape:?}"
                )));
            }
        }
        for (layer, parts) in qkv {
            let (key, weight) = self.pack_qkv(layer, &parts, cache_w)?;
            linears.insert(key, Linear { weight });
        }
        Ok((linears, raw_out))
    }

    /// Layer `layer`'s Q, K and V packed as one matrix, and its name.
    fn pack_qkv(
        &self,
        layer: usize,
        parts: &[Part; 3],
        cache_w: &mut Option<CacheWriter>,
    ) -> Result<(String, PackedWeight)> {
        let part = |i: usize, name: &str| {
            parts[i]
                .as_ref()
                .ok_or_else(|| Error::model_files(format!("missing {name} for layer {layer}")))
        };
        let (qn, qk, q) = part(0, "q_proj")?;
        let (kn, kk, k) = part(1, "k_proj")?;
        let (vn, vk, v) = part(2, "v_proj")?;
        check_files!(*qk == *kk && *kk == *vk, "qkv input width layer {layer}");
        let mut cat = Vec::with_capacity(q.len() + k.len() + v.len());
        cat.extend_from_slice(q);
        cat.extend_from_slice(k);
        cat.extend_from_slice(v);
        let key = format!("language_model.layers.{layer}.self_attn.qkv_proj");
        let weight = pack_bf16(
            &self.backend,
            self.precision,
            &key,
            qn + kn + vn,
            *qk,
            &cat,
            cache_w,
        )?;
        Ok((key, weight))
    }

    fn finish(
        self,
        linears: Linears,
        parts: Assembled,
        embeddings: Mapped<u16>,
        load_source: WeightSource,
    ) -> Result<Model> {
        validate_weights(&linears, &parts.norms)?;
        check_files!(parts.scalars.len() == N_LAYERS, "layer_scalar count");
        Ok(Model {
            embeddings,
            vision: None,
            audio: None,
            backend: self.backend,
            linears,
            norms: parts.norms,
            scalars: parts.scalars,
            local_rope: DefaultRope::new(head_dim(0), MAX_SEQ, LOCAL_THETA)?,
            global_rope: DefaultRope::new(head_dim(5), MAX_SEQ, GLOBAL_THETA)?,
            tile: self.tile,
            max: MAX_SEQ,
            precision: self.precision,
            load_source,
            load_profile: self.profile,
        })
    }
}

/// One of a layer's Q, K and V as it arrives: (rows, cols, BF16 data).
type Part = Option<(usize, usize, Vec<u16>)>;

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
        let embedded =
            crate::assets::Manifest::parse(crate::embeddinggemma2::ASSET_MANIFEST).unwrap();
        let file = embedded.file("model.safetensors").unwrap();
        assert_eq!(
            expected_weight_pin(None).unwrap(),
            (file.sha256.clone(), file.size)
        );
        let missing = test_path("round2-missing-manifest.toml");
        let _ = std::fs::remove_file(&missing);
        let err = expected_weight_pin(Some(&missing)).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::ModelFiles);
        assert!(
            err.to_string().contains(&missing.display().to_string()),
            "{err}"
        );
        let explicit = test_path("round2-explicit-manifest.toml");
        std::fs::write(
            &explicit,
            crate::embeddinggemma2::ASSET_MANIFEST.replace(&file.sha256, &"a".repeat(64)),
        )
        .unwrap();
        assert_eq!(
            expected_weight_pin(Some(&explicit)).unwrap(),
            ("a".repeat(64), file.size)
        );
        std::fs::write(&explicit, "malformed manifest").unwrap();
        let err = expected_weight_pin(Some(&explicit)).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::ModelFiles);
        assert!(
            err.to_string().contains(&explicit.display().to_string()),
            "{err}"
        );
        std::fs::remove_file(explicit).unwrap();
    }

    #[test]
    fn missing_projection_weights_are_rejected_without_assets() {
        let linears: HashMap<String, Linear> = HashMap::new();
        assert!(validate_weights(&linears, &HashMap::new()).is_err());
    }
}
