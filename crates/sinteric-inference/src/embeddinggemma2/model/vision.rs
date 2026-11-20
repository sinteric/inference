//! Gemma 4 vision tower and the EG2 image projector: patches in, one
//! 512-wide soft token per 3x3 patch block out.
//!
//! ViT with 16 layers (768 wide, 12 heads x 64, gelu-tanh MLP 3072) in the
//! text tower's norm order, bidirectional attention over the real patches
//! with scale 1, and 2-D RoPE (theta 100): head dims `[0, 32)` rotate by the
//! patch column, `[32, 64)` by the row, each half by its own rotate-half.
//! Q/K rows are stored with the middle two quarters swapped, which turns
//! that into one full-width rotate-half (the dot products are unchanged),
//! so the text tower's fused QKV split/norm/RoPE pass serves both.
//! Then: 3x3 average pool in row-major block order, `* sqrt(768)`, unscaled
//! RMSNorm and `embed_vision` (768 -> 512).

use crate::cpu::kernels::{add_rmsnorm, bf16_bits_to_f32, qkv_split_norm_rope, rmsnorm_nthreads};
use crate::cpu::onednn::{Attention, OnednnCpu};
use crate::{check_shape, Result};

use super::tower::{load_tower, Packer, TowerParts, TowerSource, Views};
use super::Precision;
use crate::embeddinggemma2::config::{HIDDEN, RMS_EPS};
use crate::embeddinggemma2::media::{ImagePatches, PATCH_PIXELS, POOL};
use crate::tensor_file::{Mapped, TensorFile};

pub(crate) const WIDTH: usize = 768;
const LAYERS: usize = 16;
const HEADS: usize = 12;
const HEAD_DIM: usize = 64;
const INTER: usize = 3072;
const POSITIONS: usize = 10_240;
const THETA: f32 = 100.0;
const POS_TABLE: &str = "vision_tower.patch_embedder.position_embedding_table";

/// Stored Q/K head dim `j` holds checkpoint dim `rope_src(j)`.
fn rope_src(j: usize) -> usize {
    match j {
        16..32 => j + 16,
        32..48 => j - 16,
        _ => j,
    }
}

fn layer(i: usize) -> String {
    format!("vision_tower.encoder.layers.{i}")
}

pub(crate) struct VisionTower {
    parts: TowerParts,
    positions: Mapped<u16>,
    inv_freq: [f32; HEAD_DIM / 4],
}

fn build(v: &Views<'_>, p: &mut Packer<'_>) -> Result<()> {
    let proj = "vision_tower.patch_embedder.input_proj.weight";
    p.pack(
        "vision_tower.patch_embedder.input_proj",
        WIDTH,
        PATCH_PIXELS,
        v.bf16(proj, &[WIDTH, PATCH_PIXELS])?,
    )?;
    v.release(proj);
    for i in 0..LAYERS {
        let base = layer(i);
        let mut qkv = Vec::with_capacity(3 * WIDTH * WIDTH);
        for part in ["q", "k", "v"] {
            let name = format!("{base}.self_attn.{part}_proj.linear.weight");
            let w = v.bf16(&name, &[WIDTH, WIDTH])?;
            for row in 0..WIDTH {
                let src = if part == "v" {
                    row
                } else {
                    row - row % HEAD_DIM + rope_src(row % HEAD_DIM)
                };
                qkv.extend_from_slice(&w[src * WIDTH..(src + 1) * WIDTH]);
            }
            v.release(&name);
        }
        p.pack(&format!("{base}.self_attn.qkv"), 3 * WIDTH, WIDTH, &qkv)?;
        for part in ["q", "k"] {
            let name = format!("{base}.self_attn.{part}_norm.weight");
            let w = v.f32(&name, &[HEAD_DIM])?;
            p.raw(&name, (0..HEAD_DIM).map(|j| w[rope_src(j)]).collect());
        }
        for (name, n, k) in [
            ("self_attn.o_proj", WIDTH, WIDTH),
            ("mlp.gate_proj", INTER, WIDTH),
            ("mlp.up_proj", INTER, WIDTH),
            ("mlp.down_proj", WIDTH, INTER),
        ] {
            let t = format!("{base}.{name}.linear.weight");
            p.pack(&format!("{base}.{name}"), n, k, v.bf16(&t, &[n, k])?)?;
            v.release(&t);
        }
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            let name = format!("{base}.{norm}.weight");
            p.raw(&name, v.f32(&name, &[WIDTH])?);
        }
    }
    let name = "embed_vision.embedding_projection.weight";
    p.pack(
        "embed_vision.embedding_projection",
        HIDDEN,
        WIDTH,
        v.bf16(name, &[HIDDEN, WIDTH])?,
    )?;
    Ok(())
}

impl VisionTower {
    pub fn load(src: &TowerSource<'_>, backend: &OnednnCpu, precision: Precision) -> Result<Self> {
        let tag = if precision == Precision::Fp32 {
            "vision.fp32"
        } else {
            "vision"
        };
        let parts = load_tower(src, backend, precision, tag, build)?;
        parts.check_linear(
            "vision_tower.patch_embedder.input_proj",
            WIDTH,
            PATCH_PIXELS,
        )?;
        parts.check_linear("embed_vision.embedding_projection", HIDDEN, WIDTH)?;
        for i in 0..LAYERS {
            let base = layer(i);
            parts.check_linear(&format!("{base}.self_attn.qkv"), 3 * WIDTH, WIDTH)?;
            parts.check_linear(&format!("{base}.self_attn.o_proj"), WIDTH, WIDTH)?;
            parts.check_linear(&format!("{base}.mlp.gate_proj"), INTER, WIDTH)?;
            parts.check_linear(&format!("{base}.mlp.up_proj"), INTER, WIDTH)?;
            parts.check_linear(&format!("{base}.mlp.down_proj"), WIDTH, INTER)?;
            for part in ["q", "k"] {
                parts.raw(&format!("{base}.self_attn.{part}_norm.weight"), HEAD_DIM)?;
            }
        }
        let positions =
            TensorFile::open(src.st_path)?.into_view(POS_TABLE, &[2, POSITIONS, WIDTH])?;
        let mut inv_freq = [0f32; HEAD_DIM / 4];
        for (i, f) in inv_freq.iter_mut().enumerate() {
            // `1 / theta^(arange(0, 32, 2) / 32)` in f32.
            *f = 1.0 / THETA.powf((2 * i) as f32 / (HEAD_DIM / 2) as f32);
        }
        Ok(Self {
            parts,
            positions,
            inv_freq,
        })
    }

    pub fn load_source(&self) -> &crate::cpu::runtime::WeightSource {
        &self.parts.source
    }

    fn lin(&self, hw: &OnednnCpu, precision: Precision, x: &[f32], name: &str) -> Result<Vec<f32>> {
        self.parts.matmul(hw, precision, x, name)
    }

    /// Per-patch cos/sin rows in the stored (quarter-swapped) layout.
    fn rope(&self, img: &ImagePatches) -> (Vec<f32>, Vec<f32>) {
        let n = img.patches();
        let mut cos = vec![0f32; n * HEAD_DIM];
        let mut sin = vec![0f32; n * HEAD_DIM];
        for k in 0..n {
            let (x, y) = ((k % img.pw) as f32, (k / img.pw) as f32);
            for j in 0..HEAD_DIM {
                let pos = if j % 32 < 16 { x } else { y };
                let a = pos * self.inv_freq[j % 16];
                cos[k * HEAD_DIM + j] = a.cos();
                sin[k * HEAD_DIM + j] = a.sin();
            }
        }
        (cos, sin)
    }

    /// Soft tokens `[img.soft_tokens(), 512]` for one image.
    pub fn forward(
        &self,
        hw: &OnednnCpu,
        precision: Precision,
        tile: usize,
        img: &ImagePatches,
    ) -> Result<Vec<f32>> {
        let n = img.patches();
        check_shape!(
            n > 0
                && img.pw.is_multiple_of(POOL)
                && img.ph.is_multiple_of(POOL)
                && img.pixels.len() == n * PATCH_PIXELS
                && img.pw.max(img.ph) <= POSITIONS,
            "vision patches {}x{}",
            img.pw,
            img.ph
        );
        let threads = hw.threads();
        hw.begin_shape(1, n);
        hw.validate_workspace(crate::checked_size(
            "EG2 vision activations",
            &[n, WIDTH * 8 + INTER * 3 + n * HEADS.min(4), 4],
        )?)?;
        let mut x = self.lin(
            hw,
            precision,
            &img.pixels,
            "vision_tower.patch_embedder.input_proj",
        )?;
        let table = self.positions.as_slice();
        for (k, row) in x.as_chunks_mut::<WIDTH>().0.iter_mut().enumerate() {
            let xs = &table[(k % img.pw) * WIDTH..][..WIDTH];
            let ys = &table[(POSITIONS + k / img.pw) * WIDTH..][..WIDTH];
            for ((v, &a), &b) in row.iter_mut().zip(xs).zip(ys) {
                *v += bf16_bits_to_f32(a) + bf16_bits_to_f32(b);
            }
        }
        let (cos, sin) = self.rope(img);
        let raw = |name: &str, len| self.parts.raw(name, len);
        let mut h = hw.buffer(x.len())?;
        for i in 0..LAYERS {
            let base = layer(i);
            rmsnorm_nthreads(
                &x,
                WIDTH,
                Some(raw(&format!("{base}.input_layernorm.weight"), WIDTH)?),
                RMS_EPS,
                &mut h,
                threads,
            )?;
            let fused = self.lin(hw, precision, &h, &format!("{base}.self_attn.qkv"))?;
            let mut q = hw.buffer(n * WIDTH)?;
            let mut k = hw.buffer(n * WIDTH)?;
            let mut v = hw.buffer(n * WIDTH)?;
            qkv_split_norm_rope(
                &fused,
                &mut q,
                &mut k,
                &mut v,
                1,
                n,
                HEADS,
                HEADS,
                HEAD_DIM,
                Some(raw(&format!("{base}.self_attn.q_norm.weight"), HEAD_DIM)?),
                Some(raw(&format!("{base}.self_attn.k_norm.weight"), HEAD_DIM)?),
                RMS_EPS,
                (&cos, &sin),
                threads,
            )?;
            hw.recycle(fused);
            let shape = Attention {
                batch: 1,
                seq: n,
                q_heads: HEADS,
                kv_heads: HEADS,
                head_dim: HEAD_DIM,
                tile,
                window: None,
                scale: 1.0,
                padding: &[vec![]],
            };
            let attn = if precision == Precision::Fp32 {
                hw.attention_gqa_f32(&shape, &q, &k, &v)?
            } else {
                hw.attention_gqa(&shape, &q, &k, &v)?
            };
            hw.recycle(q);
            hw.recycle(k);
            hw.recycle(v);
            check_shape!(attn.len() >= n * WIDTH, "vision attention rows");
            let o = self.lin(
                hw,
                precision,
                &attn[..n * WIDTH],
                &format!("{base}.self_attn.o_proj"),
            )?;
            hw.recycle(attn);
            add_rmsnorm(
                &mut x,
                &o,
                WIDTH,
                Some(raw(
                    &format!("{base}.post_attention_layernorm.weight"),
                    WIDTH,
                )?),
                1.0,
                RMS_EPS,
                threads,
            )?;
            hw.recycle(o);
            rmsnorm_nthreads(
                &x,
                WIDTH,
                Some(raw(
                    &format!("{base}.pre_feedforward_layernorm.weight"),
                    WIDTH,
                )?),
                RMS_EPS,
                &mut h,
                threads,
            )?;
            let up = self.lin(hw, precision, &h, &format!("{base}.mlp.up_proj"))?;
            let gated = hw.gated_linear_tanh(
                &h,
                self.parts.linear(&format!("{base}.mlp.gate_proj"))?,
                &up,
            )?;
            hw.recycle(up);
            let down = self.lin(hw, precision, &gated, &format!("{base}.mlp.down_proj"))?;
            hw.recycle(gated);
            add_rmsnorm(
                &mut x,
                &down,
                WIDTH,
                Some(raw(
                    &format!("{base}.post_feedforward_layernorm.weight"),
                    WIDTH,
                )?),
                1.0,
                RMS_EPS,
                threads,
            )?;
            hw.recycle(down);
        }
        hw.recycle(h);
        // 3x3 average pool (row-major blocks), then `* sqrt(768)` in f32.
        let (bw, bh) = (img.pw / POOL, img.ph / POOL);
        let mut pooled = vec![0f32; bw * bh * WIDTH];
        let ninth = 1.0f32 / (POOL * POOL) as f32;
        let root = (WIDTH as f64).sqrt() as f32;
        for (b, out) in pooled.as_chunks_mut::<WIDTH>().0.iter_mut().enumerate() {
            let (bx, by) = (b % bw, b / bw);
            for dy in 0..POOL {
                for dx in 0..POOL {
                    let k = (by * POOL + dy) * img.pw + bx * POOL + dx;
                    for (o, &v) in out.iter_mut().zip(&x[k * WIDTH..(k + 1) * WIDTH]) {
                        *o += v * ninth;
                    }
                }
            }
            for o in out.iter_mut() {
                *o *= root;
            }
        }
        hw.recycle(x);
        let mut normed = vec![0f32; pooled.len()];
        rmsnorm_nthreads(&pooled, WIDTH, None, RMS_EPS, &mut normed, threads)?;
        self.lin(hw, precision, &normed, "embed_vision.embedding_projection")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarter_swap_pairs_each_half() {
        // Full rotate-half pairs (j, j + 32) are the checkpoint's per-axis
        // pairs (i, i + 16) in [0, 32) and (32 + i, 48 + i).
        for j in 0..32 {
            let (a, b) = (rope_src(j), rope_src(j + 32));
            assert_eq!(b, a + 16, "{j}");
            assert_eq!(a / 32, b / 32);
        }
        let mut seen: Vec<usize> = (0..HEAD_DIM).map(rope_src).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..HEAD_DIM).collect::<Vec<_>>());
    }
}
