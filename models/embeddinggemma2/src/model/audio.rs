//! Gemma 4 audio tower (USM conformer) and the EG2 audio projector: log-mel
//! frames in, one 512-wide soft token per 4 frames out.
//!
//! Subsampling: two 3x3 stride-2 convolutions over (time, mel), each with a
//! channel LayerNorm and ReLU (128 then 32 channels), flattened `(mel, ch)`
//! and projected to 1024. Then 12 conformer layers: half-step FFN, chunked
//! local self-attention (each query sees itself and the 11 frames before
//! it, with a learned relative-position key, logit softcap 50), a light
//! conv block (GLU, causal depthwise conv k=5) and a second half-step FFN.
//! Most linears clamp their input and output to stored scalar bounds.
//! Then `output_proj` (1024 -> 1536, bias), unscaled RMSNorm and
//! `embed_audio` (1536 -> 512).
//!
//! Only real frames enter the tower. The reference masks padded frames out
//! of every block, so running the live prefix alone is the same result.

use rt_backend::Hardware;
use rt_core::{check_shape, Result};
use rt_kernels::{add_rmsnorm, rmsnorm_inplace, rmsnorm_nthreads};

use super::tower::{load_tower, par_rows, Packer, TowerParts, TowerSource, Views};
use super::Precision;
use crate::config::{HIDDEN, RMS_EPS};
use crate::media::{AudioFeatures, MEL_BINS};

const WIDTH: usize = 1024;
const LAYERS: usize = 12;
const HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const FF: usize = 4096;
const C0: usize = 128;
const C1: usize = 32;
/// Mel bins after each stride-2 convolution.
const F0: usize = MEL_BINS / 2;
const F1: usize = F0 / 2;
/// Keys per query: the query frame and the 11 before it.
const WINDOW: usize = 12;
const KERNEL: usize = 5;
const OUT: usize = 1536;
const SOFTCAP: f32 = 50.0;
const RESIDUAL: f32 = 0.5;
/// Sinusoid timescales of the relative position encoding (`hidden / 2`).
const TIMESCALES: usize = WIDTH / 2;

const SSCP: &str = "audio_tower.subsample_conv_projection";

fn layer(i: usize) -> String {
    format!("audio_tower.layers.{i}")
}

/// Every clipped linear of one layer, `(name, out, in)`.
const CLIPPED: [(&str, usize, usize); 9] = [
    ("feed_forward1.ffw_layer_1", FF, WIDTH),
    ("feed_forward1.ffw_layer_2", WIDTH, FF),
    ("feed_forward2.ffw_layer_1", FF, WIDTH),
    ("feed_forward2.ffw_layer_2", WIDTH, FF),
    ("self_attn.q_proj", WIDTH, WIDTH),
    ("self_attn.k_proj", WIDTH, WIDTH),
    ("self_attn.v_proj", WIDTH, WIDTH),
    ("self_attn.post", WIDTH, WIDTH),
    ("lconv1d.linear_start", 2 * WIDTH, WIDTH),
];

const NORMS: [&str; 9] = [
    "feed_forward1.pre_layer_norm",
    "feed_forward1.post_layer_norm",
    "feed_forward2.pre_layer_norm",
    "feed_forward2.post_layer_norm",
    "lconv1d.pre_layer_norm",
    "lconv1d.conv_norm",
    "norm_pre_attn",
    "norm_post_attn",
    "norm_out",
];

pub(crate) struct AudioTower<B: Hardware> {
    parts: TowerParts<B>,
}

/// Relative-position keys `[WINDOW, WIDTH]`: `relative_k_proj` of the
/// `[sin, cos]` encoding of each query-to-key distance `0..WINDOW`.
fn relative_keys(w: &[f32]) -> Vec<f32> {
    let inc = (10_000f64).ln() / (TIMESCALES - 1) as f64;
    let inv: Vec<f32> = (0..TIMESCALES)
        .map(|i| (i as f32 * -(inc as f32)).exp())
        .collect();
    let mut out = vec![0f32; WINDOW * WIDTH];
    for (dist, row) in out.as_chunks_mut::<WIDTH>().0.iter_mut().enumerate() {
        let mut pe = vec![0f32; WIDTH];
        for i in 0..TIMESCALES {
            let t = dist as f32 * inv[i];
            pe[i] = t.sin();
            pe[TIMESCALES + i] = t.cos();
        }
        for (o, r) in row.iter_mut().enumerate() {
            let wr = &w[o * WIDTH..(o + 1) * WIDTH];
            *r = wr
                .iter()
                .zip(&pe)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum::<f64>() as f32;
        }
    }
    out
}

fn build<B: Hardware>(v: &Views<'_>, p: &mut Packer<'_, B>) -> Result<()> {
    let name = format!("{SSCP}.layer0.conv.weight");
    p.raw(&name, v.f32(&name, &[C0, 1, 3, 3])?);
    // layer1 as a matmul over im2col rows `[(kh * 3 + kw) * C0 + c]`.
    let name = format!("{SSCP}.layer1.conv.weight");
    let mut cols = vec![0u16; C1 * 9 * C0];
    let bits = v.bf16(&name, &[C1, C0, 3, 3])?;
    for o in 0..C1 {
        for c in 0..C0 {
            for k in 0..9 {
                cols[o * 9 * C0 + k * C0 + c] = bits[(o * C0 + c) * 9 + k];
            }
        }
    }
    p.pack(&format!("{SSCP}.layer1.conv"), C1, 9 * C0, &cols)?;
    for (l, c) in [("layer0", C0), ("layer1", C1)] {
        let name = format!("{SSCP}.{l}.norm.weight");
        p.raw(&name, v.f32(&name, &[c])?);
    }
    let name = format!("{SSCP}.input_proj_linear.weight");
    p.pack(
        &format!("{SSCP}.input_proj_linear"),
        WIDTH,
        F1 * C1,
        v.bf16(&name, &[WIDTH, F1 * C1])?,
    )?;
    for i in 0..LAYERS {
        let base = layer(i);
        for (lin, n, k) in CLIPPED
            .iter()
            .copied()
            .chain([("lconv1d.linear_end", WIDTH, WIDTH)])
        {
            let t = format!("{base}.{lin}.linear.weight");
            p.pack(&format!("{base}.{lin}"), n, k, v.bf16(&t, &[n, k])?)?;
            v.release(&t);
            let mut clip = Vec::with_capacity(4);
            for b in ["input_min", "input_max", "output_min", "output_max"] {
                clip.extend(v.f32(&format!("{base}.{lin}.{b}"), &[])?);
            }
            p.raw(&format!("{base}.{lin}.clip"), clip);
        }
        for norm in NORMS {
            let name = format!("{base}.{norm}.weight");
            p.raw(&name, v.f32(&name, &[WIDTH])?);
        }
        let name = format!("{base}.self_attn.per_dim_scale");
        p.raw(&name, v.f32(&name, &[HEAD_DIM])?);
        let name = format!("{base}.lconv1d.depthwise_conv1d.weight");
        p.raw(&name, v.f32(&name, &[WIDTH, 1, KERNEL])?);
        let name = format!("{base}.self_attn.relative_k_proj.weight");
        p.raw(
            &format!("{base}.self_attn.relative_keys"),
            relative_keys(&v.f32(&name, &[WIDTH, WIDTH])?),
        );
        v.release(&name);
    }
    let name = "audio_tower.output_proj.weight";
    p.pack(
        "audio_tower.output_proj",
        OUT,
        WIDTH,
        v.bf16(name, &[OUT, WIDTH])?,
    )?;
    let name = "audio_tower.output_proj.bias";
    p.raw(name, v.f32(name, &[OUT])?);
    let name = "embed_audio.embedding_projection.weight";
    p.pack(
        "embed_audio.embedding_projection",
        HIDDEN,
        OUT,
        v.bf16(name, &[HIDDEN, OUT])?,
    )?;
    Ok(())
}

fn clamp(x: &mut [f32], lo: f32, hi: f32, threads: usize) {
    if x.iter().all(|&v| v >= lo && v <= hi) {
        return;
    }
    par_rows(threads, x, 1, 4096, |_, v| v[0] = v[0].clamp(lo, hi));
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    // torch softplus (beta 1, threshold 20).
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

impl<B: Hardware> AudioTower<B> {
    pub fn load(src: &TowerSource<'_>, backend: &B, precision: Precision) -> Result<Self> {
        let tag = if precision == Precision::Fp32 {
            "audio.fp32"
        } else {
            "audio"
        };
        let parts = load_tower(src, backend, precision, tag, build)?;
        parts.raw(&format!("{SSCP}.layer0.conv.weight"), C0 * 9)?;
        parts.check_linear(&format!("{SSCP}.layer1.conv"), C1, 9 * C0)?;
        parts.check_linear(&format!("{SSCP}.input_proj_linear"), WIDTH, F1 * C1)?;
        for (l, c) in [("layer0", C0), ("layer1", C1)] {
            parts.raw(&format!("{SSCP}.{l}.norm.weight"), c)?;
        }
        for i in 0..LAYERS {
            let base = layer(i);
            for (lin, n, k) in CLIPPED
                .iter()
                .copied()
                .chain([("lconv1d.linear_end", WIDTH, WIDTH)])
            {
                parts.check_linear(&format!("{base}.{lin}"), n, k)?;
                parts.raw(&format!("{base}.{lin}.clip"), 4)?;
            }
            for norm in NORMS {
                parts.raw(&format!("{base}.{norm}.weight"), WIDTH)?;
            }
            parts.raw(&format!("{base}.self_attn.per_dim_scale"), HEAD_DIM)?;
            parts.raw(
                &format!("{base}.lconv1d.depthwise_conv1d.weight"),
                WIDTH * KERNEL,
            )?;
            parts.raw(&format!("{base}.self_attn.relative_keys"), WINDOW * WIDTH)?;
        }
        parts.check_linear("audio_tower.output_proj", OUT, WIDTH)?;
        parts.raw("audio_tower.output_proj.bias", OUT)?;
        parts.check_linear("embed_audio.embedding_projection", HIDDEN, OUT)?;
        Ok(Self { parts })
    }

    pub fn load_source(&self) -> &'static str {
        self.parts.source
    }

    fn norm(&self, name: &str) -> Result<&[f32]> {
        self.parts.raw(&format!("{name}.weight"), WIDTH)
    }

    /// Clamped linear: input to `[input_min, input_max]`, output to
    /// `[output_min, output_max]`.
    fn clipped(&self, hw: &B, precision: Precision, x: &[f32], name: &str) -> Result<Vec<f32>> {
        let c = self.parts.raw(&format!("{name}.clip"), 4)?;
        let threads = hw.threads();
        let mut y = if x.iter().all(|&v| v >= c[0] && v <= c[1]) {
            self.parts.matmul(hw, precision, x, name)?
        } else {
            let mut xc = hw.buffer(x.len())?;
            xc.copy_from_slice(x);
            clamp(&mut xc, c[0], c[1], threads);
            let y = self.parts.matmul(hw, precision, &xc, name);
            hw.recycle(xc);
            y?
        };
        clamp(&mut y, c[2], c[3], threads);
        Ok(y)
    }

    /// Half-step FFN: `x += 0.5 * rms_post(l2(silu(l1(rms_pre(x)))))`.
    fn feed_forward(&self, hw: &B, precision: Precision, x: &mut [f32], base: &str) -> Result<()> {
        let threads = hw.threads();
        let mut h = hw.buffer(x.len())?;
        rmsnorm_nthreads(
            x,
            WIDTH,
            Some(self.norm(&format!("{base}.pre_layer_norm"))?),
            RMS_EPS,
            &mut h,
            threads,
        )?;
        let mut up = self.clipped(hw, precision, &h, &format!("{base}.ffw_layer_1"))?;
        hw.recycle(h);
        par_rows(threads, &mut up, FF, 4, |_, row| {
            for v in row {
                *v = silu(*v);
            }
        });
        let mut down = self.clipped(hw, precision, &up, &format!("{base}.ffw_layer_2"))?;
        hw.recycle(up);
        // `add_rmsnorm` scales the sum; the half step scales only the branch.
        rmsnorm_inplace(
            &mut down,
            WIDTH,
            Some(self.norm(&format!("{base}.post_layer_norm"))?),
            RMS_EPS,
            threads,
        )?;
        for (o, d) in x.iter_mut().zip(&down) {
            *o += RESIDUAL * d;
        }
        hw.recycle(down);
        Ok(())
    }

    /// Chunked local attention over `t` frames; returns `post(attn)`.
    fn attention(
        &self,
        hw: &B,
        precision: Precision,
        h: &[f32],
        t: usize,
        base: &str,
    ) -> Result<Vec<f32>> {
        let threads = hw.threads();
        let q = self.clipped(hw, precision, h, &format!("{base}.q_proj"))?;
        let k = self.clipped(hw, precision, h, &format!("{base}.k_proj"))?;
        let v = self.clipped(hw, precision, h, &format!("{base}.v_proj"))?;
        let pds = self.parts.raw(&format!("{base}.per_dim_scale"), HEAD_DIM)?;
        let rel = self
            .parts
            .raw(&format!("{base}.relative_keys"), WINDOW * WIDTH)?;
        let q_scale = ((HEAD_DIM as f64).powf(-0.5) / std::f64::consts::LN_2) as f32;
        let k_scale = ((1.0 + std::f64::consts::E).ln() / std::f64::consts::LN_2) as f32;
        let qs: Vec<f32> = pds.iter().map(|&s| softplus(s)).collect();
        let mut out = hw.buffer(t * WIDTH)?;
        par_rows(threads, &mut out, WIDTH, 2, |i, row| {
            let lo = (i + 1).saturating_sub(WINDOW);
            let mut qh = [0f32; HEAD_DIM];
            let mut logits = [0f32; WINDOW];
            for hd in 0..HEADS {
                let off = hd * HEAD_DIM;
                for (d, x) in qh.iter_mut().enumerate() {
                    *x = q[i * WIDTH + off + d] * q_scale * qs[d];
                }
                let mut max = f32::NEG_INFINITY;
                for (j, kv) in (lo..=i).enumerate() {
                    let kr = &k[kv * WIDTH + off..][..HEAD_DIM];
                    let rr = &rel[(i - kv) * WIDTH + off..][..HEAD_DIM];
                    let mut ac = 0f32;
                    let mut bd = 0f32;
                    for d in 0..HEAD_DIM {
                        ac += qh[d] * (kr[d] * k_scale);
                        bd += qh[d] * rr[d];
                    }
                    let l = ((ac + bd) / SOFTCAP).tanh() * SOFTCAP;
                    logits[j] = l;
                    max = max.max(l);
                }
                // Masked keys (INVALID logits) get exactly 0 weight in f32,
                // so only the live window enters the softmax.
                let n = i + 1 - lo;
                let mut sum = 0f32;
                for l in &mut logits[..n] {
                    *l = (*l - max).exp();
                    sum += *l;
                }
                let dst = &mut row[off..off + HEAD_DIM];
                dst.fill(0.0);
                for (j, kv) in (lo..=i).enumerate() {
                    let p = logits[j] / sum;
                    for (o, &x) in dst.iter_mut().zip(&v[kv * WIDTH + off..][..HEAD_DIM]) {
                        *o += p * x;
                    }
                }
            }
        });
        hw.recycle(q);
        hw.recycle(k);
        hw.recycle(v);
        let y = self.clipped(hw, precision, &out, &format!("{base}.post"));
        hw.recycle(out);
        y
    }

    /// Light conv block: `x += linear_end(silu(conv_norm(dwconv(glu(linear_start(rms(x)))))))`.
    fn light_conv(
        &self,
        hw: &B,
        precision: Precision,
        x: &mut [f32],
        t: usize,
        base: &str,
    ) -> Result<()> {
        let threads = hw.threads();
        let mut h = hw.buffer(x.len())?;
        rmsnorm_nthreads(
            x,
            WIDTH,
            Some(self.norm(&format!("{base}.pre_layer_norm"))?),
            RMS_EPS,
            &mut h,
            threads,
        )?;
        let ab = self.clipped(hw, precision, &h, &format!("{base}.linear_start"))?;
        par_rows(threads, &mut h, WIDTH, 8, |r, row| {
            let src = &ab[r * 2 * WIDTH..(r + 1) * 2 * WIDTH];
            for (c, o) in row.iter_mut().enumerate() {
                let g = src[WIDTH + c];
                *o = src[c] * (1.0 / (1.0 + (-g).exp()));
            }
        });
        hw.recycle(ab);
        let w = self
            .parts
            .raw(&format!("{base}.depthwise_conv1d.weight"), WIDTH * KERNEL)?;
        let mut conv = hw.buffer(t * WIDTH)?;
        let glu = &h;
        par_rows(threads, &mut conv, WIDTH, 8, |r, row| {
            for (c, o) in row.iter_mut().enumerate() {
                let mut acc = 0f32;
                for j in 0..KERNEL {
                    // Causal: left pad KERNEL - 1.
                    if let Some(src) = (r + j).checked_sub(KERNEL - 1) {
                        acc += w[c * KERNEL + j] * glu[src * WIDTH + c];
                    }
                }
                *o = acc;
            }
        });
        rmsnorm_inplace(
            &mut conv,
            WIDTH,
            Some(self.norm(&format!("{base}.conv_norm"))?),
            RMS_EPS,
            threads,
        )?;
        par_rows(threads, &mut conv, WIDTH, 8, |_, row| {
            for v in row {
                *v = silu(*v);
            }
        });
        hw.recycle(h);
        let end = self.clipped(hw, precision, &conv, &format!("{base}.linear_end"))?;
        hw.recycle(conv);
        for (o, e) in x.iter_mut().zip(&end) {
            *o += e;
        }
        hw.recycle(end);
        Ok(())
    }

    /// Two stride-2 convolutions + LayerNorm + ReLU, then the input projection.
    fn subsample(
        &self,
        hw: &B,
        precision: Precision,
        f: &AudioFeatures,
    ) -> Result<(Vec<f32>, usize)> {
        let threads = hw.threads();
        let t0 = f.frames.div_ceil(2);
        let t1 = t0.div_ceil(2);
        let w0 = self
            .parts
            .raw(&format!("{SSCP}.layer0.conv.weight"), C0 * 9)?;
        let n0 = self.parts.raw(&format!("{SSCP}.layer0.norm.weight"), C0)?;
        let n1 = self.parts.raw(&format!("{SSCP}.layer1.norm.weight"), C1)?;
        // layer0 out: [t0][F0][C0], channel last.
        let mut a = hw.buffer(rt_core::checked_size(
            "EG2 audio subsampling",
            &[t0, F0, C0],
        )?)?;
        let mel = &f.mel;
        par_rows(threads, &mut a, F0 * C0, 4, |t, row| {
            for (fo, px) in row.as_chunks_mut::<C0>().0.iter_mut().enumerate() {
                let mut taps = [0f32; 9];
                for kh in 0..3 {
                    for kw in 0..3 {
                        let (y, x) = ((2 * t + kh).checked_sub(1), (2 * fo + kw).checked_sub(1));
                        if let (Some(y), Some(x)) = (y, x) {
                            if y < f.frames && x < MEL_BINS {
                                taps[kh * 3 + kw] = mel[y * MEL_BINS + x];
                            }
                        }
                    }
                }
                for (c, o) in px.iter_mut().enumerate() {
                    let wc = &w0[c * 9..c * 9 + 9];
                    *o = taps.iter().zip(wc).map(|(a, b)| a * b).sum();
                }
                layer_norm_relu(px, n0);
            }
        });
        // layer1 via im2col rows [(t1, f1)][(kh * 3 + kw) * C0 + c].
        let mut cols = hw.buffer(rt_core::checked_size(
            "EG2 audio im2col",
            &[t1, F1, 9 * C0],
        )?)?;
        let src = &a;
        par_rows(threads, &mut cols, F1 * 9 * C0, 2, |t, row| {
            for (fo, col) in row.as_chunks_mut::<{ 9 * C0 }>().0.iter_mut().enumerate() {
                for kh in 0..3 {
                    for kw in 0..3 {
                        let dst = &mut col[(kh * 3 + kw) * C0..][..C0];
                        match ((2 * t + kh).checked_sub(1), (2 * fo + kw).checked_sub(1)) {
                            (Some(y), Some(x)) if y < t0 && x < F0 => {
                                dst.copy_from_slice(&src[(y * F0 + x) * C0..][..C0]);
                            }
                            _ => dst.fill(0.0),
                        }
                    }
                }
            }
        });
        hw.recycle(a);
        let mut b = self
            .parts
            .matmul(hw, precision, &cols, &format!("{SSCP}.layer1.conv"))?;
        hw.recycle(cols);
        par_rows(threads, &mut b, C1, 64, |_, px| layer_norm_relu(px, n1));
        let x = self
            .parts
            .matmul(hw, precision, &b, &format!("{SSCP}.input_proj_linear"))?;
        hw.recycle(b);
        Ok((x, t1))
    }

    /// Soft tokens `[f.soft_tokens(), 512]` for one clip.
    pub fn forward(&self, hw: &B, precision: Precision, f: &AudioFeatures) -> Result<Vec<f32>> {
        check_shape!(
            f.frames > 0 && f.mel.len() == f.frames * MEL_BINS,
            "audio features: {} frames, {} values",
            f.frames,
            f.mel.len()
        );
        let threads = hw.threads();
        let t1 = f.frames.div_ceil(2).div_ceil(2);
        hw.begin_shape(1, t1);
        hw.validate_workspace(rt_core::checked_size(
            "EG2 audio activations",
            &[t1, F1 * 9 * C0 + WIDTH * 8 + FF * 2, 4],
        )?)?;
        let (mut x, t) = self.subsample(hw, precision, f)?;
        for i in 0..LAYERS {
            let base = layer(i);
            self.feed_forward(hw, precision, &mut x, &format!("{base}.feed_forward1"))?;
            let mut h = hw.buffer(x.len())?;
            rmsnorm_nthreads(
                &x,
                WIDTH,
                Some(self.norm(&format!("{base}.norm_pre_attn"))?),
                RMS_EPS,
                &mut h,
                threads,
            )?;
            let attn = self.attention(hw, precision, &h, t, &format!("{base}.self_attn"))?;
            hw.recycle(h);
            add_rmsnorm(
                &mut x,
                &attn,
                WIDTH,
                Some(self.norm(&format!("{base}.norm_post_attn"))?),
                1.0,
                RMS_EPS,
                threads,
            )?;
            hw.recycle(attn);
            self.light_conv(hw, precision, &mut x, t, &format!("{base}.lconv1d"))?;
            self.feed_forward(hw, precision, &mut x, &format!("{base}.feed_forward2"))?;
            rmsnorm_inplace(
                &mut x,
                WIDTH,
                Some(self.norm(&format!("{base}.norm_out"))?),
                RMS_EPS,
                threads,
            )?;
        }
        let mut y = self
            .parts
            .matmul(hw, precision, &x, "audio_tower.output_proj")?;
        hw.recycle(x);
        let bias = self.parts.raw("audio_tower.output_proj.bias", OUT)?;
        par_rows(threads, &mut y, OUT, 8, |_, row| {
            for (o, b) in row.iter_mut().zip(bias) {
                *o += b;
            }
        });
        rmsnorm_inplace(&mut y, OUT, None, RMS_EPS, threads)?;
        let soft = self
            .parts
            .matmul(hw, precision, &y, "embed_audio.embedding_projection");
        hw.recycle(y);
        soft
    }
}

/// `nn.LayerNorm(weight only, eps 1e-6)` over one pixel's channels, then ReLU.
fn layer_norm_relu(px: &mut [f32], w: &[f32]) {
    let n = px.len() as f32;
    let mean = px.iter().sum::<f32>() / n;
    let var = px.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + RMS_EPS).sqrt();
    for (v, &g) in px.iter_mut().zip(w) {
        *v = ((*v - mean) * inv * g).max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_keys_start_from_the_zero_distance_encoding() {
        // Identity projection: distance 0 is [sin 0 = 0..., cos 0 = 1...].
        let mut w = vec![0f32; WIDTH * WIDTH];
        for i in 0..WIDTH {
            w[i * WIDTH + i] = 1.0;
        }
        let r = relative_keys(&w);
        assert!(r[..TIMESCALES].iter().all(|&v| v == 0.0));
        assert!(r[TIMESCALES..WIDTH].iter().all(|&v| v == 1.0));
        // Distance 1, fastest timescale: sin(1), cos(1).
        assert!((r[WIDTH] - 1f32.sin()).abs() < 1e-6);
        assert!((r[WIDTH + TIMESCALES] - 1f32.cos()).abs() < 1e-6);
    }

    #[test]
    fn layer_norm_relu_normalizes_then_clips() {
        let mut px = [1.0f32, 3.0];
        layer_norm_relu(&mut px, &[1.0, 1.0]);
        assert_eq!(px[0], 0.0);
        assert!((px[1] - 1.0).abs() < 1e-5);
    }
}
