//! Small hardware trait between a per-model graph and a backend.
//!
//! Graphs stay per-model. The trait keeps them free of CPU-only assumptions
//! so a future CUDA graph can implement the same operators (matmul with
//! post-ops, attention, norm, eltwise, plus a backend-owned buffer pool).

pub use rt_core::{Error, Result};
use serde_json::Value;

/// Packed oneDNN memory-descriptor fingerprint stored in the weight cache.
pub const PACKED_LAYOUT_TAG_LEN: usize = 256;

/// Packed weight handle owned by a [`Hardware`] backend.
pub trait Packed: Send + Sync {
    fn n(&self) -> usize;
    fn k(&self) -> usize;
    fn bytes(&self) -> usize;
    /// ISA-dependent blocked-layout fingerprint. Default is zeros.
    fn layout_tag(&self) -> [u8; PACKED_LAYOUT_TAG_LEN] {
        [0u8; PACKED_LAYOUT_TAG_LEN]
    }
}

/// Hardware a model graph talks to. Implemented today for oneDNN-CPU.
pub trait Hardware: Send + Sync {
    type Packed: Packed;

    fn info(&self) -> &str;
    fn threads(&self) -> usize;
    /// `dnnl_get_effective_cpu_isa` (0 when the backend has no ISA).
    fn cpu_isa(&self) -> u64 {
        0
    }

    /// Rows consumed by packed projections, including any backend padding.
    fn residual_rows(&self, tokens: usize) -> usize {
        tokens
    }
    /// Whole-row unit for glue over the residual layout; independent of pool width.
    fn residual_row_tile(&self, _live_rows: usize) -> usize {
        1
    }

    /// Validate the graph's planned activation storage before allocating it.
    fn validate_workspace(&self, _bytes: usize) -> Result<()> {
        Ok(())
    }
    fn begin_shape(&self, batch: usize, seq: usize);
    /// Acquire `len` FP32 elements, returning allocation/size/budget errors.
    /// Successful buffers always have exactly `len` initialized elements.
    fn buffer(&self, len: usize) -> Result<Vec<f32>>;
    fn recycle(&self, buf: Vec<f32>);
    /// Pre-touch `n` f32 buffers of `len` so later `buffer` calls do not page-fault.
    fn reserve(&self, len: usize, n: usize) -> Result<()> {
        let _ = (len, n);
        Ok(())
    }
    /// Acquire `len` initialized BF16 elements with fallible allocation.
    fn buffer_u16(&self, len: usize) -> Result<Vec<u16>> {
        rt_core::try_vec(len)
    }
    fn recycle_u16(&self, buf: Vec<u16>) {
        let _ = buf;
    }
    fn reserve_u16(&self, len: usize, n: usize) -> Result<()> {
        let _ = (len, n);
        Ok(())
    }
    /// Accumulate a named wall-time bucket (QK / softmax / PV / glue). Default no-op.
    fn add_category_s(&self, _key: &str, _dt: f64) {}

    fn pack(&self, n: usize, k: usize, src: &[f32]) -> Result<Self::Packed>;
    fn restore_packed(&self, n: usize, k: usize, bytes: &[u8]) -> Result<Self::Packed>;
    /// Restore a packed blob and require `layout` to match the live MD.
    fn restore_packed_checked(
        &self,
        n: usize,
        k: usize,
        bytes: &[u8],
        layout: &[u8],
    ) -> Result<Self::Packed> {
        let _ = layout;
        self.restore_packed(n, k, bytes)
    }
    fn unpack_into(&self, packed: &Self::Packed, dst: &mut [u8]) -> Result<()>;

    fn linear(&self, x: &[f32], w: &Self::Packed, out: &mut [f32], sum: bool) -> Result<()>;
    fn gated_linear(&self, x: &[f32], w: &Self::Packed, gate: &[f32]) -> Result<Vec<f32>>;
    fn mm(
        &self,
        g: usize,
        m: usize,
        n: usize,
        k: usize,
        ab: usize,
        ar: usize,
        bb: usize,
        br: usize,
        trans: bool,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>>;

    fn layernorm(
        &self,
        m: usize,
        n: usize,
        x: &[f32],
        scale: &[f32],
        shift: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>>;
    fn softmax(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>>;
    fn gelu_erf(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>>;
    fn relu(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>>;
    fn mul(&self, m: usize, n: usize, x: &[f32], y: &[f32]) -> Result<Vec<f32>>;

    /// RMSNorm over the last dim (`n`), in f32. `scale=None` is unscaled (`v_norm`).
    fn rmsnorm(
        &self,
        m: usize,
        n: usize,
        x: &[f32],
        scale: Option<&[f32]>,
        eps: f32,
    ) -> Result<Vec<f32>> {
        default_rmsnorm(self, m, n, x, scale, eps)
    }

    /// `gelu_pytorch_tanh` / oneDNN gelu-tanh. Default is a portable scalar.
    fn gelu_tanh(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        default_gelu_tanh(self, m, n, x)
    }

    /// Pack `[n, k]` native BF16 weights into the backend's blocked layout.
    /// Default upcasts to f32 and calls [`Hardware::pack`] (not for production EG2).
    fn pack_bf16(&self, n: usize, k: usize, src: &[u16]) -> Result<Self::Packed> {
        rt_core::check_shape!(
            src.len() == rt_core::checked_size("BF16 weights", &[n, k])?,
            "BF16 weight shape [{n}, {k}]"
        );
        let mut f = rt_core::try_vec::<f32>(src.len())?;
        for (d, &b) in f.iter_mut().zip(src) {
            *d = f32::from_bits((b as u32) << 16);
        }
        self.pack(n, k, &f)
    }

    /// Restore a packed BF16 weight blob. Default is [`Hardware::restore_packed`].
    fn restore_packed_bf16(&self, n: usize, k: usize, bytes: &[u8]) -> Result<Self::Packed> {
        self.restore_packed(n, k, bytes)
    }

    fn restore_packed_bf16_checked(
        &self,
        n: usize,
        k: usize,
        bytes: &[u8],
        layout: &[u8],
    ) -> Result<Self::Packed> {
        let _ = layout;
        self.restore_packed_bf16(n, k, bytes)
    }

    /// Pack `[n, k]` BF16 weights for FP32 [`Hardware::linear`] /
    /// [`Hardware::gated_linear_tanh`]. Products equal those of the upcast
    /// weights (BF16 widens exactly); a backend may keep BF16 storage and
    /// widen in the kernel. Default upcasts and calls [`Hardware::pack`].
    fn pack_bf16_f32(&self, n: usize, k: usize, src: &[u16]) -> Result<Self::Packed> {
        rt_core::check_shape!(
            src.len() == rt_core::checked_size("BF16 weights", &[n, k])?,
            "BF16 weight shape [{n}, {k}]"
        );
        let mut f = rt_core::try_vec::<f32>(src.len())?;
        for (d, &b) in f.iter_mut().zip(src) {
            *d = f32::from_bits((b as u32) << 16);
        }
        self.pack(n, k, &f)
    }

    /// Restore a [`Hardware::pack_bf16_f32`] blob. Default is [`Hardware::restore_packed_checked`].
    fn restore_packed_bf16_f32_checked(
        &self,
        n: usize,
        k: usize,
        bytes: &[u8],
        layout: &[u8],
    ) -> Result<Self::Packed> {
        self.restore_packed_checked(n, k, bytes, layout)
    }

    /// BF16×BF16 matmul with f32 accumulation. Default falls back to f32 [`Hardware::linear`].
    fn linear_bf16(&self, x: &[f32], w: &Self::Packed, out: &mut [f32], sum: bool) -> Result<()> {
        self.linear(x, w, out, sum)
    }

    /// Like [`Hardware::linear_bf16`] but the activation is already BF16 bits.
    fn linear_bf16_src(
        &self,
        x: &[u16],
        w: &Self::Packed,
        out: &mut [f32],
        sum: bool,
    ) -> Result<()> {
        let mut f = rt_core::try_vec::<f32>(x.len())?;
        for (d, &b) in f.iter_mut().zip(x) {
            *d = f32::from_bits((b as u32) << 16);
        }
        self.linear_bf16(&f, w, out, sum)
    }

    /// Batched BF16 matmul, f32 in/out. Default falls back to [`Hardware::mm`].
    fn mm_bf16(
        &self,
        g: usize,
        m: usize,
        n: usize,
        k: usize,
        ab: usize,
        ar: usize,
        bb: usize,
        br: usize,
        trans: bool,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>> {
        self.mm(g, m, n, k, ab, ar, bb, br, trans, a, b)
    }

    /// Batched BF16 matmul from already-converted activations.
    fn mm_bf16_src(
        &self,
        g: usize,
        m: usize,
        n: usize,
        k: usize,
        ab: usize,
        ar: usize,
        bb: usize,
        br: usize,
        trans: bool,
        a: &[u16],
        b: &[u16],
    ) -> Result<Vec<f32>> {
        let mut af = rt_core::try_vec::<f32>(a.len())?;
        let mut bf = rt_core::try_vec::<f32>(b.len())?;
        for (d, &v) in af.iter_mut().zip(a) {
            *d = f32::from_bits((v as u32) << 16);
        }
        for (d, &v) in bf.iter_mut().zip(b) {
            *d = f32::from_bits((v as u32) << 16);
        }
        self.mm_bf16(g, m, n, k, ab, ar, bb, br, trans, &af, &bf)
    }

    /// GeGLU with `gelu_pytorch_tanh` instead of GELU-erf.
    fn gated_linear_tanh(&self, x: &[f32], w: &Self::Packed, gate: &[f32]) -> Result<Vec<f32>> {
        let mut y = rt_core::try_vec::<f32>(gate.len())?;
        self.linear(x, w, &mut y, false)?;
        let gelu = self.gelu_tanh(1, y.len(), &y)?;
        self.mul(1, gelu.len(), &gelu, gate)
    }

    /// [`Hardware::gated_linear_tanh`] with a preconverted BF16 activation.
    fn gated_linear_tanh_src(&self, x: &[u16], w: &Self::Packed, gate: &[f32]) -> Result<Vec<f32>> {
        let mut f = rt_core::try_vec::<f32>(x.len())?;
        for (d, &b) in f.iter_mut().zip(x) {
            *d = f32::from_bits((b as u32) << 16);
        }
        self.gated_linear_tanh(&f, w, gate)
    }

    /// GQA bidirectional attention. `q` is `[b, nq, s, d]`, `k`/`v` are `[b, nkv, s, d]`.
    /// Output is `[b, s, nq, d]`. `scale` is applied to QKᵀ (EG2 uses 1.0).
    /// Returns token-major rows in `residual_rows(b * s)`; any extra rows
    /// are zero. A backend overriding the residual layout must provide the
    /// corresponding attention implementation. Keys and queries stay live.
    fn attention_gqa(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        b: usize,
        s: usize,
        nq: usize,
        nkv: usize,
        d: usize,
        tile: usize,
        local: bool,
        radius: usize,
        scale: f32,
        padding: &[Vec<usize>],
    ) -> Result<Vec<f32>> {
        default_attention_gqa(
            self, q, k, v, b, s, nq, nkv, d, tile, local, radius, scale, padding,
        )
    }

    /// [`Hardware::attention_gqa`] with preconverted BF16 Q/K/V (RNE bits).
    fn attention_gqa_src(
        &self,
        q: &[u16],
        k: &[u16],
        v: &[u16],
        b: usize,
        s: usize,
        nq: usize,
        nkv: usize,
        d: usize,
        tile: usize,
        local: bool,
        radius: usize,
        scale: f32,
        padding: &[Vec<usize>],
    ) -> Result<Vec<f32>> {
        let up = |x: &[u16]| -> Vec<f32> {
            x.iter()
                .map(|&b| f32::from_bits((b as u32) << 16))
                .collect()
        };
        self.attention_gqa(
            &up(q),
            &up(k),
            &up(v),
            b,
            s,
            nq,
            nkv,
            d,
            tile,
            local,
            radius,
            scale,
            padding,
        )
    }

    /// [`Hardware::attention_gqa`] with FP32 QK, softmax and PV. Output is
    /// exactly `[b * s, nq * d]` token-major rows (no residual padding).
    fn attention_gqa_f32(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        b: usize,
        s: usize,
        nq: usize,
        nkv: usize,
        d: usize,
        tile: usize,
        local: bool,
        radius: usize,
        scale: f32,
        padding: &[Vec<usize>],
    ) -> Result<Vec<f32>> {
        default_attention_gqa(
            self, q, k, v, b, s, nq, nkv, d, tile, local, radius, scale, padding,
        )
    }

    /// Default attention: tiled QK^T + excluded-only mask + softmax + PV.
    /// A CUDA backend can override with a fused kernel.
    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        b: usize,
        s: usize,
        nh: usize,
        d: usize,
        tile: usize,
        local: bool,
        radius: usize,
        padding: &[Vec<usize>],
    ) -> Result<Vec<f32>> {
        default_attention(self, q, k, v, b, s, nh, d, tile, local, radius, padding)
    }

    fn stats(&self) -> Value;
}

fn default_rmsnorm<H: Hardware + ?Sized>(
    hw: &H,
    m: usize,
    n: usize,
    x: &[f32],
    scale: Option<&[f32]>,
    eps: f32,
) -> Result<Vec<f32>> {
    if n == 0 || x.len() != rt_core::checked_size("operator shape", &[m, n])? {
        return Err(rt_core::Error::Shape("rmsnorm shape".into()));
    }
    if let Some(s) = scale {
        if s.len() != n {
            return Err(rt_core::Error::Shape("rmsnorm scale".into()));
        }
    }
    let mut y = hw.buffer(x.len())?;
    let inv_n = 1.0 / n as f32;
    for r in 0..m {
        let row = &x[r * n..(r + 1) * n];
        let out = &mut y[r * n..(r + 1) * n];
        let mut mean_sq = 0.0f32;
        for &v in row {
            mean_sq += v * v;
        }
        let inv = (mean_sq * inv_n + eps).powf(-0.5);
        if let Some(s) = scale {
            for c in 0..n {
                out[c] = row[c] * inv * s[c];
            }
        } else {
            for c in 0..n {
                out[c] = row[c] * inv;
            }
        }
    }
    Ok(y)
}

fn default_gelu_tanh<H: Hardware + ?Sized>(
    hw: &H,
    m: usize,
    n: usize,
    x: &[f32],
) -> Result<Vec<f32>> {
    if n == 0 || x.len() != rt_core::checked_size("operator shape", &[m, n])? {
        return Err(rt_core::Error::Shape("gelu_tanh shape".into()));
    }
    const SQRT_2_OVER_PI: f32 = 0.797_884_583_473_205_6;
    const COEFF: f32 = 0.044_715;
    let mut y = hw.buffer(x.len())?;
    for (yi, &v) in y.iter_mut().zip(x) {
        *yi = 0.5 * v * (1.0 + (SQRT_2_OVER_PI * (v + COEFF * v * v * v)).tanh());
    }
    Ok(y)
}

fn default_attention_gqa<H: Hardware + ?Sized>(
    hw: &H,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    s: usize,
    nq: usize,
    nkv: usize,
    d: usize,
    tile: usize,
    local: bool,
    radius: usize,
    scale: f32,
    padding: &[Vec<usize>],
) -> Result<Vec<f32>> {
    validate_attention(q, k, v, b, s, nq, nkv, d, tile, radius, padding)?;
    if nq == 0 || nkv == 0 || !nq.is_multiple_of(nkv) {
        return Err(rt_core::Error::Shape("GQA head grouping".into()));
    }
    let groups = nq / nkv;
    // Repeat KV heads to match Q (Hugging Face eager `repeat_kv`).
    let mut k_rep = hw.buffer(rt_core::checked_size(
        "activation elements",
        &[b, nq, s, d],
    )?)?;
    let mut v_rep = hw.buffer(k_rep.len())?;
    for bi in 0..b {
        for kvh in 0..nkv {
            let src = ((bi * nkv + kvh) * s) * d;
            for g in 0..groups {
                let qh = kvh * groups + g;
                let dst = ((bi * nq + qh) * s) * d;
                k_rep[dst..dst + s * d].copy_from_slice(&k[src..src + s * d]);
                v_rep[dst..dst + s * d].copy_from_slice(&v[src..src + s * d]);
            }
        }
    }
    let out = if (scale - 1.0).abs() > 0.0 {
        let mut qs = hw.buffer(q.len())?;
        for (d, &v) in qs.iter_mut().zip(q) {
            *d = v * scale;
        }
        let y = default_attention(
            hw, &qs, &k_rep, &v_rep, b, s, nq, d, tile, local, radius, padding,
        )?;
        hw.recycle(qs);
        y
    } else {
        default_attention(
            hw, q, &k_rep, &v_rep, b, s, nq, d, tile, local, radius, padding,
        )?
    };
    hw.recycle(k_rep);
    hw.recycle(v_rep);
    Ok(out)
}

fn default_attention<H: Hardware + ?Sized>(
    hw: &H,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    s: usize,
    nh: usize,
    d: usize,
    tile: usize,
    local: bool,
    radius: usize,
    padding: &[Vec<usize>],
) -> Result<Vec<f32>> {
    validate_attention(q, k, v, b, s, nh, nh, d, tile, radius, padding)?;
    let g = b * nh;
    let mut out = hw.buffer(rt_core::checked_size(
        "activation elements",
        &[b, s, nh, d],
    )?)?;
    let tile = tile.min(s).max(1);
    for start in (0..s).step_by(tile) {
        let n = tile.min(s - start);
        let ks = if local {
            start.saturating_sub(radius)
        } else {
            0
        };
        let ke = if local {
            (start + n + radius).min(s)
        } else {
            s
        };
        let nk = ke - ks;
        let mut scores = hw.mm(
            g,
            n,
            nk,
            d,
            s * d,
            d,
            s * d,
            d,
            true,
            &q[start * d..],
            &k[ks * d..],
        )?;
        for (gi, group) in scores.chunks_exact_mut(n * nk).enumerate() {
            let excluded = &padding[gi / nh];
            if !local && excluded.is_empty() {
                continue;
            }
            for (qi, row) in group.chunks_exact_mut(nk).enumerate() {
                let lo = if local {
                    (start + qi).saturating_sub(radius).saturating_sub(ks)
                } else {
                    0
                };
                let hi = if local {
                    (start + qi + radius + 1).min(ke) - ks
                } else {
                    nk
                };
                for score in &mut row[..lo] {
                    *score += f32::MIN;
                }
                for score in &mut row[hi..] {
                    *score += f32::MIN;
                }
                for &pos in excluded {
                    if pos >= ks + lo && pos < ks + hi {
                        row[pos - ks] += f32::MIN;
                    }
                }
            }
        }
        let probs = hw.softmax(g * n, nk, &scores)?;
        hw.recycle(scores);
        let y = hw.mm(
            g,
            n,
            d,
            nk,
            n * nk,
            nk,
            s * d,
            d,
            false,
            &probs,
            &v[ks * d..],
        )?;
        hw.recycle(probs);
        for bi in 0..b {
            for head in 0..nh {
                for qi in 0..n {
                    let src = ((bi * nh + head) * n + qi) * d;
                    let dst = (bi * s + start + qi) * (nh * d) + head * d;
                    out[dst..dst + d].copy_from_slice(&y[src..src + d]);
                }
            }
        }
        hw.recycle(y);
    }
    Ok(out)
}

/// Validate attention tensors once before any row-parallel work or pointer use.
pub fn validate_attention<T>(
    q: &[T],
    k: &[T],
    v: &[T],
    b: usize,
    s: usize,
    nq: usize,
    nkv: usize,
    d: usize,
    tile: usize,
    radius: usize,
    padding: &[Vec<usize>],
) -> Result<()> {
    rt_core::check_shape!(
        b > 0 && s > 0 && nq > 0 && nkv > 0 && nq.is_multiple_of(nkv) && d > 0 && tile > 0,
        "attention shape [{b}, {nq}/{nkv}, {s}, {d}], tile {tile}"
    );
    let qn = rt_core::checked_size("attention queries", &[b, nq, s, d])?;
    let kn = rt_core::checked_size("attention keys", &[b, nkv, s, d])?;
    rt_core::check_shape!(
        q.len() == qn && k.len() == kn && v.len() == kn,
        "attention tensor lengths {}/{}/{}, expected {qn}/{kn}/{kn}",
        q.len(),
        k.len(),
        v.len()
    );
    rt_core::check_shape!(
        padding.len() == b && padding.iter().all(|r| r.iter().all(|&p| p < s)),
        "attention padding indices outside batch {b}, sequence {s}"
    );
    s.checked_add(radius)
        .and_then(|n| n.checked_add(1))
        .ok_or(Error::SizeOverflow("attention radius"))?;
    rt_core::checked_size("attention score bytes", &[b, nq, s, s.min(tile), 4])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NullPacked {
        n: usize,
        k: usize,
    }
    impl Packed for NullPacked {
        fn n(&self) -> usize {
            self.n
        }
        fn k(&self) -> usize {
            self.k
        }
        fn bytes(&self) -> usize {
            self.n * self.k * 4
        }
    }

    struct NullHw;
    impl Hardware for NullHw {
        type Packed = NullPacked;
        fn info(&self) -> &str {
            "null"
        }
        fn threads(&self) -> usize {
            1
        }
        fn begin_shape(&self, _b: usize, _s: usize) {}
        fn buffer(&self, len: usize) -> Result<Vec<f32>> {
            rt_core::try_vec(len)
        }
        fn recycle(&self, _buf: Vec<f32>) {}
        fn pack(&self, n: usize, k: usize, _src: &[f32]) -> Result<Self::Packed> {
            Ok(NullPacked { n, k })
        }
        fn restore_packed(&self, n: usize, k: usize, _bytes: &[u8]) -> Result<Self::Packed> {
            Ok(NullPacked { n, k })
        }
        fn unpack_into(&self, _p: &Self::Packed, _dst: &mut [u8]) -> Result<()> {
            Ok(())
        }
        fn linear(&self, _x: &[f32], _w: &Self::Packed, _o: &mut [f32], _s: bool) -> Result<()> {
            Ok(())
        }
        fn gated_linear(&self, _x: &[f32], _w: &Self::Packed, g: &[f32]) -> Result<Vec<f32>> {
            Ok(g.to_vec())
        }
        fn mm(
            &self,
            g: usize,
            m: usize,
            n: usize,
            _k: usize,
            _ab: usize,
            _ar: usize,
            _bb: usize,
            _br: usize,
            _t: bool,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            Ok(vec![0.; g * m * n])
        }
        fn layernorm(
            &self,
            m: usize,
            n: usize,
            x: &[f32],
            _s: &[f32],
            _h: &[f32],
            _e: f32,
        ) -> Result<Vec<f32>> {
            Ok(x[..m * n].to_vec())
        }
        fn softmax(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
            Ok(x[..m * n].to_vec())
        }
        fn gelu_erf(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
            Ok(x[..m * n].to_vec())
        }
        fn relu(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
            Ok(x[..m * n].to_vec())
        }
        fn mul(&self, m: usize, n: usize, x: &[f32], _y: &[f32]) -> Result<Vec<f32>> {
            Ok(x[..m * n].to_vec())
        }
        fn stats(&self) -> Value {
            Value::Null
        }
    }

    #[test]
    fn default_attention_shape() {
        let hw = NullHw;
        let q = vec![0f32; 2 * 4 * 2 * 8]; // b=2 s=4 nh=2 d=8
        let y = hw
            .attention(&q, &q, &q, 2, 4, 2, 8, 4, false, 0, &[vec![], vec![]])
            .unwrap();
        assert_eq!(y.len(), 2 * 4 * 2 * 8);
    }

    #[test]
    fn default_attention_gqa_shape() {
        let hw = NullHw;
        let b = 2;
        let s = 4;
        let nq = 4;
        let nkv = 2;
        let d = 8;
        let q = vec![0f32; b * nq * s * d];
        let k = vec![0f32; b * nkv * s * d];
        let y = hw
            .attention_gqa(
                &q,
                &k,
                &k,
                b,
                s,
                nq,
                nkv,
                d,
                4,
                true,
                2,
                1.0,
                &[vec![], vec![]],
            )
            .unwrap();
        assert_eq!(y.len(), b * s * nq * d);
    }

    #[test]
    fn rmsnorm_unscaled() {
        let hw = NullHw;
        let x = [3.0f32, 4.0];
        let y = hw.rmsnorm(1, 2, &x, None, 0.0).unwrap();
        let n = (y[0] * y[0] + y[1] * y[1]).sqrt();
        assert!((n - 2.0f32.sqrt()).abs() < 1e-5);
    }
}
