//! The operators the model graphs call: the matmuls (plain, BF16, gated,
//! with a residual sum) and the row-wise ops around them.

use super::{Key, MatDt, OnednnCpu, PackedWeight};
use crate::cpu::kernels::rmsnorm_nthreads;
use crate::{check_shape, Result, Result as AResult};

impl OnednnCpu {
    pub(crate) fn linear(
        &self,
        x: &[f32],
        w: &PackedWeight,
        out: &mut [f32],
        sum: bool,
    ) -> AResult<()> {
        check_shape!(x.len() % w.k == 0, "linear input width");
        let m = x.len() / w.k;
        check_shape!(
            out.len() == crate::checked_size("linear output", &[m, w.n])?,
            "linear destination"
        );
        self.run(
            Key {
                kind: 0,
                g: 1,
                m,
                n: w.n,
                k: w.k,
                ab: 0,
                ar: w.k,
                bb: 0,
                br: w.k,
                trans: true,
                param: 0,
                sum,
                weight: w.layout.packed_md as usize,
                // BF16-stored FP32 weights keep FP32 math; BF16 weights
                // belong to `linear_bf16`.
                dt: if w.dt == MatDt::F32Bf16 {
                    MatDt::F32Bf16
                } else {
                    MatDt::F32
                },
            },
            x,
            None,
            None,
            out,
            Some(w),
        )
    }

    pub(crate) fn linear_bf16(
        &self,
        x: &[f32],
        w: &PackedWeight,
        out: &mut [f32],
        sum: bool,
    ) -> AResult<()> {
        check_shape!(
            w.dt == MatDt::Bf16,
            "linear_bf16 requires packed BF16 weights"
        );
        check_shape!(x.len() % w.k == 0, "linear input width");
        let m = x.len() / w.k;
        check_shape!(
            out.len() == crate::checked_size("linear output", &[m, w.n])?,
            "linear destination"
        );
        let m_pad = self.strategy.gemm_pad_m(m);
        if m_pad == m {
            return self.run(
                self.packed_bf16_key(0, m, w, sum),
                x,
                None,
                None,
                out,
                Some(w),
            );
        }
        let mut src = self.buffer(crate::checked_size("activation elements", &[m_pad, w.k])?)?;
        src[..x.len()].copy_from_slice(x);
        src[x.len()..].fill(0.0);
        let mut dst = self.buffer(crate::checked_size("activation elements", &[m_pad, w.n])?)?;
        if sum {
            // The sum post-op accumulates into `dst`, so it starts as `out`.
            dst[..out.len()].copy_from_slice(out);
            dst[out.len()..].fill(0.0);
        }
        let rc = self.run(
            self.packed_bf16_key(0, m_pad, w, sum),
            &src,
            None,
            None,
            &mut dst,
            Some(w),
        );
        if rc.is_ok() {
            out.copy_from_slice(&dst[..out.len()]);
        }
        self.recycle(dst);
        self.recycle(src);
        rc
    }

    pub(crate) fn gated_linear(
        &self,
        x: &[f32],
        w: &PackedWeight,
        gate: &[f32],
    ) -> AResult<Vec<f32>> {
        check_shape!(x.len() % w.k == 0, "gated input");
        let m = x.len() / w.k;
        check_shape!(
            gate.len() == crate::checked_size("gate output", &[m, w.n])?,
            "gate shape"
        );
        let mut out = self.buffer(gate.len())?;
        self.run(
            Key {
                kind: 7,
                g: 1,
                m,
                n: w.n,
                k: w.k,
                ab: 0,
                ar: w.k,
                bb: 0,
                br: w.k,
                trans: true,
                param: 0,
                sum: false,
                weight: w.layout.packed_md as usize,
                dt: MatDt::F32,
            },
            x,
            None,
            Some(gate),
            &mut out,
            Some(w),
        )?;
        Ok(out)
    }

    pub(crate) fn gated_linear_tanh(
        &self,
        x: &[f32],
        w: &PackedWeight,
        gate: &[f32],
    ) -> AResult<Vec<f32>> {
        check_shape!(x.len() % w.k == 0, "gated input");
        let m = x.len() / w.k;
        check_shape!(
            gate.len() == crate::checked_size("gate output", &[m, w.n])?,
            "gate shape"
        );
        if w.dt != MatDt::Bf16 {
            // FP32 math: no BF16 conversion and no AMX row padding.
            let mut out = self.buffer(gate.len())?;
            self.run(
                Key {
                    dt: w.dt,
                    ..self.packed_bf16_key(8, m, w, false)
                },
                x,
                None,
                Some(gate),
                &mut out,
                Some(w),
            )?;
            return Ok(out);
        }
        let m_pad = self.strategy.gemm_pad_m(m);
        if m_pad == m {
            let mut out = self.buffer(gate.len())?;
            self.run(
                self.packed_bf16_key(8, m, w, false),
                x,
                None,
                Some(gate),
                &mut out,
                Some(w),
            )?;
            return Ok(out);
        }
        let mut src = self.buffer(crate::checked_size("activation elements", &[m_pad, w.k])?)?;
        src[..x.len()].copy_from_slice(x);
        src[x.len()..].fill(0.0);
        let mut gate_pad =
            self.buffer(crate::checked_size("activation elements", &[m_pad, w.n])?)?;
        gate_pad[..gate.len()].copy_from_slice(gate);
        gate_pad[gate.len()..].fill(0.0);
        let mut dst = self.buffer(crate::checked_size("activation elements", &[m_pad, w.n])?)?;
        let rc = self.run(
            self.packed_bf16_key(8, m_pad, w, false),
            &src,
            None,
            Some(&gate_pad),
            &mut dst,
            Some(w),
        );
        self.recycle(src);
        self.recycle(gate_pad);
        rc?;
        let mut out = self.buffer(gate.len())?;
        out.copy_from_slice(&dst[..gate.len()]);
        self.recycle(dst);
        Ok(out)
    }

    pub(crate) fn linear_bf16_src(
        &self,
        x: &[u16],
        w: &PackedWeight,
        out: &mut [f32],
        sum: bool,
    ) -> AResult<()> {
        check_shape!(
            w.dt == MatDt::Bf16,
            "linear_bf16_src requires packed BF16 weights"
        );
        check_shape!(x.len() % w.k == 0, "linear input width");
        let m = x.len() / w.k;
        check_shape!(
            out.len() == crate::checked_size("linear output", &[m, w.n])?,
            "linear destination"
        );
        let m_pad = self.strategy.gemm_pad_m(m);
        if m_pad == m {
            return self.run_ex(
                self.packed_bf16_key(0, m, w, sum),
                &[],
                Some(x),
                None,
                None,
                None,
                out,
                Some(w),
            );
        }
        let mut src =
            self.buffer_u16(crate::checked_size("activation elements", &[m_pad, w.k])?)?;
        src[..x.len()].copy_from_slice(x);
        src[x.len()..].fill(0);
        let mut dst = self.buffer(crate::checked_size("activation elements", &[m_pad, w.n])?)?;
        if sum {
            dst[..out.len()].copy_from_slice(out);
            dst[out.len()..].fill(0.0);
        }
        let rc = self.run_ex(
            self.packed_bf16_key(0, m_pad, w, sum),
            &[],
            Some(&src),
            None,
            None,
            None,
            &mut dst,
            Some(w),
        );
        if rc.is_ok() {
            out.copy_from_slice(&dst[..out.len()]);
        }
        self.recycle(dst);
        self.recycle_u16(src);
        rc
    }

    pub(crate) fn gated_linear_tanh_src(
        &self,
        x: &[u16],
        w: &PackedWeight,
        gate: &[f32],
    ) -> AResult<Vec<f32>> {
        check_shape!(
            w.dt == MatDt::Bf16,
            "gated_linear_tanh_src requires packed BF16 weights"
        );
        check_shape!(x.len() % w.k == 0, "gated input");
        let m = x.len() / w.k;
        check_shape!(
            gate.len() == crate::checked_size("gate output", &[m, w.n])?,
            "gate shape"
        );
        let m_pad = self.strategy.gemm_pad_m(m);
        if m_pad == m {
            let mut out = self.buffer(gate.len())?;
            self.run_ex(
                self.packed_bf16_key(8, m, w, false),
                &[],
                Some(x),
                None,
                None,
                Some(gate),
                &mut out,
                Some(w),
            )?;
            return Ok(out);
        }
        let mut src =
            self.buffer_u16(crate::checked_size("activation elements", &[m_pad, w.k])?)?;
        src[..x.len()].copy_from_slice(x);
        src[x.len()..].fill(0);
        let mut gate_pad =
            self.buffer(crate::checked_size("activation elements", &[m_pad, w.n])?)?;
        gate_pad[..gate.len()].copy_from_slice(gate);
        gate_pad[gate.len()..].fill(0.0);
        let mut dst = self.buffer(crate::checked_size("activation elements", &[m_pad, w.n])?)?;
        let rc = self.run_ex(
            self.packed_bf16_key(8, m_pad, w, false),
            &[],
            Some(&src),
            None,
            None,
            Some(&gate_pad),
            &mut dst,
            Some(w),
        );
        self.recycle_u16(src);
        self.recycle(gate_pad);
        rc?;
        let mut out = self.buffer(gate.len())?;
        out.copy_from_slice(&dst[..gate.len()]);
        self.recycle(dst);
        Ok(out)
    }

    fn packed_bf16_key(&self, kind: i32, m: usize, w: &PackedWeight, sum: bool) -> Key {
        Key {
            kind,
            g: 1,
            m,
            n: w.n,
            k: w.k,
            ab: 0,
            ar: w.k,
            bb: 0,
            br: w.k,
            trans: true,
            param: 0,
            sum,
            weight: w.layout.packed_md as usize,
            dt: MatDt::Bf16,
        }
    }

    pub(crate) fn mm(
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
    ) -> AResult<Vec<f32>> {
        let na = crate::matrix_span(g, m, k, ab, ar)?;
        let nb = if trans {
            crate::matrix_span(g, n, k, bb, br)?
        } else {
            crate::matrix_span(g, k, n, bb, br)?
        };
        self.check_workspace("matmul output", &[g, m, n, 4])?;
        check_shape!(a.len() >= na && b.len() >= nb, "matmul slice bounds");
        let mut out = self.buffer(crate::checked_size("activation elements", &[g, m, n])?)?;
        self.run(
            Key {
                kind: 0,
                g,
                m,
                n,
                k,
                ab,
                ar,
                bb,
                br,
                trans,
                param: 0,
                sum: false,
                weight: 0,
                dt: MatDt::F32,
            },
            a,
            Some(b),
            None,
            &mut out,
            None,
        )?;
        Ok(out)
    }

    pub(crate) fn mm_bf16_src(
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
    ) -> AResult<Vec<f32>> {
        let na = crate::matrix_span(g, m, k, ab, ar)?;
        let nb = if trans {
            crate::matrix_span(g, n, k, bb, br)?
        } else {
            crate::matrix_span(g, k, n, bb, br)?
        };
        self.check_workspace("matmul output", &[g, m, n, 4])?;
        check_shape!(a.len() >= na && b.len() >= nb, "matmul slice bounds");
        let m_pad = self.strategy.attn_pad_m(m);
        let tight = ar == k && (g == 1 || ab == m * ar);
        if m_pad == m || !tight {
            let mut out = self.buffer(crate::checked_size("activation elements", &[g, m, n])?)?;
            self.run_ex(
                Key {
                    kind: 0,
                    g,
                    m,
                    n,
                    k,
                    ab,
                    ar,
                    bb,
                    br,
                    trans,
                    param: 0,
                    sum: false,
                    weight: 0,
                    dt: MatDt::Bf16,
                },
                &[],
                Some(a),
                None,
                Some(b),
                None,
                &mut out,
                None,
            )?;
            return Ok(out);
        }
        let mut a_pad =
            self.buffer_u16(crate::checked_size("activation elements", &[g, m_pad, k])?)?;
        for gi in 0..g {
            let src = gi * ab;
            let dst = gi * m_pad * k;
            a_pad[dst..dst + m * k].copy_from_slice(&a[src..src + m * k]);
            a_pad[dst + m * k..dst + m_pad * k].fill(0);
        }
        let mut dst = self.buffer(crate::checked_size("activation elements", &[g, m_pad, n])?)?;
        let rc = self.run_ex(
            Key {
                kind: 0,
                g,
                m: m_pad,
                n,
                k,
                ab: m_pad * k,
                ar: k,
                bb,
                br,
                trans,
                param: 0,
                sum: false,
                weight: 0,
                dt: MatDt::Bf16,
            },
            &[],
            Some(&a_pad),
            None,
            Some(b),
            None,
            &mut dst,
            None,
        );
        self.recycle_u16(a_pad);
        rc?;
        let mut out = self.buffer(crate::checked_size("activation elements", &[g, m, n])?)?;
        for gi in 0..g {
            let src = gi * m_pad * n;
            let dst_o = gi * m * n;
            out[dst_o..dst_o + m * n].copy_from_slice(&dst[src..src + m * n]);
        }
        self.recycle(dst);
        Ok(out)
    }

    /// FP32 SiLU for OCR's SVTR MLP. Existing model op kinds are unchanged.
    pub fn swish(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.op_inner(9, m, n, x, None, None, 0.)
    }

    pub fn op_inner(
        &self,
        kind: i32,
        m: usize,
        n: usize,
        x: &[f32],
        other: Option<&[f32]>,
        shift: Option<&[f32]>,
        eps: f32,
    ) -> AResult<Vec<f32>> {
        check_shape!(
            m > 0 && n > 0 && x.len() == crate::checked_size("operator shape", &[m, n])?,
            "operator [{m}, {n}] input length {}",
            x.len()
        );
        if kind == 1 {
            check_shape!(
                other.map(|v| v.len()) == Some(n) && shift.map(|v| v.len()) == Some(n),
                "layernorm parameters"
            );
        }
        if kind == 4 {
            check_shape!(other.map(|v| v.len()) == Some(x.len()), "binary shape");
        }
        let mut out = self.buffer(x.len())?;
        self.run(
            Key {
                kind,
                g: 1,
                m,
                n,
                k: 0,
                ab: 0,
                ar: n,
                bb: 0,
                br: n,
                trans: false,
                param: eps.to_bits(),
                sum: false,
                weight: 0,
                dt: MatDt::F32,
            },
            x,
            other,
            shift,
            &mut out,
            None,
        )?;
        Ok(out)
    }

    pub(crate) fn layernorm(
        &self,
        m: usize,
        n: usize,
        x: &[f32],
        scale: &[f32],
        shift: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>> {
        self.op_inner(1, m, n, x, Some(scale), Some(shift), eps)
    }

    pub(crate) fn softmax(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.op_inner(2, m, n, x, None, None, 0.)
    }

    pub(crate) fn gelu_erf(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.op_inner(3, m, n, x, None, None, 0.)
    }

    pub(crate) fn mul(&self, m: usize, n: usize, x: &[f32], y: &[f32]) -> Result<Vec<f32>> {
        self.op_inner(4, m, n, x, Some(y), None, 0.)
    }

    pub(crate) fn rmsnorm(
        &self,
        m: usize,
        n: usize,
        x: &[f32],
        scale: Option<&[f32]>,
        eps: f32,
    ) -> Result<Vec<f32>> {
        if x.len() != m.saturating_mul(n) {
            return Err(crate::Error::internal("rmsnorm shape"));
        }
        let mut y = self.buffer(x.len())?;
        rmsnorm_nthreads(x, n, scale, eps, &mut y, self.threads.max(1) as usize)?;
        Ok(y)
    }
}
