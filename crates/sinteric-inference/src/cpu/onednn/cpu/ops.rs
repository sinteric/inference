//! The operators the model graphs call: the matmuls (plain, BF16, gated,
//! with a residual sum) and the row-wise ops around them.

use super::{Key, MatDt, OnednnCpu, Operand, Operands, PackedWeight};
use crate::cpu::kernels::rmsnorm_nthreads;
use crate::{check_shape, checked_size, Result};

/// A batched matmul `C[i] = A[i] · B[i]` for `i < g`: `[m, k]` by `[k, n]`,
/// or by `[n, k]` transposed when `b_trans`. Each operand is addressed by
/// its stride between groups and between rows, in elements; `C` is dense
/// `[g, m, n]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Gemm {
    pub(crate) g: usize,
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) a_group: usize,
    pub(crate) a_row: usize,
    pub(crate) b_group: usize,
    pub(crate) b_row: usize,
    pub(crate) b_trans: bool,
}

impl Gemm {
    /// `A · B` with dense rows: `A` rows are `k` apart, `B` rows `n` apart.
    pub(crate) fn nn(
        g: usize,
        m: usize,
        n: usize,
        k: usize,
        a_group: usize,
        b_group: usize,
    ) -> Self {
        let (a_row, b_row, b_trans) = (k, n, false);
        Self {
            g,
            m,
            n,
            k,
            a_group,
            a_row,
            b_group,
            b_row,
            b_trans,
        }
    }

    /// `A · Bᵀ` with dense rows: both operands' rows are `k` apart.
    pub(crate) fn nt(
        g: usize,
        m: usize,
        n: usize,
        k: usize,
        a_group: usize,
        b_group: usize,
    ) -> Self {
        let (a_row, b_row, b_trans) = (k, k, true);
        Self {
            g,
            m,
            n,
            k,
            a_group,
            a_row,
            b_group,
            b_row,
            b_trans,
        }
    }

    /// The plan key for this matmul with operands of `dt`.
    pub(super) fn key(self, dt: MatDt) -> Key {
        Key {
            kind: 0,
            g: self.g,
            m: self.m,
            n: self.n,
            k: self.k,
            ab: self.a_group,
            ar: self.a_row,
            bb: self.b_group,
            br: self.b_row,
            trans: self.b_trans,
            param: 0,
            sum: false,
            weight: 0,
            dt,
        }
    }

    /// Checks that operands of `a_len` and `b_len` elements hold every
    /// element the matmul reads.
    fn check_operands(&self, a_len: usize, b_len: usize) -> Result<()> {
        let na = crate::matrix_span(self.g, self.m, self.k, self.a_group, self.a_row)?;
        let nb = if self.b_trans {
            crate::matrix_span(self.g, self.n, self.k, self.b_group, self.b_row)?
        } else {
            crate::matrix_span(self.g, self.k, self.n, self.b_group, self.b_row)?
        };
        check_shape!(a_len >= na && b_len >= nb, "matmul slice bounds");
        Ok(())
    }
}

/// A row-wise oneDNN operator over an `[m, n]` buffer.
#[derive(Clone, Copy, Debug)]
pub(crate) enum RowOp<'a> {
    /// Layer normalisation of each row, then a per-column scale and shift.
    LayerNorm {
        scale: &'a [f32],
        shift: &'a [f32],
        eps: f32,
    },
    /// Softmax of each row.
    Softmax,
    /// GELU with the exact erf.
    GeluErf,
    /// The elementwise product with a second `[m, n]` buffer.
    Mul(&'a [f32]),
    /// SiLU, `x · sigmoid(x)`.
    Swish,
}

/// A pooled buffer padded with zero rows, in the type of the operand it
/// replaces.
enum Padded {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
}

impl Padded {
    fn operand(&self) -> Operand<'_> {
        match self {
            Self::F32(v) => Operand::F32(v),
            Self::Bf16(v) => Operand::Bf16(v),
        }
    }

    fn recycle(self, cpu: &OnednnCpu) {
        match self {
            Self::F32(v) => cpu.recycle(v),
            Self::Bf16(v) => cpu.recycle_u16(v),
        }
    }
}

impl OnednnCpu {
    /// `out = x · wᵀ` (`+ out` when `sum`) in FP32 math; BF16-stored
    /// weights widen exactly to f32. BF16 weights belong to
    /// [`Self::linear_bf16`].
    pub(crate) fn linear(
        &self,
        x: &[f32],
        w: &PackedWeight,
        out: &mut [f32],
        sum: bool,
    ) -> Result<()> {
        let m = packed_rows(x.len(), w, out.len())?;
        let dt = if w.dt == MatDt::F32Bf16 {
            MatDt::F32Bf16
        } else {
            MatDt::F32
        };
        self.run(
            packed_key(0, m, w, sum, dt),
            Operands::f32(x).weight(w),
            out,
        )
    }

    /// `out = x · wᵀ` (`+ out` when `sum`) against BF16 weights; `x` is
    /// converted to BF16 on the way in.
    pub(crate) fn linear_bf16(
        &self,
        x: &[f32],
        w: &PackedWeight,
        out: &mut [f32],
        sum: bool,
    ) -> Result<()> {
        check_shape!(
            w.dt == MatDt::Bf16,
            "linear_bf16 requires packed BF16 weights"
        );
        let m = packed_rows(x.len(), w, out.len())?;
        self.run_padded(
            packed_key(0, m, w, sum, MatDt::Bf16),
            Operands::f32(x).weight(w),
            out,
        )
    }

    /// [`Self::linear_bf16`] of an input already in BF16.
    pub(crate) fn linear_bf16_src(
        &self,
        x: &[u16],
        w: &PackedWeight,
        out: &mut [f32],
        sum: bool,
    ) -> Result<()> {
        check_shape!(
            w.dt == MatDt::Bf16,
            "linear_bf16_src requires packed BF16 weights"
        );
        let m = packed_rows(x.len(), w, out.len())?;
        self.run_padded(
            packed_key(0, m, w, sum, MatDt::Bf16),
            Operands::bf16(x).weight(w),
            out,
        )
    }

    /// `gelu_erf(x · wᵀ) · gate` in FP32 math.
    pub(crate) fn gated_linear(
        &self,
        x: &[f32],
        w: &PackedWeight,
        gate: &[f32],
    ) -> Result<Vec<f32>> {
        let m = packed_rows(x.len(), w, gate.len())?;
        let mut out = self.buffer(gate.len())?;
        let ops = Operands::f32(x).shift(gate).weight(w);
        self.run(packed_key(7, m, w, false, MatDt::F32), ops, &mut out)?;
        Ok(out)
    }

    /// `gelu_tanh(x · wᵀ) · gate`: BF16 math against BF16 weights, FP32
    /// math (no conversion and no row padding) against any other.
    pub(crate) fn gated_linear_tanh(
        &self,
        x: &[f32],
        w: &PackedWeight,
        gate: &[f32],
    ) -> Result<Vec<f32>> {
        let m = packed_rows(x.len(), w, gate.len())?;
        let ops = Operands::f32(x).shift(gate).weight(w);
        if w.dt != MatDt::Bf16 {
            let mut out = self.buffer(gate.len())?;
            self.run(packed_key(8, m, w, false, w.dt), ops, &mut out)?;
            return Ok(out);
        }
        self.run_gated(packed_key(8, m, w, false, MatDt::Bf16), ops)
    }

    /// [`Self::gated_linear_tanh`] of an input already in BF16, against
    /// BF16 weights.
    pub(crate) fn gated_linear_tanh_src(
        &self,
        x: &[u16],
        w: &PackedWeight,
        gate: &[f32],
    ) -> Result<Vec<f32>> {
        check_shape!(
            w.dt == MatDt::Bf16,
            "gated_linear_tanh_src requires packed BF16 weights"
        );
        let m = packed_rows(x.len(), w, gate.len())?;
        let ops = Operands::bf16(x).shift(gate).weight(w);
        self.run_gated(packed_key(8, m, w, false, MatDt::Bf16), ops)
    }

    /// Runs a gated matmul into a new buffer of the gate's length.
    fn run_gated(&self, key: Key, ops: Operands) -> Result<Vec<f32>> {
        let len = ops.shift.map_or(0, <[f32]>::len);
        let mut out = self.buffer(len)?;
        self.run_padded(key, ops, &mut out)?;
        Ok(out)
    }

    /// Runs a packed BF16 matmul with its rows padded to the count the
    /// ISA's kernels want ([`IsaStrategy::gemm_pad_m`](crate::cpu::onednn::IsaStrategy::gemm_pad_m)). The padded
    /// rows are zero and only the first `key.m` rows of the result reach
    /// `out`; a sum post-op accumulates into the padded output, so that
    /// starts as `out`.
    fn run_padded(&self, key: Key, ops: Operands, out: &mut [f32]) -> Result<()> {
        let (m, n, k) = (key.m, key.n, key.k);
        let m_pad = self.strategy.gemm_pad_m(m);
        if m_pad == m {
            return self.run(key, ops, out);
        }
        let key = Key { m: m_pad, ..key };
        let src_len = checked_size("activation elements", &[m_pad, k])?;
        let dst_len = checked_size("activation elements", &[m_pad, n])?;
        let src = self.padded(ops.a, src_len)?;
        let gate = ops.shift.map(|g| self.padded_f32(g, dst_len)).transpose()?;
        let mut dst = if key.sum {
            self.padded_f32(out, dst_len)?
        } else {
            self.buffer(dst_len)?
        };
        let padded = Operands {
            a: src.operand(),
            shift: gate.as_deref(),
            ..ops
        };
        let rc = self.run(key, padded, &mut dst);
        src.recycle(self);
        if let Some(gate) = gate {
            self.recycle(gate);
        }
        if rc.is_ok() {
            out.copy_from_slice(&dst[..out.len()]);
        }
        self.recycle(dst);
        rc
    }

    /// `x` followed by zeros, `len` elements in all, in a pooled buffer.
    fn padded_f32(&self, x: &[f32], len: usize) -> Result<Vec<f32>> {
        check_shape!(x.len() <= len, "padding {} elements to {len}", x.len());
        let mut buf = self.buffer(len)?;
        buf[..x.len()].copy_from_slice(x);
        buf[x.len()..].fill(0.0);
        Ok(buf)
    }

    /// [`Self::padded_f32`] of either operand type.
    fn padded(&self, x: Operand, len: usize) -> Result<Padded> {
        Ok(match x {
            Operand::F32(x) => Padded::F32(self.padded_f32(x, len)?),
            Operand::Bf16(x) => {
                check_shape!(x.len() <= len, "padding {} elements to {len}", x.len());
                let mut buf = self.buffer_u16(len)?;
                buf[..x.len()].copy_from_slice(x);
                buf[x.len()..].fill(0);
                Padded::Bf16(buf)
            }
        })
    }

    /// An f32 batched matmul.
    pub(crate) fn mm(&self, gemm: Gemm, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        let Gemm { g, m, n, .. } = gemm;
        gemm.check_operands(a.len(), b.len())?;
        self.check_workspace("matmul output", &[g, m, n, 4])?;
        let mut out = self.buffer(checked_size("activation elements", &[g, m, n])?)?;
        self.run(
            gemm.key(MatDt::F32),
            Operands::f32(a).b(Operand::F32(b)),
            &mut out,
        )?;
        Ok(out)
    }

    /// A batched BF16 matmul against BF16 `b`; an f32 `a` is converted on
    /// the way in. Short dense query blocks are padded to the row count the
    /// ISA's kernels want, as
    /// [`IsaStrategy::attn_pad_m`](crate::cpu::onednn::IsaStrategy::attn_pad_m) says.
    pub(crate) fn mm_bf16(&self, gemm: Gemm, a: Operand, b: &[u16]) -> Result<Vec<f32>> {
        let Gemm {
            g,
            m,
            n,
            k,
            a_group,
            a_row,
            ..
        } = gemm;
        gemm.check_operands(a.len(), b.len())?;
        self.check_workspace("matmul output", &[g, m, n, 4])?;
        let out_len = checked_size("activation elements", &[g, m, n])?;
        let m_pad = self.strategy.attn_pad_m(m);
        let tight = a_row == k && (g == 1 || a_group == m * a_row);
        let ops = Operands {
            a,
            ..Operands::bf16(&[])
        }
        .b(Operand::Bf16(b));
        if m_pad == m || !tight {
            let mut out = self.buffer(out_len)?;
            self.run(gemm.key(MatDt::Bf16), ops, &mut out)?;
            return Ok(out);
        }
        let padded = Gemm {
            m: m_pad,
            a_group: m_pad * k,
            a_row: k,
            ..gemm
        };
        let a_pad = self.padded_groups(a, &gemm, m_pad)?;
        let mut dst = self.buffer(checked_size("activation elements", &[g, m_pad, n])?)?;
        let rc = self.run(
            padded.key(MatDt::Bf16),
            Operands {
                a: a_pad.operand(),
                ..ops
            },
            &mut dst,
        );
        a_pad.recycle(self);
        rc?;
        let mut out = self.buffer(out_len)?;
        for (to, from) in out.chunks_exact_mut(m * n).zip(dst.chunks_exact(m_pad * n)) {
            to.copy_from_slice(&from[..m * n]);
        }
        self.recycle(dst);
        Ok(out)
    }

    /// Each of `gemm.g` dense `[m, k]` groups of `a`, followed by zero rows
    /// up to `m_pad`.
    fn padded_groups(&self, a: Operand, gemm: &Gemm, m_pad: usize) -> Result<Padded> {
        fn copy<T: Copy + Default>(a: &[T], gemm: &Gemm, m_pad: usize, out: &mut [T]) {
            let (live, group) = (gemm.m * gemm.k, m_pad * gemm.k);
            for (gi, to) in out.chunks_exact_mut(group).enumerate().take(gemm.g) {
                to[..live].copy_from_slice(&a[gi * gemm.a_group..][..live]);
                to[live..].fill(T::default());
            }
        }
        let len = checked_size("activation elements", &[gemm.g, m_pad, gemm.k])?;
        Ok(match a {
            Operand::F32(a) => {
                let mut out = self.buffer(len)?;
                copy(a, gemm, m_pad, &mut out);
                Padded::F32(out)
            }
            Operand::Bf16(a) => {
                let mut out = self.buffer_u16(len)?;
                copy(a, gemm, m_pad, &mut out);
                Padded::Bf16(out)
            }
        })
    }

    /// Runs a row-wise operator over the `[m, n]` buffer `x` into a new
    /// buffer.
    pub(crate) fn row_op(&self, op: RowOp, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        check_shape!(
            m > 0 && n > 0 && x.len() == checked_size("operator shape", &[m, n])?,
            "operator [{m}, {n}] input length {}",
            x.len()
        );
        let (kind, eps, ops) = match op {
            RowOp::LayerNorm { scale, shift, eps } => {
                check_shape!(scale.len() == n && shift.len() == n, "layernorm parameters");
                (1, eps, Operands::f32(x).b(Operand::F32(scale)).shift(shift))
            }
            RowOp::Softmax => (2, 0.0, Operands::f32(x)),
            RowOp::GeluErf => (3, 0.0, Operands::f32(x)),
            RowOp::Mul(y) => {
                check_shape!(y.len() == x.len(), "binary shape");
                (4, 0.0, Operands::f32(x).b(Operand::F32(y)))
            }
            RowOp::Swish => (9, 0.0, Operands::f32(x)),
        };
        let key = Key {
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
            param: f32::to_bits(eps),
            sum: false,
            weight: 0,
            dt: MatDt::F32,
        };
        let mut out = self.buffer(x.len())?;
        self.run(key, ops, &mut out)?;
        Ok(out)
    }

    /// SiLU, for OCR's SVTR MLP.
    pub(crate) fn swish(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.row_op(RowOp::Swish, m, n, x)
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
        self.row_op(RowOp::LayerNorm { scale, shift, eps }, m, n, x)
    }

    pub(crate) fn softmax(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.row_op(RowOp::Softmax, m, n, x)
    }

    pub(crate) fn gelu_erf(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.row_op(RowOp::GeluErf, m, n, x)
    }

    pub(crate) fn mul(&self, m: usize, n: usize, x: &[f32], y: &[f32]) -> Result<Vec<f32>> {
        self.row_op(RowOp::Mul(y), m, n, x)
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

/// The row count of an input of `x_len` elements against `w`, checking
/// that the output holds `out_len` = rows × `w.n` elements.
fn packed_rows(x_len: usize, w: &PackedWeight, out_len: usize) -> Result<usize> {
    check_shape!(w.k > 0 && x_len.is_multiple_of(w.k), "linear input width");
    let m = x_len / w.k;
    check_shape!(
        out_len == checked_size("linear output", &[m, w.n])?,
        "linear destination"
    );
    Ok(m)
}

/// The plan key of an `m`-row matmul of `kind` against the packed `w`.
fn packed_key(kind: i32, m: usize, w: &PackedWeight, sum: bool, dt: MatDt) -> Key {
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
        dt,
    }
}
