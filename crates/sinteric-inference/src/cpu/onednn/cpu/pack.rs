//! Packing weights into oneDNN's blocked layouts, and restoring and
//! unpacking them for the packed-weight cache. One reorder per `(n, k,
//! dtype)` layout is built once and shared by every weight of that shape.

use super::{Aligned, ExecWorkspace, MatDt, OnednnCpu, Operand, PackedWeight, WeightLayout};
use crate::cpu::onednn::api::Api;
use crate::cpu::onednn::cache::{empty_layout_tag, LayoutTag, LAYOUT_TAG_LEN};
use crate::{check_shape, Result, Result as AResult, ResultContext as Context};
use onednn_sys::*;
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

impl OnednnCpu {
    fn layout_tag_of(&self, md: dnnl_memory_desc_t, packed_bytes: usize) -> LayoutTag {
        let mut tag = empty_layout_tag();
        tag[0..4].copy_from_slice(b"MD01");
        let dt = self.md_query_i32(md, dnnl_query_data_type) as u32;
        let fk = self.md_query_i32(md, dnnl_query_format_kind) as u32;
        let ndims = self.md_query_i32(md, dnnl_query_ndims_s32) as u32;
        let nblks = self.md_query_i32(md, dnnl_query_inner_nblks_s32) as u32;
        tag[4..8].copy_from_slice(&dt.to_le_bytes());
        tag[8..12].copy_from_slice(&fk.to_le_bytes());
        tag[12..16].copy_from_slice(&ndims.to_le_bytes());
        tag[16..20].copy_from_slice(&nblks.to_le_bytes());
        tag[24..32].copy_from_slice(&(packed_bytes as u64).to_le_bytes());
        let dims = self.md_query_dims(md, dnnl_query_dims);
        let padded = self.md_query_dims(md, dnnl_query_padded_dims);
        let strides = self.md_query_dims(md, dnnl_query_strides);
        let inner_blks = self.md_query_dims(md, dnnl_query_inner_blks);
        let inner_idxs = self.md_query_dims(md, dnnl_query_inner_idxs);
        for i in 0..4 {
            let o = 32 + i * 8;
            tag[o..o + 8].copy_from_slice(&dims[i].to_le_bytes());
            tag[64 + i * 8..72 + i * 8].copy_from_slice(&padded[i].to_le_bytes());
            tag[96 + i * 8..104 + i * 8].copy_from_slice(&strides[i].to_le_bytes());
            tag[128 + i * 8..136 + i * 8].copy_from_slice(&inner_blks[i].to_le_bytes());
            tag[160 + i * 8..168 + i * 8].copy_from_slice(&inner_idxs[i].to_le_bytes());
        }
        tag
    }

    fn select_weight_md(&self, n: i64, k: i64, dt: MatDt) -> AResult<(dnnl_memory_desc_t, usize)> {
        // Layout probe uses a fixed activation M=64. Packed-weight cache is
        // ISA + weight shape, not live token count or thread width.
        let a = self.md_strides_dt(&[64, k], &[k, 1], dt.src())?;
        let b = self.md_any_dt(&[k, n], dt.weights())?;
        let c = self.md_strides(&[64, n], &[n, 1])?;
        let attr = self.attr(false, None)?;
        let mut pd = ptr::null_mut();
        // SAFETY: the three descriptors and the attribute are live (created
        // above), and oneDNN writes the new handle to a live local.
        let rc =
            unsafe { (self.api.matmul_pd)(&mut pd, self.engine, a, b, ptr::null(), c, attr.raw()) };
        // SAFETY: created above for this probe alone; the primitive
        // descriptor keeps its own copies.
        unsafe {
            (self.api.md_destroy)(a);
            (self.api.md_destroy)(b);
            (self.api.md_destroy)(c);
        }
        Api::check(rc, "weight-format matmul pd")?;
        // SAFETY: `pd` is the live descriptor built above; the weights
        // descriptor is its own, alive while it is.
        let queried = unsafe { (self.api.pd_query_md)(pd, dnnl_query_weights_md, 0) };
        let packed = self.clone_md(queried);
        // SAFETY: a copy of the descriptor, alive past `pd`.
        let bytes = packed
            .as_ref()
            .map_or(0, |&md| unsafe { (self.api.md_size)(md) });
        // SAFETY: nothing uses `pd` after this.
        unsafe { (self.api.pd_destroy)(pd) };
        Ok((packed?, bytes))
    }

    fn make_layout(&self, n: usize, k: usize, dt: MatDt) -> AResult<Arc<WeightLayout>> {
        check_shape!(
            n > 0 && k > 0 && n <= i64::MAX as usize && k <= i64::MAX as usize,
            "packed matrix [{n}, {k}]"
        );
        let raw_bytes = crate::checked_size(
            "packed matrix bytes",
            &[n, k, if dt == MatDt::F32 { 4 } else { 2 }],
        )?;
        check_shape!(
            raw_bytes <= 1024 * 1024 * 1024,
            "packed matrix [{n}, {k}] exceeds 1 GiB weight limit"
        );
        let _ = self.api.apply_threads(self.threads);
        let (packed_md, bytes) = self.select_weight_md(n as i64, k as i64, dt)?;
        let raw_md = self.md_strides_dt(&[k as i64, n as i64], &[1, k as i64], dt.weights())?;
        let attr = self.attr(false, None)?;
        let mut pd = ptr::null_mut();
        let engine = self.engine;
        // SAFETY: both descriptors and the attribute are live, and oneDNN
        // writes the new handle to a live local.
        let rc = unsafe {
            (self.api.reorder_pd)(&mut pd, raw_md, engine, packed_md, engine, attr.raw())
        };
        Api::check(rc, "reorder pd")?;
        let scratch = self.scratch_size(pd);
        self.check_workspace("operator scratch", &[scratch])?;
        // SAFETY: `pd` is the live descriptor built above.
        let q = unsafe { (self.api.pd_query_md)(pd, dnnl_query_scratchpad_md, 0) };
        let scratch_md = if q.is_null() {
            self.md_strides(&[16], &[1])?
        } else {
            self.clone_md(q)?
        };
        let reorder = self.create_prim(pd)?;
        // SAFETY: the primitive keeps what it needs; nothing uses `pd` after.
        unsafe { (self.api.pd_destroy)(pd) };
        check_shape!(
            bytes > 0 && bytes <= 1024 * 1024 * 1024 && scratch <= 512 * 1024 * 1024,
            "packing allocation size"
        );
        let api = self.api.clone();
        Ok(Arc::new(WeightLayout {
            api,
            packed_md,
            raw_md,
            scratch_md,
            reorder,
            bytes,
            scratch,
        }))
    }

    fn layout_for(&self, n: usize, k: usize, dt: MatDt) -> AResult<Arc<WeightLayout>> {
        if let Some(l) = self.lock_state()?.layouts.get(&(n, k, dt)) {
            return Ok(l.clone());
        }
        let l = self.make_layout(n, k, dt)?;
        self.lock_state()?.layouts.insert((n, k, dt), l.clone());
        Ok(l)
    }

    /// Packs a row-major `[n, k]` f32 weight for FP32 math.
    pub fn pack(&self, n: usize, k: usize, src: &[f32]) -> AResult<PackedWeight> {
        self.reorder_pack(n, k, MatDt::F32, Operand::F32(src))
    }

    /// Packs a row-major `[n, k]` BF16 weight for BF16 math.
    pub fn pack_bf16(&self, n: usize, k: usize, src: &[u16]) -> AResult<PackedWeight> {
        self.reorder_pack(n, k, MatDt::Bf16, Operand::Bf16(src))
    }

    /// Pack BF16 weights for f32 activations: BF16 storage, FP32 math.
    pub fn pack_bf16_f32(&self, n: usize, k: usize, src: &[u16]) -> AResult<PackedWeight> {
        self.reorder_pack(n, k, MatDt::F32Bf16, Operand::Bf16(src))
    }

    fn reorder_pack(&self, n: usize, k: usize, dt: MatDt, src: Operand) -> AResult<PackedWeight> {
        check_shape!(
            src.len() == n.checked_mul(k).context("weight size")?,
            "weight shape"
        );
        check_shape!(
            matches!(src, Operand::F32(_)) == (dt == MatDt::F32),
            "{dt:?} weights in the wrong element type"
        );
        let layout = self.layout_for(n, k, dt)?;
        let mut ws = self
            .exec
            .acquire(|| ExecWorkspace::new(&self.api, self.engine, self.threadpool()))?;
        let mut s = self.lock_state()?;
        let data = Aligned::new(layout.bytes)?;
        if ws.scratch.bytes < layout.scratch {
            ws.scratch = Aligned::new(layout.scratch)?;
        }
        let _ = self.api.apply_threads(self.threads);
        let from = match src {
            Operand::F32(v) => v.as_ptr().cast::<c_void>(),
            Operand::Bf16(v) => v.as_ptr().cast::<c_void>(),
        };
        let bound = [
            (DNNL_ARG_FROM, layout.raw_md, from.cast_mut()),
            (DNNL_ARG_TO, layout.packed_md, data.ptr()),
            (DNNL_ARG_SCRATCHPAD, layout.scratch_md, ws.scratch.ptr()),
        ];
        let mut args = Vec::with_capacity(bound.len());
        let mut rc = Ok(());
        for (arg, md, handle) in bound {
            let memory = match self.mem(md) {
                Ok(memory) => memory,
                Err(e) => {
                    rc = Err(e);
                    break;
                }
            };
            args.push(dnnl_exec_arg_t { arg, memory });
            // SAFETY: `memory` was created just above for this call alone,
            // and `handle` holds what `md` describes: the source is `n * k`
            // elements of `dt`'s weight type (both checked above), `data` is
            // `layout.bytes`, and the scratch at least `layout.scratch`.
            rc = Api::check(
                unsafe { (self.api.mem_set)(memory, handle) },
                "reorder handle",
            );
            if rc.is_err() {
                break;
            }
        }
        if rc.is_ok() {
            let (n_args, stream) = (args.len() as i32, ws.stream);
            // SAFETY: the reorder is alive (`layout` holds it), every argument
            // is bound above, and waiting on the stream before returning keeps
            // each bound buffer alive for the whole execute.
            rc = Api::check(
                unsafe { (self.api.prim_execute)(layout.reorder, stream, n_args, args.as_ptr()) },
                "reorder execute",
            )
            // SAFETY: `stream` is this workspace's live stream.
            .and_then(|()| Api::check(unsafe { (self.api.stream_wait)(stream) }, "reorder wait"));
        }
        for a in &args {
            // SAFETY: created above for this call alone, and the reorder has
            // finished with it.
            unsafe { (self.api.mem_destroy)(a.memory) };
        }
        rc?;
        s.weight_bytes += layout.bytes;
        let tag = self.layout_tag_of(layout.packed_md, layout.bytes);
        Ok(PackedWeight {
            layout,
            data,
            n,
            k,
            dt,
            tag,
        })
    }

    /// A packed weight from the packed-weight cache: `bytes` as
    /// [`Self::unpack_into`] wrote them, under the layout tag `layout`,
    /// which must match the layout this CPU picks for `(n, k, dt)`.
    fn restore_packed(
        &self,
        n: usize,
        k: usize,
        dt: MatDt,
        bytes: &[u8],
        layout: &[u8],
    ) -> AResult<PackedWeight> {
        let mut expected = empty_layout_tag();
        if layout.len() == LAYOUT_TAG_LEN {
            expected.copy_from_slice(layout);
        }
        let layout = self.layout_for(n, k, dt)?;
        let tag = self.layout_tag_of(layout.packed_md, layout.bytes);
        check_shape!(
            expected == tag,
            "packed layout ISA/format mismatch (cache vs live oneDNN MD)"
        );
        check_shape!(bytes.len() == layout.bytes, "cached packed size");
        let data = Aligned::copy_from(bytes)?;
        self.lock_state()?.weight_bytes += layout.bytes;
        Ok(PackedWeight {
            layout,
            data,
            n,
            k,
            dt,
            tag,
        })
    }

    /// [`Self::restore_packed`] of an f32 weight.
    pub(crate) fn restore_packed_checked(
        &self,
        n: usize,
        k: usize,
        bytes: &[u8],
        layout: &[u8],
    ) -> Result<PackedWeight> {
        self.restore_packed(n, k, MatDt::F32, bytes, layout)
    }

    /// [`Self::restore_packed`] of a BF16 weight for BF16 math.
    pub(crate) fn restore_packed_bf16_checked(
        &self,
        n: usize,
        k: usize,
        bytes: &[u8],
        layout: &[u8],
    ) -> Result<PackedWeight> {
        self.restore_packed(n, k, MatDt::Bf16, bytes, layout)
    }

    /// [`Self::restore_packed`] of a BF16 weight for FP32 math.
    pub(crate) fn restore_packed_bf16_f32_checked(
        &self,
        n: usize,
        k: usize,
        bytes: &[u8],
        layout: &[u8],
    ) -> Result<PackedWeight> {
        self.restore_packed(n, k, MatDt::F32Bf16, bytes, layout)
    }

    /// Copies a packed weight's bytes into `dst`, for the packed-weight
    /// cache.
    pub(crate) fn unpack_into(&self, packed: &PackedWeight, dst: &mut [u8]) -> Result<()> {
        let src = packed.as_bytes();
        if dst.len() < src.len() {
            return Err(crate::Error::internal("packed unpack dest"));
        }
        dst[..src.len()].copy_from_slice(src);
        Ok(())
    }
}
