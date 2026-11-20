//! Packing weights into oneDNN's blocked layouts, and restoring and
//! unpacking them for the packed-weight cache. One reorder per `(n, k,
//! dtype)` layout is built once and shared by every weight of that shape.

use super::{Aligned, ExecWorkspace, MatDt, OnednnCpu, PackedWeight, WeightLayout};
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
        let (attr, post) = if dt != MatDt::F32 {
            self.bf16_attr(false, None)?
        } else {
            self.strict_attr(false, None)?
        };
        let mut pd = ptr::null_mut();
        let st = unsafe { (self.api.matmul_pd)(&mut pd, self.engine, a, b, ptr::null(), c, attr) };
        unsafe {
            (self.api.attr_destroy)(attr);
            if let Some(p) = post {
                (self.api.post_ops_destroy)(p);
            }
            (self.api.md_destroy)(a);
            (self.api.md_destroy)(b);
            (self.api.md_destroy)(c);
        }
        Api::check(st, "weight-format matmul pd")?;
        let queried = unsafe { (self.api.pd_query_md)(pd, dnnl_query_weights_md, 0) };
        let packed = self.clone_md(queried)?;
        let bytes = unsafe { (self.api.md_size)(packed) };
        unsafe { (self.api.pd_destroy)(pd) };
        Ok((packed, bytes))
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
        let (attr, post) = if dt != MatDt::F32 {
            self.bf16_attr(false, None)?
        } else {
            self.strict_attr(false, None)?
        };
        let mut pd = ptr::null_mut();
        let st = unsafe {
            (self.api.reorder_pd)(&mut pd, raw_md, self.engine, packed_md, self.engine, attr)
        };
        unsafe {
            (self.api.attr_destroy)(attr);
            if let Some(p) = post {
                (self.api.post_ops_destroy)(p);
            }
        }
        Api::check(st, "reorder pd")?;
        let scratch = self.scratch_size(pd);
        self.check_workspace("operator scratch", &[scratch])?;
        let q = unsafe { (self.api.pd_query_md)(pd, dnnl_query_scratchpad_md, 0) };
        let scratch_md = if q.is_null() {
            self.md_strides(&[16], &[1])?
        } else {
            self.clone_md(q)?
        };
        let reorder = self.create_prim(pd)?;
        unsafe { (self.api.pd_destroy)(pd) };
        check_shape!(
            bytes > 0 && bytes <= 1024 * 1024 * 1024 && scratch <= 512 * 1024 * 1024,
            "packing allocation size"
        );
        Ok(Arc::new(WeightLayout {
            api: self.api.clone(),
            packed_md,
            raw_md,
            scratch_md,
            reorder,
            bytes,
            scratch,
        }))
    }

    fn layout_for(&self, n: usize, k: usize, dt: MatDt) -> AResult<Arc<WeightLayout>> {
        let s = self
            .state
            .lock()
            .map_err(|_| crate::Error::Poisoned("backend"))?;
        if let Some(l) = s.layouts.get(&(n, k, dt)) {
            return Ok(l.clone());
        }
        drop(s);
        let l = self.make_layout(n, k, dt)?;
        let mut s = self
            .state
            .lock()
            .map_err(|_| crate::Error::Poisoned("backend"))?;
        s.layouts.insert((n, k, dt), l.clone());
        Ok(l)
    }

    pub fn pack(&self, n: usize, k: usize, src: &[f32]) -> AResult<PackedWeight> {
        check_shape!(
            src.len() == n.checked_mul(k).context("weight size")?,
            "weight shape"
        );
        self.reorder_pack(n, k, MatDt::F32, src.as_ptr() as *const u8)
    }

    pub fn pack_bf16(&self, n: usize, k: usize, src: &[u16]) -> AResult<PackedWeight> {
        check_shape!(
            src.len() == n.checked_mul(k).context("weight size")?,
            "weight shape"
        );
        self.reorder_pack(n, k, MatDt::Bf16, src.as_ptr() as *const u8)
    }

    /// Pack BF16 weights for f32 activations: BF16 storage, FP32 math.
    pub fn pack_bf16_f32(&self, n: usize, k: usize, src: &[u16]) -> AResult<PackedWeight> {
        check_shape!(
            src.len() == n.checked_mul(k).context("weight size")?,
            "weight shape"
        );
        self.reorder_pack(n, k, MatDt::F32Bf16, src.as_ptr() as *const u8)
    }

    fn reorder_pack(&self, n: usize, k: usize, dt: MatDt, src: *const u8) -> AResult<PackedWeight> {
        let layout = self.layout_for(n, k, dt)?;
        let mut ws = self
            .exec
            .acquire(|| ExecWorkspace::new(&self.api, self.engine, self.threadpool()))?;
        let mut s = self
            .state
            .lock()
            .map_err(|_| crate::Error::Poisoned("backend"))?;
        let data = Aligned::new(layout.bytes)?;
        if ws.scratch.bytes < layout.scratch {
            ws.scratch = Aligned::new(layout.scratch)?;
        }
        let _ = self.api.apply_threads(self.threads);
        let src_mem = self.mem(layout.raw_md)?;
        let dst_mem = self.mem(layout.packed_md)?;
        let scratch_mem = self.mem(layout.scratch_md)?;
        unsafe {
            Api::check(
                (self.api.mem_set)(src_mem, src as *mut c_void),
                "reorder src",
            )?;
            Api::check((self.api.mem_set)(dst_mem, data.ptr()), "reorder dst")?;
            Api::check(
                (self.api.mem_set)(scratch_mem, ws.scratch.ptr()),
                "reorder scratch",
            )?;
            let args = [
                dnnl_exec_arg_t {
                    arg: DNNL_ARG_FROM,
                    memory: src_mem,
                },
                dnnl_exec_arg_t {
                    arg: DNNL_ARG_TO,
                    memory: dst_mem,
                },
                dnnl_exec_arg_t {
                    arg: DNNL_ARG_SCRATCHPAD,
                    memory: scratch_mem,
                },
            ];
            Api::check(
                (self.api.prim_execute)(layout.reorder, ws.stream, 3, args.as_ptr()),
                "reorder execute",
            )?;
            Api::check((self.api.stream_wait)(ws.stream), "reorder wait")?;
            (self.api.mem_destroy)(src_mem);
            (self.api.mem_destroy)(dst_mem);
            (self.api.mem_destroy)(scratch_mem);
        }
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

    pub fn pack_from_cache(&self, n: usize, k: usize, bytes: &[u8]) -> AResult<PackedWeight> {
        self.pack_from_cache_dt(n, k, bytes, MatDt::F32, None)
    }

    pub fn pack_from_cache_bf16(&self, n: usize, k: usize, bytes: &[u8]) -> AResult<PackedWeight> {
        self.pack_from_cache_dt(n, k, bytes, MatDt::Bf16, None)
    }

    fn pack_from_cache_dt(
        &self,
        n: usize,
        k: usize,
        bytes: &[u8],
        dt: MatDt,
        expected: Option<&LayoutTag>,
    ) -> AResult<PackedWeight> {
        let layout = self.layout_for(n, k, dt)?;
        let tag = self.layout_tag_of(layout.packed_md, layout.bytes);
        if let Some(exp) = expected {
            check_shape!(
                *exp == tag,
                "packed layout ISA/format mismatch (cache vs live oneDNN MD)"
            );
        }
        check_shape!(bytes.len() == layout.bytes, "cached packed size");
        let data = Aligned::copy_from(bytes)?;
        let mut s = self
            .state
            .lock()
            .map_err(|_| crate::Error::Poisoned("backend"))?;
        s.weight_bytes += layout.bytes;
        Ok(PackedWeight {
            layout,
            data,
            n,
            k,
            dt,
            tag,
        })
    }

    pub(crate) fn restore_packed_checked(
        &self,
        n: usize,
        k: usize,
        bytes: &[u8],
        layout: &[u8],
    ) -> Result<PackedWeight> {
        let mut tag = empty_layout_tag();
        if layout.len() == LAYOUT_TAG_LEN {
            tag.copy_from_slice(layout);
        }
        self.pack_from_cache_dt(n, k, bytes, MatDt::F32, Some(&tag))
    }

    pub(crate) fn unpack_into(&self, packed: &PackedWeight, dst: &mut [u8]) -> Result<()> {
        let src = packed.as_bytes();
        if dst.len() < src.len() {
            return Err(crate::Error::Shape("packed unpack dest".into()));
        }
        dst[..src.len()].copy_from_slice(src);
        Ok(())
    }

    pub(crate) fn restore_packed_bf16_checked(
        &self,
        n: usize,
        k: usize,
        bytes: &[u8],
        layout: &[u8],
    ) -> Result<PackedWeight> {
        let mut tag = empty_layout_tag();
        if layout.len() == LAYOUT_TAG_LEN {
            tag.copy_from_slice(layout);
        }
        self.pack_from_cache_dt(n, k, bytes, MatDt::Bf16, Some(&tag))
    }

    pub(crate) fn restore_packed_bf16_f32_checked(
        &self,
        n: usize,
        k: usize,
        bytes: &[u8],
        layout: &[u8],
    ) -> Result<PackedWeight> {
        let mut tag = empty_layout_tag();
        if layout.len() == LAYOUT_TAG_LEN {
            tag.copy_from_slice(layout);
        }
        self.pack_from_cache_dt(n, k, bytes, MatDt::F32Bf16, Some(&tag))
    }
}
