//! Primitive plans: the memory descriptors and attributes a primitive is
//! built from, the plan cache keyed by shape, and executing a plan on a
//! leased stream and scratchpad.
//!
//! Every pointer handed to oneDNN here is either a handle this backend
//! created and still owns, a live local it writes back through, or a buffer
//! [`Operands::check`] has measured against the plan's shape. That check is
//! what makes binding caller buffers to a primitive sound.

use super::{
    fill, fill_bf16, heap_in_use, Aligned, ExecWorkspace, Key, MatDt, MemSlot, OnednnCpu,
    PackedWeight, Plan, State,
};
use crate::cpu::onednn::api::{memory_none, Api};
use crate::cpu::threads::compute_lock;
use crate::{check_shape, Result as AResult, ResultContext as Context};
use onednn_sys::*;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, MutexGuard};
use std::time::Instant;

/// One operand buffer, in the type the caller holds it in. An f32 operand of
/// a BF16 matmul is converted into the execute workspace first.
#[derive(Clone, Copy)]
pub(crate) enum Operand<'a> {
    F32(&'a [f32]),
    Bf16(&'a [u16]),
}

impl std::fmt::Debug for Operand<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32(v) => write!(f, "F32({} elements)", v.len()),
            Self::Bf16(v) => write!(f, "Bf16({} elements)", v.len()),
        }
    }
}

impl Operand<'_> {
    pub(super) fn len(self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::Bf16(v) => v.len(),
        }
    }
}

/// The buffers one execution reads, by the role each plays in the plan.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Operands<'a> {
    /// `SRC` (`SRC_0` for the binary multiply).
    pub(super) a: Operand<'a>,
    /// `WEIGHTS` of an unpacked matmul, `SCALE` of layernorm, `SRC_1` of the
    /// binary multiply.
    pub(super) b: Option<Operand<'a>>,
    /// `SHIFT` of layernorm; the gate of the gated matmuls.
    pub(super) shift: Option<&'a [f32]>,
    /// The packed `WEIGHTS`, when the key names one.
    pub(super) weight: Option<&'a PackedWeight>,
}

impl<'a> Operands<'a> {
    pub(crate) fn f32(a: &'a [f32]) -> Self {
        Self {
            a: Operand::F32(a),
            b: None,
            shift: None,
            weight: None,
        }
    }

    pub(crate) fn bf16(a: &'a [u16]) -> Self {
        Self {
            a: Operand::Bf16(a),
            b: None,
            shift: None,
            weight: None,
        }
    }

    pub(crate) fn b(self, b: Operand<'a>) -> Self {
        Self { b: Some(b), ..self }
    }

    pub(crate) fn shift(self, shift: &'a [f32]) -> Self {
        Self {
            shift: Some(shift),
            ..self
        }
    }

    pub(crate) fn weight(self, weight: &'a PackedWeight) -> Self {
        Self {
            weight: Some(weight),
            ..self
        }
    }

    /// Checks that each buffer holds every element the key's primitive reads
    /// or writes through it, so that binding the buffers is sound.
    fn check(&self, key: &Key, out: usize) -> AResult<()> {
        let (g, m, n, k) = (key.g, key.m, key.n, key.k);
        let (mut b_need, mut shift_need) = (0, 0);
        let (a_need, out_need) = if key.is_matmul() {
            let out_need = crate::checked_size("matmul output", &[g, m, n])?;
            match self.weight {
                Some(w) => check_shape!(
                    key.weight == w.layout.packed_md as usize && w.k == k && w.n == n,
                    "packed weight does not match the plan"
                ),
                None => {
                    check_shape!(key.weight == 0, "plan needs its packed weight");
                    let (rows, cols) = if key.trans { (n, k) } else { (k, n) };
                    b_need = crate::matrix_span(g, rows, cols, key.bb, key.br)?;
                }
            }
            if key.is_gated() {
                shift_need = out_need;
            }
            (crate::matrix_span(g, m, k, key.ab, key.ar)?, out_need)
        } else {
            let mn = crate::checked_size("operator shape", &[m, n])?;
            match key.kind {
                1 => (b_need, shift_need) = (n, n),
                4 => b_need = mn,
                _ => {}
            }
            (mn, mn)
        };
        check_shape!(
            self.a.len() >= a_need
                && out >= out_need
                && self.b.map_or(0, Operand::len) >= b_need
                && self.shift.map_or(0, <[f32]>::len) >= shift_need,
            "operator buffers smaller than kind {} [{g}, {m}, {n}, {k}] reads",
            key.kind
        );
        Ok(())
    }
}

/// A primitive attribute and its post-ops, destroyed together on drop.
pub(super) struct Attr<'a> {
    api: &'a Api,
    attr: dnnl_primitive_attr_t,
    post: dnnl_post_ops_t,
}

impl Attr<'_> {
    pub(super) fn raw(&self) -> dnnl_primitive_attr_t {
        self.attr
    }
}

impl Drop for Attr<'_> {
    fn drop(&mut self) {
        // SAFETY: both handles, where not null, were created by
        // `OnednnCpu::attr` and are owned by this value alone; a descriptor
        // built from them keeps its own copy.
        unsafe {
            if !self.attr.is_null() {
                (self.api.attr_destroy)(self.attr);
            }
            if !self.post.is_null() {
                (self.api.post_ops_destroy)(self.post);
            }
        }
    }
}

/// The descriptors one primitive is built from.
struct Descs {
    pd: dnnl_primitive_desc_t,
    src: dnnl_memory_desc_t,
    weights: dnnl_memory_desc_t,
    dst: dnnl_memory_desc_t,
    scale: dnnl_memory_desc_t,
    gate: dnnl_memory_desc_t,
}

impl Key {
    fn is_matmul(&self) -> bool {
        matches!(self.kind, 0 | 7 | 8)
    }

    fn is_gated(&self) -> bool {
        matches!(self.kind, 7 | 8)
    }
}

impl OnednnCpu {
    /// The strict attribute every primitive here is built with: strict
    /// fpmath and accumulation, a user scratchpad, then optionally a sum
    /// post-op and a gate (`alg(dst) * gate`).
    pub(super) fn attr(
        &self,
        sum: bool,
        gate: Option<(dnnl_memory_desc_t, dnnl_alg_kind_t)>,
    ) -> AResult<Attr<'_>> {
        let mut attr = Attr {
            api: &self.api,
            attr: ptr::null_mut(),
            post: ptr::null_mut(),
        };
        // SAFETY: oneDNN writes the new handle to a live local.
        Api::check(unsafe { (self.api.attr_create)(&mut attr.attr) }, "attr")?;
        Api::check(
            // SAFETY: the live attribute created above, owned by `attr`.
            unsafe { (self.api.attr_fpmath)(attr.attr, dnnl_fpmath_mode_strict) },
            "fpmath",
        )?;
        // Strict accumulation is f32 for non-quantized (incl. BF16) matmuls.
        // Explicit `accumulation_mode_f32` is a non-default attr that brgemm
        // / gemm_bf16 reject as unimplemented.
        Api::check(
            // SAFETY: the live attribute created above.
            unsafe { (self.api.attr_accum)(attr.attr, dnnl_accumulation_mode_strict) },
            "accumulation",
        )?;
        Api::check(
            // SAFETY: the live attribute created above.
            unsafe { (self.api.attr_scratch)(attr.attr, dnnl_scratchpad_mode_user) },
            "scratchpad",
        )?;
        if sum || gate.is_some() {
            self.post_ops(&mut attr, sum, gate)?;
        }
        Ok(attr)
    }

    fn post_ops(
        &self,
        attr: &mut Attr,
        sum: bool,
        gate: Option<(dnnl_memory_desc_t, dnnl_alg_kind_t)>,
    ) -> AResult<()> {
        // SAFETY: oneDNN writes the new handle into `attr`, which owns it.
        let rc = unsafe { (self.api.post_ops_create)(&mut attr.post) };
        Api::check(rc, "post_ops")?;
        let post = attr.post;
        if sum {
            // SAFETY: `post` is the live post-op list created above.
            let rc = unsafe { (self.api.post_ops_sum)(post, 1.0, 0, dnnl_data_type_undef) };
            Api::check(rc, "sum post-op")?;
        }
        if let Some((md, alg)) = gate {
            // SAFETY: `post` is the live post-op list created above.
            let rc = unsafe { (self.api.post_ops_eltwise)(post, alg, 0.0, 0.0) };
            Api::check(rc, "gelu post-op")?;
            // SAFETY: `post` as above; `md` is a live descriptor the caller
            // owns, which oneDNN copies.
            let rc = unsafe { (self.api.post_ops_binary)(post, dnnl_binary_mul, md) };
            Api::check(rc, "binary-mul post-op")?;
        }
        // SAFETY: both are the live handles created above; oneDNN copies the
        // list into the attribute.
        Api::check(
            unsafe { (self.api.attr_post_ops)(attr.attr, post) },
            "set post_ops",
        )
    }

    pub(super) fn md_strides(&self, dims: &[i64], strides: &[i64]) -> AResult<dnnl_memory_desc_t> {
        self.md_strides_dt(dims, strides, dnnl_f32)
    }

    pub(super) fn md_strides_dt(
        &self,
        dims: &[i64],
        strides: &[i64],
        dt: dnnl_data_type_t,
    ) -> AResult<dnnl_memory_desc_t> {
        check_shape!(
            dims.len() == strides.len() && dims.len() <= 12,
            "memory desc rank"
        );
        let d = fill(dims);
        let s = fill(strides);
        let mut md = ptr::null_mut();
        let ndims = dims.len() as i32;
        // SAFETY: `d` and `s` hold 12 entries, at least `ndims` (checked
        // above), and oneDNN writes the new handle to a live local.
        let rc = unsafe { (self.api.md_strides)(&mut md, ndims, d.as_ptr(), dt, s.as_ptr()) };
        Api::check(rc, "memory desc strides")?;
        Ok(md)
    }

    pub(super) fn md_any_dt(
        &self,
        dims: &[i64],
        dt: dnnl_data_type_t,
    ) -> AResult<dnnl_memory_desc_t> {
        check_shape!(dims.len() <= 12, "memory desc rank");
        let d = fill(dims);
        let mut md = ptr::null_mut();
        let (ndims, any) = (dims.len() as i32, dnnl_format_tag_any);
        // SAFETY: `d` holds 12 entries, at least `ndims` (checked above), and
        // oneDNN writes the new handle to a live local.
        let rc = unsafe { (self.api.md_tag)(&mut md, ndims, d.as_ptr(), dt, any) };
        Api::check(rc, "memory desc any")?;
        Ok(md)
    }

    pub(super) fn clone_md(&self, src: const_dnnl_memory_desc_t) -> AResult<dnnl_memory_desc_t> {
        check_shape!(!src.is_null(), "null memory desc");
        let mut md = ptr::null_mut();
        // SAFETY: `src` is a live descriptor (not null, checked above) and
        // oneDNN writes the copy's handle to a live local.
        Api::check(unsafe { (self.api.md_clone)(&mut md, src) }, "clone md")?;
        Ok(md)
    }

    pub(super) fn md_query_i32(&self, md: dnnl_memory_desc_t, what: dnnl_query_t) -> i32 {
        let mut v = 0i32;
        // SAFETY: `md` is a live descriptor the caller owns, and every query
        // this is called with writes one i32 to the live local.
        let rc = unsafe { (self.api.md_query)(md, what, (&mut v as *mut i32).cast()) };
        if rc == dnnl_success {
            v
        } else {
            0
        }
    }

    pub(super) fn md_query_dims(&self, md: dnnl_memory_desc_t, what: dnnl_query_t) -> [i64; 12] {
        let mut p: *const dnnl_dim_t = ptr::null();
        // SAFETY: `md` is a live descriptor the caller owns, and the dims
        // queries write one pointer to the live local.
        let rc =
            unsafe { (self.api.md_query)(md, what, (&mut p as *mut *const dnnl_dim_t).cast()) };
        let mut out = [0i64; 12];
        if rc == dnnl_success && !p.is_null() {
            // SAFETY: a dims query points at the descriptor's own array of
            // DNNL_MAX_NDIMS (12) entries, alive while `md` is.
            out.copy_from_slice(unsafe { std::slice::from_raw_parts(p, 12) });
        }
        out
    }

    fn impl_name(&self, pd: dnnl_primitive_desc_t) -> String {
        let mut p: *const std::ffi::c_char = ptr::null();
        let out = (&mut p as *mut *const std::ffi::c_char).cast();
        // SAFETY: `pd` is a live primitive descriptor and the query writes
        // one string pointer to the live local.
        let rc = unsafe { (self.api.pd_query)(pd, dnnl_query_impl_info_str, 0, out) };
        if rc != dnnl_success || p.is_null() {
            return "unknown".into();
        }
        // SAFETY: the implementation string is the descriptor's own, alive
        // while `pd` is, and copied out here.
        unsafe { crate::cpu::onednn::api::cstr(p) }.to_string()
    }

    pub(super) fn scratch_size(&self, pd: dnnl_primitive_desc_t) -> usize {
        // SAFETY: `pd` is a live primitive descriptor the caller owns.
        let md = unsafe { (self.api.pd_query_md)(pd, dnnl_query_scratchpad_md, 0) };
        if md.is_null() {
            0
        } else {
            // SAFETY: `md` is the descriptor's own, alive while `pd` is.
            unsafe { (self.api.md_size)(md) }
        }
    }

    pub(super) fn create_prim(&self, pd: dnnl_primitive_desc_t) -> AResult<dnnl_primitive_t> {
        let mut prim = ptr::null_mut();
        // SAFETY: `pd` is a live primitive descriptor and oneDNN writes the
        // new handle to a live local.
        let rc = unsafe { (self.api.prim_create)(&mut prim, pd) };
        Api::check(rc, "primitive")?;
        Ok(prim)
    }

    pub(super) fn mem(&self, md: dnnl_memory_desc_t) -> AResult<dnnl_memory_t> {
        let mut mem = ptr::null_mut();
        // SAFETY: `md` is a live descriptor and `self.engine` this backend's
        // engine; the memory starts with no data handle (bound per execute).
        let rc = unsafe { (self.api.mem_create)(&mut mem, md, self.engine, memory_none()) };
        Api::check(rc, "memory")?;
        Ok(mem)
    }

    /// The descriptors of a matmul key (kinds 0, 7 and 8). The ones created
    /// here go to `owned`, which the plan keeps.
    fn matmul_descs(
        &self,
        key: &Key,
        w: Option<&PackedWeight>,
        owned: &mut Vec<dnnl_memory_desc_t>,
    ) -> AResult<Descs> {
        let (g, m, n, k) = (key.g as i64, key.m as i64, key.n as i64, key.k as i64);
        check_shape!(k > 0, "matmul K");
        let (ab, ar, bb, br) = (key.ab as i64, key.ar as i64, key.bb as i64, key.br as i64);
        let src_dt = key.dt.src();
        let src = if g == 1 {
            self.md_strides_dt(&[m, k], &[ar, 1], src_dt)?
        } else {
            self.md_strides_dt(&[g, m, k], &[ab, ar, 1], src_dt)?
        };
        owned.push(src);
        let weights = if let Some(weight) = w {
            check_shape!(g == 1, "batched packed weight");
            check_shape!(
                weight.dt == key.dt,
                "packed weight dtype does not match plan"
            );
            weight.layout.packed_md
        } else {
            let wdt = key.dt.weights();
            let md = match (g == 1, key.trans) {
                (true, true) => self.md_strides_dt(&[k, n], &[1, br], wdt)?,
                (true, false) => self.md_strides_dt(&[k, n], &[br, 1], wdt)?,
                (false, true) => self.md_strides_dt(&[g, k, n], &[bb, 1, br], wdt)?,
                (false, false) => self.md_strides_dt(&[g, k, n], &[bb, br, 1], wdt)?,
            };
            owned.push(md);
            md
        };
        let dst = if g == 1 {
            self.md_strides(&[m, n], &[n, 1])?
        } else {
            self.md_strides(&[g, m, n], &[m * n, n, 1])?
        };
        owned.push(dst);
        if key.is_gated() {
            check_shape!(!key.sum, "gated projection sum");
        }
        let gate = if key.is_gated() { dst } else { ptr::null_mut() };
        let alg = match key.kind {
            7 => Some(dnnl_eltwise_gelu_erf),
            8 => Some(dnnl_eltwise_gelu_tanh),
            _ => None,
        };
        let attr = self.attr(key.sum, alg.map(|alg| (gate, alg)))?;
        let mut pd = ptr::null_mut();
        let (engine, no_bias) = (self.engine, ptr::null());
        // SAFETY: every descriptor is live (created above or the packed
        // weight's own), as is the attribute; oneDNN writes the new handle to
        // a live local.
        let rc = unsafe {
            (self.api.matmul_pd)(&mut pd, engine, src, weights, no_bias, dst, attr.raw())
        };
        Api::check(rc, "matmul pd")?;
        Ok(Descs {
            pd,
            src,
            weights,
            dst,
            scale: ptr::null_mut(),
            gate,
        })
    }

    /// The descriptors of a row-wise operator key (layernorm, softmax,
    /// GELU, SiLU, the binary multiply). The ones created here go to
    /// `owned`, which the plan keeps.
    fn operator_descs(&self, key: &Key, owned: &mut Vec<dnnl_memory_desc_t>) -> AResult<Descs> {
        let (kind, m, n) = (key.kind, key.m as i64, key.n as i64);
        let src = self.md_strides(&[m, n], &[n, 1])?;
        owned.push(src);
        let attr = self.attr(false, None)?;
        let (mut pd, engine, inference) = (ptr::null_mut(), self.engine, dnnl_forward_inference);
        let mut descs = Descs {
            pd: ptr::null_mut(),
            src,
            weights: ptr::null_mut(),
            dst: src,
            scale: ptr::null_mut(),
            gate: ptr::null_mut(),
        };
        let attr = attr.raw();
        // SAFETY (each primitive descriptor below): `src`, `scale` and the
        // attribute are live, and oneDNN writes the new handle to a live
        // local.
        let rc = match kind {
            1 => {
                descs.scale = self.md_strides(&[n], &[1])?;
                owned.push(descs.scale);
                let eps = f32::from_bits(key.param);
                let flags = dnnl_use_scale | dnnl_use_shift;
                let no_stats = ptr::null();
                // SAFETY: see above.
                unsafe {
                    (self.api.layernorm_pd)(
                        &mut pd, engine, inference, src, src, no_stats, eps, flags, attr,
                    )
                }
            }
            2 => {
                let alg = dnnl_softmax_accurate;
                // SAFETY: see above.
                unsafe { (self.api.softmax_pd)(&mut pd, engine, inference, alg, src, src, 1, attr) }
            }
            3 | 9 => {
                let (alg, alpha) = if kind == 3 {
                    (dnnl_eltwise_gelu_erf, 0.0)
                } else {
                    (dnnl_eltwise_swish, 1.0)
                };
                // SAFETY: see above.
                unsafe {
                    (self.api.eltwise_pd)(
                        &mut pd, engine, inference, alg, src, src, alpha, 0.0, attr,
                    )
                }
            }
            4 => {
                descs.weights = src;
                let mul = dnnl_binary_mul;
                // SAFETY: see above.
                unsafe { (self.api.binary_pd)(&mut pd, engine, mul, src, src, src, attr) }
            }
            _ => crate::invalid_shape!("operator kind {kind}"),
        };
        let what = match kind {
            1 => "layernorm pd",
            2 => "softmax pd",
            4 => "binary pd",
            _ => "eltwise pd",
        };
        Api::check(rc, what)?;
        descs.pd = pd;
        Ok(descs)
    }

    fn build_plan(&self, key: &Key, w: Option<&PackedWeight>) -> AResult<Plan> {
        let _ = self.api.apply_threads(self.threads);
        let mut owned = Vec::new();
        let d = if key.is_matmul() {
            self.matmul_descs(key, w, &mut owned)?
        } else {
            self.operator_descs(key, &mut owned)?
        };
        let scratch = self.scratch_size(d.pd);
        self.check_workspace("operator scratch", &[scratch])?;
        check_shape!(scratch <= 512 * 1024 * 1024, "operator scratch budget");
        let name = self.impl_name(d.pd);
        let prim = self.create_prim(d.pd)?;
        // SAFETY: `d.pd` is the live primitive descriptor built above.
        let q = unsafe { (self.api.pd_query_md)(d.pd, dnnl_query_scratchpad_md, 0) };
        let scratch_md = if q.is_null() {
            self.md_strides(&[16], &[1])?
        } else {
            self.clone_md(q)?
        };
        owned.push(scratch_md);
        // SAFETY: the primitive keeps what it needs; nothing uses `d.pd` after.
        unsafe { (self.api.pd_destroy)(d.pd) };
        let mut slot_specs = vec![
            (
                if key.kind == 4 {
                    DNNL_ARG_SRC_0
                } else {
                    DNNL_ARG_SRC
                },
                d.src,
            ),
            (DNNL_ARG_DST, d.dst),
            (DNNL_ARG_SCRATCHPAD, scratch_md),
        ];
        match key.kind {
            0 | 7 | 8 => slot_specs.push((DNNL_ARG_WEIGHTS, d.weights)),
            1 => slot_specs.extend([(DNNL_ARG_SCALE, d.scale), (DNNL_ARG_SHIFT, d.scale)]),
            4 => slot_specs.push((DNNL_ARG_SRC_1, d.src)),
            _ => {}
        }
        if key.is_gated() {
            slot_specs.push((GATE_ARG, d.gate));
        }
        let mut slots = Vec::with_capacity(slot_specs.len());
        for &(arg, md) in &slot_specs {
            slots.push(MemSlot {
                arg,
                mem: self.mem(md)?,
            });
        }
        Ok(Plan {
            api: self.api.clone(),
            prim,
            slot_specs,
            slots,
            mds: owned,
            scratch,
            name,
            used: AtomicU64::new(0),
            bytes: 0,
        })
    }

    /// Builds `key`'s plan and measures the heap it took.
    fn measured_plan(&self, key: &Key, w: Option<&PackedWeight>) -> AResult<(Plan, Instant)> {
        let t = Instant::now();
        let heap = heap_in_use();
        let mut plan = self.build_plan(key, w)?;
        // Other threads allocate meanwhile, so this is an estimate; a plan
        // holds well under 64 MiB.
        plan.bytes = heap
            .zip(heap_in_use())
            .map_or(0, |(a, b)| b.saturating_sub(a).min(64 << 20));
        Ok((plan, t))
    }

    pub(super) fn lock_state(&self) -> AResult<MutexGuard<'_, State>> {
        self.state
            .lock()
            .map_err(|_| crate::Error::poisoned("backend"))
    }

    /// Puts `key`'s plan in the cache under the state lock, building it here
    /// when `built` is `None`, and evicting the least recently executed plan
    /// when the cache is full.
    fn insert_plan<'s>(
        &'s self,
        mut s: MutexGuard<'s, State>,
        key: &Key,
        built: Option<(Plan, Instant)>,
        w: Option<&PackedWeight>,
    ) -> AResult<MutexGuard<'s, State>> {
        // Evicted since the caller's check (budget), or built here first time.
        let (plan, t) = match built {
            Some(built) => built,
            None => {
                drop(s);
                let built = self.measured_plan(key, w)?;
                s = self.lock_state()?;
                built
            }
        };
        // Evicting an arbitrary entry (HashMap order) kept evicting plans
        // the next layer needed: Vela 2.0 (the encoder the guard replaced)
        // spent ~150 ms per request re-creating plans once 256 distinct
        // shapes had been seen.
        if s.plans.len() >= s.capacity {
            let old = s
                .plans
                .iter()
                .min_by_key(|(_, plan)| plan.used.load(Ordering::Relaxed))
                .map(|(key, _)| key.clone());
            if let Some(plan) = old.and_then(|old| s.plans.remove(&old)) {
                self.retained.plans.fetch_sub(plan.bytes, Ordering::Relaxed);
                s.capacity_evictions += 1;
            }
        }
        self.retained.plans.fetch_add(plan.bytes, Ordering::Relaxed);
        s.plans.insert(key.clone(), Arc::new(plan));
        s.creations += 1;
        s.setup_s += t.elapsed().as_secs_f64();
        self.enforce_budget(&mut s, key);
        Ok(s)
    }

    pub(super) fn run(&self, key: Key, ops: Operands, out: &mut [f32]) -> AResult<()> {
        let a_ok = ops.a.len() > 0;
        check_shape!(a_ok && !out.is_empty(), "empty operator input");
        ops.check(&key, out.len())?;
        let mut ws = self
            .exec
            .acquire(|| ExecWorkspace::try_new(&self.api, self.engine, self.threadpool()))?;
        // Create a missing plan before taking the compute lock: primitive
        // creation (JIT, descriptors) touches no other caller's state, so a
        // cold shape on one model no longer stalls every model in the process.
        let missing = !self.lock_state()?.plans.contains_key(&key);
        let built = if missing {
            Some(self.measured_plan(&key, ops.weight)?)
        } else {
            None
        };
        // After the exec slot: AMX is programmed on this caller thread inside
        // `prim_execute`, so two in-flight executes must not overlap.
        let _compute = compute_lock();
        let mut s = self.lock_state()?;
        if !s.plans.contains_key(&key) {
            s = self.insert_plan(s, &key, built, ops.weight)?;
        }
        s.tick += 1;
        let tick = s.tick;
        // Held through the execute: inside a pool job another batch may
        // evict this plan from the cache meanwhile.
        let plan = Arc::clone(s.plans.get(&key).context("execution plan")?);
        plan.used.store(tick, Ordering::Relaxed);
        self.check_workspace("operator scratch", &[plan.scratch])?;
        if ws.scratch.bytes < plan.scratch.max(64) {
            let old = ws.scratch.bytes;
            ws.scratch = Aligned::try_new(plan.scratch.max(64))?;
            self.retained
                .exec_scratch
                .fetch_add(ws.scratch.bytes - old, Ordering::Relaxed);
        }
        // Plan-owned memories are shared, so they are only bound under the
        // state lock. With several slots, or inside a pool job, each execute
        // binds its own and lets the others run.
        let concurrent = self.exec_slots > 1 || crate::cpu::threads::in_parallel();
        let state_guard = if concurrent {
            drop(s);
            None
        } else {
            Some(s)
        };
        let _ = self.api.apply_threads(self.threads);
        let t_conv = Instant::now();
        let (handles, converted) = self.handles(&key, &ops, &mut ws, out)?;
        let convert_dt = if converted {
            t_conv.elapsed().as_secs_f64()
        } else {
            0.0
        };
        let t = Instant::now();
        self.execute(&plan, concurrent, &handles, &ws)?;
        let elapsed = t.elapsed().as_secs_f64();
        let s = match state_guard {
            Some(g) => g,
            None => self.lock_state()?,
        };
        record(s, &key, elapsed, convert_dt);
        Ok(())
    }

    /// The data handle bound to each kind of argument, converting f32
    /// operands of a BF16 matmul into the workspace first. Says whether it
    /// converted.
    fn handles(
        &self,
        key: &Key,
        ops: &Operands,
        ws: &mut ExecWorkspace,
        out: &mut [f32],
    ) -> AResult<(Handles, bool)> {
        let mut converted = false;
        let src = match ops.a {
            Operand::Bf16(bf) => bf.as_ptr() as *mut c_void,
            Operand::F32(a) if key.dt == MatDt::Bf16 => {
                converted = true;
                fill_bf16(&mut ws.bf16_a, a, self.threads)?
            }
            Operand::F32(a) => a.as_ptr() as *mut c_void,
        };
        let b = if let Some(w) = ops.weight {
            w.data.ptr()
        } else {
            match ops.b {
                Some(Operand::Bf16(bf)) => bf.as_ptr() as *mut c_void,
                Some(Operand::F32(bv)) if key.dt == MatDt::Bf16 => {
                    converted = true;
                    fill_bf16(&mut ws.bf16_b, bv, self.threads)?
                }
                Some(Operand::F32(bv)) => bv.as_ptr() as *mut c_void,
                None => ptr::null_mut(),
            }
        };
        let shift = ops
            .shift
            .map_or(ptr::null_mut(), |v| v.as_ptr() as *mut c_void);
        let dst = out.as_mut_ptr() as *mut c_void;
        Ok((Handles { src, b, shift, dst }, converted))
    }

    /// Binds the handles to the plan's memories and runs the primitive to
    /// completion on the workspace's stream.
    fn execute(
        &self,
        plan: &Plan,
        concurrent: bool,
        h: &Handles,
        ws: &ExecWorkspace,
    ) -> AResult<()> {
        let mut ephemeral = Vec::new();
        let slots: Vec<(i32, dnnl_memory_t)> = if concurrent {
            for &(arg, md) in &plan.slot_specs {
                ephemeral.push((arg, self.mem(md)?));
            }
            ephemeral.clone()
        } else {
            plan.slots.iter().map(|slot| (slot.arg, slot.mem)).collect()
        };
        let scratch = ws.scratch.ptr();
        let mut args: Vec<dnnl_exec_arg_t> = Vec::with_capacity(slots.len());
        let mut rc = Ok(());
        for &(arg, memory) in &slots {
            // `SRC_0`, the binary multiply's first input, is the same
            // argument as `SRC`.
            let handle = match arg {
                DNNL_ARG_SRC => h.src,
                DNNL_ARG_DST => h.dst,
                DNNL_ARG_SCRATCHPAD => scratch,
                DNNL_ARG_WEIGHTS | DNNL_ARG_SRC_1 | DNNL_ARG_SCALE => h.b,
                DNNL_ARG_SHIFT | GATE_ARG => h.shift,
                _ => ptr::null_mut(),
            };
            // SAFETY: `memory` is one of this plan's live memories, bound
            // only by this call (the state lock holds the plan's own; the
            // ephemeral ones are this call's); `handle` points into a buffer
            // `Operands::check` measured against the plan, or into the
            // workspace scratch sized to `plan.scratch` above.
            rc = Api::check(unsafe { (self.api.mem_set)(memory, handle) }, "set handle");
            if rc.is_err() {
                break;
            }
            args.push(dnnl_exec_arg_t { arg, memory });
        }
        if rc.is_ok() {
            let (n, stream) = (args.len() as i32, ws.stream);
            // SAFETY: `plan.prim` is alive (the caller holds the plan), every
            // argument is bound above, and waiting on the stream before
            // returning keeps each bound buffer alive and unaliased for the
            // whole execute.
            rc = Api::check(
                unsafe { (self.api.prim_execute)(plan.prim, stream, n, args.as_ptr()) },
                "execute",
            )
            // SAFETY: `stream` is this workspace's live stream.
            .and_then(|()| Api::check(unsafe { (self.api.stream_wait)(stream) }, "wait"));
        }
        for (_, mem) in ephemeral {
            // SAFETY: created above for this call alone, and the execute has
            // finished with it.
            unsafe { (self.api.mem_destroy)(mem) };
        }
        rc
    }
}

/// The post-op argument the gated matmuls read their gate from: the second
/// input of post-op 1, the binary multiply after the GELU at post-op 0.
const GATE_ARG: i32 = (DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE * 2) | DNNL_ARG_SRC_1;

/// The data handles one execute binds, by argument.
struct Handles {
    src: *mut c_void,
    b: *mut c_void,
    shift: *mut c_void,
    dst: *mut c_void,
}

/// Adds one execute to the backend's statistics.
fn record(mut s: MutexGuard<'_, State>, key: &Key, elapsed: f64, convert_dt: f64) {
    if convert_dt > 0.0 {
        s.convert_s += convert_dt;
    }
    *s.executions.entry(key.kind).or_default() += 1;
    *s.exec_s.entry(key.kind).or_default() += elapsed;
    let category = if key.is_matmul() {
        if key.weight == 0 {
            "attention_or_span"
        } else {
            "packed_linear"
        }
    } else {
        "other_primitives"
    };
    *s.category_s.entry(category.into()).or_default() += elapsed;
}
