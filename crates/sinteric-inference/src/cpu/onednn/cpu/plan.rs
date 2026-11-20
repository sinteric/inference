//! Primitive plans: the memory descriptors and attributes a primitive is
//! built from, the plan cache keyed by shape, and executing a plan on a
//! leased stream and scratchpad.

use super::{
    fill, fill_bf16, heap_in_use, Aligned, ExecWorkspace, Key, MatDt, MemSlot, OnednnCpu,
    PackedWeight, Plan,
};
use crate::cpu::onednn::api::{memory_none, Api};
use crate::cpu::threads::compute_lock;
use crate::{check_shape, Result as AResult, ResultContext as Context};
use onednn_sys::*;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

impl OnednnCpu {
    pub(super) fn strict_attr(
        &self,
        sum: bool,
        gate: Option<dnnl_memory_desc_t>,
    ) -> AResult<(dnnl_primitive_attr_t, Option<dnnl_post_ops_t>)> {
        let mut attr = ptr::null_mut();
        Api::check(unsafe { (self.api.attr_create)(&mut attr) }, "attr")?;
        Api::check(
            unsafe { (self.api.attr_fpmath)(attr, dnnl_fpmath_mode_strict) },
            "fpmath",
        )?;
        Api::check(
            unsafe { (self.api.attr_accum)(attr, dnnl_accumulation_mode_strict) },
            "accumulation",
        )?;
        Api::check(
            unsafe { (self.api.attr_scratch)(attr, dnnl_scratchpad_mode_user) },
            "scratchpad",
        )?;
        let mut post = ptr::null_mut();
        if sum || gate.is_some() {
            Api::check(unsafe { (self.api.post_ops_create)(&mut post) }, "post_ops")?;
            if sum {
                Api::check(
                    unsafe { (self.api.post_ops_sum)(post, 1.0, 0, dnnl_data_type_undef) },
                    "sum post-op",
                )?;
            }
            if let Some(md) = gate {
                Api::check(
                    unsafe { (self.api.post_ops_eltwise)(post, dnnl_eltwise_gelu_erf, 0.0, 0.0) },
                    "gelu post-op",
                )?;
                Api::check(
                    unsafe { (self.api.post_ops_binary)(post, dnnl_binary_mul, md) },
                    "binary-mul post-op",
                )?;
            }
            Api::check(
                unsafe { (self.api.attr_post_ops)(attr, post) },
                "set post_ops",
            )?;
        }
        Ok((attr, if post.is_null() { None } else { Some(post) }))
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
        let d = fill(dims);
        let s = fill(strides);
        let mut md = ptr::null_mut();
        Api::check(
            unsafe {
                (self.api.md_strides)(&mut md, dims.len() as i32, d.as_ptr(), dt, s.as_ptr())
            },
            "memory desc strides",
        )?;
        Ok(md)
    }

    pub(super) fn md_any_dt(
        &self,
        dims: &[i64],
        dt: dnnl_data_type_t,
    ) -> AResult<dnnl_memory_desc_t> {
        let d = fill(dims);
        let mut md = ptr::null_mut();
        Api::check(
            unsafe {
                (self.api.md_tag)(
                    &mut md,
                    dims.len() as i32,
                    d.as_ptr(),
                    dt,
                    dnnl_format_tag_any,
                )
            },
            "memory desc any",
        )?;
        Ok(md)
    }

    pub(super) fn bf16_attr(
        &self,
        sum: bool,
        gate: Option<(dnnl_memory_desc_t, dnnl_alg_kind_t)>,
    ) -> AResult<(dnnl_primitive_attr_t, Option<dnnl_post_ops_t>)> {
        let mut attr = ptr::null_mut();
        Api::check(unsafe { (self.api.attr_create)(&mut attr) }, "attr")?;
        Api::check(
            unsafe { (self.api.attr_fpmath)(attr, dnnl_fpmath_mode_strict) },
            "fpmath",
        )?;
        // Strict accumulation is f32 for non-quantized (incl. BF16) matmuls.
        // Explicit `accumulation_mode_f32` is a non-default attr that brgemm
        // / gemm_bf16 reject as unimplemented.
        Api::check(
            unsafe { (self.api.attr_accum)(attr, dnnl_accumulation_mode_strict) },
            "accumulation",
        )?;
        Api::check(
            unsafe { (self.api.attr_scratch)(attr, dnnl_scratchpad_mode_user) },
            "scratchpad",
        )?;
        let mut post = ptr::null_mut();
        if sum || gate.is_some() {
            Api::check(unsafe { (self.api.post_ops_create)(&mut post) }, "post_ops")?;
            if sum {
                Api::check(
                    unsafe { (self.api.post_ops_sum)(post, 1.0, 0, dnnl_data_type_undef) },
                    "sum post-op",
                )?;
            }
            if let Some((md, alg)) = gate {
                Api::check(
                    unsafe { (self.api.post_ops_eltwise)(post, alg, 0.0, 0.0) },
                    "gelu post-op",
                )?;
                Api::check(
                    unsafe { (self.api.post_ops_binary)(post, dnnl_binary_mul, md) },
                    "binary-mul post-op",
                )?;
            }
            Api::check(
                unsafe { (self.api.attr_post_ops)(attr, post) },
                "set post_ops",
            )?;
        }
        Ok((attr, if post.is_null() { None } else { Some(post) }))
    }

    pub(super) fn clone_md(&self, src: const_dnnl_memory_desc_t) -> AResult<dnnl_memory_desc_t> {
        check_shape!(!src.is_null(), "null memory desc");
        let mut md = ptr::null_mut();
        Api::check(unsafe { (self.api.md_clone)(&mut md, src) }, "clone md")?;
        Ok(md)
    }

    pub(super) fn md_query_i32(&self, md: dnnl_memory_desc_t, what: dnnl_query_t) -> i32 {
        let mut v = 0i32;
        let rc = unsafe { (self.api.md_query)(md, what, (&mut v as *mut i32).cast()) };
        if rc == dnnl_success {
            v
        } else {
            0
        }
    }

    pub(super) fn md_query_dims(&self, md: dnnl_memory_desc_t, what: dnnl_query_t) -> [i64; 12] {
        let mut p: *const dnnl_dim_t = ptr::null();
        let rc =
            unsafe { (self.api.md_query)(md, what, (&mut p as *mut *const dnnl_dim_t).cast()) };
        let mut out = [0i64; 12];
        if rc == dnnl_success && !p.is_null() {
            unsafe {
                out.copy_from_slice(std::slice::from_raw_parts(p, 12));
            }
        }
        out
    }

    fn impl_name(&self, pd: dnnl_primitive_desc_t) -> String {
        let mut p: *const std::ffi::c_char = ptr::null();
        let rc = unsafe {
            (self.api.pd_query)(
                pd,
                dnnl_query_impl_info_str,
                0,
                (&mut p as *mut *const std::ffi::c_char).cast(),
            )
        };
        if rc != dnnl_success || p.is_null() {
            return "unknown".into();
        }
        crate::cpu::onednn::api::cstr(p).to_string()
    }

    pub(super) fn scratch_size(&self, pd: dnnl_primitive_desc_t) -> usize {
        let md = unsafe { (self.api.pd_query_md)(pd, dnnl_query_scratchpad_md, 0) };
        if md.is_null() {
            0
        } else {
            unsafe { (self.api.md_size)(md) }
        }
    }

    pub(super) fn create_prim(&self, pd: dnnl_primitive_desc_t) -> AResult<dnnl_primitive_t> {
        let mut prim = ptr::null_mut();
        Api::check(
            unsafe { (self.api.prim_create)(&mut prim, pd) },
            "primitive",
        )?;
        Ok(prim)
    }

    pub(super) fn mem(&self, md: dnnl_memory_desc_t) -> AResult<dnnl_memory_t> {
        let mut mem = ptr::null_mut();
        Api::check(
            unsafe { (self.api.mem_create)(&mut mem, md, self.engine, memory_none()) },
            "memory",
        )?;
        Ok(mem)
    }

    fn build_plan(&self, key: &Key, w: Option<&PackedWeight>) -> AResult<Plan> {
        let _ = self.api.apply_threads(self.threads);
        let kind = key.kind;
        let g = key.g as i64;
        let m = key.m as i64;
        let n = key.n as i64;
        let k = key.k as i64;
        let mut owned_mds: Vec<dnnl_memory_desc_t> = Vec::new();
        let src_md;
        let mut weights_md = ptr::null_mut();
        let dst_md;
        let mut scale_md = ptr::null_mut();
        let mut gate_md = ptr::null_mut();
        let mut pd = ptr::null_mut();
        let (attr, post);

        if kind == 0 || kind == 7 || kind == 8 {
            check_shape!(k > 0, "matmul K");
            let src_dt = key.dt.src();
            src_md = if g == 1 {
                self.md_strides_dt(&[m, k], &[key.ar as i64, 1], src_dt)?
            } else {
                self.md_strides_dt(&[g, m, k], &[key.ab as i64, key.ar as i64, 1], src_dt)?
            };
            owned_mds.push(src_md);
            if let Some(weight) = w {
                check_shape!(g == 1, "batched packed weight");
                check_shape!(
                    weight.dt == key.dt,
                    "packed weight dtype does not match plan"
                );
                weights_md = weight.layout.packed_md;
            } else if g == 1 {
                weights_md = if key.trans {
                    self.md_strides_dt(&[k, n], &[1, key.br as i64], key.dt.weights())?
                } else {
                    self.md_strides_dt(&[k, n], &[key.br as i64, 1], key.dt.weights())?
                };
                owned_mds.push(weights_md);
            } else {
                weights_md = if key.trans {
                    self.md_strides_dt(
                        &[g, k, n],
                        &[key.bb as i64, 1, key.br as i64],
                        key.dt.weights(),
                    )?
                } else {
                    self.md_strides_dt(
                        &[g, k, n],
                        &[key.bb as i64, key.br as i64, 1],
                        key.dt.weights(),
                    )?
                };
                owned_mds.push(weights_md);
            }
            dst_md = if g == 1 {
                self.md_strides(&[m, n], &[n, 1])?
            } else {
                self.md_strides(&[g, m, n], &[m * n, n, 1])?
            };
            owned_mds.push(dst_md);
            if kind == 7 || kind == 8 {
                check_shape!(!key.sum, "gated projection sum");
                gate_md = dst_md;
            }
            let pair = if key.dt != MatDt::F32 || kind == 8 {
                let gate = if kind == 7 {
                    Some((gate_md, dnnl_eltwise_gelu_erf))
                } else if kind == 8 {
                    Some((gate_md, dnnl_eltwise_gelu_tanh))
                } else {
                    None
                };
                self.bf16_attr(key.sum, gate)?
            } else {
                self.strict_attr(key.sum, if kind == 7 { Some(gate_md) } else { None })?
            };
            attr = pair.0;
            post = pair.1;
            Api::check(
                unsafe {
                    (self.api.matmul_pd)(
                        &mut pd,
                        self.engine,
                        src_md,
                        weights_md,
                        ptr::null(),
                        dst_md,
                        attr,
                    )
                },
                "matmul pd",
            )?;
        } else {
            src_md = self.md_strides(&[m, n], &[n, 1])?;
            dst_md = src_md;
            owned_mds.push(src_md);
            let pair = self.strict_attr(false, None)?;
            attr = pair.0;
            post = pair.1;
            if kind == 1 {
                scale_md = self.md_strides(&[n], &[1])?;
                owned_mds.push(scale_md);
                let eps = f32::from_bits(key.param);
                Api::check(
                    unsafe {
                        (self.api.layernorm_pd)(
                            &mut pd,
                            self.engine,
                            dnnl_forward_inference,
                            src_md,
                            dst_md,
                            ptr::null(),
                            eps,
                            dnnl_use_scale | dnnl_use_shift,
                            attr,
                        )
                    },
                    "layernorm pd",
                )?;
            } else if kind == 2 {
                Api::check(
                    unsafe {
                        (self.api.softmax_pd)(
                            &mut pd,
                            self.engine,
                            dnnl_forward_inference,
                            dnnl_softmax_accurate,
                            src_md,
                            dst_md,
                            1,
                            attr,
                        )
                    },
                    "softmax pd",
                )?;
            } else if kind == 3 || kind == 9 {
                let alg = if kind == 3 {
                    dnnl_eltwise_gelu_erf
                } else {
                    dnnl_eltwise_swish
                };
                Api::check(
                    unsafe {
                        (self.api.eltwise_pd)(
                            &mut pd,
                            self.engine,
                            dnnl_forward_inference,
                            alg,
                            src_md,
                            dst_md,
                            if kind == 9 { 1.0 } else { 0.0 },
                            0.0,
                            attr,
                        )
                    },
                    "eltwise pd",
                )?;
            } else if kind == 4 {
                weights_md = src_md;
                Api::check(
                    unsafe {
                        (self.api.binary_pd)(
                            &mut pd,
                            self.engine,
                            dnnl_binary_mul,
                            src_md,
                            weights_md,
                            dst_md,
                            attr,
                        )
                    },
                    "binary pd",
                )?;
            } else {
                crate::invalid_shape!("operator kind {kind}");
            }
        }

        unsafe {
            (self.api.attr_destroy)(attr);
            if let Some(p) = post {
                (self.api.post_ops_destroy)(p);
            }
        }
        let scratch = self.scratch_size(pd);
        self.check_workspace("operator scratch", &[scratch])?;
        check_shape!(scratch <= 512 * 1024 * 1024, "operator scratch budget");
        let name = self.impl_name(pd);
        let prim = self.create_prim(pd)?;
        let q = unsafe { (self.api.pd_query_md)(pd, dnnl_query_scratchpad_md, 0) };
        let scratch_md = if q.is_null() {
            self.md_strides(&[16], &[1])?
        } else {
            self.clone_md(q)?
        };
        owned_mds.push(scratch_md);
        unsafe { (self.api.pd_destroy)(pd) };

        let mut slots = Vec::new();
        let mut slot_specs = Vec::new();
        let src_arg = if kind == 4 {
            DNNL_ARG_SRC_0
        } else {
            DNNL_ARG_SRC
        };
        let mut push_slot = |arg: i32, md: dnnl_memory_desc_t| -> AResult<()> {
            slots.push(MemSlot {
                arg,
                mem: self.mem(md)?,
            });
            slot_specs.push((arg, md));
            Ok(())
        };
        push_slot(src_arg, src_md)?;
        push_slot(DNNL_ARG_DST, dst_md)?;
        push_slot(DNNL_ARG_SCRATCHPAD, scratch_md)?;
        if kind == 0 || kind == 7 || kind == 8 {
            push_slot(DNNL_ARG_WEIGHTS, weights_md)?;
        } else if kind == 1 {
            push_slot(DNNL_ARG_SCALE, scale_md)?;
            push_slot(DNNL_ARG_SHIFT, scale_md)?;
        } else if kind == 4 {
            push_slot(DNNL_ARG_SRC_1, src_md)?;
        }
        if kind == 7 || kind == 8 {
            push_slot(
                (DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE * 2) | DNNL_ARG_SRC_1,
                gate_md,
            )?;
        }

        Ok(Plan {
            api: self.api.clone(),
            prim,
            slot_specs,
            slots,
            mds: owned_mds,
            scratch,
            name,
            used: AtomicU64::new(0),
            bytes: 0,
        })
    }

    pub(super) fn run(
        &self,
        key: Key,
        a: &[f32],
        b: Option<&[f32]>,
        shift: Option<&[f32]>,
        out: &mut [f32],
        w: Option<&PackedWeight>,
    ) -> AResult<()> {
        self.run_ex(key, a, None, b, None, shift, out, w)
    }

    pub(super) fn run_ex(
        &self,
        key: Key,
        a: &[f32],
        a_bf: Option<&[u16]>,
        b: Option<&[f32]>,
        b_bf: Option<&[u16]>,
        shift: Option<&[f32]>,
        out: &mut [f32],
        w: Option<&PackedWeight>,
    ) -> AResult<()> {
        let a_ok = a_bf.map(|v| !v.is_empty()).unwrap_or(!a.is_empty());
        check_shape!(a_ok && !out.is_empty(), "empty operator input");
        let mut ws = self
            .exec
            .acquire(|| ExecWorkspace::new(&self.api, self.engine, self.threadpool()))?;
        // Create a missing plan before taking the compute lock: primitive
        // creation (JIT, descriptors) touches no other caller's state, so a
        // cold shape on one model no longer stalls every model in the process.
        let missing = !self
            .state
            .lock()
            .map_err(|_| crate::Error::Poisoned("backend"))?
            .plans
            .contains_key(&key);
        let mut built = None;
        if missing {
            let t = Instant::now();
            let heap = heap_in_use();
            let mut plan = self.build_plan(&key, w)?;
            // Other threads allocate meanwhile, so this is an estimate; a
            // plan holds well under 64 MiB.
            plan.bytes = heap
                .zip(heap_in_use())
                .map_or(0, |(a, b)| b.saturating_sub(a).min(64 << 20));
            built = Some((plan, t));
        }
        // After the exec slot: AMX is programmed on this caller thread inside
        // `prim_execute`, so two in-flight executes must not overlap.
        let _compute = compute_lock();
        let mut s = self
            .state
            .lock()
            .map_err(|_| crate::Error::Poisoned("backend"))?;
        if !s.plans.contains_key(&key) {
            // Evicted since the check above (budget), or built here first time.
            let (plan, t) = match built {
                Some(built) => built,
                None => {
                    let t = Instant::now();
                    drop(s);
                    let heap = heap_in_use();
                    let mut plan = self.build_plan(&key, w)?;
                    plan.bytes = heap
                        .zip(heap_in_use())
                        .map_or(0, |(a, b)| b.saturating_sub(a).min(64 << 20));
                    s = self
                        .state
                        .lock()
                        .map_err(|_| crate::Error::Poisoned("backend"))?;
                    (plan, t)
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
            self.enforce_budget(&mut s, &key);
        }
        s.tick += 1;
        let tick = s.tick;
        // Held through the execute: inside a pool job another batch may
        // evict this plan from the cache meanwhile.
        let plan = Arc::clone(s.plans.get(&key).context("execution plan")?);
        plan.used.store(tick, Ordering::Relaxed);
        let scratch_need = plan.scratch;
        self.check_workspace("operator scratch", &[scratch_need])?;
        if ws.scratch.bytes < scratch_need.max(64) {
            let old = ws.scratch.bytes;
            ws.scratch = Aligned::new(scratch_need.max(64))?;
            self.retained
                .exec_scratch
                .fetch_add(ws.scratch.bytes - old, Ordering::Relaxed);
        }
        let prim = plan.prim;
        // Plan-owned memories are shared, so they are only bound under the
        // state lock. With several slots, or inside a pool job, each execute
        // binds its own and lets the others run.
        let concurrent = self.exec_slots > 1 || crate::cpu::threads::in_parallel();
        let slot_specs = if concurrent {
            plan.slot_specs.clone()
        } else {
            Vec::new()
        };
        let slots: Vec<(i32, dnnl_memory_t)> = if concurrent {
            Vec::new()
        } else {
            plan.slots.iter().map(|slot| (slot.arg, slot.mem)).collect()
        };
        let mut state_guard = if concurrent {
            drop(s);
            None
        } else {
            Some(s)
        };
        let _ = self.api.apply_threads(self.threads);
        let t_conv = Instant::now();
        let mut converted = false;
        let src_p = if let Some(bf) = a_bf {
            bf.as_ptr() as *mut c_void
        } else if key.dt == MatDt::Bf16 {
            converted = true;
            fill_bf16(&mut ws.bf16_a, a, self.threads)?
        } else {
            a.as_ptr() as *mut c_void
        };
        let bp = if let Some(v) = w {
            v.data.ptr.as_ptr() as *mut c_void
        } else if let Some(bf) = b_bf {
            bf.as_ptr() as *mut c_void
        } else if key.dt == MatDt::Bf16 {
            if let Some(bv) = b {
                converted = true;
                fill_bf16(&mut ws.bf16_b, bv, self.threads)?
            } else {
                ptr::null_mut()
            }
        } else {
            b.map(|v| v.as_ptr() as *mut c_void)
                .unwrap_or(ptr::null_mut())
        };
        let convert_dt = if converted {
            t_conv.elapsed().as_secs_f64()
        } else {
            0.0
        };
        let shift_p = shift
            .map(|v| v.as_ptr() as *mut c_void)
            .unwrap_or(ptr::null_mut());
        let t = Instant::now();
        let stream = ws.stream;
        let scratch_ptr = ws.scratch.ptr();
        let mut ephemeral = Vec::new();
        let slot_pairs: Vec<(i32, dnnl_memory_t)> = if concurrent {
            for (arg, md) in slot_specs {
                ephemeral.push((arg, self.mem(md)?));
            }
            ephemeral.clone()
        } else {
            slots
        };
        unsafe {
            let mut args: Vec<dnnl_exec_arg_t> = Vec::with_capacity(slot_pairs.len());
            for (arg, mem) in &slot_pairs {
                let handle = if *arg == DNNL_ARG_SRC || *arg == DNNL_ARG_SRC_0 {
                    src_p
                } else if *arg == DNNL_ARG_DST {
                    out.as_mut_ptr() as *mut c_void
                } else if *arg == DNNL_ARG_SCRATCHPAD {
                    scratch_ptr
                } else if *arg == DNNL_ARG_WEIGHTS
                    || *arg == DNNL_ARG_SRC_1
                    || *arg == DNNL_ARG_SCALE
                {
                    bp
                } else if *arg == DNNL_ARG_SHIFT
                    || *arg == (DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE * 2) | DNNL_ARG_SRC_1
                {
                    shift_p
                } else {
                    ptr::null_mut()
                };
                Api::check((self.api.mem_set)(*mem, handle), "set handle")?;
                args.push(dnnl_exec_arg_t {
                    arg: *arg,
                    memory: *mem,
                });
            }
            Api::check(
                (self.api.prim_execute)(prim, stream, args.len() as i32, args.as_ptr()),
                "execute",
            )?;
            Api::check((self.api.stream_wait)(stream), "wait")?;
            if concurrent {
                for (_, mem) in ephemeral {
                    (self.api.mem_destroy)(mem);
                }
            }
        }
        let elapsed = t.elapsed().as_secs_f64();
        let mut s = match state_guard.take() {
            Some(g) => g,
            None => self
                .state
                .lock()
                .map_err(|_| crate::Error::Poisoned("backend"))?,
        };
        if convert_dt > 0.0 {
            s.convert_s += convert_dt;
        }
        *s.executions.entry(key.kind).or_default() += 1;
        *s.exec_s.entry(key.kind).or_default() += elapsed;
        let category = if key.kind == 0 || key.kind == 7 || key.kind == 8 {
            if key.weight == 0 {
                "attention_or_span"
            } else {
                "packed_linear"
            }
        } else {
            "other_primitives"
        };
        *s.category_s.entry(category.into()).or_default() += elapsed;
        Ok(())
    }
}
