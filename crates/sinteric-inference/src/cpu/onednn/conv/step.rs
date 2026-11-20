//! One step of a plan: a primitive with its bound arguments, or a glue
//! pass (spatial mean, concat, channel shuffle, pointwise tile, zero pad)
//! run on the pool, and the memory and stream handles it executes with.

use super::{
    native_key, Binding, Bound, Md, MdOwner, Memory, NativeCache, Node, Pd, Primitive, Step, Stream,
};
use crate::cpu::onednn::api::Api;
use crate::cpu::onednn::OnednnCpu;
use crate::{check_shape, checked_size, Result};
use onednn_sys::*;
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

impl Step {
    pub(super) fn new(pd: Pd, specs: Vec<(i32, Md, Binding)>) -> Result<Self> {
        Self::new_cached(pd, specs, None)
    }
    pub(super) fn new_cached(
        pd: Pd,
        specs: Vec<(i32, Md, Binding)>,
        cache: Option<(&mut NativeCache, &Node, usize)>,
    ) -> Result<Self> {
        let (key, cache) = if let Some((cache, node, capacity)) = cache {
            let key = native_key(node, &specs)?;
            if cache.len() >= capacity {
                cache.retain(|_, p| p.strong_count() > 0);
            }
            (Some((key, capacity)), Some(cache))
        } else {
            (None, None)
        };
        let cached = key
            .as_ref()
            .and_then(|(key, _)| cache.as_ref()?.get(key)?.upgrade());
        let primitive = if let Some(p) = cached {
            p
        } else {
            let mut primitive = Primitive(ptr::null_mut());
            Api::check(
                // SAFETY: `pd` is live; oneDNN writes the new handle into
                // `primitive`, which owns it.
                unsafe { dnnl_primitive_create(&mut primitive.0, pd.0) },
                "conv primitive",
            )?;
            let primitive = Arc::new(primitive);
            if let (Some((key, capacity)), Some(cache)) = (key, cache) {
                if cache.len() < capacity {
                    cache.insert(key, Arc::downgrade(&primitive));
                }
            }
            primitive
        };
        let mut info = ptr::null::<std::ffi::c_char>();
        Api::check(
            // SAFETY: `pd` is live and `info` is a valid out-pointer for the
            // string this query returns.
            unsafe {
                dnnl_primitive_desc_query(
                    pd.0,
                    dnnl_query_impl_info_str,
                    0,
                    (&mut info as *mut *const std::ffi::c_char).cast(),
                )
            },
            "conv implementation query",
        )?;
        // SAFETY: the implementation string is the descriptor's own, alive
        // while `pd` is, and copied out here.
        let implementation = unsafe { crate::cpu::onednn::api::cstr(info) }.to_owned();
        // SAFETY: `pd` is live; the descriptor returned is the descriptor's
        // own and is cloned below before `pd` can go.
        let md = unsafe { dnnl_primitive_desc_query_md(pd.0, dnnl_query_scratchpad_md, 0) };
        let scratch = if md.is_null() {
            None
        } else {
            let mut copy = ptr::null_mut();
            Api::check(
                // SAFETY: `md` is non-null and alive with `pd`; oneDNN writes
                // the clone into `copy`, which `MdOwner` then owns.
                unsafe { dnnl_memory_desc_clone(&mut copy, md) },
                "scratch descriptor",
            )?;
            Some(Md(Arc::new(MdOwner(copy))))
        };
        let scratch_bytes = scratch
            .as_ref()
            // SAFETY: `m` owns a live descriptor.
            .map(|m| unsafe { dnnl_memory_desc_get_size(m.raw()) })
            .unwrap_or(0)
            .max(64);
        check_shape!(scratch_bytes <= isize::MAX as usize, "scratchpad too large");
        Ok(Self {
            primitive,
            specs,
            scratch,
            scratch_bytes,
            implementation,
            mean: None,
            concat: None,
            shuffle: None,
            tile: None,
            pad: None,
        })
    }
    /// Checks that every buffer holds what this step reads or writes
    /// through it: each bound descriptor's size, and the extents the glue
    /// passes index. `blk` is the CPU's channel block.
    fn check_extents(&self, bufs: &[Bound], scratch: Bound, blk: usize) -> Result<()> {
        check_shape!(bufs.len() == self.specs.len(), "conv step bindings");
        check_shape!(
            self.scratch_bytes <= scratch.bytes,
            "conv scratch past its buffer"
        );
        let floats = |i: usize| bufs[i].bytes / 4;
        let mut sizes = Vec::with_capacity(self.specs.len());
        for (_, md, _) in &self.specs {
            sizes.push(md.bytes()?);
        }
        if let Some(t) = self.tile {
            // The tile's descriptors describe its scratch windows, not the
            // tensors it copies them from and to.
            let ib = t.shape[1].div_ceil(blk);
            let ob = t.output_channels.div_ceil(blk);
            let window = |blocks: usize| checked_size("pointwise tile", &[blocks, t.width, blk, 4]);
            check_shape!(
                sizes.len() >= 2
                    && window(ib)? <= sizes[0]
                    && window(ob)? <= sizes[1]
                    && t.source_offset
                        .checked_add(sizes[0])
                        .is_some_and(|e| e <= t.destination_offset)
                    && t.destination_offset
                        .checked_add(sizes[1])
                        .is_some_and(|e| e <= scratch.bytes)
                    && checked_size("tile source", &[t.shape[0], ib, t.plane, blk])? <= floats(0)
                    && checked_size("tile output", &[t.shape[0], ob, t.plane, blk])? <= floats(1)
                    && sizes[2..]
                        .iter()
                        .zip(&bufs[2..])
                        .all(|(&n, b)| n <= b.bytes),
                "pointwise tile extents"
            );
            return Ok(());
        }
        for (&n, b) in sizes.iter().zip(bufs) {
            check_shape!(
                n <= b.bytes,
                "conv tensor of {n} bytes past its buffer of {}",
                b.bytes
            );
        }
        if let Some(p) = self.pad {
            let src = checked_size("pad source", &[p.planes, p.height, p.width, p.lanes])?;
            let dst = checked_size(
                "pad output",
                &[p.planes, p.padded_height, p.padded_width, p.lanes],
            )?;
            check_shape!(
                bufs.len() >= 2
                    && src <= floats(0)
                    && dst <= floats(1)
                    && p.top + p.height <= p.padded_height
                    && p.left + p.width <= p.padded_width,
                "pad extents"
            );
        }
        if let Some(s) = self.shuffle {
            let (n, c, h, w) = (s[0], s[1] / 4, s[2], s[3]);
            let src = checked_size("shuffle source", &[n, s[1].div_ceil(blk), blk, h, w])?;
            // Blocked only when the channels fill whole blocks, so either
            // way the output holds exactly its `n * c * 4 * h * w` floats.
            let dst = checked_size("shuffle output", &[n, c, 4, h, w])?;
            check_shape!(
                bufs.len() >= 2 && src <= floats(0) && dst <= floats(1),
                "shuffle extents"
            );
        }
        if let Some((batch, plane, channels)) = &self.concat {
            let full = checked_size("concat output", &[*batch, channels.iter().sum(), *plane])?;
            check_shape!(
                bufs.len() == channels.len() + 1 && full <= floats(0),
                "concat extents"
            );
            for (i, &c) in channels.iter().enumerate() {
                check_shape!(
                    checked_size("concat input", &[*batch, c, *plane])? <= floats(i + 1),
                    "concat input extents"
                );
            }
        }
        if let Some((channels, plane, _)) = self.mean {
            let src = checked_size("mean source", &[channels, plane])?;
            check_shape!(
                bufs.len() >= 2 && src <= floats(0) && channels <= floats(1),
                "mean extents"
            );
        }
        Ok(())
    }

    /// Runs the step over `bufs`, one per binding, and the workspace's
    /// `scratch`.
    ///
    /// Every extent is checked against its buffer first
    /// ([`Step::check_extents`]). What the glue's raw copies rely on beyond
    /// that is the plan's: a step's input and output tensors are live at
    /// the same step, so the arena never places them in overlapping bytes
    /// (concat alone may find an input already in place, and copies with
    /// `ptr::copy`).
    pub(super) fn execute(
        &self,
        cpu: &OnednnCpu,
        stream: &Stream,
        bufs: &[Bound],
        scratch: Bound,
    ) -> Result<()> {
        let blk = cpu.conv_block();
        self.check_extents(bufs, scratch, blk)?;
        let threads = cpu.threads();
        if let Some(p) = self.pad {
            // SAFETY: the extents are checked above, and the plan keeps the
            // two tensors apart.
            unsafe { pad(p, threads, bufs) };
        } else if let Some(s) = self.shuffle {
            // SAFETY: as for the pad.
            unsafe { depth_to_space(s, blk, threads, bufs) };
        } else if let Some((batch, plane, channels)) = &self.concat {
            // SAFETY: as for the pad; an input may already be in place.
            unsafe { concat(*batch, *plane, channels, bufs) };
        } else if let Some((channels, plane, order)) = self.mean {
            // SAFETY: `channels * plane` floats of the source, checked in
            // `check_extents`, which the output does not overlap.
            let input =
                unsafe { std::slice::from_raw_parts(bufs[0].ptr.cast::<f32>(), channels * plane) };
            // SAFETY: `channels` floats of the output, checked likewise, and
            // nothing else refers to them during the call.
            let output =
                unsafe { std::slice::from_raw_parts_mut(bufs[1].ptr.cast::<f32>(), channels) };
            return crate::cpu::kernels::spatial_mean(input, output, plane, order, threads);
        } else {
            return self.run_primitive(cpu, stream, bufs, scratch, blk);
        }
        Ok(())
    }

    /// Run the primitive on the bound buffers: once, or for a pointwise
    /// tile once per window of the plane, staged through the scratch.
    fn run_primitive(
        &self,
        cpu: &OnednnCpu,
        stream: &Stream,
        bufs: &[Bound],
        scratch: Bound,
        blk: usize,
    ) -> Result<()> {
        let mut bound: Vec<*mut c_void> = bufs.iter().map(|b| b.ptr).collect();
        if let Some(tile) = self.tile {
            // SAFETY: both windows lie within the scratch (`check_extents`).
            bound[0] = unsafe { scratch.ptr.cast::<u8>().add(tile.source_offset).cast() };
            // SAFETY: as above.
            bound[1] = unsafe { scratch.ptr.cast::<u8>().add(tile.destination_offset).cast() };
        }
        let mut memories = Vec::with_capacity(bufs.len() + 1);
        let mut args = Vec::with_capacity(bufs.len() + 1);
        for ((arg, md, _), &data) in self.specs.iter().zip(&bound) {
            let mem = make_memory(cpu, md, data)?;
            args.push(dnnl_exec_arg_t {
                arg: *arg,
                memory: mem.0,
            });
            memories.push(mem);
        }
        if let Some(md) = &self.scratch {
            let mem = make_memory(cpu, md, scratch.ptr)?;
            args.push(dnnl_exec_arg_t {
                arg: DNNL_ARG_SCRATCHPAD,
                memory: mem.0,
            });
            memories.push(mem);
        }
        let execute = || {
            let (prim, n) = (self.primitive.0, args.len() as i32);
            // SAFETY: the primitive is alive (`self` holds it), each memory
            // is bound to a buffer that holds its descriptor's size
            // (`check_extents`), and waiting on the stream keeps them all
            // alive for the whole execute.
            let rc = unsafe { dnnl_primitive_execute(prim, stream.0, n, args.as_ptr()) };
            let result = Api::check(rc, "conv execute");
            // SAFETY: the workspace's live stream.
            let wait = Api::check(unsafe { dnnl_stream_wait(stream.0) }, "conv wait");
            result.and(wait)
        };
        let Some(tile) = self.tile else {
            return execute();
        };
        let ib = tile.shape[1].div_ceil(blk);
        let ob = tile.output_channels.div_ceil(blk);
        for b in 0..tile.shape[0] {
            for start in (0..tile.plane).step_by(tile.width) {
                let count = tile.width.min(tile.plane - start) * blk;
                for block in 0..ib {
                    // SAFETY: the source tensor and the scratch window both
                    // hold these floats (`check_extents`) and do not overlap;
                    // only the last tile needs its tail zeroed.
                    unsafe {
                        let dst = bound[0].cast::<f32>().add(block * tile.width * blk);
                        ptr::copy_nonoverlapping(
                            bufs[0]
                                .ptr
                                .cast::<f32>()
                                .add(((b * ib + block) * tile.plane + start) * blk),
                            dst,
                            count,
                        );
                        ptr::write_bytes(dst.add(count), 0, tile.width * blk - count);
                    }
                }
                execute()?;
                for block in 0..ob {
                    // SAFETY: as above, from the output window to the output
                    // tensor.
                    unsafe {
                        ptr::copy_nonoverlapping(
                            bound[1].cast::<f32>().add(block * tile.width * blk),
                            bufs[1]
                                .ptr
                                .cast::<f32>()
                                .add(((b * ob + block) * tile.plane + start) * blk),
                            count,
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

/// Copy each plane into the middle of a zeroed, larger one.
///
/// # Safety
///
/// `bufs` were checked against the step that holds `p`
/// ([`Step::check_extents`]), and its two tensors do not overlap.
unsafe fn pad(p: super::PadGlue, threads: usize, bufs: &[Bound]) {
    let src = bufs[0].ptr as usize;
    let dst = bufs[1].ptr as usize;
    let row = p.width * p.lanes;
    let padded_row = p.padded_width * p.lanes;
    let padded_plane = p.padded_height * padded_row;
    crate::cpu::kernels::for_ranges(threads, p.planes, 1, |lo, hi| {
        for plane in lo..hi {
            // SAFETY: both extents are checked and the tensors do not
            // overlap (this function's contract); each task writes whole
            // output planes no other task writes.
            unsafe {
                let out = (dst as *mut f32).add(plane * padded_plane);
                ptr::write_bytes(out, 0, padded_plane);
                for y in 0..p.height {
                    ptr::copy_nonoverlapping(
                        (src as *const f32).add((plane * p.height + y) * row),
                        out.add((y + p.top) * padded_row + p.left * p.lanes),
                        row,
                    );
                }
            }
        }
    });
}

/// Move each group of four channels to a 2x2 block of pixels: blocked
/// `[n, c * 4, h, w]` to `[n, c, h * 2, w * 2]`.
///
/// # Safety
///
/// As for [`pad`], with `s` the source shape.
unsafe fn depth_to_space(s: super::Shape, blk: usize, threads: usize, bufs: &[Bound]) {
    let (n, c, h, w) = (s[0], s[1] / 4, s[2], s[3]);
    let src = bufs[0].ptr as usize;
    let dst = bufs[1].ptr as usize;
    let dst_blocks = c.div_ceil(blk);
    crate::cpu::kernels::for_ranges(threads, n * dst_blocks * h, 16, |lo, hi| {
        for row in lo..hi {
            let b = row / (dst_blocks * h);
            let block = row / h % dst_blocks;
            let y = row % h;
            for x in 0..w {
                for lane in 0..blk.min(c - block * blk) {
                    let ch = block * blk + lane;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let ic = ch * 4 + dy * 2 + dx;
                            let si = ((b * s[1].div_ceil(blk) + ic / blk) * h * w + y * w + x)
                                * blk
                                + ic % blk;
                            let di = if c % blk == 0 {
                                ((b * dst_blocks + block) * 4 * h * w
                                    + (y * 2 + dy) * w * 2
                                    + x * 2
                                    + dx)
                                    * blk
                                    + lane
                            } else {
                                ((b * c + ch) * h * 2 + y * 2 + dy) * w * 2 + x * 2 + dx
                            };
                            // SAFETY: `si` and `di` lie within the
                            // extents checked in `check_extents`, the
                            // tensors do not overlap, and each task
                            // writes whole output rows no other task
                            // writes.
                            unsafe {
                                *((dst as *mut f32).add(di)) = *((src as *const f32).add(si));
                            }
                        }
                    }
                }
            }
        }
    });
}

/// Copy each input after the last along the channel axis.
///
/// # Safety
///
/// `bufs` were checked against the step that holds the concat
/// ([`Step::check_extents`]); an input may already sit at its place in
/// the output, and nothing else overlaps.
unsafe fn concat(batch: usize, plane: usize, channels: &[usize], bufs: &[Bound]) {
    let full = channels.iter().sum::<usize>() * plane;
    let mut offset = 0;
    for (i, &c) in channels.iter().enumerate() {
        let count = c * plane;
        for n in 0..batch {
            // SAFETY: both ranges lie within the extents checked in
            // `check_extents`; complete channel blocks are contiguous,
            // and a producer may already have written its slice in
            // place, which `ptr::copy` allows.
            unsafe {
                std::ptr::copy(
                    bufs[i + 1].ptr.cast::<f32>().add(n * count),
                    bufs[0].ptr.cast::<f32>().add(n * full + offset),
                    count,
                );
            }
        }
        offset += count;
    }
}

pub(super) fn make_memory(cpu: &OnednnCpu, md: &Md, data: *mut c_void) -> Result<Memory> {
    let mut m = Memory(ptr::null_mut());
    Api::check(
        // SAFETY: the descriptor and engine are live; `data` is a buffer of at
        // least the descriptor's size, which the caller checked
        // (`Step::check_extents`) and keeps alive while the memory is.
        unsafe { dnnl_memory_create(&mut m.0, md.raw(), cpu.conv_engine(), data) },
        "conv memory",
    )?;
    Ok(m)
}

pub(super) fn make_stream(cpu: &OnednnCpu) -> Result<Stream> {
    let mut s = Stream(ptr::null_mut());
    let api = cpu.conv_api();
    Api::check(
        // SAFETY: the engine and thread pool are live while `cpu` is, which
        // outlives every workspace it makes; oneDNN writes the new handle into
        // `s`, which owns it.
        unsafe { (api.stream_create)(&mut s.0, cpu.conv_engine(), cpu.threadpool()) },
        "conv stream",
    )?;
    Ok(s)
}

pub(super) fn clone_md(md: &Md) -> Result<Md> {
    Ok(md.clone())
}
