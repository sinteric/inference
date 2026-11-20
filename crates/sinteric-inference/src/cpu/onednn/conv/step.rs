//! One step of a plan: a primitive with its bound arguments, or a glue
//! pass (spatial mean, concat, channel shuffle, pointwise tile, zero pad)
//! run on the pool, and the memory and stream handles it executes with.

use super::{
    native_key, Binding, Md, MdOwner, Memory, NativeCache, Node, Pd, Primitive, Step, Stream,
};
use crate::cpu::onednn::api::Api;
use crate::cpu::onednn::OnednnCpu;
use crate::{check_shape, Result};
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
        let implementation = crate::cpu::onednn::api::cstr(info).to_owned();
        let md = unsafe { dnnl_primitive_desc_query_md(pd.0, dnnl_query_scratchpad_md, 0) };
        let scratch = if md.is_null() {
            None
        } else {
            let mut copy = ptr::null_mut();
            Api::check(
                unsafe { dnnl_memory_desc_clone(&mut copy, md) },
                "scratch descriptor",
            )?;
            Some(Md(Arc::new(MdOwner(copy))))
        };
        let scratch_bytes = scratch
            .as_ref()
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
    pub(super) fn execute(
        &self,
        cpu: &OnednnCpu,
        stream: &Stream,
        pointers: &[*mut c_void],
        scratch: *mut c_void,
    ) -> Result<()> {
        debug_assert_eq!(pointers.len(), self.specs.len());
        if let Some(p) = self.pad {
            let src = pointers[0] as usize;
            let dst = pointers[1] as usize;
            let row = p.width * p.lanes;
            let padded_row = p.padded_width * p.lanes;
            let padded_plane = p.padded_height * padded_row;
            crate::cpu::kernels::for_ranges(cpu.threads(), p.planes, 1, |lo, hi| {
                for plane in lo..hi {
                    // The checked plan owns disjoint input/output tensors;
                    // each task writes complete, disjoint output planes.
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
            return Ok(());
        }
        if let Some(s) = self.shuffle {
            let (n, c, h, w) = (s[0], s[1] / 4, s[2], s[3]);
            let src = pointers[0] as usize;
            let dst = pointers[1] as usize;
            let blk = cpu.conv_block();
            let dst_blocks = c.div_ceil(blk);
            crate::cpu::kernels::for_ranges(cpu.threads(), n * dst_blocks * h, 16, |lo, hi| {
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
                                    let si =
                                        ((b * s[1].div_ceil(blk) + ic / blk) * h * w + y * w + x)
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
                                    // The checked plan owns disjoint input/output tensors;
                                    // each task writes complete, disjoint output rows.
                                    unsafe {
                                        *((dst as *mut f32).add(di)) =
                                            *((src as *const f32).add(si));
                                    }
                                }
                            }
                        }
                    }
                }
            });
            return Ok(());
        }
        if let Some((batch, plane, channels)) = &self.concat {
            let full = channels.iter().sum::<usize>() * plane;
            let mut offset = 0;
            for (i, &c) in channels.iter().enumerate() {
                let count = c * plane;
                for n in 0..*batch {
                    // Complete channel blocks are contiguous. Checked producer
                    // views can already occupy their destination slices.
                    unsafe {
                        std::ptr::copy(
                            pointers[i + 1].cast::<f32>().add(n * count),
                            pointers[0].cast::<f32>().add(n * full + offset),
                            count,
                        );
                    }
                }
                offset += count;
            }
            return Ok(());
        }
        if let Some((channels, plane, order)) = self.mean {
            // Planned tensors have checked extents, plain NCHW storage, and
            // disjoint lifetimes. No per-element validation enters the kernel.
            let input =
                unsafe { std::slice::from_raw_parts(pointers[0].cast::<f32>(), channels * plane) };
            let output =
                unsafe { std::slice::from_raw_parts_mut(pointers[1].cast::<f32>(), channels) };
            return crate::cpu::kernels::spatial_mean(input, output, plane, order, cpu.threads());
        }
        let mut bound = pointers.to_vec();
        if let Some(tile) = self.tile {
            bound[0] = unsafe { scratch.cast::<u8>().add(tile.source_offset).cast() };
            bound[1] = unsafe { scratch.cast::<u8>().add(tile.destination_offset).cast() };
        }
        let mut memories = Vec::with_capacity(pointers.len() + 1);
        let mut args = Vec::with_capacity(pointers.len() + 1);
        for ((arg, md, _), &data) in self.specs.iter().zip(&bound) {
            let mem = make_memory(cpu, md, data)?;
            args.push(dnnl_exec_arg_t {
                arg: *arg,
                memory: mem.0,
            });
            memories.push(mem);
        }
        if let Some(md) = &self.scratch {
            let mem = make_memory(cpu, md, scratch)?;
            args.push(dnnl_exec_arg_t {
                arg: DNNL_ARG_SCRATCHPAD,
                memory: mem.0,
            });
            memories.push(mem);
        }
        let execute = || {
            let result = Api::check(
                unsafe {
                    dnnl_primitive_execute(
                        self.primitive.0,
                        stream.0,
                        args.len() as i32,
                        args.as_ptr(),
                    )
                },
                "conv execute",
            );
            let wait = Api::check(unsafe { dnnl_stream_wait(stream.0) }, "conv wait");
            result.and(wait)
        };
        if let Some(tile) = self.tile {
            let blk = cpu.conv_block();
            let ib = tile.shape[1].div_ceil(blk);
            let ob = tile.output_channels.div_ceil(blk);
            for b in 0..tile.shape[0] {
                for start in (0..tile.plane).step_by(tile.width) {
                    let count = tile.width.min(tile.plane - start) * blk;
                    for block in 0..ib {
                        // Blocked spatial tiles are contiguous in each channel
                        // block. Extents and scratch offsets are checked at plan
                        // creation; only the last tile needs zero padding.
                        unsafe {
                            let dst = bound[0].cast::<f32>().add(block * tile.width * blk);
                            ptr::copy_nonoverlapping(
                                pointers[0]
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
                        unsafe {
                            ptr::copy_nonoverlapping(
                                bound[1].cast::<f32>().add(block * tile.width * blk),
                                pointers[1]
                                    .cast::<f32>()
                                    .add(((b * ob + block) * tile.plane + start) * blk),
                                count,
                            );
                        }
                    }
                }
            }
            Ok(())
        } else {
            execute()
        }
    }
}

pub(super) fn make_memory(cpu: &OnednnCpu, md: &Md, data: *mut c_void) -> Result<Memory> {
    let mut m = Memory(ptr::null_mut());
    Api::check(
        unsafe { dnnl_memory_create(&mut m.0, md.raw(), cpu.conv_engine(), data) },
        "conv memory",
    )?;
    Ok(m)
}

pub(super) fn make_stream(cpu: &OnednnCpu) -> Result<Stream> {
    let mut s = Stream(ptr::null_mut());
    let api = cpu.conv_api();
    Api::check(
        unsafe { (api.stream_create)(&mut s.0, cpu.conv_engine(), cpu.threadpool()) },
        "conv stream",
    )?;
    Ok(s)
}

pub(super) fn clone_md(md: &Md) -> Result<Md> {
    Ok(md.clone())
}
