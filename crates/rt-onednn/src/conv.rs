//! FP32 convolution graphs with fixed packed weights, a shape LRU and a checked
//! liveness arena. Graphs are supplied by model code; no ONNX interpretation.
use crate::api::Api;
use crate::cpu::Aligned;
use crate::{Hardware, OnednnCpu};
use onednn_sys::*;
use rt_core::{check_shape, checked_size, Error, Result};
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Condvar, Mutex, Weak};

pub use rt_kernels::SpatialMean;
use sha2::{Digest, Sha256};

pub type Shape = [usize; 4];

fn dimensions(d: &[usize]) -> Result<Vec<i64>> {
    check_shape!(
        !d.is_empty() && d.len() <= 12 && !d.contains(&0),
        "invalid dimensions {d:?}"
    );
    let bytes = checked_size("convolution tensor", d)?
        .checked_mul(4)
        .ok_or(Error::SizeOverflow("tensor bytes"))?;
    check_shape!(
        bytes <= isize::MAX as usize,
        "tensor extent exceeds address space"
    );
    d.iter()
        .map(|&x| i64::try_from(x).map_err(|_| Error::SizeOverflow("oneDNN dimension")))
        .collect()
}

macro_rules! handle {
    ($name:ident, $ty:ty, $destroy:path) => {
        struct $name($ty);
        impl Drop for $name {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    unsafe {
                        $destroy(self.0);
                    }
                }
            }
        }
        // Descriptors and primitives are immutable after construction. Execute
        // creates separate memory handles/stream/scratch for every invocation.
        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}
    };
}
handle!(MdOwner, dnnl_memory_desc_t, dnnl_memory_desc_destroy);
#[derive(Clone)]
struct Md(Arc<MdOwner>);
handle!(Pd, dnnl_primitive_desc_t, dnnl_primitive_desc_destroy);
handle!(Primitive, dnnl_primitive_t, dnnl_primitive_destroy);
handle!(Attr, dnnl_primitive_attr_t, dnnl_primitive_attr_destroy);
handle!(Posts, dnnl_post_ops_t, dnnl_post_ops_destroy);
handle!(Memory, dnnl_memory_t, dnnl_memory_destroy);
handle!(Stream, dnnl_stream_t, dnnl_stream_destroy);

impl Md {
    fn tagged(dims: &[usize], tag: i32) -> Result<Self> {
        let d = dimensions(dims)?;
        let mut md = ptr::null_mut();
        Api::check(
            unsafe {
                dnnl_memory_desc_create_with_tag(&mut md, d.len() as i32, d.as_ptr(), dnnl_f32, tag)
            },
            "conv memory descriptor",
        )?;
        Ok(Self(Arc::new(MdOwner(md))))
    }
    fn raw(&self) -> dnnl_memory_desc_t {
        self.0 .0
    }
    fn bytes(&self) -> Result<usize> {
        let n = unsafe { dnnl_memory_desc_get_size(self.raw()) };
        check_shape!(
            n > 0 && n <= isize::MAX as usize && n % 4 == 0,
            "invalid memory descriptor extent {n}"
        );
        Ok(n)
    }
}

/// Immutable, spatial-shape-independent convolution weights. Source layout is
/// OIHW (GOIHW for groups); deconvolution also uses logical OIHW dimensions.
pub struct ConvWeights {
    md: Md,
    data: Aligned,
    bias: Option<Vec<f32>>,
    output: usize,
    input: usize,
    kernel: [usize; 2],
    transpose: bool,
    // Own the engine until the last weight/primitive is released.
    cpu: Arc<OnednnCpu>,
    tag: i32,
}
impl ConvWeights {
    pub fn pack(
        cpu: Arc<OnednnCpu>,
        output: usize,
        input: usize,
        kernel: [usize; 2],
        groups: usize,
        transpose: bool,
        data: &[f32],
        bias: Option<&[f32]>,
    ) -> Result<Arc<Self>> {
        check_shape!(
            groups > 0 && input > 0 && output > 0 && input % groups == 0 && output % groups == 0,
            "invalid conv channels/groups"
        );
        if groups != 1 && (groups != input || groups != output) {
            return Err(Error::Unsupported {
                what: "grouped convolution",
                reason: "OCR supports groups=1 or depthwise groups=C".into(),
            });
        }
        let dims = if groups == 1 {
            vec![output, input, kernel[0], kernel[1]]
        } else {
            vec![
                groups,
                output / groups,
                input / groups,
                kernel[0],
                kernel[1],
            ]
        };
        dimensions(&dims)?;
        check_shape!(
            data.len() == checked_size("conv weights", &dims)?,
            "wrong conv weight length"
        );
        check_shape!(
            bias.is_none_or(|b| b.len() == output),
            "wrong conv bias length"
        );
        check_shape!(
            !transpose || groups == 1,
            "grouped deconvolution is not supported"
        );
        let wide = cpu.conv_block() == 16;
        let tag = if transpose {
            // Deconvolution lowers to convolution backward-data. Its JIT
            // expects the opposite inner block order to forward convolution;
            // oneDNN names that layout for 16-channel blocks only.
            if !wide {
                return Err(Error::Unsupported {
                    what: "deconvolution",
                    reason: "oneDNN has no 8-channel deconvolution weight layout; needs AVX-512"
                        .into(),
                });
            }
            dnnl_IOhw16i16o
        } else if groups == 1 && input < cpu.conv_block() {
            if wide {
                dnnl_Ohwi16o
            } else {
                dnnl_Ohwi8o
            }
        } else if groups == 1 {
            if wide {
                dnnl_OIhw16i16o
            } else {
                dnnl_OIhw8i8o
            }
        } else if groups == input && groups == output {
            if wide {
                dnnl_Goihw16g
            } else {
                dnnl_Goihw8g
            }
        } else if wide {
            dnnl_gOIhw16i16o
        } else {
            dnnl_gOIhw8i8o
        };
        let raw = Md::tagged(&dims, if groups == 1 { dnnl_oihw } else { dnnl_goihw })?;
        let md = Md::tagged(&dims, tag)?;
        let packed = Aligned::new(md.bytes()?)?;
        let mut pd = Pd(ptr::null_mut());
        let engine = cpu.conv_engine();
        Api::check(
            unsafe {
                dnnl_reorder_primitive_desc_create(
                    &mut pd.0,
                    raw.raw(),
                    engine,
                    md.raw(),
                    engine,
                    ptr::null(),
                )
            },
            "weight reorder descriptor",
        )?;
        let step = Step::new(
            pd,
            vec![
                (DNNL_ARG_FROM, raw, Binding::External),
                (DNNL_ARG_TO, Md::tagged(&dims, tag)?, Binding::External),
            ],
        )?;
        let _compute = rt_threads::compute_lock();
        cpu.conv_api().apply_threads(cpu.threads() as i32)?;
        let stream = make_stream(&cpu)?;
        let scratch = Aligned::new(step.scratch_bytes)?;
        step.execute(
            &cpu,
            &stream,
            &[data.as_ptr() as *mut c_void, packed.ptr()],
            scratch.ptr(),
        )?;
        let bias = bias
            .map(|b| {
                let mut v = rt_core::try_vec(b.len())?;
                v.copy_from_slice(b);
                Ok::<_, Error>(v)
            })
            .transpose()?;
        cpu.conv_weight_bytes
            .fetch_add(packed.bytes, std::sync::atomic::Ordering::Relaxed);
        Ok(Arc::new(Self {
            md,
            data: packed,
            bias,
            output,
            input,
            kernel,
            transpose,
            cpu,
            tag,
        }))
    }
    pub fn format_tag(&self) -> i32 {
        self.tag
    }
    /// Too few input channels for a block: the convolution reads plain NCHW.
    fn reads_plain(&self) -> bool {
        self.tag == dnnl_Ohwi16o || self.tag == dnnl_Ohwi8o
    }
    /// Dense blocked weights, as a pointwise tile requires.
    fn dense_blocked(&self) -> bool {
        self.tag == dnnl_OIhw16i16o || self.tag == dnnl_OIhw8i8o
    }
    pub fn packed_bytes(&self) -> usize {
        self.data.bytes
    }
}

impl Drop for ConvWeights {
    fn drop(&mut self) {
        self.cpu
            .conv_weight_bytes
            .fetch_sub(self.data.bytes, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Spatial convolution/pool parameters, using ONNX dilation (1 = no dilation).
#[derive(Clone, Debug)]
pub struct Spatial {
    pub strides: [usize; 2],
    pub dilations: [usize; 2],
    pub pads_begin: [usize; 2],
    pub pads_end: [usize; 2],
}
impl Default for Spatial {
    fn default() -> Self {
        Self {
            strides: [1; 2],
            dilations: [1; 2],
            pads_begin: [0; 2],
            pads_end: [0; 2],
        }
    }
}
impl Spatial {
    fn arrays(&self) -> Result<[[i64; 2]; 4]> {
        check_shape!(
            !self.strides.contains(&0) && !self.dilations.contains(&0),
            "zero stride/dilation"
        );
        let mut a = [[0; 2]; 4];
        for (i, v) in [
            self.strides,
            self.dilations.map(|x| x - 1),
            self.pads_begin,
            self.pads_end,
        ]
        .iter()
        .enumerate()
        {
            for j in 0..2 {
                a[i][j] =
                    i64::try_from(v[j]).map_err(|_| Error::SizeOverflow("spatial attributes"))?;
            }
        }
        Ok(a)
    }
    fn output(
        &self,
        input: Shape,
        kernel: [usize; 2],
        channels: usize,
        transpose: bool,
    ) -> Result<Shape> {
        self.arrays()?;
        dimensions(&kernel)?;
        let mut out = [input[0], channels, 0, 0];
        for j in 0..2 {
            let effective = (kernel[j] - 1)
                .checked_mul(self.dilations[j])
                .and_then(|v| v.checked_add(1))
                .ok_or(Error::SizeOverflow("dilated kernel"))?;
            let pad = self.pads_begin[j]
                .checked_add(self.pads_end[j])
                .ok_or(Error::SizeOverflow("padding"))?;
            let v = if transpose {
                (input[j + 2] - 1)
                    .checked_mul(self.strides[j])
                    .and_then(|v| v.checked_add(effective))
                    .and_then(|v| v.checked_sub(pad))
                    .ok_or_else(|| Error::Shape("deconvolution output extent".into()))?
            } else {
                let padded = input[j + 2]
                    .checked_add(pad)
                    .ok_or(Error::SizeOverflow("padded input"))?;
                check_shape!(padded >= effective, "kernel exceeds padded input");
                (padded - effective) / self.strides[j] + 1
            };
            out[j + 2] = v;
        }
        dimensions(&out)?;
        Ok(out)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Relu,
    HardSwish,
    HardSigmoid { alpha: f32, beta: f32 },
    Logistic,
    Swish,
    GeluErf,
}
impl Activation {
    fn params(self) -> Result<(i32, f32, f32)> {
        Ok(match self {
            Self::Relu => (dnnl_eltwise_relu, 0., 0.),
            Self::HardSwish => (dnnl_eltwise_hardswish, 1. / 6., 0.5),
            Self::HardSigmoid { alpha, beta } => {
                check_shape!(
                    alpha.is_finite() && beta.is_finite(),
                    "nonfinite activation parameter"
                );
                (dnnl_eltwise_hardsigmoid, alpha, beta)
            }
            Self::Logistic => (dnnl_eltwise_logistic, 0., 0.),
            Self::Swish => (dnnl_eltwise_swish, 1., 0.),
            Self::GeluErf => (dnnl_eltwise_gelu_erf, 0., 0.),
        })
    }
}
#[derive(Clone, Copy, Debug)]
pub enum Binary {
    Add,
    Mul,
}
impl Binary {
    fn alg(self) -> i32 {
        match self {
            Self::Add => dnnl_binary_add,
            Self::Mul => dnnl_binary_mul,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub enum Pool {
    Max,
    AverageIncludePadding,
    AverageExcludePadding,
}
impl Pool {
    fn alg(self) -> i32 {
        match self {
            Self::Max => dnnl_pooling_max,
            Self::AverageIncludePadding => dnnl_pooling_avg_include_padding,
            Self::AverageExcludePadding => dnnl_pooling_avg_exclude_padding,
        }
    }
}
#[derive(Clone)]
pub enum PostOp {
    Eltwise(Activation),
    /// Broadcast scalar arithmetic, kept as separate operations (no affine FMA).
    Scalar {
        kind: Binary,
        value: Arc<Vec<f32>>,
    },
    /// Add FP32 channel bias after accumulation, before subsequent post-ops.
    ChannelBias(Arc<Vec<f32>>),
    Binary {
        kind: Binary,
        tensor: usize,
    },
}
#[derive(Clone)]
pub enum Op {
    Conv {
        weights: Arc<ConvWeights>,
        spatial: Spatial,
    },
    /// Rearrange [N,4*C,H,W] into [N,C,2*H,2*W] (ONNX CRD order).
    DepthToSpace2x,
    Pool {
        kind: Pool,
        kernel: [usize; 2],
        spatial: Spatial,
    },
    /// oneDNN nearest uses half-pixel nearest. For integral upsampling this
    /// equals ONNX asymmetric/floor. Non-integral nearest is rejected.
    Resize {
        height: usize,
        width: usize,
        linear: bool,
    },
    /// Integral spatial upsampling, resolved when a shape plan is built.
    UpsampleNearest {
        scales: [usize; 2],
    },
    /// Spatial FP32 mean with an explicitly fixed reduction order.
    SpatialMean(SpatialMean),
    ReduceMean {
        axes: [bool; 4],
    },
    Concat {
        axis: usize,
    },
    /// Channel concat whose inputs all have complete channel blocks
    /// ([`OnednnCpu::conv_block`]).
    ConcatAlignedChannels,
    Eltwise(Activation),
    Binary(Binary),
    Prelu {
        slopes: Vec<f32>,
    },
    Reorder {
        blocked: bool,
    },
    /// Zero-pad the spatial axes; the output keeps the input's layout.
    /// Lowering inserts one before a convolution whose padding oneDNN would
    /// take only on its reference kernel, and runs that convolution unpadded.
    ZeroPad {
        begin: [usize; 2],
        end: [usize; 2],
    },
}
/// Activation 0 is the NCHW network input. Node i produces activation i+1.
#[derive(Clone)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<usize>,
    pub post_ops: Vec<PostOp>,
}
impl Node {
    pub fn new(op: Op, inputs: impl Into<Vec<usize>>) -> Self {
        Self {
            op,
            inputs: inputs.into(),
            post_ops: vec![],
        }
    }
}

struct Tensor {
    shape: Shape,
    md: Md,
}
#[derive(Clone)]
enum Binding {
    Tensor(usize),
    Weights(Arc<ConvWeights>),
    Bias(Arc<ConvWeights>),
    Slopes(Arc<Vec<f32>>),
    External,
}
#[derive(Clone, Copy)]
struct PointwiseTile {
    shape: Shape,
    output_channels: usize,
    plane: usize,
    width: usize,
    source_offset: usize,
    destination_offset: usize,
}
struct Step {
    primitive: Arc<Primitive>,
    specs: Vec<(i32, Md, Binding)>,
    scratch: Option<Md>,
    scratch_bytes: usize,
    implementation: String,
    mean: Option<(usize, usize, SpatialMean)>,
    concat: Option<(usize, usize, Vec<usize>)>,
    shuffle: Option<Shape>,
    tile: Option<PointwiseTile>,
    pad: Option<PadGlue>,
}
/// A zero pad over `planes` contiguous planes of `height x width` pixels of
/// `lanes` floats each (the channel block, or 1 for plain NCHW).
#[derive(Clone, Copy)]
struct PadGlue {
    planes: usize,
    height: usize,
    width: usize,
    lanes: usize,
    top: usize,
    left: usize,
    padded_height: usize,
    padded_width: usize,
}
impl Step {
    fn new(pd: Pd, specs: Vec<(i32, Md, Binding)>) -> Result<Self> {
        Self::new_cached(pd, specs, None)
    }
    fn new_cached(
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
        let implementation = crate::api::cstr(info).to_owned();
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
    fn execute(
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
            rt_kernels::for_ranges(cpu.threads(), p.planes, 1, |lo, hi| {
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
            rt_kernels::for_ranges(cpu.threads(), n * dst_blocks * h, 16, |lo, hi| {
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
            return rt_kernels::spatial_mean(input, output, plane, order, cpu.threads());
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
fn make_memory(cpu: &OnednnCpu, md: &Md, data: *mut c_void) -> Result<Memory> {
    let mut m = Memory(ptr::null_mut());
    Api::check(
        unsafe { dnnl_memory_create(&mut m.0, md.raw(), cpu.conv_engine(), data) },
        "conv memory",
    )?;
    Ok(m)
}
fn make_stream(cpu: &OnednnCpu) -> Result<Stream> {
    let mut s = Stream(ptr::null_mut());
    let api = cpu.conv_api();
    Api::check(
        unsafe { (api.stream_create)(&mut s.0, cpu.conv_engine(), cpu.threadpool()) },
        "conv stream",
    )?;
    Ok(s)
}
fn clone_md(md: &Md) -> Result<Md> {
    Ok(md.clone())
}
fn broadcast(a: Shape, b: Shape) -> Result<()> {
    check_shape!(
        (0..4).all(|i| b[i] == 1 || b[i] == a[i]),
        "broadcast {b:?} into {a:?}"
    );
    Ok(())
}

/// `block` is the backend's channel block ([`OnednnCpu::conv_block`]).
fn infer_shape(node: &Node, shapes: &[Shape], block: usize) -> Result<Shape> {
    let s = shapes[node.inputs[0]];
    let unary = !matches!(
        node.op,
        Op::Concat { .. } | Op::ConcatAlignedChannels | Op::Binary(_)
    );
    check_shape!(
        !unary || node.inputs.len() == 1,
        "unary operator has multiple inputs"
    );
    let output = match &node.op {
        Op::Conv {
            weights: w,
            spatial,
        } => {
            check_shape!(s[1] == w.input, "conv input channels");
            spatial.output(s, w.kernel, w.output, w.transpose)?
        }
        Op::Pool {
            kernel, spatial, ..
        } => spatial.output(s, *kernel, s[1], false)?,
        Op::DepthToSpace2x => {
            check_shape!(s[1] % 4 == 0, "depth-to-space requires 4*C channels");
            [
                s[0],
                s[1] / 4,
                s[2].checked_mul(2)
                    .ok_or(Error::SizeOverflow("depth-to-space height"))?,
                s[3].checked_mul(2)
                    .ok_or(Error::SizeOverflow("depth-to-space width"))?,
            ]
        }
        Op::Resize {
            height,
            width,
            linear,
        } => {
            dimensions(&[*height, *width])?;
            check_shape!(
                *linear || (*height % s[2] == 0 && *width % s[3] == 0),
                "nearest resize requires integral upsampling"
            );
            [s[0], s[1], *height, *width]
        }
        Op::UpsampleNearest { scales } => {
            check_shape!(!scales.contains(&0), "zero resize scale");
            [
                s[0],
                s[1],
                s[2].checked_mul(scales[0])
                    .ok_or(Error::SizeOverflow("resize height"))?,
                s[3].checked_mul(scales[1])
                    .ok_or(Error::SizeOverflow("resize width"))?,
            ]
        }
        Op::SpatialMean(_) => [s[0], s[1], 1, 1],
        Op::ZeroPad { begin, end } => {
            let mut o = s;
            for j in 0..2 {
                o[j + 2] = s[j + 2]
                    .checked_add(begin[j])
                    .and_then(|v| v.checked_add(end[j]))
                    .ok_or(Error::SizeOverflow("zero pad"))?;
            }
            o
        }
        Op::ReduceMean { axes } => std::array::from_fn(|i| if axes[i] { 1 } else { s[i] }),
        Op::ConcatAlignedChannels | Op::Concat { .. } => {
            let axis = if let Op::Concat { axis } = node.op {
                axis
            } else {
                1
            };
            if matches!(node.op, Op::ConcatAlignedChannels) {
                check_shape!(
                    node.inputs.iter().all(|&id| shapes[id][1] % block == 0),
                    "channel concat needs complete blocks"
                );
            }
            check_shape!(axis < 4, "concat axis");
            let mut o = s;
            o[axis] = 0;
            for &id in &node.inputs {
                let t = shapes[id];
                check_shape!((0..4).all(|i| i == axis || t[i] == s[i]), "concat shapes");
                o[axis] = o[axis]
                    .checked_add(t[axis])
                    .ok_or(Error::SizeOverflow("concat"))?;
            }
            o
        }
        Op::Binary(_) => {
            check_shape!(node.inputs.len() == 2, "binary needs two inputs");
            broadcast(s, shapes[node.inputs[1]])?;
            s
        }
        _ => s,
    };
    dimensions(&output)?;
    Ok(output)
}
fn output_blocked(node: &Node, output: Shape, block: usize) -> bool {
    match &node.op {
        Op::Reorder { blocked } => *blocked,
        Op::DepthToSpace2x => output[1] % block == 0,
        Op::Concat { .. } | Op::ReduceMean { .. } | Op::SpatialMean(_) | Op::Prelu { .. } => false,
        // JIT eltwise requires no padded channels for nonzero-preserving ops.
        Op::Eltwise(Activation::HardSigmoid { .. } | Activation::Logistic)
            if output[1] % block != 0 =>
        {
            false
        }
        _ => true,
    }
}

struct Plan {
    shape: Shape,
    output_shape: Shape,
    steps: Vec<Arc<Step>>,
    offsets: Vec<usize>,
    /// Merged byte ranges of blocked tensors with padded channel lanes; only
    /// these need zeroing before a run (dense tensors are fully overwritten).
    zero: Vec<(usize, usize)>,
    arena_bytes: usize,
    scratch_bytes: usize,
    /// Heap growth while the plan was built (primitives and descriptors not
    /// shared with an earlier plan), an estimate reported in stats.
    bytes: usize,
}
/// Inspectable cache/arena information, including primitive implementation names.
#[derive(Clone, Debug)]
pub struct PlanInfo {
    pub shape: Shape,
    pub output_shape: Shape,
    pub arena_bytes: usize,
    pub scratch_bytes: usize,
    pub implementations: Vec<String>,
}
impl Plan {
    fn info(&self) -> PlanInfo {
        PlanInfo {
            shape: self.shape,
            output_shape: self.output_shape,
            arena_bytes: self.arena_bytes,
            scratch_bytes: self.scratch_bytes,
            implementations: self
                .steps
                .iter()
                .map(|s| s.implementation.clone())
                .collect(),
        }
    }
}

/// One immutable graph with a bounded plan/primitive LRU. All convolution
/// segments on a backend share a grow-only arena, scratchpad and stream
/// ([`Workspaces`]). Taking the workspace serializes calls and bounds
/// retained activation memory, independent of the number of cached plans.
pub(crate) struct Workspace {
    arena: Aligned,
    scratch: Aligned,
    stream: Stream,
}
impl Workspace {
    pub(crate) fn bytes(&self) -> (usize, usize) {
        (self.arena.bytes, self.scratch.bytes)
    }
}

/// A backend's convolution workspaces: normally one, which callers wait for.
/// Inside a pool job (independent inputs running one per worker) a busy
/// workspace gets a sibling instead, so up to one per worker is retained.
#[derive(Default)]
pub(crate) struct Workspaces {
    pub(crate) state: Mutex<WorkspaceSet>,
    cv: Condvar,
}
#[derive(Default)]
pub(crate) struct WorkspaceSet {
    pub(crate) idle: Vec<Workspace>,
    created: usize,
}
struct WorkspaceLease<'a> {
    owner: &'a Workspaces,
    workspace: Option<Workspace>,
}
impl Workspaces {
    /// A lease on an idle workspace, or an empty lease when the caller
    /// should create one.
    fn take(&self) -> Result<WorkspaceLease<'_>> {
        let mut set = self
            .state
            .lock()
            .map_err(|_| Error::Poisoned("conv workspace"))?;
        let parallel = rt_threads::in_parallel();
        loop {
            // Outside a job take the largest workspace, inside one the
            // smallest: only one grows to the largest input (the detector's
            // page) instead of each in turn.
            let bytes = |w: &Workspace| w.arena.bytes + w.scratch.bytes;
            let pick = if parallel {
                set.idle.iter().enumerate().min_by_key(|(_, w)| bytes(w))
            } else {
                set.idle.iter().enumerate().max_by_key(|(_, w)| bytes(w))
            }
            .map(|(i, _)| i);
            if let Some(i) = pick {
                return Ok(WorkspaceLease {
                    owner: self,
                    workspace: Some(set.idle.swap_remove(i)),
                });
            }
            if set.created == 0 || parallel {
                set.created += 1;
                return Ok(WorkspaceLease {
                    owner: self,
                    workspace: None,
                });
            }
            set = self
                .cv
                .wait(set)
                .map_err(|_| Error::Poisoned("conv workspace"))?;
        }
    }
}
impl Drop for WorkspaceLease<'_> {
    fn drop(&mut self) {
        let mut set = self.owner.state.lock().unwrap_or_else(|e| e.into_inner());
        match self.workspace.take() {
            Some(workspace) => set.idle.push(workspace),
            // Creation failed: let the next caller try.
            None => set.created -= 1,
        }
        drop(set);
        self.owner.cv.notify_one();
    }
}

type StepKey = (usize, Vec<usize>, Vec<Shape>);
type StepLru = VecDeque<(StepKey, Arc<Step>)>;
type NativeCache = HashMap<[u8; 32], Weak<Primitive>>;
#[derive(Default)]
struct Caches {
    descriptors: Descriptors,
    steps: Vec<StepLru>,
    natives: NativeCache,
}

#[derive(Default)]
struct Descriptors(HashMap<(Vec<usize>, i32), Weak<MdOwner>>);
impl Descriptors {
    fn tagged(&mut self, dims: &[usize], tag: i32) -> Result<Md> {
        let key = (dims.to_vec(), tag);
        if let Some(md) = self.0.get(&key).and_then(Weak::upgrade) {
            return Ok(Md(md));
        }
        // Weak entries do not retain descriptor storage. Bound dead keys too.
        if self.0.len() >= 16384 {
            self.0.retain(|_, p| p.strong_count() > 0);
        }
        let md = Md::tagged(dims, tag)?;
        if self.0.len() < 16384 {
            self.0.insert(key, Arc::downgrade(&md.0));
        }
        Ok(md)
    }
}

// Primitive code depends on descriptors and attributes, not bound weight,
// scalar or bias values. Share it across repeated backbone blocks as well
// as widths. Full descriptor blobs preserve strides, padding and block order.
fn native_key(node: &Node, specs: &[(i32, Md, Binding)]) -> Result<[u8; 32]> {
    let mut h = Sha256::new();
    let mut int = |v: i64| h.update(v.to_le_bytes());
    match &node.op {
        Op::Conv { weights, spatial } => {
            int(if weights.transpose { 1 } else { 2 });
            for a in spatial.arrays()? {
                for v in a {
                    int(v);
                }
            }
        }
        Op::Pool {
            kind,
            kernel,
            spatial,
        } => {
            int(3);
            int(kind.alg() as i64);
            for v in dimensions(kernel)? {
                int(v);
            }
            for a in spatial.arrays()? {
                for v in a {
                    int(v);
                }
            }
        }
        Op::Resize { linear, .. } => {
            int(4);
            int(i64::from(*linear));
        }
        Op::ReduceMean { .. } => int(5),
        Op::Concat { axis } => {
            int(6);
            int(*axis as i64);
        }
        Op::Eltwise(a) => {
            int(7);
            let (alg, alpha, beta) = a.params()?;
            int(alg as i64);
            int(alpha.to_bits() as i64);
            int(beta.to_bits() as i64);
        }
        Op::Binary(b) => {
            int(8);
            int(b.alg() as i64);
        }
        Op::Prelu { .. } => int(9),
        Op::Reorder { .. } => int(10),
        _ => return Err(Error::Backend("glue in native primitive cache".into())),
    }
    int(node.post_ops.len() as i64);
    for post in &node.post_ops {
        match post {
            PostOp::Eltwise(a) => {
                int(1);
                let (alg, alpha, beta) = a.params()?;
                int(alg as i64);
                int(alpha.to_bits() as i64);
                int(beta.to_bits() as i64);
            }
            PostOp::ChannelBias(_) => {
                int(2);
                int(dnnl_binary_add as i64);
            }
            PostOp::Scalar { kind, .. } | PostOp::Binary { kind, .. } => {
                int(2);
                int(kind.alg() as i64);
            }
        }
    }
    for (arg, md, _) in specs {
        h.update(arg.to_le_bytes());
        let mut size = 0;
        Api::check(
            unsafe { dnnl_memory_desc_get_blob(ptr::null_mut(), &mut size, md.raw()) },
            "descriptor blob size",
        )?;
        let mut blob = rt_core::try_vec(size)?;
        Api::check(
            unsafe { dnnl_memory_desc_get_blob(blob.as_mut_ptr(), &mut size, md.raw()) },
            "descriptor blob",
        )?;
        h.update(size.to_le_bytes());
        h.update(blob);
    }
    Ok(h.finalize().into())
}

pub struct ConvNet {
    pointwise_tiles: bool,
    primitive_cache: Mutex<Caches>,
    cache: Mutex<VecDeque<Arc<Plan>>>,
    nodes: Vec<Node>,
    capacity: usize,
    cpu: Arc<OnednnCpu>,
}
impl ConvNet {
    pub fn new(cpu: Arc<OnednnCpu>, nodes: Vec<Node>, capacity: usize) -> Result<Self> {
        check_shape!(
            !nodes.is_empty() && capacity > 0,
            "empty ConvNet or zero plan capacity"
        );
        for (i, n) in nodes.iter().enumerate() {
            check_shape!(
                !n.inputs.is_empty() && n.inputs.iter().all(|&x| x <= i),
                "invalid/non-topological inputs at node {i}"
            );
            for p in &n.post_ops {
                if let PostOp::Binary { tensor, .. } = p {
                    check_shape!(*tensor <= i, "non-topological binary post-op");
                }
            }
            check_shape!(n.post_ops.len() <= 32, "too many post-ops");
            if let Op::Conv { weights, .. } = &n.op {
                check_shape!(
                    Arc::ptr_eq(&cpu, &weights.cpu),
                    "weights belong to another backend"
                );
            }
        }
        Ok(Self {
            cpu,
            nodes,
            capacity,
            pointwise_tiles: false,
            cache: Mutex::new(VecDeque::new()),
            primitive_cache: Mutex::new(Caches::default()),
        })
    }
    /// Reuse 1x1 convolution code across spatial sizes. Only independent
    /// pointwise operations with scalar/channel post-ops are eligible; spatial
    /// convolutions, reductions and attention keep their exact shapes.
    #[doc(hidden)]
    pub fn with_pointwise_tiles(mut self) -> Self {
        self.pointwise_tiles = true;
        self
    }
    /// Diagnostic execution of a compiled graph prefix (1-based node count).
    #[doc(hidden)]
    #[cfg(feature = "debug-dumps")]
    pub fn prefix(&self, count: usize) -> Result<Self> {
        check_shape!(
            count > 0 && count <= self.nodes.len(),
            "invalid graph prefix"
        );
        let mut net = Self::new(self.cpu.clone(), self.nodes[..count].to_vec(), 1)?;
        net.pointwise_tiles = self.pointwise_tiles;
        Ok(net)
    }
    pub fn cached_shapes(&self) -> Result<Vec<Shape>> {
        Ok(self
            .cache
            .lock()
            .map_err(|_| Error::Poisoned("conv plan cache"))?
            .iter()
            .map(|p| p.shape)
            .collect())
    }
    pub fn warmup(&self, shape: Shape) -> Result<PlanInfo> {
        Ok(self.plan(shape)?.info())
    }
    pub fn run(&self, shape: Shape, input: &[f32]) -> Result<(Shape, Vec<f32>)> {
        dimensions(&shape)?;
        check_shape!(
            input.len() == checked_size("conv input", &shape)?,
            "input buffer/shape mismatch"
        );
        let plan = self.plan(shape)?;
        let mut lease = self.cpu.conv_workspace.take()?;
        if lease.workspace.is_none() {
            let bytes = plan.arena_bytes + plan.scratch_bytes;
            lease.workspace = Some(Workspace {
                arena: Aligned::new(plan.arena_bytes)?,
                scratch: Aligned::new(plan.scratch_bytes)?,
                stream: make_stream(&self.cpu)?,
            });
            self.cpu
                .retained
                .conv_workspace
                .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        }
        let workspace = lease
            .workspace
            .as_mut()
            .ok_or_else(|| Error::Backend("conv workspace missing".into()))?;
        let retained = workspace
            .arena
            .bytes
            .max(plan.arena_bytes)
            .checked_add(workspace.scratch.bytes.max(plan.scratch_bytes))
            .ok_or(Error::SizeOverflow("retained convolution workspace"))?;
        // Over the backend's memory budget the arena is cut back to this
        // request's size instead of staying at the largest page seen.
        let trim = retained > self.cpu.conv_limit() || self.cpu.over_budget();
        let before = workspace.arena.bytes + workspace.scratch.bytes;
        // Release the old buffer before allocating its replacement so a
        // growing arena never holds both (peak RSS would be old + new).
        if workspace.arena.bytes < plan.arena_bytes
            || (trim && workspace.arena.bytes != plan.arena_bytes)
        {
            workspace.arena = Aligned::new(0)?;
            workspace.arena = Aligned::new(plan.arena_bytes)?;
        }
        if workspace.scratch.bytes < plan.scratch_bytes
            || (trim && workspace.scratch.bytes != plan.scratch_bytes)
        {
            workspace.scratch = Aligned::new(0)?;
            workspace.scratch = Aligned::new(plan.scratch_bytes)?;
        }
        let after = workspace.arena.bytes + workspace.scratch.bytes;
        let total = &self.cpu.retained.conv_workspace;
        if after > before {
            total.fetch_add(after - before, std::sync::atomic::Ordering::Relaxed);
        } else if after < before {
            total.fetch_sub(before - after, std::sync::atomic::Ordering::Relaxed);
            self.cpu.note_released(before - after);
        }
        let arena = unsafe {
            std::slice::from_raw_parts_mut(
                workspace.arena.ptr().cast::<f32>(),
                plan.arena_bytes / 4,
            )
        };
        // Zero reused padded lanes before producers use a liveness alias.
        // Dense tensors are fully written by their producers before any read.
        for &(start, bytes) in &plan.zero {
            arena[start / 4..(start + bytes) / 4].fill(0.);
        }

        let _compute = rt_threads::compute_lock();
        self.cpu
            .conv_api()
            .apply_threads(self.cpu.threads() as i32)?;
        for step in &plan.steps {
            let mut pointers = Vec::with_capacity(step.specs.len());
            for (_, _, b) in &step.specs {
                pointers.push(match b {
                    Binding::Tensor(0) => input.as_ptr() as *mut c_void,
                    Binding::Tensor(id) => unsafe {
                        arena
                            .as_mut_ptr()
                            .cast::<u8>()
                            .add(plan.offsets[*id])
                            .cast()
                    },
                    Binding::Weights(w) => w.data.ptr(),
                    Binding::Bias(w) => w
                        .bias
                        .as_ref()
                        .ok_or_else(|| Error::Missing("conv bias".into()))?
                        .as_ptr() as *mut c_void,
                    Binding::Slopes(s) => s.as_ptr() as *mut c_void,
                    Binding::External => {
                        return Err(Error::Backend("external binding in graph".into()))
                    }
                });
            }
            step.execute(
                &self.cpu,
                &workspace.stream,
                &pointers,
                workspace.scratch.ptr(),
            )?;
        }
        let last = plan.offsets.len() - 1;
        let len = checked_size("conv output", &plan.output_shape)?;
        let mut output = rt_core::try_vec(len)?;
        let start = plan.offsets[last] / 4;
        output.copy_from_slice(&arena[start..start + len]);
        Ok((plan.output_shape, output))
    }
    fn plan(&self, shape: Shape) -> Result<Arc<Plan>> {
        dimensions(&shape)?;
        let lookup = |cache: &mut VecDeque<Arc<Plan>>| {
            let i = cache.iter().position(|p| p.shape == shape)?;
            let p = cache.remove(i)?;
            cache.push_front(p.clone());
            Some(p)
        };
        let lock = || {
            self.cache
                .lock()
                .map_err(|_| Error::Poisoned("conv plan cache"))
        };
        if let Some(p) = lookup(&mut *lock()?) {
            return Ok(p);
        }
        // Build without the cache lock so batches running side by side do
        // not wait on each other's cold shapes.
        self.cpu
            .conv_api()
            .apply_threads(self.cpu.threads() as i32)?;
        let heap = crate::cpu::heap_in_use();
        let mut plan = self.build_plan(shape)?;
        // Other threads allocate meanwhile, so this is an estimate.
        plan.bytes = heap
            .zip(crate::cpu::heap_in_use())
            .map_or(0, |(a, b)| b.saturating_sub(a).min(256 << 20));
        let mut cache = lock()?;
        if let Some(p) = lookup(&mut cache) {
            // Another caller built the same shape first.
            return Ok(p);
        }
        let p = Arc::new(plan);
        let plans = &self.cpu.retained.conv_plans;
        if cache.len() == self.capacity {
            if let Some(old) = cache.pop_back() {
                plans.fetch_sub(old.bytes, std::sync::atomic::Ordering::Relaxed);
            }
        }
        plans.fetch_add(p.bytes, std::sync::atomic::Ordering::Relaxed);
        cache.push_front(p.clone());
        // `run` cuts the arena back to this plan's size when over budget, so
        // count it at that size: plans are evicted only if trimming the arena
        // is not enough.
        let workspace = p.arena_bytes.saturating_add(p.scratch_bytes);
        if self.cpu.over_budget_with_workspace(workspace) {
            self.evict(&mut cache, workspace)?;
        }
        Ok(p)
    }
    /// Over the backend's memory budget: drop the least recently used plans
    /// (never the one just returned), then the cached steps only those plans
    /// held. An evicted shape is rebuilt the next time it runs.
    fn evict(&self, cache: &mut VecDeque<Arc<Plan>>, workspace: usize) -> Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        let retained = &self.cpu.retained;
        let mut released = 0;
        while self.cpu.over_budget_with_workspace(workspace) && cache.len() > 1 {
            let Some(old) = cache.pop_back() else { break };
            retained.conv_plans.fetch_sub(old.bytes, Relaxed);
            retained.conv_evictions.fetch_add(1, Relaxed);
            retained.conv_evicted.fetch_add(old.bytes, Relaxed);
            released += old.bytes;
        }
        if released > 0 {
            let mut caches = self
                .primitive_cache
                .lock()
                .map_err(|_| Error::Poisoned("conv primitive cache"))?;
            for steps in &mut caches.steps {
                steps.retain(|(_, step)| Arc::strong_count(step) > 1);
            }
            self.cpu.note_released(released);
        }
        Ok(())
    }
    /// Whether oneDNN would take this convolution's padding only on its
    /// reference kernel while the same convolution over an explicitly padded
    /// input has an optimized one. oneDNN's blocked AVX-512 and AVX2 kernels
    /// accept left padding only up to their register block width, which
    /// depends on the ISA and the shape (`jit_avx512_common_conv_kernel.cpp`,
    /// `jit_avx2_conv_kernel_f32.cpp`, oneDNN 3.13.4), so it is asked rather
    /// than predicted. Measured 2026-11-01: UVDoc's bridge convolutions
    /// dilated 7 to 18 on a 45 x 31 map took about 36 s per pass on the
    /// reference kernel under AVX-512; under AVX2, every 5x5 convolution
    /// dilated 3 fell to it. The probe omits the node's post-ops: one that
    /// they alone send to the reference kernel is padded for nothing.
    fn padding_needs_glue(
        &self,
        input: Shape,
        layouts: [bool; 2],
        w: &ConvWeights,
        spatial: &Spatial,
        output: Shape,
    ) -> Result<bool> {
        let reference = |input: Shape, spatial: &Spatial| -> Result<bool> {
            let tag = |blocked| match (blocked, self.cpu.conv_block()) {
                (false, _) => dnnl_nchw,
                (true, 16) => dnnl_nChw16c,
                (true, _) => dnnl_nChw8c,
            };
            let src = Md::tagged(&input, tag(layouts[0]))?;
            let dst = Md::tagged(&output, tag(layouts[1]))?;
            let bias = w
                .bias
                .as_ref()
                .map(|_| Md::tagged(&[w.output], dnnl_x))
                .transpose()?;
            let a = spatial.arrays()?;
            let mut attr = Attr(ptr::null_mut());
            let mut pd = Pd(ptr::null_mut());
            let mut info = ptr::null::<std::ffi::c_char>();
            unsafe {
                Api::check(dnnl_primitive_attr_create(&mut attr.0), "conv probe attr")?;
                Api::check(
                    dnnl_primitive_attr_set_fpmath_mode(attr.0, dnnl_fpmath_mode_strict),
                    "conv probe strict math",
                )?;
                Api::check(
                    dnnl_primitive_attr_set_scratchpad_mode(attr.0, dnnl_scratchpad_mode_user),
                    "conv probe user scratch",
                )?;
                Api::check(
                    dnnl_convolution_forward_primitive_desc_create(
                        &mut pd.0,
                        self.cpu.conv_engine(),
                        dnnl_forward_inference,
                        dnnl_convolution_direct,
                        src.raw(),
                        w.md.raw(),
                        bias.as_ref().map_or(ptr::null_mut(), |b| b.raw()),
                        dst.raw(),
                        a[0].as_ptr(),
                        a[1].as_ptr(),
                        a[2].as_ptr(),
                        a[3].as_ptr(),
                        attr.0,
                    ),
                    "conv padding probe",
                )?;
                Api::check(
                    dnnl_primitive_desc_query(
                        pd.0,
                        dnnl_query_impl_info_str,
                        0,
                        (&mut info as *mut *const std::ffi::c_char).cast(),
                    ),
                    "conv probe implementation",
                )?;
            }
            Ok(crate::api::cstr(info).starts_with("ref"))
        };
        if w.transpose
            || spatial
                .pads_begin
                .iter()
                .chain(&spatial.pads_end)
                .all(|&p| p == 0)
        {
            return Ok(false);
        }
        if !reference(input, spatial)? {
            return Ok(false);
        }
        let mut padded = input;
        for j in 0..2 {
            padded[j + 2] += spatial.pads_begin[j] + spatial.pads_end[j];
        }
        let unpadded = Spatial {
            pads_begin: [0; 2],
            pads_end: [0; 2],
            ..spatial.clone()
        };
        Ok(!reference(padded, &unpadded)?)
    }
    fn lower_nodes(&self, shape: Shape) -> Result<(Vec<Node>, Vec<usize>, Vec<bool>)> {
        let block = self.cpu.conv_block();
        let mut nodes = Vec::new();
        let mut identities = Vec::new();
        let mut shapes = vec![shape];
        let mut layouts = vec![false];
        let mut mapping = vec![0usize];
        for (identity, mut node) in self.nodes.clone().into_iter().enumerate() {
            for id in &mut node.inputs {
                *id = mapping[*id];
            }
            for post in &mut node.post_ops {
                if let PostOp::Binary { tensor, .. } = post {
                    *tensor = mapping[*tensor];
                }
            }
            if matches!(node.op, Op::Concat { axis: 1 })
                && node.inputs.iter().all(|&id| shapes[id][1] % block == 0)
            {
                node.op = Op::ConcatAlignedChannels;
            }
            let output = infer_shape(&node, &shapes, block)?;
            if matches!(node.op, Op::UpsampleNearest { .. }) {
                node.op = Op::Resize {
                    height: output[2],
                    width: output[3],
                    linear: false,
                };
            }
            if matches!(node.op,Op::ReduceMean{axes} if !axes.iter().any(|&a|a)) {
                node.op = Op::Reorder { blocked: true };
            }
            let want = match &node.op {
                Op::Conv { weights, .. } if weights.reads_plain() => false,
                Op::Reorder { .. } | Op::ZeroPad { .. } => layouts[node.inputs[0]],
                Op::DepthToSpace2x => true,
                _ => output_blocked(&node, output, block),
            };
            for id in &mut node.inputs {
                if layouts[*id] != want {
                    let old = *id;
                    nodes.push(Node::new(Op::Reorder { blocked: want }, [old]));
                    identities.push(identity * 2 + 1);
                    shapes.push(shapes[old]);
                    layouts.push(want);
                    *id = nodes.len();
                }
            }
            let blocked = if matches!(node.op, Op::ZeroPad { .. }) {
                want
            } else {
                output_blocked(&node, output, block)
            };
            if let Op::Conv { weights, spatial } = &node.op {
                let src = node.inputs[0];
                if self.padding_needs_glue(
                    shapes[src],
                    [want, blocked],
                    weights,
                    spatial,
                    output,
                )? {
                    let pad = Node::new(
                        Op::ZeroPad {
                            begin: spatial.pads_begin,
                            end: spatial.pads_end,
                        },
                        [src],
                    );
                    let padded = infer_shape(&pad, &shapes, block)?;
                    node.op = Op::Conv {
                        weights: weights.clone(),
                        spatial: Spatial {
                            pads_begin: [0; 2],
                            pads_end: [0; 2],
                            ..spatial.clone()
                        },
                    };
                    nodes.push(pad);
                    identities.push(identity * 2 + 1);
                    shapes.push(padded);
                    layouts.push(want);
                    node.inputs[0] = nodes.len();
                }
            }
            nodes.push(node);
            identities.push(identity * 2);
            shapes.push(output);
            layouts.push(blocked);
            mapping.push(nodes.len());
        }
        if layouts[nodes.len()] {
            nodes.push(Node::new(Op::Reorder { blocked: false }, [nodes.len()]));
            identities.push(self.nodes.len() * 2);
            layouts.push(false);
        }
        Ok((nodes, identities, layouts))
    }
    fn build_plan(&self, shape: Shape) -> Result<Plan> {
        let (nodes, identities, layouts) = self.lower_nodes(shape)?;
        let block = self.cpu.conv_block();
        let blocked_tag = if block == 16 {
            dnnl_nChw16c
        } else {
            dnnl_nChw8c
        };
        let mut caches = self
            .primitive_cache
            .lock()
            .map_err(|_| Error::Poisoned("conv primitive cache"))?;
        let Caches {
            steps: primitive_cache,
            natives,
            descriptors,
        } = &mut *caches;
        primitive_cache.resize_with(nodes.len(), VecDeque::new);
        let mut tensors = vec![Tensor {
            shape,
            md: descriptors.tagged(&shape, dnnl_nchw)?,
        }];
        let mut shapes = vec![shape];
        for node in &nodes {
            let output = infer_shape(node, &shapes, block)?;
            let blocked = layouts[tensors.len()];
            tensors.push(Tensor {
                shape: output,
                md: descriptors.tagged(&output, if blocked { blocked_tag } else { dnnl_nchw })?,
            });
            shapes.push(output);
        }
        let mut last: Vec<usize> = (0..tensors.len()).collect();
        for (i, n) in nodes.iter().enumerate() {
            for &id in &n.inputs {
                last[id] = i + 1;
            }
            for p in &n.post_ops {
                if let PostOp::Binary { tensor, .. } = p {
                    last[*tensor] = i + 1;
                }
            }
        }
        let last_id = tensors.len() - 1;
        last[last_id] = tensors.len();
        // A complete-channel concat at N=1 can share storage with its
        // producers if each input dies at the concat. Reserve the joined
        // interval conservatively and place the producers into its slices.
        let mut parent = (0..tensors.len()).collect::<Vec<_>>();
        let mut relative = vec![0usize; tensors.len()];
        let mut first = (0..tensors.len()).collect::<Vec<_>>();
        for (index, node) in nodes.iter().enumerate() {
            let dst = index + 1;
            if matches!(node.op, Op::ConcatAlignedChannels)
                && shapes[dst][0] == 1
                && node
                    .inputs
                    .iter()
                    .all(|&id| id > 0 && parent[id] == id && first[id] == id && last[id] == dst)
            {
                let mut offset = 0usize;
                for &id in &node.inputs {
                    parent[id] = dst;
                    relative[id] = offset;
                    first[dst] = first[dst].min(id);
                    offset = offset
                        .checked_add(tensors[id].md.bytes()?)
                        .ok_or(Error::SizeOverflow("concat placement"))?;
                }
                check_shape!(
                    offset == tensors[dst].md.bytes()?,
                    "concat placement extent"
                );
            }
        }
        // Largest-first interval placement avoids holes left by early large
        // feature maps. Network input is borrowed and occupies no arena space.
        let mut sizes = tensors
            .iter()
            .map(|t| rt_core::align_up(t.md.bytes()?, 64))
            .collect::<Result<Vec<_>>>()?;
        sizes[0] = 0;
        let mut order = (1..tensors.len())
            .filter(|&id| parent[id] == id)
            .collect::<Vec<_>>();
        order.sort_by_key(|&id| std::cmp::Reverse(sizes[id]));
        let mut offsets = vec![0usize; tensors.len()];
        let mut placed = Vec::new();
        let mut peak = 0usize;
        for id in order {
            let mut conflicts = placed
                .iter()
                .copied()
                .filter(|&old| first[id] <= last[old] && first[old] <= last[id])
                .map(|old| (offsets[old], sizes[old]))
                .collect::<Vec<_>>();
            conflicts.sort_unstable();
            let mut offset = 0usize;
            for (start, bytes) in conflicts {
                let end = offset
                    .checked_add(sizes[id])
                    .ok_or(Error::SizeOverflow("conv arena"))?;
                if end <= start {
                    break;
                }
                offset = offset.max(
                    start
                        .checked_add(bytes)
                        .ok_or(Error::SizeOverflow("conv arena"))?,
                );
            }
            offsets[id] = offset;
            peak = peak.max(
                offset
                    .checked_add(sizes[id])
                    .ok_or(Error::SizeOverflow("conv arena"))?,
            );
            placed.push(id);
        }
        for id in 1..tensors.len() {
            if parent[id] != id {
                offsets[id] = offsets[parent[id]]
                    .checked_add(relative[id])
                    .ok_or(Error::SizeOverflow("concat offset"))?;
            }
        }
        peak = peak
            .checked_add(64)
            .ok_or(Error::SizeOverflow("conv arena alignment"))?;
        let limit = self.cpu.conv_limit();
        if peak > limit {
            return Err(Error::WorkspaceLimit {
                operation: "ConvNet activation arena",
                requested: peak,
                limit,
            });
        }

        let mut steps = Vec::new();
        for (index, node) in nodes.iter().enumerate() {
            let dst_id = index + 1;
            let src_id = node.inputs[0];
            let src = &tensors[src_id];
            let s = src.shape;
            let dst = &tensors[dst_id];
            let output = dst.shape;
            // Stable original-node identity and binding IDs also protect reuse
            // if a different channel count changes inserted layout reorders.
            // Key only on the actual operand shapes, never the network width:
            // SE 1x1 gates and downstream ceil-divided widths share primitives.
            let mut key = vec![output];
            let mut bindings = node.inputs.clone();
            key.extend(node.inputs.iter().map(|&id| tensors[id].shape));
            for post in &node.post_ops {
                if let PostOp::Binary { tensor, .. } = post {
                    key.push(tensors[*tensor].shape);
                    bindings.push(*tensor);
                }
            }
            let key = (identities[index], bindings, key);
            let cache = &mut primitive_cache[index];
            if let Some(i) = cache.iter().position(|(k, _)| *k == key) {
                let entry = cache
                    .remove(i)
                    .ok_or_else(|| Error::Backend("conv primitive LRU index".into()))?;
                steps.push(entry.1.clone());
                cache.push_front(entry);
                continue;
            }
            if let Op::ZeroPad { begin, .. } = node.op {
                check_shape!(node.post_ops.is_empty(), "zero pad post-ops unsupported");
                let lanes = if layouts[src_id] { block } else { 1 };
                let groups = if layouts[src_id] {
                    s[1].div_ceil(block)
                } else {
                    s[1]
                };
                let step = Arc::new(Step {
                    primitive: Arc::new(Primitive(ptr::null_mut())),
                    specs: vec![
                        (DNNL_ARG_SRC, clone_md(&src.md)?, Binding::Tensor(src_id)),
                        (DNNL_ARG_DST, clone_md(&dst.md)?, Binding::Tensor(dst_id)),
                    ],
                    scratch: None,
                    scratch_bytes: 64,
                    implementation: "glue:zero_pad".into(),
                    mean: None,
                    concat: None,
                    shuffle: None,
                    tile: None,
                    pad: Some(PadGlue {
                        planes: checked_size("zero pad planes", &[s[0], groups])?,
                        height: s[2],
                        width: s[3],
                        lanes,
                        top: begin[0],
                        left: begin[1],
                        padded_height: output[2],
                        padded_width: output[3],
                    }),
                });
                retain_step(cache, self.capacity, key, step.clone());
                steps.push(step);
                continue;
            }
            if matches!(node.op, Op::DepthToSpace2x) {
                check_shape!(
                    node.post_ops.is_empty(),
                    "depth-to-space post-ops unsupported"
                );
                let step = Arc::new(Step {
                    primitive: Arc::new(Primitive(ptr::null_mut())),
                    specs: vec![
                        (DNNL_ARG_SRC, clone_md(&src.md)?, Binding::Tensor(src_id)),
                        (DNNL_ARG_DST, clone_md(&dst.md)?, Binding::Tensor(dst_id)),
                    ],
                    scratch: None,
                    scratch_bytes: 64,
                    implementation: "glue:depth_to_space_2x".into(),
                    mean: None,
                    concat: None,
                    shuffle: Some(s),
                    tile: None,
                    pad: None,
                });
                retain_step(cache, self.capacity, key, step.clone());
                steps.push(step);
                continue;
            }
            if matches!(node.op, Op::ConcatAlignedChannels) {
                check_shape!(
                    node.post_ops.is_empty(),
                    "channel concat post-ops unsupported"
                );
                let mut specs = vec![(DNNL_ARG_DST, clone_md(&dst.md)?, Binding::Tensor(dst_id))];
                let mut channels = Vec::new();
                for &id in &node.inputs {
                    specs.push((
                        DNNL_ARG_SRC,
                        clone_md(&tensors[id].md)?,
                        Binding::Tensor(id),
                    ));
                    channels.push(tensors[id].shape[1]);
                }
                let step = Arc::new(Step {
                    primitive: Arc::new(Primitive(ptr::null_mut())),
                    specs,
                    scratch: None,
                    scratch_bytes: 64,
                    implementation: if node.inputs.iter().all(|&id| parent[id] == dst_id) {
                        "glue:concat_producer_views".into()
                    } else {
                        "glue:aligned_channel_concat".into()
                    },
                    mean: None,
                    shuffle: None,
                    tile: None,
                    pad: None,
                    concat: Some((s[0], checked_size("concat plane", &[s[2], s[3]])?, channels)),
                });
                retain_step(cache, self.capacity, key, step.clone());
                steps.push(step);
                continue;
            }
            if let Op::SpatialMean(order) = node.op {
                check_shape!(
                    node.post_ops.is_empty(),
                    "spatial mean post-ops unsupported"
                );
                let step = Arc::new(Step {
                    primitive: Arc::new(Primitive(ptr::null_mut())),
                    specs: vec![
                        (DNNL_ARG_SRC, clone_md(&src.md)?, Binding::Tensor(src_id)),
                        (DNNL_ARG_DST, clone_md(&dst.md)?, Binding::Tensor(dst_id)),
                    ],
                    scratch: None,
                    scratch_bytes: 64,
                    implementation: format!("glue:fp32_spatial_mean_{order:?}"),
                    concat: None,
                    shuffle: None,
                    tile: None,
                    pad: None,
                    mean: Some((
                        checked_size("spatial channels", &[s[0], s[1]])?,
                        checked_size("spatial plane", &[s[2], s[3]])?,
                        order,
                    )),
                });
                retain_step(cache, self.capacity, key, step.clone());
                steps.push(step);
                continue;
            }
            let plane = checked_size("pointwise plane", &[s[2], s[3]])?;
            let tile = if self.pointwise_tiles
                && plane > 1
                && node
                    .post_ops
                    .iter()
                    .all(|p| !matches!(p, PostOp::Binary { .. }))
                && matches!(&node.op, Op::Conv { weights, spatial }
                    if !weights.transpose && weights.dense_blocked()
                        && weights.kernel == [1; 2] && spatial.strides == [1; 2]
                        && spatial.dilations == [1; 2] && spatial.pads_begin == [0; 2]
                        && spatial.pads_end == [0; 2])
            {
                Some(PointwiseTile {
                    shape: s,
                    plane,
                    output_channels: output[1],
                    width: plane.min(512).next_power_of_two().max(16),
                    source_offset: 0,
                    destination_offset: 0,
                })
            } else {
                None
            };
            let src_md = if let Some(t) = tile {
                descriptors.tagged(&[1, s[1], 1, t.width], blocked_tag)?
            } else {
                clone_md(&src.md)?
            };
            let dst_md = if let Some(t) = tile {
                descriptors.tagged(&[1, output[1], 1, t.width], blocked_tag)?
            } else {
                clone_md(&dst.md)?
            };
            let mut attr = Attr(ptr::null_mut());
            Api::check(
                unsafe { dnnl_primitive_attr_create(&mut attr.0) },
                "conv attr",
            )?;
            Api::check(
                unsafe { dnnl_primitive_attr_set_fpmath_mode(attr.0, dnnl_fpmath_mode_strict) },
                "conv strict math",
            )?;

            Api::check(
                unsafe {
                    dnnl_primitive_attr_set_scratchpad_mode(attr.0, dnnl_scratchpad_mode_user)
                },
                "conv user scratch",
            )?;
            let mut specs = vec![
                (DNNL_ARG_SRC, clone_md(&src_md)?, Binding::Tensor(src_id)),
                (DNNL_ARG_DST, clone_md(&dst_md)?, Binding::Tensor(dst_id)),
            ];
            let mut posts = Posts(ptr::null_mut());
            Api::check(
                unsafe { dnnl_post_ops_create(&mut posts.0) },
                "conv post-ops",
            )?;
            for (i, p) in node.post_ops.iter().enumerate() {
                match p {
                    PostOp::Scalar { kind, value } => {
                        check_shape!(value.len() == 1 && value[0].is_finite(), "scalar post-op");
                        let md = descriptors.tagged(&[1, 1, 1, 1], dnnl_nchw)?;
                        Api::check(
                            unsafe { dnnl_post_ops_append_binary(posts.0, kind.alg(), md.raw()) },
                            "scalar binary post-op",
                        )?;
                        specs.push((
                            (DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE * (i as i32 + 1)) | DNNL_ARG_SRC_1,
                            md,
                            Binding::Slopes(value.clone()),
                        ));
                    }
                    PostOp::Eltwise(a) => {
                        let (alg, alpha, beta) = a.params()?;
                        Api::check(
                            unsafe { dnnl_post_ops_append_eltwise(posts.0, alg, alpha, beta) },
                            "conv eltwise post-op",
                        )?;
                    }
                    PostOp::ChannelBias(bias) => {
                        check_shape!(bias.len() == output[1], "channel post-bias extent");
                        let md = descriptors.tagged(&[1, output[1], 1, 1], dnnl_nchw)?;
                        Api::check(
                            unsafe {
                                dnnl_post_ops_append_binary(posts.0, dnnl_binary_add, md.raw())
                            },
                            "channel bias post-op",
                        )?;
                        specs.push((
                            (DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE * (i as i32 + 1)) | DNNL_ARG_SRC_1,
                            md,
                            Binding::Slopes(bias.clone()),
                        ));
                    }
                    PostOp::Binary { kind, tensor } => {
                        let t = &tensors[*tensor];
                        broadcast(output, t.shape)?;
                        Api::check(
                            unsafe { dnnl_post_ops_append_binary(posts.0, kind.alg(), t.md.raw()) },
                            "conv binary post-op",
                        )?;
                        specs.push((
                            (DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE * (i as i32 + 1)) | DNNL_ARG_SRC_1,
                            clone_md(&t.md)?,
                            Binding::Tensor(*tensor),
                        ));
                    }
                }
            }
            Api::check(
                unsafe { dnnl_primitive_attr_set_post_ops(attr.0, posts.0) },
                "conv post-op attributes",
            )?;
            let mut pd = Pd(ptr::null_mut());
            let e = self.cpu.conv_engine();
            let st = unsafe {
                match &node.op {
                    Op::Conv {
                        weights: w,
                        spatial,
                    } => {
                        let a = spatial.arrays()?;
                        let bias = if w.bias.is_some() {
                            Some(descriptors.tagged(&[w.output], dnnl_x)?)
                        } else {
                            None
                        };
                        let bias_ptr = bias.as_ref().map(|b| b.raw()).unwrap_or(ptr::null_mut());
                        specs.push((
                            DNNL_ARG_WEIGHTS,
                            clone_md(&w.md)?,
                            Binding::Weights(w.clone()),
                        ));
                        if let Some(b) = bias {
                            specs.push((DNNL_ARG_BIAS, b, Binding::Bias(w.clone())));
                        }
                        let create = if w.transpose {
                            dnnl_deconvolution_forward_primitive_desc_create
                        } else {
                            dnnl_convolution_forward_primitive_desc_create
                        };
                        create(
                            &mut pd.0,
                            e,
                            dnnl_forward_inference,
                            if w.transpose {
                                dnnl_deconvolution_direct
                            } else {
                                dnnl_convolution_direct
                            },
                            src_md.raw(),
                            w.md.raw(),
                            bias_ptr,
                            dst_md.raw(),
                            a[0].as_ptr(),
                            a[1].as_ptr(),
                            a[2].as_ptr(),
                            a[3].as_ptr(),
                            attr.0,
                        )
                    }
                    Op::Pool {
                        kind,
                        kernel,
                        spatial,
                    } => {
                        let a = spatial.arrays()?;
                        let k = dimensions(kernel)?;
                        dnnl_pooling_forward_primitive_desc_create(
                            &mut pd.0,
                            e,
                            dnnl_forward_inference,
                            kind.alg(),
                            src_md.raw(),
                            dst_md.raw(),
                            a[0].as_ptr(),
                            k.as_ptr(),
                            a[1].as_ptr(),
                            a[2].as_ptr(),
                            a[3].as_ptr(),
                            attr.0,
                        )
                    }
                    Op::Resize { linear, .. } => dnnl_resampling_forward_primitive_desc_create(
                        &mut pd.0,
                        e,
                        dnnl_forward_inference,
                        if *linear {
                            dnnl_resampling_linear
                        } else {
                            dnnl_resampling_nearest
                        },
                        ptr::null(),
                        src_md.raw(),
                        dst_md.raw(),
                        attr.0,
                    ),
                    Op::DepthToSpace2x => {
                        return Err(Error::Backend("unexpected depth-to-space primitive".into()))
                    }
                    Op::SpatialMean(_) => {
                        return Err(Error::Backend("unexpected spatial mean primitive".into()))
                    }
                    Op::ZeroPad { .. } => {
                        return Err(Error::Backend("unexpected zero pad primitive".into()))
                    }
                    Op::UpsampleNearest { .. } => {
                        return Err(Error::Backend("unresolved resize scale".into()))
                    }
                    Op::ReduceMean { .. } => dnnl_reduction_primitive_desc_create(
                        &mut pd.0,
                        e,
                        dnnl_reduction_mean,
                        src_md.raw(),
                        dst_md.raw(),
                        0.,
                        0.,
                        attr.0,
                    ),
                    Op::ConcatAlignedChannels | Op::Concat { .. } => {
                        let axis = if let Op::Concat { axis } = node.op {
                            axis
                        } else {
                            1
                        };
                        specs.remove(0);
                        let mut mds = Vec::new();
                        for (i, &id) in node.inputs.iter().enumerate() {
                            mds.push(tensors[id].md.raw() as const_dnnl_memory_desc_t);
                            specs.push((
                                DNNL_ARG_MULTIPLE_SRC + i as i32,
                                clone_md(&tensors[id].md)?,
                                Binding::Tensor(id),
                            ));
                        }
                        dnnl_concat_primitive_desc_create(
                            &mut pd.0,
                            e,
                            dst_md.raw(),
                            mds.len() as i32,
                            axis as i32,
                            mds.as_ptr(),
                            attr.0,
                        )
                    }
                    Op::Eltwise(a) => {
                        let (alg, alpha, beta) = a.params()?;
                        dnnl_eltwise_forward_primitive_desc_create(
                            &mut pd.0,
                            e,
                            dnnl_forward_inference,
                            alg,
                            src_md.raw(),
                            dst_md.raw(),
                            alpha,
                            beta,
                            attr.0,
                        )
                    }
                    Op::Binary(kind) => {
                        let t = &tensors[node.inputs[1]];
                        specs.push((
                            DNNL_ARG_SRC_1,
                            clone_md(&t.md)?,
                            Binding::Tensor(node.inputs[1]),
                        ));
                        dnnl_binary_primitive_desc_create(
                            &mut pd.0,
                            e,
                            kind.alg(),
                            src_md.raw(),
                            t.md.raw(),
                            dst_md.raw(),
                            attr.0,
                        )
                    }
                    Op::Prelu { slopes } => {
                        check_shape!(slopes.len() == 1 || slopes.len() == s[1], "PReLU slopes");
                        let slopes = if slopes.len() == 1 {
                            let mut expanded = rt_core::try_vec(s[1])?;
                            expanded.fill(slopes[0]);
                            expanded
                        } else {
                            slopes.clone()
                        };
                        let md = descriptors.tagged(&[1, slopes.len(), 1, 1], dnnl_nchw)?;
                        let st = dnnl_prelu_forward_primitive_desc_create(
                            &mut pd.0,
                            e,
                            dnnl_forward_inference,
                            src_md.raw(),
                            md.raw(),
                            dst_md.raw(),
                            attr.0,
                        );
                        specs.push((DNNL_ARG_WEIGHTS, md, Binding::Slopes(Arc::new(slopes))));
                        st
                    }
                    Op::Reorder { .. } => {
                        check_shape!(node.post_ops.is_empty(), "post-ops on reorder unsupported");
                        dnnl_reorder_primitive_desc_create(
                            &mut pd.0,
                            src_md.raw(),
                            e,
                            dst_md.raw(),
                            e,
                            attr.0,
                        )
                    }
                }
            };
            Api::check(st, "conv graph primitive descriptor")?;
            let mut step = Step::new_cached(
                pd,
                specs,
                Some((natives, node, self.capacity.saturating_mul(nodes.len()))),
            )?;
            if let Some(mut tile) = tile {
                tile.source_offset = rt_core::align_up(step.scratch_bytes, 64)?;
                tile.destination_offset = tile
                    .source_offset
                    .checked_add(src_md.bytes()?)
                    .ok_or(Error::SizeOverflow("pointwise scratch"))?;
                step.scratch_bytes = tile
                    .destination_offset
                    .checked_add(dst_md.bytes()?)
                    .ok_or(Error::SizeOverflow("pointwise scratch"))?;
                step.tile = Some(tile);
            }
            let step = Arc::new(step);
            retain_step(cache, self.capacity, key, step.clone());
            steps.push(step);
        }
        let mut zero = Vec::new();
        for id in 1..tensors.len() {
            let bytes = tensors[id].md.bytes()?;
            let dense = checked_size(
                "conv tensor bytes",
                &[checked_size("conv tensor", &tensors[id].shape)?, 4],
            )?;
            if bytes != dense {
                zero.push((offsets[id], rt_core::align_up(bytes, 64)?));
            }
        }
        zero.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(zero.len());
        for (start, bytes) in zero {
            match merged.last_mut() {
                Some((s0, b0)) if start <= *s0 + *b0 => *b0 = (*b0).max(start + bytes - *s0),
                _ => merged.push((start, bytes)),
            }
        }
        let scratch_bytes = steps.iter().map(|s| s.scratch_bytes).max().unwrap_or(64);
        let total = peak
            .checked_add(scratch_bytes)
            .ok_or(Error::SizeOverflow("conv workspace"))?;
        let limit = self.cpu.conv_limit();
        if total > limit {
            return Err(Error::WorkspaceLimit {
                operation: "ConvNet activation arena + scratch",
                requested: total,
                limit,
            });
        }
        Ok(Plan {
            shape,
            output_shape: shapes[last_id],
            steps,
            offsets,
            zero: merged,
            arena_bytes: peak,
            scratch_bytes,
            bytes: 0,
        })
    }
}

fn retain_step(cache: &mut StepLru, capacity: usize, key: StepKey, step: Arc<Step>) {
    if cache.len() == capacity {
        cache.pop_back();
    }
    cache.push_front((key, step));
}
