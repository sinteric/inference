//! FP32 convolution graphs with fixed packed weights, a shape LRU and a checked
//! liveness arena. Graphs are supplied by model code; no ONNX interpretation.
//!
//! The operators: convolution, depthwise and transposed convolution, pooling,
//! resampling, spatial means, concat, PReLU, reorders, and activation and
//! binary post-ops, on the model's backend with the runtime's pinned pool and
//! ISA. Grouped convolution beyond one group or depthwise is unsupported.
//! Integral nearest upsampling matches ONNX asymmetric/floor; linear
//! resampling uses half-pixel centres, not align-corners. Spatial means use
//! fixed FP32 accumulation orders to match ONNX Runtime.
//!
//! **Layouts.** Weights use explicit tags fixed by channels and ISA,
//! independent of height and width. Graph inputs and outputs are NCHW;
//! activations inside use channel blocks of the ISA's vector width (nChw16c
//! with AVX-512, nChw8c below it), with plain reorders where an optimized
//! small-channel convolution, a nonzero-preserving eltwise, a reduction or
//! PReLU needs them. The block follows the ISA because oneDNN's AVX2
//! convolutions take only 8-channel blocks: given 16, every convolution fell
//! to the reference kernel (measured 2026-11-01 on a 32-channel 5x5:
//! 19 MMAC/s against 74 GMAC/s for the AVX-512 kernel). oneDNN names no
//! 8-channel transposed-convolution weight layout, so below AVX-512 a
//! transposed convolution is [`crate::ErrorKind::Unsupported`].
//!
//! **Wide padding.** oneDNN 3.13.4's jit convolutions take left padding no
//! wider than their register block, which shrinks with the ISA's vector
//! width; wider padding falls to the reference kernel. When the plan sees
//! that, it zero-pads the input itself and runs the convolution unpadded:
//! the same sums on a jit kernel. The probe asks oneDNN without the node's
//! post-ops, so a convolution its post-ops alone send to the reference
//! kernel is padded for nothing and stays there.
//!
//! **Plans and memory.** [`ConvNet::try_new`] takes a positive LRU capacity of
//! shape plans. A plan is built per input shape and graph identity; plans
//! share packed weights and immutable memory descriptors, and identical
//! native descriptors share primitive code. A largest-first liveness arena
//! borrows the network input; aligned channel concats place their producers
//! straight into the joined storage when lifetimes permit. Arena plus the
//! largest scratchpad is refused ([`crate::ErrorKind::TooLarge`]) before any
//! activation is allocated when it exceeds the backend's workspace limit.
//! Segments on one backend reuse one aligned arena, scratchpad and stream,
//! grown to the largest needed. Nothing is cached on disk.
use crate::cpu::onednn::api::Api;
use crate::cpu::onednn::cpu::Aligned;
use crate::cpu::onednn::OnednnCpu;
use crate::{check_shape, checked_size, Error, Result};
use onednn_sys::*;
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Condvar, Mutex, Weak};

pub use crate::cpu::kernels::SpatialMean;

mod build;
mod lower;
mod step;
mod weights;

use build::native_key;
use lower::{broadcast, infer_shape};
use step::{clone_md, make_stream};

pub type Shape = [usize; 4];

fn dimensions(d: &[usize]) -> Result<Vec<i64>> {
    check_shape!(
        !d.is_empty() && d.len() <= 12 && !d.contains(&0),
        "invalid dimensions {d:?}"
    );
    let bytes = checked_size("convolution tensor", d)?
        .checked_mul(4)
        .ok_or(Error::size_overflow("tensor bytes"))?;
    check_shape!(
        bytes <= isize::MAX as usize,
        "tensor extent exceeds address space"
    );
    d.iter()
        .map(|&x| i64::try_from(x).map_err(|_| Error::size_overflow("oneDNN dimension")))
        .collect()
}

/// An owned oneDNN handle, destroyed on drop. `shared` marks a handle that is
/// immutable once created, so any thread may use or share it; `owned` marks
/// one a single thread uses at a time, which may still move between threads.
/// A handle with neither stays on the thread that made it.
macro_rules! handle {
    ($name:ident, $ty:ty, $destroy:path) => {
        struct $name($ty);
        impl Drop for $name {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    // SAFETY: the handle is non-null, this value owns it, and
                    // nothing uses it after this drop.
                    unsafe {
                        $destroy(self.0);
                    }
                }
            }
        }
    };
    ($name:ident, $ty:ty, $destroy:path,shared) => {
        handle!($name, $ty, $destroy);
        // SAFETY: oneDNN memory descriptors, primitive descriptors and
        // primitives are immutable after creation and documented as safe to
        // use from several threads at once; each execute brings its own
        // memories, stream and scratchpad.
        unsafe impl Send for $name {}
        // SAFETY: as for `Send` above; no method mutates the handle.
        unsafe impl Sync for $name {}
    };
    ($name:ident, $ty:ty, $destroy:path,owned) => {
        handle!($name, $ty, $destroy);
        // SAFETY: a oneDNN stream is not tied to the thread that created it;
        // the workspace that owns it is taken under a mutex, so one thread
        // uses it at a time.
        unsafe impl Send for $name {}
    };
}
handle!(
    MdOwner,
    dnnl_memory_desc_t,
    dnnl_memory_desc_destroy,
    shared
);
#[derive(Clone)]
struct Md(Arc<MdOwner>);
handle!(
    Pd,
    dnnl_primitive_desc_t,
    dnnl_primitive_desc_destroy,
    shared
);
handle!(Primitive, dnnl_primitive_t, dnnl_primitive_destroy, shared);
handle!(Attr, dnnl_primitive_attr_t, dnnl_primitive_attr_destroy);
handle!(Posts, dnnl_post_ops_t, dnnl_post_ops_destroy);
handle!(Memory, dnnl_memory_t, dnnl_memory_destroy);
handle!(Stream, dnnl_stream_t, dnnl_stream_destroy, owned);

impl Md {
    fn tagged(dims: &[usize], tag: i32) -> Result<Self> {
        let d = dimensions(dims)?;
        let mut md = ptr::null_mut();
        Api::check(
            // SAFETY: `d` holds `d.len()` dimensions (at most 12, checked by
            // `dimensions`), and `md` is a valid out-pointer.
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
        // SAFETY: `self` owns a live descriptor.
        let n = unsafe { dnnl_memory_desc_get_size(self.raw()) };
        check_shape!(
            n > 0 && n <= isize::MAX as usize && n % 4 == 0,
            "invalid memory descriptor extent {n}"
        );
        Ok(n)
    }
}

/// The shape of a convolution's weights: `output` filters, each `kernel`
/// (height, width) over `input / groups` channels. Deconvolution weights keep
/// the same logical order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Filters {
    /// Output channels.
    pub output: usize,
    /// Input channels.
    pub input: usize,
    /// Kernel height and width.
    pub kernel: [usize; 2],
    /// 1, or the channel count for depthwise filters.
    pub groups: usize,
    /// Deconvolution (transposed convolution) filters.
    pub transpose: bool,
}
impl Filters {
    /// Dense `k x k` filters from `input` to `output` channels.
    pub fn square(output: usize, input: usize, k: usize) -> Self {
        Self::new(output, input, [k, k])
    }
    /// Dense `kernel` filters from `input` to `output` channels.
    pub fn new(output: usize, input: usize, kernel: [usize; 2]) -> Self {
        Self {
            output,
            input,
            kernel,
            groups: 1,
            transpose: false,
        }
    }
    /// One `k x k` filter per channel.
    pub fn depthwise(channels: usize, k: usize) -> Self {
        Self {
            groups: channels,
            ..Self::square(channels, channels, k)
        }
    }
}

/// Immutable, spatial-shape-independent convolution weights. Source layout is
/// OIHW (GOIHW for groups); deconvolution also uses logical OIHW dimensions.
pub struct ConvWeights {
    md: Md,
    data: Aligned,
    bias: Option<Vec<f32>>,
    filters: Filters,
    // Own the engine until the last weight/primitive is released.
    cpu: Arc<OnednnCpu>,
    tag: i32,
}

impl Drop for ConvWeights {
    fn drop(&mut self) {
        self.cpu
            .conv_weight_bytes
            .fetch_sub(self.data.bytes(), std::sync::atomic::Ordering::Relaxed);
    }
}

/// Spatial convolution/pool parameters, using ONNX dilation (1 = no dilation).
#[derive(Clone, Copy, Debug)]
pub struct Spatial {
    /// Step between windows, per axis.
    pub strides: [usize; 2],
    /// Spacing between kernel taps, per axis (1 = adjacent).
    pub dilations: [usize; 2],
    /// Zero padding before the first row and column.
    pub pads_begin: [usize; 2],
    /// Zero padding after the last row and column.
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
    /// The same stride, dilation and padding on both axes and both sides.
    pub fn square(stride: usize, dilation: usize, pad: usize) -> Self {
        Self {
            strides: [stride; 2],
            dilations: [dilation; 2],
            pads_begin: [pad; 2],
            pads_end: [pad; 2],
        }
    }
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
                    i64::try_from(v[j]).map_err(|_| Error::size_overflow("spatial attributes"))?;
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
                .ok_or(Error::size_overflow("dilated kernel"))?;
            let pad = self.pads_begin[j]
                .checked_add(self.pads_end[j])
                .ok_or(Error::size_overflow("padding"))?;
            let v = if transpose {
                (input[j + 2] - 1)
                    .checked_mul(self.strides[j])
                    .and_then(|v| v.checked_add(effective))
                    .and_then(|v| v.checked_sub(pad))
                    .ok_or_else(|| Error::internal("deconvolution output extent"))?
            } else {
                let padded = input[j + 2]
                    .checked_add(pad)
                    .ok_or(Error::size_overflow("padded input"))?;
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
    AverageExcludePadding,
}
impl Pool {
    fn alg(self) -> i32 {
        match self {
            Self::Max => dnnl_pooling_max,
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
    /// Rearrange `[N, 4*C, H, W]` into `[N, C, 2*H, 2*W]` (ONNX CRD order).
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
    Concat {
        axis: usize,
    },
    /// Channel concat whose inputs all have complete channel blocks
    /// ([`OnednnCpu::conv_block`]).
    ConcatAlignedChannels,
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
/// A buffer a step binds: where it starts, and how many bytes from there
/// the step may read or write through it.
#[derive(Clone, Copy)]
struct Bound {
    ptr: *mut c_void,
    bytes: usize,
}

impl Bound {
    /// A buffer the step only reads.
    fn of(v: &[f32]) -> Self {
        Self {
            ptr: v.as_ptr().cast_mut().cast(),
            bytes: std::mem::size_of_val(v),
        }
    }
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
    /// oneDNN's (or the glue's) name for what runs; tests check it.
    #[cfg_attr(not(test), expect(dead_code, reason = "only tests read plans' names"))]
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
#[cfg(test)]
#[derive(Clone, Debug)]
pub struct PlanInfo {
    pub output_shape: Shape,
    pub arena_bytes: usize,
    pub scratch_bytes: usize,
    pub implementations: Vec<String>,
}
impl Plan {
    #[cfg(test)]
    fn info(&self) -> PlanInfo {
        PlanInfo {
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
        (self.arena.bytes(), self.scratch.bytes())
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
            .map_err(|_| Error::poisoned("conv workspace"))?;
        let parallel = crate::cpu::threads::in_parallel();
        loop {
            // Outside a job take the largest workspace, inside one the
            // smallest: only one grows to the largest input (the detector's
            // page) instead of each in turn.
            let bytes = |w: &Workspace| w.arena.bytes() + w.scratch.bytes();
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
                .map_err(|_| Error::poisoned("conv workspace"))?;
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

pub struct ConvNet {
    pointwise_tiles: bool,
    primitive_cache: Mutex<Caches>,
    cache: Mutex<VecDeque<Arc<Plan>>>,
    nodes: Vec<Node>,
    capacity: usize,
    cpu: Arc<OnednnCpu>,
}
impl ConvNet {
    pub fn try_new(cpu: Arc<OnednnCpu>, nodes: Vec<Node>, capacity: usize) -> Result<Self> {
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
    /// The graph's first `count` nodes as a network of their own, for the
    /// detector's `dump_means` diagnostic test.
    #[cfg(all(test, feature = "debug-dumps"))]
    pub fn prefix(&self, count: usize) -> Result<Self> {
        check_shape!(
            count > 0 && count <= self.nodes.len(),
            "invalid graph prefix"
        );
        let mut net = Self::try_new(self.cpu.clone(), self.nodes[..count].to_vec(), 1)?;
        net.pointwise_tiles = self.pointwise_tiles;
        Ok(net)
    }
    #[cfg(test)]
    pub fn cached_shapes(&self) -> Result<Vec<Shape>> {
        Ok(self
            .cache
            .lock()
            .map_err(|_| Error::poisoned("conv plan cache"))?
            .iter()
            .map(|p| p.shape)
            .collect())
    }
    /// Build the plan for `shape` now and describe it.
    #[cfg(test)]
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
                arena: Aligned::try_new(plan.arena_bytes)?,
                scratch: Aligned::try_new(plan.scratch_bytes)?,
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
            .ok_or_else(|| Error::internal("conv workspace missing"))?;
        let retained = workspace
            .arena
            .bytes()
            .max(plan.arena_bytes)
            .checked_add(workspace.scratch.bytes().max(plan.scratch_bytes))
            .ok_or(Error::size_overflow("retained convolution workspace"))?;
        // Over the backend's memory budget the arena is cut back to this
        // request's size instead of staying at the largest page seen.
        let trim = retained > self.cpu.conv_limit() || self.cpu.over_budget();
        let before = workspace.arena.bytes() + workspace.scratch.bytes();
        // Release the old buffer before allocating its replacement so a
        // growing arena never holds both (peak RSS would be old + new).
        if workspace.arena.bytes() < plan.arena_bytes
            || (trim && workspace.arena.bytes() != plan.arena_bytes)
        {
            workspace.arena = Aligned::try_new(0)?;
            workspace.arena = Aligned::try_new(plan.arena_bytes)?;
        }
        if workspace.scratch.bytes() < plan.scratch_bytes
            || (trim && workspace.scratch.bytes() != plan.scratch_bytes)
        {
            workspace.scratch = Aligned::try_new(0)?;
            workspace.scratch = Aligned::try_new(plan.scratch_bytes)?;
        }
        let after = workspace.arena.bytes() + workspace.scratch.bytes();
        let total = &self.cpu.retained.conv_workspace;
        if after > before {
            total.fetch_add(after - before, std::sync::atomic::Ordering::Relaxed);
        } else if after < before {
            total.fetch_sub(before - after, std::sync::atomic::Ordering::Relaxed);
            self.cpu.note_released(before - after);
        }
        let arena = &mut workspace.arena.as_f32_mut()[..plan.arena_bytes / 4];
        // Zero reused padded lanes before producers use a liveness alias.
        // Dense tensors are fully written by their producers before any read.
        for &(start, bytes) in &plan.zero {
            arena[start / 4..(start + bytes) / 4].fill(0.);
        }

        let _compute = crate::cpu::threads::compute_lock();
        self.cpu
            .conv_api()
            .apply_threads(self.cpu.threads() as i32)?;
        // Every step binds tensors through this one pointer, so that none
        // of them outlives a later borrow of the arena.
        let (base, arena_len) = (arena.as_mut_ptr(), arena.len());
        let scratch = Bound {
            ptr: workspace.scratch.ptr(),
            bytes: workspace.scratch.bytes(),
        };
        for step in &plan.steps {
            let mut bufs = Vec::with_capacity(step.specs.len());
            for (_, _, b) in &step.specs {
                bufs.push(match b {
                    // The graph input: no step writes tensor 0, which no node
                    // produces.
                    Binding::Tensor(0) => Bound::of(input),
                    Binding::Tensor(id) => {
                        let at = plan.offsets.get(*id).map_or(usize::MAX, |&o| o);
                        check_shape!(
                            at % 4 == 0 && at / 4 <= arena_len,
                            "conv tensor offset {at}"
                        );
                        // SAFETY: `at / 4` is at most the arena's length
                        // (checked just above).
                        let ptr = unsafe { base.add(at / 4) }.cast();
                        Bound {
                            ptr,
                            bytes: (arena_len - at / 4) * 4,
                        }
                    }
                    Binding::Weights(w) => Bound {
                        ptr: w.data.ptr(),
                        bytes: w.data.bytes(),
                    },
                    Binding::Bias(w) => {
                        Bound::of(w.bias.as_ref().ok_or_else(|| Error::missing("conv bias"))?)
                    }
                    Binding::Slopes(s) => Bound::of(s),
                    Binding::External => return Err(Error::internal("external binding in graph")),
                });
            }
            step.execute(&self.cpu, &workspace.stream, &bufs, scratch)?;
        }
        let last = plan.offsets.len() - 1;
        let len = checked_size("conv output", &plan.output_shape)?;
        let mut output = crate::try_vec(len)?;
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
                .map_err(|_| Error::poisoned("conv plan cache"))
        };
        if let Some(p) = lookup(&mut *lock()?) {
            return Ok(p);
        }
        // Build without the cache lock so batches running side by side do
        // not wait on each other's cold shapes.
        self.cpu
            .conv_api()
            .apply_threads(self.cpu.threads() as i32)?;
        let heap = crate::cpu::onednn::cpu::heap_in_use();
        let mut plan = self.build_plan(shape)?;
        // Other threads allocate meanwhile, so this is an estimate.
        plan.bytes = heap
            .zip(crate::cpu::onednn::cpu::heap_in_use())
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
                .map_err(|_| Error::poisoned("conv primitive cache"))?;
            for steps in &mut caches.steps {
                steps.retain(|(_, step)| Arc::strong_count(step) > 1);
            }
            self.cpu.note_released(released);
        }
        Ok(())
    }
}
