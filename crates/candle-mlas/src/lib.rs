//! Model-independent CPU FP32 GEMM, immutable packed weights and fresh Candle outputs.
//! Native code is explicitly loaded once; see `initialize` for its safety contract.
use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, Error, InplaceOp2, Layout, Result, Shape, Tensor,
};
use libloading::Library;
use std::ffi::CStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
#[derive(Clone, Default, serde::Serialize)]
pub struct ProjectionAllocation {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub residual: bool,
    pub calls: usize,
    pub initialized_bytes: usize,
    pub allocation_s: f64,
    pub compute_s: f64,
}
#[derive(Default)]
struct ProjectionProfile {
    enabled: bool,
    records: std::collections::BTreeMap<(usize, usize, usize, bool), ProjectionAllocation>,
}
thread_local! {static PROJECTION_PROFILE:std::cell::RefCell<ProjectionProfile>=std::cell::RefCell::new(ProjectionProfile::default());}
#[doc(hidden)]
pub struct ProfileScope {
    previous: bool,
}
impl ProfileScope {
    pub fn enter(enabled: bool) -> Self {
        let previous = PROJECTION_PROFILE.with(|p| {
            let mut p = p.borrow_mut();
            let old = p.enabled;
            p.enabled = enabled;
            p.records.clear();
            old
        });
        Self { previous }
    }
}
impl Drop for ProfileScope {
    fn drop(&mut self) {
        PROJECTION_PROFILE.with(|p| p.borrow_mut().enabled = self.previous);
    }
}
/// Diagnostic totals from the most recent profiled forward on the calling thread.
pub fn projection_allocation_profile() -> Vec<ProjectionAllocation> {
    PROJECTION_PROFILE.with(|p| p.borrow().records.values().cloned().collect())
}
fn projection_timer() -> Option<std::time::Instant> {
    PROJECTION_PROFILE
        .with(|p| p.borrow().enabled)
        .then(std::time::Instant::now)
}
fn record_projection(
    p: &PackedWeight,
    m: usize,
    residual: bool,
    bytes: usize,
    allocation: Option<f64>,
    compute: Option<std::time::Instant>,
) {
    if let (Some(allocation), Some(compute)) = (allocation, compute) {
        let compute_s = compute.elapsed().as_secs_f64();
        PROJECTION_PROFILE.with(|state| {
            let mut state = state.borrow_mut();
            let value = state
                .records
                .entry((m, p.n, p.k, residual))
                .or_insert_with(|| ProjectionAllocation {
                    m,
                    n: p.n,
                    k: p.k,
                    residual,
                    ..Default::default()
                });
            value.calls += 1;
            value.initialized_bytes += bytes;
            value.allocation_s += allocation;
            value.compute_s += compute_s;
        });
    }
}

type Sgemm = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    f32,
    *const f32,
    i32,
    *const f32,
    i32,
    f32,
    *mut f32,
    i32,
);
type PackSize = unsafe extern "C" fn(i32, i32, i32, i32) -> usize;
type Pack = unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, f32, *const f32, i32, *mut f32);
type Compute = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f32,
    i32,
    *const f32,
    i32,
    f32,
    *mut f32,
    i32,
);
type GeGlu = unsafe extern "C" fn(*const f32, *mut f32, usize, usize);
type SgemmBatch = unsafe extern "C" fn(
    i32,
    i32,
    usize,
    usize,
    usize,
    usize,
    *const f32,
    usize,
    usize,
    *const f32,
    usize,
    usize,
    *mut f32,
    usize,
    usize,
);
type Norm = unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32, usize, usize, f32);
struct PackApi {
    size: PackSize,
    pack: Pack,
    compute: Compute,
}
struct Blas {
    _library: Library,
    sgemm: Sgemm,
    packed: Option<PackApi>,
    m_independent: bool,
    geglu: Option<GeGlu>,
    batch: Option<SgemmBatch>,
    norm: Option<Norm>,
    aligned_buffers: bool,
    debug_alignment: bool,
    gemm_residual: bool,
    info: String,
}
static BLAS: OnceLock<Blas> = OnceLock::new();
static BLAS_INIT: Mutex<()> = Mutex::new(());
/// Whether the process-wide backend has been initialized successfully.
pub fn is_initialized() -> bool {
    BLAS.get().is_some()
}
/// Immutable configuration applied before loading a model.
pub fn aligned_buffers() -> bool {
    BLAS.get().map(|b| b.aligned_buffers).unwrap_or(false)
}
/// Configuration and provenance of the initialized native backend.
pub fn backend_info() -> Option<&'static str> {
    BLAS.get().map(|b| b.info.as_str())
}
/// Explicit process-wide settings; the reusable crate does not read or mutate
/// model-specific environment variables, affinity, or OpenMP wait policy.
/// Explicit GEMM partition strategy; independent of model environment variables.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(i32)]
pub enum GemmPartition {
    /// Split rows into 16-row tasks (the generic default).
    #[default]
    Rows = 0,
    /// Preserve upstream adaptive M/N partitioning for sufficiently large GEMMs.
    Auto = 1,
}
#[derive(Clone, Copy, Debug)]
pub struct BackendOptions {
    pub threads: i32,
    pub partition: GemmPartition,
    pub aligned_buffers: bool,
    pub residual_fusion: bool,
    pub debug_alignment: bool,
}
impl Default for BackendOptions {
    fn default() -> Self {
        Self {
            threads: 1,
            partition: GemmPartition::Rows,
            aligned_buffers: false,
            residual_fusion: false,
            debug_alignment: false,
        }
    }
}
/// Load the pinned standalone ABI once per process. Failed duplicate/invalid budgets
/// never change an active backend. All calls/models share its immutable thread budget.
///
/// # Safety
/// `path` must be a trusted native library implementing standalone ABI version 2,
/// built from the documented MLAS revision for this host CPU/OS. Its symbols must
/// honor the declared FP32 pointer/length contracts and permit concurrent calls.
/// Loading arbitrary machine code cannot be made safe by inspecting symbol names.
pub unsafe fn initialize(path: &str, threads: i32) -> anyhow::Result<String> {
    initialize_with_options(
        path,
        BackendOptions {
            threads,
            ..Default::default()
        },
    )
}
/// Initialize with explicit settings, under the same safety contract as `initialize`.
/// # Safety
/// See `initialize`: the host supplies and qualifies trusted ABI-compatible code.
pub unsafe fn initialize_with_options(
    path: &str,
    options: BackendOptions,
) -> anyhow::Result<String> {
    let threads = options.threads;
    anyhow::ensure!(
        (1..=1024).contains(&threads),
        "invalid BLAS thread count: expected 1..=1024"
    );
    let _guard = BLAS_INIT
        .lock()
        .map_err(|_| anyhow::anyhow!("BLAS initialization lock poisoned"))?;
    anyhow::ensure!(BLAS.get().is_none(), "BLAS already initialized");
    let lib = Library::new(path)?;
    let abi = *lib.get::<unsafe extern "C" fn() -> u32>(b"candle_mlas_abi_version\0")?;
    anyhow::ensure!(abi() == 2, "unsupported standalone MLAS ABI version");
    let config =
        *lib.get::<unsafe extern "C" fn() -> *const libc::c_char>(b"candle_mlas_get_config\0")?;
    let text = config();
    anyhow::ensure!(!text.is_null(), "null standalone MLAS configuration");
    let text = CStr::from_ptr(text).to_string_lossy();
    anyhow::ensure!(
        text.contains("f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7"),
        "unsupported standalone MLAS source revision"
    );
    let set = *lib.get::<unsafe extern "C" fn(i32)>(b"candle_mlas_set_num_threads\0")?;
    let get = *lib.get::<unsafe extern "C" fn() -> i32>(b"candle_mlas_get_num_threads\0")?;
    let set_partition = *lib.get::<unsafe extern "C" fn(i32)>(b"candle_mlas_set_partition\0")?;
    let get_partition =
        *lib.get::<unsafe extern "C" fn() -> i32>(b"candle_mlas_get_partition\0")?;
    let sgemm = *lib.get::<Sgemm>(b"cblas_sgemm\0")?;
    let packed = Some(PackApi {
        size: *lib.get::<PackSize>(b"cblas_sgemm_pack_get_size\0")?,
        pack: *lib.get::<Pack>(b"cblas_sgemm_pack\0")?,
        compute: *lib.get::<Compute>(b"cblas_sgemm_compute\0")?,
    });
    let geglu = lib.get::<GeGlu>(b"candle_mlas_geglu\0").ok().map(|v| *v);
    let batch = lib
        .get::<SgemmBatch>(b"candle_mlas_sgemm_batch\0")
        .ok()
        .map(|v| *v);
    let norm = lib
        .get::<Norm>(b"candle_mlas_layer_norm\0")
        .ok()
        .map(|v| *v);
    set(threads);
    anyhow::ensure!(get() == threads, "BLAS thread budget was not applied");
    set_partition(options.partition as i32);
    anyhow::ensure!(
        get_partition() == options.partition as i32,
        "BLAS partition was not applied"
    );
    let aligned_buffers = options.aligned_buffers;
    let gemm_residual = options.residual_fusion;
    let info=format!("{}; threads={threads}; partition={:?}; library={path}; aligned_buffers={aligned_buffers}; gemm_residual={gemm_residual}",text,options.partition);
    let backend = Blas {
        _library: lib,
        sgemm,
        info: info.clone(),
        packed,
        m_independent: true,
        geglu,
        batch,
        norm,
        aligned_buffers,
        gemm_residual,
        debug_alignment: options.debug_alignment,
    };
    BLAS.set(backend)
        .map_err(|_| anyhow::anyhow!("BLAS already initialized"))?;
    Ok(info)
}
struct OutputLayout {
    enabled: bool,
    offset: AtomicUsize,
}
impl OutputLayout {
    fn new() -> Self {
        Self {
            enabled: BLAS.get().map(|b| b.aligned_buffers).unwrap_or(false),
            offset: AtomicUsize::new(0),
        }
    }
    fn allocate(&self, count: usize) -> Result<Vec<f32>> {
        let length = count
            .checked_add(if self.enabled { 16 } else { 0 })
            .ok_or_else(|| Error::Msg("aligned output size overflow".into()))?;
        let out = vec![0.; length];
        let offset = if self.enabled {
            (64 - out.as_ptr() as usize % 64) % 64 / 4
        } else {
            0
        };
        self.offset.store(offset, Ordering::Relaxed);
        Ok(out)
    }
    fn allocate_from(&self, source: &[f32]) -> Result<Vec<f32>> {
        let length = source
            .len()
            .checked_add(if self.enabled { 16 } else { 0 })
            .ok_or_else(|| Error::Msg("residual output size overflow".into()))?;
        let mut out = Vec::<f32>::with_capacity(length);
        let offset = if self.enabled {
            (64 - out.as_ptr() as usize % 64) % 64 / 4
        } else {
            0
        };
        self.offset.store(offset, Ordering::Relaxed);
        out.resize(offset, 0.);
        out.extend_from_slice(source);
        out.resize(length, 0.);
        Ok(out)
    }
    fn pointer(&self, out: &mut [f32]) -> *mut f32 {
        out.as_mut_ptr()
            .wrapping_add(self.offset.load(Ordering::Relaxed))
    }
    fn stored_shape(&self, logical: Shape, out: &[f32]) -> Shape {
        if self.enabled {
            Shape::from((out.len(),))
        } else {
            logical
        }
    }
    fn finish(&self, value: Tensor, logical: Shape) -> Result<Tensor> {
        if self.enabled {
            value
                .narrow(0, self.offset.load(Ordering::Relaxed), logical.elem_count())?
                .reshape(logical)
        } else {
            Ok(value)
        }
    }
}

pub fn gemm_residual_enabled() -> bool {
    BLAS.get().map(|b| b.gemm_residual).unwrap_or(false)
}
struct Matmul;
impl CustomOp2 for Matmul {
    fn name(&self) -> &'static str {
        "candle-mlas-matmul-f32"
    }
    fn cpu_fwd(
        &self,
        a: &CpuStorage,
        la: &Layout,
        b: &CpuStorage,
        lb: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let av = a.as_slice::<f32>()?;
        let bv = b.as_slice::<f32>()?;
        let da = la.dims();
        let db = lb.dims();
        let r = da.len();
        if r < 2 || r != db.len() || da[..r - 2] != db[..r - 2] || da[r - 1] != db[r - 2] {
            return Err(Error::Msg("CBLAS unsupported batch/shape".into()));
        }
        let (m, k, n) = (da[r - 2], da[r - 1], db[r - 1]);
        if [m, k, n].iter().any(|&d| d == 0 || d > i32::MAX as usize) {
            return Err(Error::Msg("CBLAS invalid dimensions".into()));
        }
        let matrix = |l: &Layout| -> Result<(i32, i32)> {
            let d = l.dims();
            let s = l.stride();
            let r = d.len();
            if s[r - 1] == 1 && s[r - 2] >= d[r - 1] {
                Ok((111, s[r - 2].try_into().map_err(Error::wrap)?))
            } else if s[r - 2] == 1 && s[r - 1] >= d[r - 2] {
                Ok((112, s[r - 1].try_into().map_err(Error::wrap)?))
            } else {
                Err(Error::Msg(
                    "CBLAS requires contiguous or transposed matrix strides".into(),
                ))
            }
        };
        let (ta, lda) = matrix(la)?;
        let (tb, ldb) = matrix(lb)?;
        let batch = da[..r - 2]
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| Error::Msg("CBLAS batch size overflow".into()))?;
        let count = batch
            .checked_mul(m)
            .and_then(|v| v.checked_mul(n))
            .ok_or_else(|| Error::Msg("CBLAS output size overflow".into()))?;
        let mut out = vec![0f32; count];
        let offset = |l: &Layout, mut bi: usize| {
            let mut off = l.start_offset();
            for j in (0..r - 2).rev() {
                let idx = bi % l.dims()[j];
                bi /= l.dims()[j];
                off += idx * l.stride()[j];
            }
            off
        };
        let backend = BLAS
            .get()
            .ok_or_else(|| Error::Msg("CBLAS was not initialized".into()))?;
        if let Some(fun) = backend.batch.filter(|_| batch > 1) {
            let (oa0, ob0) = (offset(la, 0), offset(lb, 0));
            let (sa, sb) = (offset(la, 1) - oa0, offset(lb, 1) - ob0);
            let mut regular = true;
            for bi in 0..batch {
                let (oa, ob) = (offset(la, bi), offset(lb, bi));
                let amax = oa + (m - 1) * la.stride()[r - 2] + (k - 1) * la.stride()[r - 1];
                let bmax = ob + (k - 1) * lb.stride()[r - 2] + (n - 1) * lb.stride()[r - 1];
                if amax >= av.len() || bmax >= bv.len() {
                    return Err(Error::Msg("batched CBLAS operand bounds".into()));
                }
                regular &= oa == oa0 + bi * sa && ob == ob0 + bi * sb;
            }
            if regular {
                unsafe {
                    fun(
                        ta,
                        tb,
                        batch,
                        m,
                        n,
                        k,
                        av.as_ptr().add(oa0),
                        lda as usize,
                        sa,
                        bv.as_ptr().add(ob0),
                        ldb as usize,
                        sb,
                        out.as_mut_ptr(),
                        n,
                        m * n,
                    );
                }
                let mut shape = da[..r - 2].to_vec();
                shape.extend([m, n]);
                return Ok((CpuStorage::F32(out), Shape::from(shape)));
            }
        }
        let gemm = backend.sgemm;
        for bi in 0..batch {
            let oa = offset(la, bi);
            let ob = offset(lb, bi);
            let amax = oa + (m - 1) * la.stride()[r - 2] + (k - 1) * la.stride()[r - 1];
            let bmax = ob + (k - 1) * lb.stride()[r - 2] + (n - 1) * lb.stride()[r - 1];
            if amax >= av.len() || bmax >= bv.len() {
                return Err(Error::Msg("CBLAS operand bounds".into()));
            }
            // Row-major CBLAS with validated strides, lengths and 32-bit dimensions.
            unsafe {
                gemm(
                    101,
                    ta,
                    tb,
                    m as i32,
                    n as i32,
                    k as i32,
                    1.,
                    av.as_ptr().add(oa),
                    lda,
                    bv.as_ptr().add(ob),
                    ldb,
                    0.,
                    out.as_mut_ptr().add(bi * m * n),
                    n as i32,
                );
            }
        }
        let mut shape = da[..r - 2].to_vec();
        shape.extend([m, n]);
        Ok((CpuStorage::F32(out), Shape::from(shape)))
    }
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.apply_op2_no_bwd(b, &Matmul)
}
pub fn geglu(x: &Tensor) -> Result<Tensor> {
    let op = FusedGeGlu {
        output: OutputLayout::new(),
    };
    let mut dims = x.dims().to_vec();
    *dims
        .last_mut()
        .ok_or_else(|| Error::Msg("GeGLU rank".into()))? /= 2;
    let value = x.apply_op1_no_bwd(&op)?;
    op.output.finish(value, Shape::from(dims))
}
struct FusedGeGlu {
    output: OutputLayout,
}
impl CustomOp1 for FusedGeGlu {
    fn name(&self) -> &'static str {
        "vela-mlas-geglu-erf"
    }
    fn cpu_fwd(&self, x: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let data = x.as_slice::<f32>()?;
        let (start, end) = l
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("GeGLU needs contiguous input".into()))?;
        let mut dims = l.dims().to_vec();
        let channels = *dims.last().ok_or_else(|| Error::Msg("GeGLU rank".into()))?;
        if channels == 0 || channels % 2 != 0 {
            return Err(Error::Msg("GeGLU channels".into()));
        }
        let width = channels / 2;
        let rows = (end - start) / channels;
        *dims.last_mut().unwrap() = width;
        let mut output = self.output.allocate(rows * width)?;
        let fun = BLAS
            .get()
            .and_then(|b| b.geglu)
            .ok_or_else(|| Error::Msg("fused GeGLU requires the standalone MLAS bridge".into()))?;
        unsafe {
            fun(
                data.as_ptr().add(start),
                self.output.pointer(&mut output),
                rows,
                width,
            );
        }
        let shape = self.output.stored_shape(Shape::from(dims), &output);
        Ok((CpuStorage::F32(output), shape))
    }
}

pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, epsilon: f32) -> Result<Tensor> {
    if !epsilon.is_finite() || epsilon <= 0. {
        return Err(Error::Msg(
            "normalization epsilon must be finite and positive".into(),
        ));
    }
    let op = FusedNorm {
        epsilon,
        output: OutputLayout::new(),
    };
    let value = x.apply_op3_no_bwd(weight, bias, &op)?;
    op.output.finish(value, x.shape().clone())
}
struct FusedNorm {
    epsilon: f32,
    output: OutputLayout,
}
impl CustomOp3 for FusedNorm {
    fn name(&self) -> &'static str {
        "vela-layer-norm-f32"
    }
    fn cpu_fwd(
        &self,
        x: &CpuStorage,
        l: &Layout,
        w: &CpuStorage,
        lw: &Layout,
        b: &CpuStorage,
        lb: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let width = *l
            .dims()
            .last()
            .ok_or_else(|| Error::Msg("norm rank".into()))?;
        if width == 0 || lw.dims() != [width] || lb.dims() != [width] {
            return Err(Error::Msg("norm weight shape".into()));
        }
        let (start, end) = l
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("norm input strides".into()))?;
        let (ws, _) = lw
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("norm weight strides".into()))?;
        let (bs, _) = lb
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("norm bias strides".into()))?;
        let mut output = self.output.allocate(end - start)?;
        let fun = BLAS
            .get()
            .and_then(|v| v.norm)
            .ok_or_else(|| Error::Msg("native norm requires MLAS bridge".into()))?;
        unsafe {
            fun(
                x.as_slice::<f32>()?.as_ptr().add(start),
                w.as_slice::<f32>()?.as_ptr().add(ws),
                b.as_slice::<f32>()?.as_ptr().add(bs),
                self.output.pointer(&mut output),
                (end - start) / width,
                width,
                self.epsilon,
            );
        }
        let shape = self.output.stored_shape(l.shape().clone(), &output);
        Ok((CpuStorage::F32(output), shape))
    }
}
/// Own one packed weight buffer. Packing is M-independent; no original weight tensor is retained.
/// The validated bridge uses the documented CBLAS packed-B constants.
pub struct PackedWeight {
    pointer: std::ptr::NonNull<f32>,
    layout: std::alloc::Layout,
    m: usize,
    n: usize,
    k: usize,
    m_independent: bool,
}
unsafe impl Send for PackedWeight {}
// PackedWeight B is immutable after construction; each GEMM owns its output buffer.
unsafe impl Sync for PackedWeight {}
pub fn m_independent() -> bool {
    BLAS.get().map(|b| b.m_independent).unwrap_or(false)
}
impl Drop for PackedWeight {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.pointer.as_ptr().cast(), self.layout);
        }
    }
}
impl PackedWeight {
    pub fn output_storage_bytes(&self, m: usize) -> Result<usize> {
        if m == 0 || m > i32::MAX as usize {
            return Err(Error::Msg("workspace GEMM row count".into()));
        }
        m.checked_mul(self.n)
            .and_then(|v| {
                v.checked_add(if BLAS.get().map(|b| b.aligned_buffers).unwrap_or(false) {
                    16
                } else {
                    0
                })
            })
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| Error::Msg("workspace GEMM output size overflow".into()))
    }
    pub fn initialized_output(&self, m: usize) -> Result<Tensor> {
        self.output_storage_bytes(m)?;
        let layout = OutputLayout::new();
        let values = layout.allocate(m * self.n)?;
        let shape = layout.stored_shape(Shape::from((m, self.n)), &values);
        let tensor = Tensor::from_vec(values, shape, &candle_core::Device::Cpu)?;
        layout.finish(tensor, Shape::from((m, self.n)))
    }
    /// Write into inference-owned initialized output storage.
    /// # Safety
    /// Caller must own this scratch for the entire operation and must not expose
    /// snapshots as immutable outputs while this storage can be overwritten.
    pub unsafe fn forward_into(&self, input: &Tensor, output: &Tensor) -> Result<()> {
        self.write_into(input, output, 0.)
    }
    /// Accumulate into inference-owned initialized residual storage.
    /// # Safety
    /// Same exclusive scratch/no-escaping-snapshot contract as `forward_into`.
    pub unsafe fn forward_residual_into(&self, input: &Tensor, output: &Tensor) -> Result<()> {
        self.write_into(input, output, 1.)
    }
    fn write_into(&self, input: &Tensor, output: &Tensor, beta: f32) -> Result<()> {
        // Reject shared backing storage before taking a destination write lock.
        // The public in-place API would otherwise block on its own read lock.
        {
            let (source, _) = input.storage_and_layout();
            let (dest, _) = output.storage_and_layout();
            match (&*source, &*dest) {
                (
                    candle_core::Storage::Cpu(CpuStorage::F32(a)),
                    candle_core::Storage::Cpu(CpuStorage::F32(b)),
                ) => {
                    if std::ptr::eq(a.as_ptr(), b.as_ptr()) {
                        return Err(Error::Msg("workspace GEMM input aliases output".into()));
                    }
                }
                _ => return Err(Error::Msg("workspace GEMM needs CPU FP32 storage".into())),
            }
        }
        output.inplace_op2(input, &PackedWeightInto { packed: self, beta })
    }
    pub fn new(weight: &Tensor) -> Result<Self> {
        let m = 1;
        let (n, k) = weight.dims2()?;
        if [m, n, k].iter().any(|&v| v == 0 || v > i32::MAX as usize) {
            return Err(Error::Msg("packed GEMM invalid dimensions".into()));
        }
        let api = BLAS.get().and_then(|b| b.packed.as_ref()).ok_or_else(|| {
            Error::Msg("packed SGEMM requires the initialized standalone MLAS ABI".into())
        })?;
        let data = weight.flatten_all()?.to_vec1::<f32>()?;
        let bytes = unsafe { (api.size)(162, m as i32, n as i32, k as i32) };
        if bytes == 0 || bytes > 1024 * 1024 * 1024 {
            return Err(Error::Msg("packed GEMM invalid buffer size".into()));
        }
        let layout = std::alloc::Layout::from_size_align(bytes, 64).map_err(Error::wrap)?;
        let pointer = std::ptr::NonNull::new(unsafe { std::alloc::alloc(layout) }.cast::<f32>())
            .ok_or_else(|| Error::Msg("packed GEMM allocation failed".into()))?;
        let out = Self {
            pointer,
            layout,
            m,
            n,
            k,
            m_independent: m_independent(),
        };
        // Pack transposed [out,in] weights as B for row-major X @ W.T.
        unsafe {
            (api.pack)(
                101,
                162,
                112,
                m as i32,
                n as i32,
                k as i32,
                1.,
                data.as_ptr(),
                k as i32,
                out.pointer.as_ptr(),
            );
        }
        Ok(out)
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (m, _) = x.dims2()?;
        let op = PackedWeightMatmul {
            packed: self,
            output: OutputLayout::new(),
        };
        let value = x.apply_op1_no_bwd(&op)?;
        op.output.finish(value, Shape::from((m, self.n)))
    }
    pub fn forward_residual(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        let (m, _) = x.dims2()?;
        let op = PackedWeightResidual {
            packed: self,
            output: OutputLayout::new(),
        };
        let value = x.apply_op2_no_bwd(residual, &op)?;
        op.output.finish(value, Shape::from((m, self.n)))
    }
    pub fn n(&self) -> usize {
        self.n
    }
}
struct PackedWeightInto<'a> {
    packed: &'a PackedWeight,
    beta: f32,
}
impl InplaceOp2 for PackedWeightInto<'_> {
    fn name(&self) -> &'static str {
        "vela-initialized-workspace-gemm"
    }
    fn cpu_fwd(
        &self,
        output: &mut CpuStorage,
        lo: &Layout,
        input: &CpuStorage,
        li: &Layout,
    ) -> Result<()> {
        let p = self.packed;
        let (m, k) = li.shape().dims2()?;
        if m == 0
            || m > i32::MAX as usize
            || k != p.k
            || (!p.m_independent && m != p.m)
            || lo.dims() != [m, p.n]
        {
            return Err(Error::Msg("workspace GEMM shape".into()));
        }
        let (start, end) = li
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("workspace GEMM input strides".into()))?;
        let (out_start, out_end) = lo
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("workspace GEMM output strides".into()))?;
        let data = input.as_slice::<f32>()?;
        let values = match output {
            CpuStorage::F32(values) => values,
            _ => return Err(Error::Msg("workspace GEMM output dtype".into())),
        };
        if end - start != m * p.k
            || out_end - out_start != m * p.n
            || end > data.len()
            || out_end > values.len()
        {
            return Err(Error::Msg("workspace GEMM storage bounds".into()));
        }
        let api = BLAS
            .get()
            .and_then(|b| b.packed.as_ref())
            .ok_or_else(|| Error::Msg("workspace GEMM missing packed API".into()))?;
        let compute = projection_timer();
        // The destination remains initialized. beta=0 overwrites scratch;
        // beta=1 updates a private per-forward residual activation in place.
        unsafe {
            (api.compute)(
                101,
                111,
                151,
                m as i32,
                p.n as i32,
                p.k as i32,
                data.as_ptr().add(start),
                p.k as i32,
                p.pointer.as_ptr(),
                p.n as i32,
                self.beta,
                values.as_mut_ptr().add(out_start),
                p.n as i32,
            );
        }
        record_projection(p, m, self.beta == 1., 0, Some(0.), compute);
        Ok(())
    }
}
struct PackedWeightMatmul<'a> {
    packed: &'a PackedWeight,
    output: OutputLayout,
}
impl CustomOp1 for PackedWeightMatmul<'_> {
    fn name(&self) -> &'static str {
        "candle-mlas-packed-f32"
    }
    fn cpu_fwd(&self, x: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let p = self.packed;
        let data = x.as_slice::<f32>()?;
        let (start, end) = l
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("packed GEMM requires contiguous input".into()))?;
        let (m, k) = l.shape().dims2()?;
        if m == 0
            || m > i32::MAX as usize
            || k != p.k
            || (!p.m_independent && m != p.m)
            || m.checked_mul(p.k) != Some(end - start)
            || end > data.len()
        {
            return Err(Error::Msg("packed GEMM input shape".into()));
        }
        let allocate = projection_timer();
        let mut out = self.output.allocate(m * p.n)?;
        let allocation = allocate.map(|t| t.elapsed().as_secs_f64());
        let api = BLAS
            .get()
            .and_then(|b| b.packed.as_ref())
            .ok_or_else(|| Error::Msg("missing packed GEMM API".into()))?;
        if BLAS.get().map(|b| b.debug_alignment).unwrap_or(false) {
            eprintln!(
                "gemm_alignment m={m} n={} k={} A_mod64={} B_mod64={} C_mod64={}",
                p.n,
                p.k,
                data.as_ptr().wrapping_add(start) as usize % 64,
                p.pointer.as_ptr() as usize % 64,
                self.output.pointer(&mut out) as usize % 64
            );
        }
        let compute = projection_timer();
        unsafe {
            (api.compute)(
                101,
                111,
                151,
                m as i32,
                p.n as i32,
                p.k as i32,
                data.as_ptr().add(start),
                p.k as i32,
                p.pointer.as_ptr(),
                p.n as i32,
                0.,
                self.output.pointer(&mut out),
                p.n as i32,
            );
        }
        record_projection(p, m, false, out.len() * 4, allocation, compute);
        let shape = self.output.stored_shape(Shape::from((m, p.n)), &out);
        Ok((CpuStorage::F32(out), shape))
    }
}

struct PackedWeightResidual<'a> {
    packed: &'a PackedWeight,
    output: OutputLayout,
}
impl CustomOp2 for PackedWeightResidual<'_> {
    fn name(&self) -> &'static str {
        "candle-mlas-residual-f32"
    }
    fn cpu_fwd(
        &self,
        x: &CpuStorage,
        l: &Layout,
        residual: &CpuStorage,
        lr: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let p = self.packed;
        let (m, k) = l.shape().dims2()?;
        if m == 0
            || m > i32::MAX as usize
            || k != p.k
            || (!p.m_independent && m != p.m)
            || lr.dims() != [m, p.n]
        {
            return Err(Error::Msg("residual GEMM shape".into()));
        }
        let (start, end) = l
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("residual GEMM input strides".into()))?;
        let (rs, re) = lr
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("residual GEMM residual strides".into()))?;
        if end - start != m * p.k || re - rs != m * p.n {
            return Err(Error::Msg("residual GEMM input size".into()));
        }
        // Safe initialized copy: no uninitialized output is exposed to Rust or C.
        let allocate = projection_timer();
        let mut out = self
            .output
            .allocate_from(&residual.as_slice::<f32>()?[rs..re])?;
        let allocation = allocate.map(|t| t.elapsed().as_secs_f64());
        let api = BLAS
            .get()
            .and_then(|b| b.packed.as_ref())
            .ok_or_else(|| Error::Msg("missing packed GEMM API".into()))?;
        let compute = projection_timer();
        unsafe {
            (api.compute)(
                101,
                111,
                151,
                m as i32,
                p.n as i32,
                p.k as i32,
                x.as_slice::<f32>()?.as_ptr().add(start),
                p.k as i32,
                p.pointer.as_ptr(),
                p.n as i32,
                1.,
                self.output.pointer(&mut out),
                p.n as i32,
            );
        }
        record_projection(p, m, true, out.len() * 4, allocation, compute);
        let shape = self.output.stored_shape(Shape::from((m, p.n)), &out);
        Ok((CpuStorage::F32(out), shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_budget_and_missing_library_fail_before_backend_publication() {
        for budget in [-1, 0, 1025, i32::MAX] {
            let error = unsafe { initialize("/nonexistent/candle-mlas-test", budget) }.unwrap_err();
            assert!(error.to_string().contains("invalid BLAS thread count"));
        }
        assert!(unsafe { initialize("/nonexistent/candle-mlas-test", 1) }.is_err());
        assert!(!is_initialized());
        #[cfg(target_os = "linux")]
        assert!(unsafe { initialize("libm.so.6", 1) }
            .unwrap_err()
            .to_string()
            .contains("candle_mlas_abi_version"));
        assert!(!is_initialized());
    }
    #[test]
    fn rejected_shapes_dtypes_and_epsilon_do_not_enter_native_code() {
        let x = Tensor::zeros((2, 3), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        let bad =
            Tensor::zeros((2, 4), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        assert!(matmul(&x, &bad).is_err());
        let half =
            Tensor::zeros((3, 4), candle_core::DType::F16, &candle_core::Device::Cpu).unwrap();
        assert!(matmul(&x, &half).is_err());
        for epsilon in [0., -1., f32::NAN, f32::INFINITY] {
            assert!(layer_norm(&x, &bad, &bad, epsilon).is_err());
        }
    }
    #[test]
    fn aligned_outputs_retain_allocator_ownership_and_initialized_padding() {
        for count in [1, 15, 16, 17, 64, 77, 512] {
            let data: Vec<f32> = (0..count).map(|i| i as f32 - 11.).collect();
            let layout = OutputLayout {
                enabled: true,
                offset: AtomicUsize::new(0),
            };
            let mut buffer = layout.allocate_from(&data).unwrap();
            let offset = layout.offset.load(Ordering::Relaxed);
            assert_eq!(layout.pointer(&mut buffer) as usize % 64, 0);
            assert_eq!(&buffer[offset..offset + count], data.as_slice());
            assert!(buffer[..offset]
                .iter()
                .chain(&buffer[offset + count..])
                .all(|&v| v == 0.));
            let tensor =
                Tensor::from_vec(buffer, (count + 16,), &candle_core::Device::Cpu).unwrap();
            assert_eq!(
                layout
                    .finish(tensor, Shape::from((count,)))
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                data
            );
        }
    }
    #[test]
    #[ignore = "requires qualified native library; run separately for Rows and Auto"]
    fn explicit_partition_ignores_vela_environment() {
        let env = rt_core::env::TestOverrides::from_env().unwrap();
        let library_path = env
            .candle_mlas_library
            .expect("set CANDLE_TEST_MLAS_LIBRARY");
        let path = library_path.to_str().expect("MLAS ABI uses a Unicode path");
        let partition = if env.candle_partition_auto.unwrap_or(false) {
            GemmPartition::Auto
        } else {
            GemmPartition::Rows
        };
        let contradictory = if partition == GemmPartition::Auto {
            "rows"
        } else {
            "auto"
        };
        unsafe { std::env::set_var("VELA_MLAS_PARTITION", contradictory) };
        unsafe {
            initialize_with_options(
                path,
                BackendOptions {
                    threads: 2,
                    partition,
                    ..Default::default()
                },
            )
        }
        .unwrap();
        let lib = unsafe { Library::new(path) }.unwrap();
        let get = unsafe {
            *lib.get::<unsafe extern "C" fn() -> i32>(b"candle_mlas_get_partition\0")
                .unwrap()
        };
        assert_eq!(unsafe { get() }, partition as i32);
        let weight = Tensor::ones(
            (128, 128),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let packed = PackedWeight::new(&weight).unwrap();
        let x = Tensor::ones(
            (64, 128),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let before = packed.forward(&x).unwrap().to_vec2::<f32>().unwrap();
        unsafe {
            std::env::set_var(
                "VELA_MLAS_PARTITION",
                if contradictory == "auto" {
                    "rows"
                } else {
                    "auto"
                },
            )
        };
        let after = packed.forward(&x).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(before, after);
        assert!(after.iter().flatten().all(|&v| v == 128.));
        assert_eq!(unsafe { get() }, partition as i32);
    }
    #[test]
    #[ignore = "requires CANDLE_TEST_MLAS_LIBRARY; explicitly run on qualified CI/VM builds"]
    fn immutable_packed_weights_support_different_shapes_retained_outputs_and_concurrent_calls() {
        let env = rt_core::env::TestOverrides::from_env().unwrap();
        let library_path = env
            .candle_mlas_library
            .expect("set CANDLE_TEST_MLAS_LIBRARY");
        let path = library_path.to_str().expect("MLAS ABI uses a Unicode path");
        unsafe { initialize(path, 2) }.unwrap();
        assert!(unsafe { initialize(path, 4) }
            .unwrap_err()
            .to_string()
            .contains("already initialized"));
        let weight = Tensor::from_vec(
            vec![1f32, 2., -3., 1., 0.5, -2.],
            (3, 2),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let packed = std::sync::Arc::new(PackedWeight::new(&weight).unwrap());
        drop(weight);
        let x =
            Tensor::from_vec(vec![1f32, 2., -1., 3.], (2, 2), &candle_core::Device::Cpu).unwrap();
        let first = packed.forward(&x).unwrap();
        let retained = first.to_vec2::<f32>().unwrap();
        let other = Tensor::from_vec(vec![2f32, -1.], (1, 2), &candle_core::Device::Cpu).unwrap();
        assert_eq!(
            packed.forward(&other).unwrap().to_vec2::<f32>().unwrap(),
            vec![vec![0., -7., 3.]]
        );
        assert_eq!(first.to_vec2::<f32>().unwrap(), retained);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let p = packed.clone();
                let x = x.clone();
                let expected = retained.clone();
                scope.spawn(move || {
                    for _ in 0..20 {
                        assert_eq!(p.forward(&x).unwrap().to_vec2::<f32>().unwrap(), expected);
                    }
                });
            }
        });
        let scratch = packed.initialized_output(2).unwrap();
        unsafe { packed.forward_into(&x, &scratch) }.unwrap();
        assert_eq!(scratch.to_vec2::<f32>().unwrap(), retained);
        let alias = scratch
            .flatten_all()
            .unwrap()
            .narrow(0, 0, 4)
            .unwrap()
            .reshape((2, 2))
            .unwrap();
        assert!(unsafe { packed.forward_into(&alias, &scratch) }
            .unwrap_err()
            .to_string()
            .contains("aliases output"));
        assert_eq!(scratch.to_vec2::<f32>().unwrap(), retained);
        let bad =
            Tensor::zeros((1, 3), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        assert!(packed.forward(&bad).is_err());
    }
}
