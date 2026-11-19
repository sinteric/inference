//! Vela-specific fused QKV/RoPE, local-mask softmax and head layout.
//! Model-independent numerical operations live in the `candle-mlas` crate.
use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, Error, InplaceOp1, InplaceOp2, Layout, Result,
    Shape, Tensor,
};
pub(crate) use candle_mlas::{
    geglu, gemm_residual_enabled, layer_norm, m_independent, matmul, PackedWeight as Packed,
    ProfileScope,
};
pub use candle_mlas::{projection_allocation_profile, ProjectionAllocation};
use libloading::Library;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
type QkvRope = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const f32,
    *mut f32,
    usize,
    usize,
    usize,
    usize,
    f32,
);
type MaskSoftmax = unsafe extern "C" fn(
    *const f32,
    *const u32,
    *mut f32,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    i32,
);
type UnpackHeads = unsafe extern "C" fn(*const f32, *mut f32, usize, usize, usize, usize);
struct VelaKernels {
    _library: Library,
    qkv: Option<QkvRope>,
    softmax: Option<MaskSoftmax>,
    unpack: Option<UnpackHeads>,
}
static KERNELS: OnceLock<VelaKernels> = OnceLock::new();
static INIT: Mutex<()> = Mutex::new(());
pub fn is_initialized() -> bool {
    KERNELS.get().is_some() && candle_mlas::is_initialized()
}
/// Initialize the Vela extension and shared model-independent MLAS backend once.
/// # Safety
/// `path` must be the trusted qualified Vela bridge built by `build_mlas.sh`,
/// with its documented ABI/source revision and CPU-compatible native code.
pub unsafe fn initialize(path: &str, threads: i32) -> anyhow::Result<String> {
    initialize_with_options(
        path,
        candle_mlas::BackendOptions {
            threads,
            ..Default::default()
        },
    )
}

/// Explicit experiment options; never reads process environment.
/// # Safety
/// Same qualified native-library contract as [`initialize`].
#[doc(hidden)]
pub unsafe fn initialize_with_options(
    path: &str,
    options: candle_mlas::BackendOptions,
) -> anyhow::Result<String> {
    let threads = options.threads;
    anyhow::ensure!(
        (1..=1024).contains(&threads),
        "invalid BLAS thread count: expected 1..=1024"
    );
    let _guard = INIT
        .lock()
        .map_err(|_| anyhow::anyhow!("Vela kernel initialization lock poisoned"))?;
    anyhow::ensure!(KERNELS.get().is_none(), "BLAS already initialized");
    let lib = Library::new(path)?;
    let qkv = lib.get::<QkvRope>(b"vela_mlas_qkv_rope\0").ok().map(|v| *v);
    let softmax = lib
        .get::<MaskSoftmax>(b"vela_mlas_masked_softmax_u32\0")
        .ok()
        .map(|v| *v);
    let unpack = lib
        .get::<UnpackHeads>(b"vela_mlas_unpack_heads\0")
        .ok()
        .map(|v| *v);
    anyhow::ensure!(
        qkv.is_some() && softmax.is_some() && unpack.is_some(),
        "Vela bridge is missing fused kernel symbols"
    );
    let info = candle_mlas::initialize_with_options(path, options)?;
    KERNELS
        .set(VelaKernels {
            _library: lib,
            qkv,
            softmax,
            unpack,
        })
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
            enabled: candle_mlas::aligned_buffers(),
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
    #[cfg(test)]
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

pub fn qkv_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, heads: usize) -> Result<Tensor> {
    x.apply_op3_no_bwd(cos, sin, &FusedQkv { heads })
}
pub(crate) fn qkv_rope_shape(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    heads: usize,
) -> Result<Vec<usize>> {
    let (b, s, width) = x.dims3()?;
    let groups = heads
        .checked_mul(3)
        .filter(|&n| n > 0)
        .ok_or_else(|| Error::Msg("QKV heads overflow or zero".into()))?;
    if b == 0 || s == 0 || width == 0 || width % groups != 0 {
        return Err(Error::Msg("QKV channels".into()));
    }
    let dim = width / groups;
    if dim % 2 != 0
        || cos.dims() != sin.dims()
        || cos.dims().len() != 2
        || cos.dims()[0] < s
        || cos.dims()[1] != dim / 2
    {
        return Err(Error::Msg("QKV RoPE table shape".into()));
    }
    Ok(vec![3, b, heads, s, dim])
}
/// Internal initialized scratch, used only while its forward owns the lease.
pub(crate) fn qkv_rope_into(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    heads: usize,
    output: &Tensor,
) -> Result<()> {
    let expected = qkv_rope_shape(x, cos, sin, heads)?;
    if output.dims() != expected {
        return Err(Error::Msg("QKV workspace output shape".into()));
    }
    let (cos_storage, lc) = cos.storage_and_layout();
    let (sin_storage, ls) = sin.storage_and_layout();

    let cosine = cpu_f32_storage(&cos_storage)?;
    let sine = cpu_f32_storage(&sin_storage)?;
    let (cs, ce) = lc
        .contiguous_offsets()
        .ok_or_else(|| Error::Msg("QKV cosine strides".into()))?;
    let (ss, se) = ls
        .contiguous_offsets()
        .ok_or_else(|| Error::Msg("QKV sine strides".into()))?;
    if ce > cosine.len() || se > sine.len() {
        return Err(Error::Msg("QKV table storage bounds".into()));
    }
    {
        // Check every read operand before taking the destination write lock.
        // Table guards remain read-locked throughout the native call.
        let (source, _) = x.storage_and_layout();
        let (destination, _) = output.storage_and_layout();
        let source = cpu_f32_storage(&source)?;
        let destination = cpu_f32_storage(&destination)?;
        if [source.as_ptr(), cosine.as_ptr(), sine.as_ptr()].contains(&destination.as_ptr()) {
            return Err(Error::Msg(
                "QKV workspace read operand aliases output".into(),
            ));
        }
    }
    output.inplace_op2(
        x,
        &QkvInto {
            heads,
            cosine: &cosine[cs..ce],
            sine: &sine[ss..se],
            expected,
        },
    )
}
fn cpu_f32_storage(storage: &candle_core::Storage) -> Result<&[f32]> {
    match storage {
        candle_core::Storage::Cpu(CpuStorage::F32(values)) => Ok(values),
        _ => Err(Error::Msg("QKV workspace needs CPU FP32 storage".into())),
    }
}
struct QkvInto<'a> {
    heads: usize,
    cosine: &'a [f32],
    sine: &'a [f32],
    expected: Vec<usize>,
}
impl InplaceOp2 for QkvInto<'_> {
    fn name(&self) -> &'static str {
        "vela-initialized-qkv-rope-workspace"
    }
    fn cpu_fwd(
        &self,
        output: &mut CpuStorage,
        lo: &Layout,
        input: &CpuStorage,
        li: &Layout,
    ) -> Result<()> {
        let (b, s, width) = li.shape().dims3()?;
        let dim = self.expected[4];
        let count = b
            .checked_mul(s)
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| Error::Msg("QKV workspace size overflow".into()))?;
        let table_count = s
            .checked_mul(dim / 2)
            .ok_or_else(|| Error::Msg("QKV workspace table size overflow".into()))?;
        let (start, end) = li
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("QKV workspace input strides".into()))?;
        let (os, oe) = lo
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("QKV workspace output strides".into()))?;
        let source = input.as_slice::<f32>()?;
        let destination = match output {
            CpuStorage::F32(values) => values,
            _ => return Err(Error::Msg("QKV workspace output dtype".into())),
        };
        if lo.dims() != self.expected
            || end - start != count
            || oe - os != count
            || end > source.len()
            || oe > destination.len()
            || self.cosine.len() < table_count
            || self.sine.len() < table_count
        {
            return Err(Error::Msg("QKV workspace storage bounds".into()));
        }
        let fun = KERNELS
            .get()
            .and_then(|v| v.qkv)
            .ok_or_else(|| Error::Msg("QKV workspace requires MLAS bridge".into()))?;
        unsafe {
            fun(
                source.as_ptr().add(start),
                self.cosine.as_ptr(),
                self.sine.as_ptr(),
                destination.as_mut_ptr().add(os),
                b,
                s,
                self.heads,
                dim,
                (dim as f64).powf(-0.5) as f32,
            );
        }
        Ok(())
    }
}
pub fn masked_softmax(
    x: &Tensor,
    mask: &Tensor,
    query_start: usize,
    key_start: usize,
    radius: Option<usize>,
) -> Result<Tensor> {
    x.apply_op2_no_bwd(
        mask,
        &FusedSoftmax {
            query_start,
            key_start,
            radius,
        },
    )
}
/// The caller owns fresh native-GEMM score storage; no score tensor is returned by Model.
pub(crate) fn private_masked_softmax(
    scores: &Tensor,
    mask: &Tensor,
    query_start: usize,
    key_start: usize,
    radius: Option<usize>,
) -> Result<()> {
    let (batch, heads, queries, keys) = scores.dims4()?;
    let (mb, seq) = mask.dims2()?;
    if [batch, heads, queries, keys].contains(&0)
        || mb != batch
        || query_start
            .checked_add(queries)
            .map(|n| n > seq)
            .unwrap_or(true)
        || key_start.checked_add(keys).map(|n| n > seq).unwrap_or(true)
        || radius.map(|r| r > i32::MAX as usize).unwrap_or(false)
    {
        return Err(Error::Msg("private softmax mask shape/range".into()));
    }
    let (mask_storage, lm) = mask.storage_and_layout();
    let values = match &*mask_storage {
        candle_core::Storage::Cpu(v) => v.as_slice::<u32>()?,
        _ => return Err(Error::Msg("private softmax needs CPU mask".into())),
    };
    let (ms, me) = lm
        .contiguous_offsets()
        .ok_or_else(|| Error::Msg("private softmax mask strides".into()))?;
    let count = batch
        .checked_mul(seq)
        .ok_or_else(|| Error::Msg("private softmax mask overflow".into()))?;
    if me - ms != count || me > values.len() {
        return Err(Error::Msg("private softmax mask bounds".into()));
    }
    {
        let (storage, _) = scores.storage_and_layout();
        let data = cpu_f32_storage(&storage)?;
        if std::ptr::eq(data.as_ptr().cast::<u32>(), values.as_ptr()) {
            return Err(Error::Msg("private softmax mask aliases scores".into()));
        }
    }
    scores.inplace_op1(&PrivateSoftmax {
        mask: &values[ms..me],
        batch,
        heads,
        queries,
        keys,
        seq,
        query_start,
        key_start,
        radius,
    })
}
struct PrivateSoftmax<'a> {
    mask: &'a [u32],
    batch: usize,
    heads: usize,
    queries: usize,
    keys: usize,
    seq: usize,
    query_start: usize,
    key_start: usize,
    radius: Option<usize>,
}
impl InplaceOp1 for PrivateSoftmax<'_> {
    fn name(&self) -> &'static str {
        "vela-private-score-softmax"
    }
    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> Result<()> {
        if layout.dims() != [self.batch, self.heads, self.queries, self.keys] {
            return Err(Error::Msg("private softmax score shape".into()));
        }
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("private softmax score strides".into()))?;
        let count = self
            .batch
            .checked_mul(self.heads)
            .and_then(|n| n.checked_mul(self.queries))
            .and_then(|n| n.checked_mul(self.keys))
            .ok_or_else(|| Error::Msg("private softmax score overflow".into()))?;
        let values = match storage {
            CpuStorage::F32(values) => values,
            _ => return Err(Error::Msg("private softmax score dtype".into())),
        };
        if end - start != count || end > values.len() {
            return Err(Error::Msg("private softmax score bounds".into()));
        }
        let fun = KERNELS
            .get()
            .and_then(|b| b.softmax)
            .ok_or_else(|| Error::Msg("private softmax requires MLAS bridge".into()))?;
        let pointer = values.as_mut_ptr().wrapping_add(start);
        // Each native row reads an element before overwriting the same element,
        // then invokes the same already-in-place MLAS softmax. Rows are disjoint.
        unsafe {
            fun(
                pointer.cast_const(),
                self.mask.as_ptr(),
                pointer,
                self.batch,
                self.heads,
                self.queries,
                self.keys,
                self.seq,
                self.query_start,
                self.key_start,
                self.radius.map(|r| r as i32).unwrap_or(-1),
            );
        }
        Ok(())
    }
}
pub fn unpack_heads(x: &Tensor) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    let op = FusedUnpack {
        output: OutputLayout::new(),
    };
    let value = x.apply_op1_no_bwd(&op)?;
    op.output.finish(value, Shape::from((b, s, h * d)))
}
struct FusedUnpack {
    output: OutputLayout,
}
impl CustomOp1 for FusedUnpack {
    fn name(&self) -> &'static str {
        "vela-head-output-layout"
    }
    fn cpu_fwd(&self, x: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let (b, h, s, d) = l.shape().dims4()?;
        let (start, end) = l
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("head output strides".into()))?;
        let mut output = self.output.allocate(end - start)?;
        let fun = KERNELS
            .get()
            .and_then(|v| v.unpack)
            .ok_or_else(|| Error::Msg("head output layout requires MLAS bridge".into()))?;
        unsafe {
            fun(
                x.as_slice::<f32>()?.as_ptr().add(start),
                self.output.pointer(&mut output),
                b,
                h,
                s,
                d,
            );
        }
        let shape = self
            .output
            .stored_shape(Shape::from((b, s, h * d)), &output);
        Ok((CpuStorage::F32(output), shape))
    }
}
struct FusedQkv {
    heads: usize,
}
impl CustomOp3 for FusedQkv {
    fn name(&self) -> &'static str {
        "vela-qkv-layout-rope"
    }
    fn cpu_fwd(
        &self,
        x: &CpuStorage,
        l: &Layout,
        cos: &CpuStorage,
        lc: &Layout,
        sin: &CpuStorage,
        ls: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (b, s, width) = l.shape().dims3()?;
        if self.heads == 0 || width % (3 * self.heads) != 0 {
            return Err(Error::Msg("QKV channels".into()));
        }
        let dim = width / (3 * self.heads);
        if dim % 2 != 0
            || lc.dims() != ls.dims()
            || lc.dims().len() != 2
            || lc.dims()[0] < s
            || lc.dims()[1] != dim / 2
        {
            return Err(Error::Msg("QKV RoPE table shape".into()));
        }
        let (start, _) = l
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("QKV input strides".into()))?;
        let (cs, _) = lc
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("QKV cosine strides".into()))?;
        let (ss, _) = ls
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("QKV sine strides".into()))?;
        let mut output = vec![0.; b * s * width];
        let fun = KERNELS
            .get()
            .and_then(|v| v.qkv)
            .ok_or_else(|| Error::Msg("fused QKV requires MLAS bridge".into()))?;
        unsafe {
            fun(
                x.as_slice::<f32>()?.as_ptr().add(start),
                cos.as_slice::<f32>()?.as_ptr().add(cs),
                sin.as_slice::<f32>()?.as_ptr().add(ss),
                output.as_mut_ptr(),
                b,
                s,
                self.heads,
                dim,
                (dim as f64).powf(-0.5) as f32,
            );
        }
        Ok((
            CpuStorage::F32(output),
            Shape::from((3, b, self.heads, s, dim)),
        ))
    }
}
struct FusedSoftmax {
    query_start: usize,
    key_start: usize,
    radius: Option<usize>,
}
impl CustomOp2 for FusedSoftmax {
    fn name(&self) -> &'static str {
        "vela-masked-softmax"
    }
    fn cpu_fwd(
        &self,
        x: &CpuStorage,
        l: &Layout,
        mask: &CpuStorage,
        lm: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (b, h, q, k) = l.shape().dims4()?;
        let (mb, s) = lm.shape().dims2()?;
        if mb != b || self.query_start + q > s || self.key_start + k > s {
            return Err(Error::Msg("softmax mask shape".into()));
        }
        let (start, end) = l
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("softmax input strides".into()))?;
        let (ms, _) = lm
            .contiguous_offsets()
            .ok_or_else(|| Error::Msg("softmax mask strides".into()))?;
        let mut output = vec![0.; end - start];
        let fun = KERNELS
            .get()
            .and_then(|v| v.softmax)
            .ok_or_else(|| Error::Msg("fused softmax requires MLAS bridge".into()))?;
        unsafe {
            fun(
                x.as_slice::<f32>()?.as_ptr().add(start),
                mask.as_slice::<u32>()?.as_ptr().add(ms),
                output.as_mut_ptr(),
                b,
                h,
                q,
                k,
                s,
                self.query_start,
                self.key_start,
                self.radius.map(|v| v as i32).unwrap_or(-1),
            );
        }
        Ok((CpuStorage::F32(output), l.shape().clone()))
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the qualified VELA_TEST_MLAS_LIBRARY; run explicitly in CI and VM"]
    fn initialization_rejects_invalid_and_duplicate_budgets_before_side_effects() {
        for budget in [-1, 0, 1025, i32::MAX] {
            let error =
                unsafe { initialize("/nonexistent/vela-test-library", budget) }.unwrap_err();
            assert!(error.to_string().contains("invalid BLAS thread count"));
        }

        let env = rt_core::env::TestOverrides::from_env().unwrap();
        let path = env
            .vela_mlas_library
            .expect("set VELA_TEST_MLAS_LIBRARY to the hash-verified standalone library");
        let path = path.to_str().expect("MLAS ABI uses a Unicode path");
        unsafe { initialize(path, 2) }.unwrap();
        let library = unsafe { Library::new(path) }.unwrap();
        let get =
            unsafe { library.get::<unsafe extern "C" fn() -> i32>(b"vela_mlas_get_num_threads\0") }
                .unwrap();
        assert_eq!(unsafe { get() }, 2);
        assert!(unsafe { initialize(path, 4) }
            .unwrap_err()
            .to_string()
            .contains("already initialized"));
        assert_eq!(
            unsafe { get() },
            2,
            "failed duplicate initialization changed the active budget"
        );
        std::thread::scope(|scope| {
            for budget in [1, 4, 8, 16] {
                let path = &path;
                scope.spawn(move || assert!(unsafe { initialize(path, budget) }.is_err()));
            }
        });
        assert_eq!(
            unsafe { get() },
            2,
            "concurrent duplicate initialization changed the active budget"
        );
        assert!(unsafe { initialize(path, 1025) }.is_err());
        assert_eq!(unsafe { get() }, 2);
        let weight = Tensor::from_vec(
            vec![1f32, 2., -3., 1., 0.5, -2.],
            (3, 2),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let x =
            Tensor::from_vec(vec![1f32, 2., -1., 3.], (2, 2), &candle_core::Device::Cpu).unwrap();
        let residual = Tensor::from_vec(
            vec![0.25f32, 1., 2., -2., 0., 3.],
            (2, 3),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let packed = Packed::new(&weight).unwrap();
        let value = packed.forward_residual(&x, &residual).unwrap();
        assert_eq!(
            value.to_vec2::<f32>().unwrap(),
            vec![vec![5.25, 0., -1.5], vec![3., 6., -3.5]]
        );
        let cached = packed.initialized_output(2).unwrap();
        unsafe { packed.forward_into(&x, &cached) }.unwrap();
        assert_eq!(
            cached.to_vec2::<f32>().unwrap(),
            packed.forward(&x).unwrap().to_vec2::<f32>().unwrap()
        );
        let changed =
            Tensor::from_vec(vec![0f32, 0., 2., -1.], (2, 2), &candle_core::Device::Cpu).unwrap();
        unsafe { packed.forward_into(&changed, &cached) }.unwrap();
        assert_eq!(
            cached.to_vec2::<f32>().unwrap(),
            packed.forward(&changed).unwrap().to_vec2::<f32>().unwrap(),
            "cached GEMM accumulated stale output"
        );
        let alias = cached
            .flatten_all()
            .unwrap()
            .narrow(0, 0, 4)
            .unwrap()
            .reshape((2, 2))
            .unwrap();
        assert!(unsafe { packed.forward_into(&alias, &cached) }
            .unwrap_err()
            .to_string()
            .contains("aliases output"));
        let before = cached.to_vec2::<f32>().unwrap();
        assert!(unsafe { packed.forward_residual_into(&alias, &cached) }
            .unwrap_err()
            .to_string()
            .contains("aliases output"));
        assert_eq!(
            cached.to_vec2::<f32>().unwrap(),
            before,
            "alias rejection modified the destination"
        );
        let private_residual = Tensor::from_vec(
            residual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            (2, 3),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let expected = packed
            .forward_residual(&x, &residual)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        unsafe { packed.forward_residual_into(&x, &private_residual) }.unwrap();
        assert_eq!(private_residual.to_vec2::<f32>().unwrap(), expected);
        let qkv = Tensor::from_vec(
            (0..24).map(|n| n as f32 - 7.).collect::<Vec<_>>(),
            (2, 2, 6),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let cos = Tensor::from_vec(vec![1f32, 0.5], (2, 1), &candle_core::Device::Cpu).unwrap();
        let sin = Tensor::from_vec(vec![0f32, -0.25], (2, 1), &candle_core::Device::Cpu).unwrap();
        let shape = qkv_rope_shape(&qkv, &cos, &sin, 1).unwrap();
        let output = Tensor::zeros(
            shape.as_slice(),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        qkv_rope_into(&qkv, &cos, &sin, 1, &output).unwrap();
        assert_eq!(
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            qkv_rope(&qkv, &cos, &sin, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        let different = qkv.affine(-0.5, 2.).unwrap();
        qkv_rope_into(&different, &cos, &sin, 1, &output).unwrap();
        let before = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            before,
            qkv_rope(&different, &cos, &sin, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            "QKV cache retained stale output"
        );
        let alias = output.reshape((2, 2, 6)).unwrap();
        assert!(qkv_rope_into(&alias, &cos, &sin, 1, &output)
            .unwrap_err()
            .to_string()
            .contains("aliases output"));
        let table_alias = output
            .flatten_all()
            .unwrap()
            .narrow(0, 0, 2)
            .unwrap()
            .reshape((2, 1))
            .unwrap();
        assert!(qkv_rope_into(&qkv, &table_alias, &sin, 1, &output)
            .unwrap_err()
            .to_string()
            .contains("aliases output"));
        assert_eq!(
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            before,
            "alias rejection mutated QKV storage"
        );
        let disabled = crate::mlas::workspace::Pool::new(false);
        let mut lease = disabled.checkout().unwrap();
        let fallback = lease.qkv_rope(&qkv, &cos, &sin, 1).unwrap();
        assert_eq!(
            fallback.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            qkv_rope(&qkv, &cos, &sin, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        assert_eq!(disabled.stats().qkv_rope_budget_fallbacks, 1);
        assert_eq!(disabled.stats().reserved_cache_bytes, 0);
        let pool = crate::mlas::workspace::Pool::new(true);
        let mut lease = pool.checkout().unwrap();
        let _ = lease.project(0, &packed, &x).unwrap();
        assert_eq!(
            pool.stats().reserved_cache_bytes,
            packed.output_storage_bytes(2).unwrap()
        );
        let one = Tensor::from_vec(vec![2f32, -1.], (1, 2), &candle_core::Device::Cpu).unwrap();
        let _ = lease.project(0, &packed, &one).unwrap();
        assert_eq!(
            pool.stats().reserved_cache_bytes,
            packed.output_storage_bytes(1).unwrap(),
            "shape change did not release the old reservation first"
        );
        drop(lease);
        drop(pool);
        let masks = Tensor::from_vec(
            vec![1u32, 1, 0, 1, 0, 1, 0, 1, 1, 0],
            (2, 5),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        for radius in [None, Some(1)] {
            let data = (0..60).map(|i| i as f32 * 0.25 - 5.).collect::<Vec<_>>();
            let source =
                Tensor::from_vec(data.clone(), (2, 2, 3, 5), &candle_core::Device::Cpu).unwrap();
            let expected = masked_softmax(&source, &masks, 1, 0, radius)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let private = Tensor::from_vec(data, (2, 2, 3, 5), &candle_core::Device::Cpu).unwrap();
            private_masked_softmax(&private, &masks, 1, 0, radius).unwrap();
            let actual = private.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "private softmax changed numeric output"
            );
            let before = actual;
            assert!(private_masked_softmax(&private, &masks, 4, 0, radius).is_err());
            assert_eq!(
                private.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                before,
                "rejected mask range mutated scores"
            );
        }
    }
    #[test]
    fn aligned_storage_preserves_initialized_values_and_logical_view() {
        for count in [1, 15, 16, 17, 64, 77, 512] {
            let source = (0..count).map(|i| i as f32 - 11.).collect::<Vec<_>>();
            let layout = OutputLayout {
                enabled: true,
                offset: AtomicUsize::new(0),
            };
            let mut buffer = layout.allocate_from(&source).unwrap();
            let offset = layout.offset.load(Ordering::Relaxed);
            assert!(offset < 16);
            assert_eq!(layout.pointer(&mut buffer) as usize % 64, 0);
            assert_eq!(&buffer[offset..offset + count], source.as_slice());
            assert!(buffer[..offset]
                .iter()
                .chain(buffer[offset + count..].iter())
                .all(|&v| v == 0.));
            let tensor =
                Tensor::from_vec(buffer.clone(), (buffer.len(),), &candle_core::Device::Cpu)
                    .unwrap();
            assert_eq!(
                layout
                    .finish(tensor, Shape::from((count,)))
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                source
            );
        }
    }
}
