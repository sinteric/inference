//! Convolution weights packed once into the layout the primitives read,
//! with the bias and the depthwise and grouped variants.

use super::{dimensions, make_stream, Binding, Bound, ConvWeights, Filters, Md, Pd, Step};
use crate::cpu::onednn::api::Api;
use crate::cpu::onednn::cpu::Aligned;
use crate::cpu::onednn::OnednnCpu;
use crate::{check_shape, checked_size, Error, Result};
use onednn_sys::*;
use std::ptr;
use std::sync::Arc;

impl Filters {
    /// The weight tensor's dimensions: OIHW, or GOIHW for depthwise groups.
    fn dims(&self) -> Result<Vec<usize>> {
        let Self {
            output,
            input,
            kernel,
            groups,
            transpose,
        } = *self;
        check_shape!(
            groups > 0
                && input > 0
                && output > 0
                && input.is_multiple_of(groups)
                && output.is_multiple_of(groups),
            "invalid conv channels/groups"
        );
        if groups != 1 && (groups != input || groups != output) {
            return Err(Error::unsupported(
                "grouped convolution",
                "groups=1 or depthwise groups=C only",
            ));
        }
        check_shape!(
            !transpose || groups == 1,
            "grouped deconvolution is not supported"
        );
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
        Ok(dims)
    }

    /// The blocked layout the primitives read these filters in, for a CPU
    /// whose channel block is `block` (16 with AVX-512, else 8).
    fn layout(&self, block: usize) -> Result<i32> {
        let wide = block == 16;
        let pick = |w, n| if wide { w } else { n };
        Ok(if self.transpose {
            // Deconvolution lowers to convolution backward-data. Its JIT
            // expects the opposite inner block order to forward convolution;
            // oneDNN names that layout for 16-channel blocks only.
            if !wide {
                return Err(Error::unsupported(
                    "deconvolution",
                    "oneDNN has no 8-channel deconvolution weight layout; needs AVX-512",
                ));
            }
            dnnl_IOhw16i16o
        } else if self.groups == 1 && self.input < block {
            pick(dnnl_Ohwi16o, dnnl_Ohwi8o)
        } else if self.groups == 1 {
            pick(dnnl_OIhw16i16o, dnnl_OIhw8i8o)
        } else if self.groups == self.input && self.groups == self.output {
            pick(dnnl_Goihw16g, dnnl_Goihw8g)
        } else {
            pick(dnnl_gOIhw16i16o, dnnl_gOIhw8i8o)
        })
    }
}

impl ConvWeights {
    /// Pack `data` (OIHW, or GOIHW for depthwise groups) once into the
    /// layout the primitives read; `bias` has one value per output channel.
    pub fn pack(
        cpu: Arc<OnednnCpu>,
        filters: Filters,
        data: &[f32],
        bias: Option<&[f32]>,
    ) -> Result<Arc<Self>> {
        let dims = filters.dims()?;
        check_shape!(
            data.len() == checked_size("conv weights", &dims)?,
            "wrong conv weight length"
        );
        check_shape!(
            bias.is_none_or(|b| b.len() == filters.output),
            "wrong conv bias length"
        );
        let tag = filters.layout(cpu.conv_block())?;
        let plain = if filters.groups == 1 {
            dnnl_oihw
        } else {
            dnnl_goihw
        };
        let (md, packed) = reorder(&cpu, &dims, plain, tag, data)?;
        let bias = bias
            .map(|b| {
                let mut v = crate::try_vec(b.len())?;
                v.copy_from_slice(b);
                Ok::<_, Error>(v)
            })
            .transpose()?;
        cpu.conv_weight_bytes
            .fetch_add(packed.bytes(), std::sync::atomic::Ordering::Relaxed);
        Ok(Arc::new(Self {
            md,
            data: packed,
            bias,
            filters,
            cpu,
            tag,
        }))
    }
    /// Too few input channels for a block: the convolution reads plain NCHW.
    pub(super) fn reads_plain(&self) -> bool {
        self.tag == dnnl_Ohwi16o || self.tag == dnnl_Ohwi8o
    }
    /// Dense blocked weights, as a pointwise tile requires.
    pub(super) fn dense_blocked(&self) -> bool {
        self.tag == dnnl_OIhw16i16o || self.tag == dnnl_OIhw8i8o
    }
}

/// Reorder `data`, laid out as `from`, into a new buffer laid out as `to`:
/// that layout's descriptor and the buffer.
fn reorder(
    cpu: &Arc<OnednnCpu>,
    dims: &[usize],
    from: i32,
    to: i32,
    data: &[f32],
) -> Result<(Md, Aligned)> {
    let raw = Md::tagged(dims, from)?;
    let md = Md::tagged(dims, to)?;
    let packed = Aligned::new(md.bytes()?)?;
    let mut pd = Pd(ptr::null_mut());
    let engine = cpu.conv_engine();
    Api::check(
        // SAFETY: both descriptors and the engine are live, no attribute
        // means the default one, and oneDNN writes the new handle into `pd`,
        // which owns it.
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
            (DNNL_ARG_TO, Md::tagged(dims, to)?, Binding::External),
        ],
    )?;
    let _compute = crate::cpu::threads::compute_lock();
    cpu.conv_api().apply_threads(cpu.threads() as i32)?;
    let stream = make_stream(cpu)?;
    let scratch = Aligned::new(step.scratch_bytes)?;
    let target = Bound {
        ptr: packed.ptr(),
        bytes: packed.bytes(),
    };
    let scratch = Bound {
        ptr: scratch.ptr(),
        bytes: scratch.bytes(),
    };
    step.execute(cpu, &stream, &[Bound::of(data), target], scratch)?;
    Ok((md, packed))
}
