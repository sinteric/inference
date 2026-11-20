//! Convolution weights packed once into the layout the primitives read,
//! with the bias and the depthwise and grouped variants.

use super::{dimensions, make_stream, Binding, ConvWeights, Md, Pd, Step};
use crate::cpu::onednn::api::Api;
use crate::cpu::onednn::cpu::Aligned;
use crate::cpu::onednn::OnednnCpu;
use crate::{check_shape, checked_size, Error, Result};
use onednn_sys::*;
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

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
            return Err(Error::unsupported(
                "grouped convolution",
                "groups=1 or depthwise groups=C only",
            ));
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
                return Err(Error::unsupported(
                    "deconvolution",
                    "oneDNN has no 8-channel deconvolution weight layout; needs AVX-512",
                ));
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
        let _compute = crate::cpu::threads::compute_lock();
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
                let mut v = crate::try_vec(b.len())?;
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
    /// Too few input channels for a block: the convolution reads plain NCHW.
    pub(super) fn reads_plain(&self) -> bool {
        self.tag == dnnl_Ohwi16o || self.tag == dnnl_Ohwi8o
    }
    /// Dense blocked weights, as a pointwise tile requires.
    pub(super) fn dense_blocked(&self) -> bool {
        self.tag == dnnl_OIhw16i16o || self.tag == dnnl_OIhw8i8o
    }
}
