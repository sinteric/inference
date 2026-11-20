//! Lowering a graph for one input shape: each node's output shape, which
//! tensors stay channel-blocked, and the padding glue a node needs.

use super::{dimensions, Attr, ConvNet, ConvWeights, Md, Node, Op, Pd, PostOp, Shape, Spatial};
use crate::cpu::onednn::api::Api;
use crate::{check_shape, Error, Result};
use onednn_sys::*;
use std::ptr;

pub(super) fn broadcast(a: Shape, b: Shape) -> Result<()> {
    check_shape!(
        (0..4).all(|i| b[i] == 1 || b[i] == a[i]),
        "broadcast {b:?} into {a:?}"
    );
    Ok(())
}

/// `block` is the backend's channel block ([`OnednnCpu::conv_block`]).
pub(super) fn infer_shape(node: &Node, shapes: &[Shape], block: usize) -> Result<Shape> {
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
            check_shape!(s[1] == w.filters.input, "conv input channels");
            spatial.output(s, w.filters.kernel, w.filters.output, w.filters.transpose)?
        }
        Op::Pool {
            kernel, spatial, ..
        } => spatial.output(s, *kernel, s[1], false)?,
        Op::DepthToSpace2x => {
            check_shape!(
                s[1].is_multiple_of(4),
                "depth-to-space requires 4*C channels"
            );
            [
                s[0],
                s[1] / 4,
                s[2].checked_mul(2)
                    .ok_or(Error::size_overflow("depth-to-space height"))?,
                s[3].checked_mul(2)
                    .ok_or(Error::size_overflow("depth-to-space width"))?,
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
                    .ok_or(Error::size_overflow("resize height"))?,
                s[3].checked_mul(scales[1])
                    .ok_or(Error::size_overflow("resize width"))?,
            ]
        }
        Op::SpatialMean(_) => [s[0], s[1], 1, 1],
        Op::ZeroPad { begin, end } => {
            let mut o = s;
            for j in 0..2 {
                o[j + 2] = s[j + 2]
                    .checked_add(begin[j])
                    .and_then(|v| v.checked_add(end[j]))
                    .ok_or(Error::size_overflow("zero pad"))?;
            }
            o
        }
        Op::ConcatAlignedChannels | Op::Concat { .. } => concat_shape(node, shapes, block)?,
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

/// The output of a concat node: its inputs' shapes summed along the axis.
fn concat_shape(node: &Node, shapes: &[Shape], block: usize) -> Result<Shape> {
    let s = shapes[node.inputs[0]];
    let axis = if let Op::Concat { axis } = node.op {
        axis
    } else {
        1
    };
    if matches!(node.op, Op::ConcatAlignedChannels) {
        check_shape!(
            node.inputs
                .iter()
                .all(|&id| shapes[id][1].is_multiple_of(block)),
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
            .ok_or(Error::size_overflow("concat"))?;
    }
    Ok(o)
}

pub(super) fn output_blocked(node: &Node, output: Shape, block: usize) -> bool {
    match &node.op {
        Op::Reorder { blocked } => *blocked,
        Op::DepthToSpace2x => output[1].is_multiple_of(block),
        Op::Concat { .. } | Op::SpatialMean(_) | Op::Prelu { .. } => false,
        _ => true,
    }
}

impl ConvNet {
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
                .map(|_| Md::tagged(&[w.filters.output], dnnl_x))
                .transpose()?;
            let a = spatial.arrays()?;
            let mut attr = Attr(ptr::null_mut());
            let mut pd = Pd(ptr::null_mut());
            let mut info = ptr::null::<std::ffi::c_char>();
            // SAFETY: oneDNN writes each new handle into its owner (`attr`,
            // `pd`) before the next call reads it; every descriptor is live,
            // each spatial array holds the two entries oneDNN reads, and
            // `info` is a valid out-pointer for the implementation string.
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
            // SAFETY: the implementation string is the probe descriptor's
            // own, alive while it is.
            Ok(unsafe { crate::cpu::onednn::api::cstr(info) }.starts_with("ref"))
        };
        if w.filters.transpose
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
            ..*spatial
        };
        Ok(!reference(padded, &unpadded)?)
    }

    pub(super) fn lower_nodes(&self, shape: Shape) -> Result<(Vec<Node>, Vec<usize>, Vec<bool>)> {
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
                            ..*spatial
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
}
