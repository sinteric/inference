//! Building a plan for one input shape: a primitive per lowered node,
//! shared across nodes and shapes by descriptor, and the liveness arena
//! the steps run in.

use super::{
    broadcast, clone_md, dimensions, infer_shape, Attr, Binding, Caches, ConvNet, Md, Node, Op,
    PadGlue, Pd, Plan, PointwiseTile, PostOp, Posts, Primitive, Shape, Step, StepKey, StepLru,
    Tensor,
};
use crate::cpu::onednn::api::Api;
use crate::{check_shape, checked_size, Error, Result};
use onednn_sys::*;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::ptr;
use std::sync::Arc;

// Primitive code depends on descriptors and attributes, not bound weight,
// scalar or bias values. Share it across repeated backbone blocks as well
// as widths. Full descriptor blobs preserve strides, padding and block order.
pub(super) fn native_key(node: &Node, specs: &[(i32, Md, Binding)]) -> Result<[u8; 32]> {
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
        Op::Concat { axis } => {
            int(6);
            int(*axis as i64);
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
        let mut blob = crate::try_vec(size)?;
        Api::check(
            unsafe { dnnl_memory_desc_get_blob(blob.as_mut_ptr(), &mut size, md.raw()) },
            "descriptor blob",
        )?;
        h.update(size.to_le_bytes());
        h.update(blob);
    }
    Ok(h.finalize().into())
}

pub(super) fn retain_step(cache: &mut StepLru, capacity: usize, key: StepKey, step: Arc<Step>) {
    if cache.len() == capacity {
        cache.pop_back();
    }
    cache.push_front((key, step));
}

impl ConvNet {
    pub(super) fn build_plan(&self, shape: Shape) -> Result<Plan> {
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
            .map(|t| crate::align_up(t.md.bytes()?, 64))
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
                            let mut expanded = crate::try_vec(s[1])?;
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
                tile.source_offset = crate::align_up(step.scratch_bytes, 64)?;
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
                zero.push((offsets[id], crate::align_up(bytes, 64)?));
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
