//! Building a plan for one input shape: a primitive per lowered node,
//! shared across nodes and shapes by descriptor, and the liveness arena
//! the steps run in.

use super::{
    broadcast, clone_md, dimensions, infer_shape, Attr, Binding, Caches, ConvNet, ConvWeights,
    Descriptors, Md, NativeCache, Node, Op, PadGlue, Pd, Plan, PointwiseTile, Pool, PostOp, Posts,
    Primitive, Shape, Spatial, Step, StepKey, StepLru, Tensor,
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
            int(if weights.filters.transpose { 1 } else { 2 });
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
        _ => return Err(Error::internal("glue in native primitive cache")),
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
            // SAFETY: with a null blob oneDNN only writes the size to a live
            // local; `md` is live.
            unsafe { dnnl_memory_desc_get_blob(ptr::null_mut(), &mut size, md.raw()) },
            "descriptor blob size",
        )?;
        let mut blob = crate::try_vec(size)?;
        Api::check(
            // SAFETY: `blob` holds the `size` bytes oneDNN asked for above.
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

/// Where the arena puts each tensor: its byte offset, the arena's size, and
/// for each tensor the concat whose bytes it is a view into (itself when
/// none).
struct Placement {
    offsets: Vec<usize>,
    peak: usize,
    parent: Vec<usize>,
}

/// Concat views: for each tensor, the concat it is a view into (`parent`,
/// itself when none), its byte offset inside it (`relative`), and the first
/// step a concat's interval covers (`first`, from its earliest view).
struct Views {
    parent: Vec<usize>,
    relative: Vec<usize>,
    first: Vec<usize>,
}

/// One node of a plan being built, with the tensors it reads and writes.
struct NodeCtx<'a> {
    node: &'a Node,
    /// The node's position; its output is tensor `index + 1`.
    index: usize,
    tensors: &'a [Tensor],
    /// Whether each tensor is in the blocked layout.
    layouts: &'a [bool],
    parent: &'a [usize],
    block: usize,
    blocked_tag: dnnl_format_tag_t,
}

/// The descriptors and attribute a node's primitive descriptor is made from.
struct PdArgs {
    engine: dnnl_engine_t,
    src: dnnl_memory_desc_t,
    dst: dnnl_memory_desc_t,
    attr: dnnl_primitive_attr_t,
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
            .map_err(|_| Error::poisoned("conv primitive cache"))?;
        let Caches {
            steps: primitive_cache,
            natives,
            descriptors,
        } = &mut *caches;
        primitive_cache.resize_with(nodes.len(), VecDeque::new);
        let (tensors, shapes) =
            tensors_of(shape, &nodes, &layouts, block, blocked_tag, descriptors)?;
        let placement = self.place(&nodes, &tensors, &shapes)?;
        let mut steps = Vec::new();
        for (index, node) in nodes.iter().enumerate() {
            let key = step_key(identities[index], node, &tensors, tensors[index + 1].shape);
            let cache = &mut primitive_cache[index];
            if let Some(i) = cache.iter().position(|(k, _)| *k == key) {
                let entry = cache
                    .remove(i)
                    .ok_or_else(|| Error::internal("conv primitive LRU index"))?;
                steps.push(entry.1.clone());
                cache.push_front(entry);
                continue;
            }
            let at = NodeCtx {
                node,
                index,
                tensors: &tensors,
                layouts: &layouts,
                parent: &placement.parent,
                block,
                blocked_tag,
            };
            let step = match at.glue()? {
                Some(step) => step,
                None => self.primitive_step(&at, descriptors, natives, nodes.len())?,
            };
            let step = Arc::new(step);
            retain_step(cache, self.capacity, key, step.clone());
            steps.push(step);
        }
        let zero = zero_ranges(&tensors, &placement.offsets)?;
        let scratch_bytes = steps.iter().map(|s| s.scratch_bytes).max().unwrap_or(64);
        let total = placement
            .peak
            .checked_add(scratch_bytes)
            .ok_or(Error::size_overflow("conv workspace"))?;
        let limit = self.cpu.conv_limit();
        if total > limit {
            return Err(Error::workspace_limit(
                "ConvNet activation arena + scratch",
                total,
                limit,
            ));
        }
        Ok(Plan {
            shape,
            output_shape: shapes[shapes.len() - 1],
            steps,
            offsets: placement.offsets,
            zero,
            arena_bytes: placement.peak,
            scratch_bytes,
            bytes: 0,
        })
    }

    /// Places every tensor but the input (which the caller lends) in one
    /// arena, two tensors overlapping only when no step has both live.
    fn place(&self, nodes: &[Node], tensors: &[Tensor], shapes: &[Shape]) -> Result<Placement> {
        let last = last_uses(nodes, tensors.len());
        let Views {
            parent,
            relative,
            first,
        } = concat_views(nodes, tensors, shapes, &last)?;
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
                    .ok_or(Error::size_overflow("conv arena"))?;
                if end <= start {
                    break;
                }
                offset = offset.max(
                    start
                        .checked_add(bytes)
                        .ok_or(Error::size_overflow("conv arena"))?,
                );
            }
            offsets[id] = offset;
            peak = peak.max(
                offset
                    .checked_add(sizes[id])
                    .ok_or(Error::size_overflow("conv arena"))?,
            );
            placed.push(id);
        }
        for id in 1..tensors.len() {
            if parent[id] != id {
                offsets[id] = offsets[parent[id]]
                    .checked_add(relative[id])
                    .ok_or(Error::size_overflow("concat offset"))?;
            }
        }
        peak = peak
            .checked_add(64)
            .ok_or(Error::size_overflow("conv arena alignment"))?;
        let limit = self.cpu.conv_limit();
        if peak > limit {
            return Err(Error::workspace_limit(
                "ConvNet activation arena",
                peak,
                limit,
            ));
        }
        Ok(Placement {
            offsets,
            peak,
            parent,
        })
    }

    /// The step a oneDNN primitive runs for `at`'s node.
    fn primitive_step(
        &self,
        at: &NodeCtx,
        descriptors: &mut Descriptors,
        natives: &mut NativeCache,
        nodes: usize,
    ) -> Result<Step> {
        let node = at.node;
        let (src, dst) = (&at.tensors[at.src_id()], &at.tensors[at.dst_id()]);
        let (s, output) = (src.shape, dst.shape);
        let tile = self.pointwise_tile(node, s, output)?;
        let src_md = match tile {
            Some(t) => descriptors.tagged(&[1, s[1], 1, t.width], at.blocked_tag)?,
            None => clone_md(&src.md)?,
        };
        let dst_md = match tile {
            Some(t) => descriptors.tagged(&[1, output[1], 1, t.width], at.blocked_tag)?,
            None => clone_md(&dst.md)?,
        };
        let mut specs = vec![
            (
                DNNL_ARG_SRC,
                clone_md(&src_md)?,
                Binding::Tensor(at.src_id()),
            ),
            (
                DNNL_ARG_DST,
                clone_md(&dst_md)?,
                Binding::Tensor(at.dst_id()),
            ),
        ];
        let (attr, _posts) = attr_with_post_ops(node, output, at.tensors, descriptors, &mut specs)?;
        let args = PdArgs {
            engine: self.cpu.conv_engine(),
            src: src_md.raw(),
            dst: dst_md.raw(),
            attr: attr.0,
        };
        let pd = at.primitive_desc(&args, descriptors, &mut specs)?;
        let capacity = self.capacity.saturating_mul(nodes);
        let mut step = Step::new_cached(pd, specs, Some((natives, node, capacity)))?;
        if let Some(mut tile) = tile {
            tile.source_offset = crate::align_up(step.scratch_bytes, 64)?;
            tile.destination_offset = tile
                .source_offset
                .checked_add(src_md.bytes()?)
                .ok_or(Error::size_overflow("pointwise scratch"))?;
            step.scratch_bytes = tile
                .destination_offset
                .checked_add(dst_md.bytes()?)
                .ok_or(Error::size_overflow("pointwise scratch"))?;
            step.tile = Some(tile);
        }
        Ok(step)
    }

    /// The pointwise tile a 1x1 convolution runs in, when tiles are on and
    /// the node qualifies.
    fn pointwise_tile(
        &self,
        node: &Node,
        s: Shape,
        output: Shape,
    ) -> Result<Option<PointwiseTile>> {
        let plane = checked_size("pointwise plane", &[s[2], s[3]])?;
        let fits = self.pointwise_tiles
            && plane > 1
            && node
                .post_ops
                .iter()
                .all(|p| !matches!(p, PostOp::Binary { .. }))
            && matches!(&node.op, Op::Conv { weights, spatial }
                if !weights.filters.transpose && weights.dense_blocked()
                    && weights.filters.kernel == [1; 2] && spatial.strides == [1; 2]
                    && spatial.dilations == [1; 2] && spatial.pads_begin == [0; 2]
                    && spatial.pads_end == [0; 2]);
        Ok(fits.then(|| PointwiseTile {
            shape: s,
            plane,
            output_channels: output[1],
            width: plane.min(512).next_power_of_two().max(16),
            source_offset: 0,
            destination_offset: 0,
        }))
    }
}

impl Step {
    /// A glue pass over `specs`: no primitive and no scratch; the caller
    /// sets which pass it is.
    fn glue(specs: Vec<(i32, Md, Binding)>, implementation: String) -> Self {
        Self {
            primitive: Arc::new(Primitive(ptr::null_mut())),
            specs,
            scratch: None,
            scratch_bytes: 64,
            implementation,
            mean: None,
            concat: None,
            shuffle: None,
            tile: None,
            pad: None,
        }
    }
}

impl NodeCtx<'_> {
    fn src_id(&self) -> usize {
        self.node.inputs[0]
    }

    fn dst_id(&self) -> usize {
        self.index + 1
    }

    /// The node's source and destination, bound as most steps bind them.
    fn io_specs(&self) -> Result<Vec<(i32, Md, Binding)>> {
        let (src, dst) = (self.src_id(), self.dst_id());
        Ok(vec![
            (
                DNNL_ARG_SRC,
                clone_md(&self.tensors[src].md)?,
                Binding::Tensor(src),
            ),
            (
                DNNL_ARG_DST,
                clone_md(&self.tensors[dst].md)?,
                Binding::Tensor(dst),
            ),
        ])
    }

    /// The node's step when a glue pass runs it rather than a primitive.
    fn glue(&self) -> Result<Option<Step>> {
        let node = self.node;
        let s = self.tensors[self.src_id()].shape;
        let output = self.tensors[self.dst_id()].shape;
        let no_post_ops = |what: &str| -> Result<()> {
            check_shape!(node.post_ops.is_empty(), "{what} post-ops unsupported");
            Ok(())
        };
        Ok(Some(match node.op {
            Op::ZeroPad { begin, .. } => {
                no_post_ops("zero pad")?;
                let blocked = self.layouts[self.src_id()];
                let lanes = if blocked { self.block } else { 1 };
                let groups = if blocked {
                    s[1].div_ceil(self.block)
                } else {
                    s[1]
                };
                let pad = PadGlue {
                    planes: checked_size("zero pad planes", &[s[0], groups])?,
                    height: s[2],
                    width: s[3],
                    lanes,
                    top: begin[0],
                    left: begin[1],
                    padded_height: output[2],
                    padded_width: output[3],
                };
                Step {
                    pad: Some(pad),
                    ..Step::glue(self.io_specs()?, "glue:zero_pad".into())
                }
            }
            Op::DepthToSpace2x => {
                no_post_ops("depth-to-space")?;
                let name = "glue:depth_to_space_2x".into();
                Step {
                    shuffle: Some(s),
                    ..Step::glue(self.io_specs()?, name)
                }
            }
            Op::ConcatAlignedChannels => {
                no_post_ops("channel concat")?;
                let dst = self.dst_id();
                let mut specs = vec![(
                    DNNL_ARG_DST,
                    clone_md(&self.tensors[dst].md)?,
                    Binding::Tensor(dst),
                )];
                let mut channels = Vec::new();
                for &id in &node.inputs {
                    specs.push((
                        DNNL_ARG_SRC,
                        clone_md(&self.tensors[id].md)?,
                        Binding::Tensor(id),
                    ));
                    channels.push(self.tensors[id].shape[1]);
                }
                let name = if node.inputs.iter().all(|&id| self.parent[id] == dst) {
                    "glue:concat_producer_views"
                } else {
                    "glue:aligned_channel_concat"
                };
                let concat = (s[0], checked_size("concat plane", &[s[2], s[3]])?, channels);
                Step {
                    concat: Some(concat),
                    ..Step::glue(specs, name.into())
                }
            }
            Op::SpatialMean(order) => {
                no_post_ops("spatial mean")?;
                let mean = (
                    checked_size("spatial channels", &[s[0], s[1]])?,
                    checked_size("spatial plane", &[s[2], s[3]])?,
                    order,
                );
                let name = format!("glue:fp32_spatial_mean_{order:?}");
                Step {
                    mean: Some(mean),
                    ..Step::glue(self.io_specs()?, name)
                }
            }
            _ => return Ok(None),
        }))
    }

    /// The primitive descriptor for the node, adding the bindings its op
    /// reads beyond source and destination to `specs`.
    fn primitive_desc(
        &self,
        args: &PdArgs,
        descriptors: &mut Descriptors,
        specs: &mut Vec<(i32, Md, Binding)>,
    ) -> Result<Pd> {
        let (node, tensors) = (self.node, self.tensors);
        let PdArgs {
            engine: e,
            src,
            dst,
            attr,
        } = *args;
        let mut pd = Pd(ptr::null_mut());
        let st = match &node.op {
            Op::Conv { weights, spatial } => {
                conv_pd(&mut pd, args, weights, spatial, descriptors, specs)?
            }
            Op::Pool {
                kind,
                kernel,
                spatial,
            } => pool_pd(&mut pd, args, *kind, *kernel, spatial)?,
            Op::Resize { linear, .. } => {
                let alg = if *linear {
                    dnnl_resampling_linear
                } else {
                    dnnl_resampling_nearest
                };
                let (inference, no_factors) = (dnnl_forward_inference, ptr::null());
                // SAFETY: the descriptors and attribute are live; no factors
                // means oneDNN derives them from the two shapes; it writes
                // the new handle into `pd`.
                unsafe {
                    dnnl_resampling_forward_primitive_desc_create(
                        &mut pd.0, e, inference, alg, no_factors, src, dst, attr,
                    )
                }
            }
            Op::DepthToSpace2x => {
                return Err(Error::internal("unexpected depth-to-space primitive"))
            }
            Op::SpatialMean(_) => return Err(Error::internal("unexpected spatial mean primitive")),
            Op::ZeroPad { .. } => return Err(Error::internal("unexpected zero pad primitive")),
            Op::UpsampleNearest { .. } => return Err(Error::internal("unresolved resize scale")),
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
                let (n, axis) = (mds.len() as i32, axis as i32);
                // SAFETY: `mds` holds `n` live descriptors, as do `dst` and
                // the attribute; oneDNN writes the new handle into `pd`.
                unsafe {
                    dnnl_concat_primitive_desc_create(
                        &mut pd.0,
                        e,
                        dst,
                        n,
                        axis,
                        mds.as_ptr(),
                        attr,
                    )
                }
            }
            Op::Binary(kind) => {
                let t = &tensors[node.inputs[1]];
                specs.push((
                    DNNL_ARG_SRC_1,
                    clone_md(&t.md)?,
                    Binding::Tensor(node.inputs[1]),
                ));
                let (alg, other) = (kind.alg(), t.md.raw());
                // SAFETY: the three descriptors and the attribute are live;
                // oneDNN writes the new handle into `pd`.
                unsafe {
                    dnnl_binary_primitive_desc_create(&mut pd.0, e, alg, src, other, dst, attr)
                }
            }
            Op::Prelu { slopes } => prelu_pd(
                &mut pd,
                args,
                slopes,
                self.tensors[self.src_id()].shape,
                descriptors,
                specs,
            )?,
            Op::Reorder { .. } => {
                check_shape!(node.post_ops.is_empty(), "post-ops on reorder unsupported");
                // SAFETY: both descriptors and the attribute are live; oneDNN
                // writes the new handle into `pd`.
                unsafe { dnnl_reorder_primitive_desc_create(&mut pd.0, src, e, dst, e, attr) }
            }
        };
        Api::check(st, "conv graph primitive descriptor")?;
        Ok(pd)
    }
}

/// The (de)convolution descriptor, binding its weights and bias.
fn conv_pd(
    pd: &mut Pd,
    args: &PdArgs,
    w: &Arc<ConvWeights>,
    spatial: &Spatial,
    descriptors: &mut Descriptors,
    specs: &mut Vec<(i32, Md, Binding)>,
) -> Result<dnnl_status_t> {
    let a = spatial.arrays()?;
    let bias = if w.bias.is_some() {
        Some(descriptors.tagged(&[w.filters.output], dnnl_x)?)
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
    let (create, alg) = if w.filters.transpose {
        (
            dnnl_deconvolution_forward_primitive_desc_create as ConvCreate,
            dnnl_deconvolution_direct,
        )
    } else {
        (
            dnnl_convolution_forward_primitive_desc_create as ConvCreate,
            dnnl_convolution_direct,
        )
    };
    // SAFETY: every descriptor and the attribute are live (`bias_ptr` is the
    // bias descriptor `specs` now holds, or null for none), each array holds
    // the two spatial entries oneDNN reads, and oneDNN writes the new handle
    // into `pd`.
    Ok(unsafe {
        create(
            &mut pd.0,
            args.engine,
            dnnl_forward_inference,
            alg,
            args.src,
            w.md.raw(),
            bias_ptr,
            args.dst,
            a[0].as_ptr(),
            a[1].as_ptr(),
            a[2].as_ptr(),
            a[3].as_ptr(),
            args.attr,
        )
    })
}

/// The pooling descriptor.
fn pool_pd(
    pd: &mut Pd,
    args: &PdArgs,
    kind: Pool,
    kernel: [usize; 2],
    spatial: &Spatial,
) -> Result<dnnl_status_t> {
    let a = spatial.arrays()?;
    let k = dimensions(&kernel)?;
    let (strides, pads) = (a[0].as_ptr(), [a[2].as_ptr(), a[3].as_ptr()]);
    let alg = kind.alg();
    // SAFETY: every descriptor and the attribute are live, each array holds
    // the two spatial entries oneDNN reads, and oneDNN writes the new handle
    // into `pd`, which owns it.
    Ok(unsafe {
        dnnl_pooling_forward_primitive_desc_create(
            &mut pd.0,
            args.engine,
            dnnl_forward_inference,
            alg,
            args.src,
            args.dst,
            strides,
            k.as_ptr(),
            a[1].as_ptr(),
            pads[0],
            pads[1],
            args.attr,
        )
    })
}

/// The signature oneDNN's convolution and deconvolution descriptors share.
type ConvCreate = unsafe extern "C" fn(
    *mut dnnl_primitive_desc_t,
    dnnl_engine_t,
    dnnl_prop_kind_t,
    dnnl_alg_kind_t,
    const_dnnl_memory_desc_t,
    const_dnnl_memory_desc_t,
    const_dnnl_memory_desc_t,
    const_dnnl_memory_desc_t,
    *const dnnl_dim_t,
    *const dnnl_dim_t,
    *const dnnl_dim_t,
    *const dnnl_dim_t,
    const_dnnl_primitive_attr_t,
) -> dnnl_status_t;

/// The PReLU descriptor, binding its slopes (one shared slope is expanded to
/// one per channel of `s`).
fn prelu_pd(
    pd: &mut Pd,
    args: &PdArgs,
    slopes: &[f32],
    s: Shape,
    descriptors: &mut Descriptors,
    specs: &mut Vec<(i32, Md, Binding)>,
) -> Result<dnnl_status_t> {
    check_shape!(slopes.len() == 1 || slopes.len() == s[1], "PReLU slopes");
    let slopes = if slopes.len() == 1 {
        let mut expanded = crate::try_vec(s[1])?;
        expanded.fill(slopes[0]);
        expanded
    } else {
        slopes.to_vec()
    };
    let md = descriptors.tagged(&[1, slopes.len(), 1, 1], dnnl_nchw)?;
    // SAFETY: the descriptors and the attribute are live; oneDNN writes the
    // new handle into `pd`.
    let st = unsafe {
        dnnl_prelu_forward_primitive_desc_create(
            &mut pd.0,
            args.engine,
            dnnl_forward_inference,
            args.src,
            md.raw(),
            args.dst,
            args.attr,
        )
    };
    specs.push((DNNL_ARG_WEIGHTS, md, Binding::Slopes(Arc::new(slopes))));
    Ok(st)
}

/// The strict attribute a node's primitive is built with, carrying the
/// node's post-ops; binds each post-op operand in `specs`. The post-op list
/// is returned beside the attribute, which holds a copy of it.
fn attr_with_post_ops(
    node: &Node,
    output: Shape,
    tensors: &[Tensor],
    descriptors: &mut Descriptors,
    specs: &mut Vec<(i32, Md, Binding)>,
) -> Result<(Attr, Posts)> {
    let mut attr = Attr(ptr::null_mut());
    // SAFETY: oneDNN writes the new handle into `attr`, which owns it.
    let rc = unsafe { dnnl_primitive_attr_create(&mut attr.0) };
    Api::check(rc, "conv attr")?;
    // SAFETY: the live attribute created above.
    let rc = unsafe { dnnl_primitive_attr_set_fpmath_mode(attr.0, dnnl_fpmath_mode_strict) };
    Api::check(rc, "conv strict math")?;
    // SAFETY: as above.
    let rc = unsafe { dnnl_primitive_attr_set_scratchpad_mode(attr.0, dnnl_scratchpad_mode_user) };
    Api::check(rc, "conv user scratch")?;
    let mut posts = Posts(ptr::null_mut());
    // SAFETY: oneDNN writes the new handle into `posts`, which owns it.
    let rc = unsafe { dnnl_post_ops_create(&mut posts.0) };
    Api::check(rc, "conv post-ops")?;
    for (i, p) in node.post_ops.iter().enumerate() {
        // The argument a binary post-op at position `i` reads its operand as.
        let operand = (DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE * (i as i32 + 1)) | DNNL_ARG_SRC_1;
        match p {
            PostOp::Scalar { kind, value } => {
                check_shape!(value.len() == 1 && value[0].is_finite(), "scalar post-op");
                let md = descriptors.tagged(&[1, 1, 1, 1], dnnl_nchw)?;
                // SAFETY: the live list created above and a live descriptor,
                // which oneDNN copies.
                let rc = unsafe { dnnl_post_ops_append_binary(posts.0, kind.alg(), md.raw()) };
                Api::check(rc, "scalar binary post-op")?;
                specs.push((operand, md, Binding::Slopes(value.clone())));
            }
            PostOp::Eltwise(a) => {
                let (alg, alpha, beta) = a.params()?;
                // SAFETY: the live list created above.
                let rc = unsafe { dnnl_post_ops_append_eltwise(posts.0, alg, alpha, beta) };
                Api::check(rc, "conv eltwise post-op")?;
            }
            PostOp::ChannelBias(bias) => {
                check_shape!(bias.len() == output[1], "channel post-bias extent");
                let md = descriptors.tagged(&[1, output[1], 1, 1], dnnl_nchw)?;
                // SAFETY: as for the scalar post-op.
                let rc = unsafe { dnnl_post_ops_append_binary(posts.0, dnnl_binary_add, md.raw()) };
                Api::check(rc, "channel bias post-op")?;
                specs.push((operand, md, Binding::Slopes(bias.clone())));
            }
            PostOp::Binary { kind, tensor } => {
                let t = &tensors[*tensor];
                broadcast(output, t.shape)?;
                // SAFETY: as for the scalar post-op.
                let rc = unsafe { dnnl_post_ops_append_binary(posts.0, kind.alg(), t.md.raw()) };
                Api::check(rc, "conv binary post-op")?;
                specs.push((operand, clone_md(&t.md)?, Binding::Tensor(*tensor)));
            }
        }
    }
    // SAFETY: both are the live handles created above; oneDNN copies the
    // list into the attribute.
    let rc = unsafe { dnnl_primitive_attr_set_post_ops(attr.0, posts.0) };
    Api::check(rc, "conv post-op attributes")?;
    Ok((attr, posts))
}

/// The tensors a plan for `shape` holds, the input first and then each
/// node's output, blocked where `layouts` says; and their shapes.
fn tensors_of(
    shape: Shape,
    nodes: &[Node],
    layouts: &[bool],
    block: usize,
    blocked_tag: dnnl_format_tag_t,
    descriptors: &mut Descriptors,
) -> Result<(Vec<Tensor>, Vec<Shape>)> {
    let mut tensors = vec![Tensor {
        shape,
        md: descriptors.tagged(&shape, dnnl_nchw)?,
    }];
    let mut shapes = vec![shape];
    for node in nodes {
        let output = infer_shape(node, &shapes, block)?;
        let tag = if layouts[tensors.len()] {
            blocked_tag
        } else {
            dnnl_nchw
        };
        tensors.push(Tensor {
            shape: output,
            md: descriptors.tagged(&output, tag)?,
        });
        shapes.push(output);
    }
    Ok((tensors, shapes))
}

/// The key a node's step is cached under. Stable original-node identity and
/// binding ids also protect reuse if a different channel count changes
/// inserted layout reorders. It keys only on the actual operand shapes, never
/// the network width: SE 1x1 gates and downstream ceil-divided widths share
/// primitives.
fn step_key(identity: usize, node: &Node, tensors: &[Tensor], output: Shape) -> StepKey {
    let mut key = vec![output];
    let mut bindings = node.inputs.clone();
    key.extend(node.inputs.iter().map(|&id| tensors[id].shape));
    for post in &node.post_ops {
        if let PostOp::Binary { tensor, .. } = post {
            key.push(tensors[*tensor].shape);
            bindings.push(*tensor);
        }
    }
    (identity, bindings, key)
}

/// The last step that reads each tensor, counting steps from 1; the output
/// outlives every step.
fn last_uses(nodes: &[Node], count: usize) -> Vec<usize> {
    let mut last: Vec<usize> = (0..count).collect();
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
    last[count - 1] = count;
    last
}

/// A complete-channel concat at N=1 can share storage with its producers if
/// each input dies at the concat. Reserve the joined interval conservatively
/// and place the producers into its slices.
fn concat_views(
    nodes: &[Node],
    tensors: &[Tensor],
    shapes: &[Shape],
    last: &[usize],
) -> Result<Views> {
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
                    .ok_or(Error::size_overflow("concat placement"))?;
            }
            check_shape!(
                offset == tensors[dst].md.bytes()?,
                "concat placement extent"
            );
        }
    }
    Ok(Views {
        parent,
        relative,
        first,
    })
}

/// The merged byte ranges of blocked tensors whose padded channel lanes a
/// run must zero first; dense tensors are fully overwritten.
fn zero_ranges(tensors: &[Tensor], offsets: &[usize]) -> Result<Vec<(usize, usize)>> {
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
    Ok(merged)
}
