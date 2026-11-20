//! Hand-written UVDoc: a dilated residual backbone, a six-branch dilated
//! bridge and the 2D point head. Node numbers are stable imported weight
//! names; the authenticated graph.json identifies the export and never drives
//! execution.
use crate::cpu::onednn::conv::*;
use crate::cpu::onednn::OnednnCpu;
use crate::ppocr::weights::Assets;
use crate::{Error, Result};
use std::sync::Arc;

/// The export resizes every page to this before the backbone.
pub(crate) const INPUT_HEIGHT: usize = 712;
pub(crate) const INPUT_WIDTH: usize = 488;
/// The 2D grid the point head answers: four stride-2 stages of 712 x 488.
pub(crate) const GRID_HEIGHT: usize = 45;
pub(crate) const GRID_WIDTH: usize = 31;
/// Channels the bridge hands the head, and the head's hidden width.
pub(crate) const BRIDGE_CHANNELS: usize = 128;
pub(crate) const HEAD_CHANNELS: usize = 32;
/// The head's two 5x5 convolutions read reflect-padded inputs.
pub(crate) const HEAD_PAD: usize = 2;
/// The three convolution networks. The head's reflect pads run on the host
/// (`crate::image::reflect_pad_nchw`).
pub(crate) struct Nets {
    /// `[1, 3, 712, 488]` -> `[1, 128, 45, 31]`: the stem, three residual
    /// stages, the bridge's six dilated branches concatenated in order, and
    /// conv042 over them.
    pub backbone: ConvNet,
    /// `[1, 128, 49, 35]` -> `[1, 32, 45, 31]`: conv043, then PReLU.
    pub head: ConvNet,
    /// `[1, 32, 49, 35]` -> `[1, 2, 45, 31]`: conv044, x then y.
    pub points: ConvNet,
}

struct Graph<'a> {
    assets: &'a Assets,
    cpu: Arc<OnednnCpu>,
    nodes: Vec<Node>,
}
impl Graph<'_> {
    fn new<'a>(assets: &'a Assets, cpu: &Arc<OnednnCpu>) -> Graph<'a> {
        Graph {
            assets,
            cpu: cpu.clone(),
            nodes: vec![],
        }
    }
    fn push(&mut self, op: Op, inputs: impl Into<Vec<usize>>) -> usize {
        self.nodes.push(Node::new(op, inputs));
        self.nodes.len()
    }
    /// A square `k x k` convolution with folded batch norm (`.w`, `.b`).
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        id: usize,
        src: usize,
        oc: usize,
        ic: usize,
        k: usize,
        stride: usize,
        dilation: usize,
        pad: usize,
    ) -> Result<usize> {
        let key = format!("uvdoc.conv{id:03}");
        let tensors = &self.assets.graph().tensors;
        if tensors.get(&format!("{key}.w")) != Some(&vec![oc, ic, k, k])
            || tensors.get(&format!("{key}.b")) != Some(&vec![oc])
        {
            return Err(Error::Asset(format!(
                "compiled unwarper weight shape {key}"
            )));
        }
        let raw = self.assets.tensor(&format!("{key}.w"))?;
        let bias = self.assets.tensor(&format!("{key}.b"))?;
        let weights = ConvWeights::pack(
            self.cpu.clone(),
            oc,
            ic,
            [k, k],
            1,
            false,
            &raw,
            Some(&bias),
        )?;
        Ok(self.push(
            Op::Conv {
                weights,
                spatial: Spatial {
                    strides: [stride; 2],
                    dilations: [dilation; 2],
                    pads_begin: [pad; 2],
                    pads_end: [pad; 2],
                },
            },
            [src],
        ))
    }
    fn relu(&mut self, src: usize) -> usize {
        debug_assert_eq!(src, self.nodes.len());
        self.nodes[src - 1]
            .post_ops
            .push(PostOp::Eltwise(Activation::Relu));
        src
    }
    /// `relu(src + residual)` fused into the producer of `src`.
    fn add_relu(&mut self, src: usize, residual: usize) -> usize {
        debug_assert_eq!(src, self.nodes.len());
        let post = &mut self.nodes[src - 1].post_ops;
        post.push(PostOp::Binary {
            kind: Binary::Add,
            tensor: residual,
        });
        post.push(PostOp::Eltwise(Activation::Relu));
        src
    }
    /// A 5x5 residual block. `shortcut` names the downsampling convolution
    /// that replaces the identity on the first block of a stage.
    fn block(
        &mut self,
        src: usize,
        id: usize,
        ic: usize,
        oc: usize,
        dilation: usize,
        stride: usize,
        shortcut: Option<usize>,
    ) -> Result<usize> {
        let pad = 2 * dilation;
        let residual = match shortcut {
            Some(s) => self.conv(s, src, oc, ic, 5, stride, 1, 2)?,
            None => src,
        };
        let a = self.conv(id, src, oc, ic, 5, stride, dilation, pad)?;
        let a = self.relu(a);
        let b = self.conv(id + 1, a, oc, oc, 5, 1, dilation, pad)?;
        Ok(self.add_relu(b, residual))
    }
    fn finish(self, capacity: usize) -> Result<ConvNet> {
        ConvNet::new(self.cpu, self.nodes, capacity)
    }
}

fn backbone(cpu: &Arc<OnednnCpu>, assets: &Assets, capacity: usize) -> Result<ConvNet> {
    let mut g = Graph::new(assets, cpu);
    let a = g.conv(0, 0, 32, 3, 5, 2, 1, 2)?;
    let a = g.relu(a);
    let a = g.conv(1, a, 32, 32, 5, 2, 1, 2)?;
    let mut a = g.relu(a);
    // Stage 1: 32 channels at 178 x 122.
    a = g.block(a, 2, 32, 32, 1, 1, None)?;
    a = g.block(a, 4, 32, 32, 3, 1, None)?;
    a = g.block(a, 6, 32, 32, 3, 1, None)?;
    // Stage 2: 64 channels at 89 x 61.
    a = g.block(a, 9, 32, 64, 1, 2, Some(8))?;
    for id in [11, 13, 15] {
        a = g.block(a, id, 64, 64, 3, 1, None)?;
    }
    // Stage 3: 128 channels at 45 x 31.
    a = g.block(a, 18, 64, 128, 1, 2, Some(17))?;
    for id in [20, 22, 24, 26, 28] {
        a = g.block(a, id, 128, 128, 3, 1, None)?;
    }
    // The bridge: each branch reads the stage-3 features.
    let c = BRIDGE_CHANNELS;
    let mut ends = vec![];
    for convs in BRANCHES {
        let mut b = a;
        for &(id, dilation) in convs {
            b = g.conv(id, b, c, c, 3, 1, dilation, dilation)?;
            b = g.relu(b);
        }
        ends.push(b);
    }
    let bridge = g.push(Op::Concat { axis: 1 }, ends);
    let a = g.conv(42, bridge, c, BRANCHES.len() * c, 1, 1, 1, 0)?;
    g.relu(a);
    g.finish(capacity)
}

/// The bridge's branches: chains of 3x3 dilated convolutions padded by their
/// dilation, each followed by ReLU, as (weight node, dilation). Padding
/// wider than oneDNN's jit kernels take is applied by the plan itself
/// (`crate::cpu::onednn::conv::ConvNet`).
const BRANCHES: [&[(usize, usize)]; 6] = [
    &[(30, 1)],
    &[(31, 2)],
    &[(32, 5)],
    &[(33, 8), (34, 3), (35, 2)],
    &[(36, 12), (37, 7), (38, 4)],
    &[(39, 18), (40, 12), (41, 6)],
];

fn head(cpu: &Arc<OnednnCpu>, assets: &Assets, capacity: usize) -> Result<ConvNet> {
    // The one PReLU shares a single slope across its channels.
    let name = assets
        .graph()
        .nodes
        .iter()
        .find(|n| n.op == "PRelu")
        .and_then(|n| n.inputs.get(1))
        .ok_or_else(|| Error::Asset("compiled unwarper PReLU".into()))?;
    if assets.graph().tensors.get(name) != Some(&vec![1, 1, 1]) {
        return Err(Error::Asset("compiled unwarper PReLU slope".into()));
    }
    let slope = assets.tensor(name)?;
    let mut g = Graph::new(assets, cpu);
    let a = g.conv(43, 0, HEAD_CHANNELS, BRIDGE_CHANNELS, 5, 1, 1, 0)?;
    g.push(Op::Prelu { slopes: slope }, [a]);
    g.finish(capacity)
}

fn points(cpu: &Arc<OnednnCpu>, assets: &Assets, capacity: usize) -> Result<ConvNet> {
    let mut g = Graph::new(assets, cpu);
    g.conv(44, 0, 2, HEAD_CHANNELS, 5, 1, 1, 0)?;
    g.finish(capacity)
}

pub(crate) fn build(cpu: Arc<OnednnCpu>, assets: &Assets, capacity: usize) -> Result<Nets> {
    Ok(Nets {
        backbone: backbone(&cpu, assets, capacity)?,
        head: head(&cpu, assets, capacity)?,
        points: points(&cpu, assets, capacity)?,
    })
}
