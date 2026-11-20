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
    /// A convolution with folded batch norm (`.w`, `.b`).
    fn conv(&mut self, id: usize, src: usize, filters: Filters, spatial: Spatial) -> Result<usize> {
        let Filters {
            output,
            input,
            kernel,
            groups,
            ..
        } = filters;
        let key = format!("uvdoc.conv{id:03}");
        let tensors = &self.assets.graph().tensors;
        if tensors.get(&format!("{key}.w"))
            != Some(&vec![output, input / groups, kernel[0], kernel[1]])
            || tensors.get(&format!("{key}.b")) != Some(&vec![output])
        {
            return Err(Error::model_files(format!(
                "compiled unwarper weight shape {key}"
            )));
        }
        let raw = self.assets.tensor(&format!("{key}.w"))?;
        let bias = self.assets.tensor(&format!("{key}.b"))?;
        let weights = ConvWeights::pack(self.cpu.clone(), filters, &raw, Some(&bias))?;
        Ok(self.push(Op::Conv { weights, spatial }, [src]))
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
    /// A 5x5 residual block: two convolutions with `dilation` over
    /// `channels`, and the identity as its shortcut.
    fn block(&mut self, src: usize, id: usize, channels: usize, dilation: usize) -> Result<usize> {
        let filters = Filters::square(channels, channels, 5);
        self.residual(
            src,
            src,
            id,
            filters,
            Spatial::square(1, dilation, 2 * dilation),
        )
    }
    /// The first block of a stage: it halves the resolution, and the 5x5
    /// stride-2 convolution `shortcut` replaces the identity.
    fn down_block(
        &mut self,
        src: usize,
        shortcut: usize,
        id: usize,
        input: usize,
        output: usize,
    ) -> Result<usize> {
        let (filters, spatial) = (Filters::square(output, input, 5), Spatial::square(2, 1, 2));
        let residual = self.conv(shortcut, src, filters, spatial)?;
        self.residual(src, residual, id, filters, spatial)
    }
    /// `relu(b(relu(a(src))) + residual)`: convolution `id` with `filters`
    /// and `spatial`, then convolution `id + 1`, a 5x5 over its output
    /// channels with the same dilation and padding at stride 1.
    fn residual(
        &mut self,
        src: usize,
        residual: usize,
        id: usize,
        filters: Filters,
        spatial: Spatial,
    ) -> Result<usize> {
        let a = self.conv(id, src, filters, spatial)?;
        let a = self.relu(a);
        let second = Filters::square(filters.output, filters.output, 5);
        let b = self.conv(
            id + 1,
            a,
            second,
            Spatial {
                strides: [1; 2],
                ..spatial
            },
        )?;
        Ok(self.add_relu(b, residual))
    }
    fn finish(self, capacity: usize) -> Result<ConvNet> {
        ConvNet::new(self.cpu, self.nodes, capacity)
    }
}

fn backbone(cpu: &Arc<OnednnCpu>, assets: &Assets, capacity: usize) -> Result<ConvNet> {
    let mut g = Graph::new(assets, cpu);
    let a = g.conv(0, 0, Filters::square(32, 3, 5), Spatial::square(2, 1, 2))?;
    let a = g.relu(a);
    let a = g.conv(1, a, Filters::square(32, 32, 5), Spatial::square(2, 1, 2))?;
    let mut a = g.relu(a);
    // Stage 1: 32 channels at 178 x 122.
    a = g.block(a, 2, 32, 1)?;
    a = g.block(a, 4, 32, 3)?;
    a = g.block(a, 6, 32, 3)?;
    // Stage 2: 64 channels at 89 x 61.
    a = g.down_block(a, 8, 9, 32, 64)?;
    for id in [11, 13, 15] {
        a = g.block(a, id, 64, 3)?;
    }
    // Stage 3: 128 channels at 45 x 31.
    a = g.down_block(a, 17, 18, 64, 128)?;
    for id in [20, 22, 24, 26, 28] {
        a = g.block(a, id, 128, 3)?;
    }
    // The bridge: each branch reads the stage-3 features.
    let c = BRIDGE_CHANNELS;
    let mut ends = vec![];
    for convs in BRANCHES {
        let mut b = a;
        for &(id, dilation) in convs {
            b = g.conv(
                id,
                b,
                Filters::square(c, c, 3),
                Spatial::square(1, dilation, dilation),
            )?;
            b = g.relu(b);
        }
        ends.push(b);
    }
    let bridge = g.push(Op::Concat { axis: 1 }, ends);
    let a = g.conv(
        42,
        bridge,
        Filters::square(c, BRANCHES.len() * c, 1),
        Spatial::default(),
    )?;
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
        .ok_or_else(|| Error::model_files("compiled unwarper PReLU"))?;
    if assets.graph().tensors.get(name) != Some(&vec![1, 1, 1]) {
        return Err(Error::model_files("compiled unwarper PReLU slope"));
    }
    let slope = assets.tensor(name)?;
    let mut g = Graph::new(assets, cpu);
    let a = g.conv(
        43,
        0,
        Filters::square(HEAD_CHANNELS, BRIDGE_CHANNELS, 5),
        Spatial::default(),
    )?;
    g.push(Op::Prelu { slopes: slope }, [a]);
    g.finish(capacity)
}

fn points(cpu: &Arc<OnednnCpu>, assets: &Assets, capacity: usize) -> Result<ConvNet> {
    let mut g = Graph::new(assets, cpu);
    g.conv(
        44,
        0,
        Filters::square(2, HEAD_CHANNELS, 5),
        Spatial::default(),
    )?;
    g.finish(capacity)
}

pub(crate) fn build(cpu: Arc<OnednnCpu>, assets: &Assets, capacity: usize) -> Result<Nets> {
    Ok(Nets {
        backbone: backbone(&cpu, assets, capacity)?,
        head: head(&cpu, assets, capacity)?,
        points: points(&cpu, assets, capacity)?,
    })
}
