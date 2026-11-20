//! Compiled PPLCNetV3 + SVTR convolution segments. The authenticated graph.json
//! identifies the export; it never drives execution.
use crate::cpu::onednn::conv::*;
use crate::cpu::onednn::OnednnCpu;
use crate::ppocr::weights::Assets;
use crate::{Error, Result};
use std::sync::Arc;

struct Graph<'a> {
    assets: &'a Assets,
    cpu: Arc<OnednnCpu>,
    nodes: Vec<Node>,
}
impl Graph<'_> {
    fn push(&mut self, op: Op, inputs: impl Into<Vec<usize>>) -> usize {
        self.nodes.push(Node::new(op, inputs));
        self.nodes.len()
    }
    fn conv(
        &mut self,
        id: usize,
        src: usize,
        oc: usize,
        ic: usize,
        kernel: [usize; 2],
        strides: [usize; 2],
        groups: usize,
    ) -> Result<usize> {
        let key = format!("rec.conv{id:03}");
        if self.assets.graph().tensors.get(&format!("{key}.w"))
            != Some(&vec![oc, ic / groups, kernel[0], kernel[1]])
        {
            return Err(Error::Asset(format!("compiled recognizer shape {key}")));
        }
        let raw = self.assets.tensor(&format!("{key}.w"))?;
        let bias = self.assets.tensor(&format!("{key}.b"))?;
        let tiny = matches!(id, 22 | 23 | 26 | 27);
        let weights = ConvWeights::pack(
            self.cpu.clone(),
            oc,
            ic,
            kernel,
            groups,
            false,
            &raw,
            if tiny { Some(&bias) } else { None },
        )?;
        let pad = kernel.map(|k| k / 2);
        let n = self.push(
            Op::Conv {
                weights,
                spatial: Spatial {
                    strides,
                    pads_begin: pad,
                    pads_end: pad,
                    ..Default::default()
                },
            },
            [src],
        );
        if !tiny {
            self.nodes[n - 1]
                .post_ops
                .push(PostOp::ChannelBias(Arc::new(bias)));
        }
        Ok(n)
    }
    fn act(&mut self, n: usize, act: Activation) -> usize {
        debug_assert_eq!(n, self.nodes.len());
        self.nodes[n - 1].post_ops.push(PostOp::Eltwise(act));
        n
    }
    fn affine(&mut self, n: usize, mul: usize, add: usize) -> Result<()> {
        for (id, kind) in [(mul, Binary::Mul), (add, Binary::Add)] {
            self.nodes[n - 1].post_ops.push(PostOp::Scalar {
                kind,
                value: Arc::new(self.assets.tensor(&format!("rec.const{id:03}"))?),
            });
        }
        Ok(())
    }
    fn learned_act(&mut self, n: usize, first: usize) -> Result<usize> {
        self.affine(n, first, first + 1)?;
        self.act(n, Activation::HardSwish);
        self.affine(n, first + 2, first + 3)?;
        Ok(n)
    }
    fn se(&mut self, n: usize, id: usize, channels: usize) -> Result<usize> {
        let mean = self.push(Op::SpatialMean(SpatialMean::Sequential), [n]);
        let a = self.conv(id, mean, channels / 4, channels, [1, 1], [1, 1], 1)?;
        self.act(a, Activation::Relu);
        let b = self.conv(id + 1, a, channels, channels / 4, [1, 1], [1, 1], 1)?;
        self.act(
            b,
            Activation::HardSigmoid {
                alpha: 0.1666667,
                beta: 0.5,
            },
        );
        Ok(self.push(Op::Binary(Binary::Mul), [n, b]))
    }
    fn finish(self, capacity: usize) -> Result<ConvNet> {
        ConvNet::new(self.cpu, self.nodes, capacity).map(ConvNet::with_pointwise_tiles)
    }
}
fn graph(cpu: Arc<OnednnCpu>, assets: &Assets) -> Graph<'_> {
    Graph {
        cpu,
        assets,
        nodes: vec![],
    }
}

pub(super) struct Segments {
    pub backbone: ConvNet,
    pub neck_in: ConvNet,
    pub neck_out: ConvNet,
    pub fusion: ConvNet,
}
pub(super) fn build(cpu: Arc<OnednnCpu>, assets: &Assets, capacity: usize) -> Result<Segments> {
    let mut g = graph(cpu.clone(), assets);
    let mut a = g.conv(0, 0, 16, 3, [3, 3], [2, 2], 1)?;
    // Conv000 contains folded BN/affine weights; Conv001 onward retain
    // explicit trained scalar affines around hard-swish in the pinned export.
    for (id, oc, ic, k, groups, pre, post) in [
        (1, 16, 16, 3, 16, 103, 105),
        (2, 32, 16, 1, 1, 127, 149),
        (3, 32, 32, 3, 32, 171, 193),
        (4, 64, 32, 1, 1, 207, 209),
        (5, 64, 64, 3, 64, 211, 213),
        (6, 64, 64, 1, 1, 107, 109),
    ] {
        a = g.conv(id, a, oc, ic, [k, k], [1, 1], groups)?;
        g.affine(a, pre, pre + 1)?;
        g.act(a, Activation::HardSwish);
        g.affine(a, post, post + 1)?;
    }
    for (id, oc, ic, k, stride, groups, first) in [
        (7, 64, 64, 3, [2, 1], 64, 111),
        (8, 128, 64, 1, [1, 1], 1, 115),
        (9, 128, 128, 3, [1, 1], 128, 119),
        (10, 128, 128, 1, [1, 1], 1, 123),
        (11, 128, 128, 3, [1, 2], 128, 129),
        (12, 240, 128, 1, [1, 1], 1, 133),
        (13, 240, 240, 5, [1, 1], 240, 137),
        (14, 240, 240, 1, [1, 1], 1, 141),
        (15, 240, 240, 5, [1, 1], 240, 145),
        (16, 240, 240, 1, [1, 1], 1, 151),
        (17, 240, 240, 5, [1, 1], 240, 155),
        (18, 240, 240, 1, [1, 1], 1, 159),
        (19, 240, 240, 5, [1, 1], 240, 163),
        (20, 240, 240, 1, [1, 1], 1, 167),
        (21, 240, 240, 5, [2, 1], 240, 173),
    ] {
        a = g.conv(id, a, oc, ic, [k, k], stride, groups)?;
        a = g.learned_act(a, first)?;
    }
    a = g.se(a, 22, 240)?;
    for (id, oc, ic, k, stride, groups, first) in [
        (24, 480, 240, 1, [1, 1], 1, 177),
        (25, 480, 480, 5, [1, 1], 480, 181),
    ] {
        a = g.conv(id, a, oc, ic, [k, k], stride, groups)?;
        a = g.learned_act(a, first)?;
    }
    a = g.se(a, 26, 480)?;
    for (id, k, stride, groups, first) in [
        (28, 1, [1, 1], 1, 185),
        (29, 5, [2, 1], 480, 189),
        (30, 1, [1, 1], 1, 195),
        (31, 5, [1, 1], 480, 199),
        (32, 1, [1, 1], 1, 203),
    ] {
        a = g.conv(id, a, 480, 480, [k, k], stride, groups)?;
        a = g.learned_act(a, first)?;
    }
    g.push(
        Op::Pool {
            kind: Pool::AverageExcludePadding,
            kernel: [3, 2],
            spatial: Spatial {
                strides: [3, 2],
                ..Default::default()
            },
        },
        [a],
    );
    let backbone = g.finish(capacity)?;
    let mut g = graph(cpu.clone(), assets);
    let a = g.conv(33, 0, 60, 480, [1, 3], [1, 1], 1)?;
    g.act(a, Activation::Swish);
    let a = g.conv(34, a, 120, 60, [1, 1], [1, 1], 1)?;
    g.act(a, Activation::Swish);
    let neck_in = g.finish(capacity)?;
    let mut g = graph(cpu.clone(), assets);
    let a = g.conv(35, 0, 480, 120, [1, 1], [1, 1], 1)?;
    g.act(a, Activation::Swish);
    let neck_out = g.finish(capacity)?;
    let mut g = graph(cpu, assets);
    let a = g.conv(36, 0, 60, 960, [1, 3], [1, 1], 1)?;
    g.act(a, Activation::Swish);
    let a = g.conv(37, a, 120, 60, [1, 1], [1, 1], 1)?;
    g.act(a, Activation::Swish);
    Ok(Segments {
        backbone,
        neck_in,
        neck_out,
        fusion: g.finish(capacity)?,
    })
}
