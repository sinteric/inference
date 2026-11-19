//! Hand-written PP-OCRv6 tiny HGNet/LCNet + RSE-FPN + DB probability head.
//! Node numbers are stable imported weight names, not an executable JSON graph.
use crate::weights::Assets;
use crate::{Error, Result};
use rt_onednn::conv::*;
use rt_onednn::OnednnCpu;
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
        k: usize,
        stride: usize,
        pad: usize,
        groups: usize,
    ) -> Result<usize> {
        self.convolution(
            format!("det.conv{id:03}"),
            src,
            oc,
            ic,
            k,
            stride,
            [pad; 2],
            [pad; 2],
            groups,
            false,
            ![
                6, 7, 17, 18, 28, 29, 36, 37, 47, 48, 55, 56, 58, 59, 61, 62, 64, 65, 68, 69, 72,
                73, 76, 77, 80, 81,
            ]
            .contains(&id),
        )
    }
    fn convolution(
        &mut self,
        name: String,
        src: usize,
        oc: usize,
        ic: usize,
        k: usize,
        stride: usize,
        begin: [usize; 2],
        end: [usize; 2],
        groups: usize,
        transpose: bool,
        late_bias: bool,
    ) -> Result<usize> {
        let key = format!("{name}.w");
        let expected = if transpose {
            vec![ic, oc, k, k]
        } else {
            vec![oc, ic / groups, k, k]
        };
        if self.assets.graph().tensors.get(&key) != Some(&expected) {
            return Err(Error::Asset(format!(
                "compiled detector weight shape {key}"
            )));
        }
        let raw = self.assets.tensor(&key)?;
        // Non-overlapping 2x2/stride-2 transpose convolution equals a
        // 1x1 convolution to 4*OC followed by depth-to-space. Pack once.
        let (data, packed_oc, packed_k) = if transpose {
            rt_core::check_shape!(
                k == 2 && stride == 2 && groups == 1 && begin == [0; 2] && end == [0; 2],
                "DB head transpose convolution"
            );
            let mut t = rt_core::try_vec(raw.len())?;
            for i in 0..ic {
                for o in 0..oc {
                    for xy in 0..4 {
                        t[(o * 4 + xy) * ic + i] = raw[(i * oc + o) * 4 + xy];
                    }
                }
            }
            (t, oc * 4, 1)
        } else {
            (raw, oc, k)
        };
        let bias = if self
            .assets
            .graph()
            .tensors
            .contains_key(&format!("{name}.b"))
        {
            Some(self.assets.tensor(&format!("{name}.b"))?)
        } else {
            None
        };
        let bias = if transpose {
            bias.map(|b| b.into_iter().flat_map(|v| [v; 4]).collect::<Vec<_>>())
        } else {
            bias
        };
        let late_bias = late_bias || transpose;
        let w = ConvWeights::pack(
            self.cpu.clone(),
            packed_oc,
            ic,
            [packed_k, packed_k],
            groups,
            false,
            &data,
            if late_bias { None } else { bias.as_deref() },
        )?;
        let node = self.push(
            Op::Conv {
                weights: w,
                spatial: Spatial {
                    strides: if transpose { [1; 2] } else { [stride; 2] },
                    pads_begin: begin,
                    pads_end: end,
                    ..Spatial::default()
                },
            },
            [src],
        );
        // MLAS adds bias after the convolution sum. In-primitive oneDNN 1x1
        // convolution initializes its accumulator with bias; on near-constant
        // page backgrounds that changes the long FP32 global-pool rounding.
        if late_bias {
            if let Some(bias) = bias {
                self.nodes[node - 1]
                    .post_ops
                    .push(PostOp::ChannelBias(Arc::new(bias)));
            }
        }
        Ok(node)
    }
    fn act(&mut self, src: usize, a: Activation) -> usize {
        debug_assert_eq!(src, self.nodes.len());
        self.nodes[src - 1].post_ops.push(PostOp::Eltwise(a));
        src
    }
    fn add(&mut self, a: usize, b: usize) -> usize {
        self.push(Op::Binary(Binary::Add), [a, b])
    }
    fn up(&mut self, src: usize, scale: usize) -> usize {
        self.push(Op::UpsampleNearest { scales: [scale; 2] }, [src])
    }
    fn se(&mut self, src: usize, id: usize, c: usize, alpha: f32, residual: bool) -> Result<usize> {
        // The five backbone ReduceMean nodes use the reference build's Eigen
        // AVX order; RSE-FPN GlobalAveragePool uses sequential accumulation.
        let order = if id < 54 {
            SpatialMean::EigenAvx
        } else {
            SpatialMean::Sequential
        };
        let mean = self.push(Op::SpatialMean(order), [src]);
        let a = self.conv(id, mean, c / 4, c, 1, 1, 0, 1)?;
        let a = self.act(a, Activation::Relu);
        let b = self.conv(id + 1, a, c, c / 4, 1, 1, 0, 1)?;
        let b = self.act(b, Activation::HardSigmoid { alpha, beta: 0.5 });
        let m = self.push(Op::Binary(Binary::Mul), [src, b]);
        if residual {
            self.nodes[m - 1].post_ops.push(PostOp::Binary {
                kind: Binary::Add,
                tensor: src,
            });
        }
        Ok(m)
    }
    fn mlp(
        &mut self,
        src: usize,
        id: usize,
        c: usize,
        hidden: usize,
        output: usize,
        residual: bool,
    ) -> Result<usize> {
        let a = self.conv(id, src, hidden, c, 1, 1, 0, 1)?;
        let a = self.act(a, Activation::GeluErf);
        let b = self.conv(id + 1, a, output, hidden, 1, 1, 0, 1)?;
        if residual {
            self.nodes[b - 1].post_ops.push(PostOp::Binary {
                kind: Binary::Add,
                tensor: src,
            });
        }
        Ok(b)
    }
    fn rse(
        &mut self,
        src: usize,
        id: usize,
        ic: usize,
        oc: usize,
        depthwise: bool,
    ) -> Result<usize> {
        let (src, id) = if depthwise {
            (self.conv(id, src, ic, ic, 5, 1, 2, ic)?, id + 1)
        } else {
            (src, id)
        };
        let a = self.conv(id, src, oc, ic, 1, 1, 0, 1)?;
        self.se(a, id + 1, oc, 0.2, true)
    }
}
pub(super) fn build(cpu: Arc<OnednnCpu>, assets: &Assets, capacity: usize) -> Result<ConvNet> {
    let mut g = Graph {
        assets,
        cpu: cpu.clone(),
        nodes: vec![],
    };
    let a = g.conv(0, 0, 16, 3, 3, 2, 1, 1)?;
    let a = g.act(a, Activation::Relu);
    let b = g.convolution(
        "det.conv001".into(),
        a,
        8,
        16,
        2,
        1,
        [0; 2],
        [1; 2],
        1,
        false,
        true,
    )?;
    let b = g.act(b, Activation::Relu);
    let b = g.convolution(
        "det.conv002".into(),
        b,
        16,
        8,
        2,
        1,
        [0; 2],
        [1; 2],
        1,
        false,
        true,
    )?;
    let b = g.act(b, Activation::Relu);
    let pool = g.push(
        Op::Pool {
            kind: Pool::Max,
            kernel: [2; 2],
            spatial: Spatial {
                pads_end: [1; 2],
                ..Spatial::default()
            },
        },
        [a],
    );
    let cat = g.push(Op::Concat { axis: 1 }, [pool, b]);
    let a = g.conv(3, cat, 16, 32, 3, 2, 1, 1)?;
    let a = g.act(a, Activation::Relu);
    let a = g.conv(4, a, 32, 16, 1, 1, 0, 1)?;
    let a = g.act(a, Activation::Relu);
    let a = g.conv(5, a, 32, 32, 3, 1, 1, 32)?;
    let a = g.se(a, 6, 32, 0.1666667, false)?;
    let a = g.mlp(a, 8, 32, 64, 32, true)?;
    let a = g.conv(10, a, 32, 32, 3, 1, 1, 32)?;
    let c2 = g.mlp(a, 11, 32, 64, 32, true)?;
    let a = g.conv(13, c2, 32, 32, 3, 2, 1, 32)?;
    let a = g.mlp(a, 14, 32, 64, 48, false)?;
    let a = g.conv(16, a, 48, 48, 3, 1, 1, 48)?;
    let a = g.se(a, 17, 48, 0.1666667, false)?;
    let a = g.mlp(a, 19, 48, 96, 48, true)?;
    let a = g.conv(21, a, 48, 48, 3, 1, 1, 48)?;
    let c3 = g.mlp(a, 22, 48, 96, 48, true)?;
    let a = g.conv(24, c3, 48, 48, 3, 2, 1, 48)?;
    let a = g.mlp(a, 25, 48, 96, 64, false)?;
    let a = g.conv(27, a, 64, 64, 3, 1, 1, 64)?;
    let a = g.se(a, 28, 64, 0.1666667, false)?;
    let a = g.mlp(a, 30, 64, 128, 64, true)?;
    let a = g.conv(32, a, 64, 64, 3, 1, 1, 64)?;
    let a = g.mlp(a, 33, 64, 128, 64, true)?;
    let a = g.conv(35, a, 64, 64, 3, 1, 1, 64)?;
    let a = g.se(a, 36, 64, 0.1666667, false)?;
    let a = g.mlp(a, 38, 64, 128, 64, true)?;
    let a = g.conv(40, a, 64, 64, 3, 1, 1, 64)?;
    let c4 = g.mlp(a, 41, 64, 128, 64, true)?;
    let a = g.conv(43, c4, 64, 64, 3, 2, 1, 64)?;
    let a = g.mlp(a, 44, 64, 128, 160, false)?;
    let a = g.conv(46, a, 160, 160, 3, 1, 1, 160)?;
    let a = g.se(a, 47, 160, 0.1666667, false)?;
    let a = g.mlp(a, 49, 160, 320, 160, true)?;
    let a = g.conv(51, a, 160, 160, 3, 1, 1, 160)?;
    let c5 = g.mlp(a, 52, 160, 320, 160, true)?;
    let p5 = g.rse(c5, 54, 160, 64, false)?;
    let p4 = g.rse(c4, 57, 64, 64, false)?;
    let p3 = g.rse(c3, 60, 48, 64, false)?;
    let p2 = g.rse(c2, 63, 32, 64, false)?;
    let up = g.up(p5, 2);
    let p4 = g.add(p4, up);
    let up = g.up(p4, 2);
    let p3 = g.add(p3, up);
    let up = g.up(p3, 2);
    let p2 = g.add(p2, up);
    let p5 = g.rse(p5, 66, 64, 16, true)?;
    let p4 = g.rse(p4, 70, 64, 16, true)?;
    let p3 = g.rse(p3, 74, 64, 16, true)?;
    let p2 = g.rse(p2, 78, 64, 16, true)?;
    let p5 = g.up(p5, 8);
    let p4 = g.up(p4, 4);
    let p3 = g.up(p3, 2);
    let cat = g.push(Op::Concat { axis: 1 }, [p5, p4, p3, p2]);
    let a = g.conv(82, cat, 16, 64, 3, 1, 1, 1)?;
    let a = g.act(a, Activation::Relu);
    let a = g.convolution(
        "det.convtranspose000".into(),
        a,
        16,
        16,
        2,
        2,
        [0; 2],
        [0; 2],
        1,
        true,
        false,
    )?;
    let a = g.act(a, Activation::Relu);
    let a = g.push(Op::DepthToSpace2x, [a]);
    let a = g.convolution(
        "det.convtranspose001".into(),
        a,
        1,
        16,
        2,
        2,
        [0; 2],
        [0; 2],
        1,
        true,
        false,
    )?;
    g.push(Op::DepthToSpace2x, [a]);
    // The final Sigmoid runs outside the graph as ORT's exact MLAS kernel
    // (`Detector::forward`): DB thresholds compare probabilities directly.
    ConvNet::new(cpu, g.nodes, capacity)
}
