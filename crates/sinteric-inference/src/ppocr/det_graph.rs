//! Hand-written PP-OCRv6 tiny HGNet/LCNet + RSE-FPN + DB probability head.
//! Node numbers are stable imported weight names, not an executable JSON graph.
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
    /// A convolution read from `det.conv{id}` with `stride` and `pad` on
    /// both axes.
    fn conv(
        &mut self,
        id: usize,
        src: usize,
        filters: Filters,
        stride: usize,
        pad: usize,
    ) -> Result<usize> {
        // Convolutions whose bias goes inside the primitive, where that
        // matches the reference's rounding; the rest add it after the sum
        // (see `convolution`).
        const EARLY_BIAS: [usize; 26] = [
            6, 7, 17, 18, 28, 29, 36, 37, 47, 48, 55, 56, 58, 59, 61, 62, 64, 65, 68, 69, 72, 73,
            76, 77, 80, 81,
        ];
        let name = format!("det.conv{id:03}");
        self.convolution(
            name,
            src,
            filters,
            Spatial::square(stride, 1, pad),
            !EARLY_BIAS.contains(&id),
        )
    }
    fn convolution(
        &mut self,
        name: String,
        src: usize,
        filters: Filters,
        spatial: Spatial,
        late_bias: bool,
    ) -> Result<usize> {
        let Filters {
            output: oc,
            input: ic,
            kernel,
            groups,
            transpose,
        } = filters;
        let key = format!("{name}.w");
        let expected = if transpose {
            vec![ic, oc, kernel[0], kernel[1]]
        } else {
            vec![oc, ic / groups, kernel[0], kernel[1]]
        };
        if self.assets.graph().tensors.get(&key) != Some(&expected) {
            return Err(Error::model_files(format!(
                "compiled detector weight shape {key}"
            )));
        }
        let raw = self.assets.tensor(&key)?;
        // Non-overlapping 2x2/stride-2 transpose convolution equals a
        // 1x1 convolution to 4*OC followed by depth-to-space. Pack once.
        let (data, packed_oc, packed_k) = if transpose {
            crate::check_shape!(
                kernel == [2; 2]
                    && spatial.strides == [2; 2]
                    && groups == 1
                    && spatial.pads_begin == [0; 2]
                    && spatial.pads_end == [0; 2],
                "DB head transpose convolution"
            );
            let mut t = crate::try_vec(raw.len())?;
            for i in 0..ic {
                for o in 0..oc {
                    for xy in 0..4 {
                        t[(o * 4 + xy) * ic + i] = raw[(i * oc + o) * 4 + xy];
                    }
                }
            }
            (t, oc * 4, [1; 2])
        } else {
            (raw, oc, kernel)
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
        let packed = Filters {
            output: packed_oc,
            kernel: packed_k,
            transpose: false,
            ..filters
        };
        let early = if late_bias { None } else { bias.as_deref() };
        let w = ConvWeights::pack(self.cpu.clone(), packed, &data, early)?;
        let spatial = if transpose {
            Spatial {
                strides: [1; 2],
                ..spatial
            }
        } else {
            spatial
        };
        let node = self.push(
            Op::Conv {
                weights: w,
                spatial,
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
        let a = self.conv(id, mean, Filters::square(c / 4, c, 1), 1, 0)?;
        let a = self.act(a, Activation::Relu);
        let b = self.conv(id + 1, a, Filters::square(c, c / 4, 1), 1, 0)?;
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
        let a = self.conv(id, src, Filters::square(hidden, c, 1), 1, 0)?;
        let a = self.act(a, Activation::GeluErf);
        let b = self.conv(id + 1, a, Filters::square(output, hidden, 1), 1, 0)?;
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
            (self.conv(id, src, Filters::depthwise(ic, 5), 1, 2)?, id + 1)
        } else {
            (src, id)
        };
        let a = self.conv(id, src, Filters::square(oc, ic, 1), 1, 0)?;
        self.se(a, id + 1, oc, 0.2, true)
    }
}
impl Graph<'_> {
    /// The stem: a stride-2 convolution, a max-pool and convolution branch
    /// concatenated, and two more convolutions.
    fn stem(&mut self) -> Result<usize> {
        let a = self.conv(0, 0, Filters::square(16, 3, 3), 2, 1)?;
        let a = self.act(a, Activation::Relu);
        let b = self.convolution(
            "det.conv001".into(),
            a,
            Filters::square(8, 16, 2),
            Spatial {
                pads_end: [1; 2],
                ..Spatial::default()
            },
            true,
        )?;
        let b = self.act(b, Activation::Relu);
        let b = self.convolution(
            "det.conv002".into(),
            b,
            Filters::square(16, 8, 2),
            Spatial {
                pads_end: [1; 2],
                ..Spatial::default()
            },
            true,
        )?;
        let b = self.act(b, Activation::Relu);
        let pool = self.push(
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
        let cat = self.push(Op::Concat { axis: 1 }, [pool, b]);
        let a = self.conv(3, cat, Filters::square(16, 32, 3), 2, 1)?;
        let a = self.act(a, Activation::Relu);
        let a = self.conv(4, a, Filters::square(32, 16, 1), 1, 0)?;
        let a = self.act(a, Activation::Relu);
        Ok(a)
    }
    /// The four backbone stages from the stem; their outputs at strides 4,
    /// 8, 16 and 32.
    fn backbone(&mut self, a: usize) -> Result<[usize; 4]> {
        let a = self.conv(5, a, Filters::depthwise(32, 3), 1, 1)?;
        let a = self.se(a, 6, 32, 0.1666667, false)?;
        let a = self.mlp(a, 8, 32, 64, 32, true)?;
        let a = self.conv(10, a, Filters::depthwise(32, 3), 1, 1)?;
        let c2 = self.mlp(a, 11, 32, 64, 32, true)?;
        let a = self.conv(13, c2, Filters::depthwise(32, 3), 2, 1)?;
        let a = self.mlp(a, 14, 32, 64, 48, false)?;
        let a = self.conv(16, a, Filters::depthwise(48, 3), 1, 1)?;
        let a = self.se(a, 17, 48, 0.1666667, false)?;
        let a = self.mlp(a, 19, 48, 96, 48, true)?;
        let a = self.conv(21, a, Filters::depthwise(48, 3), 1, 1)?;
        let c3 = self.mlp(a, 22, 48, 96, 48, true)?;
        let a = self.conv(24, c3, Filters::depthwise(48, 3), 2, 1)?;
        let a = self.mlp(a, 25, 48, 96, 64, false)?;
        let a = self.conv(27, a, Filters::depthwise(64, 3), 1, 1)?;
        let a = self.se(a, 28, 64, 0.1666667, false)?;
        let a = self.mlp(a, 30, 64, 128, 64, true)?;
        let a = self.conv(32, a, Filters::depthwise(64, 3), 1, 1)?;
        let a = self.mlp(a, 33, 64, 128, 64, true)?;
        let a = self.conv(35, a, Filters::depthwise(64, 3), 1, 1)?;
        let a = self.se(a, 36, 64, 0.1666667, false)?;
        let a = self.mlp(a, 38, 64, 128, 64, true)?;
        let a = self.conv(40, a, Filters::depthwise(64, 3), 1, 1)?;
        let c4 = self.mlp(a, 41, 64, 128, 64, true)?;
        let a = self.conv(43, c4, Filters::depthwise(64, 3), 2, 1)?;
        let a = self.mlp(a, 44, 64, 128, 160, false)?;
        let a = self.conv(46, a, Filters::depthwise(160, 3), 1, 1)?;
        let a = self.se(a, 47, 160, 0.1666667, false)?;
        let a = self.mlp(a, 49, 160, 320, 160, true)?;
        let a = self.conv(51, a, Filters::depthwise(160, 3), 1, 1)?;
        let c5 = self.mlp(a, 52, 160, 320, 160, true)?;
        Ok([c2, c3, c4, c5])
    }
    /// The RSE-FPN neck: the four stage outputs fused top-down, each refined,
    /// brought to stride 4 and concatenated.
    fn fpn(&mut self, [c2, c3, c4, c5]: [usize; 4]) -> Result<usize> {
        let p5 = self.rse(c5, 54, 160, 64, false)?;
        let p4 = self.rse(c4, 57, 64, 64, false)?;
        let p3 = self.rse(c3, 60, 48, 64, false)?;
        let p2 = self.rse(c2, 63, 32, 64, false)?;
        let up = self.up(p5, 2);
        let p4 = self.add(p4, up);
        let up = self.up(p4, 2);
        let p3 = self.add(p3, up);
        let up = self.up(p3, 2);
        let p2 = self.add(p2, up);
        let p5 = self.rse(p5, 66, 64, 16, true)?;
        let p4 = self.rse(p4, 70, 64, 16, true)?;
        let p3 = self.rse(p3, 74, 64, 16, true)?;
        let p2 = self.rse(p2, 78, 64, 16, true)?;
        let p5 = self.up(p5, 8);
        let p4 = self.up(p4, 4);
        let p3 = self.up(p3, 2);
        let cat = self.push(Op::Concat { axis: 1 }, [p5, p4, p3, p2]);
        Ok(cat)
    }
    /// The DB head: a convolution and two 2x upsampling transposed
    /// convolutions to the probability map's logits.
    fn head(&mut self, cat: usize) -> Result<()> {
        let a = self.conv(82, cat, Filters::square(16, 64, 3), 1, 1)?;
        let a = self.act(a, Activation::Relu);
        let a = self.convolution(
            "det.convtranspose000".into(),
            a,
            Filters {
                transpose: true,
                ..Filters::square(16, 16, 2)
            },
            Spatial {
                strides: [2; 2],
                ..Spatial::default()
            },
            false,
        )?;
        let a = self.act(a, Activation::Relu);
        let a = self.push(Op::DepthToSpace2x, [a]);
        let a = self.convolution(
            "det.convtranspose001".into(),
            a,
            Filters {
                transpose: true,
                ..Filters::square(1, 16, 2)
            },
            Spatial {
                strides: [2; 2],
                ..Spatial::default()
            },
            false,
        )?;
        self.push(Op::DepthToSpace2x, [a]);
        Ok(())
    }
}
pub(super) fn build(cpu: Arc<OnednnCpu>, assets: &Assets, capacity: usize) -> Result<ConvNet> {
    let mut g = Graph {
        assets,
        cpu: cpu.clone(),
        nodes: vec![],
    };
    let stem = g.stem()?;
    let levels = g.backbone(stem)?;
    let fused = g.fpn(levels)?;
    g.head(fused)?;
    // The final Sigmoid runs outside the graph as ORT's exact MLAS kernel
    // (`Detector::forward`): DB thresholds compare probabilities directly.
    ConvNet::new(cpu, g.nodes, capacity)
}
