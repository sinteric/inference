//! Extra coverage for public boundaries, live residuals and broadcast post-ops.
use crate::cpu::onednn::conv::*;
use crate::cpu::onednn::{OnednnCpu, SessionOptions};
use std::sync::Arc;

use crate::cpu::kernels::SpatialMean;

fn cpu(limit: usize) -> Arc<OnednnCpu> {
    OnednnCpu::load_with(
        &crate::cpu::threads::ThreadBudget::new(1).unwrap(),
        SessionOptions {
            max_cpu_isa: crate::test_support::env::conv_isa(),
            workspace_limit: limit,
            ..Default::default()
        },
    )
    .unwrap()
}
fn identity(cpu: &Arc<OnednnCpu>, channels: usize) -> Arc<ConvWeights> {
    let mut w = vec![0.; channels * channels];
    for i in 0..channels {
        w[i * channels + i] = 1.;
    }
    ConvWeights::pack(
        cpu.clone(),
        Filters::new(channels, channels, [1, 1]),
        &w,
        None,
    )
    .unwrap()
}
fn input(shape: Shape) -> Vec<f32> {
    (0..shape.iter().product())
        .map(|i| (i % 61) as f32 / 61. - 0.5)
        .collect()
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let error = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(error < 5e-7, "max error {error}");
}

#[test]
fn residual_fusion_lifetime_and_spatial_reuse() {
    crate::test_support::own_process!(residual_fusion_lifetime_and_spatial_reuse);
    let cpu = cpu(1 << 20);
    let packed = identity(&cpu, 17);
    let mut conv = Node::new(
        Op::Conv {
            weights: packed,
            spatial: Spatial::default(),
        },
        vec![0],
    );
    // The binary source is kept live through the producing convolution and its
    // post-ops. The eltwise at index 0 also exercises binary argument index 1.
    conv.post_ops = vec![
        PostOp::Eltwise(Activation::Relu),
        PostOp::Binary {
            kind: Binary::Add,
            tensor: 0,
        },
    ];
    let nodes = vec![
        conv,
        Node::new(Op::Prelu { slopes: vec![0.25] }, vec![1]),
        Node::new(Op::Binary(Binary::Mul), vec![2, 0]),
        Node::new(Op::Concat { axis: 1 }, vec![3, 1, 0]),
    ];
    let net = ConvNet::new(cpu, nodes, 2).unwrap();
    for shape in [[2, 17, 3, 5], [1, 17, 7, 9], [2, 17, 3, 5]] {
        let x = input(shape);
        let (out_shape, output) = net.run(shape, &x).unwrap();
        assert_eq!(out_shape, [shape[0], 51, shape[2], shape[3]]);
        let mut expected = Vec::new();
        let batch = shape[1] * shape[2] * shape[3];
        for row in x.chunks(batch) {
            expected.extend(row.iter().map(|&v| {
                let r = v.max(0.) + v;
                v * if r > 0. { r } else { 0.25 * r }
            }));
            expected.extend(row.iter().map(|&v| v.max(0.) + v));
            expected.extend(row);
        }
        close(&output, &expected);
    }
    assert_eq!(net.cached_shapes().unwrap().len(), 2);
}

#[test]
fn binary_post_ops_broadcast_a_spatial_mean() {
    crate::test_support::own_process!(binary_post_ops_broadcast_a_spatial_mean);
    let cpu = cpu(1 << 20);
    let shape = [2, 16, 3, 5];
    let x = input(shape);
    // Per sample and channel: the squeeze-excite broadcast the models use.
    let axes = [false, false, true, true];
    let mut node = Node::new(
        Op::Conv {
            weights: identity(&cpu, 16),
            spatial: Spatial::default(),
        },
        vec![0],
    );
    node.post_ops = vec![
        PostOp::Eltwise(Activation::Relu),
        PostOp::Binary {
            kind: Binary::Mul,
            tensor: 1,
        },
        PostOp::Binary {
            kind: Binary::Add,
            tensor: 0,
        },
    ];
    let net = ConvNet::new(
        cpu.clone(),
        vec![
            Node::new(Op::SpatialMean(SpatialMean::Sequential), vec![0]),
            node,
        ],
        1,
    )
    .unwrap();
    let y = net.run(shape, &x).unwrap().1;
    let reduced: Shape = std::array::from_fn(|i| if axes[i] { 1 } else { shape[i] });
    let mut means = vec![0.; reduced.iter().product()];
    let mut counts = vec![0usize; means.len()];
    let index = |n: usize, c: usize, h: usize, w: usize| {
        ((n % reduced[0] * reduced[1] + c % reduced[1]) * reduced[2] + h % reduced[2]) * reduced[3]
            + w % reduced[3]
    };
    for n in 0..2 {
        for c in 0..16 {
            for h in 0..3 {
                for w in 0..5 {
                    let j = index(n, c, h, w);
                    means[j] += x[((n * 16 + c) * 3 + h) * 5 + w];
                    counts[j] += 1;
                }
            }
        }
    }
    for (m, count) in means.iter_mut().zip(counts) {
        *m /= count as f32;
    }
    let mut expected = vec![0.; x.len()];
    for n in 0..2 {
        for c in 0..16 {
            for h in 0..3 {
                for w in 0..5 {
                    let i = ((n * 16 + c) * 3 + h) * 5 + w;
                    expected[i] = x[i].max(0.) * means[index(n, c, h, w)] + x[i];
                }
            }
        }
    }
    close(&y, &expected);
}

#[test]
fn malformed_graphs_weights_and_spatial_options_return_errors() {
    crate::test_support::own_process!(malformed_graphs_weights_and_spatial_options_return_errors);
    let cpu = cpu(8192);
    for (oc, ic, k, groups, transpose, w, b) in [
        (0, 1, [1, 1], 1, false, vec![], None),
        (1, 0, [1, 1], 1, false, vec![], None),
        (1, 1, [0, 1], 1, false, vec![], None),
        (1, 1, [usize::MAX, 2], 1, false, vec![], None),
        (2, 3, [1, 1], 2, false, vec![0.; 6], None),
        (2, 2, [1, 1], 2, true, vec![0.; 2], None),
        (1, 1, [1, 1], 1, false, vec![0.; 2], None),
        (1, 1, [1, 1], 1, false, vec![0.], Some(vec![0.; 2])),
    ] {
        assert!(ConvWeights::pack(
            cpu.clone(),
            Filters {
                output: oc,
                input: ic,
                kernel: k,
                groups,
                transpose
            },
            &w,
            b.as_deref()
        )
        .is_err());
    }
    for spatial in [
        Spatial {
            strides: [0, 1],
            ..Default::default()
        },
        Spatial {
            dilations: [0, 1],
            ..Default::default()
        },
        Spatial {
            pads_begin: [usize::MAX, 0],
            ..Default::default()
        },
        Spatial {
            dilations: [usize::MAX, 1],
            ..Default::default()
        },
    ] {
        let weights =
            ConvWeights::pack(cpu.clone(), Filters::new(1, 1, [3, 3]), &[0.; 9], None).unwrap();
        let net = ConvNet::new(
            cpu.clone(),
            vec![Node::new(Op::Conv { weights, spatial }, vec![0])],
            1,
        )
        .unwrap();
        assert!(net.warmup([1, 1, 5, 5]).is_err());
    }
    for node in [
        Node::new(Op::Prelu { slopes: vec![0.] }, vec![]),
        Node::new(Op::Binary(Binary::Add), vec![0]),
        Node::new(Op::Prelu { slopes: vec![0.] }, vec![0, 0]),
        Node::new(Op::Concat { axis: 4 }, vec![0]),
        Node::new(Op::Prelu { slopes: vec![] }, vec![0]),
        Node::new(
            Op::Prelu {
                slopes: vec![1.; 2],
            },
            vec![0],
        ),
        Node::new(
            Op::Resize {
                height: 7,
                width: 7,
                linear: false,
            },
            vec![0],
        ),
    ] {
        if let Ok(net) = ConvNet::new(cpu.clone(), vec![node], 1) {
            assert!(net.warmup([1, 3, 5, 5]).is_err());
        }
    }
    let node = Node::new(Op::Prelu { slopes: vec![0.] }, vec![0]);
    assert!(ConvNet::new(cpu.clone(), vec![node.clone()], 0).is_err());
    let net = ConvNet::new(cpu, vec![node], 1).unwrap();
    assert!(net
        .warmup([1, 16, 128, 128])
        .is_err_and(|e| e.kind() == crate::ErrorKind::TooLarge));
}
