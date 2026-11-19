use rt_onednn::conv::*;
use rt_onednn::{CpuIsaCap, OnednnCpu, SessionOptions};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
static MAX_ERROR: AtomicU32 = AtomicU32::new(0);
static PRIMITIVES: AtomicUsize = AtomicUsize::new(0);

fn cpu() -> Arc<OnednnCpu> {
    static CPU: OnceLock<Arc<OnednnCpu>> = OnceLock::new();
    CPU.get_or_init(|| backend(1, 32 * 1024 * 1024)).clone()
}
fn backend(threads: usize, limit: usize) -> Arc<OnednnCpu> {
    OnednnCpu::load_with(
        &rt_threads::ThreadBudget::new(threads).unwrap(),
        SessionOptions {
            max_cpu_isa: CpuIsaCap::for_tests(),
            workspace_limit: limit,
            ..SessionOptions::default()
        },
    )
    .unwrap()
}
fn values(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 17 % 31) as f32 - 15.) / 31.).collect()
}
fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|v| v.is_finite()));
    let err = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    MAX_ERROR.fetch_max(err.to_bits(), Ordering::Relaxed);
    assert!(err <= tol, "max error {err} > {tol}");
}
fn no_ref(info: &PlanInfo) {
    if cfg!(target_arch = "x86_64") && std::is_x86_feature_detected!("avx512f") {
        PRIMITIVES.fetch_add(info.implementations.len(), Ordering::Relaxed);
        for name in &info.implementations {
            assert!(!name.contains("ref:"), "reference implementation: {name}");
        }
    }
}
fn run(cpu: Arc<OnednnCpu>, shape: Shape, input: &[f32], nodes: Vec<Node>) -> (Shape, Vec<f32>) {
    let net = ConvNet::new(cpu, nodes, 2).unwrap();
    let info = net
        .warmup(shape)
        .unwrap_or_else(|e| panic!("warmup shape={shape:?}: {e}"));
    no_ref(&info);
    net.run(shape, input).unwrap()
}
fn index(s: Shape, n: usize, c: usize, y: usize, x: usize) -> usize {
    ((n * s[1] + c) * s[2] + y) * s[3] + x
}
fn naive_conv(
    s: Shape,
    x: &[f32],
    oc: usize,
    k: [usize; 2],
    g: usize,
    p: &Spatial,
    w: &[f32],
    b: &[f32],
    deconv: bool,
) -> (Shape, Vec<f32>) {
    let mut d = [s[0], oc, 0, 0];
    for j in 0..2 {
        let e = (k[j] - 1) * p.dilations[j] + 1;
        d[j + 2] = if deconv {
            (s[j + 2] - 1) * p.strides[j] + e - p.pads_begin[j] - p.pads_end[j]
        } else {
            (s[j + 2] + p.pads_begin[j] + p.pads_end[j] - e) / p.strides[j] + 1
        };
    }
    let mut out = vec![0.; d.iter().product()];
    for n in 0..d[0] {
        for c in 0..oc {
            for y in 0..d[2] {
                for z in 0..d[3] {
                    let mut v = b[c];
                    for ic in 0..s[1] / g {
                        for ky in 0..k[0] {
                            for kx in 0..k[1] {
                                let (iy, ix) = if deconv {
                                    let yy = y as isize + p.pads_begin[0] as isize
                                        - (ky * p.dilations[0]) as isize;
                                    let xx = z as isize + p.pads_begin[1] as isize
                                        - (kx * p.dilations[1]) as isize;
                                    if yy < 0
                                        || xx < 0
                                        || yy % p.strides[0] as isize != 0
                                        || xx % p.strides[1] as isize != 0
                                    {
                                        continue;
                                    }
                                    (yy / p.strides[0] as isize, xx / p.strides[1] as isize)
                                } else {
                                    (
                                        (y * p.strides[0] + ky * p.dilations[0]) as isize
                                            - p.pads_begin[0] as isize,
                                        (z * p.strides[1] + kx * p.dilations[1]) as isize
                                            - p.pads_begin[1] as isize,
                                    )
                                };
                                if iy < 0 || ix < 0 || iy >= s[2] as isize || ix >= s[3] as isize {
                                    continue;
                                }
                                let channel = c / (oc / g) * (s[1] / g) + ic;
                                v += x[index(s, n, channel, iy as usize, ix as usize)]
                                    * w[((c * (s[1] / g) + ic) * k[0] + ky) * k[1] + kx];
                            }
                        }
                    }
                    out[index(d, n, c, y, z)] = v;
                }
            }
        }
    }
    (d, out)
}
fn convolution_cases(cpu: Arc<OnednnCpu>) -> Vec<f32> {
    let mut results = vec![];
    for (ic, oc, g, k, stride, dilation, deconv) in [
        (3, 17, 1, [3, 3], [2, 1], [1, 1], false),
        (16, 16, 1, [1, 3], [2, 1], [1, 1], false),
        (16, 32, 1, [3, 1], [1, 2], [1, 1], false),
        (16, 16, 1, [3, 3], [1, 1], [2, 2], false),
        (16, 16, 16, [5, 5], [1, 2], [1, 1], false),
        (17, 17, 17, [3, 3], [2, 1], [2, 2], false),
        (16, 32, 1, [1, 3], [1, 2], [1, 1], false),
        (16, 16, 1, [2, 2], [2, 2], [1, 1], true),
        (3, 7, 1, [2, 2], [2, 1], [1, 1], true),
    ] {
        let s = [2, ic, 9, 11];
        let x = values(s.iter().product());
        let w = values(oc * (ic / g) * k[0] * k[1]);
        let b = values(oc);
        let p = Spatial {
            strides: stride,
            dilations: dilation,
            pads_begin: if deconv {
                [0, 0]
            } else {
                [(k[0] - 1).min(2), (k[1] - 1).min(1)]
            },
            pads_end: if deconv {
                [0, 0]
            } else {
                [(k[0] - 1).min(1), (k[1] - 1).min(2)]
            },
        };
        let packed = ConvWeights::pack(cpu.clone(), oc, ic, k, g, deconv, &w, Some(&b));
        if deconv && cpu.conv_block() != 16 {
            // oneDNN names no 8-channel deconvolution weight layout.
            assert!(matches!(packed, Err(rt_core::Error::Unsupported { .. })));
            continue;
        }
        let weights = packed.unwrap();
        let (d, expected) = naive_conv(s, &x, oc, k, g, &p, &w, &b, deconv);
        let (got_d, got) = run(
            cpu.clone(),
            s,
            &x,
            vec![Node::new(
                Op::Conv {
                    weights,
                    spatial: p,
                },
                [0],
            )],
        );
        assert_eq!(got_d, d);
        close(&got, &expected, 8e-6);
        results.extend(got);
    }
    results
}
#[test]
fn convolution_depthwise_deconvolution() {
    convolution_cases(cpu());
}

/// Padding wider than oneDNN's jit register block sends a convolution to the
/// reference kernel; the plan then zero-pads the input itself and runs the
/// convolution unpadded. The dilated 3x3 at pad 12 is wider than that block
/// on every ISA; the 5x5 at pad 6 only on AVX2. Both must match the padded
/// convolution, and a fused residual and ReLU must survive the rewrite. The
/// residual comes from a blocked producer, as in a network: a plain one sends
/// the convolution to the reference kernel whatever its padding.
#[test]
fn wide_padding_runs_on_jit_kernels() {
    let cpu = cpu();
    for (c, k, dilation, h, w) in [(128, 3, 12, 45, 31), (32, 5, 3, 37, 29)] {
        let pad = (k - 1) / 2 * dilation;
        let s = [1, c, h, w];
        let x = values(s.iter().product());
        let scaled = |n: usize, by: usize| {
            values(n)
                .into_iter()
                .map(|v| v / by as f32)
                .collect::<Vec<_>>()
        };
        let (w1, b1) = (scaled(c * c, c), values(c));
        let (w2, b2) = (scaled(c * c * k * k, c * k), values(c));
        let none = Spatial {
            strides: [1, 1],
            dilations: [1, 1],
            pads_begin: [0, 0],
            pads_end: [0, 0],
        };
        let wide = Spatial {
            dilations: [dilation; 2],
            pads_begin: [pad; 2],
            pads_end: [pad; 2],
            ..none.clone()
        };
        let (_, mid) = naive_conv(s, &x, c, [1, 1], 1, &none, &w1, &b1, false);
        let (d, conv) = naive_conv(s, &mid, c, [k, k], 1, &wide, &w2, &b2, false);
        assert_eq!(d, s);
        let expected = conv
            .iter()
            .zip(&mid)
            .map(|(y, r)| (y + r).max(0.))
            .collect::<Vec<_>>();
        let pack = |k: usize, w: &[f32], b: &[f32]| {
            ConvWeights::pack(cpu.clone(), c, c, [k, k], 1, false, w, Some(b)).unwrap()
        };
        let first = Node::new(
            Op::Conv {
                weights: pack(1, &w1, &b1),
                spatial: none.clone(),
            },
            [0],
        );
        let mut second = Node::new(
            Op::Conv {
                weights: pack(k, &w2, &b2),
                spatial: wide,
            },
            [1],
        );
        second.post_ops = vec![
            PostOp::Binary {
                kind: Binary::Add,
                tensor: 1,
            },
            PostOp::Eltwise(Activation::Relu),
        ];
        let net = ConvNet::new(cpu.clone(), vec![first, second], 2).unwrap();
        let info = net.warmup(s).unwrap();
        for name in &info.implementations {
            assert!(!name.contains("ref:"), "{k}x{k} pad {pad}: {name}");
        }
        if k == 3 {
            assert!(
                info.implementations.iter().any(|i| i == "glue:zero_pad"),
                "{:?}",
                info.implementations
            );
        }
        let (got_d, got) = net.run(s, &x).unwrap();
        assert_eq!(got_d, s);
        close(&got, &expected, 2e-5);
    }
}

fn activation(a: Activation, x: f32) -> f32 {
    match a {
        Activation::Relu => x.max(0.),
        Activation::HardSwish => x * (x / 6. + 0.5).clamp(0., 1.),
        Activation::HardSigmoid { alpha, beta } => (alpha * x + beta).clamp(0., 1.),
        Activation::Logistic => 1. / (1. + (-x).exp()),
        Activation::Swish => x / (1. + (-x).exp()),
        Activation::GeluErf => {
            // Abramowitz-Stegun erf approximation, absolute error < 1.5e-7.
            let z = x / std::f32::consts::SQRT_2;
            let t = 1. / (1. + 0.3275911 * z.abs());
            let erf = (1.
                - (((((1.0614054 * t - 1.4531521) * t) + 1.4214138) * t - 0.28449672) * t
                    + 0.2548296)
                    * t
                    * (-z * z).exp())
                * z.signum();
            0.5 * x * (1. + erf)
        }
    }
}
fn fusion_cases(cpu: Arc<OnednnCpu>) -> Vec<f32> {
    let s = [1, 17, 5, 7];
    let x = values(s.iter().product());
    let mut results = vec![];
    let w = values(17 * 17);
    let b = values(17);
    let weights = ConvWeights::pack(cpu.clone(), 17, 17, [1, 1], 1, false, &w, Some(&b)).unwrap();
    let (_, base) = naive_conv(s, &x, 17, [1, 1], 1, &Spatial::default(), &w, &b, false);
    for a in [
        Activation::Relu,
        Activation::HardSwish,
        Activation::HardSigmoid {
            alpha: 0.1666667,
            beta: 0.5,
        },
        Activation::Logistic,
        Activation::Swish,
        Activation::GeluErf,
    ] {
        let mut n = Node::new(
            Op::Conv {
                weights: weights.clone(),
                spatial: Spatial::default(),
            },
            [0],
        );
        n.post_ops.push(PostOp::Eltwise(a));
        let (_, got) = run(cpu.clone(), s, &x, vec![n]);
        close(
            &got,
            &base.iter().map(|&v| activation(a, v)).collect::<Vec<_>>(),
            2e-6,
        );
        results.extend(got);
        let (_, got) = run(cpu.clone(), s, &x, vec![Node::new(Op::Eltwise(a), [0])]);
        close(
            &got,
            &x.iter().map(|&v| activation(a, v)).collect::<Vec<_>>(),
            2e-6,
        );
        results.extend(got);
    }
    for axes in [
        [false, false, true, true],
        [false, true, true, true],
        [false, false, false, false],
    ] {
        let shape: Shape = std::array::from_fn(|i| if axes[i] { 1 } else { s[i] });
        let mut mean = vec![0.; shape.iter().product()];
        let mut counts = vec![0; mean.len()];
        for c in 0..s[1] {
            for y in 0..s[2] {
                for z in 0..s[3] {
                    let dst = index(
                        shape,
                        0,
                        if axes[1] { 0 } else { c },
                        if axes[2] { 0 } else { y },
                        if axes[3] { 0 } else { z },
                    );
                    mean[dst] += x[index(s, 0, c, y, z)];
                    counts[dst] += 1;
                }
            }
        }
        for (v, c) in mean.iter_mut().zip(counts) {
            *v /= c as f32;
        }
        for kind in [Binary::Add, Binary::Mul] {
            let reduced = Node::new(Op::ReduceMean { axes }, [0]);
            let mut n = Node::new(
                Op::Conv {
                    weights: weights.clone(),
                    spatial: Spatial::default(),
                },
                [0],
            );
            n.post_ops = vec![
                PostOp::Eltwise(Activation::Relu),
                PostOp::Binary { kind, tensor: 1 },
                PostOp::Eltwise(Activation::Logistic),
            ];
            let (_, got) = run(cpu.clone(), s, &x, vec![reduced.clone(), n]);
            let mut expected = base.clone();
            for c in 0..s[1] {
                for y in 0..s[2] {
                    for z in 0..s[3] {
                        let i = index(s, 0, c, y, z);
                        let m = mean[index(
                            shape,
                            0,
                            if axes[1] { 0 } else { c },
                            if axes[2] { 0 } else { y },
                            if axes[3] { 0 } else { z },
                        )];
                        let v = expected[i].max(0.);
                        expected[i] = activation(
                            Activation::Logistic,
                            match kind {
                                Binary::Add => v + m,
                                Binary::Mul => v * m,
                            },
                        );
                    }
                }
            }
            close(&got, &expected, 2e-6);
            results.extend(got);
            let (_, got) = run(
                cpu.clone(),
                s,
                &x,
                vec![reduced, Node::new(Op::Binary(kind), [0, 1])],
            );
            for c in 0..s[1] {
                for y in 0..s[2] {
                    for z in 0..s[3] {
                        let i = index(s, 0, c, y, z);
                        let m = mean[index(
                            shape,
                            0,
                            if axes[1] { 0 } else { c },
                            if axes[2] { 0 } else { y },
                            if axes[3] { 0 } else { z },
                        )];
                        expected[i] = match kind {
                            Binary::Add => x[i] + m,
                            Binary::Mul => x[i] * m,
                        };
                    }
                }
            }
            close(&got, &expected, 2e-6);
            results.extend(got);
        }
    }
    results
}
#[test]
fn activations_reduction_binary_and_fusions() {
    fusion_cases(cpu());
}

fn other_cases(cpu: Arc<OnednnCpu>) -> Vec<f32> {
    let s = [1, 17, 6, 8];
    let x = values(s.iter().product());
    let mut results = vec![];
    for kind in [
        Pool::Max,
        Pool::AverageIncludePadding,
        Pool::AverageExcludePadding,
    ] {
        let p = Spatial {
            strides: [2, 1],
            pads_begin: [1, 1],
            pads_end: [1, 0],
            ..Spatial::default()
        };
        let (d, got) = run(
            cpu.clone(),
            s,
            &x,
            vec![Node::new(
                Op::Pool {
                    kind,
                    kernel: [3, 2],
                    spatial: p,
                },
                [0],
            )],
        );
        let mut want = vec![0.; got.len()];
        for c in 0..s[1] {
            for y in 0..d[2] {
                for z in 0..d[3] {
                    let mut v = if matches!(kind, Pool::Max) {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    };
                    let mut count = 0;
                    for ky in 0..3 {
                        for kx in 0..2 {
                            let yy = (y * 2 + ky) as isize - 1;
                            let xx = (z + kx) as isize - 1;
                            if yy >= 0 && xx >= 0 && yy < s[2] as isize && xx < s[3] as isize {
                                let a = x[index(s, 0, c, yy as usize, xx as usize)];
                                v = if matches!(kind, Pool::Max) {
                                    v.max(a)
                                } else {
                                    v + a
                                };
                                count += 1;
                            }
                        }
                    }
                    want[index(d, 0, c, y, z)] = match kind {
                        Pool::Max => v,
                        Pool::AverageIncludePadding => v / 6.,
                        Pool::AverageExcludePadding => v / count as f32,
                    };
                }
            }
        }
        close(&got, &want, 1e-6);
        results.extend(got);
    }
    for linear in [false, true] {
        let (d, got) = run(
            cpu.clone(),
            s,
            &x,
            vec![Node::new(
                Op::Resize {
                    height: 12,
                    width: 24,
                    linear,
                },
                [0],
            )],
        );
        let mut want = vec![0.; got.len()];
        for c in 0..s[1] {
            for y in 0..d[2] {
                for z in 0..d[3] {
                    want[index(d, 0, c, y, z)] = if !linear {
                        x[index(s, 0, c, y / 2, z / 3)]
                    } else {
                        let fy = ((y as f32 + 0.5) * s[2] as f32 / d[2] as f32 - 0.5).max(0.);
                        let fx = ((z as f32 + 0.5) * s[3] as f32 / d[3] as f32 - 0.5).max(0.);
                        let y0 = fy as usize;
                        let x0 = fx as usize;
                        let wy = fy - y0 as f32;
                        let wx = fx - x0 as f32;
                        let a = x[index(s, 0, c, y0, x0)] * (1. - wx)
                            + x[index(s, 0, c, y0, (x0 + 1).min(s[3] - 1))] * wx;
                        let b = x[index(s, 0, c, (y0 + 1).min(s[2] - 1), x0)] * (1. - wx)
                            + x[index(s, 0, c, (y0 + 1).min(s[2] - 1), (x0 + 1).min(s[3] - 1))]
                                * wx;
                        a * (1. - wy) + b * wy
                    };
                }
            }
        }
        close(&got, &want, 1e-6);
        results.extend(got);
    }
    let (_, got) = run(
        cpu.clone(),
        s,
        &x,
        vec![
            Node::new(Op::Reorder { blocked: false }, [0]),
            Node::new(Op::Reorder { blocked: true }, [1]),
            Node::new(Op::Concat { axis: 1 }, [0, 2]),
        ],
    );
    close(&got, &[x.clone(), x.clone()].concat(), 0.);
    results.extend(got);
    for slopes in [vec![0.2], values(17)] {
        let (_, got) = run(
            cpu.clone(),
            s,
            &x,
            vec![Node::new(
                Op::Prelu {
                    slopes: slopes.clone(),
                },
                [0],
            )],
        );
        let want: Vec<_> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if v >= 0. {
                    v
                } else {
                    v * slopes[if slopes.len() == 1 {
                        0
                    } else {
                        i / (s[2] * s[3])
                    }]
                }
            })
            .collect();
        close(&got, &want, 1e-6);
        results.extend(got);
    }
    results
}
#[test]
fn pool_resize_concat_reorder_prelu() {
    other_cases(cpu());
}

#[test]
fn lru_liveness_bounds_and_invalid_input() {
    let c = cpu();
    let nodes = (0..16)
        .map(|i| Node::new(Op::Eltwise(Activation::Relu), [i]))
        .collect();
    let net = ConvNet::new(c.clone(), nodes, 2).unwrap();
    let s = [1, 16, 8, 8];
    let info = net.warmup(s).unwrap();
    assert!(
        info.arena_bytes <= 3 * 16 * 8 * 8 * 4,
        "{}",
        info.arena_bytes
    );
    net.warmup([1, 16, 16, 8]).unwrap();
    net.warmup(s).unwrap();
    net.warmup([1, 16, 24, 8]).unwrap();
    assert_eq!(net.cached_shapes().unwrap(), vec![[1, 16, 24, 8], s]);
    assert!(net.run(s, &[]).is_err());
    assert!(net.warmup([0, 16, 8, 8]).is_err());
    assert!(net.warmup([1, 16, usize::MAX, 8]).is_err());
    assert!(matches!(
        net.warmup([1, 16, 1024, 1024]),
        Err(rt_core::Error::WorkspaceLimit { .. })
    ));
    assert!(
        ConvNet::new(c.clone(), vec![Node::new(Op::Binary(Binary::Add), [0])], 1)
            .unwrap()
            .warmup(s)
            .is_err()
    );
    assert!(
        ConvNet::new(c.clone(), vec![Node::new(Op::Concat { axis: 4 }, [0])], 1)
            .unwrap()
            .warmup(s)
            .is_err()
    );
    assert!(ConvNet::new(
        c.clone(),
        vec![Node::new(Op::Eltwise(Activation::Relu), [1])],
        1
    )
    .is_err());
    assert!(ConvNet::new(c.clone(), vec![], 1).is_err());
    assert!(ConvWeights::pack(c, 16, 16, [0, 3], 1, false, &[], None).is_err());
}

#[test]
fn memory_budget_evicts_conv_plans_without_changing_results() {
    let load = |memory_budget| {
        OnednnCpu::load_with(
            &rt_threads::ThreadBudget::new(1).unwrap(),
            SessionOptions {
                max_cpu_isa: CpuIsaCap::for_tests(),
                workspace_limit: 32 * 1024 * 1024,
                memory_budget,
                ..SessionOptions::default()
            },
        )
        .unwrap()
    };
    let net = |cpu: Arc<OnednnCpu>| {
        let p = Spatial {
            strides: [1, 1],
            pads_begin: [1, 1],
            pads_end: [1, 1],
            dilations: [1, 1],
        };
        let w = ConvWeights::pack(
            cpu.clone(),
            16,
            16,
            [3, 3],
            1,
            false,
            &values(16 * 16 * 9),
            Some(&values(16)),
        )
        .unwrap();
        let nodes = vec![
            Node::new(
                Op::Conv {
                    weights: w,
                    spatial: p,
                },
                [0],
            ),
            Node::new(Op::Eltwise(Activation::Relu), [1]),
        ];
        ConvNet::new(cpu, nodes, 8).unwrap()
    };
    let (plain_cpu, tight_cpu) = (load(0), load(1));
    let (plain, tight) = (net(plain_cpu), net(tight_cpu.clone()));
    let shapes = [[1, 16, 8, 8], [1, 16, 8, 24], [1, 16, 16, 8], [1, 16, 8, 8]];
    for s in shapes {
        let input = values(s.iter().product());
        let (a, b) = (plain.run(s, &input).unwrap(), tight.run(s, &input).unwrap());
        assert_eq!(a.0, b.0);
        assert!(
            a.1.iter()
                .map(|v| v.to_bits())
                .eq(b.1.iter().map(|v| v.to_bits())),
            "{s:?} differs under a memory budget"
        );
    }
    // A one-byte budget keeps only the plan of the last call.
    assert_eq!(plain.cached_shapes().unwrap().len(), 3);
    assert_eq!(tight.cached_shapes().unwrap(), vec![shapes[3]]);
    let stats = tight_cpu.stats_json();
    assert!(stats["budget_evictions"].as_u64().unwrap() >= 2, "{stats}");
}

// Each width has its own process because the library's pool width is immutable.
#[test]
fn thread_count_child() {
    let Some(width) = rt_core::env::TestOverrides::from_env().unwrap().ocr_threads else {
        return;
    };
    let c = backend(width, 32 * 1024 * 1024);
    let result = [
        convolution_cases(c.clone()),
        fusion_cases(c.clone()),
        other_cases(c),
    ]
    .concat();
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    for v in &result {
        hash.update(v.to_bits().to_le_bytes());
    }
    println!(
        "OCR_BITS {} {} max_abs={} primitives={}",
        result.len(),
        hex::encode(hash.finalize()),
        f32::from_bits(MAX_ERROR.load(Ordering::Relaxed)),
        PRIMITIVES.load(Ordering::Relaxed)
    );
}
#[test]
fn bitwise_identical_across_threads() {
    let mut baseline = None;
    for width in [1, 2, 4] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "thread_count_child", "--nocapture"])
            .env("PPOCR_TEST_THREADS", width.to_string())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        let fingerprint = stdout
            .lines()
            .find(|l| l.starts_with("OCR_BITS"))
            .unwrap()
            .to_string();
        eprintln!("threads={width}: {fingerprint}");
        if let Some(base) = &baseline {
            assert_eq!(base, &fingerprint);
        } else {
            baseline = Some(fingerprint);
        }
    }
}

#[path = "support/conv_boundaries.rs"]
mod boundaries;

#[test]
fn nonoverlapping_deconv_depth_to_space() {
    for oc in [1, 16, 17] {
        let s = [2, 16, 7, 11];
        let x = values(s.iter().product());
        let raw = values(oc * 16 * 4);
        let bias = values(oc);
        let mut w = vec![0.; raw.len()];
        for o in 0..oc {
            for i in 0..16 {
                for xy in 0..4 {
                    w[(o * 4 + xy) * 16 + i] = raw[(o * 16 + i) * 4 + xy];
                }
            }
        }
        let b = bias.iter().flat_map(|&v| [v; 4]).collect::<Vec<_>>();
        let weights = ConvWeights::pack(cpu(), oc * 4, 16, [1; 2], 1, false, &w, Some(&b)).unwrap();
        let mut conv = Node::new(
            Op::Conv {
                weights,
                spatial: Spatial::default(),
            },
            [0],
        );
        conv.post_ops.push(PostOp::Eltwise(Activation::Relu));
        let (shape, got) = run(cpu(), s, &x, vec![conv, Node::new(Op::DepthToSpace2x, [1])]);
        let (expected_shape, mut expected) = naive_conv(
            s,
            &x,
            oc,
            [2; 2],
            1,
            &Spatial {
                strides: [2; 2],
                ..Default::default()
            },
            &raw,
            &bias,
            true,
        );
        expected.iter_mut().for_each(|v| *v = v.max(0.));
        assert_eq!(shape, expected_shape);
        close(&got, &expected, 2e-6);
    }
    let net = ConvNet::new(cpu(), vec![Node::new(Op::DepthToSpace2x, [0])], 1).unwrap();
    assert!(net.warmup([1, 3, 8, 8]).is_err());
    assert!(net.warmup([1, 4, usize::MAX, 8]).is_err());
}

#[test]
fn pointwise_tiles_preserve_fused_values_and_partial_tiles() {
    let weights =
        ConvWeights::pack(cpu(), 17, 16, [1; 2], 1, false, &values(17 * 16), None).unwrap();
    let mut node = Node::new(
        Op::Conv {
            weights,
            spatial: Spatial::default(),
        },
        [0],
    );
    node.post_ops = vec![
        PostOp::ChannelBias(Arc::new(values(17))),
        PostOp::Scalar {
            kind: Binary::Mul,
            value: Arc::new(vec![0.75]),
        },
        PostOp::Eltwise(Activation::HardSwish),
    ];
    let plain = ConvNet::new(cpu(), vec![node.clone()], 2).unwrap();
    let tiled = ConvNet::new(cpu(), vec![node], 2)
        .unwrap()
        .with_pointwise_tiles();
    // Different network widths use the same pointwise native code. Exercise
    // multiple batches, spatial rows, exact tiles and a padded final tile.
    for shape in [
        [1, 16, 1, 17],
        [2, 16, 3, 517],
        [2, 16, 2, 256],
        [1, 16, 1, 17],
    ] {
        let input = values(shape.iter().product());
        let (a, expected) = plain.run(shape, &input).unwrap();
        let (b, got) = tiled.run(shape, &input).unwrap();
        assert_eq!(a, b);
        close(&got, &expected, 1e-6);
    }
}

#[test]
fn concat_producer_views_and_shared_workspace() {
    let mut first = Node::new(Op::Eltwise(Activation::Relu), [0]);
    first.post_ops.push(PostOp::Scalar {
        kind: Binary::Mul,
        value: Arc::new(vec![2.]),
    });
    let nodes = vec![
        first,
        Node::new(Op::Eltwise(Activation::Relu), [0]),
        Node::new(Op::Concat { axis: 1 }, [1, 2]),
    ];
    let net = ConvNet::new(cpu(), nodes, 2).unwrap();
    for n in [1, 2, 1] {
        let shape = [n, 16, 3, 5];
        let input = values(shape.iter().product());
        let info = net.warmup(shape).unwrap();
        assert_eq!(
            info.implementations
                .iter()
                .any(|s| s == "glue:concat_producer_views"),
            n == 1
        );
        let (_, got) = net.run(shape, &input).unwrap();
        let mut expected = Vec::new();
        for batch in input.as_chunks::<{ 16 * 3 * 5 }>().0 {
            expected.extend(batch.iter().map(|x| x.max(0.) * 2.));
            expected.extend(batch.iter().map(|x| x.max(0.)));
        }
        close(&got, &expected, 0.);
    }
}
