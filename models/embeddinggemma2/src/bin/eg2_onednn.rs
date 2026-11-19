//! Persistent JSON driver for EmbeddingGemma 2 (text tower).
use anyhow::{bail, ensure, Context, Result};
use embeddinggemma2::{mrl_truncate, Batch, Frontend, Model, Precision, PromptKind, EMBED_DIM};
use rt_core::env::{EnvOverrides, ModelKind};
use rt_onednn::{CacheOptions, CpuIsaCap, OnednnCpu, SessionOptions, TuningPlan};
use rt_threads::{CpuTopology, ThreadBudget};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

fn cpu() -> f64 {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut t) };
    t.tv_sec as f64 + t.tv_nsec as f64 * 1e-9
}

fn memory() -> Value {
    let mut v = serde_json::Map::new();
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for l in s.lines() {
            if l.starts_with("VmRSS:") || l.starts_with("VmHWM:") {
                if let Some(kb) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u64>().ok())
                {
                    v.insert(
                        if l.starts_with("VmRSS:") {
                            "rss_bytes"
                        } else {
                            "peak_rss_bytes"
                        }
                        .into(),
                        json!(kb * 1024),
                    );
                }
            }
        }
    }
    Value::Object(v)
}

/// Explicit opt-in for Zen 5 / AVX512_CORE_BF16 emulation via
/// `dnnl_set_max_cpu_isa` (not `ONEDNN_MAX_CPU_ISA`).
fn cpu_isa_cap(pin: Option<&str>) -> Result<CpuIsaCap> {
    match pin {
        None => Ok(CpuIsaCap::BestAvailable),
        Some(isa) => CpuIsaCap::from_name(isa).map_err(|e| anyhow::anyhow!("{e}")),
    }
}

fn run_wave(
    models: &[Arc<Model<OnednnCpu>>],
    batch: &Batch,
    warmup: usize,
    repeat: usize,
    concurrency: usize,
) -> Result<(Vec<f64>, Vec<f64>, Option<Vec<Vec<f32>>>)> {
    ensure!(concurrency >= 1 && concurrency <= 64, "concurrency");
    ensure!(!models.is_empty(), "no models");
    for (i, model) in models.iter().enumerate() {
        let n = if i == 0 { warmup } else { warmup.min(1) };
        for _ in 0..n {
            model.forward(batch)?;
        }
    }
    let mut wall = Vec::with_capacity(repeat);
    let mut work = Vec::with_capacity(repeat);
    let mut out = None;
    if concurrency == 1 {
        let model = &models[0];
        for _ in 0..repeat {
            let t = Instant::now();
            let c = cpu();
            let result = model.forward(batch)?;
            wall.push(t.elapsed().as_secs_f64());
            work.push(cpu() - c);
            out = Some(result);
        }
        return Ok((wall, work, out));
    }
    let batch = Arc::new(batch.clone());
    for _ in 0..repeat {
        let t = Instant::now();
        let c = cpu();
        let mut joins = Vec::with_capacity(concurrency);
        for i in 0..concurrency {
            let model = Arc::clone(&models[i % models.len()]);
            let batch = Arc::clone(&batch);
            joins.push(
                thread::Builder::new()
                    .name(format!("eg2-slot-{i}"))
                    .spawn(move || model.forward(&batch))
                    .context("spawn concurrent forward")?,
            );
        }
        for (i, h) in joins.into_iter().enumerate() {
            let result = h
                .join()
                .map_err(|_| anyhow::anyhow!("concurrent forward {i} panicked"))??;
            if i == 0 {
                out = Some(result);
            }
        }
        wall.push(t.elapsed().as_secs_f64());
        work.push(cpu() - c);
    }
    Ok((wall, work, out))
}

fn embeddings_json(rows: &[Vec<f32>], mrl: usize) -> Value {
    json!(rows
        .iter()
        .map(|r| {
            if mrl < r.len() {
                mrl_truncate(r, mrl)
            } else {
                r.clone()
            }
        })
        .collect::<Vec<_>>())
}

fn main() -> Result<()> {
    ensure!(cfg!(target_os = "linux"), "eg2-onednn runs on Linux only");
    let mut path = "model".to_string();
    let mut threads = None::<usize>;
    let mut tile = None::<usize>;
    let mut exec_slots = 1usize;
    let mut instances = 1usize;
    let mut load_frontend = false;
    let mut warmup = false;
    let mut cpuset = None::<String>;
    let mut pin_isa = None::<String>;
    let mut precision = Precision::Bf16;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => path = args.next().context("model")?,
            "--library" => {
                // Ignored: oneDNN is statically linked. Kept so existing scripts parse.
                let _ = args.next().context("library")?;
            }
            "--threads" => threads = Some(args.next().context("threads")?.parse()?),
            "--tile" => tile = Some(args.next().context("tile")?.parse()?),
            "--exec-slots" => exec_slots = args.next().context("exec-slots")?.parse()?,
            "--instances" => instances = args.next().context("instances")?.parse()?,
            "--cpuset" => cpuset = Some(args.next().context("cpuset")?),
            "--frontend" => load_frontend = true,
            "--warmup" => warmup = true,
            "--pin-isa" => pin_isa = Some(args.next().context("pin-isa")?),
            "--precision" => precision = args.next().context("precision")?.parse()?,
            _ => bail!("unknown argument {a}"),
        }
    }
    let env = EnvOverrides::from_env()?;
    let topo = CpuTopology::detect();
    let threads = threads.unwrap_or(topo.physical_cores.max(1));
    ensure!(
        threads >= 1 && exec_slots >= 1 && instances >= 1 && instances <= 8,
        "execution settings"
    );
    if let Some(t) = tile {
        ensure!(t > 0, "tile must be > 0");
    }
    let mut budget = ThreadBudget::new(threads)?;
    if let Some(cs) = cpuset {
        budget = budget.with_cpuset(cs)?;
    }
    budget.apply_process_policy()?;
    let session = SessionOptions {
        max_cpu_isa: if pin_isa.is_some() {
            cpu_isa_cap(pin_isa.as_deref())?
        } else {
            env.max_cpu_isa.map(Into::into).unwrap_or_default()
        },
        exec_slots,
        ..SessionOptions::default().with_overrides(&env)
    };
    let cache = CacheOptions::default().with_overrides(&env, ModelKind::EmbeddingGemma2);
    let t = Instant::now();
    let work = cpu();
    let tokenizer = if load_frontend {
        let p = path.clone();
        Some(
            thread::Builder::new()
                .name("eg2-tokenizer".into())
                .spawn(move || Frontend::load(Path::new(&p)))
                .context("spawn tokenizer")?,
        )
    } else {
        None
    };
    let mut models = Vec::with_capacity(instances);
    let mut backend_info = String::new();
    let mut cpu_isa = 0u64;
    let mut plan = None::<TuningPlan>;
    let mut resolved_tile = 0usize;
    for i in 0..instances {
        let backend = OnednnCpu::load_with(&budget, session.clone())?;
        backend_info = backend.info.clone();
        cpu_isa = backend.cpu_isa;
        if i == 0 {
            let p = TuningPlan::new(topo, backend.cpu_isa, threads);
            resolved_tile = tile.unwrap_or(p.eg2_attn_tile);
            ensure!(resolved_tile > 0, "tile must be > 0");
            plan = Some(p);
        }
        let model = Model::load_with_precision(
            Path::new(&path),
            backend,
            resolved_tile,
            &cache,
            precision,
        )?;
        if warmup {
            model.warmup()?;
        }
        models.push(Arc::new(model));
    }
    let tile = resolved_tile;
    let plan = plan.expect("at least one instance");
    let mut frontend = if let Some(h) = tokenizer {
        Some(
            h.join()
                .map_err(|_| anyhow::anyhow!("tokenizer thread panicked"))??,
        )
    } else {
        None
    };
    println!(
        "{}",
        json!({"ready":true,"backend":"eg2-onednn","precision":precision.as_str(),"threads":threads,"affinity":budget.cpus,"tile":tile,"exec_slots":exec_slots,"instances":instances,"warmup":warmup,"tuning":{"physical_cores":topo.physical_cores,"l1d_bytes":topo.l1d_bytes,"l2_bytes":topo.l2_bytes,"l3_bytes":topo.l3_bytes,"eltwise_grain":plan.eltwise_grain},"native_frontend_loaded":load_frontend,"load_wall_s":t.elapsed().as_secs_f64(),"load_cpu_s":cpu()-work,"memory":memory(),"numerical_backend":backend_info,"cpu_isa":cpu_isa,"embedding_dim":EMBED_DIM,"vision":false,"audio":false,"onednn":"static","primitive_stats":models[0].stats()})
    );
    io::stdout().flush()?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = (|| -> Result<Value> {
            let req: Value = serde_json::from_str(&line)?;
            let operation = req["operation"].as_str().unwrap_or("tensor");
            if operation == "stats" {
                return Ok(models[0].stats());
            }
            let repeat = req["repeat"].as_u64().unwrap_or(1) as usize;
            let warmup = req["warmup"].as_u64().unwrap_or(0) as usize;
            let concurrency = req["concurrency"].as_u64().unwrap_or(1) as usize;
            ensure!(
                (1..=10000).contains(&repeat) && warmup <= 10000,
                "repeat count"
            );
            let mrl = req["mrl"].as_u64().unwrap_or(EMBED_DIM as u64) as usize;
            let run_batch = |batch: &Batch| -> Result<(Vec<f64>, Vec<f64>, Option<Vec<Vec<f32>>>)> {
                run_wave(&models, batch, warmup, repeat, concurrency)
            };
            if operation == "tensor" {
                let input_ids: Vec<Vec<u32>> = serde_json::from_value(req["input_ids"].clone())?;
                let attention_mask: Vec<Vec<u32>> =
                    serde_json::from_value(req["attention_mask"].clone())?;
                let batch = Batch {
                    input_ids,
                    attention_mask,
                };
                let (wall, work, out) = run_batch(&batch)?;
                let mut v = json!({"wall_s":wall,"cpu_s":work,"memory":memory(),"primitive_stats":models[0].stats(),"concurrency":concurrency,"instances":models.len(),"exec_slots":exec_slots});
                if req["return_embeddings"].as_bool().unwrap_or(true) {
                    v["embeddings"] = embeddings_json(&out.unwrap(), mrl);
                }
                return Ok(v);
            }
            let f = frontend
                .as_mut()
                .context("encode requires startup --frontend")?;
            if operation == "encode" {
                let texts: Vec<String> = if let Some(t) = req["text"].as_str() {
                    vec![t.to_string()]
                } else {
                    serde_json::from_value(req["texts"].clone())?
                };
                let kind = match req["prompt"].as_str().unwrap_or("query") {
                    "document" | "doc" => PromptKind::Document,
                    _ => PromptKind::Query,
                };
                let batch = f.encode(&texts, kind)?;
                let (wall, work, out) = run_batch(&batch)?;
                let mut v = json!({"wall_s":wall,"cpu_s":work,"memory":memory()});
                if req["return_embeddings"].as_bool().unwrap_or(true) {
                    v["embeddings"] = embeddings_json(&out.unwrap(), mrl);
                }
                return Ok(v);
            }
            bail!("unsupported operation {operation}")
        })();
        let out = response.unwrap_or_else(|e| json!({"error":{"message":format!("{e:#}")}}));
        println!("{out}");
        io::stdout().flush()?;
    }
    Ok(())
}
