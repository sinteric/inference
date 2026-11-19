//! Persistent VM-only driver: model-only and complete native API have separate timing boundaries.
use anyhow::{bail, ensure, Context, Result};
use rt_core::env::{EnvOverrides, ModelKind};
use rt_onednn::{CacheOptions, CpuIsaCap, OnednnCpu, SessionOptions, TuningPlan};
use rt_text::{Frontend, Inputs};
use rt_threads::{CpuTopology, ThreadBudget};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::thread;
use std::time::Instant;
use vela2_0p3b::onednn::model::Model;
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
fn main() -> Result<()> {
    ensure!(
        cfg!(target_os = "linux"),
        "authorized inference environment is Remote Linux VM"
    );
    let mut path = "model".to_string();
    let mut threads = None::<usize>;
    let mut tile = None::<usize>;
    let mut load_frontend = false;
    let mut cpuset = None::<String>;
    let mut pin_isa = None::<String>;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => path = args.next().context("model")?,
            "--library" => {
                let _ = args.next().context("library")?;
            }
            "--threads" => threads = Some(args.next().context("threads")?.parse()?),
            "--tile" => tile = Some(args.next().context("tile")?.parse()?),
            "--cpuset" => cpuset = Some(args.next().context("cpuset")?),
            "--frontend" => load_frontend = true,
            "--pin-isa" => pin_isa = Some(args.next().context("pin-isa")?),
            _ => bail!("unknown argument {a}"),
        }
    }
    let env = EnvOverrides::from_env()?;
    let topo = CpuTopology::detect();
    let threads = threads.unwrap_or(topo.physical_cores.max(1));
    ensure!(threads >= 1, "threads must be >= 1");
    if let Some(t) = tile {
        ensure!(t > 0, "tile must be > 0");
    }
    let mut budget = ThreadBudget::new(threads)?;
    if let Some(cs) = cpuset {
        budget = budget.with_cpuset(cs)?;
    }
    budget.apply_process_policy()?;
    let session = SessionOptions {
        max_cpu_isa: match pin_isa.as_deref() {
            None => env.max_cpu_isa.map(Into::into).unwrap_or_default(),
            Some(isa) => CpuIsaCap::from_name(isa).map_err(|e| anyhow::anyhow!("{e}"))?,
        },
        ..SessionOptions::default().with_overrides(&env)
    };
    let t = Instant::now();
    let work = cpu();
    let tokenizer = if load_frontend {
        let p = path.clone();
        Some(
            thread::Builder::new()
                .name("vela-tokenizer".into())
                .spawn(move || Frontend::load(&p))
                .context("spawn tokenizer")?,
        )
    } else {
        None
    };
    let cache = CacheOptions::default().with_overrides(&env, ModelKind::Vela);
    let profiling = env.vela_standalone_profile.unwrap_or(false);
    let backend = OnednnCpu::load_with(&budget, session)?;
    let plan = TuningPlan::new(topo, backend.cpu_isa, threads);
    let tile = tile.unwrap_or(plan.vela_attn_tile);
    ensure!(tile > 0, "tile must be > 0");
    let model = Model::load_with(Path::new(&path), backend.clone(), tile, &cache, profiling)?;
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
        json!({"ready":true,"backend":"standalone-rust-public-onednn","precision":"FP32","threads":threads,"affinity":budget.cpus,"tile":tile,"tuning":{"physical_cores":topo.physical_cores,"l1d_bytes":topo.l1d_bytes,"l2_bytes":topo.l2_bytes,"l3_bytes":topo.l3_bytes,"eltwise_grain":plan.eltwise_grain},"native_frontend_loaded":load_frontend,"load_wall_s":t.elapsed().as_secs_f64(),"load_cpu_s":cpu()-work,"memory":memory(),"numerical_backend":backend.info.as_str(),"dependencies":{"candle":false,"mlas":false,"libtorch":false,"onnx":false},"glue":"Rust embedding/gather/indexing/mask/RoPE/prefix pooling/L2/cosine; public oneDNN C API matmul/reorder/layernorm/softmax/GELU/binary/ReLU","primitive_stats":model.stats(),"onednn":"static"})
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
                return Ok(model.stats());
            }
            let repeat = req["repeat"].as_u64().unwrap_or(1) as usize;
            let warmup = req["warmup"].as_u64().unwrap_or(0) as usize;
            ensure!(
                (1..=10000).contains(&repeat) && warmup <= 10000,
                "repeat count"
            );
            if operation == "tensor" {
                let input: Inputs = serde_json::from_value(req.clone())?;
                for _ in 0..warmup {
                    model.forward(&input)?;
                }
                let mut wall = Vec::with_capacity(repeat);
                let mut work = Vec::with_capacity(repeat);
                let mut out = None;
                for _ in 0..repeat {
                    let t = Instant::now();
                    let c = cpu();
                    let result = model.forward(&input)?;
                    wall.push(t.elapsed().as_secs_f64());
                    work.push(cpu() - c);
                    out = Some(result);
                }
                let mut v = json!({"wall_s":wall,"cpu_s":work,"memory":memory(),"primitive_stats":model.stats()});
                if req["return_logits"].as_bool().unwrap_or(true) {
                    let out = out.unwrap();
                    v["opt_logits"] = json!(out.opt_logits);
                    v["span_logits"] = json!(out.span_logits);
                }
                return Ok(v);
            }
            let f = frontend
                .as_mut()
                .context("full API requires startup --frontend")?;
            if let Some(b) = req["batch_size"].as_u64() {
                ensure!(b > 0, "batch_size");
                f.batch_size = b as usize;
            }
            if let Some(b) = req["token_budget"].as_u64() {
                ensure!(b > 0, "token_budget");
                f.token_budget = b as usize;
            }
            let execute = || -> Result<Value> {
                match operation {
                    "quality" => {
                        let rows = req["rows"].as_array().context("rows")?;
                        let raw = f.predict_raw(rows, |r| model.forward(r))?;
                        let decoded = raw
                            .iter()
                            .zip(rows)
                            .map(|(r, row)| f.calibrated(r, row, &req["thresholds"]))
                            .collect::<rt_core::Result<Vec<_>>>()?;
                        Ok(json!({"raw":raw,"decoded":decoded}))
                    }
                    "generic" => {
                        let rows = req["rows"].as_array().context("rows")?;
                        let raw = f.predict_raw(rows, |r| model.forward(r))?;
                        Ok(json!(raw.iter().zip(rows).map(|(r,row)|Ok(json!({"answers":f.calibrated(r,row,&req["thresholds"] )?,"meta":{"tokens":r["n_tokens"],"windows":r["windows"]}}))).collect::<Result<Vec<_>>>()?))
                    }
                    "predict" => Ok(f.predict(&req["req"], |r| model.forward(r))?),
                    "system_one" => Ok(f.system_one(
                        &req["state"],
                        &req["questions"],
                        req["model"].as_str(),
                        |r| model.forward(r),
                    )?),
                    "system_one_request" => {
                        Ok(f.system_one_request(&req["body"], |r| model.forward(r))?)
                    }
                    "prepare" => Ok(serde_json::to_value(
                        f.prepare(req["rows"].as_array().context("rows")?)?,
                    )?),
                    _ => bail!("unsupported operation {operation}"),
                }
            };
            if req["benchmark"].as_bool().unwrap_or(false) {
                for _ in 0..warmup {
                    execute()?;
                }
                let mut wall = Vec::new();
                let mut work = Vec::new();
                let mut result = Value::Null;
                for _ in 0..repeat {
                    let t = Instant::now();
                    let c = cpu();
                    result = execute()?;
                    wall.push(t.elapsed().as_secs_f64());
                    work.push(cpu() - c);
                }
                Ok(json!({"result":result,"wall_s":wall,"cpu_s":work,"memory":memory()}))
            } else {
                execute()
            }
        })();
        let out=response.unwrap_or_else(|e|json!({"error":{"type":if matches!(e.downcast_ref::<rt_core::Error>(), Some(rt_core::Error::Request(_))){"SystemOneError"}else{"ValueError"},"message":format!("{e:#}")}}));
        println!("{out}");
        io::stdout().flush()?;
    }
    Ok(())
}
