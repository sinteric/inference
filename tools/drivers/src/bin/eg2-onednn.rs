//! The bench harness's persistent JSON worker for EmbeddingGemma 2's text
//! tower. It loads the model, prints one `ready` line, then answers one JSON
//! request per stdin line with one JSON response line:
//!
//! - `{"operation": "tensor", "input_ids", "attention_mask"}` embeds token
//!   ids (`bench/protocols/embeddinggemma2/requests.jsonl`);
//! - `{"operation": "encode", "texts" | "text", "prompt"}` tokenizes first;
//! - `{"operation": "stats"}` returns the backend counters.
//!
//! `warmup`, `repeat` and `concurrency` shape a timed wave; the response
//! carries per-repeat `wall_s` and process `cpu_s`.

use std::io::{self, BufRead, Write};
use std::thread;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use drivers::flags::Flags;
use drivers::process::{affinity, cpu_seconds, memory};
use serde_json::{json, Value};
use sinteric_inference::embeddinggemma2::{
    Batch, Config, EmbeddingGemma2, Embeddings, Precision, PromptKind,
};
use sinteric_inference::{CpuIsa, Runtime, RuntimeOptions};

const VALUED: &[&str] = &[
    "--model",
    "--threads",
    "--tile",
    "--exec-slots",
    "--instances",
    "--cpuset",
    "--pin-isa",
    "--precision",
];
const SWITCHES: &[&str] = &["--warmup"];

/// One timed wave: `warmup` untimed calls on each model, then `repeat`
/// timed rounds of `concurrency` calls spread over the models.
struct Wave {
    warmup: usize,
    repeat: usize,
    concurrency: usize,
}

struct Timed {
    wall_s: Vec<f64>,
    cpu_s: Vec<f64>,
    last: Embeddings,
}

impl Wave {
    fn from_request(req: &Value) -> Result<Self> {
        let count = |key: &str, default: u64| -> Result<usize> {
            match &req[key] {
                Value::Null => Ok(default as usize),
                v => Ok(v.as_u64().with_context(|| format!("{key} {v}"))? as usize),
            }
        };
        let wave = Self {
            warmup: count("warmup", 0)?,
            repeat: count("repeat", 1)?,
            concurrency: count("concurrency", 1)?,
        };
        ensure!(
            (1..=10_000).contains(&wave.repeat) && wave.warmup <= 10_000,
            "repeat count"
        );
        ensure!((1..=64).contains(&wave.concurrency), "concurrency");
        Ok(wave)
    }

    fn run(&self, models: &[EmbeddingGemma2], batch: &Batch) -> Result<Timed> {
        for (i, model) in models.iter().enumerate() {
            let n = if i == 0 {
                self.warmup
            } else {
                self.warmup.min(1)
            };
            for _ in 0..n {
                model.embed_batch(batch)?;
            }
        }
        let mut wall_s = Vec::with_capacity(self.repeat);
        let mut cpu_s = Vec::with_capacity(self.repeat);
        let mut last = None;
        for _ in 0..self.repeat {
            let start = Instant::now();
            let cpu = cpu_seconds();
            let first = if self.concurrency == 1 {
                models[0].embed_batch(batch)?
            } else {
                concurrent(models, batch, self.concurrency)?
            };
            wall_s.push(start.elapsed().as_secs_f64());
            cpu_s.push(cpu_seconds() - cpu);
            last = Some(first);
        }
        Ok(Timed {
            wall_s,
            cpu_s,
            last: last.expect("repeat >= 1"),
        })
    }
}

/// `concurrency` calls at once, round-robin over `models`; the first
/// call's embeddings.
fn concurrent(models: &[EmbeddingGemma2], batch: &Batch, concurrency: usize) -> Result<Embeddings> {
    thread::scope(|s| {
        let calls: Vec<_> = (0..concurrency)
            .map(|i| {
                let model = &models[i % models.len()];
                thread::Builder::new()
                    .name(format!("eg2-slot-{i}"))
                    .spawn_scoped(s, move || model.embed_batch(batch))
            })
            .collect::<io::Result<_>>()
            .context("spawn a concurrent call")?;
        let mut first = None;
        for (i, call) in calls.into_iter().enumerate() {
            let out = call
                .join()
                .map_err(|_| anyhow::anyhow!("concurrent call {i} panicked"))??;
            first.get_or_insert(out);
        }
        Ok(first.expect("concurrency >= 1"))
    })
}

fn embeddings_json(e: &Embeddings) -> Value {
    json!(e.rows().collect::<Vec<_>>())
}

fn answer(models: &[EmbeddingGemma2], exec_slots: usize, line: &str) -> Result<Value> {
    let req: Value = serde_json::from_str(line)?;
    let operation = req["operation"].as_str().unwrap_or("tensor");
    let batch = match operation {
        "stats" => return Ok(models[0].backend_stats()),
        "tensor" => Batch {
            input_ids: serde_json::from_value(req["input_ids"].clone())?,
            attention_mask: serde_json::from_value(req["attention_mask"].clone())?,
        },
        "encode" => {
            let texts: Vec<String> = match req["text"].as_str() {
                Some(t) => vec![t.to_owned()],
                None => serde_json::from_value(req["texts"].clone())?,
            };
            let kind = match req["prompt"].as_str().unwrap_or("query") {
                "document" | "doc" => PromptKind::Document,
                "query" => PromptKind::Query,
                other => anyhow::bail!("unknown prompt {other}"),
            };
            models[0].tokenizer().encode(&texts, kind)?
        }
        other => anyhow::bail!("unsupported operation {other}"),
    };
    let timed = Wave::from_request(&req)?.run(models, &batch)?;
    let mut v = json!({
        "wall_s": timed.wall_s,
        "cpu_s": timed.cpu_s,
        "memory": memory(),
        "primitive_stats": models[0].backend_stats(),
        "instances": models.len(),
        "exec_slots": exec_slots,
    });
    if req["return_embeddings"].as_bool().unwrap_or(true) {
        v["embeddings"] = embeddings_json(&timed.last);
    }
    Ok(v)
}

fn main() -> Result<()> {
    let flags = Flags::parse(std::env::args().skip(1), VALUED, SWITCHES)?;
    let model_dir: String = flags.required("--model")?;
    let exec_slots = flags.get("--exec-slots")?.unwrap_or(1usize);
    let instances = flags.get("--instances")?.unwrap_or(1usize);
    let warmup = flags.switch("--warmup");
    let precision: Precision = flags.get("--precision")?.unwrap_or(Precision::Bf16);
    ensure!((1..=8).contains(&instances), "--instances outside 1..=8");

    let mut options = RuntimeOptions::new()
        .with_pin_process(true)
        .with_exec_slots(exec_slots);
    if let Some(threads) = flags.get("--threads")? {
        options = options.with_threads(threads);
    }
    if let Some(cpuset) = flags.get::<String>("--cpuset")? {
        options = options.with_cpuset(cpuset);
    }
    if let Some(isa) = flags.get::<String>("--pin-isa")? {
        options = options.with_cpu_isa(CpuIsa::from_name(&isa)?);
    }
    let start = Instant::now();
    let cpu = cpu_seconds();
    let rt = Runtime::new(options)?;
    let plan = rt.plan();
    let tile: Option<usize> = flags.get("--tile")?;
    let mut config = Config::new(&model_dir)
        .with_precision(precision)
        .with_warmup(warmup);
    if let Some(tile) = tile {
        config = config.with_tile(tile);
    }
    let models = (0..instances)
        .map(|_| rt.load::<EmbeddingGemma2>(config.clone()))
        .collect::<sinteric_inference::Result<Vec<_>>>()?;
    let topology = plan.topology;
    println!(
        "{}",
        json!({
            "ready": true,
            "backend": "eg2-onednn",
            "precision": models[0].precision().as_str(),
            "threads": rt.options().threads,
            "affinity": affinity()?,
            "tile": tile,
            "exec_slots": exec_slots,
            "instances": instances,
            "warmup": warmup,
            "tuning": {
                "physical_cores": topology.physical_cores,
                "l1d_bytes": topology.l1d_bytes,
                "l2_bytes": topology.l2_bytes,
                "l3_bytes": topology.l3_bytes,
                "eltwise_grain": plan.eltwise_grain,
            },
            "load_wall_s": start.elapsed().as_secs_f64(),
            "load_cpu_s": cpu_seconds() - cpu,
            "memory": memory(),
            "cpu_isa": rt.cpu_isa(),
            "cpu_isa_name": rt.cpu_isa_name(),
            "embedding_dim": models[0].embedding_dim(),
            "onednn": "static",
            "primitive_stats": models[0].backend_stats(),
        })
    );
    io::stdout().flush()?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let out = answer(&models, exec_slots, &line)
            .unwrap_or_else(|e| json!({"error": {"message": format!("{e:#}")}}));
        println!("{out}");
        io::stdout().flush()?;
    }
    Ok(())
}
