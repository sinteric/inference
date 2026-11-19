//! Load generator for `serve-sim serve`.
//!
//! Phases, in order:
//! 1. `warmup`: one sequential pass over every input of every enabled endpoint.
//! 2. `serial`: a second sequential pass. Its digests are the reference and
//!    its latencies the solo baseline; they must equal the warmup's digests.
//! 3. `idle`: health/work probes alone, for the server's own baseline.
//! 4. `load`: closed-loop clients per endpoint plus the probes, for
//!    `--duration` seconds. Optionally unloads and reloads one model on a
//!    timer (`--churn`).
//! 5. `after`: a last sequential pass, checked against the reference.
//!
//! Every response digest is compared with the reference for the same input,
//! so any cross-request interference shows up as a mismatch.
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{Method, Request};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use serde::Serialize;
use serde_json::{json, Value};

use crate::inputs;

type Http = Client<HttpConnector, Full<Bytes>>;

#[derive(clap::Args, Debug)]
pub struct LoadArgs {
    #[arg(long, default_value = "http://127.0.0.1:8090")]
    url: String,
    /// Seconds of concurrent load.
    #[arg(long, default_value_t = 30)]
    duration: u64,
    /// Closed-loop `/embed` clients.
    #[arg(long, default_value_t = 0)]
    embed: usize,
    /// Closed-loop `/vela` clients.
    #[arg(long, default_value_t = 0)]
    vela: usize,
    /// Closed-loop `/ocr` clients.
    #[arg(long, default_value_t = 0)]
    ocr: usize,
    /// Page list (the gate's JSONL with a `path` per line) for `/ocr`.
    #[arg(long)]
    pages: Option<PathBuf>,
    /// Directory the page paths are relative to (default: current dir).
    #[arg(long)]
    pages_root: Option<PathBuf>,
    #[arg(long, default_value_t = 6)]
    max_pages: usize,
    /// `/health` probes per second (open loop).
    #[arg(long, default_value_t = 20.0)]
    probe_hz: f64,
    /// Also probe `/work?us=N` (the host's own CPU work) at the same rate.
    #[arg(long, default_value_t = 0)]
    work_us: u64,
    /// Seconds of probes before the load starts.
    #[arg(long, default_value_t = 3)]
    idle: u64,
    /// Unload and reload this model (`eg2`, `vela`, `ocr`) during the load.
    #[arg(long)]
    churn: Option<String>,
    /// Seconds between churn cycles.
    #[arg(long, default_value_t = 10)]
    churn_every: u64,
    /// Per-request timeout in seconds.
    #[arg(long, default_value_t = 300)]
    timeout: u64,
    /// Seconds to wait for the server to come up.
    #[arg(long, default_value_t = 900)]
    wait: u64,
    /// Raw per-request records (JSONL).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Free-form label copied into the summary.
    #[arg(long, default_value = "")]
    label: String,
}

struct Input {
    id: String,
    body: Bytes,
    content_type: &'static str,
}

struct Target {
    endpoint: &'static str,
    clients: usize,
    inputs: Vec<Input>,
}

#[derive(Clone, Debug, Serialize)]
struct Record {
    phase: &'static str,
    endpoint: &'static str,
    input: String,
    /// Start time, seconds since the load generator started.
    t_s: f64,
    ms: f64,
    status: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    digest_ok: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    queue_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    infer_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

struct Ctx {
    http: Http,
    base: String,
    timeout: Duration,
    epoch: Instant,
    records: Mutex<Vec<Record>>,
}

impl Ctx {
    async fn send(
        &self,
        method: Method,
        path: &str,
        body: Option<(&Bytes, &'static str)>,
    ) -> Result<(u16, Value)> {
        let mut req = Request::builder()
            .method(method)
            .uri(format!("{}{}", self.base, path));
        let body = match body {
            Some((b, ct)) => {
                req = req.header("content-type", ct);
                Full::new(b.clone())
            }
            None => Full::new(Bytes::new()),
        };
        let fut = async {
            let resp = self.http.request(req.body(body)?).await?;
            let status = resp.status().as_u16();
            let bytes = resp.into_body().collect().await?.to_bytes();
            let value = serde_json::from_slice(&bytes)
                .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
            anyhow::Ok((status, value))
        };
        tokio::time::timeout(self.timeout, fut)
            .await
            .context("request timed out")?
    }

    /// One request, recorded. `reference` is the expected digest, if known.
    async fn hit(
        &self,
        phase: &'static str,
        endpoint: &'static str,
        path: &str,
        input: Option<&Input>,
        reference: Option<&str>,
    ) -> Record {
        let start = Instant::now();
        let t_s = start.duration_since(self.epoch).as_secs_f64();
        let method = if input.is_some() {
            Method::POST
        } else {
            Method::GET
        };
        let result = self
            .send(method, path, input.map(|i| (&i.body, i.content_type)))
            .await;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let mut rec = Record {
            phase,
            endpoint,
            input: input.map(|i| i.id.clone()).unwrap_or_default(),
            t_s,
            ms,
            status: 0,
            digest: None,
            digest_ok: None,
            queue_ms: None,
            infer_ms: None,
            decode_ms: None,
            error: None,
        };
        match result {
            Ok((status, body)) => {
                rec.status = status;
                if status != 200 {
                    rec.error = Some(
                        body.get("error")
                            .map_or(body.to_string(), |e| e.to_string()),
                    );
                }
                rec.digest = body
                    .get("digest")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                rec.queue_ms = body.pointer("/timing/queue_ms").and_then(Value::as_f64);
                rec.infer_ms = body.pointer("/timing/infer_ms").and_then(Value::as_f64);
                rec.decode_ms = body.get("decode_ms").and_then(Value::as_f64);
                if let (Some(want), Some(got)) = (reference, rec.digest.as_deref()) {
                    rec.digest_ok = Some(want == got);
                }
            }
            Err(e) => rec.error = Some(format!("{e:#}")),
        }
        self.records.lock().unwrap().push(rec.clone());
        rec
    }
}

fn pages(args: &LoadArgs) -> Result<Vec<Input>> {
    let Some(list) = &args.pages else {
        bail!("--ocr needs --pages");
    };
    let root = args.pages_root.clone().unwrap_or_else(|| ".".into());
    let mut out = Vec::new();
    for line in std::fs::read_to_string(list)?.lines() {
        if line.trim().is_empty() || out.len() >= args.max_pages {
            continue;
        }
        let v: Value = serde_json::from_str(line)?;
        let id = v["img_id"].as_str().unwrap_or("").to_owned();
        if id.starts_with("warm_") {
            continue;
        }
        let path = root.join(v["path"].as_str().context("page path")?);
        let bytes = std::fs::read(&path).with_context(|| path.display().to_string())?;
        out.push(Input {
            id,
            body: Bytes::from(bytes),
            content_type: "application/octet-stream",
        });
    }
    Ok(out)
}

fn json_inputs(list: Vec<inputs::Input>) -> Vec<Input> {
    list.into_iter()
        .map(|i| Input {
            id: i.id,
            body: Bytes::from(serde_json::to_vec(&i.body).expect("serializable input")),
            content_type: "application/json",
        })
        .collect()
}

fn targets(args: &LoadArgs) -> Result<Vec<Target>> {
    let mut out = Vec::new();
    if args.embed > 0 {
        out.push(Target {
            endpoint: "embed",
            clients: args.embed,
            inputs: json_inputs(inputs::embed_requests()),
        });
    }
    if args.vela > 0 {
        out.push(Target {
            endpoint: "vela",
            clients: args.vela,
            inputs: json_inputs(inputs::vela_bodies()),
        });
    }
    if args.ocr > 0 {
        out.push(Target {
            endpoint: "ocr",
            clients: args.ocr,
            inputs: pages(args)?,
        });
    }
    Ok(out)
}

type References = BTreeMap<(&'static str, String), String>;

async fn sequential(
    ctx: &Ctx,
    phase: &'static str,
    targets: &[Target],
    refs: Option<&References>,
) -> Vec<Record> {
    let mut out = Vec::new();
    for t in targets {
        for input in &t.inputs {
            let want = refs.and_then(|r| r.get(&(t.endpoint, input.id.clone())));
            let path = format!("/{}", t.endpoint);
            out.push(
                ctx.hit(
                    phase,
                    t.endpoint,
                    &path,
                    Some(input),
                    want.map(String::as_str),
                )
                .await,
            );
        }
    }
    out
}

/// `/health` (and `/work`) at a fixed rate until `stop`; each probe is its own
/// task so a slow one does not delay the schedule.
async fn probes(ctx: Arc<Ctx>, phase: &'static str, hz: f64, work_us: u64, stop: Instant) {
    if hz <= 0.0 {
        tokio::time::sleep_until(stop.into()).await;
        return;
    }
    let mut tick = tokio::time::interval(Duration::from_secs_f64(1.0 / hz));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut tasks = Vec::new();
    while Instant::now() < stop {
        tick.tick().await;
        let c = ctx.clone();
        tasks.push(tokio::spawn(async move {
            c.hit(phase, "health", "/health", None, None).await;
        }));
        if work_us > 0 {
            let c = ctx.clone();
            tasks.push(tokio::spawn(async move {
                c.hit(phase, "work", &format!("/work?us={work_us}"), None, None)
                    .await;
            }));
        }
    }
    for t in tasks {
        let _ = t.await;
    }
}

#[derive(Debug, Default, Serialize)]
struct Dist {
    n: usize,
    p50: f64,
    p95: f64,
    p99: f64,
    max: f64,
    mean: f64,
}

fn dist(mut v: Vec<f64>) -> Dist {
    if v.is_empty() {
        return Dist::default();
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let at = |p: f64| v[((p * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1];
    let r = |x: f64| (x * 10.0).round() / 10.0;
    Dist {
        n: v.len(),
        p50: r(at(0.50)),
        p95: r(at(0.95)),
        p99: r(at(0.99)),
        max: r(*v.last().unwrap()),
        mean: r(v.iter().sum::<f64>() / v.len() as f64),
    }
}

fn summarize(records: &[Record], phase: &str, endpoint: &str, secs: f64) -> Value {
    let sel: Vec<&Record> = records
        .iter()
        .filter(|r| r.phase == phase && r.endpoint == endpoint)
        .collect();
    let ok: Vec<&Record> = sel.iter().copied().filter(|r| r.status == 200).collect();
    let mut statuses = BTreeMap::new();
    for r in &sel {
        *statuses.entry(r.status.to_string()).or_insert(0usize) += 1;
    }
    let mut v = json!({
        "n": sel.len(),
        "ok": ok.len(),
        "statuses": statuses,
        "ms": dist(ok.iter().map(|r| r.ms).collect()),
    });
    if secs > 0.0 {
        v["rps"] = json!((ok.len() as f64 / secs * 100.0).round() / 100.0);
    }
    let checked: Vec<_> = ok.iter().filter_map(|r| r.digest_ok).collect();
    if !checked.is_empty() {
        v["digest_checked"] = json!(checked.len());
        v["digest_mismatches"] = json!(checked.iter().filter(|ok| !**ok).count());
    }
    let queue: Vec<f64> = ok.iter().filter_map(|r| r.queue_ms).collect();
    if !queue.is_empty() {
        v["queue_ms"] = json!(dist(queue));
        v["infer_ms"] = json!(dist(ok.iter().filter_map(|r| r.infer_ms).collect()));
    }
    let decode: Vec<f64> = ok.iter().filter_map(|r| r.decode_ms).collect();
    if !decode.is_empty() {
        v["decode_ms"] = json!(dist(decode));
    }
    if let Some(e) = sel.iter().find_map(|r| r.error.clone()) {
        v["first_error"] = json!(e);
    }
    v
}

pub fn run(args: LoadArgs) -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run_async(args))
}

async fn run_async(args: LoadArgs) -> Result<()> {
    let http: Http = Client::builder(TokioExecutor::new())
        .pool_max_idle_per_host(256)
        .build_http();
    let ctx = Arc::new(Ctx {
        http,
        base: args.url.trim_end_matches('/').to_owned(),
        timeout: Duration::from_secs(args.timeout),
        epoch: Instant::now(),
        records: Mutex::new(Vec::new()),
    });
    let targets = Arc::new(targets(&args)?);
    if targets.is_empty() {
        bail!("nothing to load: pass --embed, --vela or --ocr");
    }

    let deadline = Instant::now() + Duration::from_secs(args.wait);
    loop {
        if let Ok((200, _)) = ctx.send(Method::GET, "/health", None).await {
            break;
        }
        if Instant::now() > deadline {
            bail!("server at {} did not come up", args.url);
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    let (_, server_start) = ctx.send(Method::GET, "/stats", None).await?;

    let warm = sequential(&ctx, "warmup", &targets, None).await;
    let mut refs = References::new();
    for r in &warm {
        if let Some(d) = &r.digest {
            refs.insert((r.endpoint, r.input.clone()), d.clone());
        }
    }
    let serial = sequential(&ctx, "serial", &targets, Some(&refs)).await;
    if let Some(bad) = warm.iter().chain(&serial).find(|r| r.status != 200) {
        bail!(
            "{} {} failed serially: {} {}",
            bad.endpoint,
            bad.input,
            bad.status,
            bad.error.clone().unwrap_or_default()
        );
    }

    // RSS once every input has run serially: the single-request working set.
    let (_, server_serial) = ctx.send(Method::GET, "/stats", None).await?;

    let idle_stop = Instant::now() + Duration::from_secs(args.idle);
    probes(ctx.clone(), "idle", args.probe_hz, args.work_us, idle_stop).await;

    // Load phase.
    let load_start = Instant::now();
    let stop = load_start + Duration::from_secs(args.duration);
    let refs = Arc::new(refs);
    let mut tasks = Vec::new();
    for (ti, t) in targets.iter().enumerate() {
        for client in 0..t.clients {
            let (ctx, targets, refs) = (ctx.clone(), targets.clone(), refs.clone());
            tasks.push(tokio::spawn(async move {
                let t = &targets[ti];
                let path = format!("/{}", t.endpoint);
                let mut i = client * t.inputs.len().max(1) / t.clients.max(1);
                while Instant::now() < stop {
                    let input = &t.inputs[i % t.inputs.len()];
                    let want = refs.get(&(t.endpoint, input.id.clone())).cloned();
                    let rec = ctx
                        .hit("load", t.endpoint, &path, Some(input), want.as_deref())
                        .await;
                    if rec.status == 503 {
                        // Model unloaded (churn): back off like a client
                        // retrying, instead of spinning on instant 503s.
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        continue;
                    }
                    i += 1;
                }
            }));
        }
    }
    let probe_task = tokio::spawn(probes(
        ctx.clone(),
        "load",
        args.probe_hz,
        args.work_us,
        stop,
    ));
    let samples = Arc::new(Mutex::new(Vec::<Value>::new()));
    let sampler = {
        let (ctx, samples) = (ctx.clone(), samples.clone());
        tokio::spawn(async move {
            while Instant::now() < stop {
                if let Ok((200, s)) = ctx.send(Method::GET, "/stats", None).await {
                    samples.lock().unwrap().push(json!({
                        "t_s": load_start.elapsed().as_secs_f64(),
                        "proc": s["proc"],
                        "inflight": {
                            "eg2": s["models"]["eg2"]["inflight"],
                            "vela": s["models"]["vela"]["inflight"],
                            "ocr": s["models"]["ocr"]["inflight"],
                        },
                    }));
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        })
    };
    let churn = args.churn.clone().map(|model| {
        let ctx = ctx.clone();
        let every = Duration::from_secs(args.churn_every.max(1));
        tokio::spawn(async move {
            let mut events = Vec::new();
            loop {
                tokio::time::sleep(every).await;
                if Instant::now() + every / 2 > stop {
                    break;
                }
                let t = load_start.elapsed().as_secs_f64();
                let unload = ctx
                    .send(Method::POST, &format!("/admin/unload/{model}"), None)
                    .await
                    .map(|(_, v)| v)
                    .unwrap_or_else(|e| json!({"error": format!("{e:#}")}));
                let load = ctx
                    .send(Method::POST, &format!("/admin/load/{model}"), None)
                    .await
                    .map(|(_, v)| v)
                    .unwrap_or_else(|e| json!({"error": format!("{e:#}")}));
                events.push(json!({"t_s": t, "unload": unload, "load": load}));
            }
            events
        })
    });
    for t in tasks {
        t.await?;
    }
    let load_secs = load_start.elapsed().as_secs_f64();
    probe_task.await?;
    sampler.await?;
    let churn_events = match churn {
        Some(task) => task.await?,
        None => Vec::new(),
    };
    let (_, server_end) = ctx.send(Method::GET, "/stats", None).await?;
    sequential(&ctx, "after", &targets, Some(&refs)).await;
    // What a trim gives back with every model still loaded: freed heap the
    // allocator kept, as opposed to memory the models still hold.
    let trim = match ctx.send(Method::POST, "/admin/trim", None).await {
        Ok((200, v)) => v,
        _ => Value::Null,
    };

    let records = ctx.records.lock().unwrap().clone();
    if let Some(path) = &args.out {
        let mut buf = String::new();
        for r in &records {
            buf.push_str(&serde_json::to_string(r)?);
            buf.push('\n');
        }
        std::fs::write(path, buf)?;
    }
    let samples = samples.lock().unwrap().clone();
    let peak = |key: &str| {
        samples
            .iter()
            .filter_map(|s| s["proc"][key].as_u64())
            .max()
            .unwrap_or(0)
    };
    let cpu = |s: &Value| s["proc"]["cpu_ms"].as_f64().unwrap_or(0.0);
    let cpu_cores = match (samples.first(), samples.last()) {
        (Some(a), Some(b)) if b["t_s"].as_f64() > a["t_s"].as_f64() => {
            (cpu(b) - cpu(a)) / 1e3 / (b["t_s"].as_f64().unwrap() - a["t_s"].as_f64().unwrap())
        }
        _ => 0.0,
    };

    let mut endpoints = serde_json::Map::new();
    for t in targets.iter() {
        endpoints.insert(
            t.endpoint.into(),
            json!({
                "clients": t.clients,
                "inputs": t.inputs.len(),
                "serial": summarize(&records, "serial", t.endpoint, 0.0),
                "load": summarize(&records, "load", t.endpoint, load_secs),
                "after": summarize(&records, "after", t.endpoint, 0.0),
            }),
        );
    }
    let mut probes_out = serde_json::Map::new();
    for ep in ["health", "work"] {
        if ep == "work" && args.work_us == 0 {
            continue;
        }
        probes_out.insert(
            ep.into(),
            json!({
                "idle": summarize(&records, "idle", ep, 0.0),
                "load": summarize(&records, "load", ep, 0.0),
            }),
        );
    }
    let summary = json!({
        "label": args.label,
        "server": server_start["runtime"],
        "load_s": (load_secs * 10.0).round() / 10.0,
        "endpoints": endpoints,
        "probes": probes_out,
        "proc": {
            "rss_mib_start": server_start["proc"]["rss_mib"],
            "rss_mib_after_serial": server_serial["proc"]["rss_mib"],
            "rss_mib_peak_sampled": peak("rss_mib"),
            "rss_mib_end": server_end["proc"]["rss_mib"],
            "hwm_mib": server_end["proc"]["hwm_mib"],
            "threads_start": server_start["proc"]["threads"],
            "threads_peak": peak("threads"),
            "cpu_cores_used": (cpu_cores * 100.0).round() / 100.0,
            "heap_in_use_mib_peak_sampled": peak("heap_in_use_mib"),
            "heap_free_mib_peak_sampled": peak("heap_free_mib"),
            "snapshots": {
                "start": server_start["proc"],
                "after_serial": server_serial["proc"],
                "end": server_end["proc"],
            },
            // What each model keeps between calls (in-process server only).
            "models_after_serial": server_serial["memory"],
            "models_end": server_end["memory"],
            "trim": trim,
            // RSS and live heap over the load phase, one point per sample.
            "series": samples
                .iter()
                .map(|s| json!([s["t_s"], s["proc"]["rss_mib"], s["proc"]["heap_in_use_mib"]]))
                .collect::<Vec<_>>(),
        },
        "churn": churn_events,
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
