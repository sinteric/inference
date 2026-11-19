//! One process per model behind a front server.
//!
//! `serve-sim split` starts a `serve-sim serve` child per configured model,
//! each with its own runtime, thread pool and (optionally) cpuset, and
//! proxies the inference and admin routes to them over loopback HTTP. The
//! front answers `/health` and `/work` itself and sums the children's
//! process counters into its `/stats`, so `serve-sim load` drives it
//! unchanged.
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Context};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use http_body_util::{BodyExt, Full};
use hyper::Request;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use serde_json::{json, Value};

use crate::proc_stats;
use crate::server::{health, work, Order};

#[derive(clap::Args, Clone, Debug)]
pub struct SplitArgs {
    #[arg(long, default_value = "127.0.0.1:8090")]
    addr: String,
    /// Children listen on consecutive loopback ports from here.
    #[arg(long, default_value_t = 8190)]
    child_port: u16,
    #[arg(long)]
    eg2: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    eg2_threads: usize,
    /// Pins the EG2 child (runtime pool and its tokio threads) to these CPUs.
    #[arg(long)]
    eg2_cpuset: Option<String>,
    #[arg(long)]
    eg2_warmup: bool,
    #[arg(long)]
    vela: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    vela_threads: usize,
    #[arg(long)]
    vela_cpuset: Option<String>,
    #[arg(long)]
    ocr: Option<PathBuf>,
    #[arg(long, default_value_t = 2)]
    ocr_threads: usize,
    #[arg(long)]
    ocr_cpuset: Option<String>,
    /// Passed to every child.
    #[arg(long, default_value_t = 1)]
    exec_slots: usize,
    #[arg(long, default_value = "best_available")]
    cpu_isa: String,
    #[arg(long, default_value_t = 0)]
    max_inflight: usize,
    #[arg(long, default_value_t = 0)]
    workspace_limit_mib: usize,
    #[arg(long, default_value_t = 0)]
    memory_budget_mib: usize,
    #[arg(long, default_value_t = 0)]
    malloc_arena_max: usize,
    #[arg(long, value_enum, default_value_t = Order::Fifo)]
    compute_order: Order,
    /// Tokio workers per child and for the front.
    #[arg(long, default_value_t = 2)]
    tokio_workers: usize,
    /// Seconds to wait for every child to load and listen.
    #[arg(long, default_value_t = 300)]
    wait: u64,
}

type Http = Client<HttpConnector, Full<Bytes>>;

struct Front {
    http: Http,
    /// Model name -> child base URL.
    children: BTreeMap<&'static str, String>,
    procs: Mutex<Vec<Child>>,
    started: Instant,
}

type Shared = Arc<Front>;

async fn forward(
    front: &Front,
    model: &str,
    method: Method,
    path: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> Response {
    let Some(base) = front.children.get(model) else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": format!("{model} not configured")})),
        )
            .into_response();
    };
    let mut req = Request::builder()
        .method(method)
        .uri(format!("{base}{path}"));
    if let Some(ct) = headers.get("content-type") {
        req = req.header("content-type", ct);
    }
    let result = async {
        let resp = front.http.request(req.body(Full::new(body))?).await?;
        let status = resp.status();
        let ct = resp.headers().get("content-type").cloned();
        let bytes = resp.into_body().collect().await?.to_bytes();
        anyhow::Ok((status, ct, bytes))
    }
    .await;
    match result {
        Ok((status, ct, bytes)) => {
            let mut out = (status, bytes).into_response();
            if let Some(ct) = ct {
                out.headers_mut().insert("content-type", ct);
            }
            out
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": format!("{model} child: {e}")})),
        )
            .into_response(),
    }
}

async fn get_json(front: &Front, base: &str, method: Method, path: &str) -> Option<Value> {
    let req = Request::builder()
        .method(method)
        .uri(format!("{base}{path}"))
        .body(Full::new(Bytes::new()))
        .ok()?;
    let resp = front.http.request(req).await.ok()?;
    let bytes = resp.into_body().collect().await.ok()?.to_bytes();
    serde_json::from_slice(&bytes).ok()
}

async fn embed(State(f): State<Shared>, headers: HeaderMap, body: Bytes) -> Response {
    forward(&f, "eg2", Method::POST, "/embed", &headers, body).await
}

async fn vela(State(f): State<Shared>, headers: HeaderMap, body: Bytes) -> Response {
    forward(&f, "vela", Method::POST, "/vela", &headers, body).await
}

async fn ocr(State(f): State<Shared>, headers: HeaderMap, body: Bytes) -> Response {
    forward(&f, "ocr", Method::POST, "/ocr", &headers, body).await
}

async fn admin(
    State(f): State<Shared>,
    Path((action, name)): Path<(String, String)>,
    headers: HeaderMap,
) -> Response {
    let path = format!("/admin/{action}/{name}");
    forward(&f, &name, Method::POST, &path, &headers, Bytes::new()).await
}

/// Counters summed over the front and every child.
const SUMMED: [&str; 11] = [
    "rss_mib",
    "hwm_mib",
    "threads",
    "cpu_ms",
    "voluntary_ctxt_switches",
    "nonvoluntary_ctxt_switches",
    "rss_anon_mib",
    "rss_file_mib",
    "heap_in_use_mib",
    "heap_free_mib",
    "heap_mmap_mib",
];

fn sum_procs<'a>(own: Value, children: impl Iterator<Item = &'a Value>) -> Value {
    let mut total = own;
    for c in children {
        for key in SUMMED {
            let add = c[key].as_u64().unwrap_or(0);
            total[key] = json!(total[key].as_u64().unwrap_or(0) + add);
        }
    }
    total
}

async fn stats(State(f): State<Shared>) -> Json<Value> {
    let mut children = serde_json::Map::new();
    for (name, base) in &f.children {
        let v = get_json(&f, base, Method::GET, "/stats")
            .await
            .unwrap_or(Value::Null);
        children.insert((*name).into(), v);
    }
    let own = serde_json::to_value(proc_stats::read()).unwrap_or(Value::Null);
    let proc = sum_procs(own, children.values().map(|c| &c["proc"]));
    Json(json!({
        "uptime_s": f.started.elapsed().as_secs_f64(),
        "proc": proc,
        "runtime": {"split": children.iter().map(|(k, v)| (k.clone(), v["runtime"].clone())).collect::<serde_json::Map<_, _>>()},
        "children": children,
    }))
}

async fn trim(State(f): State<Shared>) -> Json<Value> {
    let mut children = serde_json::Map::new();
    for (name, base) in &f.children {
        let v = get_json(&f, base, Method::POST, "/admin/trim")
            .await
            .unwrap_or(Value::Null);
        children.insert((*name).into(), v);
    }
    let before = sum_procs(json!({}), children.values().map(|c| &c["before"]));
    let after = sum_procs(json!({}), children.values().map(|c| &c["after"]));
    Json(json!({"before": before, "after": after, "children": children}))
}

fn spawn_children(
    args: &SplitArgs,
) -> anyhow::Result<(BTreeMap<&'static str, String>, Vec<Child>)> {
    let exe = std::env::current_exe().context("current_exe")?;
    let plan: [(&'static str, &Option<PathBuf>, usize, &Option<String>); 3] = [
        ("eg2", &args.eg2, args.eg2_threads, &args.eg2_cpuset),
        ("vela", &args.vela, args.vela_threads, &args.vela_cpuset),
        ("ocr", &args.ocr, args.ocr_threads, &args.ocr_cpuset),
    ];
    let mut routes = BTreeMap::new();
    let mut procs = Vec::new();
    for (i, (name, dir, threads, cpuset)) in plan.into_iter().enumerate() {
        let Some(dir) = dir else { continue };
        let port = args.child_port + i as u16;
        let mut cmd = Command::new(&exe);
        cmd.arg("serve")
            .arg("--addr")
            .arg(format!("127.0.0.1:{port}"))
            .arg(format!("--{name}"))
            .arg(dir)
            .arg("--threads")
            .arg(threads.to_string())
            .arg("--exec-slots")
            .arg(args.exec_slots.to_string())
            .arg("--cpu-isa")
            .arg(&args.cpu_isa)
            .arg("--max-inflight")
            .arg(args.max_inflight.to_string())
            .arg("--workspace-limit-mib")
            .arg(args.workspace_limit_mib.to_string())
            .arg("--memory-budget-mib")
            .arg(args.memory_budget_mib.to_string())
            .arg("--malloc-arena-max")
            .arg(args.malloc_arena_max.to_string())
            .arg("--compute-order")
            .arg(match args.compute_order {
                Order::Fifo => "fifo",
                Order::Any => "any",
            })
            .arg("--tokio-workers")
            .arg(args.tokio_workers.to_string());
        if let Some(c) = cpuset {
            cmd.arg("--cpuset").arg(c).arg("--pin-process");
        }
        if name == "eg2" && args.eg2_warmup {
            cmd.arg("--eg2-warmup");
        }
        let child = cmd
            .stdin(Stdio::null())
            .spawn()
            .with_context(|| format!("spawn {name} child"))?;
        procs.push(child);
        routes.insert(name, format!("http://127.0.0.1:{port}"));
    }
    if routes.is_empty() {
        bail!("nothing to serve: pass --eg2, --vela or --ocr");
    }
    Ok((routes, procs))
}

fn stop_children(procs: &mut [Child]) {
    for c in procs.iter() {
        // SIGINT: the child's graceful shutdown unloads its model.
        // SAFETY: kill only signals the pid we spawned.
        unsafe { libc::kill(c.id() as libc::pid_t, libc::SIGINT) };
    }
    for c in procs.iter_mut() {
        let _ = c.wait();
    }
}

pub fn run(args: SplitArgs) -> anyhow::Result<()> {
    let (children, procs) = spawn_children(&args)?;
    let front = Arc::new(Front {
        http: Client::builder(TokioExecutor::new())
            .pool_max_idle_per_host(256)
            .build_http(),
        children,
        procs: Mutex::new(procs),
        started: Instant::now(),
    });
    let tokio = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(args.tokio_workers.max(1))
        .build()?;
    let result = tokio.block_on(async {
        let deadline = Instant::now() + Duration::from_secs(args.wait);
        for (name, base) in &front.children {
            while get_json(&front, base, Method::GET, "/stats").await.is_none() {
                if Instant::now() > deadline {
                    bail!("{name} child did not come up at {base}");
                }
                if let Some(status) = front
                    .procs
                    .lock()
                    .unwrap()
                    .iter_mut()
                    .find_map(|c| c.try_wait().ok().flatten())
                {
                    bail!("a child exited during startup: {status}");
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
        }
        let router = Router::new()
            .route("/health", get(health))
            .route("/work", get(work))
            .route("/stats", get(stats))
            .route("/embed", post(embed))
            .route("/vela", post(vela))
            .route("/ocr", post(ocr))
            .route("/admin/trim", post(trim))
            .route("/admin/{action}/{name}", post(admin))
            .layer(DefaultBodyLimit::max(64 << 20))
            .with_state(front.clone());
        let listener = tokio::net::TcpListener::bind(&args.addr).await?;
        eprintln!(
            "{}",
            json!({"event": "ready", "addr": listener.local_addr()?.to_string(), "children": front.children})
        );
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        anyhow::Ok(())
    });
    stop_children(&mut front.procs.lock().unwrap());
    result
}
