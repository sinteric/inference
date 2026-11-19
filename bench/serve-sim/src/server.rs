//! The host: an axum server on a multi-threaded tokio runtime that owns one
//! [`Runtime`] and calls the models in-process, the way an application
//! embeds `sinteric-inference`.
use std::path::{Path as FsPath, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use anyhow::Context;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sinteric_inference::embeddinggemma2::{self as eg2, EmbeddingGemma2};
use sinteric_inference::ppocr::{
    DecodedLine, DetConfig, Detector, ImageView, PixelFormat, Quad, RecConfig, Recognizer,
    DEFAULT_WORKSPACE_LIMIT_BYTES,
};
use sinteric_inference::vela2::{self, Vela2};
use sinteric_inference::worker::{Isolated, Remote, WorkerOptions};
use sinteric_inference::{BackendOverrides, CpuIsa, Error, Runtime, RuntimeOptions};
use tokio::sync::Semaphore;

use crate::proc_stats;

/// Where a handler runs the blocking model call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Dispatch {
    /// `tokio::task::spawn_blocking`, as `docs/library.md` says.
    Blocking,
    /// Directly on the async worker. The anti-pattern, kept so its effect on
    /// the rest of the server can be measured.
    Inline,
}

#[derive(clap::Args, Clone, Debug)]
pub struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1:8090")]
    addr: String,
    /// `RuntimeOptions::threads` (pinned pool width).
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// `RuntimeOptions::exec_slots`.
    #[arg(long, default_value_t = 1)]
    exec_slots: usize,
    /// `RuntimeOptions::cpuset`.
    #[arg(long)]
    cpuset: Option<String>,
    /// `RuntimeOptions::pin_process`. Set before tokio starts, so tokio's
    /// threads inherit it.
    #[arg(long)]
    pin_process: bool,
    /// `best_available` or `avx512_core_bf16`.
    #[arg(long, default_value = "best_available")]
    cpu_isa: String,
    /// `RuntimeOptions::workspace_limit_bytes` in MiB (0 = library default):
    /// how much activation and convolution workspace each model keeps
    /// between calls.
    #[arg(long, default_value_t = 0)]
    workspace_limit_mib: usize,
    /// `RuntimeOptions::memory_budget_bytes` in MiB (0 = unbounded): what
    /// each model may keep between calls.
    #[arg(long, default_value_t = 0)]
    memory_budget_mib: usize,
    /// `RuntimeOptions::malloc_arena_max` (0 = glibc default).
    #[arg(long, default_value_t = 0)]
    malloc_arena_max: usize,
    /// EmbeddingGemma 2 model directory.
    #[arg(long)]
    eg2: Option<PathBuf>,
    #[arg(long)]
    eg2_warmup: bool,
    /// Vela 2.0 model directory.
    #[arg(long)]
    vela: Option<PathBuf>,
    /// PP-OCR root (`det/` and `rec/`).
    #[arg(long)]
    ocr: Option<PathBuf>,
    /// Tokio worker threads (tokio's default is one per CPU).
    #[arg(long)]
    tokio_workers: Option<usize>,
    /// Cap on tokio's blocking-thread pool.
    #[arg(long, default_value_t = 512)]
    max_blocking: usize,
    /// In-flight inference calls allowed per model; the rest wait on a
    /// semaphore (0 = unlimited).
    #[arg(long, default_value_t = 0)]
    max_inflight: usize,
    #[arg(long, value_enum, default_value_t = Dispatch::Blocking)]
    dispatch: Dispatch,
    /// Which waiting call gets the library's compute lock next.
    #[arg(long, value_enum, default_value_t = Order::Fifo)]
    compute_order: Order,
    /// Run each model in its own worker process (`worker::Isolated`) with
    /// the `--{eg2,vela,ocr}-threads` / `-cpuset` / `-budget-mib` below;
    /// this process only serves HTTP.
    #[arg(long)]
    isolate: bool,
    #[arg(long, default_value_t = 1)]
    eg2_threads: usize,
    #[arg(long)]
    eg2_cpuset: Option<String>,
    #[arg(long, default_value_t = 2)]
    vela_threads: usize,
    #[arg(long)]
    vela_cpuset: Option<String>,
    #[arg(long, default_value_t = 1)]
    ocr_threads: usize,
    #[arg(long)]
    ocr_cpuset: Option<String>,
    /// Per-worker `--memory-budget-mib` overrides.
    #[arg(long)]
    eg2_budget_mib: Option<usize>,
    #[arg(long)]
    vela_budget_mib: Option<usize>,
    #[arg(long)]
    ocr_budget_mib: Option<usize>,
}

/// `rt_threads::ComputeOrder`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Order {
    Fifo,
    Any,
}

struct Slot<M> {
    name: &'static str,
    dir: Option<PathBuf>,
    model: RwLock<Option<Arc<M>>>,
    gate: Option<Semaphore>,
    inflight: AtomicUsize,
    served: AtomicU64,
    failed: AtomicU64,
    unavailable: AtomicU64,
}

impl<M> Slot<M> {
    fn new(name: &'static str, dir: Option<PathBuf>, max_inflight: usize) -> Self {
        Self {
            name,
            dir,
            model: RwLock::new(None),
            gate: (max_inflight > 0).then(|| Semaphore::new(max_inflight)),
            inflight: AtomicUsize::new(0),
            served: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            unavailable: AtomicU64::new(0),
        }
    }

    fn get(&self) -> Option<Arc<M>> {
        self.model.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn replace(&self, model: Option<Arc<M>>) -> Option<Arc<M>> {
        std::mem::replace(
            &mut *self.model.write().unwrap_or_else(|e| e.into_inner()),
            model,
        )
    }

    fn stats(&self) -> Value {
        json!({
            "configured": self.dir.is_some(),
            "loaded": self.get().is_some(),
            "inflight": self.inflight.load(Ordering::Relaxed),
            "served": self.served.load(Ordering::Relaxed),
            "failed": self.failed.load(Ordering::Relaxed),
            "unavailable": self.unavailable.load(Ordering::Relaxed),
        })
    }
}

/// A model in this process or in a worker process.
enum Hosted<M: Remote> {
    Local(M),
    Worker(Box<Isolated<M>>),
}

/// What a slot can release once its last caller is done.
trait Unload {
    fn unload(self);
}

impl<M: Remote> Unload for Hosted<M> {
    fn unload(self) {
        Hosted::unload(self)
    }
}

impl<M: Remote> Hosted<M> {
    fn load(
        app_rt: &Runtime,
        config: M::Config,
        worker: Option<WorkerOptions>,
    ) -> sinteric_inference::Result<Self> {
        Ok(match worker {
            Some(options) => Self::Worker(Box::new(Isolated::load(app_rt, config, options)?)),
            None => Self::Local(app_rt.load(config)?),
        })
    }
    fn unload(self) {
        match self {
            Self::Local(m) => m.unload(),
            Self::Worker(m) => m.unload(),
        }
    }
}

impl Hosted<EmbeddingGemma2> {
    fn embed(
        &self,
        texts: &[String],
        kind: eg2::PromptKind,
    ) -> sinteric_inference::Result<eg2::Embeddings> {
        match self {
            Self::Local(m) => m.embed(texts, kind),
            Self::Worker(m) => m.embed(texts, kind),
        }
    }
    fn backend_stats(&self) -> Value {
        match self {
            Self::Local(m) => m.backend_stats(),
            Self::Worker(m) => m.backend_stats().unwrap_or_default(),
        }
    }
}

impl Hosted<Vela2> {
    fn system_one(&self, body: &Value) -> sinteric_inference::Result<Value> {
        match self {
            Self::Local(m) => m.system_one(body),
            Self::Worker(m) => m.system_one(body),
        }
    }
    fn backend_stats(&self) -> Value {
        match self {
            Self::Local(m) => m.backend_stats(),
            Self::Worker(m) => m.backend_stats().unwrap_or_default(),
        }
    }
}

impl Hosted<Detector> {
    fn detect(&self, image: &ImageView<'_>) -> sinteric_inference::Result<Vec<Quad>> {
        match self {
            Self::Local(m) => m.detect(image),
            Self::Worker(m) => m.detect(image),
        }
    }
    fn backend_stats(&self) -> Value {
        match self {
            Self::Local(m) => m.backend_stats(),
            Self::Worker(m) => m.backend_stats().unwrap_or_default(),
        }
    }
}

impl Hosted<Recognizer> {
    fn recognize(
        &self,
        image: &ImageView<'_>,
        quads: &[Quad],
    ) -> sinteric_inference::Result<Vec<DecodedLine>> {
        match self {
            Self::Local(m) => m.recognize(image, quads),
            Self::Worker(m) => m.recognize(image, quads),
        }
    }
}

/// Page OCR as a host composes it: PP-OCR detector, then the recognizer on
/// the detected quads, lines scoring below [`DROP_SCORE`] dropped. In process
/// both models share one backend; isolated, each runs in its own worker.
struct PageOcr {
    det: Hosted<Detector>,
    rec: Hosted<Recognizer>,
}

/// PP-OCR's default recognition score threshold.
const DROP_SCORE: f32 = 0.5;

#[derive(Serialize)]
struct Line {
    quad: Quad,
    text: String,
    score: f32,
}

impl PageOcr {
    fn load(
        app_rt: &Runtime,
        root: &FsPath,
        worker: Option<WorkerOptions>,
    ) -> sinteric_inference::Result<Self> {
        let (det, rec) = (
            DetConfig::new(root.join("det")),
            RecConfig::new(root.join("rec")),
        );
        let rt = match worker {
            Some(_) => app_rt.clone(),
            None => app_rt.shared_backend(
                BackendOverrides::default()
                    .with_workspace_limit_bytes(Some(DEFAULT_WORKSPACE_LIMIT_BYTES)),
            )?,
        };
        Ok(Self {
            det: Hosted::load(&rt, det, worker.clone())?,
            rec: Hosted::load(&rt, rec, worker)?,
        })
    }
    fn recognize(&self, image: &ImageView<'_>) -> sinteric_inference::Result<Vec<Line>> {
        let quads = self.det.detect(image)?;
        let lines = self.rec.recognize(image, &quads)?;
        Ok(quads
            .into_iter()
            .zip(lines)
            .filter(|(_, l)| l.score >= DROP_SCORE)
            .map(|(quad, l)| Line {
                quad,
                text: l.text,
                score: l.score,
            })
            .collect())
    }
    /// The detector's backend counters; in process that backend is shared
    /// with the recognizer.
    fn backend_stats(&self) -> Value {
        self.det.backend_stats()
    }
}

impl Unload for PageOcr {
    fn unload(self) {
        self.det.unload();
        self.rec.unload();
    }
}

impl<M: Unload> Slot<M> {
    /// Take the model out of the slot so new requests get 503, wait for the
    /// in-flight calls to drop their `Arc`, then unload. Blocking.
    fn unload(&self) -> Option<Duration> {
        let mut model = self.replace(None)?;
        let start = Instant::now();
        loop {
            match Arc::try_unwrap(model) {
                Ok(m) => {
                    m.unload();
                    return Some(start.elapsed());
                }
                Err(back) => {
                    model = back;
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }
}

struct App {
    rt: Runtime,
    args: ServeArgs,
    started: Instant,
    ocr_max_pixels: usize,
    eg2: Slot<Hosted<EmbeddingGemma2>>,
    vela: Slot<Hosted<Vela2>>,
    ocr: Slot<PageOcr>,
}

type Shared = Arc<App>;

impl App {
    /// The worker for `name` with `--isolate`, else `None` (in-process).
    fn worker(&self, name: &str) -> Option<WorkerOptions> {
        let a = &self.args;
        let (threads, cpuset, budget) = match name {
            "eg2" => (a.eg2_threads, &a.eg2_cpuset, a.eg2_budget_mib),
            "vela" => (a.vela_threads, &a.vela_cpuset, a.vela_budget_mib),
            _ => (a.ocr_threads, &a.ocr_cpuset, a.ocr_budget_mib),
        };
        let mut options = WorkerOptions::new(threads);
        if let Some(c) = cpuset {
            options = options.with_cpuset(c.clone());
        }
        if let Some(mib) = budget {
            options = options.with_memory_budget_bytes(mib << 20);
        }
        a.isolate.then_some(options)
    }
}

impl App {
    /// Load one configured model into its slot. Blocking.
    fn load(&self, name: &str) -> Result<Duration, ApiError> {
        let start = Instant::now();
        fn fill<M>(
            slot: &Slot<M>,
            load: impl FnOnce(&FsPath) -> sinteric_inference::Result<M>,
        ) -> Result<(), ApiError> {
            let dir = slot.dir.as_ref().ok_or(ApiError::Unavailable(slot.name))?;
            if slot.get().is_none() {
                slot.replace(Some(Arc::new(load(dir).map_err(ApiError::Model)?)));
            }
            Ok(())
        }
        let worker = self.worker(name);
        match name {
            "eg2" => fill(&self.eg2, |d| {
                let config = eg2::Config::new(d).with_warmup(self.args.eg2_warmup);
                Hosted::load(&self.rt, config, worker)
            })?,
            "vela" => fill(&self.vela, |d| {
                Hosted::load(&self.rt, vela2::Config::new(d), worker)
            })?,
            "ocr" => fill(&self.ocr, |d| PageOcr::load(&self.rt, d, worker))?,
            other => return Err(ApiError::BadRequest(format!("unknown model {other}"))),
        }
        Ok(start.elapsed())
    }

    fn unload(&self, name: &str) -> Result<Option<Duration>, ApiError> {
        Ok(match name {
            "eg2" => self.eg2.unload(),
            "vela" => self.vela.unload(),
            "ocr" => self.ocr.unload(),
            other => return Err(ApiError::BadRequest(format!("unknown model {other}"))),
        })
    }
}

enum ApiError {
    Unavailable(&'static str),
    BadRequest(String),
    Model(Error),
    Join(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, msg) = match self {
            Self::Unavailable(m) => (StatusCode::SERVICE_UNAVAILABLE, format!("{m} not loaded")),
            Self::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
            Self::Model(e) => {
                let status = match e {
                    Error::Request(_) | Error::Shape(_) | Error::Config(_) => {
                        StatusCode::UNPROCESSABLE_ENTITY
                    }
                    _ => StatusCode::INTERNAL_SERVER_ERROR,
                };
                (status, e.to_string())
            }
            Self::Join(m) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("model call panicked: {m}"),
            ),
        };
        (status, Json(json!({ "error": msg }))).into_response()
    }
}

/// Server-side split of one request's latency.
#[derive(Clone, Copy, Debug, Default, Serialize)]
struct Timing {
    /// Handler entry until the model call starts (semaphore + blocking pool).
    queue_ms: f64,
    /// The model call itself, including waits on the library's own locks.
    infer_ms: f64,
}

fn ms(d: Duration) -> f64 {
    (d.as_secs_f64() * 1e6).round() / 1e3
}

struct InflightGuard<'a>(&'a AtomicUsize);

impl Drop for InflightGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

async fn call<M, T, F>(app: &App, slot: &Slot<M>, f: F) -> Result<(T, Timing), ApiError>
where
    M: Send + Sync + 'static,
    T: Send + 'static,
    F: FnOnce(&M) -> sinteric_inference::Result<T> + Send + 'static,
{
    let arrived = Instant::now();
    let Some(model) = slot.get() else {
        slot.unavailable.fetch_add(1, Ordering::Relaxed);
        return Err(ApiError::Unavailable(slot.name));
    };
    let _permit = match &slot.gate {
        Some(gate) => Some(
            gate.acquire()
                .await
                .map_err(|_| ApiError::Unavailable(slot.name))?,
        ),
        None => None,
    };
    slot.inflight.fetch_add(1, Ordering::Relaxed);
    let _inflight = InflightGuard(&slot.inflight);
    let run = move || {
        let start = Instant::now();
        let out = f(&model);
        (out, start, Instant::now())
    };
    let (out, start, end) = match app.args.dispatch {
        Dispatch::Blocking => tokio::task::spawn_blocking(run)
            .await
            .map_err(|e| ApiError::Join(e.to_string()))?,
        Dispatch::Inline => run(),
    };
    let timing = Timing {
        queue_ms: ms(start - arrived),
        infer_ms: ms(end - start),
    };
    match out {
        Ok(v) => {
            slot.served.fetch_add(1, Ordering::Relaxed);
            Ok((v, timing))
        }
        Err(e) => {
            slot.failed.fetch_add(1, Ordering::Relaxed);
            Err(ApiError::Model(e))
        }
    }
}

fn digest(bytes: &[u8]) -> String {
    hex::encode(&Sha256::digest(bytes)[..8])
}

#[derive(Deserialize)]
struct EmbedRequest {
    texts: Vec<String>,
    #[serde(default)]
    kind: Kind,
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Kind {
    #[default]
    Query,
    Document,
}

async fn embed(
    State(app): State<Shared>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<Value>, ApiError> {
    let kind = match req.kind {
        Kind::Query => eg2::PromptKind::Query,
        Kind::Document => eg2::PromptKind::Document,
    };
    let texts = req.texts;
    let (rows, timing) = call(&app, &app.eg2, move |m| m.embed(&texts, kind)).await?;
    let bits: Vec<u8> = rows
        .data
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect();
    Ok(Json(json!({
        "n": rows.n(),
        "dim": rows.dim,
        "digest": digest(&bits),
        "timing": timing,
    })))
}

async fn vela(State(app): State<Shared>, Json(body): Json<Value>) -> Result<Json<Value>, ApiError> {
    let (out, timing) = call(&app, &app.vela, move |m| m.system_one(&body)).await?;
    let bytes = serde_json::to_vec(&out).map_err(|e| ApiError::BadRequest(e.to_string()))?;
    Ok(Json(json!({
        "digest": digest(&bytes),
        "response": out,
        "timing": timing,
    })))
}

async fn ocr(State(app): State<Shared>, body: Bytes) -> Result<Json<Value>, ApiError> {
    let max_pixels = app.ocr_max_pixels;
    let ((lines, decode_ms), timing) = call(&app, &app.ocr, move |m| {
        let start = Instant::now();
        let (bgr, w, h) = ppocr::decode::decode_bgr(&body, max_pixels)?;
        let decode_ms = ms(start.elapsed());
        let stride = bgr_stride(w)?;
        let image = ImageView::new(&bgr, w, h, stride, PixelFormat::Bgr8)?;
        Ok((m.recognize(&image)?, decode_ms))
    })
    .await?;
    let bytes = serde_json::to_vec(&lines).map_err(|e| ApiError::BadRequest(e.to_string()))?;
    Ok(Json(json!({
        "lines": lines.len(),
        "chars": lines.iter().map(|l| l.text.chars().count()).sum::<usize>(),
        "digest": digest(&bytes),
        "decode_ms": decode_ms,
        "timing": timing,
    })))
}

fn bgr_stride(w: u32) -> sinteric_inference::Result<usize> {
    (w as usize)
        .checked_mul(3)
        .ok_or_else(|| Error::Shape("image stride overflows".into()))
}

pub(crate) async fn health() -> &'static str {
    "ok"
}

#[derive(Deserialize)]
pub(crate) struct WorkQuery {
    #[serde(default)]
    us: u64,
}

/// The host's own CPU-bound handler work (JSON shaping, auth, templating),
/// done inline on the async worker as a real handler would.
pub(crate) async fn work(Query(q): Query<WorkQuery>) -> &'static str {
    let end = Instant::now() + Duration::from_micros(q.us.min(1_000_000));
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    while Instant::now() < end {
        for _ in 0..64 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
        }
    }
    std::hint::black_box(x);
    "ok"
}

async fn stats(State(app): State<Shared>) -> Json<Value> {
    // A worker's `backend_stats` is a round trip that can queue behind its
    // calls in flight: keep it off the async worker like any model call.
    let stats = tokio::task::spawn_blocking(move || collect_stats(&app)).await;
    Json(stats.unwrap_or_else(|e| json!({"error": e.to_string()})))
}

fn collect_stats(app: &App) -> Value {
    json!({
        "uptime_s": app.started.elapsed().as_secs_f64(),
        "proc": proc_stats::read(),
        "runtime": {
            "threads": app.rt.options().threads,
            "exec_slots": app.rt.options().exec_slots,
            "workspace_limit_mib": app.rt.workspace_limit() >> 20,
            "memory_budget_mib": app.rt.options().memory_budget_bytes >> 20,
            "malloc_arena_max": app.rt.options().malloc_arena_max,
            "dispatch": format!("{:?}", app.args.dispatch),
            "isolate": app.args.isolate,
            "compute_order": format!("{:?}", rt_threads::compute_order()),
        },
        "models": {
            "eg2": app.eg2.stats(),
            "vela": app.vela.stats(),
            "ocr": app.ocr.stats(),
        },
        "memory": {
            "eg2": app.eg2.get().map(|m| memory(&m.backend_stats())),
            "vela": app.vela.get().map(|m| memory(&m.backend_stats())),
            "ocr": app.ocr.get().map(|m| memory(&m.backend_stats())),
        },
    })
}

/// What a backend keeps between calls, from its counters, in MiB.
fn memory(stats: &Value) -> Value {
    let mib = |k: &str| stats[k].as_u64().map(|b| b >> 20);
    json!({
        "packed_weights": mib("packed_weight_bytes"),
        "conv_packed_weights": mib("conv_packed_weight_bytes"),
        "conv_arena": mib("conv_arena_bytes"),
        "conv_scratch": mib("conv_scratch_bytes"),
        "execute_scratch": mib("execute_scratch_bytes"),
        "retained_buffers": mib("retained_output_buffer_bytes"),
        "cached_plans": stats["cached_plans"],
        "plans": mib("plan_bytes"),
        "conv_plans": mib("conv_plan_bytes"),
        "retained": mib("retained_bytes"),
        "budget": mib("memory_budget_bytes"),
        "budget_evictions": stats["budget_evictions"],
        "budget_evicted": mib("budget_evicted_bytes"),
        "budget_trims": stats["budget_trims"],
    })
}

async fn admin_unload(
    State(app): State<Shared>,
    Path(name): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let before = proc_stats::read().rss_mib;
    let a = app.clone();
    let drained = tokio::task::spawn_blocking(move || a.unload(&name))
        .await
        .map_err(|e| ApiError::Join(e.to_string()))??;
    Ok(Json(json!({
        "unloaded": drained.is_some(),
        "drain_ms": drained.map(ms),
        "rss_before_mib": before,
        "rss_after_mib": proc_stats::read().rss_mib,
    })))
}

async fn admin_load(
    State(app): State<Shared>,
    Path(name): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let before = proc_stats::read().rss_mib;
    let a = app.clone();
    let took = tokio::task::spawn_blocking(move || a.load(&name))
        .await
        .map_err(|e| ApiError::Join(e.to_string()))??;
    Ok(Json(json!({
        "load_ms": ms(took),
        "rss_before_mib": before,
        "rss_after_mib": proc_stats::read().rss_mib,
    })))
}

/// `malloc_trim(0)` with models loaded: how much of RSS is freed heap the
/// allocator kept rather than memory the models still hold.
async fn admin_trim() -> Result<Json<Value>, ApiError> {
    let before = proc_stats::read();
    let t = Instant::now();
    let released = tokio::task::spawn_blocking(sinteric_inference::release_free_memory)
        .await
        .map_err(|e| ApiError::Join(e.to_string()))?;
    Ok(Json(json!({
        "released": released,
        "trim_ms": ms(t.elapsed()),
        "before": before,
        "after": proc_stats::read(),
    })))
}

fn log(event: &str, extra: Value) {
    let mut v = json!({"event": event, "proc": proc_stats::read()});
    if let (Some(o), Value::Object(e)) = (v.as_object_mut(), extra) {
        o.extend(e);
    }
    eprintln!("{v}");
}

pub fn run(args: ServeArgs) -> anyhow::Result<()> {
    rt_threads::set_compute_order(match args.compute_order {
        Order::Fifo => rt_threads::ComputeOrder::Fifo,
        Order::Any => rt_threads::ComputeOrder::Any,
    });
    let mut options = RuntimeOptions::new()
        .with_threads(args.threads)
        .with_exec_slots(args.exec_slots)
        .with_cpu_isa(CpuIsa::from_name(&args.cpu_isa)?)
        .with_pin_process(args.pin_process)
        .with_workspace_limit_bytes(args.workspace_limit_mib << 20)
        .with_memory_budget_bytes(args.memory_budget_mib << 20);
    if args.malloc_arena_max > 0 {
        options = options.with_malloc_arena_max(args.malloc_arena_max);
    }
    if let Some(cpuset) = &args.cpuset {
        options = options.with_cpuset(cpuset.clone());
    }
    // Before tokio exists: `pin_process` affinity is per thread on Linux and
    // only threads created afterwards inherit it.
    let rt = Runtime::new(options).context("Runtime::new")?;
    log("runtime", json!({}));
    let app = Arc::new(App {
        ocr_max_pixels: DetConfig::new("unused").options.max_pixels,
        eg2: Slot::new("eg2", args.eg2.clone(), args.max_inflight),
        vela: Slot::new("vela", args.vela.clone(), args.max_inflight),
        ocr: Slot::new("ocr", args.ocr.clone(), args.max_inflight),
        rt,
        args: args.clone(),
        started: Instant::now(),
    });
    for name in ["eg2", "vela", "ocr"] {
        let configured = match name {
            "eg2" => app.eg2.dir.is_some(),
            "vela" => app.vela.dir.is_some(),
            _ => app.ocr.dir.is_some(),
        };
        if configured {
            let took = app
                .load(name)
                .map_err(|e| anyhow::anyhow!("load {name}: {}", api_error_text(e)))?;
            log("loaded", json!({"model": name, "load_ms": ms(took)}));
        }
    }

    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder
        .enable_all()
        .max_blocking_threads(args.max_blocking.max(1));
    if let Some(n) = args.tokio_workers {
        builder.worker_threads(n.max(1));
    }
    let tokio = builder.build()?;
    tokio.block_on(async {
        let router = Router::new()
            .route("/health", get(health))
            .route("/work", get(work))
            .route("/stats", get(stats))
            .route("/embed", post(embed))
            .route("/vela", post(vela))
            .route("/ocr", post(ocr))
            .route("/admin/unload/{name}", post(admin_unload))
            .route("/admin/load/{name}", post(admin_load))
            .route("/admin/trim", post(admin_trim))
            .layer(DefaultBodyLimit::max(64 << 20))
            .with_state(app.clone());
        let listener = tokio::net::TcpListener::bind(&args.addr).await?;
        log("ready", json!({"addr": listener.local_addr()?.to_string()}));
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        anyhow::Ok(())
    })?;
    for name in ["eg2", "vela", "ocr"] {
        let _ = app.unload(name);
    }
    log("stopped", json!({}));
    Ok(())
}

fn api_error_text(e: ApiError) -> String {
    match e {
        ApiError::Unavailable(m) => format!("{m} not configured"),
        ApiError::BadRequest(m) | ApiError::Join(m) => m,
        ApiError::Model(e) => e.to_string(),
    }
}
