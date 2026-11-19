//! Run a model in its own worker process.
//!
//! ```ignore
//! fn main() {
//!     // First line of the host's main: a worker process re-runs this binary
//!     // and never returns from here.
//!     sinteric_inference::worker::main_if_worker();
//!
//!     let rt = Runtime::new(RuntimeOptions::new().with_threads(1))?;
//!     let vela = Isolated::<Vela2>::load(&rt, vela2::Config::new(dir), WorkerOptions::new(2).with_cpuset("1-2"))?;
//!     let reply = vela.system_one(&body)?;
//! }
//! ```
//!
//! [`Isolated<M>`] has the same call methods as `M`; the model computes in a
//! child process with its own thread pool, heap and memory budget, so models
//! in different workers run in parallel instead of taking turns on one
//! process's pool. The child is the host binary itself, started with a marker
//! argument that [`main_if_worker`] intercepts, so there is one artifact to
//! deploy. Requests and results go over a Unix socket pair.
//!
//! If the worker dies (a crash, the OOM killer), calls in flight fail with
//! [`Error::Worker`] and the next call starts a new worker. Dropping or
//! [`Isolated::unload`] ends the process, which returns all of its memory.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::io::{self, Read, Write};
use std::marker::PhantomData;
use std::os::fd::{AsFd, AsRawFd, FromRawFd};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use rt_core::{Error, Result, SystemOneError};
use rt_runtime::{Model, Runtime, RuntimeOptions};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const MARKER: &str = "--sinteric-inference-worker";
/// The child's end of the socket pair.
const WORKER_FD: i32 = 3;
const MAX_HEADER: u32 = 256 << 20;
const MAX_PAYLOAD: u64 = 4 << 30;
/// How long a dropped worker gets to exit on its own before it is killed.
const EXIT_GRACE: Duration = Duration::from_secs(5);

/// Where and how a worker runs.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct WorkerOptions {
    /// The worker's thread pool width.
    pub threads: usize,
    /// CPUs the worker is pinned to (`"1-2"`, `"0,2"`); the whole worker
    /// process is pinned. `None` leaves it unpinned.
    pub cpuset: Option<String>,
    /// Calls the worker handles at once (tokenizing one while another
    /// computes). Compute still takes turns inside the worker.
    pub handlers: usize,
    /// This worker's `RuntimeOptions::memory_budget_bytes`, so each model
    /// can get its own budget; `None` keeps the host runtime's.
    pub memory_budget_bytes: Option<usize>,
    /// Binary to start; default the running executable.
    pub executable: Option<PathBuf>,
}

impl WorkerOptions {
    pub fn new(threads: usize) -> Self {
        Self {
            threads,
            cpuset: None,
            handlers: 4,
            memory_budget_bytes: None,
            executable: None,
        }
    }
    pub fn with_cpuset(mut self, cpuset: impl Into<String>) -> Self {
        self.cpuset = Some(cpuset.into());
        self
    }
    pub fn with_handlers(mut self, n: usize) -> Self {
        self.handlers = n;
        self
    }
    pub fn with_memory_budget_bytes(mut self, n: usize) -> Self {
        self.memory_budget_bytes = Some(n);
        self
    }
    pub fn with_executable(mut self, path: impl Into<PathBuf>) -> Self {
        self.executable = Some(path.into());
        self
    }
}

/// A model that can run in a worker process.
pub trait Remote: Model {
    #[doc(hidden)]
    const KIND: &'static str;
    #[doc(hidden)]
    fn config_json(config: &Self::Config) -> Result<Value>;
}

/// `M` running in a worker process; see the [module docs](self).
pub struct Isolated<M: Remote> {
    init: Value,
    options: WorkerOptions,
    conn: Mutex<Option<Arc<Conn>>>,
    info: Value,
    _model: PhantomData<fn() -> M>,
}

impl<M: Remote> Isolated<M> {
    /// Start a worker and load the model there. The worker's runtime copies
    /// `rt`'s options except threads, cpuset, pinning and (when set) the
    /// memory budget, which come from `options`.
    pub fn load(rt: &Runtime, config: M::Config, options: WorkerOptions) -> Result<Self> {
        if options.threads == 0 || options.handlers == 0 {
            return Err(Error::Config(
                "worker threads and handlers must be positive".into(),
            ));
        }
        let mut runtime = rt.options().clone().with_threads(options.threads);
        runtime.cpuset = options.cpuset.clone();
        runtime.pin_process = options.cpuset.is_some();
        if let Some(budget) = options.memory_budget_bytes {
            runtime.memory_budget_bytes = budget;
        }
        let init = json!({
            "kind": M::KIND,
            "runtime": runtime,
            "config": M::config_json(&config)?,
            "handlers": options.handlers,
        });
        let conn = Conn::spawn(&init, &options)?;
        let info = conn.info.clone();
        Ok(Self {
            init,
            options,
            conn: Mutex::new(Some(Arc::new(conn))),
            info,
            _model: PhantomData,
        })
    }

    /// End the worker process; all of its memory goes back to the OS.
    pub fn unload(self) {}

    /// The worker's process id, if one is running.
    pub fn pid(&self) -> Option<u32> {
        let conn = self.conn.lock().ok()?;
        conn.as_ref().filter(|c| c.alive()).map(|c| c.pid)
    }

    /// Backend counters reported by the worker (`backend_stats` of the model).
    #[doc(hidden)]
    pub fn backend_stats(&self) -> Result<Value> {
        Ok(self.call("backend_stats", Value::Null, Vec::new())?.0)
    }

    fn conn(&self) -> Result<Arc<Conn>> {
        let mut slot = self
            .conn
            .lock()
            .map_err(|_| Error::Poisoned("worker connection"))?;
        if let Some(conn) = slot.as_ref().filter(|c| c.alive()) {
            return Ok(conn.clone());
        }
        // The old worker is gone: reap it and start another.
        *slot = None;
        let conn = Arc::new(Conn::spawn(&self.init, &self.options)?);
        *slot = Some(conn.clone());
        Ok(conn)
    }

    fn call(&self, method: &str, args: Value, payload: Vec<u8>) -> Result<(Value, Vec<u8>)> {
        self.conn()?.call(method, args, payload)
    }
}

// ---- parent side ----------------------------------------------------------

type Reply = Result<(Value, Vec<u8>)>;

/// State the reader thread shares with callers. `None` once the worker is
/// gone: new calls fail at once instead of waiting forever.
struct Pending(Mutex<Option<HashMap<u64, mpsc::Sender<Reply>>>>);

impl Pending {
    fn fail_all(&self, reason: &str) {
        let waiting = self.0.lock().unwrap_or_else(|e| e.into_inner()).take();
        for (_, tx) in waiting.into_iter().flatten() {
            let _ = tx.send(Err(Error::Worker(reason.into())));
        }
    }
}

struct Conn {
    pid: u32,
    info: Value,
    writer: Mutex<UnixStream>,
    pending: Arc<Pending>,
    next: AtomicU64,
    child: Mutex<Child>,
    reader: Option<JoinHandle<()>>,
}

impl Conn {
    fn spawn(init: &Value, options: &WorkerOptions) -> Result<Self> {
        let (stream, theirs) = UnixStream::pair()?;
        let exe = match &options.executable {
            Some(path) => path.clone(),
            None => std::env::current_exe()?,
        };
        let fd = theirs.as_raw_fd();
        let mut command = Command::new(exe);
        // The worker never writes to the host's stdout (a host may speak a
        // protocol there); anything it prints goes to stderr.
        let stderr = io::stderr().as_fd().try_clone_to_owned()?;
        command
            .arg(MARKER)
            .stdin(Stdio::null())
            .stdout(Stdio::from(stderr));
        // SAFETY: only async-signal-safe calls (dup2, fcntl) between fork
        // and exec.
        unsafe {
            command.pre_exec(move || {
                if fd == WORKER_FD {
                    if libc::fcntl(fd, libc::F_SETFD, 0) < 0 {
                        return Err(io::Error::last_os_error());
                    }
                } else if libc::dup2(fd, WORKER_FD) < 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = command
            .spawn()
            .map_err(|e| Error::Worker(format!("start worker: {e}")))?;
        drop(theirs);
        let ready = (|| {
            write_frame(&stream, 0, &serde_json::to_vec(init)?, &[])?;
            match read_frame(&mut &stream)? {
                Some((_, header, _)) => decode(&header),
                None => Err(Error::Worker("worker exited while loading".into())),
            }
        })();
        let info = match ready {
            Ok(info) => info,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(e);
            }
        };
        let pending = Arc::new(Pending(Mutex::new(Some(HashMap::new()))));
        let reader = {
            let mut stream = stream.try_clone()?;
            let pending = pending.clone();
            std::thread::Builder::new()
                .name("sinteric-worker-reader".into())
                .spawn(move || read_replies(&mut stream, &pending))?
        };
        Ok(Self {
            pid: child.id(),
            info,
            writer: Mutex::new(stream),
            pending,
            next: AtomicU64::new(1),
            child: Mutex::new(child),
            reader: Some(reader),
        })
    }

    fn alive(&self) -> bool {
        self.pending.0.lock().map(|p| p.is_some()).unwrap_or(false)
    }

    fn call(&self, method: &str, args: Value, payload: Vec<u8>) -> Reply {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::channel();
        {
            let mut pending = self
                .pending
                .0
                .lock()
                .map_err(|_| Error::Poisoned("worker calls"))?;
            let Some(pending) = pending.as_mut() else {
                return Err(Error::Worker("worker process exited".into()));
            };
            pending.insert(id, tx);
        }
        let header = serde_json::to_vec(&json!({"method": method, "args": args}))?;
        let sent = {
            let writer = self
                .writer
                .lock()
                .map_err(|_| Error::Poisoned("worker socket"))?;
            write_frame(&writer, id, &header, &payload)
        };
        if let Err(e) = sent {
            if let Ok(mut pending) = self.pending.0.lock() {
                if let Some(pending) = pending.as_mut() {
                    pending.remove(&id);
                }
            }
            return Err(Error::Worker(format!("send to worker: {e}")));
        }
        rx.recv()
            .unwrap_or_else(|_| Err(Error::Worker("worker process exited".into())))
    }
}

impl Drop for Conn {
    fn drop(&mut self) {
        // EOF tells the worker to exit; it finishes no new work.
        if let Ok(writer) = self.writer.lock() {
            let _ = writer.shutdown(std::net::Shutdown::Both);
        }
        if let Ok(mut child) = self.child.lock() {
            let deadline = Instant::now() + EXIT_GRACE;
            while matches!(child.try_wait(), Ok(None)) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            if matches!(child.try_wait(), Ok(None)) {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

fn read_replies(stream: &mut UnixStream, pending: &Pending) {
    loop {
        let (id, header, payload) = match read_frame(stream) {
            Ok(Some(frame)) => frame,
            Ok(None) => break pending.fail_all("worker process exited"),
            Err(e) => break pending.fail_all(&format!("worker socket: {e}")),
        };
        let tx = pending
            .0
            .lock()
            .ok()
            .and_then(|mut p| p.as_mut()?.remove(&id));
        if let Some(tx) = tx {
            let _ = tx.send(decode(&header).map(|value| (value, payload)));
        }
    }
}

// ---- wire format ------------------------------------------------------------

/// `id: u64, header length: u32, payload length: u64` (little endian), then a
/// JSON header and a binary payload.
fn write_frame(mut w: &UnixStream, id: u64, header: &[u8], payload: &[u8]) -> io::Result<()> {
    let mut head = Vec::with_capacity(20 + header.len());
    head.extend_from_slice(&id.to_le_bytes());
    head.extend_from_slice(&(header.len() as u32).to_le_bytes());
    head.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    head.extend_from_slice(header);
    w.write_all(&head)?;
    w.write_all(payload)
}

fn read_frame(r: &mut impl Read) -> io::Result<Option<(u64, Vec<u8>, Vec<u8>)>> {
    let mut head = [0u8; 20];
    let mut got = 0;
    while got < head.len() {
        match r.read(&mut head[got..]) {
            Ok(0) if got == 0 => return Ok(None),
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => got += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    let id = u64::from_le_bytes(head[..8].try_into().unwrap_or_default());
    let header_len = u32::from_le_bytes(head[8..12].try_into().unwrap_or_default());
    let payload_len = u64::from_le_bytes(head[12..].try_into().unwrap_or_default());
    if header_len > MAX_HEADER || payload_len > MAX_PAYLOAD {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "worker frame too large",
        ));
    }
    let mut header = vec![0; header_len as usize];
    r.read_exact(&mut header)?;
    let mut payload = vec![0; payload_len as usize];
    r.read_exact(&mut payload)?;
    Ok(Some((id, header, payload)))
}

/// A reply header: `{"ok": value}` or `{"err": {...}}`.
fn decode(header: &[u8]) -> Result<Value> {
    let mut reply: Value = serde_json::from_slice(header)?;
    if let Some(ok) = reply.get_mut("ok") {
        return Ok(ok.take());
    }
    let err = reply.get("err").cloned().unwrap_or(Value::Null);
    Err(remote_error(&err))
}

fn remote_error(err: &Value) -> Error {
    let message = err["message"]
        .as_str()
        .unwrap_or("worker error")
        .to_string();
    match err["kind"].as_str() {
        Some("request") => Error::Request(SystemOneError {
            location: err["location"].as_array().cloned().unwrap_or_default(),
            message,
            kind: err["request_kind"].as_str().unwrap_or_default().into(),
        }),
        Some("config") => Error::Config(message),
        Some("shape") => Error::Shape(message),
        _ => Error::Message(message),
    }
}

fn encode_error(e: &Error) -> Value {
    match e {
        Error::Request(r) => json!({
            "kind": "request",
            "message": r.message,
            "location": r.location,
            "request_kind": r.kind,
        }),
        Error::Config(m) => json!({"kind": "config", "message": m}),
        Error::Shape(m) => json!({"kind": "shape", "message": m}),
        other => json!({"kind": "other", "message": other.to_string()}),
    }
}

// ---- worker side ------------------------------------------------------------

/// A loaded model in the worker: method name, JSON args and payload in;
/// JSON result and payload out.
pub(crate) trait Handler: Send + Sync {
    fn info(&self) -> Value {
        Value::Null
    }
    fn call(&self, method: &str, args: Value, payload: Vec<u8>) -> Result<(Value, Vec<u8>)>;
}

#[derive(Deserialize)]
struct Init {
    kind: String,
    runtime: RuntimeOptions,
    config: Value,
    handlers: usize,
}

/// Call first in the host's `main`. In a worker process started by
/// [`Isolated::load`] this serves the model and exits the process; anywhere
/// else it returns at once.
pub fn main_if_worker() {
    if std::env::args_os().nth(1).as_deref() != Some(OsStr::new(MARKER)) {
        return;
    }
    let code = match serve() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("sinteric-inference worker: {e}");
            1
        }
    };
    std::process::exit(code);
}

fn serve() -> Result<()> {
    // SAFETY: the parent dup'ed its end of the socket pair to this fd and no
    // other code in this process owns it.
    let stream = unsafe { UnixStream::from_raw_fd(WORKER_FD) };
    // Keep it out of anything this process starts.
    // SAFETY: plain fcntl on an fd this process owns.
    unsafe { libc::fcntl(WORKER_FD, libc::F_SETFD, libc::FD_CLOEXEC) };
    let mut reader = stream.try_clone()?;
    let writer = Arc::new(Mutex::new(stream));
    let Some((_, header, _)) = read_frame(&mut reader)? else {
        return Ok(());
    };
    let loaded = serde_json::from_slice::<Init>(&header)
        .map_err(Error::from)
        .and_then(|init| {
            let rt = Runtime::new(init.runtime)?;
            Ok((load(&init.kind, &rt, init.config)?, init.handlers.max(1)))
        });
    let (handler, handlers) = match loaded {
        Ok(loaded) => loaded,
        Err(e) => {
            let reply = serde_json::to_vec(&json!({"err": encode_error(&e)}))?;
            let _ = write_frame(&lock(&writer), 0, &reply, &[]);
            return Err(e);
        }
    };
    let reply = serde_json::to_vec(&json!({"ok": handler.info()}))?;
    write_frame(&lock(&writer), 0, &reply, &[])?;

    let handler: Arc<dyn Handler> = Arc::from(handler);
    let (tx, rx) = mpsc::channel::<(u64, Vec<u8>, Vec<u8>)>();
    let rx = Arc::new(Mutex::new(rx));
    for _ in 0..handlers {
        let (rx, writer, handler) = (rx.clone(), writer.clone(), handler.clone());
        std::thread::spawn(move || loop {
            let job = rx.lock().ok().and_then(|rx| rx.recv().ok());
            let Some((id, header, payload)) = job else {
                break;
            };
            // A panic must still answer the call, or the caller waits forever.
            let reply = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run(&*handler, &header, payload)
            }))
            .unwrap_or_else(|_| Err(Error::Message("worker call panicked".into())));
            let (header, payload) = match reply {
                Ok((value, payload)) => (json!({"ok": value}), payload),
                Err(e) => (json!({"err": encode_error(&e)}), Vec::new()),
            };
            let header = serde_json::to_vec(&header).unwrap_or_default();
            if write_frame(&lock(&writer), id, &header, &payload).is_err() {
                // The parent is gone.
                std::process::exit(0);
            }
        });
    }
    // EOF from the parent (dropped, or the parent died) ends the worker.
    while let Some(frame) = read_frame(&mut reader)? {
        if tx.send(frame).is_err() {
            break;
        }
    }
    Ok(())
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn run(handler: &dyn Handler, header: &[u8], payload: Vec<u8>) -> Result<(Value, Vec<u8>)> {
    #[derive(Deserialize)]
    struct Call {
        method: String,
        #[serde(default)]
        args: Value,
    }
    let call: Call = serde_json::from_slice(header)?;
    handler.call(&call.method, call.args, payload)
}

fn load(kind: &str, rt: &Runtime, config: Value) -> Result<Box<dyn Handler>> {
    match kind {
        #[cfg(feature = "embeddinggemma2")]
        "embeddinggemma2" => Ok(Box::new(
            rt.load::<crate::embeddinggemma2::EmbeddingGemma2>(serde_json::from_value(config)?)?,
        )),
        #[cfg(feature = "vela2")]
        "vela2" => Ok(Box::new(
            rt.load::<crate::vela2::Vela2>(serde_json::from_value(config)?)?,
        )),
        #[cfg(feature = "ppocr")]
        "ppocr-det" => Ok(Box::new(
            rt.load::<crate::ppocr::Detector>(serde_json::from_value(config)?)?,
        )),
        #[cfg(feature = "ppocr")]
        "ppocr-rec" => Ok(Box::new(
            rt.load::<crate::ppocr::Recognizer>(serde_json::from_value(config)?)?,
        )),
        other => Err(Error::Config(format!(
            "this binary cannot run a `{other}` worker (feature off)"
        ))),
    }
}

fn unknown(method: &str) -> Error {
    Error::Config(format!("unknown worker method `{method}`"))
}

// ---- per model ----------------------------------------------------------------

#[cfg(feature = "embeddinggemma2")]
mod eg2 {
    use super::*;
    use crate::embeddinggemma2::{Config, EmbeddingGemma2, Embeddings, PromptKind};

    impl Remote for EmbeddingGemma2 {
        const KIND: &'static str = "embeddinggemma2";
        fn config_json(config: &Config) -> Result<Value> {
            Ok(serde_json::to_value(config)?)
        }
    }

    fn floats(e: Embeddings) -> (Value, Vec<u8>) {
        let dim = e.dim;
        let bytes = e
            .into_inner()
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        (json!({"dim": dim}), bytes)
    }

    impl Handler for EmbeddingGemma2 {
        fn info(&self) -> Value {
            json!({"embedding_dim": self.embedding_dim()})
        }
        fn call(&self, method: &str, args: Value, _: Vec<u8>) -> Result<(Value, Vec<u8>)> {
            match method {
                "embed" => {
                    let texts: Vec<String> = serde_json::from_value(args["texts"].clone())?;
                    let kind: PromptKind = serde_json::from_value(args["kind"].clone())?;
                    Ok(floats(self.embed(&texts, kind)?))
                }
                "embed_ids" => {
                    let ids = serde_json::from_value(args["input_ids"].clone())?;
                    let mask = serde_json::from_value(args["attention_mask"].clone())?;
                    Ok(floats(self.embed_ids(ids, mask)?))
                }
                "backend_stats" => Ok((self.backend_stats(), Vec::new())),
                other => Err(unknown(other)),
            }
        }
    }

    fn embeddings((header, payload): (Value, Vec<u8>)) -> Result<Embeddings> {
        let dim = header["dim"]
            .as_u64()
            .ok_or_else(|| Error::Worker("embedding reply without dim".into()))?;
        rt_core::check_shape!(payload.len() % 4 == 0, "embedding reply length");
        let data = payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        Embeddings::new(dim as usize, data)
    }

    impl Isolated<EmbeddingGemma2> {
        pub fn embed(
            &self,
            texts: impl IntoIterator<Item = impl AsRef<str>>,
            kind: PromptKind,
        ) -> Result<Embeddings> {
            let texts: Vec<String> = texts.into_iter().map(|t| t.as_ref().to_owned()).collect();
            embeddings(self.call("embed", json!({"texts": texts, "kind": kind}), Vec::new())?)
        }

        pub fn embed_ids(
            &self,
            input_ids: Vec<Vec<u32>>,
            attention_mask: Option<Vec<Vec<u32>>>,
        ) -> Result<Embeddings> {
            embeddings(self.call(
                "embed_ids",
                json!({"input_ids": input_ids, "attention_mask": attention_mask}),
                Vec::new(),
            )?)
        }

        pub fn embedding_dim(&self) -> usize {
            self.info["embedding_dim"].as_u64().unwrap_or_default() as usize
        }
    }
}

#[cfg(feature = "vela2")]
mod vela {
    use super::*;
    use crate::vela2::{Config, Vela2};

    impl Remote for Vela2 {
        const KIND: &'static str = "vela2";
        fn config_json(config: &Config) -> Result<Value> {
            Ok(serde_json::to_value(config)?)
        }
    }

    impl Handler for Vela2 {
        fn call(&self, method: &str, args: Value, _: Vec<u8>) -> Result<(Value, Vec<u8>)> {
            match method {
                "system_one" => Ok((self.system_one(&args)?, Vec::new())),
                "predict" => Ok((self.predict(&args)?, Vec::new())),
                "backend_stats" => Ok((self.backend_stats(), Vec::new())),
                other => Err(unknown(other)),
            }
        }
    }

    impl Isolated<Vela2> {
        /// SystemOne request body → response.
        pub fn system_one(&self, body: &Value) -> Result<Value> {
            Ok(self.call("system_one", body.clone(), Vec::new())?.0)
        }

        /// Generic prediction request → response.
        pub fn predict(&self, request: &Value) -> Result<Value> {
            Ok(self.call("predict", request.clone(), Vec::new())?.0)
        }
    }
}

#[cfg(feature = "ppocr")]
mod ocr {
    use super::*;
    use crate::ppocr::{
        DecodedLine, DetConfig, Detector, ImageView, PixelFormat, Quad, RecConfig, Recognizer,
    };

    impl Remote for Detector {
        const KIND: &'static str = "ppocr-det";
        fn config_json(config: &DetConfig) -> Result<Value> {
            Ok(serde_json::to_value(config)?)
        }
    }

    impl Remote for Recognizer {
        const KIND: &'static str = "ppocr-rec";
        fn config_json(config: &RecConfig) -> Result<Value> {
            Ok(serde_json::to_value(config)?)
        }
    }

    fn format_name(format: PixelFormat) -> &'static str {
        match format {
            PixelFormat::Bgr8 => "bgr8",
            PixelFormat::Rgb8 => "rgb8",
            PixelFormat::Gray8 => "gray8",
        }
    }

    fn parse_format(name: &str) -> Result<PixelFormat> {
        Ok(match name {
            "bgr8" => PixelFormat::Bgr8,
            "rgb8" => PixelFormat::Rgb8,
            "gray8" => PixelFormat::Gray8,
            other => return Err(Error::Shape(format!("unknown pixel format `{other}`"))),
        })
    }

    /// The image's geometry as call args, and only the strided extent the
    /// view covers as payload.
    fn image_args(image: &ImageView<'_>) -> (Value, Vec<u8>) {
        let data = image.data();
        let row = image.width() as usize * image.format().channels();
        let extent = (image.height() as usize - 1) * image.stride() + row;
        let args = json!({
            "width": image.width(),
            "height": image.height(),
            "stride": image.stride(),
            "format": format_name(image.format()),
        });
        (args, data[..extent].to_vec())
    }

    fn image<'a>(args: &Value, payload: &'a [u8]) -> Result<ImageView<'a>> {
        let dim = |key: &str| {
            args[key]
                .as_u64()
                .ok_or_else(|| Error::Shape(format!("image {key} missing")))
        };
        ImageView::new(
            payload,
            u32::try_from(dim("width")?).map_err(|_| Error::Shape("image width".into()))?,
            u32::try_from(dim("height")?).map_err(|_| Error::Shape("image height".into()))?,
            dim("stride")? as usize,
            parse_format(args["format"].as_str().unwrap_or_default())?,
        )
    }

    impl Handler for Detector {
        fn call(&self, method: &str, args: Value, payload: Vec<u8>) -> Result<(Value, Vec<u8>)> {
            match method {
                "detect" => Ok((
                    serde_json::to_value(self.detect(&image(&args, &payload)?)?)?,
                    Vec::new(),
                )),
                "backend_stats" => Ok((self.backend_stats(), Vec::new())),
                other => Err(unknown(other)),
            }
        }
    }

    impl Handler for Recognizer {
        fn call(&self, method: &str, args: Value, payload: Vec<u8>) -> Result<(Value, Vec<u8>)> {
            match method {
                "recognize" => {
                    let quads: Vec<Quad> = serde_json::from_value(args["quads"].clone())?;
                    Ok((
                        serde_json::to_value(self.recognize(&image(&args, &payload)?, &quads)?)?,
                        Vec::new(),
                    ))
                }
                "backend_stats" => Ok((self.backend_stats(), Vec::new())),
                other => Err(unknown(other)),
            }
        }
    }

    impl Isolated<Detector> {
        /// Text-line quads on `image`, in reading order.
        pub fn detect(&self, image: &ImageView<'_>) -> Result<Vec<Quad>> {
            let (args, payload) = image_args(image);
            let (quads, _) = self.call("detect", args, payload)?;
            Ok(serde_json::from_value(quads)?)
        }
    }

    impl Isolated<Recognizer> {
        /// Read the text in each of `quads` on `image`; one line per quad,
        /// in `quads` order.
        pub fn recognize(&self, image: &ImageView<'_>, quads: &[Quad]) -> Result<Vec<DecodedLine>> {
            let (mut args, payload) = image_args(image);
            args["quads"] = serde_json::to_value(quads)?;
            let (lines, _) = self.call("recognize", args, payload)?;
            Ok(serde_json::from_value(lines)?)
        }
    }
}
