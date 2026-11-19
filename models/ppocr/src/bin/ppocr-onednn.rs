//! 100-page OCR gate JSONL protocol. PNG/JPEG decoding (`ppocr::decode`) and process metrics belong to the drivers.
use ppocr::api::{
    DetConfig, Detector, ImageView, PixelFormat, RecConfig, Recognizer,
    DEFAULT_WORKSPACE_LIMIT_BYTES,
};
use rt_core::{Error, Result};
use rt_runtime::{BackendOverrides, CpuIsa, Runtime, RuntimeOptions};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    model: PathBuf,
    threads: usize,
    warmup: usize,
    workspace_limit_bytes: Option<usize>,
    images: PathBuf,
    out: PathBuf,
}
impl Args {
    fn parse() -> Result<Self> {
        let mut values = std::collections::HashMap::new();
        let mut args = std::env::args_os().skip(1);
        while let Some(key) = args.next() {
            if key == "--help" || key == "-h" {
                println!("ppocr-onednn --model DIR --threads N [--warmup K] [--workspace-limit-bytes BYTES] --images INPUT.jsonl --out OUTPUT.jsonl");
                std::process::exit(0);
            }
            let key = key
                .into_string()
                .map_err(|_| error("invalid argument name"))?;
            if ![
                "--model",
                "--threads",
                "--warmup",
                "--workspace-limit-bytes",
                "--images",
                "--out",
            ]
            .contains(&key.as_str())
            {
                return Err(error(format!("unknown argument {key}")));
            }
            let value = args
                .next()
                .ok_or_else(|| error(format!("missing value for {key}")))?;
            if values.insert(key.clone(), value).is_some() {
                return Err(error(format!("duplicate argument {key}")));
            }
        }
        let mut required = |key: &str| {
            values
                .remove(key)
                .ok_or_else(|| error(format!("required argument {key}")))
        };
        let model = PathBuf::from(required("--model")?);
        let threads = required("--threads")?
            .to_str()
            .ok_or_else(|| error("invalid threads"))?
            .parse::<usize>()
            .map_err(error)?;
        let images = PathBuf::from(required("--images")?);
        let out = PathBuf::from(required("--out")?);
        let warmup = values
            .remove("--warmup")
            .map(|v| {
                v.to_str()
                    .ok_or_else(|| error("invalid warmup"))?
                    .parse::<usize>()
                    .map_err(error)
            })
            .transpose()?
            .unwrap_or(0);
        let workspace_limit_bytes = values
            .remove("--workspace-limit-bytes")
            .map(|v| {
                v.to_str()
                    .ok_or_else(|| error("invalid workspace limit"))?
                    .parse::<usize>()
                    .map_err(error)
            })
            .transpose()?;
        Ok(Self {
            model,
            threads,
            warmup,
            workspace_limit_bytes,
            images,
            out,
        })
    }
}
#[derive(Deserialize)]
struct Input {
    img_id: Value,
    path: PathBuf,
}
fn error(message: impl std::fmt::Display) -> Error {
    Error::Config(message.to_string())
}
fn rounded(value: f64, digits: i32) -> f64 {
    let scale = 10f64.powi(digits);
    (value * scale).round() / scale
}
fn rss_mb() -> Result<f64> {
    let s = std::fs::read_to_string("/proc/self/status")?;
    let kb = s
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|s| s.split_whitespace().next())
        .ok_or_else(|| error("VmHWM absent from /proc/self/status"))?
        .parse::<f64>()
        .map_err(error)?;
    Ok(rounded(kb / 1024., 1))
}
fn resume(path: &Path) -> Result<(HashSet<String>, bool)> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((HashSet::new(), false)),
        Err(e) => return Err(Error::from(e).at(path)),
    };
    let mut ids = HashSet::new();
    let mut needs_newline = false;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line).map_err(|e| Error::from(e).at(path))?;
        let id = value
            .get("img_id")
            .ok_or_else(|| error("existing output lacks img_id"))?;
        ids.insert(id.to_string());
        needs_newline = !line.ends_with('\n');
    }
    Ok((ids, needs_newline))
}
/// PP-OCR's default recognition score threshold, as the ORT reference uses.
const DROP_SCORE: f32 = 0.5;
fn ms_since(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.
}
fn run(args: Args) -> Result<()> {
    if args.images == args.out {
        return Err(error("--images and --out must differ"));
    }
    let env = rt_core::env::EnvOverrides::from_env()?;
    let rt = Runtime::new(
        RuntimeOptions::new()
            .with_threads(args.threads)
            .with_cpu_isa(CpuIsa::Avx512CoreBf16)
            .with_overrides(&env),
    )?;
    // Detector and recognizer on one backend: one convolution arena sized to
    // the larger model.
    let shared = rt.shared_backend(
        BackendOverrides::default().with_workspace_limit_bytes(Some(
            args.workspace_limit_bytes
                .unwrap_or(DEFAULT_WORKSPACE_LIMIT_BYTES),
        )),
    )?;
    let det_config = DetConfig::new(args.model.join("det"));
    let max_pixels = det_config.options.max_pixels;
    let det: Detector = shared.load(det_config)?;
    let rec: Recognizer = shared.load(RecConfig::new(args.model.join("rec")))?;
    let (mut done, needs_newline) = resume(&args.out)?;
    let mut out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.out)
        .map_err(|e| Error::from(e).at(&args.out))?;
    if needs_newline {
        writeln!(out)?;
    }
    for (i, line) in BufReader::new(File::open(&args.images)?)
        .lines()
        .filter(|l| l.as_ref().map_or(true, |s| !s.trim().is_empty()))
        .enumerate()
    {
        let input: Input = serde_json::from_str(&line?)?;
        let id = input.img_id.to_string();
        if i >= args.warmup && done.contains(&id) {
            continue;
        }
        let result = (|| {
            let (bytes, w, h) = ppocr::decode::load_bgr(&input.path, max_pixels)?;
            let view = ImageView::new(&bytes, w, h, w as usize * 3, PixelFormat::Bgr8)?;
            let start = Instant::now();
            let (quads, det_t) = det.detect_timed(&view)?;
            let detect_ms = ms_since(start);
            let (lines, rec_t) = rec.recognize_timed(&view, &quads)?;
            let total_ms = ms_since(start);
            let ms = rounded(total_ms, 1);
            let timings = json!({"detect_ms":detect_ms,"crop_ms":rec_t.crop_ms,
                "recognize_ms":rec_t.recognize_ms,"total_ms":total_ms,
                "det_preprocess_ms":det_t.preprocess_ms,"det_forward_ms":det_t.forward_ms,
                "det_postprocess_ms":det_t.postprocess_ms,"rec_pack_ms":rec_t.pack_ms,
                "rec_forward_ms":rec_t.forward_ms,"rec_batches":rec_t.batches});
            if env.verbose == Some(2) {
                let backend = |stats: Value| {
                    json!({"cached_plans":stats["cached_plans"],
                    "conv_arena_bytes":stats["conv_arena_bytes"],
                    "conv_scratch_bytes":stats["conv_scratch_bytes"],
                    "execute_scratch_bytes":stats["execute_scratch_bytes"],
                    "conv_weight_bytes":stats["conv_packed_weight_bytes"],
                    "linear_weight_bytes":stats["packed_weight_bytes"],
                    "retained_pool_bytes":stats["retained_output_buffer_bytes"]})
                };
                println!(
                    "ppocr_profile,{}",
                    json!({"img_id":input.img_id,"timings":timings,
                    "page_bgr_bytes":bytes.len(),
                    "backend":backend(det.backend_stats())})
                );
            }
            let lines=lines.into_iter().zip(quads).filter(|(l,_)|l.score>=DROP_SCORE).map(|(l,q)|json!({"text":l.text,"score":rounded(l.score as f64,4),"poly":q.map(|p|p.map(|v|rounded(v as f64,1)))})).collect::<Vec<_>>();
            Ok::<_, Error>(
                json!({"img_id":input.img_id,"lines":lines,"ms":ms,"rss_mb":rss_mb()?,"timings":timings}),
            )
        })();
        if i < args.warmup {
            if let Err(e) = result {
                eprintln!("warmup {}: {e}", input.img_id);
            }
            continue;
        }
        let value = result.unwrap_or_else(|e| json!({"img_id":input.img_id,"error":e.to_string()}));
        serde_json::to_writer(&mut out, &value)?;
        writeln!(out)?;
        out.flush()?;
        done.insert(id);
    }
    Ok(())
}
fn main() {
    if let Err(e) = Args::parse().and_then(run) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
