//! ORT reference for the 100-page OCR gate. Same CLI/JSONL contract as `ppocr-onednn`;
//! same preprocessing, DB post-processing, crops, Exact batching and CTC decode
//! as `ppocr-onednn` (`Detector::detect`, `Recognizer::recognize`, score >= 0.5),
//! with ONNX Runtime executing the two networks.
//! `--model DIR` holds the imported assets (`det/`, `rec/`, for the dictionary)
//! and the pinned exports at `DIR/onnx/det.onnx` and `DIR/onnx/rec.onnx`.
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use ppocr::det::{self, DetOptions};
use ppocr::net::Tensor;
use ppocr::rec::{self, RecOptions};
use ppocr::weights::{AssetConfig, Assets, ModelKind};
use rt_core::{Error, Result};
use rt_image::{ImageSize, ImageView, PixelFormat, Quad};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    model: PathBuf,
    threads: usize,
    warmup: usize,
    images: PathBuf,
    out: PathBuf,
}
fn error(message: impl std::fmt::Display) -> Error {
    Error::Config(message.to_string())
}
fn ort_err(e: ort::Error) -> Error {
    Error::Config(format!("onnxruntime: {e}"))
}
impl Args {
    fn parse() -> Result<Self> {
        let mut values = HashMap::new();
        let mut args = std::env::args_os().skip(1);
        while let Some(key) = args.next() {
            if key == "--help" || key == "-h" {
                println!("ppocr-ort --model DIR --threads N [--warmup K] --images INPUT.jsonl --out OUTPUT.jsonl");
                std::process::exit(0);
            }
            let key = key
                .into_string()
                .map_err(|_| error("invalid argument name"))?;
            if !["--model", "--threads", "--warmup", "--images", "--out"].contains(&key.as_str()) {
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
        if threads == 0 {
            return Err(error("--threads must be >= 1"));
        }
        Ok(Self {
            model,
            threads,
            warmup,
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
    let (mut ids, mut needs_newline) = (HashSet::new(), false);
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
        ids.insert(
            value
                .get("img_id")
                .ok_or_else(|| error("existing output lacks img_id"))?
                .to_string(),
        );
        needs_newline = !line.ends_with('\n');
    }
    Ok((ids, needs_newline))
}
/// PP-OCR's default recognition score threshold.
const DROP_SCORE: f32 = 0.5;
struct Line {
    quad: Quad,
    text: String,
    score: f32,
}
struct OrtOcr {
    det: Session,
    rec: Session,
    det_options: DetOptions,
    rec_options: RecOptions,
    dictionary: Vec<String>,
}
fn session(path: &Path, threads: usize) -> Result<Session> {
    if !path.is_file() {
        return Err(Error::Asset(format!(
            "missing ONNX export {}",
            path.display()
        )));
    }
    Session::builder()
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| ort_err(e.into()))?
        .with_intra_threads(threads)
        .map_err(|e| ort_err(e.into()))?
        .with_inter_threads(1)
        .map_err(|e| ort_err(e.into()))?
        .commit_from_file(path)
        .map_err(ort_err)
}
fn run_net(s: &mut Session, shape: &[usize], values: Vec<f32>) -> Result<Tensor> {
    let input = OrtTensor::from_array((shape.to_vec(), values)).map_err(ort_err)?;
    let outputs = s.run(ort::inputs![input]).map_err(ort_err)?;
    let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
    let shape = shape
        .iter()
        .map(|&d| usize::try_from(d).map_err(|_| error("negative ORT dim")))
        .collect::<Result<Vec<_>>>()?;
    Tensor::new(shape, data.to_vec())
}
impl OrtOcr {
    fn load(model: &Path, threads: usize) -> Result<Self> {
        let dictionary = Assets::load(&AssetConfig::new(model.join("rec"), ModelKind::Rec))?
            .dictionary()
            .to_vec();
        Ok(Self {
            det: session(&model.join("onnx/det.onnx"), threads)?,
            rec: session(&model.join("onnx/rec.onnx"), threads)?,
            det_options: DetOptions::default(),
            rec_options: RecOptions::default(),
            dictionary,
        })
    }
    /// Mirrors `ppocr-onednn`: detect, recognize (Exact batching), drop
    /// lines scoring below [`DROP_SCORE`].
    fn recognize(&mut self, image: &ImageView<'_>) -> Result<Vec<Line>> {
        let input = det::preprocess(image, &self.det_options)?;
        let prob = run_net(&mut self.det, &input.shape, input.values)?;
        rt_core::check_shape!(
            prob.shape.len() == 4 && prob.shape[0] == 1 && prob.shape[1] == 1,
            "ORT det output must be [1,1,H,W]"
        );
        let boxes = det::postprocess(
            &prob.values,
            prob.shape[3],
            prob.shape[2],
            ImageSize {
                width: image.width(),
                height: image.height(),
            },
            &self.det_options,
        )?;
        let ro = &self.rec_options;
        let mut crops = Vec::with_capacity(boxes.len());
        for &q in &boxes {
            crops.push(rt_image::recognizer_crop(
                image,
                q,
                ro.min_width,
                ro.max_width,
                ro.max_pixels,
            )?);
        }
        let mut order = (0..crops.len()).collect::<Vec<_>>();
        order.sort_by_key(|&i| std::cmp::Reverse(crops[i].width));
        let mut decoded: Vec<Option<Line>> = (0..boxes.len()).map(|_| None).collect();
        let mut start = 0;
        while start < order.len() {
            let width = crops[order[start]].width;
            let mut end = start + 1;
            while end < order.len()
                && end - start < ro.max_batch
                && crops[order[end]].width == width
            {
                end += 1;
            }
            let batch = end - start;
            let mut values = Vec::with_capacity(batch * 3 * 48 * width);
            for &i in &order[start..end] {
                values.extend_from_slice(&crops[i].values);
            }
            let prob = run_net(&mut self.rec, &[batch, 3, 48, width], values)?;
            let lengths = vec![width.div_ceil(4) / 2; batch];
            // The reference keeps its original single-threaded host decode.
            let lines = rec::decode(&prob, &self.dictionary, Some(&lengths), 1)?;
            for (&i, line) in order[start..end].iter().zip(lines) {
                if line.score >= DROP_SCORE {
                    decoded[i] = Some(Line {
                        quad: boxes[i],
                        text: line.text,
                        score: line.score,
                    });
                }
            }
            start = end;
        }
        Ok(decoded.into_iter().flatten().collect())
    }
}
fn run(args: Args) -> Result<()> {
    if args.images == args.out {
        return Err(error("--images and --out must differ"));
    }
    let mut ocr = OrtOcr::load(&args.model, args.threads)?;
    let max_pixels = ocr.det_options.max_pixels;
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
            let lines = ocr.recognize(&view)?;
            let ms = rounded(start.elapsed().as_secs_f64() * 1000., 1);
            let lines = lines
                .into_iter()
                .map(|l| json!({"text": l.text, "score": rounded(l.score as f64, 4), "poly": l.quad.map(|p| p.map(|v| rounded(v as f64, 1)))}))
                .collect::<Vec<_>>();
            Ok::<_, Error>(
                json!({"img_id": input.img_id, "lines": lines, "ms": ms, "rss_mb": rss_mb()?}),
            )
        })();
        if i < args.warmup {
            if let Err(e) = result {
                eprintln!("warmup {}: {e}", input.img_id);
            }
            continue;
        }
        let value =
            result.unwrap_or_else(|e| json!({"img_id": input.img_id, "error": e.to_string()}));
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
