//! `golden record`: load one model family, run every fixed input, and write
//! one JSON line per case.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use sinteric_inference::cpu::{BackendOverrides, IsaCap, Options, Runtime};
use sinteric_inference::{embeddinggemma2 as eg2, guard, ppocr};

use crate::inputs;

/// Which models one process records. One family per process, and each
/// precision is its own family, so the peak RSS a record reports is that
/// family's alone.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Eg2(eg2::cpu::Precision),
    Guard(guard::cpu::Precision),
    Ocr,
}

pub struct Args {
    family: Family,
    dir: PathBuf,
    pages: Option<PathBuf>,
    threads: usize,
    isa: IsaCap,
    reps: usize,
    out: PathBuf,
}

impl Args {
    pub fn parse(args: &[String]) -> Result<Self> {
        let family = match crate::flag(args, "--model").as_deref() {
            Some("eg2-bf16") => Family::Eg2(eg2::cpu::Precision::Bf16),
            Some("eg2-fp32") => Family::Eg2(eg2::cpu::Precision::Fp32),
            Some("guard-bf16") => Family::Guard(guard::cpu::Precision::Bf16),
            Some("guard-fp32") => Family::Guard(guard::cpu::Precision::Fp32),
            Some("ocr") => Family::Ocr,
            _ => bail!("--model eg2-bf16|eg2-fp32|guard-bf16|guard-fp32|ocr is required"),
        };
        let dir = crate::flag_path(args, "--dir")?;
        let pages = match family {
            Family::Ocr => Some(crate::flag_path(args, "--pages")?),
            _ => None,
        };
        let threads = crate::flag(args, "--threads").map_or(Ok(4), |t| t.parse())?;
        let isa = match crate::flag(args, "--isa").as_deref().unwrap_or("best") {
            "best" => IsaCap::Best,
            "avx512_core_bf16" => IsaCap::Avx512CoreBf16,
            "avx2" => IsaCap::Avx2,
            other => bail!("--isa {other}: best, avx512_core_bf16 or avx2"),
        };
        let reps = crate::flag(args, "--reps").map_or(Ok(3), |t| t.parse())?;
        let out = crate::flag_path(args, "--out")?;
        Ok(Self {
            family,
            dir,
            pages,
            threads,
            isa,
            reps,
            out,
        })
    }
}

/// One recorded case. `compare` keys cases by `(model, case, isa, threads)`.
#[derive(Serialize, Deserialize)]
pub struct Record {
    pub model: String,
    pub case: String,
    pub isa: String,
    pub threads: usize,
    /// SHA-256 of the output's bytes (floats little-endian, text UTF-8).
    pub sha256: String,
    /// The output itself, for the size of a difference.
    pub output: Value,
    /// Whether every repeat gave the same bytes as the first call.
    pub repeatable: bool,
    /// The first call, which builds this input shape's plans.
    pub cold_ms: f64,
    /// The median of the repeats after it.
    pub warm_ms: f64,
}

/// What a process measured once: load time and peak RSS.
#[derive(Serialize, Deserialize)]
pub struct Meta {
    pub meta: String,
    pub isa: String,
    pub threads: usize,
    pub load_ms: f64,
    pub peak_rss_kib: u64,
}

struct Recorder {
    out: File,
    isa: String,
    threads: usize,
    reps: usize,
}

impl Recorder {
    /// Run `call` once cold and `reps` times warm; record its output.
    fn case<T>(
        &mut self,
        model: &str,
        case: &str,
        mut call: impl FnMut() -> Result<T>,
        bytes: impl Fn(&T) -> (Vec<u8>, Value),
    ) -> Result<T> {
        let t = Instant::now();
        let first = call().with_context(|| format!("{model} {case}"))?;
        let cold_ms = ms(t);
        let (raw, output) = bytes(&first);
        let mut warm = Vec::with_capacity(self.reps);
        let mut repeatable = true;
        for _ in 0..self.reps {
            let t = Instant::now();
            let again = call()?;
            warm.push(ms(t));
            repeatable &= bytes(&again).0 == raw;
        }
        warm.sort_by(f64::total_cmp);
        let record = Record {
            model: model.into(),
            case: case.into(),
            isa: self.isa.clone(),
            threads: self.threads,
            sha256: hex::encode(Sha256::digest(&raw)),
            output,
            repeatable,
            cold_ms,
            warm_ms: warm.get(warm.len() / 2).copied().unwrap_or(cold_ms),
        };
        writeln!(self.out, "{}", serde_json::to_string(&record)?)?;
        eprintln!("{model:>12} {case:<24} {:.1} ms", record.warm_ms);
        Ok(first)
    }

    fn meta(&mut self, family: &str, load_ms: f64) -> Result<()> {
        let meta = Meta {
            meta: family.into(),
            isa: self.isa.clone(),
            threads: self.threads,
            load_ms,
            peak_rss_kib: peak_rss_kib()?,
        };
        writeln!(self.out, "{}", serde_json::to_string(&meta)?)?;
        Ok(())
    }
}

pub fn run(args: Args) -> Result<()> {
    let rt = Runtime::try_new(
        Options::new()
            .with_threads(args.threads)
            .with_isa_cap(args.isa),
    )?;
    let mut rec = Recorder {
        out: File::create(&args.out).with_context(|| format!("create {}", args.out.display()))?,
        isa: rt.isa().name().to_string(),
        threads: args.threads,
        reps: args.reps,
    };
    match args.family {
        Family::Eg2(precision) => embeddinggemma2(&rt, &args, &mut rec, precision)?,
        Family::Guard(precision) => guard(&rt, &args, &mut rec, precision)?,
        Family::Ocr => ocr(&rt, &args, &mut rec)?,
    }
    Ok(())
}

fn embeddinggemma2(
    rt: &Runtime,
    args: &Args,
    rec: &mut Recorder,
    precision: eg2::cpu::Precision,
) -> Result<()> {
    if !eg2::cpu::EmbeddingGemma2::precisions(rt).contains(&precision) {
        eprintln!(
            "skip EmbeddingGemma 2 {precision}: no {precision} kernels on {}",
            rt.isa()
        );
        return Ok(());
    }
    let mut config = eg2::cpu::Config::new(&args.dir)
        .with_precision(precision)
        .with_image(true)
        .with_audio(true);
    config.cache.enabled = false;
    let t = Instant::now();
    let model = eg2::cpu::EmbeddingGemma2::load(rt, config)?;
    let load_ms = ms(t);
    let name = format!("eg2-{precision}");
    for case in inputs::text_cases() {
        let kind = if case.query {
            eg2::PromptKind::Query
        } else {
            eg2::PromptKind::Document
        };
        rec.case(
            &name,
            &case.id,
            || Ok(model.embed(&case.texts, kind)?),
            embeddings,
        )?;
    }
    let tex = inputs::image_pixels(300, 200, 3);
    let tall = inputs::image_pixels(64, 512, 3);
    let gray = inputs::image_pixels(150, 150, 1);
    let second = inputs::audio_samples(16_000);
    let odd = inputs::audio_samples(37_123);
    let image = eg2::ImageView::try_new(&tex, 200, 300, 200 * 3, eg2::PixelFormat::Rgb8)?;
    let wide = eg2::ImageView::try_new(&tall, 512, 64, 512 * 3, eg2::PixelFormat::Rgb8)?;
    let mono = eg2::ImageView::try_new(&gray, 150, 150, 150, eg2::PixelFormat::Gray8)?;
    let media: [(&str, Vec<Vec<eg2::Part<'_>>>); 5] = [
        ("img", vec![vec![eg2::Part::Image(image)]]),
        (
            "img-text",
            vec![vec![eg2::Part::Text("a photo of "), eg2::Part::Image(wide)]],
        ),
        ("gray", vec![vec![eg2::Part::Image(mono)]]),
        ("audio-odd", vec![vec![eg2::Part::Audio(&odd)]]),
        (
            "batch-mixed",
            vec![
                vec![eg2::Part::Text("hello "), eg2::Part::Image(image)],
                vec![eg2::Part::Audio(&second)],
            ],
        ),
    ];
    for (id, parts) in &media {
        let inputs: Vec<&[eg2::Part<'_>]> = parts.iter().map(Vec::as_slice).collect();
        rec.case(
            &name,
            id,
            || Ok(model.embed_parts(&inputs, None)?),
            embeddings,
        )?;
    }
    rec.meta(&name, load_ms)?;
    Ok(())
}

fn embeddings(e: &eg2::Embeddings) -> (Vec<u8>, Value) {
    let raw = e.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    (raw, json!({"dim": e.dim, "data": e.data}))
}

fn guard(
    rt: &Runtime,
    args: &Args,
    rec: &mut Recorder,
    precision: guard::cpu::Precision,
) -> Result<()> {
    if !guard::cpu::Guard::precisions(rt).contains(&precision) {
        eprintln!(
            "skip guard {precision}: no {precision} kernels on {}",
            rt.isa()
        );
        return Ok(());
    }
    let mut config = guard::cpu::Config::new(&args.dir).with_precision(precision);
    config.cache.enabled = false;
    let t = Instant::now();
    let model = guard::cpu::Guard::load(rt, config)?;
    let load_ms = ms(t);
    let name = format!("guard-{precision}");
    for case in inputs::text_cases() {
        rec.case(
            &name,
            &case.id,
            || Ok(model.classify(&case.texts)?),
            injections,
        )?;
    }
    rec.meta(&name, load_ms)?;
    Ok(())
}

fn injections(v: &Vec<guard::Injection>) -> (Vec<u8>, Value) {
    let mut raw = Vec::new();
    let mut out = Vec::new();
    for i in v {
        raw.extend(i.logit.to_le_bytes());
        raw.extend(i.score.to_le_bytes());
        raw.extend((i.tokens as u64).to_le_bytes());
        raw.push(u8::from(i.truncated));
        out.push(json!([i.logit, i.score, i.tokens, i.truncated]));
    }
    (raw, Value::Array(out))
}

fn ocr(rt: &Runtime, args: &Args, rec: &mut Recorder) -> Result<()> {
    let pages = inputs::pages(args.pages.as_deref().context("--pages")?)?;
    let shared = rt.shared_backend(
        BackendOverrides::default()
            .with_workspace_limit_bytes(Some(ppocr::cpu::DEFAULT_WORKSPACE_LIMIT_BYTES)),
    )?;
    let t = Instant::now();
    let det =
        ppocr::cpu::Detector::load(&shared, ppocr::cpu::DetConfig::new(args.dir.join("det")))?;
    let recognizer =
        ppocr::cpu::Recognizer::load(&shared, ppocr::cpu::RecConfig::new(args.dir.join("rec")))?;
    let unwarper = ppocr::cpu::Unwarper::load(
        &shared,
        ppocr::cpu::UvdocConfig::new(args.dir.join("uvdoc")),
    )?;
    let load_ms = ms(t);
    for page in &pages {
        let format = match page.channels {
            1 => ppocr::PixelFormat::Gray8,
            _ => ppocr::PixelFormat::Rgb8,
        };
        let stride = page.width as usize * page.channels;
        let view = ppocr::ImageView::try_new(&page.data, page.width, page.height, stride, format)?;
        let quads = rec.case(
            "ppocr-det",
            &page.name,
            || Ok(det.detect(&view)?),
            |q: &Vec<ppocr::Quad>| {
                let raw = q
                    .iter()
                    .flatten()
                    .flatten()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                (raw, json!(q))
            },
        )?;
        rec.case(
            "ppocr-rec",
            &page.name,
            || Ok(recognizer.recognize(&view, &quads)?),
            |lines: &Vec<ppocr::DecodedLine>| {
                let mut raw = Vec::new();
                for line in lines {
                    raw.extend_from_slice(line.text.as_bytes());
                    raw.push(0);
                    raw.extend_from_slice(&line.score.to_le_bytes());
                }
                let out: Vec<Value> = lines.iter().map(|l| json!([l.text, l.score])).collect();
                (raw, Value::Array(out))
            },
        )?;
        if page.name.starts_with("photo") || page.name.ends_with("-1") {
            rec.case(
                "ppocr-uvdoc",
                &page.name,
                || Ok(unwarper.unwarp(&view)?),
                |u: &ppocr::UnwarpedImage| {
                    let sum: u64 = u.data.iter().map(|&b| u64::from(b)).sum();
                    (
                        u.data.clone(),
                        json!({"width": u.width, "height": u.height, "byte_sum": sum}),
                    )
                },
            )?;
        }
    }
    rec.meta("ppocr", load_ms)
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// `VmHWM` of this process, in KiB.
fn peak_rss_kib() -> Result<u64> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
        .context("VmHWM in /proc/self/status")
}
