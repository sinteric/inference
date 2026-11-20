//! Latency and RSS of `embed_parts` on the multimodal fixture cases.
//!
//! ```text
//! cargo run --release -p sinteric-inference --example eg2_media_bench -- \
//!     --model /models/embeddinggemma2 --threads 4 --precision bf16 \
//!     --modalities image,audio --repeat 20
//! ```
//!
//! Prints one JSON object: load time, RSS after load, p50/p95 per case and
//! the process peak RSS. Run one process per configuration so peaks do not
//! mix.

#[path = "../src/test_support/media.rs"]
mod media;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use sinteric_inference::cpu::{Options, Runtime};
use sinteric_inference::embeddinggemma2::cpu::{Config, EmbeddingGemma2, Precision};
use sinteric_inference::embeddinggemma2::{ImageView, Part, PixelFormat, PromptKind};
use std::time::Instant;

fn status_kib(key: &str) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let l = s.lines().find(|l| l.starts_with(key))?;
    l.split_whitespace().nth(1)?.parse().ok()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

enum Owned {
    Text(String),
    Image(usize, usize, usize, Vec<u8>),
    Audio(Vec<f32>),
}

struct Args {
    model: String,
    fixture: String,
    threads: usize,
    precision: Option<Precision>,
    image: bool,
    audio: bool,
    cases: Option<String>,
    warmup: usize,
    repeat: usize,
}

fn parse_args() -> Result<Args> {
    let mut model = None;
    let mut fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/embeddinggemma2/multimodal.json"
    )
    .to_string();
    let mut threads = 4usize;
    let mut precision = None;
    let mut modalities = "image,audio".to_string();
    let mut cases = None::<String>;
    let (mut warmup, mut repeat) = (2usize, 10usize);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--model" => model = Some(val()?),
            "--fixture" => fixture = val()?,
            "--threads" => threads = val()?.parse()?,
            "--precision" => {
                precision = Some(val()?.parse::<Precision>()?);
            }
            "--modalities" => modalities = val()?,
            "--cases" => cases = Some(val()?),
            "--warmup" => warmup = val()?.parse()?,
            "--repeat" => repeat = val()?.parse()?,
            _ => bail!("unknown argument {a}"),
        }
    }
    Ok(Args {
        model: model.context("--model is required")?,
        fixture,
        threads,
        precision,
        image: modalities.split(',').any(|m| m == "image"),
        audio: modalities.split(',').any(|m| m == "audio"),
        cases,
        warmup,
        repeat,
    })
}

/// A fixture case's parts, with synthetic pixels and samples.
fn owned_parts(doc: &Value, case: &Value) -> Result<Vec<Owned>> {
    let mut owned = Vec::new();
    for p in case["parts"].as_array().context("parts")? {
        let (kind, name) = (
            p[0].as_str().unwrap_or_default(),
            p[1].as_str().unwrap_or_default(),
        );
        owned.push(match kind {
            "text" => Owned::Text(name.to_string()),
            "image" => {
                let d: Vec<usize> = doc["images"][name]
                    .as_array()
                    .context("image dims")?
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as usize)
                    .collect();
                Owned::Image(d[0], d[1], d[2], media::image_pixels(d[0], d[1], d[2]))
            }
            _ => Owned::Audio(media::audio_samples(
                doc["audio"][name].as_u64().context("audio length")? as usize,
            )),
        });
    }
    Ok(owned)
}

fn parts(owned: &[Owned]) -> Result<Vec<Part<'_>>> {
    let mut parts = Vec::with_capacity(owned.len());
    for o in owned {
        parts.push(match o {
            Owned::Text(s) => Part::Text(s),
            Owned::Image(h, w, c, px) => {
                let format = if *c == 1 {
                    PixelFormat::Gray8
                } else {
                    PixelFormat::Rgb8
                };
                Part::Image(ImageView::try_new(px, *w as u32, *h as u32, w * c, format)?)
            }
            Owned::Audio(s) => Part::Audio(s),
        });
    }
    Ok(parts)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&args.fixture)?)?;

    let rss_start = status_kib("VmRSS:");
    let rt = Runtime::try_new(Options::new().with_threads(args.threads))?;
    let t = Instant::now();
    let mut config = Config::new(&args.model)
        .with_image(args.image)
        .with_audio(args.audio);
    if let Some(precision) = args.precision {
        config = config.with_precision(precision);
    }
    let eg2 = EmbeddingGemma2::load(&rt, config)?;
    let load_s = t.elapsed().as_secs_f64();
    let rss_loaded = status_kib("VmRSS:");

    let mut out = Vec::new();
    for case in doc["cases"].as_array().context("cases")? {
        let id = case["id"].as_str().unwrap_or_default();
        if let Some(c) = &args.cases {
            if !c.split(',').any(|c| c == id) {
                continue;
            }
        }
        let owned = owned_parts(&doc, case)?;
        let needs_image = owned.iter().any(|o| matches!(o, Owned::Image(..)));
        let needs_audio = owned.iter().any(|o| matches!(o, Owned::Audio(_)));
        if (needs_image && !args.image) || (needs_audio && !args.audio) {
            continue;
        }
        let parts = parts(&owned)?;
        let prompt = match case["prompt"].as_str() {
            Some("query") => Some(PromptKind::Query),
            Some(_) => Some(PromptKind::Document),
            None => None,
        };
        for _ in 0..args.warmup {
            eg2.embed_parts(&[&parts], prompt)?;
        }
        let mut ms = Vec::with_capacity(args.repeat);
        for _ in 0..args.repeat {
            let t = Instant::now();
            eg2.embed_parts(&[&parts], prompt)?;
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(f64::total_cmp);
        out.push(json!({
            "id": id,
            "tokens": eg2.parts_input_ids(&parts, prompt)?.len(),
            "p50_ms": percentile(&ms, 0.5),
            "p95_ms": percentile(&ms, 0.95),
        }));
    }
    println!(
        "{}",
        json!({
            "threads": args.threads,
            "precision": eg2.precision(),
            "image": args.image,
            "audio": args.audio,
            "load_s": load_s,
            "rss_start_kib": rss_start,
            "rss_loaded_kib": rss_loaded,
            "rss_end_kib": status_kib("VmRSS:"),
            "peak_rss_kib": status_kib("VmHWM:"),
            "cases": out,
        })
    );
    Ok(())
}
