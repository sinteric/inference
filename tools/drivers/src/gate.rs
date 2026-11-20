//! The OCR gate's JSONL protocol, shared by `ppocr-onednn` and `ppocr-ort`:
//! read `{"img_id", "path"}` lines, decode each page, and append one result
//! line per page. A rerun resumes: pages the output already has are skipped.
//! The first `warmup` input lines run but are not written.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use sinteric_inference::ppocr::{ImageView, PixelFormat, Quad};

use crate::process::{peak_rss_kib, rounded};

/// PP-OCR's default recognition score threshold, which the reference uses.
pub const DROP_SCORE: f32 = 0.5;

/// One recognized line of a page.
#[derive(Debug)]
pub struct Line {
    pub quad: Quad,
    pub text: String,
    pub score: f32,
}

/// Where the pages come from and where results go.
#[derive(Debug)]
pub struct Pages {
    pub images: PathBuf,
    pub out: PathBuf,
    pub warmup: usize,
    /// Pages larger than this fail to decode.
    pub max_pixels: usize,
}

#[derive(Deserialize)]
struct Input {
    img_id: Value,
    path: PathBuf,
}

/// What a page run returns: its lines, and timings to report beside them.
pub type PageResult = Result<(Vec<Line>, Option<Value>)>;

/// Run `page` over every input page and write its result lines. A page that
/// fails is written as `{"img_id", "error"}` and the run goes on.
pub fn run(
    pages: &Pages,
    mut page: impl FnMut(&Value, &ImageView<'_>) -> PageResult,
) -> Result<()> {
    if pages.images == pages.out {
        bail!("--images and --out must differ");
    }
    let (mut done, needs_newline) = resume(&pages.out)?;
    let mut out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&pages.out)
        .with_context(|| pages.out.display().to_string())?;
    if needs_newline {
        writeln!(out)?;
    }
    let images = File::open(&pages.images).with_context(|| pages.images.display().to_string())?;
    for (i, line) in BufReader::new(images)
        .lines()
        .filter(|l| l.as_ref().map_or(true, |s| !s.trim().is_empty()))
        .enumerate()
    {
        let input: Input = serde_json::from_str(&line?)?;
        let id = input.img_id.to_string();
        if i >= pages.warmup && done.contains(&id) {
            continue;
        }
        let result = (|| {
            let (bytes, w, h) = crate::decode::load_bgr(&input.path, pages.max_pixels)?;
            let view = ImageView::new(&bytes, w, h, w as usize * 3, PixelFormat::Bgr8)?;
            let start = Instant::now();
            let (lines, timings) = page(&input.img_id, &view)?;
            let ms = rounded(start.elapsed().as_secs_f64() * 1000., 1);
            let lines: Vec<Value> = lines
                .into_iter()
                .filter(|l| l.score >= DROP_SCORE)
                .map(|l| {
                    json!({"text": l.text, "score": rounded(f64::from(l.score), 4),
                        "poly": l.quad.map(|p| p.map(|v| rounded(f64::from(v), 1)))})
                })
                .collect();
            let rss_mb = rounded(peak_rss_kib()? as f64 / 1024., 1);
            let mut v = json!({"img_id": input.img_id, "lines": lines, "ms": ms, "rss_mb": rss_mb});
            if let Some(t) = timings {
                v["timings"] = t;
            }
            Ok::<_, anyhow::Error>(v)
        })();
        if i < pages.warmup {
            if let Err(e) = result {
                eprintln!("warmup {}: {e:#}", input.img_id);
            }
            continue;
        }
        let value =
            result.unwrap_or_else(|e| json!({"img_id": input.img_id, "error": format!("{e:#}")}));
        serde_json::to_writer(&mut out, &value)?;
        writeln!(out)?;
        out.flush()?;
        done.insert(id);
    }
    Ok(())
}

/// The ids an existing output already has, and whether its last line lacks
/// a newline.
fn resume(path: &Path) -> Result<(HashSet<String>, bool)> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((HashSet::new(), false)),
        Err(e) => return Err(e).with_context(|| path.display().to_string()),
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
        let value: Value =
            serde_json::from_str(&line).with_context(|| path.display().to_string())?;
        let id = value
            .get("img_id")
            .context("existing output lacks img_id")?;
        ids.insert(id.to_string());
        needs_newline = !line.ends_with('\n');
    }
    Ok((ids, needs_newline))
}
