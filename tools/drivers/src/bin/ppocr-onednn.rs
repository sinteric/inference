//! The OCR gate's runner on this library: PP-OCR detection and recognition
//! over the pages of `--images`, one JSONL result per page (see
//! `drivers::gate`). `--profile` also prints one `ppocr_profile,` line per
//! page with stage timings and backend sizes.
//!
//! `--isa` defaults to `avx512_core_bf16`, the ISA the gate's ONNX Runtime
//! reference was measured against.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use drivers::flags::Flags;
use drivers::gate::{self, Line, Pages};
use serde_json::{json, Value};
use sinteric_inference::cpu::{BackendOverrides, BackendStats, IsaCap, Options, Runtime};
use sinteric_inference::ppocr::cpu::{
    DetConfig, Detector, RecConfig, Recognizer, DEFAULT_WORKSPACE_LIMIT_BYTES,
};

const VALUED: &[&str] = &[
    "--model",
    "--threads",
    "--warmup",
    "--workspace-limit-bytes",
    "--images",
    "--out",
    "--isa",
];
const SWITCHES: &[&str] = &["--profile"];

/// The backend sizes `--profile` reports.
fn backend_sizes(stats: &BackendStats) -> Value {
    json!({
        "cached_plans": stats.cached_plans,
        "conv_arena_bytes": stats.conv_arena_bytes,
        "conv_scratch_bytes": stats.conv_scratch_bytes,
        "execute_scratch_bytes": stats.execute_scratch_bytes,
        "conv_weight_bytes": stats.conv_packed_weight_bytes,
        "linear_weight_bytes": stats.packed_weight_bytes,
        "retained_pool_bytes": stats.retained_output_buffer_bytes,
    })
}

fn main() -> Result<()> {
    let flags = Flags::parse(std::env::args().skip(1), VALUED, SWITCHES)?;
    let model: PathBuf = flags.required("--model")?;
    let isa = flags
        .get::<String>("--isa")?
        .unwrap_or_else(|| "avx512_core_bf16".into());
    let profile = flags.switch("--profile");
    let rt = Runtime::try_new(
        Options::new()
            .with_threads(flags.required("--threads")?)
            .with_isa_cap(isa.parse::<IsaCap>()?),
    )?;
    // Detector and recognizer on one backend: one convolution arena sized
    // to the larger model.
    let shared = rt.shared_backend(
        BackendOverrides::default().with_workspace_limit_bytes(Some(
            flags
                .get("--workspace-limit-bytes")?
                .unwrap_or(DEFAULT_WORKSPACE_LIMIT_BYTES),
        )),
    )?;
    let det_config = DetConfig::new(model.join("det"));
    let max_pixels = det_config.options.max_pixels;
    let det = Detector::load(&shared, det_config)?;
    let rec = Recognizer::load(&shared, RecConfig::new(model.join("rec")))?;
    let pages = Pages {
        images: flags.required("--images")?,
        out: flags.required("--out")?,
        warmup: flags.get("--warmup")?.unwrap_or(0),
        max_pixels,
    };
    gate::run(&pages, |img_id, view| {
        let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let (quads, det_t) = det.detect_timed(view)?;
        let detect_ms = ms(start);
        let (lines, rec_t) = rec.recognize_timed(view, &quads)?;
        let timings = json!({
            "detect_ms": detect_ms,
            "crop_ms": rec_t.crop_ms,
            "recognize_ms": rec_t.recognize_ms,
            "total_ms": ms(start),
            "det_preprocess_ms": det_t.preprocess_ms,
            "det_forward_ms": det_t.forward_ms,
            "det_postprocess_ms": det_t.postprocess_ms,
            "rec_pack_ms": rec_t.pack_ms,
            "rec_forward_ms": rec_t.forward_ms,
            "rec_batches": rec_t.batches,
        });
        if profile {
            println!(
                "ppocr_profile,{}",
                json!({
                    "img_id": img_id,
                    "timings": timings,
                    "page_bgr_bytes": view.height() as usize * view.stride(),
                    "backend": backend_sizes(&det.backend_stats()),
                })
            );
        }
        let lines = lines
            .into_iter()
            .zip(quads)
            .map(|(l, quad)| Line {
                quad,
                text: l.text,
                score: l.score,
            })
            .collect();
        Ok((lines, Some(timings)))
    })
}
