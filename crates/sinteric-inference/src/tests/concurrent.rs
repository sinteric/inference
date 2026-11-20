//! A host calling several models from many threads at once (a web server's
//! blocking pool) must get the same bits as the same calls made one at a
//! time. Runs on whichever pinned assets are present: EG2 from the asset
//! cache, OCR from `SINTERIC_OCR_ASSETS`.

use std::path::PathBuf;
use std::sync::Arc;

use crate::cpu::{BackendOverrides, Options, Runtime};
use crate::embeddinggemma2::cpu::EmbeddingGemma2;
use crate::embeddinggemma2::{self as eg2};
use crate::ppocr::cpu::{
    DetConfig, Detector, RecConfig, Recognizer, DEFAULT_WORKSPACE_LIMIT_BYTES,
};
use crate::ppocr::{ImageView, PixelFormat};

fn model_dir(manifest: &str) -> Option<PathBuf> {
    let manifest = crate::assets::Manifest::parse(manifest).unwrap();
    let root = crate::test_support::assets::asset_root(&manifest)?;
    let status =
        crate::test_support::assets::locate(&manifest, &root, "model.safetensors").unwrap();
    if crate::test_support::assets::skip_if_missing(&status) {
        return None;
    }
    let crate::test_support::assets::AssetStatus::Ready(weights) = status else {
        return None;
    };
    Some(weights.parent().unwrap().to_path_buf())
}

fn ocr_root() -> Option<PathBuf> {
    let root = crate::test_support::env::TestOverrides::from_env()
        .unwrap()
        .ocr_assets?;
    root.join("rec/model.safetensors").is_file().then_some(root)
}

/// A light page with dark bars of different lengths: the detector finds
/// boxes, so the recognizer runs on several widths.
fn bar_page() -> (Vec<u8>, u32, u32) {
    let (w, h) = (640u32, 480u32);
    let mut bgr = vec![235u8; (w * h * 3) as usize];
    for (row, len) in [
        (40u32, 560u32),
        (110, 300),
        (180, 450),
        (250, 120),
        (320, 600),
        (390, 220),
    ] {
        for y in row..row + 22 {
            for x in 30..30 + len {
                let i = ((y * w + x) * 3) as usize;
                let v = if (x / 9 + y / 7) % 3 == 0 { 30 } else { 60 };
                bgr[i..i + 3].fill(v);
            }
        }
    }
    (bgr, w, h)
}

type Job = Box<dyn Fn() -> Vec<u8> + Send + Sync>;

/// Page OCR as a host composes it: detector then recognizer on one backend.
struct Ocr {
    det: Detector,
    rec: Recognizer,
}

impl Ocr {
    fn load(rt: &Runtime, root: &std::path::Path) -> Self {
        let shared = rt
            .shared_backend(
                BackendOverrides::default()
                    .with_workspace_limit_bytes(Some(DEFAULT_WORKSPACE_LIMIT_BYTES)),
            )
            .unwrap();
        Self {
            det: Detector::load(&shared, DetConfig::new(root.join("det"))).unwrap(),
            rec: Recognizer::load(&shared, RecConfig::new(root.join("rec"))).unwrap(),
        }
    }
}

fn jobs(eg2: Option<Arc<EmbeddingGemma2>>, ocr: Option<Arc<Ocr>>) -> Vec<(String, Job)> {
    let mut out: Vec<(String, Job)> = Vec::new();
    if let Some(m) = eg2 {
        let long =
            "서울에서 부산까지 KTX로 얼마나 걸리나요? The tenant shall give notice. ".repeat(12);
        for (i, (texts, kind)) in [
            (
                vec!["what is sinteric?".to_string()],
                eg2::PromptKind::Query,
            ),
            (
                vec![long.clone(), "짧은 문장".into(), long.clone() + "끝"],
                eg2::PromptKind::Document,
            ),
            (
                vec!["개인정보 처리방침".to_string(); 5],
                eg2::PromptKind::Document,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let m = m.clone();
            out.push((
                format!("eg2-{i}"),
                Box::new(move || {
                    let rows = m.embed(&texts, kind).unwrap();
                    rows.data
                        .iter()
                        .flat_map(|x| x.to_bits().to_le_bytes())
                        .collect()
                }),
            ));
        }
    }
    if let Some(m) = ocr {
        let (bgr, w, h) = bar_page();
        out.push((
            "ocr-bars".into(),
            Box::new(move || {
                let image = ImageView::new(&bgr, w, h, w as usize * 3, PixelFormat::Bgr8).unwrap();
                let quads = m.det.detect(&image).unwrap();
                let lines = m.rec.recognize(&image, &quads).unwrap();
                serde_json::to_vec(&(quads, lines)).unwrap()
            }),
        ));
    }
    out
}

#[test]
fn concurrent_mixed_calls_match_serial() {
    crate::test_support::own_process!(concurrent_mixed_calls_match_serial);
    let eg2_dir = model_dir(crate::embeddinggemma2::ASSET_MANIFEST);
    let ocr_dir = ocr_root();
    if eg2_dir.is_none() && ocr_dir.is_none() {
        eprintln!("skipping concurrent calls: no model assets present");
        return;
    }
    // exec_slots 1 shares one stream per model; 3 gives concurrent calls
    // their own stream and scratch (the per-call plan-argument path). A
    // 32 MiB memory budget makes every model evict plans and buffers while
    // the others run; results must not change.
    let mut first: Option<Vec<Vec<u8>>> = None;
    for (exec_slots, budget) in [(1, 0), (3, 0), (1, 32 << 20)] {
        let rt = Runtime::new(
            Options::new()
                .with_threads(2)
                .with_exec_slots(exec_slots)
                .with_memory_budget_bytes(budget),
        )
        .unwrap();
        let eg2 = eg2_dir
            .as_ref()
            .map(|d| Arc::new(EmbeddingGemma2::load(&rt, eg2::cpu::Config::new(d)).unwrap()));
        let ocr = ocr_dir.as_ref().map(|d| Arc::new(Ocr::load(&rt, d)));
        let jobs = jobs(eg2.clone(), ocr.clone());
        let reference: Vec<Vec<u8>> = jobs.iter().map(|(_, job)| job()).collect();
        let first = first.get_or_insert_with(|| reference.clone());
        for ((name, _), (got, want)) in jobs.iter().zip(reference.iter().zip(first.iter())) {
            assert!(
                got == want,
                "{name} differs from exec_slots 1 without a budget (exec_slots {exec_slots}, budget {budget})"
            );
        }
        // Each thread runs every job twice, starting at a different job, so
        // all model pairs overlap.
        std::thread::scope(|scope| {
            for t in 0..6 {
                let (jobs, reference) = (&jobs, &reference);
                scope.spawn(move || {
                    for k in 0..2 * jobs.len() {
                        let i = (t + k) % jobs.len();
                        let (name, job) = &jobs[i];
                        assert!(
                            job() == reference[i],
                            "{name} differs from its serial result (exec_slots {exec_slots}, budget {budget}, thread {t})"
                        );
                    }
                });
            }
        });
        drop(jobs);
        if let Some(m) = eg2.and_then(Arc::into_inner) {
            m.unload();
        }
        if let Some(Ocr { det, rec }) = ocr.and_then(Arc::into_inner) {
            det.unload();
            rec.unload();
        }
    }
}
