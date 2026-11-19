//! Models in worker processes must give the same bits as in-process, fail
//! typed while their worker is gone, start a new worker on the next call and
//! end the process on unload. `harness = false`: this test binary is also the
//! worker binary, so `main` calls `main_if_worker` first, as a host would.
#[path = "../../rt-assets/tests/support/mod.rs"]
mod asset_test_support;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde_json::json;
use sinteric_inference::embeddinggemma2::{self as eg2, EmbeddingGemma2};
use sinteric_inference::ppocr::{
    DetConfig, Detector, ImageView, PixelFormat, RecConfig, Recognizer,
};
use sinteric_inference::vela2::{self, Vela2};
use sinteric_inference::worker::{main_if_worker, Isolated, WorkerOptions};
use sinteric_inference::{Error, Runtime, RuntimeOptions};

fn model_dir(manifest: &str) -> Option<PathBuf> {
    let manifest = rt_assets::Manifest::parse(manifest).unwrap();
    let root = asset_test_support::asset_root(&manifest)?;
    let status = rt_assets::locate(&manifest, &root, "model.safetensors").unwrap();
    if rt_assets::skip_if_missing(&status) {
        return None;
    }
    let rt_assets::AssetStatus::Ready(weights) = status else {
        return None;
    };
    Some(weights.parent().unwrap().to_path_buf())
}

fn ocr_root() -> Option<PathBuf> {
    let root = sinteric_inference::env::TestOverrides::from_env()
        .unwrap()
        .ocr_assets?;
    root.join("rec/model.safetensors").is_file().then_some(root)
}

fn alive(pid: u32) -> bool {
    // SAFETY: signal 0 only checks that the process exists.
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

fn kill(pid: u32) {
    // SAFETY: SIGKILL to the worker this test started.
    assert_eq!(unsafe { libc::kill(pid as i32, libc::SIGKILL) }, 0);
}

fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Kill the worker, check the call fails typed while it is gone, then that
/// the next call runs on a new worker with the same result; unload ends it.
fn crash_and_restart<M: sinteric_inference::worker::Remote>(
    name: &str,
    model: Isolated<M>,
    call: impl Fn(&Isolated<M>) -> sinteric_inference::Result<Vec<u8>> + Sync,
    want: &[u8],
) {
    assert_eq!(call(&model).unwrap(), want, "{name} differs in a worker");
    let pid = model.pid().unwrap();
    // A call in flight when the worker dies gets `Error::Worker`.
    std::thread::scope(|scope| {
        let pending = scope.spawn(|| call(&model));
        std::thread::sleep(Duration::from_millis(2));
        kill(pid);
        match pending.join().unwrap() {
            Err(Error::Worker(_)) => {}
            Ok(got) => assert_eq!(got, want, "{name} finished before the kill"),
            Err(e) => panic!("{name}: unexpected error {e}"),
        }
    });
    wait_until("the old worker to be noticed", || model.pid() != Some(pid));
    assert_eq!(call(&model).unwrap(), want, "{name} differs after restart");
    let restarted = model.pid().unwrap();
    assert_ne!(restarted, pid);
    model.unload();
    assert!(
        !alive(restarted),
        "{name} worker still running after unload"
    );
    eprintln!("{name}: worker parity, crash and restart ok");
}

fn main() {
    main_if_worker();

    let options = RuntimeOptions::new().with_threads(2);
    let rt = Runtime::new(options).unwrap();
    let worker = || WorkerOptions::new(2);

    // A load error in the worker comes back as an error, not a hang.
    let missing =
        Isolated::<Vela2>::load(&rt, vela2::Config::new("/explicit/missing/vela"), worker());
    assert!(missing.is_err(), "loading a missing model must fail");
    assert!(Isolated::<Vela2>::load(&rt, vela2::Config::new("/x"), WorkerOptions::new(0)).is_err());

    if let Some(dir) = model_dir(embeddinggemma2::ASSET_MANIFEST) {
        let texts = vec![
            "what is sinteric?".to_string(),
            "서울에서 부산까지 KTX로 얼마나 걸리나요? ".repeat(20),
        ];
        let bits = |rows: eg2::Embeddings| -> Vec<u8> {
            rows.data
                .iter()
                .flat_map(|x| x.to_bits().to_le_bytes())
                .collect()
        };
        let local: EmbeddingGemma2 = rt.load(eg2::Config::new(&dir)).unwrap();
        let want = bits(local.embed(&texts, eg2::PromptKind::Document).unwrap());
        let dim = local.embedding_dim();
        drop(local);
        let remote =
            Isolated::<EmbeddingGemma2>::load(&rt, eg2::Config::new(&dir), worker()).unwrap();
        assert_eq!(remote.embedding_dim(), dim);
        let ids = remote.embed_ids(vec![vec![2, 100, 200]], None).unwrap();
        assert_eq!(ids.n(), 1);
        assert!(matches!(
            remote.embed_ids(Vec::new(), None),
            Err(Error::Shape(_))
        ));
        crash_and_restart(
            "eg2",
            remote,
            |m| Ok(bits(m.embed(&texts, eg2::PromptKind::Document)?)),
            &want,
        );
    }

    if let Some(dir) = model_dir(vela2_0p3b::ASSET_MANIFEST) {
        let body = json!({
            "model": "vela-2.0",
            "state": {"request": "김철수 John Smith lives in 서울 and signed the lease. ".repeat(20)},
            "questions": {
                "pii": {"type": "noul"},
                "people": {"type": "span", "criteria": {"person": "person name"}, "over": "request"},
            },
        });
        let local: Vela2 = rt.load(vela2::Config::new(&dir)).unwrap();
        let want = serde_json::to_vec(&local.system_one(&body).unwrap()).unwrap();
        drop(local);
        // A worker can have its own memory budget.
        let budget = 64 << 20;
        let budgeted = Isolated::<Vela2>::load(
            &rt,
            vela2::Config::new(&dir),
            worker().with_memory_budget_bytes(budget),
        )
        .unwrap();
        let stats = budgeted.backend_stats().unwrap();
        assert_eq!(stats["memory_budget_bytes"].as_u64(), Some(budget as u64));
        budgeted.unload();
        let remote = Isolated::<Vela2>::load(&rt, vela2::Config::new(&dir), worker()).unwrap();
        // A bad request keeps its type across the process boundary.
        assert!(matches!(
            remote.system_one(&json!({"model": "vela-2.0"})),
            Err(Error::Request(_))
        ));
        crash_and_restart(
            "vela",
            remote,
            |m| Ok(serde_json::to_vec(&m.system_one(&body)?)?),
            &want,
        );
    }

    if let Some(root) = ocr_root() {
        let (w, h) = (640u32, 480u32);
        let mut bgr = vec![235u8; (w * h * 3) as usize];
        for (row, len) in [(40u32, 560u32), (110, 300), (180, 450), (250, 120)] {
            for y in row..row + 22 {
                for x in 30..30 + len {
                    let i = ((y * w + x) * 3) as usize;
                    let v = if (x / 9 + y / 7) % 3 == 0 { 30 } else { 60 };
                    bgr[i..i + 3].fill(v);
                }
            }
        }
        let image = ImageView::new(&bgr, w, h, w as usize * 3, PixelFormat::Bgr8).unwrap();
        let (det_config, rec_config) = (
            DetConfig::new(root.join("det")),
            RecConfig::new(root.join("rec")),
        );
        let det: Detector = rt.load(det_config.clone()).unwrap();
        let rec: Recognizer = rt.load(rec_config.clone()).unwrap();
        let quads = det.detect(&image).unwrap();
        assert!(!quads.is_empty());
        let want_quads = serde_json::to_vec(&quads).unwrap();
        let want_lines = serde_json::to_vec(&rec.recognize(&image, &quads).unwrap()).unwrap();
        drop((det, rec));
        let remote = Isolated::<Detector>::load(&rt, det_config, worker()).unwrap();
        crash_and_restart(
            "ocr-det",
            remote,
            |m| Ok(serde_json::to_vec(&m.detect(&image)?)?),
            &want_quads,
        );
        let remote = Isolated::<Recognizer>::load(&rt, rec_config, worker()).unwrap();
        crash_and_restart(
            "ocr-rec",
            remote,
            |m| Ok(serde_json::to_vec(&m.recognize(&image, &quads)?)?),
            &want_lines,
        );
    }
    eprintln!("worker tests passed");
}
