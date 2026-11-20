use crate::cpu::{BackendOverrides, IsaCap, Options, Runtime};
use crate::ppocr::cpu::{
    DetConfig, Detector, RecConfig, Recognizer, DEFAULT_WORKSPACE_LIMIT_BYTES,
};
use crate::ppocr::{ImageView, PixelFormat};
use crate::ErrorKind;
use std::path::Path;
fn runtime(options: Options) -> Runtime {
    // Concurrent tests must finish process-global ISA setup before model loads.
    static STARTUP: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _startup = STARTUP.lock().unwrap();
    Runtime::try_new(options).unwrap()
}
fn options() -> Options {
    Options::new()
        .with_threads(1)
        .with_isa_cap(IsaCap::Avx512CoreBf16)
}
fn assets() -> Option<std::path::PathBuf> {
    let root = crate::test_support::env::TestOverrides::from_env()
        .unwrap()
        .ocr_assets?;
    root.join("rec/model.safetensors").is_file().then_some(root)
}
fn load(rt: &Runtime, root: &Path, det_limit: usize, rec_limit: usize) -> (Detector, Recognizer) {
    let det = Detector::load(
        rt,
        DetConfig::new(root.join("det")).with_workspace_limit_bytes(det_limit),
    )
    .unwrap();
    let rec = Recognizer::load(
        rt,
        RecConfig::new(root.join("rec")).with_workspace_limit_bytes(rec_limit),
    )
    .unwrap();
    (det, rec)
}
fn shared(rt: &Runtime, workspace: usize) -> Runtime {
    rt.shared_backend(BackendOverrides::default().with_workspace_limit_bytes(Some(workspace)))
        .unwrap()
}
#[test]
fn decoded_page_api_and_missing_assets() {
    crate::test_support::own_process!(decoded_page_api_and_missing_assets);
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Detector>();
    send_sync::<Recognizer>();
    let rt = runtime(options());
    let missing = Detector::load(&rt, DetConfig::new("/explicit/missing/ocr/det")).unwrap_err();
    assert_eq!(missing.kind(), ErrorKind::ModelFiles, "{missing}");
    let missing = Recognizer::load(&rt, RecConfig::new("/explicit/missing/ocr/rec")).unwrap_err();
    assert_eq!(missing.kind(), ErrorKind::ModelFiles, "{missing}");
    // A shared handle owns the backend settings.
    let ocr_rt = shared(&rt, DEFAULT_WORKSPACE_LIMIT_BYTES);
    assert!(ocr_rt.shares_backend());
    assert!(ocr_rt
        .shared_backend(BackendOverrides::default())
        .is_err_and(|e| e.kind() == ErrorKind::Config));
    let Some(root) = assets() else {
        return;
    };
    // Per-model backend settings on a shared handle are a config error.
    assert!(Detector::load(
        &ocr_rt,
        DetConfig::new(root.join("det")).with_workspace_limit_bytes(1)
    )
    .is_err_and(|e| e.kind() == ErrorKind::Config));
    assert!(Recognizer::load(
        &ocr_rt,
        RecConfig::new(root.join("rec")).with_memory_budget_bytes(1)
    )
    .is_err_and(|e| e.kind() == ErrorKind::Config));
    let bytes = [255; 32 * 32 * 3];
    let image = ImageView::try_new(&bytes, 32, 32, 32 * 3, PixelFormat::Rgb8).unwrap();
    let separate = load(
        &rt,
        &root,
        DEFAULT_WORKSPACE_LIMIT_BYTES,
        DEFAULT_WORKSPACE_LIMIT_BYTES,
    );
    let together = (
        Detector::load(&ocr_rt, DetConfig::new(root.join("det"))).unwrap(),
        Recognizer::load(&ocr_rt, RecConfig::new(root.join("rec"))).unwrap(),
    );
    for (det, rec) in [&separate, &together] {
        let quads = det.detect(&image).unwrap();
        assert!(quads.is_empty());
        assert!(rec.recognize(&image, &quads).unwrap().is_empty());
    }
    for (det, rec) in [separate, together] {
        det.unload();
        rec.unload();
    }
}

#[test]
fn ocr_workspace_is_independent_and_fits_2400_square() {
    crate::test_support::own_process!(ocr_workspace_is_independent_and_fits_2400_square);
    let options = options();
    // The text models keep their 512 MiB runtime default.
    assert_eq!(options.workspace_limit_bytes, 512 * 1024 * 1024);
    let Some(root) = assets() else {
        eprintln!("skipping OCR workspace plans: explicit OCR assets absent");
        return;
    };
    let rt = runtime(options.clone());
    let default = DEFAULT_WORKSPACE_LIMIT_BYTES;
    // An unset config cap is the OCR default, not the runtime's.
    let det = Detector::load(&rt, DetConfig::new(root.join("det"))).unwrap();
    let plan = det.warmup(2400, 2400).unwrap();
    let requested = plan.arena_bytes + plan.scratch_bytes;
    println!(
        "OCR 2400x2400: arena={} scratch={} total={} default={}",
        plan.arena_bytes, plan.scratch_bytes, requested, default
    );
    // The optimized head and shared producer views now fit below 512 MiB.
    assert!(requested <= options.workspace_limit_bytes);
    assert!(requested <= default);
    det.unload();

    // A caller's smaller text-model limit cannot shrink the OCR default.
    let rt = runtime(options.clone().with_workspace_limit_bytes(1024 * 1024));
    let det = Detector::load(&rt, DetConfig::new(root.join("det"))).unwrap();
    assert!(det.warmup(2400, 2400).is_ok());
    det.unload();

    let smaller = requested - 1;
    let (det, rec) = load(&rt, &root, smaller, default);
    let err = det.warmup(2400, 2400).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::TooLarge, "{err}");
    assert!(
        err.to_string()
            .ends_with(&format!("workspace limit is {smaller} bytes")),
        "{err}"
    );
    det.unload();
    rec.unload();

    // The recognizer's own cap reaches its convolution segments.
    let smaller = 1024 * 1024;
    let (det, rec) = load(&rt, &root, default, smaller);
    let err = rec.warmup(1, 1600).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::TooLarge, "{err}");
    assert!(
        err.to_string()
            .ends_with(&format!("workspace limit is {smaller} bytes")),
        "{err}"
    );
    det.unload();
    rec.unload();

    // A shared backend applies its one cap to both models.
    let ocr_rt = shared(&rt, smaller);
    let det = Detector::load(&ocr_rt, DetConfig::new(root.join("det"))).unwrap();
    let rec = Recognizer::load(&ocr_rt, RecConfig::new(root.join("rec"))).unwrap();
    assert!(det.warmup(2400, 2400).is_err());
    assert!(rec.warmup(1, 1600).is_err());
}
