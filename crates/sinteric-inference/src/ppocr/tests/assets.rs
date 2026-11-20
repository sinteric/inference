use crate::ppocr::rec::CTC_CLASSES;
use crate::ppocr::weights::{AssetConfig, Assets, ModelKind};
use crate::Error;

#[test]
fn pinned_det_and_rec_assets() {
    crate::test_support::own_process!(pinned_det_and_rec_assets);
    let Some(root) = crate::test_support::env::TestOverrides::from_env()
        .unwrap()
        .ocr_assets
    else {
        eprintln!(
            "skipping OCR assets: set SINTERIC_OCR_ASSETS to the explicit imported asset root"
        );
        return;
    };
    for relative in [
        "det/graph.json",
        "det/model.safetensors",
        "rec/graph.json",
        "rec/model.safetensors",
        "rec/dict.txt",
    ] {
        if !root.join(relative).is_file() {
            eprintln!(
                "skipping OCR assets: missing {}",
                root.join(relative).display()
            );
            return;
        }
    }
    for kind in [ModelKind::Det, ModelKind::Rec] {
        let mut config = AssetConfig::new(root.join(kind.name()), kind);
        config.verify_weights = true;
        let assets = Assets::load(&config).unwrap();
        assert_eq!(assets.graph().model, kind);
        assert!(assets
            .graph()
            .nodes
            .iter()
            .all(|n| n.op != "BatchNormalization"));
        for (name, shape) in &assets.graph().tensors {
            let t = assets.tensor(name).unwrap();
            assert_eq!(t.len(), shape.iter().product::<usize>());
            assert!(t.iter().all(|v| v.is_finite()));
        }
        assert!(matches!(assets.tensor("missing"), Err(Error::Missing(_))));
        if kind == ModelKind::Rec {
            assert_eq!(assets.dictionary().len() + 2, CTC_CLASSES);
        }
        println!(
            "{}: {} nodes, {} tensors, {} dictionary entries",
            kind.name(),
            assets.graph().nodes.len(),
            assets.graph().tensors.len(),
            assets.dictionary().len()
        );
    }
}

#[test]
fn boundary_configs() {
    crate::test_support::own_process!(boundary_configs);
    let det = crate::ppocr::api::DetConfig::new("explicit/det").options;
    assert_eq!(det.thresh, 0.2);
    assert_eq!(det.box_thresh, 0.45);
    assert_eq!(det.max_candidates, 1000);
    assert!(det.max_side.is_none());
    assert_eq!(
        crate::ppocr::api::RecConfig::new("explicit/rec")
            .options
            .max_pixels,
        64 * 1024 * 1024
    );
    assert!(crate::ppocr::net::Tensor::new(vec![1, 0], vec![]).is_err());
    assert!(crate::ppocr::net::Tensor::new(vec![usize::MAX, 2], vec![]).is_err());
}

#[test]
fn explicit_manifests_preserve_compiled_graph_identity() {
    crate::test_support::own_process!(explicit_manifests_preserve_compiled_graph_identity);
    let Some(root) = crate::test_support::env::TestOverrides::from_env()
        .unwrap()
        .ocr_assets
    else {
        return;
    };
    if !root.join("rec/model.safetensors").is_file() {
        return;
    }
    let directory = root.join(format!("manifest-test-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("ppocr.toml");
    let mut config = AssetConfig::new(root.join("rec"), ModelKind::Rec);
    config.manifest = Some(path.clone());
    assert!(Assets::load(&config).is_err());
    std::fs::write(&path, crate::ppocr::ASSET_MANIFEST).unwrap();
    assert!(Assets::load(&config).is_ok());
    std::fs::write(&path, "schema_version = 99").unwrap();
    assert!(Assets::load(&config).is_err());
    let mut value: toml::Value = toml::from_str(crate::ppocr::ASSET_MANIFEST).unwrap();
    value["rec"]["onnx_sha256"] = toml::Value::String("0".repeat(64));
    std::fs::write(&path, toml::to_string(&value).unwrap()).unwrap();
    assert!(Assets::load(&config).is_err());
    let file = std::fs::File::create(&path).unwrap();
    file.set_len(128 * 1024 * 1024 + 1).unwrap();
    assert!(Assets::load(&config).is_err());
}
