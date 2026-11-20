//! Image and audio parity against Sentence Transformers FP32
//! (`fixtures/multimodal.json`, from `scripts/generate_multimodal_fixtures.py`).
//! Media are synthetic and integer-defined, so these tests rebuild
//! byte-identical inputs instead of committing media files.

use crate::cpu::runtime::{Options, Runtime};
use crate::embeddinggemma2::cpu::{Config, EmbeddingGemma2, Precision};
use crate::embeddinggemma2::{media, ImageView, Part, PixelFormat, PromptKind};
use crate::test_support::media::{audio_samples, image_pixels};
use crate::ErrorKind;
use serde_json::Value;

fn asset_dir() -> Option<std::path::PathBuf> {
    let manifest = crate::assets::Manifest::parse(crate::embeddinggemma2::ASSET_MANIFEST).unwrap();
    let root = crate::test_support::assets::asset_root(&manifest)?;
    let st = crate::test_support::assets::locate(&manifest, &root, "model.safetensors").ok()?;
    let tok = crate::test_support::assets::locate(&manifest, &root, "tokenizer.json").ok()?;
    if crate::test_support::assets::skip_if_missing(&st)
        || crate::test_support::assets::skip_if_missing(&tok)
    {
        return None;
    }
    let crate::test_support::assets::AssetStatus::Ready(p) = st else {
        return None;
    };
    Some(p.parent().unwrap().to_path_buf())
}

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/embeddinggemma2/multimodal.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixtures/multimodal.json")).unwrap()
}

fn fnv1a64(data: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

enum Media {
    Image {
        h: usize,
        w: usize,
        c: usize,
        px: Vec<u8>,
    },
    Audio(Vec<f32>),
}

fn media(doc: &Value, kind: &str, name: &str) -> Media {
    if kind == "image" {
        let d: Vec<usize> = doc["images"][name]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        Media::Image {
            h: d[0],
            w: d[1],
            c: d[2],
            px: image_pixels(d[0], d[1], d[2]),
        }
    } else {
        Media::Audio(audio_samples(doc["audio"][name].as_u64().unwrap() as usize))
    }
}

fn view(h: usize, w: usize, c: usize, px: &[u8]) -> ImageView<'_> {
    let format = if c == 1 {
        PixelFormat::Gray8
    } else {
        PixelFormat::Rgb8
    };
    ImageView::new(px, w as u32, h as u32, w * c, format).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn preprocessing_matches_processors() {
    crate::test_support::own_process!(preprocessing_matches_processors);
    let doc = fixture();
    for p in doc["size_probes"].as_array().unwrap() {
        let get = |k: &str| p[k].as_u64().unwrap() as usize;
        let want: Vec<usize> = p["target_hw"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let got = media::image_target_size(get("h"), get("w"), get("max_soft_tokens")).unwrap();
        assert_eq!([got.0, got.1], want[..], "size probe {p}");
    }
    for case in doc["cases"].as_array().unwrap() {
        for part in case["parts"].as_array().unwrap() {
            let (kind, name) = (part[0].as_str().unwrap(), part[1].as_str().unwrap());
            if kind == "text" {
                continue;
            }
            match media(&doc, kind, name) {
                Media::Image { h, w, c, px } => {
                    let want = &case["image"];
                    let (th, tw) =
                        media::image_target_size(h, w, media::ImageTokens::default().count())
                            .unwrap();
                    assert_eq!(
                        [th as u64, tw as u64],
                        [
                            want["resized_hw"][0].as_u64().unwrap(),
                            want["resized_hw"][1].as_u64().unwrap()
                        ]
                    );
                    let rgb = crate::image::to_rgb(&view(h, w, c, &px)).unwrap();
                    let resized =
                        crate::image::resize_bicubic_aa_rgb(&rgb, w, h, tw, th, 2).unwrap();
                    assert_eq!(
                        fnv1a64(&resized),
                        want["resized_fnv1a64"].as_str().unwrap(),
                        "{} resized bytes",
                        case["id"]
                    );
                    let p = media::image_patches(&view(h, w, c, &px), 280, 2).unwrap();
                    assert_eq!(
                        p.soft_tokens() as u64,
                        want["soft_tokens"].as_u64().unwrap()
                    );
                }
                Media::Audio(samples) => {
                    let want = &case["audio"];
                    let f = media::audio_features(&samples, 2).unwrap();
                    assert_eq!(f.frames as u64, want["valid_frames"].as_u64().unwrap());
                    assert_eq!(
                        f.soft_tokens() as u64,
                        want["soft_tokens"].as_u64().unwrap()
                    );
                    let sum: f64 = f.mel.iter().map(|&v| v as f64).sum();
                    let abs: f64 = f.mel.iter().map(|&v| (v as f64).abs()).sum();
                    let (ws, wa) = (
                        want["feature_sum"].as_f64().unwrap(),
                        want["feature_abs_sum"].as_f64().unwrap(),
                    );
                    assert!(
                        (sum - ws).abs() <= 1e-4 * wa,
                        "{} mel sum {sum} vs {ws}",
                        case["id"]
                    );
                    assert!(
                        (abs - wa).abs() <= 1e-4 * wa,
                        "{} mel |sum| {abs} vs {wa}",
                        case["id"]
                    );
                    let mut worst = 0f64;
                    for (row, key) in [(0, "first_frame"), (f.frames - 1, "last_frame")] {
                        let got = &f.mel[row * media::MEL_BINS..(row + 1) * media::MEL_BINS];
                        for (g, w) in got.iter().zip(floats(&want[key])) {
                            worst = worst.max((*g as f64 - w).abs());
                        }
                    }
                    assert!(worst < 1e-3, "{} mel frame max |d| {worst}", case["id"]);
                }
            }
        }
    }
}

fn parity(precision: Precision, gate: f32) {
    let Some(dir) = asset_dir() else {
        return;
    };
    let threads = std::env::var("EG2_MM_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let rt = Runtime::new(Options::new().with_threads(threads)).unwrap();
    if !EmbeddingGemma2::precisions(&rt).contains(&precision) {
        eprintln!("skipping: {precision} does not run on {}", rt.isa());
        return;
    }
    let model = EmbeddingGemma2::load(
        &rt,
        Config::new(&dir)
            .with_precision(precision)
            .with_image(true)
            .with_audio(true),
    )
    .unwrap();
    let doc = fixture();
    let mut min_cos = 1f32;
    for case in doc["cases"].as_array().unwrap() {
        let media: Vec<(String, Media)> = case["parts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let (kind, name) = (p[0].as_str().unwrap(), p[1].as_str().unwrap());
                if kind == "text" {
                    (name.to_string(), Media::Audio(Vec::new()))
                } else {
                    (kind.to_string(), media(&doc, kind, name))
                }
            })
            .collect();
        let parts: Vec<Part<'_>> = case["parts"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&media)
            .map(|(p, (text, m))| match (p[0].as_str().unwrap(), m) {
                ("text", _) => Part::Text(text),
                (_, Media::Image { h, w, c, px }) => Part::Image(view(*h, *w, *c, px)),
                (_, Media::Audio(s)) => Part::Audio(s),
            })
            .collect();
        let prompt = match case["prompt"].as_str() {
            Some("query") => Some(PromptKind::Query),
            Some(_) => Some(PromptKind::Document),
            None => None,
        };
        let want_ids: Vec<u32> = case["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            model.parts_input_ids(&parts, prompt).unwrap(),
            want_ids,
            "{} input ids",
            case["id"]
        );
        let t = std::time::Instant::now();
        let got = model.embed_parts(&[&parts], prompt).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let want: Vec<f32> = floats(&case["embedding_fp32"])
            .into_iter()
            .map(|v| v as f32)
            .collect();
        let cos = crate::test_support::qual::cosine(&want, got.row(0).unwrap()).unwrap();
        eprintln!(
            "{precision:?} {} tokens={} cosine={cos} {ms:.0} ms",
            case["id"],
            want_ids.len()
        );
        min_cos = min_cos.min(cos);
    }
    eprintln!("eg2 multimodal {precision:?} threads={threads} min cosine vs FP32={min_cos}");
    assert!(
        min_cos >= gate,
        "{precision:?} min cosine {min_cos} < {gate}"
    );
}

#[test]
fn image_audio_parity_bf16() {
    crate::test_support::own_process!(image_audio_parity_bf16);
    parity(Precision::Bf16, 0.999);
}

#[test]
fn image_audio_parity_fp32() {
    crate::test_support::own_process!(image_audio_parity_fp32);
    parity(Precision::Fp32, 0.99999);
}

#[test]
fn text_only_load_rejects_media() {
    crate::test_support::own_process!(text_only_load_rejects_media);
    let Some(dir) = asset_dir() else {
        return;
    };
    let rt = Runtime::new(Options::new().with_threads(1)).unwrap();
    let model = EmbeddingGemma2::load(&rt, Config::new(&dir)).unwrap();
    assert!(!model.supports_image() && !model.supports_audio());
    let px = image_pixels(48, 48, 3);
    let img = [Part::Image(view(48, 48, 3, &px))];
    assert_eq!(
        model.embed_parts(&[&img], None).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    let clip = vec![0f32; 16_000];
    assert_eq!(
        model
            .embed_parts(&[&[Part::Audio(&clip)]], None)
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    // Text-only parts match the text API.
    let a = model
        .embed_parts(
            &[&[Part::Text("hello "), Part::Text("world")]],
            Some(PromptKind::Query),
        )
        .unwrap();
    let b = model.embed(["hello world"], PromptKind::Query).unwrap();
    assert_eq!(a, b);
    assert!(model
        .embed_parts(&[&[Part::Text("<|image|>")]], None)
        .is_err());
    assert!(model.embed_parts(&[&[]], None).is_err());
}
