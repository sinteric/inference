//! The fixed inputs. Changing any of them invalidates every recorded base.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const SENTENCES: &[&str] = &[
    "what is sinteric?",
    "서울에서 부산까지 KTX로 얼마나 걸리나요?",
    "Quarterly revenue grew 12% year over year, driven by enterprise contracts in Korea and Japan.",
    "개인정보 처리방침에 따라 고객의 이름, 연락처, 주소는 계약 종료 후 즉시 파기됩니다.",
    "The tenant shall give written notice no later than sixty days before the end of the lease term.",
    "회의록: 3분기 로드맵 검토, OCR 파이프라인 성능 개선, 임베딩 모델 교체 일정 확정.",
    "How do I reset my password if I no longer have access to the email address on file?",
    "본 계약은 대한민국 법률에 따라 해석되며, 분쟁 발생 시 서울중앙지방법원을 제1심 관할 법원으로 한다.",
];

/// About `words` words of the sentences above, starting at `seed`.
pub fn paragraph(words: usize, seed: usize) -> String {
    let mut out = String::new();
    let mut i = seed;
    while out.split_whitespace().count() < words {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(SENTENCES[i % SENTENCES.len()]);
        i += 3;
    }
    out
}

/// A batch of texts for an embedding model and whether it is a query.
pub struct TextCase {
    pub id: String,
    pub texts: Vec<String>,
    pub query: bool,
}

/// Single queries and documents of every length class, and padded batches
/// whose rows differ in length.
pub fn text_cases() -> Vec<TextCase> {
    let mut out: Vec<TextCase> = SENTENCES
        .iter()
        .enumerate()
        .map(|(i, s)| TextCase {
            id: format!("q{i}"),
            texts: vec![(*s).to_string()],
            query: true,
        })
        .collect();
    for (batch, words) in [(2, 40), (4, 80), (8, 30), (1, 300), (4, 160), (1, 1200)] {
        out.push(TextCase {
            id: format!("d{batch}x{words}"),
            texts: (0..batch).map(|j| paragraph(words, j)).collect(),
            query: false,
        });
    }
    out
}

/// `(h, w, c)` u8, row-major, channel last. The same pattern as the
/// EmbeddingGemma 2 multimodal fixtures.
pub fn image_pixels(h: usize, w: usize, c: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(h * w * c);
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            for ch in 0..c as i64 {
                let v = ((x * 3 + y * 5 + ch * 85) ^ ((x * y) >> 3)) + ((x / 16 + y / 16) % 2) * 64;
                out.push((v & 255) as u8);
            }
        }
    }
    out
}

/// Mono 16 kHz samples in `[-1, 1)`, as in the multimodal fixtures.
pub fn audio_samples(n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut state: i64 = 12345;
    for i in 0..n as i64 {
        state = (state * 1_103_515_245 + 12345) % (1 << 31);
        let noise = ((state >> 16) & 0x7fff) - 16384;
        let freq = 200 + (i / 1600) % 8 * 110;
        let saw = (i * freq * 65536 / 16000) % 65536 - 32768;
        let env = (i / 800) % 20;
        let s = (saw.div_euclid(4) + noise.div_euclid(16)) * env;
        out.push(s.div_euclid(20) as f32 / 32768.0);
    }
    out
}

/// Vela 2.0 SystemOne bodies over states of different lengths and every
/// question type.
pub fn vela_bodies() -> Vec<(String, Value)> {
    let questions = json!({
        "pii": {"type": "noul"},
        "topic": {"type": "choice", "instructions": "topic?",
                  "criteria": {"legal": "contract or law", "finance": "money", "other": null}},
        "rating": {"type": "score", "instructions": "Rate relevance", "criteria": ["low", "medium", "high"]},
        "people": {"type": "span", "criteria": {"person": "person name"}, "over": "request"},
        "tags": {"type": "set", "criteria": {"person": null, "place": "place"}},
    });
    let mut out: Vec<(String, Value)> = [8usize, 40, 120, 400]
        .into_iter()
        .enumerate()
        .map(|(i, words)| {
            let body = json!({
                "model": "vela-2.0",
                "state": {"request": format!("김철수 John Smith 서울. {}", paragraph(words, i))},
                "questions": questions,
            });
            (format!("v{words}"), body)
        })
        .collect();
    out.push((
        "v-short".into(),
        json!({"model": "vela-2.0", "state": "Hello", "questions": {"q": {"type": "noul"}}}),
    ));
    out
}

/// A decoded page: 8-bit RGB or gray, tightly packed.
pub struct Page {
    pub name: String,
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub channels: usize,
}

/// Every `*.png` in `dir`, by name.
pub fn pages(dir: &Path) -> Result<Vec<Page>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        bail!("no .png pages in {}", dir.display());
    }
    paths.iter().map(|p| decode_png(p)).collect()
}

fn decode_png(path: &Path) -> Result<Page> {
    let mut decoder = png::Decoder::new(std::fs::File::open(path)?);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    buf.truncate(info.buffer_size());
    let (data, channels) = match info.color_type {
        png::ColorType::Rgb => (buf, 3),
        png::ColorType::Grayscale => (buf, 1),
        png::ColorType::Rgba => (
            buf.as_chunks::<4>()
                .0
                .iter()
                .flat_map(|&[r, g, b, _]| [r, g, b])
                .collect(),
            3,
        ),
        png::ColorType::GrayscaleAlpha => {
            (buf.as_chunks::<2>().0.iter().map(|&[v, _]| v).collect(), 1)
        }
        other => bail!("{}: unsupported PNG colour type {other:?}", path.display()),
    };
    let name = path
        .file_stem()
        .context("page file name")?
        .to_string_lossy()
        .into_owned();
    Ok(Page {
        name,
        data,
        width: info.width,
        height: info.height,
        channels,
    })
}
