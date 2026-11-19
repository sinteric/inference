//! Built-in request mix. Lengths and batch sizes vary on purpose so the
//! plan cache sees several shapes, as a real host's traffic does.
use serde_json::{json, Value};

/// One request body plus the id the load generator reports it under.
pub struct Input {
    pub id: String,
    pub body: Value,
}

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

fn paragraph(words: usize, seed: usize) -> String {
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

/// `/embed` bodies: batch 1 queries up to batch 8 of paragraph-length
/// documents.
pub fn embed_requests() -> Vec<Input> {
    let mut out = Vec::new();
    for (i, s) in SENTENCES.iter().enumerate() {
        out.push(Input {
            id: format!("q{i}"),
            body: json!({"texts": [s], "kind": "query"}),
        });
    }
    for (batch, words) in [(2, 40), (4, 80), (8, 30), (1, 300), (4, 160)] {
        let texts: Vec<String> = (0..batch).map(|j| paragraph(words, j)).collect();
        out.push(Input {
            id: format!("d{batch}x{words}"),
            body: json!({"texts": texts, "kind": "document"}),
        });
    }
    out
}

/// `/vela` SystemOne bodies over states of different lengths and question
/// types.
pub fn vela_bodies() -> Vec<Input> {
    let questions = json!({
        "pii": {"type": "noul"},
        "topic": {"type": "choice", "instructions": "topic?",
                  "criteria": {"legal": "contract or law", "finance": "money", "other": null}},
        "rating": {"type": "score", "instructions": "Rate relevance", "criteria": ["low", "medium", "high"]},
        "people": {"type": "span", "criteria": {"person": "person name"}, "over": "request"},
        "tags": {"type": "set", "criteria": {"person": null, "place": "place"}},
    });
    let mut out = Vec::new();
    for (i, words) in [8usize, 40, 120, 400].into_iter().enumerate() {
        out.push(Input {
            id: format!("v{words}"),
            body: json!({
                "model": "vela-2.0",
                "state": {"request": format!("김철수 John Smith 서울. {}", paragraph(words, i))},
                "questions": questions,
            }),
        });
    }
    out.push(Input {
        id: "v-short".into(),
        body: json!({"model": "vela-2.0", "state": "Hello", "questions": {"q": {"type": "noul"}}}),
    });
    out
}
