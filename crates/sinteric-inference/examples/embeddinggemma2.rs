//! Library integration: load EmbeddingGemma 2 and embed one query.
//!
//! ```text
//! cargo run -p sinteric-inference --example embeddinggemma2 -- \
//!   /path/to/embeddinggemma2 4
//! ```
use anyhow::{Context, Result};
use sinteric_inference::embeddinggemma2::{Config, EmbeddingGemma2, PromptKind};
use sinteric_inference::{Runtime, RuntimeOptions};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().context("model directory")?;
    let threads = args
        .next()
        .unwrap_or_else(|| "4".into())
        .parse()
        .unwrap_or(4);
    let query = args.next().unwrap_or_else(|| "what is sinteric?".into());

    let rt = Runtime::new(RuntimeOptions::new().with_threads(threads))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let model: EmbeddingGemma2 = rt
        .load(Config::new(model))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let rows = model
        .embed([query.as_str()], PromptKind::Query)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let l2 = rows
        .row(0)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    println!(
        "{}",
        serde_json::json!({
            "dim": rows.dim,
            "n": rows.n(),
            "l2": l2,
        })
    );
    Ok(())
}
