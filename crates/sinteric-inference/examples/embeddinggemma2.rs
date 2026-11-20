//! Library integration: load EmbeddingGemma 2 at the precision this CPU runs
//! best and embed one query.
//!
//! ```text
//! cargo run -p sinteric-inference --example embeddinggemma2 -- \
//!   /path/to/embeddinggemma2 4
//! ```
use anyhow::{Context, Result};
use sinteric_inference::cpu::{Options, Runtime};
use sinteric_inference::embeddinggemma2::cpu::{Config, EmbeddingGemma2};
use sinteric_inference::embeddinggemma2::PromptKind;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().context("model directory")?;
    let threads = args.next().map_or(Ok(4), |t| t.parse())?;
    let query = args.next().unwrap_or_else(|| "what is sinteric?".into());

    let rt = Runtime::new(Options::new().with_threads(threads))?;
    // Long inputs are cut to the window, as Sentence Transformers cuts them.
    let model = EmbeddingGemma2::load(&rt, Config::new(model).with_truncate(true))?;
    let rows = model.embed([query.as_str()], PromptKind::Query)?;
    let l2 = rows.row(0)?.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!(
        "{}",
        serde_json::json!({
            "isa": rt.isa(),
            "precision": model.precision(),
            "dim": rows.dim,
            "n": rows.n(),
            "l2": l2,
        })
    );
    Ok(())
}
