//! RSS around load, one call and release, for each model directory given.
//!
//! ```text
//! cargo run --release -p sinteric-inference --example unload_rss -- \
//!   --eg2 /path/to/embeddinggemma2
//! ```
//!
//! Prints one JSON line per phase. `drop` against `unload` shows what the
//! heap trim in `unload` returns on top of freeing the model's objects.
use anyhow::{bail, Context, Result};
use sinteric_inference::cpu::{Options, Runtime};
use sinteric_inference::embeddinggemma2::cpu::{Config, EmbeddingGemma2};
use sinteric_inference::embeddinggemma2::PromptKind;

fn rss_kib() -> Result<(u64, u64)> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let field = |name: &str| -> Result<u64> {
        let line = status
            .lines()
            .find(|l| l.starts_with(name))
            .with_context(|| format!("{name} missing"))?;
        Ok(line.split_whitespace().nth(1).context("value")?.parse()?)
    };
    Ok((field("VmRSS:")?, field("VmHWM:")?))
}

fn report(model: &str, phase: &str) -> Result<()> {
    let (rss, hwm) = rss_kib()?;
    println!(
        "{}",
        serde_json::json!({"model": model, "phase": phase, "rss_mib": rss / 1024, "hwm_mib": hwm / 1024})
    );
    Ok(())
}

fn eg2_cycle(rt: &Runtime, dir: &str, trim: bool) -> Result<()> {
    let tag = if trim { "unload" } else { "drop" };
    let m = EmbeddingGemma2::load(rt, Config::new(dir))?;
    m.embed(["what is sinteric?"; 8], PromptKind::Query)?;
    report("eg2", &format!("loaded+embed ({tag})"))?;
    if trim {
        m.unload();
    } else {
        drop(m);
    }
    report("eg2", &format!("after {tag}"))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut eg2_dir = None;
    for pair in args.chunks(2) {
        match (pair[0].as_str(), pair.get(1)) {
            ("--eg2", Some(d)) => eg2_dir = Some(d.clone()),
            _ => bail!("usage: unload_rss [--eg2 DIR]"),
        }
    }
    let rt = Runtime::new(Options::new().with_threads(1))?;
    report("-", "runtime")?;
    for trim in [false, true] {
        if let Some(d) = &eg2_dir {
            eg2_cycle(&rt, d, trim)?;
        }
    }
    Ok(())
}
