//! Library integration: load Vela 2.0 and run a SystemOne body from stdin.
//!
//! ```text
//! cargo run -p sinteric-inference --example vela2 -- \
//!   /path/to/vela2-0p3b 2 < body.json
//! ```
use anyhow::{Context, Result};
use sinteric_inference::vela2::{Config, Vela2};
use sinteric_inference::{Runtime, RuntimeOptions};
use std::io::Read;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().context("model directory")?;
    let threads = args
        .next()
        .unwrap_or_else(|| "2".into())
        .parse()
        .unwrap_or(2);

    let rt = Runtime::new(RuntimeOptions::new().with_threads(threads))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let vela: Vela2 = rt
        .load(Config::new(model))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut body = String::new();
    std::io::stdin().read_to_string(&mut body)?;
    let body: serde_json::Value = serde_json::from_str(&body)?;
    let out = vela.system_one(&body).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("{out}");
    Ok(())
}
