//! One prepared batch through the public oneDNN library. No Python or ONNX is invoked.
use anyhow::{Context, Result};
use rt_onednn::OnednnCpu;
use rt_text::Inputs;
use rt_threads::ThreadBudget;
use std::io::Read;
use vela2_0p3b::onednn::model::Model;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().context("model directory")?;
    let threads = args
        .next()
        .unwrap_or_else(|| "1".into())
        .parse()
        .unwrap_or(1);
    let budget = ThreadBudget::new(threads)?;
    let backend = OnednnCpu::load(&budget)?;
    let model = Model::load(std::path::Path::new(&model_path), backend, 64)?;
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text)?;
    let input: Inputs = serde_json::from_str(&text)?;
    let output = model.forward(&input)?;
    println!(
        "{}",
        serde_json::json!({"opt_logits":output.opt_logits,"span_logits":output.span_logits})
    );
    Ok(())
}
