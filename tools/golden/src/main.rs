//! The refactor gate: every model's outputs on fixed inputs, recorded by one
//! build and compared with another's.
//!
//! ```text
//! golden record --model eg2-bf16 --dir EG2_DIR --threads 4 --isa best --out base-eg2-bf16.jsonl
//! golden record --model guard-fp32 --dir GUARD_DIR --out base-guard-fp32.jsonl
//! golden record --model ocr --dir PPOCR_DIR --pages PNG_DIR --out base-ocr.jsonl
//! golden compare base-eg2-bf16.jsonl new-eg2-bf16.jsonl
//! ```
//!
//! A change that only restructures code must leave every output bit for bit
//! as it was on the same machine, ISA cap and thread count; `compare` fails on
//! any difference and prints how large it is. It also reports each model's
//! warm call time, load time and peak RSS beside the base's, so a slowdown
//! shows up in the same run.
//!
//! What it cannot see: a different CPU, ISA or thread count than the one
//! recorded (record both sides on one machine, one after the other); inputs
//! it does not run (the cases are in [`inputs`]); and parity with the
//! upstream references (PyTorch, ONNX Runtime), which the model crates' own
//! tests hold. Identical outputs keep that parity only because they are
//! identical.
//!
//! A change to the library's API changes how [`record`] calls a model, so the
//! change's record is made by an adapted recorder. Only the calls may change:
//! the inputs and the way an output becomes bytes stay as the base had them,
//! or the comparison measures the recorder instead of the models.
//!
//! Times are a guard, not a benchmark: two records of one build on a 4-vCPU
//! guest differed by up to 6% in a family's summed warm time (2026-11-20), so
//! only a larger change is a finding. `bench/` measures speed.

mod compare;
mod inputs;
mod record;

use anyhow::{bail, Context, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("record") => record::run(record::Args::parse(&args[1..])?),
        Some("compare") => {
            let [base, new] = &args[1..] else {
                bail!("usage: golden compare BASE.jsonl NEW.jsonl");
            };
            let same = compare::run(&PathBuf::from(base), &PathBuf::from(new))?;
            if !same {
                std::process::exit(1);
            }
            Ok(())
        }
        _ => bail!("usage: golden record ... | golden compare BASE NEW (see the crate docs)"),
    }
}

/// The value after `--name`, if given.
fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn flag_path(args: &[String], name: &str) -> Result<PathBuf> {
    flag(args, name)
        .map(PathBuf::from)
        .with_context(|| format!("{name} DIR is required"))
}
