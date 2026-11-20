//! `golden compare`: every case of the base against the same case of the new
//! record, and each family's times and peak RSS beside the base's.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

use crate::record::{Meta, Record};

type Key = (String, String, String, usize);

struct Side {
    cases: BTreeMap<Key, Record>,
    metas: BTreeMap<String, Meta>,
}

fn load(path: &Path) -> Result<Side> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut side = Side {
        cases: BTreeMap::new(),
        metas: BTreeMap::new(),
    };
    for (i, line) in text.lines().enumerate() {
        let value: Value =
            serde_json::from_str(line).with_context(|| format!("{}:{}", path.display(), i + 1))?;
        if value.get("meta").is_some() {
            let meta: Meta = serde_json::from_value(value)?;
            side.metas.insert(meta.meta.clone(), meta);
        } else {
            let r: Record = serde_json::from_value(value)?;
            let key = (r.model.clone(), r.case.clone(), r.isa.clone(), r.threads);
            side.cases.insert(key, r);
        }
    }
    Ok(side)
}

/// Prints the comparison; returns whether every base case is in the new
/// record with identical bytes.
pub fn run(base: &Path, new: &Path) -> Result<bool> {
    let (base, new) = (load(base)?, load(new)?);
    if base.cases.is_empty() {
        bail!("the base record has no cases");
    }
    let mut same = true;
    let mut per_model: BTreeMap<&str, (usize, f64, f64)> = BTreeMap::new();
    for (key, b) in &base.cases {
        let Some(n) = new.cases.get(key) else {
            println!("MISSING  {} {} ({} t{})", key.0, key.1, key.2, key.3);
            same = false;
            continue;
        };
        if !n.repeatable {
            println!(
                "UNSTABLE {} {}: repeats differ within the new build",
                key.0, key.1
            );
            same = false;
        }
        if b.sha256 != n.sha256 {
            same = false;
            let d = Diff::of(&b.output, &n.output);
            println!(
                "DIFFERS  {} {}: max |Δ| {:.3e} over {} numbers, {} text change(s), {}",
                key.0,
                key.1,
                d.max_abs,
                d.numbers,
                d.texts,
                if d.shape {
                    "same shape"
                } else {
                    "SHAPE CHANGED"
                }
            );
        }
        let entry = per_model.entry(key.0.as_str()).or_default();
        entry.0 += 1;
        entry.1 += b.warm_ms;
        entry.2 += n.warm_ms;
    }
    for key in new.cases.keys().filter(|k| !base.cases.contains_key(*k)) {
        println!("NEW      {} {} (not in the base)", key.0, key.1);
    }
    println!("\nwarm time, sum over cases (new / base):");
    for (model, (cases, b, n)) in &per_model {
        println!(
            "  {model:<12} {cases:3} cases  {b:9.1} ms -> {n:9.1} ms  ({:+.1}%)",
            (n / b - 1.0) * 100.0
        );
    }
    println!("\nload and peak RSS (new / base):");
    for (family, b) in &base.metas {
        let Some(n) = new.metas.get(family) else {
            println!("  {family:<12} missing in the new record");
            continue;
        };
        println!(
            "  {family:<12} load {:8.1} -> {:8.1} ms   peak RSS {:7.1} -> {:7.1} MiB",
            b.load_ms,
            n.load_ms,
            b.peak_rss_kib as f64 / 1024.0,
            n.peak_rss_kib as f64 / 1024.0
        );
    }
    let total = base.cases.len();
    println!(
        "\n{}: {total} base cases",
        if same { "IDENTICAL" } else { "NOT IDENTICAL" }
    );
    Ok(same)
}

/// How far two outputs are apart: the largest difference between numbers in
/// the same place, how many strings changed, and whether the shape held.
#[derive(Default)]
struct Diff {
    max_abs: f64,
    numbers: usize,
    texts: usize,
    shape: bool,
}

impl Diff {
    fn of(a: &Value, b: &Value) -> Self {
        let mut d = Self {
            shape: true,
            ..Self::default()
        };
        d.walk(a, b);
        d
    }

    fn walk(&mut self, a: &Value, b: &Value) {
        match (a, b) {
            (Value::Number(x), Value::Number(y)) => {
                let (x, y) = (
                    x.as_f64().unwrap_or(f64::NAN),
                    y.as_f64().unwrap_or(f64::NAN),
                );
                self.numbers += 1;
                self.max_abs = self.max_abs.max((x - y).abs());
            }
            (Value::String(x), Value::String(y)) => self.texts += usize::from(x != y),
            (Value::Array(x), Value::Array(y)) => {
                self.shape &= x.len() == y.len();
                for (x, y) in x.iter().zip(y) {
                    self.walk(x, y);
                }
            }
            (Value::Object(x), Value::Object(y)) => {
                self.shape &= x.len() == y.len();
                for (k, x) in x {
                    match y.get(k) {
                        Some(y) => self.walk(x, y),
                        None => self.shape = false,
                    }
                }
            }
            (x, y) => {
                self.shape &= std::mem::discriminant(x) == std::mem::discriminant(y);
                self.texts += usize::from(x != y);
            }
        }
    }
}
