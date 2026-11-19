//! Persistent CPU JSON driver for the public Rust inference library.
use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;
use vela_mlas::mlas::{initialize_blas_with_options, Inputs, Model, ModelOptions};
#[derive(Deserialize)]
struct Request {
    #[serde(flatten)]
    inputs: Inputs,
    #[serde(default)]
    warmup: usize,
    #[serde(default = "one")]
    repeat: usize,
    #[serde(default)]
    profile: bool,
    #[serde(default = "yes")]
    return_logits: bool,
}
fn one() -> usize {
    1
}
fn yes() -> bool {
    true
}
fn cpu_time() -> f64 {
    unsafe {
        let mut ts = std::mem::zeroed();
        libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts);
        ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
    }
}
fn memory() -> serde_json::Value {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let smaps = std::fs::read_to_string("/proc/self/smaps_rollup").unwrap_or_default();
    let get = |text: &str, k: &str| {
        text.lines()
            .find(|v| v.starts_with(k))
            .and_then(|v| v.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
            * 1024
    };
    json!({"rss_bytes":get(&status,"VmRSS:"),"peak_rss_bytes":get(&status,"VmHWM:"),"pss_bytes":get(&smaps,"Pss:"),"private_clean_bytes":get(&smaps,"Private_Clean:"),"private_dirty_bytes":get(&smaps,"Private_Dirty:"),"shared_clean_bytes":get(&smaps,"Shared_Clean:"),"shared_dirty_bytes":get(&smaps,"Shared_Dirty:"),"swap_bytes":get(&smaps,"Swap:")})
}
fn main() -> Result<()> {
    ensure!(
        cfg!(target_os = "linux"),
        "model inference is authorized only on the Remote Linux VM"
    );
    let mut model = PathBuf::from("model");
    let mut o = ModelOptions {
        mode: "dense".into(),
        embedding: "eager".into(),
        chunk: 128,
        fused_norm: false,
        blas: false,
        blas_attention: false,
        packed: false,
        fused_geglu: false,
        fused_qkv: false,
        fused_softmax: false,
        native_norm: false,
        projection_workspaces: false,
        residual_inplace: false,
        qkv_rope_workspace: false,
        inplace_softmax: false,
    };
    let mut blas_library = None;
    let mut blas_threads = None;
    let mut blas_options = candle_mlas::BackendOptions::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = PathBuf::from(args.next().context("--model")?),
            "--mode" => o.mode = args.next().context("--mode")?,
            "--embedding" => o.embedding = args.next().context("--embedding")?,
            "--chunk" => o.chunk = args.next().context("--chunk")?.parse()?,
            "--fused-norm" => o.fused_norm = true,
            "--blas-library" => {
                blas_library = Some(args.next().context("--blas-library")?);
                o.blas = true;
            }
            "--blas-threads" => {
                blas_threads = Some(args.next().context("--blas-threads")?.parse::<i32>()?)
            }
            "--mlas-partition" => {
                blas_options.partition = match args.next().context("--mlas-partition")?.as_str() {
                    "rows" => candle_mlas::GemmPartition::Rows,
                    "auto" => candle_mlas::GemmPartition::Auto,
                    _ => bail!("--mlas-partition must be rows or auto"),
                }
            }
            "--aligned-buffers" => blas_options.aligned_buffers = true,
            "--gemm-residual" => blas_options.residual_fusion = true,
            "--debug-alignment" => blas_options.debug_alignment = true,
            "--blas-attention" => o.blas_attention = true,
            "--packed" => o.packed = true,
            "--fused-geglu" => o.fused_geglu = true,
            "--fused-qkv" => o.fused_qkv = true,
            "--fused-softmax" => o.fused_softmax = true,
            "--native-norm" => o.native_norm = true,
            "--projection-workspaces" => o.projection_workspaces = true,
            "--residual-inplace" => o.residual_inplace = true,
            "--qkv-rope-workspace" => o.qkv_rope_workspace = true,
            "--inplace-softmax" => o.inplace_softmax = true,
            _ => bail!("unknown argument {a}"),
        }
    }
    ensure!(
        ["dense", "blocked", "adaptive"].contains(&o.mode.as_str())
            && ["eager", "mmap"].contains(&o.embedding.as_str())
            && o.chunk > 0,
        "invalid options"
    );
    ensure!(
        (!o.blas_attention
            && !o.packed
            && !o.fused_geglu
            && !o.fused_qkv
            && !o.fused_softmax
            && !o.native_norm)
            || o.blas,
        "BLAS options require --blas-library"
    );
    let start = Instant::now();
    let cpu = cpu_time();
    let blas_info = if let Some(path) = blas_library {
        blas_options.threads = blas_threads.unwrap_or(1);
        // The CLI host supplies the pinned, CPU-qualified native bridge.
        Some(unsafe { initialize_blas_with_options(&path, blas_options)? })
    } else {
        None
    };
    let m = Model::load(&model, o.clone())?;
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    writeln!(
        stdout,
        "{}",
        json!({"ready":true,"load_wall_s":start.elapsed().as_secs_f64(),"load_cpu_s":cpu_time()-cpu,"memory":memory(),"mode":o.mode,"embedding":o.embedding,"chunk":o.chunk,"fused_norm":o.fused_norm,"blas":blas_info,"blas_attention":o.blas_attention,"packed":o.packed,"fused_geglu":o.fused_geglu,"fused_qkv":o.fused_qkv,"fused_softmax":o.fused_softmax,"native_norm":o.native_norm,"projection_workspaces":o.projection_workspaces,"residual_inplace":o.residual_inplace,"qkv_rope_workspace":o.qkv_rope_workspace,"inplace_softmax":o.inplace_softmax})
    )?;
    stdout.flush()?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = (|| -> Result<serde_json::Value> {
            let r: Request = serde_json::from_str(&line)?;
            ensure!(
                r.repeat > 0 && r.repeat <= 10000 && r.warmup <= 10000,
                "invalid repetitions"
            );
            for _ in 0..r.warmup {
                m.forward_profiled(&r.inputs, r.profile)?;
            }
            let mut wall = vec![];
            let mut cpu = vec![];
            let mut profile = BTreeMap::<String, f64>::new();
            let mut last = None;
            for _ in 0..r.repeat {
                let c = cpu_time();
                let t = Instant::now();
                let out = m.forward_profiled(&r.inputs, r.profile)?;
                wall.push(t.elapsed().as_secs_f64());
                cpu.push(cpu_time() - c);
                for (k, v) in &out.2 {
                    *profile.entry(k.clone()).or_default() += v;
                }
                last = Some(out);
            }
            let (lg, sp, _) = last.unwrap();
            let mut out =
                json!({"wall_s":wall,"cpu_s":cpu,"memory":memory(),"profile_total_s":profile});
            if r.return_logits {
                out["opt_logits"] = json!(lg.to_vec1::<f32>()?);
                out["span_logits"] = json!(sp.to_vec2::<f32>()?);
            }
            Ok(out)
        })();
        writeln!(
            stdout,
            "{}",
            response.unwrap_or_else(|e| json!({"error":format!("{e:#}")}))
        )?;
        stdout.flush()?;
    }
    Ok(())
}
