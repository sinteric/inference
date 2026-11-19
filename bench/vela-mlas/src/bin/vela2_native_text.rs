//! VM-side persistent native test driver. The library itself changes no environment.
use anyhow::{bail, ensure, Context, Result};
use rt_text::{system_preparation, Frontend, Logits};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::time::Instant;
use vela_mlas::mlas::native_text::{execute_model, recommended_environment, ModelOptions};
use vela_mlas::mlas::{initialize_blas_with_options, Model};
fn main() -> Result<()> {
    ensure!(
        cfg!(target_os = "linux"),
        "inference authorized only on Remote Linux VM"
    );
    let mut args = std::env::args().skip(1);
    let path = args.next().context("model directory")?;
    let mut library = None;
    let mut blas_options = candle_mlas::BackendOptions {
        partition: candle_mlas::GemmPartition::Auto,
        ..Default::default()
    };
    let mut threads = 2i32;
    let mut policy = "ACTIVE".to_string();
    let mut spin = 500u64;
    let mut mode = "blocked".to_string();
    let mut projection_workspaces = false;
    let mut residual_inplace = false;
    let mut qkv_rope_workspace = false;
    let mut inplace_softmax = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--library" => library = Some(args.next().context("library")?),
            "--threads" => threads = args.next().context("threads")?.parse()?,
            "--policy" => policy = args.next().context("policy")?,
            "--spin" => spin = args.next().context("spin")?.parse()?,
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
            "--mode" => mode = args.next().context("mode")?,
            "--projection-workspaces" => projection_workspaces = true,
            "--residual-inplace" => residual_inplace = true,
            "--qkv-rope-workspace" => qkv_rope_workspace = true,
            "--inplace-softmax" => inplace_softmax = true,
            _ => bail!("unknown argument {arg}"),
        }
    }
    ensure!(
        (1..=1024).contains(&threads) && ["ACTIVE", "PASSIVE"].contains(&policy.as_str()),
        "invalid execution settings"
    );
    // This executable owns its process. Configure before creating any workers or
    // loading the dynamic runtime. The public Engine API does not do this.
    for (k, v) in recommended_environment() {
        // SAFETY: this process-owned driver sets its execution policy before any OpenMP runtime.
        unsafe { std::env::set_var(k, v) };
    }
    unsafe {
        std::env::set_var("OMP_WAIT_POLICY", &policy);
        std::env::set_var("GOMP_SPINCOUNT", spin.to_string());
        std::env::set_var("OMP_NUM_THREADS", threads.to_string());
        std::env::set_var("TOKENIZERS_PARALLELISM", "false");
    }
    let start = Instant::now();
    let mut numerical_config = None;
    let model = if let Some(library) = library {
        numerical_config = Some(unsafe {
            initialize_blas_with_options(
                &library,
                candle_mlas::BackendOptions {
                    threads,
                    ..blas_options
                },
            )?
        });
        let mut options = ModelOptions::optimized();
        options.mode = mode;
        options.projection_workspaces = projection_workspaces;
        options.residual_inplace = residual_inplace;
        options.qkv_rope_workspace = qkv_rope_workspace;
        options.inplace_softmax = inplace_softmax;
        Some(Model::load(&path, options)?)
    } else {
        None
    };
    let mut frontend = Frontend::load(&path)?;
    println!(
        "{}",
        json!({"ready":true,"load_wall_s":start.elapsed().as_secs_f64(),"native_text":true,"model_loaded":model.is_some(),"policy":policy,"spin":spin,"threads":threads,"projection_workspaces":projection_workspaces,"residual_inplace":residual_inplace,"qkv_rope_workspace":qkv_rope_workspace,"inplace_softmax":inplace_softmax,"numerical_config":numerical_config})
    );
    for line in io::stdin().lock().lines() {
        let begin = Instant::now();
        let result = (|| -> Result<Value> {
            let request: Value = serde_json::from_str(&line?)?;
            if let Some(v) = request["batch_size"].as_u64() {
                frontend.batch_size = v as usize;
            }
            if let Some(v) = request["token_budget"].as_u64() {
                frontend.token_budget = v as usize;
            }
            let operation = request["operation"].as_str().unwrap_or("prepare");
            match operation {
                "workspace_stats" => Ok(json!(model
                    .as_ref()
                    .context("model not loaded")?
                    .workspace_stats())),
                "prepare" => Ok(serde_json::to_value(
                    frontend.prepare(request["rows"].as_array().context("rows")?)?,
                )?),
                "calibrate" => Ok(serde_json::to_value(frontend.calibrated(
                    &request["raw"],
                    &request["row"],
                    &request["thresholds"],
                )?)?),
                "finish" => {
                    let plan = serde_json::from_value(request["plan"].clone())?;
                    let outputs: Vec<Logits> = serde_json::from_value(request["outputs"].clone())?;
                    Ok(json!(frontend.finish(&plan, &outputs)?))
                }
                "system_prepare" => {
                    let (req, _) = system_preparation(&request["state"], &request["questions"])?;
                    Ok(req)
                }
                "quality" => {
                    let rows = request["rows"].as_array().context("rows")?;
                    let model = model.as_ref().context("model not loaded")?;
                    let raw = frontend.predict_raw(rows, |input| {
                        execute_model(model, input).map_err(|source| rt_core::Error::Context {
                            context: "MLAS experiment".into(),
                            source: source.into(),
                        })
                    })?;
                    let decoded = raw
                        .iter()
                        .zip(rows)
                        .map(|(r, row)| frontend.calibrated(r, row, &request["thresholds"]))
                        .collect::<rt_core::Result<Vec<_>>>()?;
                    Ok(json!({"raw":raw,"decoded":decoded}))
                }
                "generic" => {
                    let rows = request["rows"].as_array().context("rows")?;
                    let model = model.as_ref().context("model not loaded")?;
                    let raw = frontend.predict_raw(rows, |input| {
                        execute_model(model, input).map_err(|source| rt_core::Error::Context {
                            context: "MLAS experiment".into(),
                            source: source.into(),
                        })
                    })?;
                    Ok(json!(raw.iter().zip(rows).map(|(r,row)|Ok(json!({"answers":frontend.calibrated(r,row,&request["thresholds"] )?,"meta":{"tokens":r["n_tokens"],"windows":r["windows"]}}))).collect::<Result<Vec<_>>>()?))
                }
                "predict" => {
                    let model = model.as_ref().context("model not loaded")?;
                    Ok(frontend.predict(&request["req"], |input| {
                        execute_model(model, input).map_err(|source| rt_core::Error::Context {
                            context: "MLAS experiment".into(),
                            source: source.into(),
                        })
                    })?)
                }
                "system_one" => {
                    let model = model.as_ref().context("model not loaded")?;
                    Ok(frontend.system_one(
                        &request["state"],
                        &request["questions"],
                        request["model"].as_str(),
                        |input| {
                            execute_model(model, input).map_err(|source| rt_core::Error::Context {
                                context: "MLAS experiment".into(),
                                source: source.into(),
                            })
                        },
                    )?)
                }
                "system_one_request" => {
                    let model = model.as_ref().context("model not loaded")?;
                    Ok(frontend.system_one_request(&request["body"], |input| {
                        execute_model(model, input).map_err(|source| rt_core::Error::Context {
                            context: "MLAS experiment".into(),
                            source: source.into(),
                        })
                    })?)
                }
                _ => bail!("unsupported operation {operation}"),
            }
        })();
        let out = result.unwrap_or_else(|e| {
            let kind = if matches!(
                e.downcast_ref::<rt_core::Error>(),
                Some(rt_core::Error::Request(_))
            ) {
                "SystemOneError"
            } else {
                "ValueError"
            };
            json!({"error":{"type":kind,"message":format!("{e:#}")}})
        });
        // Response stays the released object shape; diagnostics go to stderr.
        println!("{out}");
        io::stdout().flush()?;
        eprintln!(
            "{}",
            json!({"request_wall_s":begin.elapsed().as_secs_f64()})
        );
    }
    Ok(())
}
