//! Host-process simulation for the library.
//!
//! `serve-sim serve` is a small axum/tokio web server that loads the models
//! in-process through the public `sinteric-inference` API, the way an
//! application embeds them. `serve-sim load` drives it with concurrent
//! closed-loop clients per endpoint, probes the server's own non-inference
//! endpoints at a fixed rate, checks every response against a serial
//! reference, and prints one JSON summary. `serve-sim split` puts each model
//! in its own `serve` child process behind a proxying front.
//!
//! ```text
//! serve-sim serve --threads 4 --eg2 DIR --vela DIR --ocr DIR &
//! serve-sim load --embed 4 --vela 2 --ocr 1 --pages LIST.jsonl --duration 60
//! ```
//!
//! See `bench/serve-sim/README.md` for the scenarios and what they show.
mod inputs;
mod load;
mod proc_stats;
mod server;
mod split;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(about = "Embed the library's models in a web server and load it")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the server (models loaded in-process).
    Serve(server::ServeArgs),
    /// A front server proxying to one `serve` child process per model.
    Split(split::SplitArgs),
    /// Drive a running server and print a JSON summary.
    Load(load::LoadArgs),
}

fn main() -> anyhow::Result<()> {
    // `serve --isolate` starts its model workers by re-running this binary.
    sinteric_inference::worker::main_if_worker();
    match Cli::parse().command {
        Command::Serve(args) => server::run(args),
        Command::Split(args) => split::run(args),
        Command::Load(args) => load::run(args),
    }
}
