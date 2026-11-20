//! Running the models on x86-64 CPUs: the [`Runtime`] every model loads
//! from, its [`Options`], the ISA it runs on, and what a model reports about
//! its load and its backend. oneDNN, the thread pool and the kernels oneDNN
//! does not cover are private.

pub(crate) mod kernels;
pub(crate) mod onednn;
pub(crate) mod runtime;
pub(crate) mod threads;

pub use runtime::{
    AmxHealth, BackendOverrides, BackendStats, CacheConfig, CacheWrite, Isa, IsaCap, LoadReport,
    Options, PackedWeights, Runtime, WeightSource,
};
