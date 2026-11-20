//! Everything that runs the models on x86-64 CPUs: the runtime, the thread
//! pool, oneDNN and the kernels oneDNN does not cover.

pub(crate) mod kernels;
pub(crate) mod onednn;
pub(crate) mod runtime;
pub(crate) mod threads;
