//! Research-only Candle/MLAS Vela comparison; excluded from production dependencies.
//! Enable `bench-mlas` to build the comparison library and JSON drivers.

#[cfg(feature = "bench-mlas")]
pub mod mlas;
