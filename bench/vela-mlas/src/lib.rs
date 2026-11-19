//! Research-only Candle/MLAS Vela comparison; excluded from the library crates' dependencies.
//! Enable `bench-mlas` to build the comparison library and JSON drivers.

#[cfg(feature = "bench-mlas")]
pub mod mlas;
