//! Vela graph over the [`rt_backend::Hardware`] trait, implemented by oneDNN-CPU.
pub mod backend;
pub mod model;
pub use backend::Backend;
pub type Model = model::Model<rt_onednn::OnednnCpu>;
