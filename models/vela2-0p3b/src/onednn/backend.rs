//! Production backend is [`rt_onednn::OnednnCpu`] (direct oneDNN C API).

pub use rt_onednn::{OnednnCpu as Backend, PackedWeight};
