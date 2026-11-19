//! Native-text Engine wrapper from the Vela native-text experiment.
//! The Frontend body lives in `rt-text`; this module binds it to the Candle model.
use anyhow::Result;
use serde_json::{json, Value};
use std::path::Path;

use super::Model;
use rt_text::{Frontend, Inputs, Logits};

pub use super::{initialize_blas, ModelOptions};

/// Recommended startup environment, including the measured bounded wait policy.
/// The host application decides whether and when to apply these process settings.
pub fn recommended_environment() -> [(&'static str, &'static str); 4] {
    [
        ("RAYON_NUM_THREADS", "1"),
        ("CANDLE_NUM_THREADS", "1"),
        ("OMP_WAIT_POLICY", "ACTIVE"),
        ("GOMP_SPINCOUNT", "500"),
    ]
}

pub fn execute_model(model: &Model, input: &Inputs) -> Result<Logits> {
    let prepared = super::Inputs {
        input_ids: input.input_ids.clone(),
        attention_mask: input.attention_mask.clone(),
        q_index: input.q_index.clone(),
        opt_index: input.opt_index.clone(),
        unit_index: input.unit_index.clone(),
        ent_index: input.ent_index.clone(),
    };
    let output = model.forward(&prepared)?;
    Ok(Logits {
        opt_logits: output.opt_logits.to_vec1::<f32>()?,
        span_logits: output.span_logits.to_vec2::<f32>()?,
    })
}

// Adapt experiment errors to the production frontend's typed executor boundary.
fn execute_text(model: &Model, input: &Inputs) -> rt_core::Result<Logits> {
    execute_model(model, input).map_err(|source| rt_core::Error::Context {
        context: "MLAS experiment".into(),
        source: source.into(),
    })
}

/// Complete native text frontend plus Candle-owned FP32 model. Loading does not
/// mutate the host environment or silently initialize/reconfigure an executor.
pub struct Engine {
    pub frontend: Frontend,
    model: Model,
}
impl Engine {
    pub fn load(path: impl AsRef<Path>, options: ModelOptions) -> Result<Self> {
        let model = Model::load(path.as_ref(), options)?;
        let frontend = Frontend::load(path)?;
        Ok(Self { frontend, model })
    }
    pub fn predict(&mut self, request: &Value) -> Result<Value> {
        let model = &self.model;
        Ok(self
            .frontend
            .predict(request, |input| execute_text(model, input))?)
    }
    pub fn system_one(
        &mut self,
        state: &Value,
        questions: &Value,
        model_name: Option<&str>,
    ) -> Result<Value> {
        let model = &self.model;
        Ok(self
            .frontend
            .system_one(state, questions, model_name, |input| {
                execute_text(model, input)
            })?)
    }
    pub fn system_one_request(&mut self, body: &Value) -> Result<Value> {
        let model = &self.model;
        Ok(self
            .frontend
            .system_one_request(body, |input| execute_text(model, input))?)
    }
    pub fn predict_generic(&mut self, rows: &[Value], thresholds: &Value) -> Result<Vec<Value>> {
        let model = &self.model;
        let raw = self
            .frontend
            .predict_raw(rows, |input| execute_text(model, input))?;
        raw.iter()
            .zip(rows)
            .map(|(r, row)| {
                Ok(json!({"answers":self.frontend.calibrated(r,row,thresholds)?,"meta":{"tokens":r["n_tokens"],"windows":r["windows"]}}))
            })
            .collect()
    }
}
