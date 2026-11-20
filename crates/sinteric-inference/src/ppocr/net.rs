//! Tensor boundary shared by compiled OCR nets and later reference adapters.
use crate::{checked_size, Error, Result};

#[derive(Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}
impl Tensor {
    pub fn try_new(shape: Vec<usize>, values: Vec<f32>) -> Result<Self> {
        validate(&shape, values.len())?;
        Ok(Self { shape, values })
    }
}
pub(crate) fn validate(shape: &[usize], len: usize) -> Result<()> {
    if shape.is_empty() || shape.len() > 12 || shape.contains(&0) {
        return Err(Error::internal(format!(
            "invalid OCR tensor shape {shape:?}"
        )));
    }
    let elements = checked_size("OCR input", shape)?;
    if elements > isize::MAX as usize / 4 || elements != len {
        return Err(Error::internal("OCR input extent mismatch"));
    }
    Ok(())
}
pub trait OcrNet: Send + Sync {
    fn forward(&self, shape: &[usize], input: &[f32]) -> Result<Tensor>;
}
