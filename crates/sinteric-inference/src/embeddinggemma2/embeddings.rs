//! What EmbeddingGemma 2 takes and returns on any hardware: its vectors and
//! the parts of a multimodal input.

use crate::image::ImageView;
use crate::{Error, Result};

#[allow(unused_imports)] // named in docs
use super::MAX_SAMPLES;

/// Flat embedding matrix: `data.len() == n * dim`, row-major.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Embeddings {
    pub dim: usize,
    pub data: Vec<f32>,
}

impl Embeddings {
    /// Rows of `dim` floats; `data.len()` must be a multiple of `dim`.
    pub fn new(dim: usize, data: Vec<f32>) -> Result<Self> {
        if dim == 0 || data.len() % dim != 0 {
            return Err(Error::invalid_input(format!(
                "{} floats do not form rows of {dim}",
                data.len()
            )));
        }
        Ok(Self { dim, data })
    }

    pub fn n(&self) -> usize {
        self.data.len().checked_div(self.dim).unwrap_or(0)
    }

    pub fn row(&self, i: usize) -> Result<&[f32]> {
        let n = self.n();
        if i >= n {
            return Err(Error::invalid_input(format!("row {i} >= n {n}")));
        }
        let start = i * self.dim;
        Ok(&self.data[start..start + self.dim])
    }

    pub fn rows(&self) -> impl Iterator<Item = &[f32]> {
        let dim = self.dim;
        self.data
            .chunks_exact(dim.max(1))
            .take(if dim == 0 { 0 } else { self.n() })
    }

    pub fn into_inner(self) -> Vec<f32> {
        self.data
    }
}

/// One piece of a multimodal input. Parts concatenate in order, as
/// Sentence Transformers joins them.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Part<'a> {
    Text(&'a str),
    /// Any [`PixelFormat`]; resized to the model's
    /// [`cpu::Config::image_tokens`](super::cpu::Config::image_tokens).
    Image(ImageView<'a>),
    /// 16 kHz mono samples; past 30 s ([`MAX_SAMPLES`]) is truncated.
    Audio(&'a [f32]),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_rows() {
        let e = Embeddings {
            dim: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        assert_eq!(e.n(), 2);
        assert_eq!(e.row(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(e.row(1).unwrap(), &[3.0, 4.0]);
        assert!(e.row(2).is_err());
        let rows: Vec<&[f32]> = e.rows().collect();
        assert_eq!(rows, vec![&[1.0, 2.0][..], &[3.0, 4.0][..]]);
        let empty = Embeddings {
            dim: 0,
            data: vec![1.],
        };
        assert!(empty.row(0).is_err());
        assert_eq!(empty.rows().count(), 0);
        assert_eq!(
            Embeddings::new(3, vec![0.; 4]).unwrap_err().kind(),
            crate::ErrorKind::InvalidInput
        );
    }
}
