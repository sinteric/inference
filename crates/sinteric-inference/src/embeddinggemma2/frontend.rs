//! Gemma tokenizer + retrieval prefixes for the text tower.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::embeddinggemma2::config::{DOC_PREFIX, EOS_ID, MAX_SEQ, MEDIA_IDS, QUERY_PREFIX, VOCAB};
use crate::{check_input, check_shape, Error, ErrorKind, Result};

/// Which retrieval prompt a text is embedded under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptKind {
    /// `task: search result | query: `.
    Query,
    /// `title: none | text: `.
    Document,
}

impl PromptKind {
    /// The prompt text put before the input.
    pub fn prefix(self) -> &'static str {
        match self {
            Self::Query => QUERY_PREFIX,
            Self::Document => DOC_PREFIX,
        }
    }
}

/// Token ids and masks for one batch, every row padded to the longest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Batch {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u32>>,
}

/// EmbeddingGemma 2's Gemma tokenizer with its retrieval prompts.
///
/// A text whose tokens, prompt and `<bos>`/`<eos>` included, pass the window
/// is refused ([`crate::ErrorKind::TooLarge`]), or, when the model was loaded
/// to truncate, keeps its first `window - 1` tokens and is closed with
/// `<eos>`, as Sentence Transformers truncates.
pub struct Tokenizer {
    tokenizer: tokenizers::Tokenizer,
    window: usize,
    truncate: bool,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("window", &self.window)
            .field("truncate", &self.truncate)
            .finish_non_exhaustive()
    }
}

impl Tokenizer {
    /// `tokenizer.json` from `model_dir`, with the model's full window and no
    /// truncation.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let json = serde_json::to_vec(&crate::read_json(&path)?)?;
        let mut tokenizer = tokenizers::Tokenizer::from_bytes(json)
            .map_err(|e| Error::with_source(ErrorKind::ModelFiles, "tokenizer", e).at(&path))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| Error::with_source(ErrorKind::ModelFiles, "tokenizer", e).at(&path))?;
        Ok(Self {
            tokenizer,
            window: MAX_SEQ,
            truncate: false,
        })
    }

    /// The longest sequence, in tokens, this tokenizer produces.
    pub fn window(&self) -> usize {
        self.window
    }

    /// Whether a text past the window is cut rather than refused.
    pub fn truncates(&self) -> bool {
        self.truncate
    }

    pub(crate) fn set_window(&mut self, window: usize, truncate: bool) {
        self.window = window;
        self.truncate = truncate;
    }

    /// Tokenize `texts` under `kind`'s prompt into one padded batch.
    pub fn encode(&self, texts: &[impl AsRef<str>], kind: PromptKind) -> Result<Batch> {
        check_input!(!texts.is_empty(), "empty batch");
        check_shape!(
            (1..=MAX_SEQ).contains(&self.window) && (self.window > 1 || !self.truncate),
            "invalid tokenizer window {}",
            self.window
        );
        let prefix = kind.prefix();
        let mut ids: Vec<Vec<u32>> = Vec::with_capacity(texts.len());
        for t in texts {
            let enc = self
                .tokenizer
                .encode(format!("{prefix}{}", t.as_ref()), true)
                .map_err(|e| Error::with_source(ErrorKind::InvalidInput, "tokenize", e))?;
            let mut row: Vec<u32> = enc.get_ids().to_vec();
            if row.len() > self.window {
                if !self.truncate {
                    return Err(Error::too_large(format!(
                        "{} tokens exceed the window of {}",
                        row.len(),
                        self.window
                    )));
                }
                row.truncate(self.window - 1);
                row.push(EOS_ID);
            }
            check_shape!(!row.is_empty(), "empty tokenisation");
            check_shape!(
                row.iter().all(|&id| (id as usize) < VOCAB),
                "token id out of range"
            );
            ids.push(row);
        }
        let s = ids.iter().map(Vec::len).max().unwrap_or(0);
        let mut masks = Vec::with_capacity(ids.len());
        for row in &mut ids {
            let mut mask = vec![1u32; row.len()];
            if row.len() < s {
                mask.resize(s, 0);
                row.resize(s, 0);
            }
            masks.push(mask);
        }
        Ok(Batch {
            input_ids: ids,
            attention_mask: masks,
        })
    }

    /// Token ids of `text` alone: no `<bos>`/`<eos>` and no prompt, for
    /// assembling multimodal sequences. Refuses text that spells a media
    /// placeholder token.
    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        let ids = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::with_source(ErrorKind::InvalidInput, "tokenize", e))?
            .get_ids()
            .to_vec();
        check_input!(
            ids.iter()
                .all(|id| (*id as usize) < VOCAB && !MEDIA_IDS.contains(id)),
            "text may not contain image or audio placeholder tokens"
        );
        Ok(ids)
    }
}

impl Batch {
    /// Validate token IDs, masks and lengths before any embedding lookup.
    pub fn validate(&self, max_seq: usize) -> Result<()> {
        check_input!(!self.input_ids.is_empty(), "empty batch");
        check_input!(
            self.input_ids.len() == self.attention_mask.len(),
            "batch/mask row count {} != {}",
            self.input_ids.len(),
            self.attention_mask.len()
        );
        for (i, (ids, mask)) in self.input_ids.iter().zip(&self.attention_mask).enumerate() {
            if ids.len() > max_seq {
                return Err(Error::too_large(format!(
                    "row {i}: {} tokens exceed the window of {max_seq}",
                    ids.len()
                )));
            }
            check_input!(
                !ids.is_empty() && ids.len() == mask.len(),
                "row {i}: token length {}, mask length {}",
                ids.len(),
                mask.len()
            );
            check_input!(
                ids.iter().all(|&id| (id as usize) < VOCAB),
                "row {i}: token ID outside vocabulary {VOCAB}"
            );
            check_input!(
                mask.iter().all(|&v| v <= 1) && mask.contains(&1),
                "row {i}: mask must contain live tokens and values 0/1"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes() {
        assert!(PromptKind::Query
            .prefix()
            .starts_with("task: search result"));
        assert!(PromptKind::Document.prefix().starts_with("title: none"));
        assert!(PromptKind::Query.prefix().ends_with(' '));
        assert!(PromptKind::Document.prefix().ends_with(' '));
    }
}

#[cfg(test)]
mod bad_input_tests {
    use super::*;
    #[test]
    fn invalid_batches_are_rejected() {
        let batch = |ids, mask| Batch {
            input_ids: ids,
            attention_mask: mask,
        };
        for b in [
            batch(vec![], vec![]),
            batch(vec![vec![]], vec![vec![]]),
            batch(vec![vec![1]], vec![]),
            batch(vec![vec![VOCAB as u32]], vec![vec![1]]),
            batch(vec![vec![1]], vec![vec![0]]),
            batch(vec![vec![1]], vec![vec![2]]),
        ] {
            assert_eq!(b.validate(2).unwrap_err().kind(), ErrorKind::InvalidInput);
        }
        let long = batch(vec![vec![1; 3]], vec![vec![1; 3]]);
        assert_eq!(long.validate(2).unwrap_err().kind(), ErrorKind::TooLarge);
    }
    #[test]
    fn invalid_tokenizer_json_returns_path_error() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/round1-input-tests/eg2-tokenizer");
        std::fs::create_dir_all(&dir).unwrap();
        for data in ["{", "[]", "{}"] {
            std::fs::write(dir.join("tokenizer.json"), data).unwrap();
            let err = match Tokenizer::load(&dir) {
                Ok(_) => panic!("malformed tokenizer accepted"),
                Err(e) => e,
            };
            assert!(err.to_string().contains("tokenizer.json"));
            assert_eq!(err.kind(), ErrorKind::ModelFiles);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
