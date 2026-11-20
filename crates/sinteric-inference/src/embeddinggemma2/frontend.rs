//! Gemma tokenizer + retrieval prefixes for the text tower.

use crate::{check_shape, Result, ResultContext as Context};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;

use crate::embeddinggemma2::config::{DOC_PREFIX, MAX_SEQ, MEDIA_IDS, QUERY_PREFIX, VOCAB};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptKind {
    Query,
    Document,
}

impl PromptKind {
    pub fn prefix(self) -> &'static str {
        match self {
            Self::Query => QUERY_PREFIX,
            Self::Document => DOC_PREFIX,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Batch {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u32>>,
}

pub struct Frontend {
    tokenizer: Tokenizer,
    pub max_seq: usize,
}

impl Frontend {
    pub fn load(path: &Path) -> Result<Self> {
        let tok = path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&crate::read_json(&tok)?)?)
            .map_err(|e| crate::Error::Tokenizer(e).at(&tok))?;
        Ok(Self {
            tokenizer,
            max_seq: MAX_SEQ,
        })
    }

    pub fn encode(&self, texts: &[String], kind: PromptKind) -> Result<Batch> {
        check_shape!(!texts.is_empty(), "empty encode batch");
        check_shape!(
            self.max_seq > 0 && self.max_seq <= MAX_SEQ,
            "invalid tokenizer max_seq {}",
            self.max_seq
        );
        let prefix = kind.prefix();
        let mut ids: Vec<Vec<u32>> = Vec::with_capacity(texts.len());
        for t in texts {
            let enc = self
                .tokenizer
                .encode(format!("{prefix}{t}"), true)
                .map_err(crate::Error::Tokenizer)?;
            let row: Vec<u32> = enc.get_ids().to_vec();
            check_shape!(
                row.len() <= self.max_seq,
                "EG2 sequence length {} exceeds max_seq {}",
                row.len(),
                self.max_seq
            );
            check_shape!(!row.is_empty(), "empty tokenisation");
            check_shape!(
                row.iter().all(|&id| (id as usize) < VOCAB),
                "token id out of range"
            );
            ids.push(row);
        }
        let s = ids.iter().map(Vec::len).max().context("seq")?;
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
}

impl Frontend {
    /// Token ids of `text` alone: no BOS/EOS and no prompt prefix, for
    /// assembling multimodal sequences. Rejects text that spells a media
    /// placeholder token.
    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        let ids = self
            .tokenizer
            .encode(text, false)
            .map_err(crate::Error::Tokenizer)?
            .get_ids()
            .to_vec();
        check_shape!(
            ids.iter()
                .all(|id| (*id as usize) < VOCAB && !MEDIA_IDS.contains(id)),
            "EG2 text may not contain image or audio placeholder tokens"
        );
        Ok(ids)
    }
}

impl Batch {
    /// Validate token IDs, masks and lengths before any embedding lookup.
    pub fn validate(&self, max_seq: usize) -> Result<()> {
        check_shape!(!self.input_ids.is_empty(), "EG2 empty batch");
        check_shape!(
            self.input_ids.len() == self.attention_mask.len(),
            "EG2 batch/mask row count {} != {}",
            self.input_ids.len(),
            self.attention_mask.len()
        );
        for (i, (ids, mask)) in self.input_ids.iter().zip(&self.attention_mask).enumerate() {
            check_shape!(
                !ids.is_empty() && ids.len() <= max_seq && ids.len() == mask.len(),
                "EG2 row {i}: token length {}, mask length {}, limit {max_seq}",
                ids.len(),
                mask.len()
            );
            check_shape!(
                ids.iter().all(|&id| (id as usize) < VOCAB),
                "EG2 row {i}: token ID outside vocabulary {VOCAB}"
            );
            check_shape!(
                mask.iter().all(|&v| v <= 1) && mask.contains(&1),
                "EG2 row {i}: mask must contain live tokens and values 0/1"
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
            batch(vec![vec![1; 3]], vec![vec![1; 3]]),
        ] {
            assert!(b.validate(2).is_err());
        }
    }
    #[test]
    fn invalid_tokenizer_json_returns_path_error() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/round1-input-tests/eg2-tokenizer");
        std::fs::create_dir_all(&dir).unwrap();
        for data in ["{", "[]", "{}"] {
            std::fs::write(dir.join("tokenizer.json"), data).unwrap();
            let err = match Frontend::load(&dir) {
                Ok(_) => panic!("malformed tokenizer accepted"),
                Err(e) => e,
            };
            assert!(err.to_string().contains("tokenizer.json"));
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
