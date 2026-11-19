//! Architecture constants verified against HF `config.json` and
//! `modeling_embedding_gemma2.py` (transformers main, EmbeddingGemma2).

pub const NAME: &str = "embeddinggemma2";
pub const HF_REPO: &str = "google/embeddinggemma-2";
pub const HF_REVISION: &str = "914f7f89142e33e77833254d9c9b90c3cef7303b";

pub const HIDDEN: usize = 512;
pub const INTERMEDIATE: usize = 2048;
pub const N_QUERY_HEADS: usize = 4;
pub const N_KV_SLIDING: usize = 2;
pub const N_KV_GLOBAL: usize = 1;
pub const HEAD_DIM_SLIDING: usize = 256;
pub const HEAD_DIM_GLOBAL: usize = 512;
pub const N_LAYERS: usize = 24;
pub const VOCAB: usize = 262_144;
pub const EMBED_DIM: usize = 768;
pub const PLE_WIDTH: usize = 512;
pub const GLOBAL_EVERY: usize = 6;
/// Inclusive bidirectional radius: `abs(q - kv) <= 512`.
pub const SLIDING_WINDOW: usize = 512;
/// Product context (card). `max_position_embeddings` in config.json is 262144.
pub const MAX_SEQ: usize = 8192;
pub const RMS_EPS: f32 = 1e-6;
pub const LOCAL_THETA: f32 = 10_000.0;
pub const GLOBAL_THETA: f32 = 1_000_000.0;
pub const ATTN_SCALE: f32 = 1.0;
pub const PAD_ID: u32 = 0;

pub const QUERY_PREFIX: &str = "task: search result | query: ";
pub const DOC_PREFIX: &str = "title: none | text: ";

pub const MRL_DIMS: [usize; 4] = [768, 512, 256, 128];

pub const BOS_ID: u32 = 2;
pub const EOS_ID: u32 = 1;
/// Image placeholder: `<|image>`, one `<|image|>` per soft token, `<image|>`.
pub const IMAGE_OPEN_ID: u32 = 255_999;
pub const IMAGE_TOKEN_ID: u32 = 258_880;
pub const IMAGE_CLOSE_ID: u32 = 258_882;
/// Audio placeholder: `<|audio>`, one `<|audio|>` per soft token, `<audio|>`.
pub const AUDIO_OPEN_ID: u32 = 256_000;
pub const AUDIO_TOKEN_ID: u32 = 258_881;
pub const AUDIO_CLOSE_ID: u32 = 258_883;
pub const MEDIA_IDS: [u32; 6] = [
    IMAGE_OPEN_ID,
    IMAGE_TOKEN_ID,
    IMAGE_CLOSE_ID,
    AUDIO_OPEN_ID,
    AUDIO_TOKEN_ID,
    AUDIO_CLOSE_ID,
];

pub const EMB_NAME: &str = "language_model.embed_tokens.weight";

/// Global layers are indices 5, 11, 17, 23 (`(i+1) % 6 == 0`).
pub fn is_global(layer: usize) -> bool {
    layer % GLOBAL_EVERY == GLOBAL_EVERY - 1
}

pub fn head_dim(layer: usize) -> usize {
    if is_global(layer) {
        HEAD_DIM_GLOBAL
    } else {
        HEAD_DIM_SLIDING
    }
}

pub fn n_kv(layer: usize) -> usize {
    if is_global(layer) {
        N_KV_GLOBAL
    } else {
        N_KV_SLIDING
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_layers_match_config() {
        let globals: Vec<usize> = (0..N_LAYERS).filter(|&i| is_global(i)).collect();
        assert_eq!(globals, vec![5, 11, 17, 23]);
        assert_eq!(head_dim(0), 256);
        assert_eq!(head_dim(5), 512);
        assert_eq!(n_kv(0), 2);
        assert_eq!(n_kv(5), 1);
        assert_eq!(N_QUERY_HEADS * HEAD_DIM_SLIDING, 1024);
        assert_eq!(N_QUERY_HEADS * HEAD_DIM_GLOBAL, 2048);
        assert_eq!(INTERMEDIATE, 2048);
    }
}
