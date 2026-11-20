//! Architecture constants verified against HF `config.json` and
//! `modeling_embedding_gemma2.py` (transformers main, EmbeddingGemma2).

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

/// Query rows per attention chunk on AMX (`tdpbf16ps` works in 16-row
/// tiles), measured on Emerald Rapids.
const ATTN_TILE_AMX: usize = 256;
/// Query rows per attention chunk on AVX512_CORE_BF16 and any other ISA:
/// 256 beat 64 on the E6 guest.
const ATTN_TILE_AVX512_BF16: usize = 256;

/// The attention tile EG2 runs at on `isa`, from a table measured per ISA.
///
/// It is not derived from the reported cache sizes: guests often report a
/// small L2 (512 KiB on E6), and a cache-budget formula then picked 64, which
/// was 3–22% slower than 256 on AVX512_CORE_BF16. Shape does not change it
/// either: [`crate::cpu::onednn::clamp_attn_tile`] runs a sequence in one
/// chunk when its score matrix fits and folds a short last window, which
/// beat shrinking the tile to hide a remainder. `Config::with_tile`
/// overrides it.
pub(crate) fn attn_tile(isa: u64) -> usize {
    if crate::cpu::onednn::isa_has_amx(isa) {
        ATTN_TILE_AMX
    } else {
        ATTN_TILE_AVX512_BF16
    }
}

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

    #[test]
    fn attention_tile_is_the_measured_table_on_every_isa() {
        const AMX: u64 = 0xfef;
        const AVX512_BF16: u64 = 0xe7;
        assert_eq!(attn_tile(AMX), 256);
        assert_eq!(attn_tile(AVX512_BF16), 256);
        assert_eq!(attn_tile(0), 256);
    }
}
