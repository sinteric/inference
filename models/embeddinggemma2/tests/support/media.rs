//! Synthetic media shared by the multimodal test and bench; must match
//! `image_pixels` / `audio_samples` in `scripts/generate_multimodal_fixtures.py`.

/// `(h, w, c)` u8, row-major, channel last.
pub fn image_pixels(h: usize, w: usize, c: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(h * w * c);
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            for ch in 0..c as i64 {
                let v = ((x * 3 + y * 5 + ch * 85) ^ ((x * y) >> 3)) + ((x / 16 + y / 16) % 2) * 64;
                out.push((v & 255) as u8);
            }
        }
    }
    out
}

/// Mono 16 kHz samples in `[-1, 1)` (Python floor division).
pub fn audio_samples(n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut state: i64 = 12345;
    for i in 0..n as i64 {
        state = (state * 1_103_515_245 + 12345) % (1 << 31);
        let noise = ((state >> 16) & 0x7fff) - 16384;
        let freq = 200 + (i / 1600) % 8 * 110;
        let saw = (i * freq * 65536 / 16000) % 65536 - 32768;
        let env = (i / 800) % 20;
        let s = (saw.div_euclid(4) + noise.div_euclid(16)) * env;
        out.push(s.div_euclid(20) as f32 / 32768.0);
    }
    out
}
