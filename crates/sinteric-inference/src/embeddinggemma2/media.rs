//! CPU preprocessing for EG2 image and audio inputs, matching the
//! transformers Gemma 4 processors the EG2 checkpoint ships with
//! (`Gemma4ImageProcessor`, `Gemma4AudioFeatureExtractor`).
//!
//! Images: RGB, aspect-preserving resize to a multiple of 48 px within the
//! soft-token budget (torch-exact antialiased bicubic), 16x16 patches in
//! row-major patch order, each patch `(py, px, c)`. Audio: 16 kHz mono,
//! truncated to 30 s, 20 ms Hann frames every 10 ms, 128 HTK mel bins,
//! `log(mel + 1e-3)`.

use crate::image::ImageView;
use crate::{check_shape, Result};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::sync::OnceLock;

pub const PATCH: usize = 16;
pub const POOL: usize = 3;
/// `pixels per patch`: 16 * 16 * RGB.
pub const PATCH_PIXELS: usize = PATCH * PATCH * 3;
/// Image soft-token budgets the processor accepts (`max_soft_tokens`).
pub const IMAGE_TOKEN_BUDGETS: [usize; 5] = [70, 140, 280, 560, 1120];

/// Soft tokens per image (the processor's `max_soft_tokens`). Each image is
/// resized, keeping its aspect ratio, to fit the budget: more tokens keep
/// more detail and cost more compute. Serializes as the count (`280`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "usize", into = "usize")]
#[non_exhaustive]
pub enum ImageTokens {
    T70,
    T140,
    /// The processor default.
    #[default]
    T280,
    T560,
    T1120,
}

impl ImageTokens {
    pub const ALL: [Self; 5] = [Self::T70, Self::T140, Self::T280, Self::T560, Self::T1120];

    pub const fn count(self) -> usize {
        match self {
            Self::T70 => 70,
            Self::T140 => 140,
            Self::T280 => 280,
            Self::T560 => 560,
            Self::T1120 => 1120,
        }
    }
}

impl From<ImageTokens> for usize {
    fn from(t: ImageTokens) -> Self {
        t.count()
    }
}

impl TryFrom<usize> for ImageTokens {
    type Error = String;

    fn try_from(n: usize) -> std::result::Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|t| t.count() == n)
            .ok_or_else(|| format!("image_tokens {n} not in {IMAGE_TOKEN_BUDGETS:?}"))
    }
}

pub const SAMPLE_RATE: usize = 16_000;
/// Longer audio is truncated (30 s), as the feature extractor does.
pub const MAX_SAMPLES: usize = 480_000;
pub const MEL_BINS: usize = 128;
const FRAME: usize = 320;
const HOP: usize = 160;
const FFT: usize = 512;
const BINS: usize = FFT / 2 + 1;
const MEL_FLOOR: f64 = 1e-3;

/// `get_aspect_ratio_preserving_size`: the largest `(height, width)`,
/// multiples of 48, within `max_soft_tokens * 9` patches.
pub fn image_target_size(
    height: usize,
    width: usize,
    max_soft_tokens: usize,
) -> Result<(usize, usize)> {
    check_shape!(
        IMAGE_TOKEN_BUDGETS.contains(&max_soft_tokens),
        "image token budget {max_soft_tokens} not in {IMAGE_TOKEN_BUDGETS:?}"
    );
    check_shape!(height > 0 && width > 0, "empty image");
    let max_patches = max_soft_tokens * POOL * POOL;
    let target_px = (max_patches * PATCH * PATCH) as f64;
    let factor = (target_px / (height as f64 * width as f64)).sqrt();
    let side = POOL * PATCH;
    let mut th = (factor * height as f64 / side as f64).floor() as usize * side;
    let mut tw = (factor * width as f64 / side as f64).floor() as usize * side;
    check_shape!(th > 0 || tw > 0, "image {height}x{width} resizes to 0x0");
    let max_side = max_patches / (POOL * POOL) * side;
    if th == 0 {
        th = side;
        tw = ((width as f64 / height as f64).floor() as usize * side).min(max_side);
    } else if tw == 0 {
        tw = side;
        th = ((height as f64 / width as f64).floor() as usize * side).min(max_side);
    }
    check_shape!(
        (th * tw) as f64 <= target_px,
        "image {height}x{width} resized to {th}x{tw} exceeds {max_patches} patches"
    );
    Ok((th, tw))
}

/// Patchified image: `pw x ph` patches of [`PATCH_PIXELS`], row-major over
/// patches, values already mapped to the patch embedder's `2 * (p - 0.5)`.
pub struct ImagePatches {
    pub pw: usize,
    pub ph: usize,
    pub pixels: Vec<f32>,
}

impl ImagePatches {
    pub fn patches(&self) -> usize {
        self.pw * self.ph
    }
    pub fn soft_tokens(&self) -> usize {
        self.patches() / (POOL * POOL)
    }
}

/// Resize and patchify one image (any [`crate::image::PixelFormat`]; gray is
/// replicated to RGB, BGR is swapped).
pub fn image_patches(
    src: &ImageView<'_>,
    max_soft_tokens: usize,
    threads: usize,
) -> Result<ImagePatches> {
    let (h, w) = (src.height() as usize, src.width() as usize);
    let (th, tw) = image_target_size(h, w, max_soft_tokens)?;
    let rgb = crate::image::to_rgb(src)?;
    let rgb = if (th, tw) == (h, w) {
        rgb
    } else {
        crate::image::resize_bicubic_aa_rgb(&rgb, w, h, tw, th, threads)?
    };
    patchify(&rgb, tw, th)
}

/// `u8 -> f32 * (1/255)` in f32 (the processor's rescale), then `2 * (p - 0.5)`.
fn pixel_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            #[expect(
                clippy::excessive_precision,
                reason = "the processor's rescale factor as its config writes it"
            )]
            let p = i as f32 * 0.003_921_568_627_451_f32;
            *v = 2.0 * (p - 0.5);
        }
        t
    })
}

/// Patches of a packed RGB image whose sides are multiples of [`PATCH`].
pub fn patchify(rgb: &[u8], width: usize, height: usize) -> Result<ImagePatches> {
    check_shape!(
        width.is_multiple_of(PATCH)
            && height.is_multiple_of(PATCH)
            && rgb.len() == width * height * 3,
        "patchify {width}x{height} with {} bytes",
        rgb.len()
    );
    let (pw, ph) = (width / PATCH, height / PATCH);
    let lut = pixel_lut();
    let mut pixels = crate::try_vec::<f32>(crate::checked_size(
        "image patches",
        &[pw, ph, PATCH_PIXELS],
    )?)?;
    for py in 0..ph {
        for px in 0..pw {
            let dst = &mut pixels[(py * pw + px) * PATCH_PIXELS..][..PATCH_PIXELS];
            for y in 0..PATCH {
                let row = &rgb[((py * PATCH + y) * width + px * PATCH) * 3..][..PATCH * 3];
                for (d, &s) in dst[y * PATCH * 3..][..PATCH * 3].iter_mut().zip(row) {
                    *d = lut[s as usize];
                }
            }
        }
    }
    Ok(ImagePatches { pw, ph, pixels })
}

/// Log-mel frames of one clip: `frames x` [`MEL_BINS`], every frame valid.
pub struct AudioFeatures {
    pub frames: usize,
    pub mel: Vec<f32>,
}

impl AudioFeatures {
    /// Soft tokens after the two stride-2 subsampling convolutions.
    pub fn soft_tokens(&self) -> usize {
        audio_soft_tokens(self.frames)
    }
}

pub fn audio_soft_tokens(frames: usize) -> usize {
    frames.div_ceil(2).div_ceil(2)
}

/// Frames whose whole analysis window holds real samples (the extractor's
/// mask). With 160 zeros prepended, frame `i` ends at sample `i * 160 + 160`.
pub fn audio_frames(samples: usize) -> usize {
    let n = samples.min(MAX_SAMPLES);
    if n <= HOP {
        0
    } else {
        (n - HOP - 1) / HOP + 1
    }
}

struct MelTables {
    window: [f32; FRAME],
    /// Per mel filter: first FFT bin and its nonzero weights.
    filters: Vec<(usize, Vec<f64>)>,
    twiddle: Vec<(f64, f64)>,
    bitrev: Vec<usize>,
}

fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

/// `np.linspace`: `start + i * step`, last point exactly `stop`.
fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    let step = (stop - start) / (n - 1) as f64;
    let mut v: Vec<f64> = (0..n).map(|i| i as f64 * step + start).collect();
    v[n - 1] = stop;
    v
}

fn mel_tables() -> &'static MelTables {
    static T: OnceLock<MelTables> = OnceLock::new();
    T.get_or_init(|| {
        // Periodic Hann, `np.hanning(321)[:-1]`, cast to f32.
        let mut window = [0f32; FRAME];
        for (n, w) in window.iter_mut().enumerate() {
            *w = (0.5 - 0.5 * (2.0 * PI * n as f64 / FRAME as f64).cos()) as f32;
        }
        // transformers `mel_filter_bank(norm=None, mel_scale="htk")`.
        let mels = linspace(hz_to_mel(0.0), hz_to_mel(8000.0), MEL_BINS + 2);
        let hz: Vec<f64> = mels.iter().map(|&m| mel_to_hz(m)).collect();
        let fft_hz = linspace(0.0, (SAMPLE_RATE / 2) as f64, BINS);
        let filters = (0..MEL_BINS)
            .map(|j| {
                let w: Vec<f64> = fft_hz
                    .iter()
                    .map(|&f| {
                        let down = -(hz[j] - f) / (hz[j + 1] - hz[j]);
                        let up = (hz[j + 2] - f) / (hz[j + 2] - hz[j + 1]);
                        down.min(up).max(0.0)
                    })
                    .collect();
                let lo = w.iter().position(|&v| v > 0.0).unwrap_or(0);
                let hi = w.iter().rposition(|&v| v > 0.0).map_or(lo, |i| i + 1);
                (lo, w[lo..hi].to_vec())
            })
            .collect();
        let twiddle = (0..FFT / 2)
            .map(|k| {
                let a = -2.0 * PI * k as f64 / FFT as f64;
                (a.cos(), a.sin())
            })
            .collect();
        let bits = FFT.trailing_zeros();
        let bitrev = (0..FFT)
            .map(|i| i.reverse_bits() >> (usize::BITS - bits))
            .collect();
        MelTables {
            window,
            filters,
            twiddle,
            bitrev,
        }
    })
}

/// In-place radix-2 FFT of `FFT` complex points.
fn fft(re: &mut [f64; FFT], im: &mut [f64; FFT], t: &MelTables) {
    for i in 0..FFT {
        let j = t.bitrev[i];
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= FFT {
        let step = FFT / len;
        for start in (0..FFT).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = t.twiddle[k * step];
                let (a, b) = (start + k, start + k + len / 2);
                let xr = re[b] * wr - im[b] * wi;
                let xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
            }
        }
        len *= 2;
    }
}

/// Log-mel features of 16 kHz mono samples (longer than 30 s is truncated).
pub fn audio_features(samples: &[f32], threads: usize) -> Result<AudioFeatures> {
    let samples = &samples[..samples.len().min(MAX_SAMPLES)];
    let frames = audio_frames(samples.len());
    check_shape!(
        frames > 0,
        "audio of {} samples is shorter than one 20 ms frame at 16 kHz",
        samples.len()
    );
    check_shape!(
        samples.iter().all(|v| v.is_finite()),
        "audio samples must be finite"
    );
    let t = mel_tables();
    let mut mel =
        crate::try_vec::<f32>(crate::checked_size("audio features", &[frames, MEL_BINS])?)?;
    let mel_p = mel.as_mut_ptr() as usize;
    crate::cpu::threads::par_for(threads, frames, 16, |lo, hi| {
        let mut re = [0f64; FFT];
        let mut im = [0f64; FFT];
        let mut mag = [0f64; BINS];
        for i in lo..hi {
            re.fill(0.0);
            im.fill(0.0);
            for (n, w) in t.window.iter().enumerate() {
                // Frame `i` covers samples `[i*160 - 160, i*160 + 160)`.
                let s = (i * HOP + n).checked_sub(HOP).map_or(0.0, |k| samples[k]);
                re[n] = (s * w) as f64;
            }
            fft(&mut re, &mut im, t);
            for (k, m) in mag.iter_mut().enumerate() {
                *m = (re[k] * re[k] + im[k] * im[k]).sqrt();
            }
            // SAFETY: frames [lo, hi) are disjoint across workers.
            let row = unsafe {
                std::slice::from_raw_parts_mut((mel_p as *mut f32).add(i * MEL_BINS), MEL_BINS)
            };
            for (v, (start, w)) in row.iter_mut().zip(&t.filters) {
                let e: f64 = w.iter().zip(&mag[*start..]).map(|(a, b)| a * b).sum();
                *v = (e + MEL_FLOOR).ln() as f32;
            }
        }
    });
    Ok(AudioFeatures { frames, mel })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_sizes_match_processor() {
        // From transformers get_aspect_ratio_preserving_size (fixtures/multimodal.json).
        for (h, w, s, want) in [
            (300, 200, 280, (960, 624)),
            (64, 512, 280, (240, 2256)),
            (4000, 10, 280, (13440, 48)),
            (10, 4000, 280, (48, 13440)),
            (768, 1024, 280, (672, 912)),
            (1, 1, 280, (768, 768)),
            (48, 48, 70, (384, 384)),
            (2000, 3000, 70, (288, 480)),
            (2000, 3000, 1120, (1296, 1920)),
            (5000, 7, 70, (3360, 48)),
            (17, 9000, 140, (48, 6720)),
        ] {
            assert_eq!(image_target_size(h, w, s).unwrap(), want, "{h}x{w}@{s}");
        }
        assert!(image_target_size(10, 10, 100).is_err());
    }

    #[test]
    fn image_tokens_match_budgets() {
        assert_eq!(
            ImageTokens::ALL.map(ImageTokens::count),
            IMAGE_TOKEN_BUDGETS
        );
        assert_eq!(ImageTokens::default().count(), 280);
        assert_eq!(ImageTokens::try_from(560), Ok(ImageTokens::T560));
        assert!(ImageTokens::try_from(100).is_err());
        let json = serde_json::to_string(&ImageTokens::T1120).unwrap();
        assert_eq!(json, "1120");
        assert_eq!(
            serde_json::from_str::<ImageTokens>("70").unwrap(),
            ImageTokens::T70
        );
        assert!(serde_json::from_str::<ImageTokens>("71").is_err());
    }

    #[test]
    fn audio_frame_and_token_counts() {
        assert_eq!(audio_frames(160), 0);
        assert_eq!(audio_frames(161), 1);
        assert_eq!(audio_frames(16_000), 99);
        assert_eq!(audio_frames(37_123), 232);
        assert_eq!(audio_frames(560_000), 2999);
        assert_eq!(audio_soft_tokens(99), 25);
        assert_eq!(audio_soft_tokens(232), 58);
        assert_eq!(audio_soft_tokens(2999), 750);
        assert!(audio_features(&[0.0; 100], 1).is_err());
    }

    #[test]
    fn silence_is_mel_floor_and_tone_peaks_at_its_bin() {
        let f = audio_features(&vec![0.0; 1600], 1).unwrap();
        assert!(f
            .mel
            .iter()
            .all(|&v| (v - (1e-3f64).ln() as f32).abs() < 1e-6));
        // 1 kHz tone: energy lands in the mel bin covering 1 kHz.
        let tone: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * PI * 1000.0 * i as f64 / 16_000.0).sin() as f32 * 0.5)
            .collect();
        let f = audio_features(&tone, 2).unwrap();
        let row = &f.mel[50 * MEL_BINS..51 * MEL_BINS];
        let peak = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let center =
            |j: usize| mel_to_hz(hz_to_mel(8000.0) * (j + 1) as f64 / (MEL_BINS + 1) as f64);
        assert!(
            center(peak) > 900.0 && center(peak) < 1100.0,
            "peak bin {peak}"
        );
    }

    #[test]
    fn patch_layout_is_py_px_c() {
        let (w, h) = (32, 16);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
        let p = patchify(&rgb, w, h).unwrap();
        assert_eq!((p.pw, p.ph), (2, 1));
        // Patch 1, row 3, column 5, channel 2.
        let src = rgb[(3 * w + 16 + 5) * 3 + 2];
        assert_eq!(
            p.pixels[PATCH_PIXELS + (3 * 16 + 5) * 3 + 2],
            pixel_lut()[src as usize]
        );
        assert_eq!(pixel_lut()[0], -1.0);
        assert_eq!(pixel_lut()[255], 1.0);
    }
}
