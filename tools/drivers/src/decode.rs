//! Image decoding for the OCR gate. The library takes decoded pixels; both
//! JSONL drivers (`ppocr-onednn`, `ppocr-ort`) decode through this one
//! function, so the library and the ONNX Runtime reference see the same bytes.

use std::path::Path;

use anyhow::{anyhow, bail, ensure, Context, Result};
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

/// Bytes of a `w x h` image with `channels` per pixel, or an error.
fn image_bytes(w: usize, h: usize, channels: usize) -> Result<usize> {
    w.checked_mul(h)
        .and_then(|p| p.checked_mul(channels))
        .context("image size overflows")
}

/// Decodes a PNG or JPEG (sniffed from the file signature, not the extension)
/// into packed BGR8. Returns `(bgr, width, height)`.
pub fn load_bgr(path: &Path, max_pixels: usize) -> Result<(Vec<u8>, u32, u32)> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    decode_bgr(&bytes, max_pixels).with_context(|| path.display().to_string())
}

/// [`load_bgr`] for an encoded image already in memory (an upload body).
pub fn decode_bgr(bytes: &[u8], max_pixels: usize) -> Result<(Vec<u8>, u32, u32)> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        png_bgr(bytes, max_pixels)
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        jpeg_bgr(bytes, max_pixels)
    } else {
        bail!("unsupported image format (PNG or JPEG only)")
    }
}

fn png_bgr(bytes: &[u8], max_pixels: usize) -> Result<(Vec<u8>, u32, u32)> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    let (w, h) = (reader.info().width, reader.info().height);
    let pixels = image_bytes(w as usize, h as usize, 1)?;
    ensure!(pixels <= max_pixels, "PNG exceeds max_pixels {max_pixels}");
    let mut data = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut data)?;
    let data = &data[..info.buffer_size()];
    let channels = info.color_type.samples();
    if info.bit_depth != png::BitDepth::Eight
        || !matches!(
            info.color_type,
            png::ColorType::Rgb
                | png::ColorType::Rgba
                | png::ColorType::Grayscale
                | png::ColorType::GrayscaleAlpha
        )
    {
        bail!("unsupported decoded PNG color type");
    }
    let mut bgr = vec![0; image_bytes(pixels, 1, 3)?];
    for (dst, src) in bgr
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(data.chunks_exact(channels))
    {
        match info.color_type {
            png::ColorType::Rgb | png::ColorType::Rgba => {
                dst.copy_from_slice(&[src[2], src[1], src[0]])
            }
            _ => dst.fill(src[0]),
        }
    }
    Ok((bgr, w, h))
}

fn jpeg_bgr(bytes: &[u8], max_pixels: usize) -> Result<(Vec<u8>, u32, u32)> {
    // zune-jpeg converts grayscale, YCbCr, CMYK and YCCK to RGB (it has no
    // Luma->BGR mapping); swap to BGR in place afterwards.
    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    decoder.decode_headers().map_err(|e| anyhow!("{e:?}"))?;
    let info = decoder.info().context("JPEG headers not decoded")?;
    let (w, h) = (info.width as usize, info.height as usize);
    let pixels = image_bytes(w, h, 1)?;
    ensure!(pixels <= max_pixels, "JPEG exceeds max_pixels {max_pixels}");
    let size = image_bytes(w, h, 3)?;
    ensure!(
        decoder.output_buffer_size() == Some(size),
        "JPEG decoder did not produce RGB8"
    );
    let mut bgr = vec![0; size];
    decoder
        .decode_into(&mut bgr)
        .map_err(|e| anyhow!("{e:?}"))?;
    for p in bgr.as_chunks_mut::<3>().0 {
        p.swap(0, 2);
    }
    Ok((bgr, w as u32, h as u32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;

    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("drivers-decode-{}-{name}", std::process::id()))
    }

    #[test]
    fn png_and_jpeg_decode_to_bgr() {
        // 2x1 RGB PNG: red, blue.
        let path = scratch("rb.png");
        let mut encoder = png::Encoder::new(File::create(&path).unwrap(), 2, 1);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .unwrap()
            .write_image_data(&[255, 0, 0, 0, 0, 255])
            .unwrap();
        let (bgr, w, h) = load_bgr(&path, usize::MAX).unwrap();
        assert_eq!((w, h), (2, 1));
        assert_eq!(bgr, [0, 0, 255, 255, 0, 0]);
        assert!(format!("{:#}", load_bgr(&path, 1).unwrap_err()).contains("max_pixels"));
        std::fs::remove_file(&path).unwrap();

        // Minimal 8x8 baseline grayscale JPEG, uniform mid-gray: every BGR
        // channel equals the luma value.
        let path = scratch("gray.jpg");
        std::fs::write(&path, GRAY_8X8_JPEG).unwrap();
        let (bgr, w, h) = load_bgr(&path, usize::MAX).unwrap();
        assert_eq!((w, h), (8, 8));
        assert_eq!(bgr.len(), 8 * 8 * 3);
        assert!(bgr.iter().all(|&v| v == bgr[0]));
        assert!((120..=136).contains(&bgr[0]), "{}", bgr[0]);
        assert!(format!("{:#}", load_bgr(&path, 63).unwrap_err()).contains("max_pixels"));
        std::fs::remove_file(&path).unwrap();

        assert_eq!(
            decode_bgr(GRAY_8X8_JPEG, usize::MAX).unwrap().0,
            bgr,
            "in-memory decode matches the file path"
        );

        let path = scratch("junk.bin");
        std::fs::write(&path, b"GIF89a").unwrap();
        assert!(load_bgr(&path, usize::MAX).is_err());
        assert!(decode_bgr(b"GIF89a", usize::MAX).is_err());
        std::fs::remove_file(&path).unwrap();
    }

    // Baseline 8x8 single-component JPEG (all samples 128): SOI, DQT (all 1),
    // SOF0, DHT DC/AC tables with one code each, SOS, one block (DC diff 0,
    // EOB), EOI.
    const GRAY_8X8_JPEG: &[u8] = &[
        0xff, 0xd8, // SOI
        0xff, 0xdb, 0x00, 0x43, 0x00, // DQT, table 0
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, //
        0xff, 0xc0, 0x00, 0x0b, 0x08, 0x00, 0x08, 0x00, 0x08, 0x01, 0x01, 0x11,
        0x00, // SOF0 8x8, 1 comp
        0xff, 0xc4, 0x00, 0x14, 0x00, // DHT DC0: one 1-bit code -> category 0
        0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00, //
        0xff, 0xc4, 0x00, 0x14, 0x10, // DHT AC0: one 1-bit code -> EOB
        0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00, //
        0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3f, 0x00, // SOS
        0x00, // DC code '0' (diff 0), AC code '0' (EOB), padding zeros
        0xff, 0xd9, // EOI
    ];
}
