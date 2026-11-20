//! Versioned packed-weight cache. Safetensors remains the source of truth.
//!
//! Header stores the *manifest* sha256, source size/mtime, oneDNN version,
//! `dnnl_get_effective_cpu_isa`, and per-tensor packed memory-descriptor
//! fingerprints (format kind, inner blocks, strides). A matching cache is
//! trusted: the 1.18 GiB file is not hashed and, on a hit, not mapped.
//! Packed layout is ISA-dependent (AMX vs AVX512_CORE_BF16 can share a byte
//! size and still be a different blocking), so a miss on ISA or layout tag
//! rebuilds from safetensors. Packed tensors are streamed on write.

use crate::{check_shape, Result, ResultContext as Context};
use memmap2::{Mmap, MmapOptions, UncheckedAdvice};
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const CACHE_VERSION: u32 = 5;
pub const CACHE_MAGIC: &[u8; 8] = b"SNTPKW05";
/// Fingerprint of a packed oneDNN memory descriptor (format/blocking).
pub const LAYOUT_TAG_LEN: usize = 256;
/// Offset of the little-endian u32 packed-entry count in the header.
/// magic8 + ver4 + sha32 + dnnl12 + isa8 + size8 + mtime8
const PACKED_COUNT_POS: u64 = 8 + 4 + 32 + 12 + 8 + 8 + 8;

/// Pinned oneDNN release baked into the cache header and file name.
pub fn pinned_dnnl() -> [u32; 3] {
    [
        onednn_sys::PINNED_MAJOR as u32,
        onednn_sys::PINNED_MINOR as u32,
        onednn_sys::PINNED_PATCH as u32,
    ]
}

/// Explicit `file` wins; otherwise the ISA-keyed sidecar next to safetensors.
pub fn cache_path_with(safetensors: &Path, cpu_isa: u64, file: Option<&Path>) -> PathBuf {
    if let Some(p) = file {
        if !p.as_os_str().is_empty() {
            return p.to_path_buf();
        }
    }
    let [maj, min, pat] = pinned_dnnl();
    let mut p = safetensors.as_os_str().to_os_string();
    p.push(format!(
        ".onednn-{maj}.{min}.{pat}.v{CACHE_VERSION}.isa{cpu_isa:x}"
    ));
    PathBuf::from(p)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileStamp {
    pub size: u64,
    pub mtime_secs: i64,
}

pub fn file_stamp(path: &Path) -> Result<FileStamp> {
    let m = std::fs::metadata(path)?;
    let mtime_secs = m
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    Ok(FileStamp {
        size: m.len(),
        mtime_secs,
    })
}

pub type LayoutTag = [u8; LAYOUT_TAG_LEN];

pub fn empty_layout_tag() -> LayoutTag {
    [0u8; LAYOUT_TAG_LEN]
}

pub struct PackedEntry {
    pub name: String,
    pub n: usize,
    pub k: usize,
    pub bytes: Vec<u8>,
    pub layout: LayoutTag,
}

pub struct RawTensor {
    pub name: String,
    pub data: Vec<f32>,
}

pub struct CacheMeta {
    pub source_sha256: String,
    pub source_size: u64,
    pub source_mtime_secs: i64,
    pub dnnl: [u32; 3],
    pub cpu_isa: u64,
}

/// Incremental writer: each packed tensor is flushed then dropped so the
/// mmapped safetensors and a second full packed copy never coexist.
pub struct CacheWriter {
    file: Option<BufWriter<File>>,
    tmp: PathBuf,
    dest: PathBuf,
    n_packed: u32,
    finished: bool,
}

impl CacheWriter {
    pub fn create(path: &Path, meta: &CacheMeta) -> Result<Self> {
        let sha = parse_sha(&meta.source_sha256)?;
        let tmp = path.with_extension("v5.tmp");
        let mut file = BufWriter::new(File::create(&tmp)?);
        file.write_all(CACHE_MAGIC)?;
        file.write_all(&CACHE_VERSION.to_le_bytes())?;
        file.write_all(&sha)?;
        file.write_all(&meta.dnnl[0].to_le_bytes())?;
        file.write_all(&meta.dnnl[1].to_le_bytes())?;
        file.write_all(&meta.dnnl[2].to_le_bytes())?;
        file.write_all(&meta.cpu_isa.to_le_bytes())?;
        file.write_all(&meta.source_size.to_le_bytes())?;
        file.write_all(&meta.source_mtime_secs.to_le_bytes())?;
        file.write_all(&0u32.to_le_bytes())?;
        Ok(Self {
            file: Some(file),
            tmp,
            dest: path.to_path_buf(),
            n_packed: 0,
            finished: false,
        })
    }

    pub fn dest(&self) -> &Path {
        &self.dest
    }

    /// Like [`create`], but returns `None` if the cache path is not writable
    /// so the caller can pack from safetensors and continue. Logs one stderr
    /// warning on failure.
    pub fn try_create(path: &Path, meta: &CacheMeta) -> Option<Self> {
        match Self::create(path, meta) {
            Ok(w) => Some(w),
            Err(e) => {
                cache_write_failed(path, &e);
                None
            }
        }
    }

    fn file(&mut self) -> Result<&mut BufWriter<File>> {
        self.file.as_mut().context("cache writer closed")
    }

    pub fn write_packed(&mut self, e: PackedEntry) -> Result<()> {
        {
            let f = self.file()?;
            write_name(f, &e.name)?;
            f.write_all(&(e.n as u64).to_le_bytes())?;
            f.write_all(&(e.k as u64).to_le_bytes())?;
            f.write_all(&e.layout)?;
            f.write_all(&(e.bytes.len() as u64).to_le_bytes())?;
            f.write_all(&e.bytes)?;
        }
        self.n_packed = self.n_packed.checked_add(1).context("packed count")?;
        Ok(())
    }

    pub fn finish(mut self, raw: &[RawTensor]) -> Result<()> {
        {
            let f = self.file()?;
            let count = u32::try_from(raw.len())
                .map_err(|_| crate::Error::SizeOverflow("raw cache tensor count"))?;
            f.write_all(&count.to_le_bytes())?;
            for e in raw {
                write_name(f, &e.name)?;
                let nbytes = e.data.len().checked_mul(4).context("raw bytes")?;
                f.write_all(&(nbytes as u64).to_le_bytes())?;
                f.write_all(le_f32_bytes(&e.data))?;
            }
            f.flush()?;
        }
        let mut inner = self
            .file
            .take()
            .context("cache writer closed")?
            .into_inner()
            .map_err(|e| e.into_error())?;
        inner.seek(SeekFrom::Start(PACKED_COUNT_POS))?;
        inner.write_all(&self.n_packed.to_le_bytes())?;
        inner.flush()?;
        drop(inner);
        std::fs::rename(&self.tmp, &self.dest)?;
        self.finished = true;
        Ok(())
    }
}

impl Drop for CacheWriter {
    fn drop(&mut self) {
        if !self.finished {
            self.file.take();
            let _ = std::fs::remove_file(&self.tmp);
        }
    }
}

/// Apply a trusted cache. The file is mmapped; packed bytes are passed as
/// slices of that mapping (restore copies into the final aligned buffer) and
/// pages are released as they are consumed.
///
/// Returns `Ok(None)` when the file is missing, not a matching v5 cache, the
/// source size/mtime stamp does not match, the manifest sha differs, the
/// oneDNN version differs, or the effective CPU ISA differs. IO errors on a
/// present but unreadable file also become `Ok(None)` so load can fall back
/// to safetensors.
pub fn apply_cache(
    path: &Path,
    expected_sha256: &str,
    source: Option<FileStamp>,
    dnnl: [u32; 3],
    cpu_isa: u64,
    mut on_packed: impl FnMut(&str, usize, usize, &[u8], &LayoutTag) -> Result<()>,
) -> Result<Option<Vec<RawTensor>>> {
    if !path.is_file() {
        return Ok(None);
    }
    let want = match parse_sha(expected_sha256) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };
    let mmap = match unsafe { MmapOptions::new().map(&file) } {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    let mut cur = Cursor {
        map: &mmap[..],
        off: 0,
    };
    if cur.take(8).ok() != Some(CACHE_MAGIC.as_slice()) {
        return Ok(None);
    }
    if cur.u32().ok() != Some(CACHE_VERSION) {
        return Ok(None);
    }
    let sha = cur.take(32)?;
    if sha != want {
        return Ok(None);
    }
    let ver = [cur.u32()?, cur.u32()?, cur.u32()?];
    if ver != dnnl {
        return Ok(None);
    }
    let cached_isa = cur.u64()?;
    if cached_isa != cpu_isa {
        return Ok(None);
    }
    let cached_size = cur.u64()?;
    let cached_mtime = cur.i64()?;
    if let Some(st) = source {
        if st.size != cached_size || st.mtime_secs != cached_mtime {
            return Ok(None);
        }
    }
    let n_packed = cur.u32()? as usize;
    for _ in 0..n_packed {
        let name = cur.name()?;
        let pn = cur.u64()? as usize;
        let pk = cur.u64()? as usize;
        let layout_bytes = cur.take(LAYOUT_TAG_LEN)?;
        let mut layout = empty_layout_tag();
        layout.copy_from_slice(layout_bytes);
        let nbytes = cur.u64()? as usize;
        let body_off = cur.off;
        let bytes = cur.take(nbytes)?;
        on_packed(&name, pn, pk, bytes, &layout)?;
        advise_unused(&mmap, body_off, nbytes);
    }
    let n_raw = cur.u32()? as usize;
    check_shape!(
        n_raw <= (mmap.len() - cur.off) / 10,
        "cache raw tensor count {n_raw} exceeds remaining file"
    );
    let mut raw = Vec::new();
    raw.try_reserve_exact(n_raw)
        .map_err(|_| crate::Error::Allocation {
            bytes: n_raw.saturating_mul(std::mem::size_of::<RawTensor>()),
        })?;
    for _ in 0..n_raw {
        let name = cur.name()?;
        let nbytes = cur.u64()? as usize;
        let body_off = cur.off;
        let bytes = cur.take(nbytes)?;
        raw.push(RawTensor {
            name,
            data: f32s_from_le(bytes)?,
        });
        advise_unused(&mmap, body_off, nbytes);
    }
    Ok(Some(raw))
}

/// One-line stderr notice; load continues from safetensors.
pub fn cache_write_failed(path: &Path, err: impl std::fmt::Display) {
    eprintln!(
        "warning: packed-weight cache not written at {}: {err}",
        path.display()
    );
}

struct Cursor<'a> {
    map: &'a [u8],
    off: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.off.checked_add(n).context("cache cursor")?;
        check_shape!(end <= self.map.len(), "cache truncated");
        let s = &self.map[self.off..end];
        self.off = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn name(&mut self) -> Result<String> {
        let nl = u16::from_le_bytes(self.take(2)?.try_into()?) as usize;
        String::from_utf8(self.take(nl)?.to_vec()).context("cache name utf8")
    }
}

fn advise_unused(map: &Mmap, offset: usize, len: usize) {
    if len == 0 {
        return;
    }
    let page = {
        let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if p > 0 {
            p as usize
        } else {
            4096
        }
    };
    let aligned = offset.saturating_add(page - 1) & !(page - 1);
    let end = (offset.saturating_add(len)) & !(page - 1);
    if end > aligned {
        let _ = unsafe {
            map.unchecked_advise_range(UncheckedAdvice::DontNeed, aligned, end - aligned)
        };
    }
}

fn write_name(f: &mut impl Write, name: &str) -> Result<()> {
    let nb = name.as_bytes();
    check_shape!(nb.len() <= u16::MAX as usize, "tensor name");
    f.write_all(&(nb.len() as u16).to_le_bytes())?;
    f.write_all(nb)?;
    Ok(())
}

fn parse_sha(hex: &str) -> Result<[u8; 32]> {
    let raw = hex::decode(hex.trim()).context("sha256 hex")?;
    check_shape!(raw.len() == 32, "sha256 length");
    let mut out = [0u8; 32];
    out.copy_from_slice(&raw);
    Ok(out)
}

fn le_f32_bytes(data: &[f32]) -> &[u8] {
    const {
        assert!(
            cfg!(target_endian = "little"),
            "packed cache is little-endian"
        );
    }
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), data.len().saturating_mul(4)) }
}

fn f32s_from_le(bytes: &[u8]) -> Result<Vec<f32>> {
    check_shape!(bytes.len() % 4 == 0, "raw tensor bytes");
    let n = bytes.len() / 4;
    let mut data = crate::try_vec::<f32>(n)?;
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.as_mut_ptr().cast(), bytes.len());
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_ISA: u64 = 0xe7;

    fn write_cache(
        path: &Path,
        meta: &CacheMeta,
        packed: &[PackedEntry],
        raw: &[RawTensor],
    ) -> Result<()> {
        let mut w = CacheWriter::create(path, meta)?;
        for e in packed {
            w.write_packed(PackedEntry {
                name: e.name.clone(),
                n: e.n,
                k: e.k,
                bytes: e.bytes.clone(),
                layout: e.layout,
            })?;
        }
        w.finish(raw)
    }

    fn sha256_bytes(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        hex::encode(Sha256::digest(bytes))
    }

    fn tag(fill: u8) -> LayoutTag {
        let mut t = empty_layout_tag();
        t[0] = fill;
        t[1] = 0x4d;
        t
    }

    fn sample_meta(sha: &str, size: u64, mtime: i64) -> CacheMeta {
        CacheMeta {
            source_sha256: sha.to_string(),
            source_size: size,
            source_mtime_secs: mtime,
            dnnl: pinned_dnnl(),
            cpu_isa: TEST_ISA,
        }
    }

    fn apply(
        path: &Path,
        sha: &str,
        stamp: Option<FileStamp>,
        dnnl: [u32; 3],
        isa: u64,
        mut on_packed: impl FnMut(&str, usize, usize, &[u8], &LayoutTag) -> Result<()>,
    ) -> Option<Vec<RawTensor>> {
        apply_cache(path, sha, stamp, dnnl, isa, |n, a, b, c, t| {
            on_packed(n, a, b, c, t)
        })
        .unwrap()
    }

    #[test]
    fn corrupt_matching_cache_returns_error_without_allocation() {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/round1-corrupt-cache");
        let sha = sha256_bytes(b"corrupt");
        write_cache(&p, &sample_meta(&sha, 0, 0), &[], &[]).unwrap();
        let original = std::fs::read(&p).unwrap();
        for end in [16usize, original.len() - 1] {
            std::fs::write(&p, &original[..end]).unwrap();
            assert!(
                apply_cache(&p, &sha, None, pinned_dnnl(), TEST_ISA, |_, _, _, _, _| Ok(
                    ()
                ))
                .is_err()
            );
        }
        let mut corrupt = original;
        let len = corrupt.len();
        corrupt[len - 4..].copy_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&p, corrupt).unwrap();
        assert!(
            apply_cache(&p, &sha, None, pinned_dnnl(), TEST_ISA, |_, _, _, _, _| Ok(
                ()
            ))
            .is_err()
        );
        std::fs::remove_file(p).unwrap();
    }
    #[test]
    fn roundtrip_packed_and_raw_and_mismatch() {
        let dir = std::env::temp_dir().join("sinteric-pk-cache-v5-test");
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("w.bin.onednn-3.13.4.v5");
        let sha = "00".repeat(32);
        let layout = tag(7);
        write_cache(
            &p,
            &sample_meta(&sha, 1234, 99),
            &[PackedEntry {
                name: "a.weight".into(),
                n: 2,
                k: 3,
                bytes: vec![1, 2, 3, 4],
                layout,
            }],
            &[RawTensor {
                name: "emb".into(),
                data: vec![1.0, 2.0],
            }],
        )
        .unwrap();
        let stamp = FileStamp {
            size: 1234,
            mtime_secs: 99,
        };
        let mut packed = Vec::new();
        let raw = apply(
            &p,
            &sha,
            Some(stamp),
            pinned_dnnl(),
            TEST_ISA,
            |name, n, k, bytes, layout| {
                packed.push(PackedEntry {
                    name: name.to_string(),
                    n,
                    k,
                    bytes: bytes.to_vec(),
                    layout: *layout,
                });
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(packed[0].name, "a.weight");
        assert_eq!(packed[0].bytes, vec![1, 2, 3, 4]);
        assert_eq!(packed[0].layout, layout);
        assert_eq!(raw[0].name, "emb");
        assert_eq!(raw[0].data, vec![1.0, 2.0]);
        assert!(apply(
            &p,
            &"ff".repeat(32),
            None,
            pinned_dnnl(),
            TEST_ISA,
            |_, _, _, _, _| Ok(())
        )
        .is_none());
        assert!(apply(
            &p,
            &sha,
            Some(FileStamp {
                size: 9999,
                mtime_secs: 99,
            }),
            pinned_dnnl(),
            TEST_ISA,
            |_, _, _, _, _| Ok(())
        )
        .is_none());
        assert!(apply(
            &p,
            &sha,
            Some(FileStamp {
                size: 1234,
                mtime_secs: 1,
            }),
            pinned_dnnl(),
            TEST_ISA,
            |_, _, _, _, _| Ok(())
        )
        .is_none());
        assert!(apply(
            &p,
            &sha,
            Some(stamp),
            [9, 9, 9],
            TEST_ISA,
            |_, _, _, _, _| Ok(())
        )
        .is_none());
        assert!(apply(
            &p,
            &sha,
            Some(stamp),
            pinned_dnnl(),
            0xfef,
            |_, _, _, _, _| Ok(())
        )
        .is_none());
    }

    #[test]
    fn incremental_writer_does_not_need_all_packed_resident() {
        let dir = std::env::temp_dir().join("sinteric-pk-cache-v5-stream");
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("stream.bin.onednn-3.13.4.v5");
        let sha = "11".repeat(32);
        let mut w = CacheWriter::create(&p, &sample_meta(&sha, 99, 7)).unwrap();
        w.write_packed(PackedEntry {
            name: "left".into(),
            n: 1,
            k: 2,
            bytes: vec![9, 8, 7, 6],
            layout: tag(1),
        })
        .unwrap();
        w.write_packed(PackedEntry {
            name: "right".into(),
            n: 3,
            k: 4,
            bytes: vec![5, 4, 3, 2],
            layout: tag(2),
        })
        .unwrap();
        w.finish(&[RawTensor {
            name: "bias".into(),
            data: vec![0.5],
        }])
        .unwrap();
        let mut names = Vec::new();
        let raw = apply(
            &p,
            &sha,
            Some(FileStamp {
                size: 99,
                mtime_secs: 7,
            }),
            pinned_dnnl(),
            TEST_ISA,
            |name, _, _, _, _| {
                names.push(name.to_string());
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(names, ["left", "right"]);
        assert_eq!(raw[0].data, vec![0.5]);
    }

    #[test]
    fn v1_v2_v3_v4_magic_are_ignored() {
        let dir = std::env::temp_dir().join("sinteric-pk-cache-old-ignore");
        let _ = std::fs::create_dir_all(&dir);
        for (name, magic) in [
            ("old.v1", &b"SNTPKW01"[..]),
            ("old.v2", &b"SNTPKW02"[..]),
            ("old.v3", &b"SNTPKW03"[..]),
            ("old.v4", &b"SNTPKW04"[..]),
        ] {
            let p = dir.join(name);
            std::fs::write(&p, magic).unwrap();
            assert!(apply(
                &p,
                &"00".repeat(32),
                None,
                pinned_dnnl(),
                TEST_ISA,
                |_, _, _, _, _| Ok(())
            )
            .is_none());
        }
    }

    #[test]
    fn unwritable_cache_path_returns_none() {
        let dir = std::env::temp_dir().join("sinteric-pk-cache-unwritable");
        let _ = std::fs::create_dir_all(&dir);
        let not_dir = dir.join("not-a-directory");
        std::fs::write(&not_dir, b"x").unwrap();
        let p = not_dir.join("model.safetensors.onednn-3.13.4.v5");
        let sha = "22".repeat(32);
        assert!(CacheWriter::try_create(&p, &sample_meta(&sha, 1, 0)).is_none());
    }

    #[test]
    fn cache_path_embeds_onednn_version_and_isa() {
        let p = cache_path_with(Path::new("/tmp/model.safetensors"), 0xe7, None);
        let name = p.file_name().unwrap().to_str().unwrap();
        assert!(name.contains("onednn-3.13.4.v5"));
        assert!(name.contains("isae7"));
        let explicit = cache_path_with(
            Path::new("/tmp/model.safetensors"),
            0xe7,
            Some(Path::new("/tmp/custom.cache")),
        );
        assert_eq!(explicit, PathBuf::from("/tmp/custom.cache"));
    }
}
