//! The one reader of the safetensors files every model loads. The file is
//! mapped read-only, its header is parsed and checked once, and a tensor is
//! read as a typed view of the mapping, never copied unless the caller
//! copies it.
//!
//! [`Index::parse`] holds a file to what the format promises and the models
//! rely on: the header length lies inside the file and is at most 128 MiB;
//! every tensor has a known dtype and a byte extent that matches its shape;
//! and the extents tile the data section, with no gap, overlap or trailing
//! bytes. A view checks the dtype and the shape its caller expects, and the
//! alignment of the bytes.
//!
//! What it does not check: the values (a NaN weight loads; the PP-OCR
//! assets check theirs), and the file's hash, which each model pins through
//! [`crate::assets`].

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use serde::Deserialize;
use serde_json::Value;

use crate::{checked_size, Error, Result};

/// The largest header a file may carry.
const MAX_HEADER: usize = 128 * 1024 * 1024;

/// A tensor's element type, as safetensors names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dtype {
    Bool,
    U8,
    I8,
    F16,
    Bf16,
    F32,
    I32,
    U32,
    F64,
    I64,
    U64,
}

impl Dtype {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "F32" => Self::F32,
            "I32" => Self::I32,
            "U32" => Self::U32,
            "F64" => Self::F64,
            "I64" => Self::I64,
            "U64" => Self::U64,
            _ => return None,
        })
    }

    /// Bytes per element.
    fn width(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::F16 | Self::Bf16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }
}

/// An element type a view can hold.
pub(crate) trait Element: Copy {
    const DTYPE: Dtype;
}

impl Element for f32 {
    const DTYPE: Dtype = Dtype::F32;
}

/// BF16 values, as their bit patterns.
impl Element for u16 {
    const DTYPE: Dtype = Dtype::Bf16;
}

#[derive(Deserialize)]
struct RawInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

struct Info {
    dtype: Dtype,
    shape: Vec<usize>,
    /// The tensor's bytes, as offsets from the start of the file.
    start: usize,
    end: usize,
}

/// A checked safetensors header: every tensor's dtype, shape and extent.
pub(crate) struct Index {
    tensors: BTreeMap<String, Info>,
}

impl Index {
    /// The header of the file `bytes`, checked as the module doc says.
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self> {
        let asset = |m: String| Error::model_files(format!("safetensors: {m}"));
        let len: [u8; 8] = bytes
            .get(..8)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| asset(format!("{} bytes, shorter than a header", bytes.len())))?;
        let len = u64::from_le_bytes(len);
        let data = usize::try_from(len)
            .ok()
            .filter(|&n| n <= MAX_HEADER && n <= bytes.len() - 8)
            .map(|n| n + 8)
            .ok_or_else(|| {
                asset(format!(
                    "header length {len} outside 0..={} for a {}-byte file",
                    MAX_HEADER.min(bytes.len() - 8),
                    bytes.len()
                ))
            })?;
        let header: Value = serde_json::from_slice(&bytes[8..data])?;
        let header = header
            .as_object()
            .ok_or_else(|| asset("header is not a JSON object".into()))?;
        let mut tensors = BTreeMap::new();
        for (name, v) in header {
            if name == "__metadata__" {
                continue;
            }
            let raw: RawInfo = serde_json::from_value(v.clone())?;
            let dtype = Dtype::parse(&raw.dtype)
                .ok_or_else(|| asset(format!("tensor {name}: unsupported dtype {}", raw.dtype)))?;
            let size = checked_size("tensor bytes", &raw.shape)
                .and_then(|count| checked_size("tensor bytes", &[count, dtype.width()]))
                .map_err(|_| asset(format!("tensor {name}: shape {:?} overflows", raw.shape)))?;
            let [a, b] = raw.data_offsets;
            let (start, end) = (data.checked_add(a), data.checked_add(b));
            let (Some(start), Some(end)) = (start, end) else {
                return Err(asset(format!("tensor {name}: offsets {a}..{b} overflow")));
            };
            if a > b || b - a != size || end > bytes.len() {
                return Err(asset(format!(
                    "tensor {name}: {} {:?} at {a}..{b} in a {}-byte data section",
                    raw.dtype,
                    raw.shape,
                    bytes.len() - data
                )));
            }
            tensors.insert(
                name.clone(),
                Info {
                    dtype,
                    shape: raw.shape,
                    start,
                    end,
                },
            );
        }
        let mut extents: Vec<_> = tensors.iter().map(|(n, i)| (i.start, i.end, n)).collect();
        extents.sort();
        let mut at = data;
        for (start, end, name) in extents {
            if start != at {
                return Err(asset(format!(
                    "tensor {name} starts at {start}, not at {at} where the previous one ends"
                )));
            }
            at = end;
        }
        if at != bytes.len() {
            return Err(asset(format!(
                "{} bytes after the last tensor",
                bytes.len() - at
            )));
        }
        Ok(Self { tensors })
    }

    fn info(&self, name: &str) -> Result<&Info> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::missing(format!("tensor {name}")))
    }

    /// The tensor names, sorted.
    pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// How many tensors the file holds.
    pub(crate) fn len(&self) -> usize {
        self.tensors.len()
    }

    pub(crate) fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.info(name)?.shape)
    }

    pub(crate) fn dtype(&self, name: &str) -> Result<Dtype> {
        Ok(self.info(name)?.dtype)
    }

    /// The byte range of `name` in the file, checked to be a `T` tensor of
    /// exactly `shape`.
    fn extent<T: Element>(&self, name: &str, shape: &[usize]) -> Result<(usize, usize)> {
        let i = self.info(name)?;
        if i.dtype != T::DTYPE || i.shape != shape {
            return Err(Error::model_files(format!(
                "tensor {name}: {:?} {:?}, expected {:?} {shape:?}",
                i.dtype,
                i.shape,
                T::DTYPE
            )));
        }
        Ok((i.start, i.shape.iter().product()))
    }
}

/// A mapped safetensors file and its checked [`Index`].
pub(crate) struct TensorFile {
    map: Mmap,
    index: Index,
}

impl TensorFile {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let f = File::open(path).map_err(|e| Error::from(e).at(path))?;
        // SAFETY: the file is mapped read-only; a model's files are not
        // modified while it is loaded.
        let map = unsafe { Mmap::map(&f) }.map_err(|e| Error::from(e).at(path))?;
        Self::from_map(map).map_err(|e| e.at(path))
    }

    /// A file a caller has already mapped (and checked against a pin).
    pub(crate) fn from_map(map: Mmap) -> Result<Self> {
        let index = Index::parse(&map)?;
        Ok(Self { map, index })
    }

    pub(crate) fn index(&self) -> &Index {
        &self.index
    }

    /// The tensor `name`, which must be a `T` tensor of exactly `shape`.
    pub(crate) fn view<T: Element>(&self, name: &str, shape: &[usize]) -> Result<&[T]> {
        let (start, count) = self.index.extent::<T>(name, shape)?;
        let ptr = self.map[start..].as_ptr();
        aligned::<T>(ptr, name)?;
        // SAFETY: `parse` checked that the tensor's bytes lie in the mapping
        // and number `count` elements of `T`; `aligned` checked the address
        // and the byte order; the slice borrows the mapping.
        Ok(unsafe { std::slice::from_raw_parts(ptr.cast(), count) })
    }

    /// The tensor `name` as an owned view that keeps the mapping, so only the
    /// rows a caller reads become resident.
    pub(crate) fn into_view<T: Element>(self, name: &str, shape: &[usize]) -> Result<Mapped<T>> {
        let (start, count) = self.index.extent::<T>(name, shape)?;
        aligned::<T>(self.map[start..].as_ptr(), name)?;
        Ok(Mapped {
            map: self.map,
            start,
            count,
            element: std::marker::PhantomData,
        })
    }

    /// Best effort: drop the whole pages of `name` from memory once it has
    /// been packed, so the mapping does not stay resident beside the packed
    /// copy. A dropped page is read from the file again if touched.
    pub(crate) fn release(&self, name: &str) {
        let Ok(i) = self.index.info(name) else {
            return;
        };
        let page = page_size();
        let base = self.map.as_ptr() as usize;
        let first = (base + i.start).next_multiple_of(page);
        let last = (base + i.end) & !(page - 1);
        if last > first {
            // SAFETY: the range is whole pages inside the read-only mapping.
            unsafe {
                libc::madvise(
                    first as *mut libc::c_void,
                    last - first,
                    libc::MADV_DONTNEED,
                );
            }
        }
    }
}

/// One tensor of a mapped file, owning the mapping.
pub(crate) struct Mapped<T> {
    map: Mmap,
    start: usize,
    count: usize,
    element: std::marker::PhantomData<T>,
}

impl<T: Element> Mapped<T> {
    pub(crate) fn as_slice(&self) -> &[T] {
        // SAFETY: checked as in `TensorFile::view` when this was made.
        unsafe { std::slice::from_raw_parts(self.map[self.start..].as_ptr().cast(), self.count) }
    }
}

fn aligned<T>(ptr: *const u8, name: &str) -> Result<()> {
    if !(ptr as usize).is_multiple_of(std::mem::align_of::<T>()) || cfg!(target_endian = "big") {
        return Err(Error::model_files(format!(
            "tensor {name}: misaligned or big-endian mapping"
        )));
    }
    Ok(())
}

fn page_size() -> usize {
    // SAFETY: sysconf has no preconditions.
    let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(p).ok().filter(|&p| p > 0).unwrap_or(4096)
}

#[cfg(test)]
pub(crate) mod tests {
    use serde_json::json;

    use super::*;

    /// A file holding `header`, padded to 8 bytes as writers pad it, then
    /// `body`.
    pub(crate) fn encode(header: Value, body: &[u8]) -> Vec<u8> {
        let mut header = serde_json::to_vec(&header).unwrap();
        header.resize(header.len().next_multiple_of(8), b' ');
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(body);
        bytes
    }

    #[test]
    fn truncated_and_invalid_headers_are_errors() {
        for bytes in [
            vec![],
            vec![0; 7],
            u64::MAX.to_le_bytes().to_vec(),
            [255; 16].to_vec(),
            [8u64.to_le_bytes().as_slice(), b"{}"].concat(),
            [8u64.to_le_bytes().as_slice(), b"[broken]"].concat(),
        ] {
            assert!(Index::parse(&bytes).is_err(), "{bytes:?}");
        }
    }

    #[test]
    fn extents_must_match_shapes_and_tile_the_data() {
        for (header, body) in [
            (
                json!({"x":{"dtype":"F32","shape":[usize::MAX,2],"data_offsets":[0,0]}}),
                0,
            ),
            (
                json!({"x":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}),
                0,
            ),
            (
                json!({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}),
                0,
            ),
            (
                json!({"x":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}),
                8,
            ),
            (
                json!({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,12]}}),
                12,
            ),
            (
                json!({"x":{"dtype":"Q4","shape":[2],"data_offsets":[0,1]}}),
                1,
            ),
            // A gap, an overlap and trailing bytes.
            (
                json!({
                    "a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},
                    "b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]},
                }),
                3,
            ),
            (
                json!({
                    "a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},
                    "b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]},
                }),
                3,
            ),
            (
                json!({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}),
                2,
            ),
        ] {
            let bytes = encode(header.clone(), &vec![0; body]);
            assert!(
                Index::parse(&bytes).is_err_and(|e| e.kind() == crate::ErrorKind::ModelFiles),
                "{header}"
            );
        }
        let ok = json!({
            "__metadata__": {"format": "pt"},
            "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "z": {"dtype": "BF16", "shape": [0, 4], "data_offsets": [8, 8]},
        });
        let index = Index::parse(&encode(ok, &[0; 8])).unwrap();
        assert_eq!(index.names().collect::<Vec<_>>(), ["a", "z"]);
        assert_eq!(index.dtype("z").unwrap(), Dtype::Bf16);
    }

    #[test]
    fn views_check_dtype_shape_and_name() {
        let dir = std::env::temp_dir().join(format!("sinteric-tensor-file-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("views.safetensors");
        let header = json!({
            "a": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
            "b": {"dtype": "BF16", "shape": [2], "data_offsets": [16, 20]},
        });
        let body: Vec<u8> = [1f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .chain([0x80, 0x3f, 0, 0x40])
            .collect();
        std::fs::write(&path, encode(header, &body)).unwrap();
        let file = TensorFile::open(&path).unwrap();
        assert_eq!(
            file.view::<f32>("a", &[2, 2]).unwrap(),
            [1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(file.view::<u16>("b", &[2]).unwrap(), [0x3f80, 0x4000]);
        assert!(file.view::<f32>("a", &[4]).is_err(), "wrong shape");
        assert!(file.view::<u16>("a", &[2, 2]).is_err(), "not BF16");
        assert!(file
            .view::<f32>("missing", &[1])
            .is_err_and(|e| e.kind() == crate::ErrorKind::ModelFiles));
        file.release("a");
        file.release("missing");
        let b = file.into_view::<u16>("b", &[2]).unwrap();
        assert_eq!(b.as_slice(), [0x3f80, 0x4000]);
        assert!(TensorFile::open(&dir.join("absent.safetensors"))
            .is_err_and(|e| e.kind() == crate::ErrorKind::ModelFiles));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
