/// Cache-line size used for packed buffers and scratch (x86_64).
pub const CACHELINE: usize = 64;

#[inline]
pub fn align_up(n: usize, align: usize) -> crate::Result<usize> {
    if !align.is_power_of_two() {
        return Err(crate::Error::Config(format!("invalid alignment {align}")));
    }
    n.checked_add(align - 1)
        .map(|n| n & !(align - 1))
        .ok_or(crate::Error::SizeOverflow("alignment"))
}

/// 64-byte-aligned owned f32 buffer. Layout is row-major unless documented.
#[derive(Debug)]
pub struct AlignedBuf {
    ptr: *mut f32,
    len: usize,
    layout: std::alloc::Layout,
}

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    pub fn new(len: usize) -> crate::Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
                layout: std::alloc::Layout::from_size_align(1, CACHELINE)?,
            });
        }
        let layout = std::alloc::Layout::from_size_align(
            crate::checked_size("aligned buffer", &[len, std::mem::size_of::<f32>()])?,
            CACHELINE,
        )?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut f32 };
        if ptr.is_null() {
            return Err(crate::Error::Allocation {
                bytes: layout.size(),
            });
        }
        Ok(Self { ptr, len, layout })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        if self.ptr.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        if self.ptr.is_null() {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    pub fn fill(&mut self, v: f32) {
        self.as_mut_slice().fill(v);
    }

    pub fn copy_from_slice(&mut self, src: &[f32]) -> crate::Result<()> {
        if src.len() != self.len {
            return Err(crate::Error::Shape(format!(
                "copy length {} != buf {}",
                src.len(),
                self.len
            )));
        }
        self.as_mut_slice().copy_from_slice(src);
        Ok(())
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            unsafe { std::alloc::dealloc(self.ptr as *mut u8, self.layout) }
        }
    }
}

impl AlignedBuf {
    /// Fallible replacement for `Clone`; allocation failure is returned.
    pub fn try_clone(&self) -> crate::Result<Self> {
        let mut out = Self::new(self.len)?;
        out.as_mut_slice().copy_from_slice(self.as_slice());
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_cacheline() {
        assert_eq!(align_up(0, 64).unwrap(), 0);
        assert_eq!(align_up(1, 64).unwrap(), 64);
        assert_eq!(align_up(64, 64).unwrap(), 64);
        assert_eq!(align_up(65, 64).unwrap(), 128);
    }

    #[test]
    fn buf_is_aligned() {
        let b = AlignedBuf::new(17).unwrap();
        assert_eq!((b.as_ptr() as usize) % CACHELINE, 0);
        assert_eq!(b.len(), 17);
        assert!(b.as_slice().iter().all(|&x| x == 0.0));
    }
}
