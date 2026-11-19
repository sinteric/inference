//! Scratch leases. Idea carried from the Vela Candle experiment's workspace: one
//! process-owned arena, sub-allocated per forward, never grown past a cap.

use crate::{align_up, AlignedBuf, Error, Result, CACHELINE, DEFAULT_SCRATCH_CAP_BYTES};

/// A bump-allocated scratch region. Not thread-safe; one per worker.
pub struct Workspace {
    buf: AlignedBuf,
    offset: usize,
    cap_elems: usize,
}

/// A view into [`Workspace`]. The lease is a slice; dropping it does not
/// reclaim — call [`Workspace::reset`] between forwards.
pub struct ScratchLease<'a> {
    data: &'a mut [f32],
}

impl<'a> ScratchLease<'a> {
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.data
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        self.data
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_mut_ptr()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Workspace {
    pub fn with_cap_bytes(cap_bytes: usize) -> Result<Self> {
        let cap_elems = cap_bytes / std::mem::size_of::<f32>();
        Ok(Self {
            buf: AlignedBuf::new(cap_elems)?,
            offset: 0,
            cap_elems,
        })
    }

    pub fn new() -> Result<Self> {
        Self::with_cap_bytes(DEFAULT_SCRATCH_CAP_BYTES)
    }

    pub fn cap_bytes(&self) -> usize {
        self.cap_elems * std::mem::size_of::<f32>()
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Allocate `n` f32s, 64-byte aligned within the arena.
    pub fn lease(&mut self, n: usize) -> Result<ScratchLease<'_>> {
        let aligned = align_up(self.offset, CACHELINE / std::mem::size_of::<f32>())?;
        let end = aligned
            .checked_add(n)
            .ok_or(Error::SizeOverflow("workspace lease"))?;
        if end > self.cap_elems {
            return Err(Error::Message(format!(
                "scratch cap {} elems exceeded (need {end})",
                self.cap_elems
            )));
        }
        self.offset = end;
        let slice = &mut self.buf.as_mut_slice()[aligned..end];
        Ok(ScratchLease { data: slice })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_and_reset() {
        let mut ws = Workspace::with_cap_bytes(4096).unwrap();
        {
            let mut a = ws.lease(8).unwrap();
            a.as_mut_slice()[0] = 1.0;
        }
        ws.reset();
        let b = ws.lease(8).unwrap();
        // reset does not zero; it only rewinds the bump pointer.
        assert_eq!(b.len(), 8);
    }

    #[test]
    fn cap_is_hard() {
        let mut ws = Workspace::with_cap_bytes(64).unwrap();
        assert!(ws.lease(1000).is_err());
    }
}
