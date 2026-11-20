//! Parallel loops over index ranges and over the rows of a buffer. GEMMs and
//! glue share the pinned pool in `crate::cpu::threads` (oneDNN THREADPOOL
//! runtime). Tests without a pool fall back to `std::thread::scope`.
//!
//! A kernel reads its inputs as plain shared slices and writes its output
//! through [`for_rows`] or [`for_rows2`], which hand each worker its own rows
//! as `&mut [T]`. Only a kernel whose rows do not form one contiguous range
//! per worker (a scatter) takes them one by one from [`RowsMut`], and says at
//! the call why no two workers take the same row.

use std::marker::PhantomData;

/// Calls `f(lo, hi)` across the pool over ranges that are disjoint and
/// together cover `[0, n)`, each exactly once, in chunks of at least `grain`.
pub fn for_ranges(nthreads: usize, n: usize, grain: usize, f: impl Fn(usize, usize) + Sync) {
    crate::cpu::threads::par_for(nthreads, n, grain, f);
}

/// Calls `f(lo, hi)` across the pool over row ranges that are disjoint and
/// together cover `[0, rows)`, each exactly once, and that start on a
/// multiple of `tile_rows`. `tile_rows = 1` keeps the pool's grain exactly.
pub fn for_row_tiles(
    nthreads: usize,
    rows: usize,
    tile_rows: usize,
    grain: usize,
    f: impl Fn(usize, usize) + Sync,
) -> crate::Result<()> {
    crate::check_shape!(tile_rows > 0, "row tile size must be positive");
    if tile_rows == 1 {
        for_ranges(nthreads, rows, grain, f);
    } else {
        for_ranges(
            nthreads,
            rows.div_ceil(tile_rows),
            grain.div_ceil(tile_rows),
            |lo, hi| {
                f(lo * tile_rows, hi.saturating_mul(tile_rows).min(rows));
            },
        );
    }
    Ok(())
}

/// Calls `f(lo, hi, rows)` across the pool, where `rows` is rows `[lo, hi)`
/// of `out` and a row is `width` elements. Every row goes to exactly one
/// call.
///
/// # Panics
/// If `width` is zero or does not divide `out.len()`.
pub fn for_rows<T: Send>(
    nthreads: usize,
    grain: usize,
    out: &mut [T],
    width: usize,
    f: impl Fn(usize, usize, &mut [T]) + Sync,
) {
    let out = RowsMut::new(out, width);
    for_ranges(nthreads, out.rows, grain, |lo, hi| {
        // SAFETY: `for_ranges` hands out disjoint ranges, each once, so no
        // other call holds any of rows `[lo, hi)`.
        f(lo, hi, unsafe { out.range(lo, hi) });
    });
}

/// [`for_rows`] over two outputs that share a row count: `f(lo, hi, a, b)`
/// gets rows `[lo, hi)` of `a` (rows of `a_width`) and of `b` (rows of
/// `b_width`).
///
/// # Panics
/// If a width is zero or does not divide its buffer, or the two buffers do
/// not hold the same number of rows.
pub fn for_rows2<T: Send, U: Send>(
    nthreads: usize,
    grain: usize,
    a: &mut [T],
    a_width: usize,
    b: &mut [U],
    b_width: usize,
    f: impl Fn(usize, usize, &mut [T], &mut [U]) + Sync,
) {
    let a = RowsMut::new(a, a_width);
    let b = RowsMut::new(b, b_width);
    assert_eq!(a.rows, b.rows, "both outputs need the same number of rows");
    for_ranges(nthreads, a.rows, grain, |lo, hi| {
        // SAFETY: `for_ranges` hands out disjoint ranges, each once, so no
        // other call holds any of rows `[lo, hi)` of either buffer.
        let (a, b) = unsafe { (a.range(lo, hi), b.range(lo, hi)) };
        f(lo, hi, a, b);
    });
}

/// A buffer whose rows parallel workers write, each row by one worker at a
/// time. It borrows the buffer mutably for its whole life, so nothing else
/// reads or writes it meanwhile; every access is bounds-checked, and the one
/// thing a caller promises is that no two live references cover the same row.
pub struct RowsMut<'a, T> {
    ptr: *mut T,
    rows: usize,
    width: usize,
    _buf: PhantomData<&'a mut [T]>,
}

// SAFETY: a `RowsMut` only hands out `&mut` to rows its callers keep
// disjoint across threads, so sharing it shares no element; `T: Send` lets
// each element be written from the thread that took its row.
unsafe impl<T: Send> Sync for RowsMut<'_, T> {}

impl<T> std::fmt::Debug for RowsMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RowsMut")
            .field("rows", &self.rows)
            .field("width", &self.width)
            .finish_non_exhaustive()
    }
}

impl<'a, T> RowsMut<'a, T> {
    /// # Panics
    /// If `width` is zero or does not divide `buf.len()`.
    pub fn new(buf: &'a mut [T], width: usize) -> Self {
        assert!(
            width > 0 && buf.len().is_multiple_of(width),
            "a buffer of {} elements is not whole rows of {width}",
            buf.len()
        );
        Self {
            ptr: buf.as_mut_ptr(),
            rows: buf.len() / width,
            width,
            _buf: PhantomData,
        }
    }

    /// Row `i`.
    ///
    /// # Safety
    /// No other live reference from this `RowsMut` may cover row `i`.
    ///
    /// # Panics
    /// If `i` is not a row of the buffer.
    #[expect(
        clippy::mut_from_ref,
        reason = "the caller keeps the rows it takes disjoint"
    )]
    pub unsafe fn row(&self, i: usize) -> &mut [T] {
        // SAFETY: the caller keeps the rows disjoint.
        unsafe { self.range(i, i + 1) }
    }

    /// Rows `[lo, hi)`.
    ///
    /// # Safety
    /// No other live reference from this `RowsMut` may cover any of them.
    ///
    /// # Panics
    /// If `lo > hi` or `hi` is past the last row.
    #[expect(
        clippy::mut_from_ref,
        reason = "the caller keeps the rows it takes disjoint"
    )]
    pub unsafe fn range(&self, lo: usize, hi: usize) -> &mut [T] {
        assert!(
            lo <= hi && hi <= self.rows,
            "rows {lo}..{hi} of {}",
            self.rows
        );
        // SAFETY: rows `[lo, hi)` lie inside the buffer (checked above),
        // which this value borrows mutably for `'a`, and the caller promises
        // no other reference covers them.
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.add(lo * self.width), (hi - lo) * self.width)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_all_indices() {
        let mut hits = [0u32; 17];
        let hits = std::sync::Mutex::new(&mut hits[..]);
        for_ranges(4, 17, 1, |lo, hi| {
            let mut g = hits.lock().unwrap();
            for i in lo..hi {
                g[i] += 1;
            }
        });
        assert!(hits.lock().unwrap().iter().all(|&c| c == 1));
    }

    #[test]
    fn for_rows_hands_each_row_its_own_index() {
        let mut out = vec![0usize; 3 * 37];
        for_rows(4, 1, &mut out, 3, |lo, _hi, rows| {
            for (i, row) in rows.chunks_mut(3).enumerate() {
                row.fill(lo + i);
            }
        });
        assert!(out
            .chunks(3)
            .enumerate()
            .all(|(i, row)| row.iter().all(|&v| v == i)));
    }

    #[test]
    fn for_rows2_keeps_both_outputs_on_the_same_rows() {
        let mut a = vec![0usize; 2 * 29];
        let mut b = vec![0usize; 5 * 29];
        for_rows2(4, 1, &mut a, 2, &mut b, 5, |lo, hi, a, b| {
            for i in lo..hi {
                a[(i - lo) * 2..][..2].fill(i);
                b[(i - lo) * 5..][..5].fill(i);
            }
        });
        assert!(a
            .chunks(2)
            .enumerate()
            .all(|(i, r)| r.iter().all(|&v| v == i)));
        assert!(b
            .chunks(5)
            .enumerate()
            .all(|(i, r)| r.iter().all(|&v| v == i)));
    }

    #[test]
    #[should_panic(expected = "not whole rows")]
    fn rows_mut_refuses_a_partial_row() {
        let mut buf = [0u8; 7];
        let _ = RowsMut::new(&mut buf, 2);
    }
}
