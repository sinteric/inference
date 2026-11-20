//! Tiny parallel-for over index ranges. GEMMs and glue share the pinned
//! pool in `crate::cpu::threads` (oneDNN THREADPOOL runtime). Tests without a pool fall
//! back to `std::thread::scope`.

pub fn for_ranges(nthreads: usize, n: usize, grain: usize, f: impl Fn(usize, usize) + Sync) {
    crate::cpu::threads::par_for(nthreads, n, grain, f);
}

/// Partition rows in indivisible units without changing the pool's scheduling.
/// `tile_rows=1` preserves the original per-operation grain exactly.
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

/// Exclusive rows `[lo, hi)` of `width`.
///
/// # Safety
/// `ptr` must be valid for `hi * width` elements. Callers must pass
/// disjoint `[lo, hi)` to different `for_ranges` workers — overlapping
/// `&mut` of the full buffer is aliasing UB even if writes are disjoint.
#[inline]
pub unsafe fn rows_ref<'a, T>(ptr: *const T, lo: usize, hi: usize, width: usize) -> &'a [T] {
    debug_assert!(hi >= lo);
    std::slice::from_raw_parts(ptr.add(lo * width), (hi - lo) * width)
}

/// # Safety
/// Same as [`rows_ref`].
#[inline]
pub unsafe fn rows_mut<'a, T>(ptr: *mut T, lo: usize, hi: usize, width: usize) -> &'a mut [T] {
    debug_assert!(hi >= lo);
    std::slice::from_raw_parts_mut(ptr.add(lo * width), (hi - lo) * width)
}

/// # Safety
/// `ptr` must be valid for `(i + 1) * width` elements.
#[inline]
pub unsafe fn row_ref<'a, T>(ptr: *const T, i: usize, width: usize) -> &'a [T] {
    std::slice::from_raw_parts(ptr.add(i * width), width)
}

/// # Safety
/// Same as [`row_ref`].
#[inline]
pub unsafe fn row_mut<'a, T>(ptr: *mut T, i: usize, width: usize) -> &'a mut [T] {
    std::slice::from_raw_parts_mut(ptr.add(i * width), width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_all_indices() {
        let mut hits = vec![0u32; 17];
        let hits = std::sync::Mutex::new(&mut hits[..]);
        for_ranges(4, 17, 1, |lo, hi| {
            let mut g = hits.lock().unwrap();
            for i in lo..hi {
                g[i] += 1;
            }
        });
        assert!(hits.lock().unwrap().iter().all(|&c| c == 1));
    }
}
