//! Shape policies for the effective oneDNN ISA.

/// AMX BF16 tile height.
pub const AMX_BF16_TILE_ROWS: usize = 16;
/// Fixed block used for larger packed projections.
pub const AMX_BF16_TILE_BLOCK: usize = 64;
/// Maximum global attention score allocation, including a folded query tail.
pub const ATTN_SCORE_BYTES_CAP: usize = 256 * 1024 * 1024;
// Folding an unaligned tail pads and copies the entire preceding score tile
// for both QK and PV. Bound that extra traffic independently of pool width.
const ATTN_FOLD_COPY_BYTES_CAP: usize = 16 * 1024 * 1024;

/// Execution layout selected once from the detected or pinned CPU ISA.
///
/// Every policy is a pure function of input shape and fixed ISA constants.
/// No policy accepts, reads, or depends on thread count or pool occupancy.
/// oneDNN always receives the full pool width and partitions its own GEMMs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IsaStrategy {
    Amx,
    /// Live-row policy, also used for CPUs below AVX-512.
    Avx512,
}

impl IsaStrategy {
    pub fn from_cpu_isa(isa: u64) -> Self {
        if crate::isa_has_amx(isa) {
            Self::Amx
        } else {
            Self::Avx512
        }
    }

    /// Packed projections reuse the padded residual layout throughout a forward.
    /// Preserve 16-aligned M; otherwise round to 16 below 64 rows and 64 above.
    #[inline]
    pub fn gemm_pad_m(self, m: usize) -> usize {
        if self == Self::Avx512 || m.is_multiple_of(AMX_BF16_TILE_ROWS) {
            // Preserve main's aligned GEMMs. The unnecessary 1200 -> 1216
            // pad regressed B4/S300 by over 20% in primitive-time profiles.
            return m;
        }
        let block = if m < AMX_BF16_TILE_BLOCK {
            AMX_BF16_TILE_ROWS
        } else {
            AMX_BF16_TILE_BLOCK
        };
        m.div_ceil(block).saturating_mul(block)
    }

    #[inline]
    pub fn attn_pad_m(self, m: usize) -> usize {
        self.gemm_pad_m(m)
    }

    #[inline]
    pub fn residual_rows(self, tokens: usize) -> usize {
        self.gemm_pad_m(tokens)
    }

    /// Only buffers with added residual padding need whole-tile layout work.
    /// Already aligned AMX shapes and AVX-512 retain main's row scheduling,
    /// including grain 8 at 64 rows. Norm and conversion kernels are always
    /// row-disjoint and retain their original grains.
    pub fn residual_row_tile(self, live_rows: usize) -> usize {
        if self.residual_rows(live_rows) > live_rows {
            AMX_BF16_TILE_ROWS
        } else {
            1
        }
    }

    fn score_query_cap(self, batch: usize, nq: usize, nkv: usize, s: usize) -> usize {
        if self == Self::Avx512 {
            return attn_score_max_query_rows(batch, nq, s);
        }
        let groups = nq.checked_div(nkv).unwrap_or(1).max(1);
        let per_m = batch
            .saturating_mul(nkv)
            .saturating_mul(s)
            .saturating_mul(4)
            .max(1);
        let max_m = ATTN_SCORE_BYTES_CAP / per_m;
        let block = if max_m >= AMX_BF16_TILE_BLOCK {
            AMX_BF16_TILE_BLOCK
        } else {
            AMX_BF16_TILE_ROWS
        };
        // Bound the temporary padded QK destination as well as live scores.
        // Use the block upper bound: an unaligned tail can pad farther than
        // a slightly larger, already aligned M. E.g. b16*s4224 uses 240
        // queries, ensuring every shorter tail also fits the 256 MiB cap.
        ((max_m / block * block) / groups).max(1)
    }

    /// Query ranges cover each live query exactly once. AVX-512 retains
    /// `step_by(tile)`; AMX absorbs short or wasteful grouped tails.
    /// A global fold must still fit the score cap after folding. Large
    /// unaligned folds retain main's tiles to avoid full-score pad copies.
    pub fn attn_query_ranges(
        self,
        s: usize,
        tile: usize,
        batch: usize,
        nq: usize,
        nkv: usize,
        local: bool,
    ) -> Vec<(usize, usize)> {
        if s == 0 {
            return Vec::new();
        }
        let tile = if local {
            tile.max(1)
        } else {
            tile.max(1).min(self.score_query_cap(batch, nq, nkv, s))
        };
        if s <= tile {
            return vec![(0, s)];
        }
        let rem = s % tile;
        let groups = nq.checked_div(nkv).unwrap_or(1).max(1);
        let folded_m = groups.saturating_mul(tile.saturating_add(rem));
        let folded_score_bytes = batch
            .saturating_mul(nq)
            .saturating_mul(tile.saturating_add(rem))
            .saturating_mul(s)
            .saturating_mul(4);
        // Keep main's separate tail when a fold would force large copies.
        // B16/S777 profiles attributed ~1.3 seconds to padded QK/PV copies,
        // while actual primitive time was unchanged. Aligned folds need no
        // such copies; small buffers retain the short-tail AMX benefit.
        let fold_without_large_copy =
            self.attn_pad_m(folded_m) == folded_m || folded_score_bytes <= ATTN_FOLD_COPY_BYTES_CAP;
        let fold = self == Self::Amx
            && fold_without_large_copy
            && attn_amx_fold_remainder(rem, nq, nkv)
            && (local || tile + rem <= self.score_query_cap(batch, nq, nkv, s));
        let mut out: Vec<_> = (0..s)
            .step_by(tile)
            .map(|start| (start, tile.min(s - start)))
            .collect();
        if fold {
            if let Some((_, tail)) = out.pop() {
                if let Some(last) = out.last_mut() {
                    last.1 += tail;
                }
            }
        }
        out
    }
}

pub fn attn_amx_fold_remainder(rem: usize, nq: usize, nkv: usize) -> bool {
    if rem == 0 {
        return false;
    }
    if rem < AMX_BF16_TILE_ROWS {
        return true;
    }
    let groups = nq.checked_div(nkv).unwrap_or(1).max(1);
    let m = groups.saturating_mul(rem);
    let pad = m
        .div_ceil(AMX_BF16_TILE_ROWS)
        .saturating_mul(AMX_BF16_TILE_ROWS)
        .saturating_sub(m);
    pad.saturating_mul(4) > m
}

pub fn attn_score_max_query_rows(batch: usize, nq: usize, nk: usize) -> usize {
    let per = batch
        .saturating_mul(nq)
        .saturating_mul(nk)
        .saturating_mul(4)
        .max(1);
    (ATTN_SCORE_BYTES_CAP / per).max(1)
}

/// Clamp to the live sequence, sliding window, and global score budget.
pub fn clamp_attn_tile(
    tile: usize,
    s: usize,
    local: bool,
    radius: usize,
    batch: usize,
    nq: usize,
) -> usize {
    let mut t = tile.min(s).max(1);
    if local {
        t = t
            .min(radius.saturating_mul(2).saturating_add(1))
            .min(s)
            .max(1);
    } else {
        let max_rows = attn_score_max_query_rows(batch, nq, s);
        t = if s <= max_rows {
            s.max(1)
        } else {
            t.min(max_rows).max(1)
        };
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(policy: IsaStrategy) -> Vec<(usize, usize, usize, Vec<(usize, usize)>)> {
        [
            0usize, 1, 15, 16, 17, 33, 48, 64, 100, 128, 129, 192, 256, 300, 320, 777, 4224,
        ]
        .into_iter()
        .map(|s| {
            (
                policy.gemm_pad_m(s),
                policy.residual_rows(s),
                policy.residual_row_tile(s),
                policy.attn_query_ranges(s, 256, 3, 4, 2, true),
            )
        })
        .collect()
    }

    #[test]
    fn policies_are_independent_of_thread_context() {
        for policy in [IsaStrategy::Amx, IsaStrategy::Avx512] {
            let expected = snapshot(policy);
            for threads in [1, 2, 4, 8] {
                std::thread::scope(|scope| {
                    let jobs: Vec<_> = (0..threads)
                        .map(|_| scope.spawn(|| snapshot(policy)))
                        .collect();
                    for job in jobs {
                        assert_eq!(job.join().unwrap(), expected);
                    }
                });
            }
        }
    }

    #[test]
    fn global_scores_including_amx_padding_fit_cap() {
        for policy in [IsaStrategy::Amx, IsaStrategy::Avx512] {
            for batch in [1usize, 3, 5, 16] {
                for s in [129usize, 777, 4096, 4097, 4104, 4224, 8192] {
                    for (nq, nkv) in [(4, 1), (4, 2)] {
                        let tile = clamp_attn_tile(256, s, false, 512, batch, nq);
                        let ranges = policy.attn_query_ranges(s, tile, batch, nq, nkv, false);
                        let mut end = 0;
                        for (start, n) in ranges {
                            assert_eq!(start, end);
                            end += n;
                            let grouped_m = (nq / nkv) * n;
                            let bytes = batch * nkv * policy.attn_pad_m(grouped_m) * s * 4;
                            assert!(
                                bytes <= ATTN_SCORE_BYTES_CAP,
                                "{policy:?} b{batch} s{s} n{n}: {bytes}"
                            );
                        }
                        assert_eq!(end, s);
                    }
                }
            }
        }
    }

    #[test]
    fn residual_glue_covers_rows_once_in_whole_tiles() {
        use std::sync::Mutex;
        for policy in [IsaStrategy::Amx, IsaStrategy::Avx512] {
            for live in (0..=320).chain([777, 1024, 1248]) {
                let rows = policy.residual_rows(live);
                assert!(rows >= live);
                assert_eq!(
                    policy.residual_row_tile(live),
                    if rows > live { 16 } else { 1 }
                );
                if policy == IsaStrategy::Avx512 || live.is_multiple_of(AMX_BF16_TILE_ROWS) {
                    assert_eq!(rows, live);
                }
                for threads in [1, 2, 4, 8] {
                    let hits = Mutex::new(vec![0u8; rows]);
                    rt_kernels::for_row_tiles(
                        threads,
                        rows,
                        policy.residual_row_tile(live),
                        8,
                        |lo, hi| {
                            if rows > live {
                                assert_eq!(lo % 16, 0);
                                assert_eq!(hi % 16, 0);
                            }
                            for hit in &mut hits.lock().unwrap()[lo..hi] {
                                *hit += 1;
                            }
                        },
                    )
                    .unwrap();
                    assert!(hits.into_inner().unwrap().iter().all(|&h| h == 1));
                }
            }
        }
    }
}
