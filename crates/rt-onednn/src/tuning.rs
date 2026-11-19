//! Per-ISA attention tiles and loop grains.
//!
//! Do **not** derive EG2 tiles from sysfs cache sizes. VMs often report a
//! small L2 (512 KiB on E6) and a cache-budget formula then picks 64, which
//! was 3–22% slower than 256 on AVX512_CORE_BF16. Tiles come from a
//! gate-validated table (AMX / AVX512_CORE_BF16 / default). Sequence-length
//! remainders are handled at runtime: `clamp_attn_tile` oneshots when the
//! score matrix fits, and the unaligned path folds a short last window
//! rather than shrinking this cap to 64/128.
//!
//! `advanced.tile` / CLI `--tile` is the only override (hidden, experimental).

use crate::isa_has_amx;
use rt_threads::CpuTopology;

/// Measured EG2 Q-chunk on AMX (`tdpbf16ps` 16-row). Intel matrix: 256.
const EG2_TILE_AMX: usize = 256;
/// Measured EG2 Q-chunk on AVX512_CORE_BF16 (and unknown ISA). E6: 256 beat 64.
const EG2_TILE_AVX512_BF16: usize = 256;
/// Vela 0.3B previous default; not derived from L2.
const VELA_TILE_DEFAULT: usize = 64;

/// Attention tiles and parallel grains. EG2/Vela tiles are per-ISA tables,
/// not a function of reported L2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TuningPlan {
    pub topology: CpuTopology,
    pub cpu_isa: u64,
    pub threads: usize,
    pub eg2_attn_tile: usize,
    pub vela_attn_tile: usize,
    /// Grain for long eltwise/glue `for_ranges` loops.
    pub eltwise_grain: usize,
}

impl TuningPlan {
    pub fn new(topology: CpuTopology, cpu_isa: u64, threads: usize) -> Self {
        let threads = threads.max(1);
        Self {
            topology,
            cpu_isa,
            threads,
            eg2_attn_tile: eg2_attn_tile(cpu_isa),
            vela_attn_tile: vela_attn_tile(cpu_isa),
            eltwise_grain: eltwise_grain(topology.l1d_bytes, threads),
        }
    }
}

/// EG2 attention tile from the measured per-ISA table.
///
/// Shape (M/N) does not change the cap: shrinking to hide a remainder
/// picked 64 on E6 and lost. Remainder minimization is oneshot-when-fits
/// plus last-window fold, not a smaller default tile.
pub fn eg2_attn_tile(isa: u64) -> usize {
    if isa_has_amx(isa) {
        EG2_TILE_AMX
    } else {
        EG2_TILE_AVX512_BF16
    }
}

pub fn vela_attn_tile(_isa: u64) -> usize {
    VELA_TILE_DEFAULT
}

fn eltwise_grain(l1d_bytes: usize, threads: usize) -> usize {
    let from_l1 = if l1d_bytes == 0 {
        4096
    } else {
        (l1d_bytes / 16).clamp(256, 4096)
    };
    (from_l1 / threads.max(1)).clamp(256, 4096)
}

#[cfg(test)]
mod tests {
    use super::*;

    const AMX: u64 = 0xfef;
    const AVX512_BF16: u64 = 0xe7;

    fn topo(l2: usize, l3: usize, cores: usize) -> CpuTopology {
        CpuTopology {
            physical_cores: cores,
            l1d_bytes: 32 * 1024,
            l2_bytes: l2,
            l3_bytes: l3,
        }
    }

    #[test]
    fn per_isa_table_is_256_on_amx_and_avx512_bf16() {
        assert_eq!(eg2_attn_tile(AMX), 256);
        assert_eq!(eg2_attn_tile(AVX512_BF16), 256);
        assert_eq!(eg2_attn_tile(0), 256);
        assert_eq!(vela_attn_tile(AMX), 64);
        assert_eq!(vela_attn_tile(AVX512_BF16), 64);
    }

    #[test]
    fn e6_512kib_l2_does_not_pick_tile_64() {
        // E6 reported 512 KiB L2; the old cache formula picked 64 and lost.
        let p = TuningPlan::new(topo(512 * 1024, 32 * 1024 * 1024, 4), AVX512_BF16, 4);
        assert_eq!(p.eg2_attn_tile, 256);
        assert_ne!(p.eg2_attn_tile, 64);
        assert_ne!(p.eg2_attn_tile, 128);
    }

    #[test]
    fn turin_1mib_l2_stays_on_the_table_not_128() {
        let p = TuningPlan::new(topo(1024 * 1024, 32 * 1024 * 1024, 8), AVX512_BF16, 8);
        assert_eq!(p.eg2_attn_tile, 256);
    }

    #[test]
    fn intel_2mib_amx_stays_256() {
        let p = TuningPlan::new(topo(2048 * 1024, 32 * 1024 * 1024, 4), AMX, 4);
        assert_eq!(p.eg2_attn_tile, 256);
        assert_eq!(p.eg2_attn_tile % 16, 0);
    }

    #[test]
    fn reported_l2_does_not_change_the_tile() {
        let small = TuningPlan::new(topo(256 * 1024, 8 * 1024 * 1024, 4), AVX512_BF16, 4);
        let big = TuningPlan::new(topo(2048 * 1024, 128 * 1024 * 1024, 4), AVX512_BF16, 4);
        assert_eq!(small.eg2_attn_tile, big.eg2_attn_tile);
        assert_eq!(small.eg2_attn_tile, 256);
    }

    #[test]
    fn thread_count_does_not_change_eg2_tile() {
        let p2 = TuningPlan::new(topo(2048 * 1024, 32 * 1024 * 1024, 2), AMX, 2);
        let p8 = TuningPlan::new(topo(2048 * 1024, 32 * 1024 * 1024, 8), AMX, 8);
        assert_eq!(p2.eg2_attn_tile, p8.eg2_attn_tile);
        assert!(p8.eltwise_grain <= p2.eltwise_grain);
    }

    #[test]
    fn vela_is_the_previous_default() {
        assert_eq!(vela_attn_tile(0), 64);
        assert_eq!(vela_attn_tile(AMX), 64);
    }
}
