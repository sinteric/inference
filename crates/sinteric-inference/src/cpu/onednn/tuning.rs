//! What the runtime derives from the host for every model: the topology,
//! the effective ISA, the thread count, and the grain of long element-wise
//! loops. A model's own tuning (EmbeddingGemma 2's and the guard's attention
//! tiles) lives with that model.

use crate::cpu::threads::CpuTopology;

/// The host-derived tuning every model shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TuningPlan {
    pub topology: CpuTopology,
    pub cpu_isa: u64,
    pub threads: usize,
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
            eltwise_grain: eltwise_grain(topology.l1d_bytes, threads),
        }
    }
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

    fn topo(l2: usize, l3: usize, cores: usize) -> CpuTopology {
        CpuTopology {
            physical_cores: cores,
            l1d_bytes: 32 * 1024,
            l2_bytes: l2,
            l3_bytes: l3,
        }
    }

    #[test]
    fn more_threads_take_a_finer_grain() {
        let p2 = TuningPlan::new(topo(2048 * 1024, 32 * 1024 * 1024, 2), AMX, 2);
        let p8 = TuningPlan::new(topo(2048 * 1024, 32 * 1024 * 1024, 8), AMX, 8);
        assert!(p8.eltwise_grain <= p2.eltwise_grain);
        assert!((256..=4096).contains(&p8.eltwise_grain));
    }
}
