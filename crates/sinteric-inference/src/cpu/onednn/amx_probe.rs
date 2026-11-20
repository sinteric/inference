//! AMX tile-state health probe.
//!
//! Some virtual machines lose the AMX tile registers of a thread that the
//! scheduler preempts while its tiles are live. The thread resumes with zeroed
//! tiles, so an AMX GEMM silently returns a wrong block (a 32x32 C tile in
//! oneDNN's `brg_matmul:avx10_1_512_amx`). This is below oneDNN and the thread
//! pool: raw `tdpbf16ps` loops fail the same way. It needs preemption, so it
//! shows with more runnable threads than free vCPUs, never with the AVX-512
//! kernels.
//!
//! [`amx_health`] runs a short tile GEMM that a second thread keeps preempting
//! on the same CPU and counts wrong results. [`crate::cpu::onednn::OnednnCpu::load_with`]
//! calls it before choosing the ISA: a [`CpuIsaCap::BestAvailable`] cap on an
//! affected host becomes [`CpuIsaCap::Avx512CoreBf16`]. Tests that measure AMX
//! itself turn the probe off (`set_amx_probe`); a library caller cannot.
//!
//! [`CpuIsaCap`]: crate::cpu::onednn::CpuIsaCap
//! [`CpuIsaCap::BestAvailable`]: crate::cpu::onednn::CpuIsaCap::BestAvailable
//! [`CpuIsaCap::Avx512CoreBf16`]: crate::cpu::onednn::CpuIsaCap::Avx512CoreBf16

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use onednn_sys::{sinteric_amx_tile_state_probe, SntAmxProbeResult};

static ENABLED: AtomicBool = AtomicBool::new(true);
static HEALTH: OnceLock<AmxHealth> = OnceLock::new();

/// Probe attempts, in milliseconds. A run that saw fewer than
/// [`MIN_PREEMPTIONS`] preemptions proves nothing, so the next one is longer.
const BUDGETS_MS: [i32; 4] = [40, 40, 80, 160];
const MIN_PREEMPTIONS: u64 = 10;

/// Outcome of the AMX tile-state probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AmxHealth {
    /// AMX is not usable by this process (no CPU support, no OS permission,
    /// or not x86-64 Linux).
    Unavailable,
    /// No wrong result. `conclusive` is false when the thread was preempted
    /// fewer than 10 times, so the probe could not tell.
    Healthy { conclusive: bool },
    /// Tile state was lost across preemption: AMX GEMMs on this host are
    /// unreliable under load.
    TileStateLoss {
        corrupt: u64,
        iterations: u64,
        preemptions: u64,
    },
}

impl AmxHealth {
    /// Whether AMX kernels may be used.
    #[cfg(test)]
    pub fn amx_usable(self) -> bool {
        !matches!(self, Self::Unavailable | Self::TileStateLoss { .. })
    }
}

/// Enable or disable the probe for the process (default: enabled). With it
/// off, [`amx_health`] is not consulted and `BestAvailable` keeps AMX. Set it
/// before the first model load; a result already cached is kept.
#[cfg(test)]
pub fn set_amx_probe(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn amx_probe_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Run the probe once per process (about 40 ms when healthy) and cache the
/// result. It runs its own two threads pinned to one CPU of the caller's
/// affinity mask and leaves the caller's affinity alone.
pub fn amx_health() -> AmxHealth {
    // SAFETY: the probe takes a time budget and only touches its own
    // threads' tile state.
    *HEALTH.get_or_init(|| probe_with(|ms| unsafe { sinteric_amx_tile_state_probe(ms) }))
}

fn probe_with(run: impl Fn(i32) -> SntAmxProbeResult) -> AmxHealth {
    let mut conclusive = false;
    for ms in BUDGETS_MS {
        let r = run(ms);
        if r.status != 0 {
            return AmxHealth::Unavailable;
        }
        if r.corrupt > 0 {
            return AmxHealth::TileStateLoss {
                corrupt: r.corrupt,
                iterations: r.iterations,
                preemptions: r.preemptions,
            };
        }
        if r.preemptions >= MIN_PREEMPTIONS {
            conclusive = true;
            break;
        }
    }
    AmxHealth::Healthy { conclusive }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(status: i32, corrupt: u64, preemptions: u64) -> SntAmxProbeResult {
        SntAmxProbeResult {
            status,
            iterations: 1000,
            corrupt,
            preemptions,
        }
    }

    #[test]
    fn corruption_is_reported_on_first_attempt() {
        let h = probe_with(|_| result(0, 3, 40));
        assert_eq!(
            h,
            AmxHealth::TileStateLoss {
                corrupt: 3,
                iterations: 1000,
                preemptions: 40
            }
        );
        assert!(!h.amx_usable());
    }

    #[test]
    fn quiet_attempts_extend_then_stay_inconclusive() {
        let calls = std::cell::RefCell::new(Vec::new());
        let h = probe_with(|ms| {
            calls.borrow_mut().push(ms);
            result(0, 0, 2)
        });
        assert_eq!(h, AmxHealth::Healthy { conclusive: false });
        assert_eq!(*calls.borrow(), BUDGETS_MS);
        assert!(h.amx_usable());
    }

    #[test]
    fn corruption_after_a_quiet_attempt_is_still_found() {
        let n = std::cell::Cell::new(0);
        let h = probe_with(|_| {
            n.set(n.get() + 1);
            if n.get() < 3 {
                result(0, 0, 0)
            } else {
                result(0, 1, 12)
            }
        });
        assert!(matches!(h, AmxHealth::TileStateLoss { corrupt: 1, .. }));
    }

    #[test]
    fn enough_preemptions_make_a_clean_run_conclusive() {
        assert_eq!(
            probe_with(|_| result(0, 0, 25)),
            AmxHealth::Healthy { conclusive: true }
        );
    }

    #[test]
    fn unavailable_amx_is_reported() {
        assert_eq!(probe_with(|_| result(-1, 0, 0)), AmxHealth::Unavailable);
    }

    /// The real probe: on an AMX host it must agree with itself across runs
    /// (a host that loses tile state does so every time under preemption).
    #[test]
    fn real_probe_runs() {
        let h = amx_health();
        eprintln!("amx_health = {h:?}");
        assert_eq!(h, amx_health());
    }

    /// Detection rate of one 40 ms probe run, for tuning [`BUDGETS_MS`].
    #[test]
    #[ignore = "diagnostic, 8 s"]
    fn single_run_detection_rate() {
        let runs = 200;
        let hits = (0..runs)
            // SAFETY: as in `amx_health`.
            .filter(|_| unsafe { sinteric_amx_tile_state_probe(BUDGETS_MS[0]) }.corrupt > 0)
            .count();
        eprintln!(
            "probe detected corruption in {hits}/{runs} single {} ms runs",
            BUDGETS_MS[0]
        );
    }
}
