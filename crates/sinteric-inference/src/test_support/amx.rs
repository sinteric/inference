//! Shared test-only platform check. Never used by the inference runtime.

/// Exercise AMX tile preservation under preemption on each worker CPU.
/// A same-CPU integer contender makes the probe useful even on an idle box.
/// This checks the OS/hypervisor context switch, independently of oneDNN.
pub fn amx_tile_state_is_preserved(cpus: &[usize]) -> bool {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        if std::arch::x86_64::__cpuid_count(7, 0).edx & (1 << 24) == 0 {
            return true;
        }
        let mut preserved = true;
        for &cpu in cpus {
            // SAFETY: the CPU reports AMX (checked above), and the probe runs
            // on a thread of its own, whose affinity it may change.
            let result = std::thread::spawn(move || unsafe { probe_cpu(cpu) })
                .join()
                .expect("AMX platform probe");
            match result {
                Ok(0) => {}
                Ok(bad) => {
                    eprintln!("PLATFORM AMX STATE CORRUPTION: skipping AMX cross-thread assert (CPU {cpu}: {bad}/4096 corrupted tiles)");
                    preserved = false;
                }
                Err(reason) => {
                    eprintln!("AMX platform probe unavailable: {reason}; skipping AMX cross-thread assert");
                    return false;
                }
            }
        }
        preserved
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        let _ = cpus;
        true
    }
}

/// Pin the calling thread to `cpu`.
///
/// # Safety
///
/// `cpu` is below `CPU_SETSIZE`, and the caller owns the thread's affinity.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe fn pin(cpu: usize) -> Result<(), String> {
    let mut mask: libc::cpu_set_t = std::mem::zeroed();
    libc::CPU_ZERO(&mut mask);
    libc::CPU_SET(cpu, &mut mask);
    if libc::sched_setaffinity(0, std::mem::size_of_val(&mask), &mask) != 0 {
        return Err(format!(
            "pin CPU {cpu}: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

/// Load and store one tile 4,096 times on `cpu` beside an integer contender,
/// and count the stores that came back changed.
///
/// # Safety
///
/// The CPU supports AMX, and the calling thread is the probe's own: it is
/// pinned, and its tile configuration is replaced.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe fn probe_cpu(cpu: usize) -> Result<usize, String> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    pin(cpu)?;
    if libc::syscall(libc::SYS_arch_prctl, 0x1023, 18) != 0 {
        return Err(format!(
            "ARCH_REQ_XCOMP_PERM: {}",
            std::io::Error::last_os_error()
        ));
    }
    #[repr(C, align(64))]
    struct TileConfig {
        palette: u8,
        start_row: u8,
        reserved: [u8; 14],
        columns: [u16; 16],
        rows: [u8; 16],
    }
    let mut cfg = TileConfig {
        palette: 1,
        start_row: 0,
        reserved: [0; 14],
        columns: [0; 16],
        rows: [0; 16],
    };
    cfg.columns[0] = 64;
    cfg.rows[0] = 16;
    let stop = Arc::new(AtomicBool::new(false));
    let contender_stop = Arc::clone(&stop);
    let ready = Arc::new(AtomicBool::new(false));
    let contender_ready = Arc::clone(&ready);
    let contender = std::thread::spawn(move || {
        pin(cpu).expect("pin probe contender");
        contender_ready.store(true, Ordering::Release);
        while !contender_stop.load(Ordering::Relaxed) {
            // Plain integer work: no AMX instructions or oneDNN calls.
            std::hint::black_box(1usize.wrapping_add(2));
        }
    });
    while !ready.load(Ordering::Acquire) {
        std::thread::yield_now();
    }
    let mut src = [0u8; 1024];
    let mut dst = [0u8; 1024];
    let mut bad = 0;
    core::arch::asm!("ldtilecfg [{cfg}]", cfg = in(reg) &cfg, options(nostack));
    for iteration in 0usize..4096 {
        for (i, value) in src.iter_mut().enumerate() {
            *value = (i as u8).wrapping_mul(7).wrapping_add(iteration as u8);
        }
        core::arch::asm!("tileloadd tmm0, [{src} + {stride}]", src = in(reg) src.as_ptr(), stride = in(reg) 64usize, options(nostack));
        // PAUSE keeps the live-tile interval long enough to span scheduler
        // time slices; an optimized integer loop could finish before preemption.
        for _ in 0..2000 {
            std::hint::spin_loop();
        }
        core::arch::asm!("tilestored [{dst} + {stride}], tmm0", dst = in(reg) dst.as_mut_ptr(), stride = in(reg) 64usize, options(nostack));
        bad += usize::from(src != dst);
    }
    core::arch::asm!("tilerelease", options(nostack));
    stop.store(true, Ordering::Relaxed);
    contender.join().expect("AMX probe contender");
    Ok(bad)
}
