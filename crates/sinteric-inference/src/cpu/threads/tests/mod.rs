//! Tests of the thread pool. The submodules were integration tests, each in
//! its own process.

mod capped_jobs;

use super::*;

/// Pins the calling thread to `cpu`, as `taskset -c <cpu>` would a host.
fn pin_thread(cpu: usize) {
    // SAFETY: `set` is a zeroed `cpu_set_t` with one bit set; the call reads
    // it and changes only the calling thread's mask.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        assert_eq!(
            libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set),
            0,
            "{}",
            std::io::Error::last_os_error()
        );
    }
}

#[test]
fn rejects_zero() {
    assert!(ThreadBudget::try_new(0).is_err());
}

#[test]
fn accepts_any_positive_count() {
    for n in [1usize, 2, 3, 5, 7, 8] {
        let b = ThreadBudget::try_new(n).unwrap();
        assert_eq!(b.nthreads, n);
        assert_eq!(b.cpus.len(), n);
    }
}

#[test]
fn physical_ids_are_unique_by_default() {
    let phys = physical_cpu_ids().unwrap();
    let mut seen = phys.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(phys.len(), seen.len());
    if phys.len() >= 2 {
        let b = ThreadBudget::try_new(2).unwrap();
        // Default 2-thread budget uses two *physical* cores when they exist,
        // not SMT siblings 0 and 1 of the same core.
        assert_eq!(b.cpus, phys[..2].to_vec());
    }
}

#[test]
fn lowest_smt_sibling_wins_regardless_of_discovery_order() {
    let high_first = [
        (1usize, "0,1".to_string()),
        (0, "0,1".to_string()),
        (3, "2,3".to_string()),
        (2, "2,3".to_string()),
        (5, "4,5".to_string()),
        (4, "4,5".to_string()),
        (7, "6,7".to_string()),
        (6, "6,7".to_string()),
    ];
    assert_eq!(lowest_sibling_per_core(&high_first), vec![0, 2, 4, 6]);
    let reversed_lists = [
        (7usize, "7,6".to_string()),
        (6, "7,6".to_string()),
        (1, "1,0".to_string()),
        (0, "1,0".to_string()),
    ];
    assert_eq!(lowest_sibling_per_core(&reversed_lists), vec![0, 6]);
}

#[test]
fn cpu_selection_stays_inside_the_affinity_mask() {
    let Some(allowed) = allowed_cpus() else {
        return;
    };
    let last = *allowed.last().unwrap();
    // A host started under `taskset -c <last>`: threads it spawns inherit
    // the mask, and the pool must not pin outside it.
    std::thread::spawn(move || {
        pin_thread(last);
        assert_eq!(physical_cpu_ids().unwrap(), vec![last]);
        assert_eq!(ThreadBudget::try_new(2).unwrap().cpus, vec![last, last]);
        if last > 0 {
            let err = ThreadBudget::try_new(1)
                .unwrap()
                .with_cpuset("0")
                .unwrap_err();
            assert_eq!(err.kind(), crate::ErrorKind::Config, "{err}");
        }
    })
    .join()
    .unwrap();
    assert!(physical_cpu_ids()
        .unwrap()
        .iter()
        .all(|c| allowed.contains(c)));
}

#[test]
fn explicit_cpuset_is_honoured() {
    // The set names the first and last CPUs in the affinity mask: a fixed
    // "0,2" was refused on a two-CPU CI runner, whose mask is {0, 1}.
    let allowed = allowed_cpus().unwrap_or_else(|| vec![0, 2]);
    let (first, last) = (allowed[0], allowed[allowed.len() - 1]);
    let b = ThreadBudget::try_new(2)
        .unwrap()
        .with_cpuset(format!("{first},{last}"))
        .unwrap();
    assert_eq!(b.cpus, vec![first, last]);
}

#[test]
fn does_not_read_env_for_nthreads() {
    let b = ThreadBudget::try_new(3).unwrap();
    assert_eq!(b.nthreads, 3);
}

#[test]
fn installing_the_pool_does_not_set_env() {
    crate::test_support::own_process!(installing_the_pool_does_not_set_env);
    let before = crate::test_support::env::thread_environment_snapshot();
    let b = ThreadBudget::try_new(2).unwrap();
    b.install_library_workers().unwrap();
    assert_eq!(
        crate::test_support::env::thread_environment_snapshot(),
        before
    );
}

#[test]
fn installed_pool_covers_indices() {
    crate::test_support::own_process!(installed_pool_covers_indices);
    let cpus = physical_cpu_ids().unwrap_or_else(|_| vec![0, 1]);
    install_pool(cpus.len().clamp(2, 4), &cpus).unwrap();
    let mut hits = vec![0u32; 64];
    let hits = std::sync::Mutex::new(&mut hits[..]);
    par_for(4, 64, 1, |lo, hi| {
        let mut g = hits.lock().unwrap();
        for i in lo..hi {
            g[i] += 1;
        }
    });
    assert!(hits.lock().unwrap().iter().all(|&c| c == 1));
}
