//! A job capped below the pool width runs on that many workers (`snt-par-0..`),
//! the cap a backend's thread limit relies on. Own test binary: the pool is
//! process-global and the unit tests install a 2-worker one.

use std::collections::BTreeSet;
use std::sync::Mutex;

fn workers_used(nthreads: usize) -> BTreeSet<String> {
    let seen = Mutex::new(BTreeSet::new());
    let hits: Vec<_> = (0..64)
        .map(|_| std::sync::atomic::AtomicUsize::new(0))
        .collect();
    for _ in 0..50 {
        crate::cpu::threads::par_for(nthreads, 64, 1, |lo, hi| {
            let name = std::thread::current().name().unwrap_or("?").to_owned();
            seen.lock().unwrap().insert(name);
            for h in &hits[lo..hi] {
                h.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            // Long enough that every eligible worker wakes up for the job.
            std::thread::sleep(std::time::Duration::from_micros(200));
        });
    }
    assert!(hits
        .iter()
        .all(|h| h.load(std::sync::atomic::Ordering::Relaxed) == 50));
    seen.into_inner().unwrap()
}

#[test]
fn capped_jobs_run_on_the_first_workers_only() {
    crate::test_support::own_process!(capped_jobs_run_on_the_first_workers_only);
    crate::cpu::threads::install_pool(4, &[]).unwrap();
    for cap in 2..=4 {
        let used = workers_used(cap);
        let allowed: BTreeSet<String> = (0..cap).map(|i| format!("snt-par-{i}")).collect();
        assert!(used.is_subset(&allowed), "cap {cap}: ran on {used:?}");
    }
    // Wider than the pool is the pool.
    let used = workers_used(16);
    assert!(used.iter().all(|n| n.starts_with("snt-par-")), "{used:?}");
}
