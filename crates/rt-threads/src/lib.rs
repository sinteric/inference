//! Explicit thread budget, physical-core selection, and process affinity.
//!
//! Libraries never read `OMP_NUM_THREADS` or `available_parallelism`. The
//! caller passes a [`ThreadBudget`]. Default placement is one logical CPU per
//! physical core (the first sibling in `thread_siblings_list`), so a 2-thread
//! budget on an SMT guest does not pin two hyperthreads of the same core.
//!
//! oneDNN GEMMs and glue share this pool (`threadpool_iface`). Concurrent
//! callers queue in arrival order: one `parallel_for` occupies the workers
//! at a time. This crate does not set process environment variables.

mod pool;
mod topology;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub use rt_core::{Error, Result};

pub use pool::{
    compute_contended, compute_lock, compute_order, in_parallel, install as install_pool,
    install_or_check as install_pool_or_check, installed_nthreads, par_for, set_compute_order,
    ComputeGuard, ComputeOrder,
};
pub use topology::{parse_cache_size, CpuTopology};

/// How pool workers wait between parallel regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitPolicy {
    /// Spin `spin_count` times, then condvar sleep. Matches prior benches.
    Active,
    /// Sleep on the condvar. Lower idle CPU, higher wake latency.
    Passive,
}

/// Threading contract for one model session.
#[derive(Clone, Debug)]
pub struct ThreadBudget {
    pub nthreads: usize,
    pub wait: WaitPolicy,
    pub spin_count: u32,
    /// Explicit cpuset. `None` means "pick `nthreads` physical cores".
    pub cpuset: Option<String>,
    /// Resolved logical CPU ids this budget should pin to.
    pub cpus: Vec<usize>,
}

impl ThreadBudget {
    pub fn new(nthreads: usize) -> Result<Self> {
        if nthreads == 0 || nthreads > libc::CPU_SETSIZE as usize {
            return Err(rt_core::Error::Config(
                "ThreadBudget.nthreads must be in 1..=CPU_SETSIZE".into(),
            ));
        }
        let cpus = select_cpus(nthreads, None)?;
        Ok(Self {
            nthreads,
            wait: WaitPolicy::Active,
            spin_count: 500,
            cpuset: None,
            cpus,
        })
    }

    pub fn with_wait(mut self, wait: WaitPolicy) -> Self {
        self.wait = wait;
        self
    }

    pub fn with_cpuset(mut self, cpuset: impl Into<String>) -> Result<Self> {
        let s = cpuset.into();
        self.cpus = select_cpus(self.nthreads, Some(&s))?;
        self.cpuset = Some(s);
        Ok(self)
    }

    pub fn nthreads_i32(&self) -> i32 {
        i32::try_from(self.nthreads).unwrap_or(i32::MAX)
    }

    pub fn omp_places(&self) -> String {
        self.cpus
            .iter()
            .map(|c| format!("{{{c}}}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn pool_spin(&self) -> u32 {
        match self.wait {
            WaitPolicy::Active => self.spin_count,
            WaitPolicy::Passive => 0,
        }
    }

    /// Install the pinned pool (GEMMs + glue) without process affinity.
    /// Library loads call this. Does not mutate process environment.
    pub fn install_library_workers(&self) -> Result<()> {
        pool::install_or_check_with(self.nthreads, &self.cpus, self.pool_spin())
    }

    /// Pin the calling process to [`Self::cpus`] and install the pool.
    /// Intended for process-owned drivers, not for embedding as a library.
    /// Does not set `OMP_*` / `GOMP_*` / `RAYON_*` / tokenizer env.
    pub fn apply_process_policy(&self) -> Result<()> {
        pin_process(&self.cpus)?;
        pool::install_or_check_with(self.nthreads, &self.cpus, self.pool_spin())?;
        Ok(())
    }
}

/// One logical CPU per physical core, preferring the lowest sibling id.
///
/// Matches `bench/harness/support.py:physical_cpus`: walk CPUs by id and keep
/// the first (lowest) sibling of each `thread_siblings_list` group. `readdir`
/// order is not stable, so a naive `or_insert` can pick 1,3,5,7 on SMT.
///
/// Only CPUs in the calling thread's affinity mask count, so a host started
/// under `taskset` or a cpuset cgroup keeps the pool inside its CPUs.
pub fn physical_cpu_ids() -> Result<Vec<usize>> {
    let base = Path::new("/sys/devices/system/cpu");
    if !base.is_dir() {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        return Ok((0..n).collect());
    }
    let mut found: Vec<(usize, String)> = Vec::new();
    let entries = fs::read_dir(base)?;
    for e in entries {
        let e = e?;
        let name = e.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("cpu") {
            continue;
        }
        let id: usize = match name[3..].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let sib = e.path().join("topology/thread_siblings_list");
        let key = fs::read_to_string(&sib)
            .unwrap_or_else(|_| id.to_string())
            .trim()
            .to_string();
        found.push((id, key));
    }
    if let Some(allowed) = allowed_cpus() {
        found.retain(|(id, _)| allowed.contains(id));
    }
    Ok(lowest_sibling_per_core(&found))
}

/// Keep the lowest logical CPU id in each SMT sibling group.
pub fn lowest_sibling_per_core(cpus: &[(usize, String)]) -> Vec<usize> {
    let mut found: Vec<(usize, String)> = cpus
        .iter()
        .map(|(id, key)| (*id, normalize_sibling_key(key)))
        .collect();
    found.sort_unstable_by_key(|(id, _)| *id);
    let mut groups: BTreeMap<String, usize> = BTreeMap::new();
    let mut online = Vec::new();
    for (id, key) in found {
        groups.entry(key).or_insert(id);
        online.push(id);
    }
    if groups.is_empty() {
        online.sort_unstable();
        return online;
    }
    let mut ids: Vec<usize> = groups.into_values().collect();
    ids.sort_unstable();
    ids
}

fn normalize_sibling_key(s: &str) -> String {
    match parse_cpuset(s) {
        Ok(mut ids) if !ids.is_empty() => {
            ids.sort_unstable();
            ids.dedup();
            ids.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",")
        }
        _ => s.trim().to_string(),
    }
}

/// Logical CPUs the calling thread may run on (`sched_getaffinity`), or
/// `None` when unknown.
pub fn allowed_cpus() -> Option<Vec<usize>> {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of_val(&set), &mut set) != 0 {
            return None;
        }
        let cpus: Vec<usize> = (0..libc::CPU_SETSIZE as usize)
            .filter(|&c| libc::CPU_ISSET(c, &set))
            .collect();
        (!cpus.is_empty()).then_some(cpus)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn parse_cpuset(s: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in s.split([',', ' ']) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            let lo: usize = a
                .parse()
                .map_err(|_| rt_core::Error::Message(format!("bad cpuset {s}")))?;
            let hi: usize = b
                .parse()
                .map_err(|_| rt_core::Error::Message(format!("bad cpuset {s}")))?;
            if lo > hi || hi >= libc::CPU_SETSIZE as usize {
                return Err(rt_core::Error::Config(format!(
                    "cpuset range {part} outside CPU_SETSIZE"
                )));
            }
            for c in lo..=hi {
                out.push(c);
            }
        } else {
            out.push(
                part.parse()
                    .map_err(|_| rt_core::Error::Message(format!("bad cpuset {s}")))?,
            );
        }
    }
    if out.iter().any(|&c| c >= libc::CPU_SETSIZE as usize) {
        return Err(rt_core::Error::Config(format!(
            "cpuset {s} outside CPU_SETSIZE"
        )));
    }
    out.sort_unstable();
    out.dedup();
    if out.is_empty() {
        return Err(rt_core::Error::Message("empty cpuset".into()));
    }
    Ok(out)
}

fn select_cpus(nthreads: usize, cpuset: Option<&str>) -> Result<Vec<usize>> {
    if nthreads == 0 || nthreads > libc::CPU_SETSIZE as usize {
        return Err(Error::Config(format!("invalid thread count {nthreads}")));
    }
    let allowed = allowed_cpus();
    let pool = if let Some(s) = cpuset {
        let cpus = parse_cpuset(s)?;
        if let Some(allowed) = &allowed {
            if let Some(c) = cpus.iter().find(|c| !allowed.contains(c)) {
                return Err(Error::Config(format!(
                    "cpuset {s}: CPU {c} is outside this process's affinity mask"
                )));
            }
        }
        cpus
    } else {
        let mut phys = physical_cpu_ids()?;
        if phys.len() < nthreads {
            // Fill remaining slots with leftover logical CPUs (SMT siblings).
            let mut extra = Vec::new();
            if let Ok(list) = fs::read_to_string("/sys/devices/system/cpu/online") {
                extra = parse_cpuset(list.trim()).unwrap_or_default();
            }
            for c in extra {
                if allowed.as_ref().is_some_and(|a| !a.contains(&c)) {
                    continue;
                }
                if !phys.contains(&c) {
                    phys.push(c);
                }
            }
        }
        phys
    };
    if pool.is_empty() {
        return Ok((0..nthreads).collect());
    }
    let mut out = Vec::with_capacity(nthreads);
    for i in 0..nthreads {
        out.push(pool[i % pool.len()]);
    }
    Ok(out)
}

fn pin_process(cpus: &[usize]) -> Result<()> {
    if cpus.is_empty() || cpus.iter().any(|&c| c >= libc::CPU_SETSIZE as usize) {
        return Err(Error::Config("invalid CPU affinity set".into()));
    }
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &c in cpus {
            libc::CPU_SET(c, &mut set);
        }
        if libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set) != 0 {
            return Err(rt_core::Error::Message(format!(
                "CPU affinity: {}",
                std::io::Error::last_os_error()
            )));
        }
    }
    let _ = cpus;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero() {
        assert!(ThreadBudget::new(0).is_err());
    }

    #[test]
    fn accepts_any_positive_count() {
        for n in [1usize, 2, 3, 5, 7, 8] {
            let b = ThreadBudget::new(n).unwrap();
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
            let b = ThreadBudget::new(2).unwrap();
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
            pin_process(&[last]).unwrap();
            assert_eq!(physical_cpu_ids().unwrap(), vec![last]);
            assert_eq!(ThreadBudget::new(2).unwrap().cpus, vec![last, last]);
            if last > 0 {
                let err = ThreadBudget::new(1).unwrap().with_cpuset("0").unwrap_err();
                assert!(matches!(err, Error::Config(_)), "{err}");
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
        let b = ThreadBudget::new(2)
            .unwrap()
            .with_cpuset(format!("{first},{last}"))
            .unwrap();
        assert_eq!(b.cpus, vec![first, last]);
        assert_eq!(b.omp_places(), format!("{{{first}}},{{{last}}}"));
    }

    #[test]
    fn does_not_read_env_for_nthreads() {
        let b = ThreadBudget::new(3).unwrap();
        assert_eq!(b.nthreads, 3);
        assert_eq!(b.wait, WaitPolicy::Active);
    }

    #[test]
    fn apply_process_policy_does_not_set_env() {
        let before = rt_core::env::thread_environment_snapshot();
        let b = ThreadBudget::new(2).unwrap();
        b.apply_process_policy().unwrap();
        assert_eq!(rt_core::env::thread_environment_snapshot(), before);
    }

    #[test]
    fn installed_pool_covers_indices() {
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
}
