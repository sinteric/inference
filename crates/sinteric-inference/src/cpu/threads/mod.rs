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

use crate::{Error, Result};

mod pool;
mod topology;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[cfg(test)]
pub use pool::install as install_pool;
pub use pool::{compute_contended, compute_lock, in_parallel, par_for};
pub use topology::CpuTopology;

/// Threading contract for one model session.
#[derive(Clone, Debug)]
pub struct ThreadBudget {
    pub nthreads: usize,
    /// Times a worker spins for new work before it sleeps on the condvar.
    pub spin_count: u32,
    /// Explicit cpuset. `None` means "pick `nthreads` physical cores".
    pub cpuset: Option<String>,
    /// Resolved logical CPU ids this budget should pin to.
    pub cpus: Vec<usize>,
}

impl ThreadBudget {
    pub fn new(nthreads: usize) -> Result<Self> {
        if nthreads == 0 || nthreads > libc::CPU_SETSIZE as usize {
            return Err(crate::Error::config(
                "ThreadBudget.nthreads must be in 1..=CPU_SETSIZE",
            ));
        }
        let cpus = select_cpus(nthreads, None)?;
        Ok(Self {
            nthreads,
            spin_count: 500,
            cpuset: None,
            cpus,
        })
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

    /// Install the pinned pool (GEMMs + glue). The pool's workers are
    /// pinned; the calling process is not, and no environment variable is
    /// set.
    pub fn install_library_workers(&self) -> Result<()> {
        pool::install_or_check_with(self.nthreads, &self.cpus, self.spin_count)
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
    // SAFETY: an all-zero `cpu_set_t` is a valid empty set, and
    // `sched_getaffinity` writes at most its size into it.
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
                .map_err(|_| crate::Error::config(format!("bad cpuset {s}")))?;
            let hi: usize = b
                .parse()
                .map_err(|_| crate::Error::config(format!("bad cpuset {s}")))?;
            if lo > hi || hi >= libc::CPU_SETSIZE as usize {
                return Err(crate::Error::config(format!(
                    "cpuset range {part} outside CPU_SETSIZE"
                )));
            }
            for c in lo..=hi {
                out.push(c);
            }
        } else {
            out.push(
                part.parse()
                    .map_err(|_| crate::Error::config(format!("bad cpuset {s}")))?,
            );
        }
    }
    if out.iter().any(|&c| c >= libc::CPU_SETSIZE as usize) {
        return Err(crate::Error::config(format!(
            "cpuset {s} outside CPU_SETSIZE"
        )));
    }
    out.sort_unstable();
    out.dedup();
    if out.is_empty() {
        return Err(crate::Error::config("empty cpuset"));
    }
    Ok(out)
}

fn select_cpus(nthreads: usize, cpuset: Option<&str>) -> Result<Vec<usize>> {
    if nthreads == 0 || nthreads > libc::CPU_SETSIZE as usize {
        return Err(Error::config(format!("invalid thread count {nthreads}")));
    }
    let allowed = allowed_cpus();
    let pool = if let Some(s) = cpuset {
        let cpus = parse_cpuset(s)?;
        if let Some(allowed) = &allowed {
            if let Some(c) = cpus.iter().find(|c| !allowed.contains(c)) {
                return Err(Error::config(format!(
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

#[cfg(test)]
mod tests;
