//! Cache and core topology from sysfs. No process-env reads.

use std::fs;
use std::path::Path;

use crate::cpu::threads::physical_cpu_ids;

/// Detected CPU cache / core counts. Zeros mean "unknown" (non-Linux, missing sysfs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuTopology {
    pub physical_cores: usize,
    /// L1 data, per core.
    pub l1d_bytes: usize,
    /// L2 unified, per core (sysfs size of cpu0's L2).
    pub l2_bytes: usize,
    /// L3 unified, package.
    pub l3_bytes: usize,
}

impl CpuTopology {
    pub fn detect() -> Self {
        let physical_cores = physical_cpu_ids().map(|v| v.len().max(1)).unwrap_or(1);
        let mut l1d = 0usize;
        let mut l2 = 0usize;
        let mut l3 = 0usize;
        let base = Path::new("/sys/devices/system/cpu/cpu0/cache");
        if base.is_dir() {
            if let Ok(entries) = fs::read_dir(base) {
                for e in entries.flatten() {
                    let p = e.path();
                    let name = e.file_name();
                    let name = name.to_string_lossy();
                    if !name.starts_with("index") {
                        continue;
                    }
                    let level = read_trim(&p.join("level"));
                    let kind = read_trim(&p.join("type"));
                    let size = parse_cache_size(&read_trim(&p.join("size")));
                    match (level.as_str(), kind.as_str()) {
                        ("1", "Data") => l1d = size,
                        ("2", _) => l2 = size,
                        ("3", _) => l3 = size,
                        _ => {}
                    }
                }
            }
        }
        Self {
            physical_cores,
            l1d_bytes: l1d,
            l2_bytes: l2,
            l3_bytes: l3,
        }
    }
}

fn read_trim(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_default()
        .trim()
        .to_string()
}

pub fn parse_cache_size(s: &str) -> usize {
    let s = s.trim();
    if s.is_empty() {
        return 0;
    }
    let (n, mul) = if let Some(rest) = s.strip_suffix(['K', 'k']) {
        (rest, 1024usize)
    } else if let Some(rest) = s.strip_suffix(['M', 'm']) {
        (rest, 1024 * 1024)
    } else {
        (s, 1usize)
    };
    n.trim().parse::<usize>().unwrap_or(0).saturating_mul(mul)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sysfs_sizes() {
        assert_eq!(parse_cache_size("48K"), 48 * 1024);
        assert_eq!(parse_cache_size("2048K"), 2048 * 1024);
        assert_eq!(parse_cache_size("327680K"), 327680 * 1024);
        assert_eq!(parse_cache_size("1M"), 1024 * 1024);
        assert_eq!(parse_cache_size(""), 0);
    }

    #[test]
    fn detect_has_cores() {
        let t = CpuTopology::detect();
        assert!(t.physical_cores >= 1);
    }
}
