//! What a driver reports about its own process: CPU time and memory.

use anyhow::{Context, Result};
use serde_json::{json, Value};

/// CPU seconds this process has used, on every thread.
pub fn cpu_seconds() -> f64 {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `t` is a valid timespec for the call to fill.
    unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut t) };
    t.tv_sec as f64 + t.tv_nsec as f64 * 1e-9
}

/// The CPUs this process may run on.
pub fn affinity() -> Result<Vec<usize>> {
    // SAFETY: a zeroed cpu_set_t is the empty set; the call fills it.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // SAFETY: `set` is valid for the size passed.
    let rc = unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&set), &mut set) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error()).context("sched_getaffinity");
    }
    // SAFETY: CPU_ISSET reads `set` within CPU_SETSIZE.
    Ok((0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &set) })
        .collect())
}

/// A `kB` line of `/proc/self/status`, in KiB.
fn status_kib(field: &str) -> Result<u64> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    status
        .lines()
        .find_map(|l| l.strip_prefix(field))
        .and_then(|s| s.split_whitespace().next())
        .with_context(|| format!("{field} absent from /proc/self/status"))?
        .parse()
        .with_context(|| format!("{field} in /proc/self/status"))
}

/// Peak resident memory (`VmHWM`), in KiB.
pub fn peak_rss_kib() -> Result<u64> {
    status_kib("VmHWM:")
}

/// `{rss_bytes, peak_rss_bytes}`; a field it cannot read is left out.
pub fn memory() -> Value {
    let mut v = serde_json::Map::new();
    for (field, key) in [("VmRSS:", "rss_bytes"), ("VmHWM:", "peak_rss_bytes")] {
        if let Ok(kib) = status_kib(field) {
            v.insert(key.into(), json!(kib * 1024));
        }
    }
    Value::Object(v)
}

/// `value` rounded to `digits` decimals.
pub fn rounded(value: f64, digits: i32) -> f64 {
    let scale = 10f64.powi(digits);
    (value * scale).round() / scale
}
