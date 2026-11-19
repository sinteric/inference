//! `/proc/self` counters the server reports at `/stats`.
use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ProcStats {
    pub rss_mib: u64,
    pub hwm_mib: u64,
    pub threads: u64,
    /// Process CPU time (user + system) in milliseconds.
    pub cpu_ms: u64,
    pub voluntary_ctxt_switches: u64,
    pub nonvoluntary_ctxt_switches: u64,
    /// Anonymous (heap, oneDNN scratch, JIT) and file-backed resident memory.
    pub rss_anon_mib: u64,
    pub rss_file_mib: u64,
    /// glibc `mallinfo2`: live bytes (arena and direct `mmap` chunks), bytes
    /// free inside the arenas (resident until trimmed), and the direct
    /// `mmap` share of the live bytes.
    pub heap_in_use_mib: u64,
    pub heap_free_mib: u64,
    pub heap_mmap_mib: u64,
}

pub fn read() -> ProcStats {
    let mut out = ProcStats::default();
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            let mut it = line.split_whitespace();
            let (Some(key), Some(value)) = (it.next(), it.next()) else {
                continue;
            };
            let value: u64 = value.parse().unwrap_or(0);
            match key {
                "VmRSS:" => out.rss_mib = value / 1024,
                "VmHWM:" => out.hwm_mib = value / 1024,
                "Threads:" => out.threads = value,
                "voluntary_ctxt_switches:" => out.voluntary_ctxt_switches = value,
                "nonvoluntary_ctxt_switches:" => out.nonvoluntary_ctxt_switches = value,
                "RssAnon:" => out.rss_anon_mib = value / 1024,
                "RssFile:" => out.rss_file_mib = value / 1024,
                _ => {}
            }
        }
    }
    if let Ok(stat) = std::fs::read_to_string("/proc/self/stat") {
        // Fields after the parenthesised command name; utime and stime are
        // fields 14 and 15 (1-based), in clock ticks.
        if let Some(rest) = stat.rsplit_once(')').map(|(_, r)| r) {
            let f: Vec<&str> = rest.split_whitespace().collect();
            let tick = |i: usize| f.get(i).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
            // rest starts at field 3 (state), so field n is index n - 3.
            let ticks = tick(11) + tick(12);
            // SAFETY: sysconf has no preconditions.
            let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1) as u64;
            out.cpu_ms = ticks * 1000 / hz;
        }
    }
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: mallinfo2 only reads allocator statistics.
        let m = unsafe { libc::mallinfo2() };
        const MIB: usize = 1 << 20;
        out.heap_in_use_mib = ((m.uordblks + m.hblkhd) / MIB) as u64;
        out.heap_free_mib = (m.fordblks / MIB) as u64;
        out.heap_mmap_mib = (m.hblkhd / MIB) as u64;
    }
    out
}
