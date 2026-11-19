//! Inference-only projection scratch. Every active forward owns a separate lease.
use super::blas::{self, Packed};
use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const CACHE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, serde::Serialize)]
pub struct WorkspaceStats {
    pub cache_limit_bytes: usize,
    pub reserved_cache_bytes: usize,
    pub peak_reserved_cache_bytes: usize,
    pub active_forwards: usize,
    pub peak_active_forwards: usize,
    pub workspaces_created: usize,
    pub projection_hits: usize,
    pub projection_misses: usize,
    pub qkv_rope_hits: usize,
    pub qkv_rope_misses: usize,
    pub qkv_rope_budget_fallbacks: usize,
    pub budget_fallbacks: usize,
}

struct Budget {
    limit: usize,
    used: AtomicUsize,
    peak: AtomicUsize,
    hits: AtomicUsize,
    misses: AtomicUsize,
    fallbacks: AtomicUsize,
    rope_hits: AtomicUsize,
    rope_misses: AtomicUsize,
    rope_fallbacks: AtomicUsize,
}
impl Budget {
    fn reserve(self: &Arc<Self>, bytes: usize) -> Option<Reservation> {
        let mut old = self.used.load(Ordering::Relaxed);
        loop {
            let new = old.checked_add(bytes)?;
            if new > self.limit {
                return None;
            }
            match self
                .used
                .compare_exchange_weak(old, new, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => {
                    self.peak.fetch_max(new, Ordering::Relaxed);
                    return Some(Reservation {
                        budget: self.clone(),
                        bytes,
                    });
                }
                Err(value) => old = value,
            }
        }
    }
}
struct Reservation {
    budget: Arc<Budget>,
    bytes: usize,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
// Fields drop in declaration order: owned storage is released before its quota.
struct Buffer {
    tensor: Tensor,
    _reservation: Reservation,
}

pub(crate) struct Workspace {
    buffers: [Option<Buffer>; 3],
    budget: Arc<Budget>,
    rows: usize,
}
impl Workspace {
    pub(crate) fn project(
        &mut self,
        slot: usize,
        packed: &Packed,
        input: &Tensor,
    ) -> Result<Tensor> {
        ensure!(slot < 2, "projection workspace slot");
        let (rows, _) = input.dims2()?;
        if self.rows != rows {
            self.buffers = [None, None, None];
            self.rows = rows;
        }
        let expected = [rows, packed.n()];
        if self.buffers[slot]
            .as_ref()
            .map(|b| b.tensor.dims() != expected)
            .unwrap_or(false)
        {
            // Shape changes discard old storage before trying a new reservation.
            self.buffers[slot] = None;
        }
        if self.buffers[slot].is_none() {
            self.budget.misses.fetch_add(1, Ordering::Relaxed);
            if let Some(reservation) = self.budget.reserve(packed.output_storage_bytes(rows)?) {
                let tensor = packed.initialized_output(rows)?;
                self.buffers[slot] = Some(Buffer {
                    tensor,
                    _reservation: reservation,
                });
            } else {
                self.budget.fallbacks.fetch_add(1, Ordering::Relaxed);
                return Ok(packed.forward(input)?);
            }
        } else {
            self.budget.hits.fetch_add(1, Ordering::Relaxed);
        }
        let output = &self.buffers[slot].as_ref().unwrap().tensor;
        // This active lease exclusively owns the scratch until its forward completes.
        unsafe { packed.forward_into(input, output)? };
        Ok(output.clone())
    }
    pub(crate) fn qkv_rope(
        &mut self,
        input: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        heads: usize,
    ) -> Result<Tensor> {
        let shape = blas::qkv_rope_shape(input, cos, sin, heads)?;
        let count = shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| anyhow::anyhow!("QKV workspace size overflow"))?;
        let bytes = count
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("QKV workspace byte overflow"))?;
        if self.buffers[2]
            .as_ref()
            .map(|b| b.tensor.dims() != shape)
            .unwrap_or(false)
        {
            self.buffers[2] = None;
        }
        if self.buffers[2].is_none() {
            self.budget.rope_misses.fetch_add(1, Ordering::Relaxed);
            if let Some(reservation) = self.budget.reserve(bytes) {
                let tensor = Tensor::zeros(shape.as_slice(), DType::F32, &Device::Cpu)?;
                self.buffers[2] = Some(Buffer {
                    tensor,
                    _reservation: reservation,
                });
            } else {
                self.budget.rope_fallbacks.fetch_add(1, Ordering::Relaxed);
                return Ok(blas::qkv_rope(input, cos, sin, heads)?);
            }
        } else {
            self.budget.rope_hits.fetch_add(1, Ordering::Relaxed);
        }
        let output = &self.buffers[2].as_ref().unwrap().tensor;
        blas::qkv_rope_into(input, cos, sin, heads, output)?;
        Ok(output.clone())
    }
}

pub(crate) struct Pool {
    budget: Arc<Budget>,
    idle: Mutex<Option<Workspace>>,
    active: AtomicUsize,
    peak_active: AtomicUsize,
    created: AtomicUsize,
}
impl Pool {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            budget: Arc::new(Budget {
                limit: if enabled { CACHE_BYTES } else { 0 },
                used: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
                hits: AtomicUsize::new(0),
                misses: AtomicUsize::new(0),
                fallbacks: AtomicUsize::new(0),
                rope_hits: AtomicUsize::new(0),
                rope_misses: AtomicUsize::new(0),
                rope_fallbacks: AtomicUsize::new(0),
            }),
            idle: Mutex::new(None),
            active: AtomicUsize::new(0),
            peak_active: AtomicUsize::new(0),
            created: AtomicUsize::new(0),
        }
    }
    pub(crate) fn checkout(&self) -> Result<Lease<'_>> {
        // Release this short lock before initialization or model computation.
        let workspace = self
            .idle
            .lock()
            .map_err(|_| anyhow::anyhow!("workspace pool poisoned"))?
            .take();
        let workspace = workspace.unwrap_or_else(|| {
            self.created.fetch_add(1, Ordering::Relaxed);
            Workspace {
                buffers: [None, None, None],
                budget: self.budget.clone(),
                rows: 0,
            }
        });
        let active = self.active.fetch_add(1, Ordering::AcqRel) + 1;
        self.peak_active.fetch_max(active, Ordering::Relaxed);
        Ok(Lease {
            pool: self,
            workspace: Some(workspace),
        })
    }
    pub(crate) fn stats(&self) -> WorkspaceStats {
        WorkspaceStats {
            cache_limit_bytes: self.budget.limit,
            reserved_cache_bytes: self.budget.used.load(Ordering::Acquire),
            peak_reserved_cache_bytes: self.budget.peak.load(Ordering::Relaxed),
            active_forwards: self.active.load(Ordering::Acquire),
            peak_active_forwards: self.peak_active.load(Ordering::Relaxed),
            workspaces_created: self.created.load(Ordering::Relaxed),
            projection_hits: self.budget.hits.load(Ordering::Relaxed),
            projection_misses: self.budget.misses.load(Ordering::Relaxed),
            budget_fallbacks: self.budget.fallbacks.load(Ordering::Relaxed),
            qkv_rope_hits: self.budget.rope_hits.load(Ordering::Relaxed),
            qkv_rope_misses: self.budget.rope_misses.load(Ordering::Relaxed),
            qkv_rope_budget_fallbacks: self.budget.rope_fallbacks.load(Ordering::Relaxed),
        }
    }
}
pub(crate) struct Lease<'a> {
    pool: &'a Pool,
    workspace: Option<Workspace>,
}
impl std::ops::Deref for Lease<'_> {
    type Target = Workspace;
    fn deref(&self) -> &Workspace {
        self.workspace.as_ref().unwrap()
    }
}
impl std::ops::DerefMut for Lease<'_> {
    fn deref_mut(&mut self) -> &mut Workspace {
        self.workspace.as_mut().unwrap()
    }
}
impl Drop for Lease<'_> {
    fn drop(&mut self) {
        let workspace = self.workspace.take().unwrap();
        // Keep at most one idle workspace; deallocation happens outside the lock.
        let unused = match self.pool.idle.lock() {
            Ok(mut idle) if idle.is_none() => {
                *idle = Some(workspace);
                None
            }
            _ => Some(workspace),
        };
        self.pool.active.fetch_sub(1, Ordering::AcqRel);
        drop(unused);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn concurrent_reservations_release_all_bytes_and_never_exceed_limit() {
        let pool = Arc::new(Pool::new(true));
        let exact = pool.budget.reserve(CACHE_BYTES).unwrap();
        assert!(pool.budget.reserve(1).is_none());
        drop(exact);
        let barrier = Arc::new(std::sync::Barrier::new(8));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let pool = pool.clone();
                let barrier = barrier.clone();
                scope.spawn(move || {
                    barrier.wait();
                    for _ in 0..1000 {
                        let reservation = pool.budget.reserve(1024).unwrap();
                        assert!(pool.stats().reserved_cache_bytes <= CACHE_BYTES);
                        drop(reservation);
                    }
                });
            }
        });
        assert_eq!(pool.stats().reserved_cache_bytes, 0);
        assert!(pool.stats().peak_reserved_cache_bytes <= CACHE_BYTES);
    }

    #[test]
    fn panic_releases_active_lease_and_reservations() {
        let pool = Pool::new(true);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _lease = pool.checkout().unwrap();
            let _reservation = pool.budget.reserve(4096).unwrap();
            panic!("controlled unwinding test");
        }));
        assert!(result.is_err());
        assert_eq!(pool.stats().active_forwards, 0);
        assert_eq!(pool.stats().reserved_cache_bytes, 0);
    }
    #[test]
    fn simultaneous_and_nested_leases_are_distinct_and_budget_is_bounded() {
        let pool = Pool::new(true);
        let first = pool.checkout().unwrap();
        let second = pool.checkout().unwrap();
        assert!(!std::ptr::eq(&*first, &*second));
        assert_eq!(pool.stats().active_forwards, 2);
        let reservation = pool.budget.reserve(CACHE_BYTES).unwrap();
        assert!(pool.budget.reserve(1).is_none());
        drop(reservation);
        assert_eq!(pool.stats().reserved_cache_bytes, 0);
        drop(first);
        drop(second);
        assert_eq!(pool.stats().active_forwards, 0);
        assert_eq!(pool.stats().peak_active_forwards, 2);
        let third = pool.checkout().unwrap();
        assert_eq!(pool.stats().workspaces_created, 2);
        drop(third);
    }
}
