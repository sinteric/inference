//! The activation buffer pool: graphs take a buffer, hand it back when
//! done, and reserve ahead of a shape so the hot path does not allocate.

use super::{touch_pages, OnednnCpu};
use crate::Result;
use std::sync::atomic::Ordering;

impl OnednnCpu {
    pub(crate) fn buffer(&self, len: usize) -> Result<Vec<f32>> {
        self.check_workspace("activation buffer", &[len, 4])?;
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(buffer) = pool.buffers.get_mut(&len).and_then(Vec::pop) {
            pool.buffer_bytes -= buffer.capacity() * 4;
            self.retained
                .pool
                .store(pool.buffer_bytes, Ordering::Relaxed);
            return Ok(buffer);
        }
        drop(pool);
        crate::try_vec(len)
    }

    pub(crate) fn recycle(&self, buffer: Vec<f32>) {
        const PER_LEN: usize = 4;
        let limit = self.workspace_limit;
        let bytes = buffer.capacity() * 4;
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if buffer.is_empty()
            || bytes > limit
            || pool.buffer_bytes + bytes > limit
            || pool.buffers.get(&buffer.len()).map_or(0, Vec::len) >= PER_LEN
            || self.exceeds_budget(bytes)
        {
            return;
        }
        pool.buffer_bytes += bytes;
        self.retained
            .pool
            .store(pool.buffer_bytes, Ordering::Relaxed);
        pool.buffers.entry(buffer.len()).or_default().push(buffer);
    }

    pub(crate) fn buffer_u16(&self, len: usize) -> Result<Vec<u16>> {
        self.check_workspace("BF16 activation buffer", &[len, 2])?;
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(buffer) = pool.buffers_u16.get_mut(&len).and_then(Vec::pop) {
            pool.buffer_bytes -= buffer.capacity() * 2;
            self.retained
                .pool
                .store(pool.buffer_bytes, Ordering::Relaxed);
            return Ok(buffer);
        }
        drop(pool);
        crate::try_vec(len)
    }

    pub(crate) fn recycle_u16(&self, buffer: Vec<u16>) {
        const PER_LEN: usize = 4;
        let limit = self.workspace_limit;
        let bytes = buffer.capacity() * 2;
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if buffer.is_empty()
            || bytes > limit
            || pool.buffer_bytes + bytes > limit
            || pool.buffers_u16.get(&buffer.len()).map_or(0, Vec::len) >= PER_LEN
            || self.exceeds_budget(bytes)
        {
            return;
        }
        pool.buffer_bytes += bytes;
        self.retained
            .pool
            .store(pool.buffer_bytes, Ordering::Relaxed);
        pool.buffers_u16
            .entry(buffer.len())
            .or_default()
            .push(buffer);
    }

    pub(crate) fn reserve(&self, len: usize, n: usize) -> Result<()> {
        self.check_workspace("reserve activation buffers", &[len, n, 4])?;
        self.check_workspace("reserve activation", &[len, 4])?;
        if len == 0 || n == 0 {
            return Ok(());
        }
        // Top up to `n` pooled buffers of this length. Allocating all `n`
        // on every forward (and dropping the extras on recycle) cost a fresh
        // zero-fill and page faults per forward.
        let have = self
            .pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .buffers
            .get(&len)
            .map_or(0, Vec::len);
        for _ in have..n {
            let mut buf = crate::try_vec::<f32>(len)?;
            touch_pages(&mut buf);
            self.recycle(buf);
        }
        Ok(())
    }

    pub(crate) fn reserve_u16(&self, len: usize, n: usize) -> Result<()> {
        self.check_workspace("reserve BF16 activation buffers", &[len, n, 2])?;
        self.check_workspace("reserve BF16 activation", &[len, 2])?;
        if len == 0 || n == 0 {
            return Ok(());
        }
        // Top up to `n` pooled buffers of this length. Allocating all `n`
        // on every forward (and dropping the extras on recycle) cost a fresh
        // zero-fill and page faults per forward.
        let have = self
            .pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .buffers_u16
            .get(&len)
            .map_or(0, Vec::len);
        for _ in have..n {
            let mut buf = crate::try_vec::<u16>(len)?;
            touch_pages(&mut buf);
            self.recycle_u16(buf);
        }
        Ok(())
    }
}
