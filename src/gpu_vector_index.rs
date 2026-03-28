//! GPU Vector Index — GPU-resident index for fast search.
//! Stub. Real impl would use wgpu/cudarc.
//! Ported from m2m-vector-search Python.

use ndarray::Array2;

/// GPU vector index (stub).
pub struct GpuVectorIndex {
    dim: usize,
    n_vectors: usize,
    available: bool,
}

impl GpuVectorIndex {
    pub fn new(dim: usize) -> Self {
        Self { dim, n_vectors: 0, available: false }
    }

    pub fn is_available(&self) -> bool { self.available }

    pub fn index(&mut self, _vectors: &Array2<f32>) -> Result<(), String> {
        if !self.available { return Err("GPU not available".into()); }
        self.n_vectors = _vectors.nrows();
        Ok(())
    }

    pub fn len(&self) -> usize { self.n_vectors }

    pub fn search(&self, _query: &[f32], _k: usize) -> Result<Vec<(usize, f32)>, String> {
        if !self.available { return Err("GPU not available".into()); }
        Ok(vec![])
    }

    pub fn memory_usage_mb(&self) -> f64 {
        (self.n_vectors * self.dim * 4) as f64 / (1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stub() {
        let idx = GpuVectorIndex::new(64);
        assert!(!idx.is_available());
        assert!(idx.search(&vec![0.0; 64], 5).is_err());
    }
}
