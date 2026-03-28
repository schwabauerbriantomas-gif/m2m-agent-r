//! CUDA Search — GPU-accelerated vector search.
//!
//! Stub implementation. Real implementation would use cudarc or cust.
//! Ported from m2m-vector-search Python.

use ndarray::{Array1, Array2};

/// CUDA search result.
#[derive(Debug, Clone)]
pub struct CudaSearchResult {
    pub indices: Vec<usize>,
    pub distances: Vec<f32>,
}

/// CUDA-accelerated vector search (stub).
pub struct CudaSearch {
    dim: usize,
    available: bool,
}

impl CudaSearch {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            available: false, // Would detect CUDA availability
        }
    }

    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Index vectors on GPU.
    pub fn index(&mut self, _vectors: &Array2<f32>) -> Result<(), String> {
        if !self.available {
            return Err("CUDA not available".into());
        }
        Ok(())
    }

    /// Search on GPU.
    pub fn search(&self, _query: &Array1<f32>, _k: usize) -> Result<CudaSearchResult, String> {
        if !self.available {
            return Err("CUDA not available".into());
        }
        Ok(CudaSearchResult { indices: vec![], distances: vec![] })
    }

    /// Get GPU info.
    pub fn gpu_info(&self) -> Option<serde_json::Value> {
        if self.available {
            Some(serde_json::json!({"available": true}))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_unavailable() {
        let cuda = CudaSearch::new(64);
        assert!(!cuda.is_available());
        assert!(cuda.search(&Array1::zeros(64), 5).is_err());
    }
}
