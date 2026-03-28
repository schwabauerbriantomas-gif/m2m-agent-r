//! GPU Hierarchical Search — multi-level GPU search.
//! Stub. Ported from m2m-vector-search Python.

/// GPU hierarchical search (stub).
pub struct GpuHierarchicalSearch {
    available: bool,
}

impl GpuHierarchicalSearch {
    pub fn new() -> Self { Self { available: false } }
    pub fn is_available(&self) -> bool { self.available }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stub() { let _ = GpuHierarchicalSearch::new(); }
}
