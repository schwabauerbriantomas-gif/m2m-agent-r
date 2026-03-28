//! GPU Auto-Tune — automatic GPU parameter optimization.
//! Stub. Ported from m2m-vector-search Python.

/// GPU auto-tuner (stub).
pub struct GpuAutoTuner {
    available: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuProfile {
    pub device_name: String,
    pub vram_mb: u64,
    pub optimal_batch_size: usize,
}

impl GpuAutoTuner {
    pub fn new() -> Self { Self { available: false } }

    pub fn detect_gpu(&self) -> Option<GpuProfile> { None }

    pub fn get_optimal_config(&self) -> serde_json::Value {
        serde_json::json!({"batch_size": 100, "threads": 1})
    }
}

pub fn get_gpu_tuner() -> GpuAutoTuner { GpuAutoTuner::new() }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stub() {
        let tuner = GpuAutoTuner::new();
        assert!(tuner.detect_gpu().is_none());
    }
}
