//! M2M (Machine-to-Memory) Configuration
//!
//! Centralized configuration for M2M system with auto-detection of devices.

use serde::{Deserialize, Serialize};
use std::any::Any;

/// Detect the best available compute device.
/// Priority: CUDA > Vulkan > CPU
pub fn detect_device() -> &'static str {
    // TODO: Implement actual device detection
    // For now, return CPU as fallback
    // In production, check for CUDA via rust-cuda or vulkano
    "cpu"
}

/// Configuration for M2M (Machine-to-Memory) system.
///
/// Supports devices: cpu, vulkan, cuda.
/// Use device='auto' for auto-detection (CUDA > Vulkan > CPU).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct M2MConfig {
    // --- System Configuration ---
    pub device: String,
    pub dtype: Dtype,

    // --- Latent Space Configuration ---
    pub latent_dim: usize,
    pub n_splats_init: usize,
    pub max_splats: usize,
    pub knn_k: usize,

    // --- Splat Parameters ---
    pub init_alpha: f64,
    pub init_kappa: f64,
    pub min_kappa: f64,
    pub max_kappa: f64,

    // --- Temperature (for exploration) ---
    pub splat_temperature: f64,
    pub weight_decay_start: f64,

    // --- Energy Function Weights ---
    pub energy_splat_weight: f64,
    pub energy_geom_weight: f64,
    pub energy_comp_weight: f64,
    pub global_temperature: f64,

    // --- SOC Parameters ---
    pub soc_threshold: f64,
    pub soc_buffer_capacity: usize,
    pub soc_update_interval: usize,
    pub phi_convergence_threshold: f64,

    // --- Hardware Acceleration ---
    pub enable_vulkan: bool,
    pub enable_cuda: bool,
    pub cuda_metric: String,

    // --- Memory Hierarchy ---
    pub enable_3_tier_memory: bool,
    pub memory_tier: String,

    // --- Hierarchical Context ---
    pub context_local: usize,
    pub context_medium: usize,
    pub context_global: usize,

    // --- Decoder Configuration (MoE) ---
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub moe_experts: usize,
    pub moe_active: usize,

    // --- Training Configuration ---
    pub batch_size: usize,
    pub seq_length: usize,
    pub noise_levels: Vec<f64>,

    // --- Optimization ---
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub grad_clip: f64,

    // --- Langevin Dynamics Configuration ---
    pub langevin_steps: usize,
    pub langevin_dt: f64,
    pub langevin_gamma: f64,
    pub langevin_t: f64,

    // --- API Configuration ---
    pub rest_port: u16,
    pub grpc_port: u16,
}

/// Dtype enum to represent numpy-style data types
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Dtype {
    Float32,
    Float64,
    Int32,
    Int64,
}

impl Dtype {
    pub fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Default for Dtype {
    fn default() -> Self {
        Dtype::Float32
    }
}

impl Default for M2MConfig {
    fn default() -> Self {
        Self {
            // System
            device: "cpu".to_string(),
            dtype: Dtype::Float32,

            // Latent Space
            latent_dim: 640,
            n_splats_init: 10000,
            max_splats: 100000,
            knn_k: 64,

            // Splat Parameters
            init_alpha: 1.0,
            init_kappa: 10.0,
            min_kappa: 1.0,
            max_kappa: 50.0,

            // Temperature
            splat_temperature: 0.1,
            weight_decay_start: 1.0,

            // Energy Weights
            energy_splat_weight: 1.0,
            energy_geom_weight: 0.1,
            energy_comp_weight: 0.0,
            global_temperature: 1.0,

            // SOC
            soc_threshold: 0.8,
            soc_buffer_capacity: 1000,
            soc_update_interval: 100,
            phi_convergence_threshold: 0.95,

            // Hardware
            enable_vulkan: false,
            enable_cuda: false,
            cuda_metric: "cosine".to_string(),

            // Memory
            enable_3_tier_memory: true,
            memory_tier: "3-tier".to_string(),

            // Context
            context_local: 12,
            context_medium: 64,
            context_global: 512,

            // MoE
            vocab_size: 50257,
            hidden_dim: 1024,
            moe_experts: 4,
            moe_active: 2,

            // Training
            batch_size: 32,
            seq_length: 32,
            noise_levels: vec![0.01, 0.05, 0.1, 0.2, 0.5],

            // Optimization
            learning_rate: 1e-4,
            weight_decay: 0.01,
            grad_clip: 1.0,

            // Langevin
            langevin_steps: 200,
            langevin_dt: 0.001,
            langevin_gamma: 0.1,
            langevin_t: 1.0,

            // API
            rest_port: 8080,
            grpc_port: 9090,
        }
    }
}

impl M2MConfig {
    /// Handle device auto-detection and flags.
    pub fn finalize(&mut self) {
        if self.device == "auto" {
            self.device = detect_device().to_string();
        }

        match self.device.as_str() {
            "vulkan" => self.enable_vulkan = true,
            "cuda" => self.enable_cuda = true,
            _ => {}
        }
    }

    /// Device for allocations.
    pub fn compute_device(&self) -> &str {
        if self.enable_cuda {
            "cuda"
        } else {
            "cpu"
        }
    }

    /// Device efectivo para operaciones de búsqueda.
    pub fn effective_device(&self) -> &str {
        if self.enable_cuda {
            "cuda"
        } else if self.enable_vulkan {
            "vulkan"
        } else {
            "cpu"
        }
    }

    /// Creates a 'Simple' configuration for edge computing.
    ///
    /// Disables heavy features like SOC, 3-tier memory, and Langevin dynamics.
    pub fn simple(device: Option<&str>) -> Self {
        let mut config = Self::default();
        
        if let Some(d) = device {
            config.device = d.to_string();
        }

        config.enable_3_tier_memory = false;
        config.memory_tier = "ram-only".to_string();
        config.splat_temperature = 0.0;

        config
    }

    /// Creates an 'Advanced' configuration for Agentic AI workloads.
    ///
    /// Enables 3-tier memory, SOC, EBM, and generative capabilities.
    pub fn advanced(device: Option<&str>) -> Self {
        let mut config = Self::default();
        
        if let Some(d) = device {
            config.device = d.to_string();
        }

        config.enable_3_tier_memory = true;
        config.memory_tier = "3-tier".to_string();
        config.splat_temperature = 0.1;

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = M2MConfig::default();
        assert_eq!(config.latent_dim, 640);
        assert_eq!(config.knn_k, 64);
        assert_eq!(config.device, "cpu");
    }

    #[test]
    fn test_simple_config() {
        let config = M2MConfig::simple(None);
        assert!(!config.enable_3_tier_memory);
        assert_eq!(config.memory_tier, "ram-only");
    }

    #[test]
    fn test_advanced_config() {
        let config = M2MConfig::advanced(Some("cuda"));
        assert!(config.enable_3_tier_memory);
        assert_eq!(config.device, "cuda");
    }
}
