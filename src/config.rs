//! M2M (Machine-to-Memory) Configuration
//!
//! Centralized configuration for M2M system with auto-detection of devices.

use serde::{Deserialize, Serialize};
use std::any::Any;

/// Detect the best available compute device.
/// Priority: CUDA > Vulkan > CPU
pub fn detect_device() -> &'static str {
    // Check CUDA via nvidia-smi (available on Windows/Linux with NVIDIA driver)
    if std::process::Command::new("nvidia-smi")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        return "cuda";
    }
    "cpu"
}

/// Quantization algorithm selector.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum QuantAlgorithm {
    #[default]
    TurboQuant,
    PolarQuant,
    None,
}

/// Search backend selector.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum SearchBackend {
    #[default]
    Hrm2,
    Hnsw,
    Lsh,
    Quantized,
}

/// Semantic search fusion method.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum FusionMethod {
    #[default]
    Rrf,
    Weighted,
    VectorOnly,
    Bm25Only,
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

    // --- Search Configuration ---
    pub search_backend: SearchBackend,
    pub hrm2_n_coarse: usize,
    pub hrm2_n_fine: usize,
    pub hrm2_n_probe: usize,

    // --- HNSW Configuration ---
    pub enable_hnsw: bool,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,

    // --- LSH Configuration ---
    pub enable_lsh: bool,
    pub lsh_n_tables: usize,
    pub lsh_n_projections: usize,

    // --- Quantization Configuration ---
    pub enable_quantization: bool,
    pub quant_algorithm: QuantAlgorithm,
    pub quant_bits: u8,
    pub quant_projections: usize,
    pub quant_seed: u64,
    pub quant_search_fraction: f64,

    // --- GraphSplat Configuration ---
    pub enable_graph: bool,
    pub graph_max_neighbors: usize,
    pub graph_traverse_depth: usize,
    pub graph_boost_weight: f64,

    // --- Semantic Memory Configuration ---
    pub enable_semantic_memory: bool,
    pub semantic_fusion: FusionMethod,
    pub semantic_vector_weight: f64,
    pub semantic_bm25_weight: f64,
    pub semantic_bm25_k1: f64,
    pub semantic_bm25_b: f64,
    pub semantic_decay_halflife: f64,

    // --- MapReduce Indexing ---
    pub enable_mapreduce: bool,
    pub mapreduce_n_chunks: usize,

    // --- Auto-Scaling ---
    pub enable_auto_scaling: bool,
    pub autoscale_min_nodes: usize,
    pub autoscale_max_nodes: usize,
    pub autoscale_cooldown_secs: f64,

    // --- Quality Reflection ---
    pub enable_quality_reflection: bool,
    pub quality_recall_target: f64,

    // --- Training Configuration ---
    pub enable_training: bool,
    pub training_epochs: usize,
    pub training_eval_interval: usize,
    pub training_save_interval: usize,
    pub training_noise_augmentation: bool,
    pub training_matryoshka_dims: Vec<usize>,
    pub training_distillation: bool,

    // --- Data Lake ---
    pub enable_data_lake: bool,
    pub data_lake_max_entries: usize,
    pub data_lake_compress: bool,

    // --- Entity Extraction ---
    pub enable_entity_extraction: bool,
    pub entity_ngram_sizes: Vec<usize>,
    pub entity_min_confidence: f64,

    // --- GPU Acceleration ---
    pub enable_gpu_search: bool,
    pub gpu_batch_size: usize,
    pub gpu_auto_tune: bool,
}

/// Dtype enum to represent numpy-style data types
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum Dtype {
    #[default]
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

impl Default for M2MConfig {
    fn default() -> Self {
        Self {
            // System — auto-detect GPU on startup
            device: detect_device().to_string(),
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

            // Search
            search_backend: SearchBackend::Hrm2,
            hrm2_n_coarse: 16,
            hrm2_n_fine: 64,
            hrm2_n_probe: 4,

            // HNSW
            enable_hnsw: false,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,

            // LSH
            enable_lsh: false,
            lsh_n_tables: 12,
            lsh_n_projections: 8,

            // Quantization
            enable_quantization: true,
            quant_algorithm: QuantAlgorithm::TurboQuant,
            quant_bits: 8,
            quant_projections: 160,
            quant_seed: 42,
            quant_search_fraction: 0.1,

            // Graph
            enable_graph: true,
            graph_max_neighbors: 10,
            graph_traverse_depth: 3,
            graph_boost_weight: 0.05,

            // Semantic Memory
            enable_semantic_memory: true,
            semantic_fusion: FusionMethod::Rrf,
            semantic_vector_weight: 0.7,
            semantic_bm25_weight: 0.3,
            semantic_bm25_k1: 1.5,
            semantic_bm25_b: 0.75,
            semantic_decay_halflife: 86400.0,

            // MapReduce
            enable_mapreduce: true,
            mapreduce_n_chunks: 8,

            // Auto-Scaling
            enable_auto_scaling: false,
            autoscale_min_nodes: 1,
            autoscale_max_nodes: 10,
            autoscale_cooldown_secs: 60.0,

            // Quality
            enable_quality_reflection: true,
            quality_recall_target: 0.95,

            // Training
            enable_training: false,
            training_epochs: 10,
            training_eval_interval: 100,
            training_save_interval: 1000,
            training_noise_augmentation: true,
            training_matryoshka_dims: vec![64, 128, 256, 640],
            training_distillation: false,

            // Data Lake
            enable_data_lake: false,
            data_lake_max_entries: 1_000_000,
            data_lake_compress: true,

            // Entity Extraction
            enable_entity_extraction: true,
            entity_ngram_sizes: vec![2, 3, 4],
            entity_min_confidence: 0.5,

            // GPU
            enable_gpu_search: false,
            gpu_batch_size: 1024,
            gpu_auto_tune: true,
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

    /// Device efectivo para operaciones de busqueda.
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
    /// Stripped-down: no quantization overhead, no graph, no auto-scaling.
    /// Pure HRM2 search with RAM-only memory.
    pub fn simple(device: Option<&str>) -> Self {
        let mut config = Self::default();

        if let Some(d) = device {
            config.device = d.to_string();
        }

        config.enable_3_tier_memory = false;
        config.memory_tier = "ram-only".to_string();
        config.splat_temperature = 0.0;

        // Disable heavy features
        config.enable_quantization = false;
        config.enable_graph = false;
        config.enable_semantic_memory = false;
        config.enable_mapreduce = false;
        config.enable_auto_scaling = false;
        config.enable_quality_reflection = false;
        config.enable_hnsw = false;
        config.enable_lsh = false;

        // Reduce HRM2 overhead
        config.hrm2_n_coarse = 8;
        config.hrm2_n_fine = 32;
        config.hrm2_n_probe = 2;
        config.max_splats = 10000;
        config.n_splats_init = 1000;
        config.knn_k = 32;

        // Reduce memory footprint
        config.langevin_steps = 0;
        config.context_local = 8;
        config.context_medium = 32;
        config.context_global = 128;
        config.hidden_dim = 512;
        config.moe_experts = 2;
        config.moe_active = 1;

        config
    }

    /// Creates an 'Advanced' configuration for Agentic AI workloads.
    ///
    /// Everything enabled: TurboQuant compression, GraphSplat knowledge graph,
    /// Semantic Memory with hybrid BM25+vector, auto-scaling, quality reflection.
    pub fn advanced(device: Option<&str>) -> Self {
        let mut config = Self::default();

        if let Some(d) = device {
            config.device = d.to_string();
        }

        config.enable_3_tier_memory = true;
        config.memory_tier = "3-tier".to_string();
        config.splat_temperature = 0.1;

        // Full quantization with TurboQuant for memory efficiency
        config.enable_quantization = true;
        config.quant_algorithm = QuantAlgorithm::TurboQuant;
        config.quant_bits = 4;
        config.quant_projections = 80;
        config.quant_search_fraction = 0.2;

        // GraphSplat for knowledge graph augmentation
        config.enable_graph = true;
        config.graph_max_neighbors = 20;
        config.graph_traverse_depth = 5;
        config.graph_boost_weight = 0.1;

        // Semantic Memory with RRF fusion and temporal decay
        config.enable_semantic_memory = true;
        config.semantic_fusion = FusionMethod::Rrf;
        config.semantic_vector_weight = 0.6;
        config.semantic_bm25_weight = 0.4;
        config.semantic_decay_halflife = 3600.0; // 1 hour

        // HNSW as secondary index for exact recall
        config.enable_hnsw = true;
        config.hnsw_m = 32;
        config.hnsw_ef_construction = 400;
        config.hnsw_ef_search = 100;

        // MapReduce for bulk indexing
        config.enable_mapreduce = true;
        config.mapreduce_n_chunks = 16;

        // Auto-scaling for distributed deployments
        config.enable_auto_scaling = true;
        config.autoscale_min_nodes = 2;
        config.autoscale_max_nodes = 20;
        config.autoscale_cooldown_secs = 30.0;

        // Quality reflection with high recall target
        config.enable_quality_reflection = true;
        config.quality_recall_target = 0.99;

        // Larger capacity
        config.max_splats = 1_000_000;
        config.n_splats_init = 50000;
        config.knn_k = 128;
        config.hrm2_n_coarse = 32;
        config.hrm2_n_fine = 128;
        config.hrm2_n_probe = 8;

        // Full context windows
        config.context_local = 24;
        config.context_medium = 128;
        config.context_global = 1024;

        config
    }

    /// Creates a 'Training' configuration for embedding model research.
    ///
    /// Enables training loops, evaluation, data lake ingestion, noise augmentation,
    /// and Matryoshka representation learning. Search features are secondary.
    pub fn training(device: Option<&str>) -> Self {
        let mut config = Self::default();

        if let Some(d) = device {
            config.device = d.to_string();
        }

        // Training enabled
        config.enable_training = true;
        config.training_epochs = 50;
        config.training_eval_interval = 50;
        config.training_save_interval = 500;
        config.training_noise_augmentation = true;
        config.training_matryoshka_dims = vec![32, 64, 128, 256, 640];
        config.training_distillation = true;

        // Data Lake for training data management
        config.enable_data_lake = true;
        config.data_lake_max_entries = 10_000_000;
        config.data_lake_compress = true;

        // Entity extraction for metadata enrichment
        config.enable_entity_extraction = true;
        config.entity_ngram_sizes = vec![2, 3, 4, 5];
        config.entity_min_confidence = 0.3;

        // Search features at moderate capacity for evaluation
        config.enable_quantization = false; // No compression during training
        config.enable_graph = false;
        config.enable_semantic_memory = false;
        config.max_splats = 500_000;
        config.n_splats_init = 50_000;
        config.knn_k = 32;
        config.hrm2_n_coarse = 16;
        config.hrm2_n_fine = 64;

        // Larger training batches and sequences
        config.batch_size = 128;
        config.seq_length = 64;
        config.learning_rate = 3e-4;
        config.noise_levels = vec![0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5];

        config
    }

    /// Creates a 'Distributed' configuration for multi-node cluster deployments.
    ///
    /// Enables cluster modules (router, balancer, sharding), auto-scaling,
    /// MapReduce indexing, and inter-node communication.
    pub fn distributed(device: Option<&str>) -> Self {
        let mut config = Self::default();

        if let Some(d) = device {
            config.device = d.to_string();
        }

        // Cluster and distributed features
        config.enable_auto_scaling = true;
        config.autoscale_min_nodes = 3;
        config.autoscale_max_nodes = 50;
        config.autoscale_cooldown_secs = 20.0;

        // MapReduce for distributed indexing
        config.enable_mapreduce = true;
        config.mapreduce_n_chunks = 32;

        // Compression for inter-node transfer efficiency
        config.enable_quantization = true;
        config.quant_algorithm = QuantAlgorithm::TurboQuant;
        config.quant_bits = 4;
        config.quant_projections = 80;
        config.quant_search_fraction = 0.15;

        // Quality monitoring across nodes
        config.enable_quality_reflection = true;
        config.quality_recall_target = 0.97;

        // Large scale
        config.max_splats = 10_000_000;
        config.n_splats_init = 500_000;
        config.knn_k = 64;

        // Semantic search for cross-node queries
        config.enable_semantic_memory = true;
        config.semantic_fusion = FusionMethod::Rrf;
        config.semantic_decay_halflife = 7200.0; // 2 hours

        config
    }

    /// Creates a 'GPU' configuration for CUDA-accelerated search.
    ///
    /// Enables GPU search backends, auto-tuning, and large batch operations.
    /// Falls back to CPU if no GPU is available.
    pub fn gpu(device: Option<&str>) -> Self {
        let mut config = Self::default();

        config.device = device.unwrap_or("cuda").to_string();
        config.enable_cuda = true;
        config.enable_vulkan = false;

        // GPU search acceleration
        config.enable_gpu_search = true;
        config.gpu_batch_size = 4096;
        config.gpu_auto_tune = true;

        // HNSW on GPU for high-throughput ANN
        config.enable_hnsw = true;
        config.hnsw_m = 48;
        config.hnsw_ef_construction = 800;
        config.hnsw_ef_search = 200;

        // Full quantization pipeline
        config.enable_quantization = true;
        config.quant_algorithm = QuantAlgorithm::TurboQuant;
        config.quant_bits = 4;
        config.quant_projections = 160;
        config.quant_search_fraction = 0.05;

        // Graph and semantic at scale
        config.enable_graph = true;
        config.graph_max_neighbors = 32;
        config.graph_traverse_depth = 5;
        config.enable_semantic_memory = true;

        // Large capacity with GPU memory
        config.max_splats = 5_000_000;
        config.n_splats_init = 100_000;
        config.knn_k = 256;
        config.hrm2_n_coarse = 64;
        config.hrm2_n_fine = 256;
        config.hrm2_n_probe = 16;

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
        assert!(config.device == "cpu" || config.device == "cuda");
        assert!(config.enable_quantization);
        assert!(config.enable_graph);
        assert!(config.enable_semantic_memory);
    }

    #[test]
    fn test_simple_config() {
        let config = M2MConfig::simple(None);
        assert!(!config.enable_3_tier_memory);
        assert!(!config.enable_quantization);
        assert!(!config.enable_graph);
        assert!(!config.enable_semantic_memory);
        assert_eq!(config.memory_tier, "ram-only");
        assert_eq!(config.max_splats, 10000);
    }

    #[test]
    fn test_advanced_config() {
        let config = M2MConfig::advanced(Some("cuda"));
        assert!(config.enable_3_tier_memory);
        assert!(config.enable_quantization);
        assert!(config.enable_graph);
        assert!(config.enable_semantic_memory);
        assert!(config.enable_hnsw);
        assert!(config.enable_auto_scaling);
        assert_eq!(config.device, "cuda");
        assert_eq!(config.quant_bits, 4);
        assert_eq!(config.max_splats, 1_000_000);
        assert_eq!(config.hnsw_m, 32);
    }

    #[test]
    fn test_finalized_auto_device() {
        let mut config = M2MConfig::default();
        config.device = "auto".to_string();
        config.finalize();
        // detect_device() probes nvidia-smi — will find "cuda" if GPU present, else "cpu"
        assert!(config.device == "cuda" || config.device == "cpu");
    }

    #[test]
    fn test_finalized_cuda_device() {
        let mut config = M2MConfig::default();
        config.device = "cuda".to_string();
        config.finalize();
        assert!(config.enable_cuda);
    }

    #[test]
    fn test_training_config() {
        let config = M2MConfig::training(None);
        assert!(config.enable_training);
        assert!(config.enable_data_lake);
        assert!(config.training_distillation);
        assert!(config.training_noise_augmentation);
        assert!(!config.enable_quantization); // No compression during training
        assert_eq!(config.batch_size, 128);
        assert_eq!(config.training_epochs, 50);
        assert!(config.training_matryoshka_dims.contains(&640));
    }

    #[test]
    fn test_distributed_config() {
        let config = M2MConfig::distributed(None);
        assert!(config.enable_auto_scaling);
        assert!(config.enable_mapreduce);
        assert!(config.enable_quantization);
        assert_eq!(config.max_splats, 10_000_000);
        assert_eq!(config.autoscale_max_nodes, 50);
        assert_eq!(config.mapreduce_n_chunks, 32);
    }

    #[test]
    fn test_gpu_config() {
        let config = M2MConfig::gpu(None);
        assert!(config.enable_cuda);
        assert!(config.enable_gpu_search);
        assert!(config.enable_hnsw);
        assert!(config.enable_quantization);
        assert_eq!(config.device, "cuda");
        assert_eq!(config.gpu_batch_size, 4096);
        assert_eq!(config.max_splats, 5_000_000);
        assert_eq!(config.knn_k, 256);
    }

    /// Validate each preset can create a SplatStore and run add_splat + find_neighbors
    #[test]
    fn test_all_presets_runtime() {
        use crate::splats::SplatStore;
        use ndarray::{Array1, Array2};

        let presets: Vec<(&str, M2MConfig)> = vec![
            ("default", M2MConfig::default()),
            ("simple", M2MConfig::simple(None)),
            ("advanced", M2MConfig::advanced(None)),
            ("training", M2MConfig::training(None)),
            ("distributed", M2MConfig::distributed(None)),
            ("gpu", M2MConfig::gpu(None)),
        ];

        for (name, mut config) in presets {
            // Force CPU so tests work without GPU
            config.device = "cpu".to_string();
            config.enable_cuda = false;
            config.enable_vulkan = false;
            config.enable_gpu_search = false;
            // Scale cap: distributed pide 10M splats (25.6 GB de una sola
            // preallocación) y gpu 5M (12.8 GB). Este test verifica el
            // runtime del preset con 2 splats — la escala no aporta nada y
            // aborta runners de CI con poca RAM.
            config.max_splats = config.max_splats.min(10_000);
            config.n_splats_init = config.n_splats_init.min(config.max_splats);
            config.finalize();

            let mut store = SplatStore::new(config);

            let vec1 = Array2::from_shape_vec((1, 640), vec![0.1f32; 640]).unwrap();
            let vec2 = Array2::from_shape_vec((1, 640), vec![0.2f32; 640]).unwrap();

            assert!(store.add_splat(&vec1), "{name}: first add_splat failed");
            assert!(store.add_splat(&vec2), "{name}: second add_splat failed");

            store.build_index();

            let query = Array1::from_vec(vec![0.1f32; 640]);
            let results = store.find_neighbors(&query.view(), 2);
            assert!(!results.is_empty(), "{name}: find_neighbors empty");
            assert_eq!(results[0].index, 0, "{name}: nearest should be vec1");
        }
    }

    // ─── Per-preset feature validation tests ───

    #[test]
    fn test_simple_preset_features() {
        let c = M2MConfig::simple(None);
        // Simple should have heavy features DISABLED
        assert!(!c.enable_quantization, "simple: quantization should be off");
        assert!(!c.enable_graph, "simple: graph should be off");
        assert!(
            !c.enable_semantic_memory,
            "simple: semantic memory should be off"
        );
        assert!(!c.enable_hnsw, "simple: hnsw should be off");
        assert!(!c.enable_lsh, "simple: lsh should be off");
        assert!(
            !c.enable_3_tier_memory,
            "simple: 3-tier memory should be off"
        );
        assert!(!c.enable_auto_scaling, "simple: auto-scaling should be off");
        assert!(!c.enable_mapreduce, "simple: mapreduce should be off");
        // Compact footprint
        assert_eq!(c.max_splats, 10_000, "simple: max_splats should be 10K");
        assert_eq!(c.knn_k, 32, "simple: knn_k should be 32");
        assert_eq!(c.moe_experts, 2, "simple: moe_experts should be 2");
        assert!(c.langevin_steps == 0, "simple: langevin_steps should be 0");
    }

    #[test]
    fn test_advanced_preset_features() {
        let c = M2MConfig::advanced(None);
        // Advanced should have EVERYTHING enabled
        assert!(c.enable_quantization, "advanced: quantization should be on");
        assert!(c.enable_graph, "advanced: graph should be on");
        assert!(
            c.enable_semantic_memory,
            "advanced: semantic memory should be on"
        );
        assert!(c.enable_hnsw, "advanced: hnsw should be on");
        assert!(
            c.enable_3_tier_memory,
            "advanced: 3-tier memory should be on"
        );
        assert!(c.enable_auto_scaling, "advanced: auto-scaling should be on");
        assert!(c.enable_mapreduce, "advanced: mapreduce should be on");
        assert!(
            c.enable_quality_reflection,
            "advanced: quality reflection should be on"
        );
        // Quantization specifics
        assert_eq!(c.quant_algorithm, QuantAlgorithm::TurboQuant);
        assert_eq!(c.quant_bits, 4);
        // Semantic fusion
        assert_eq!(c.semantic_fusion, FusionMethod::Rrf);
        assert!((c.semantic_vector_weight - 0.6).abs() < 1e-6);
        assert!((c.semantic_bm25_weight - 0.4).abs() < 1e-6);
        // Large capacity
        assert_eq!(c.max_splats, 1_000_000);
        assert_eq!(c.knn_k, 128);
    }

    #[test]
    fn test_training_preset_features() {
        let c = M2MConfig::training(None);
        // Training-specific features
        assert!(c.enable_training, "training: training should be on");
        assert!(
            c.training_noise_augmentation,
            "training: noise augmentation should be on"
        );
        assert!(
            c.training_distillation,
            "training: distillation should be on"
        );
        assert!(c.enable_data_lake, "training: data lake should be on");
        assert!(
            c.enable_entity_extraction,
            "training: entity extraction should be on"
        );
        // Search features off during training
        assert!(
            !c.enable_quantization,
            "training: quantization should be off"
        );
        assert!(!c.enable_graph, "training: graph should be off");
        assert!(
            !c.enable_semantic_memory,
            "training: semantic memory should be off"
        );
        // Matryoshka dims
        assert_eq!(c.training_matryoshka_dims, vec![32, 64, 128, 256, 640]);
        // Noise levels for augmentation
        assert!(
            !c.noise_levels.is_empty(),
            "training: should have noise levels"
        );
    }

    #[test]
    fn test_distributed_preset_features() {
        let c = M2MConfig::distributed(None);
        // Distributed-specific
        assert!(
            c.enable_auto_scaling,
            "distributed: auto-scaling should be on"
        );
        assert!(c.enable_mapreduce, "distributed: mapreduce should be on");
        assert!(
            c.enable_quantization,
            "distributed: quantization should be on"
        );
        assert!(
            c.enable_quality_reflection,
            "distributed: quality reflection should be on"
        );
        assert!(
            c.enable_semantic_memory,
            "distributed: semantic memory should be on"
        );
        // Scale
        assert_eq!(c.max_splats, 10_000_000);
        assert!(c.autoscale_max_nodes >= 10);
        assert!(c.mapreduce_n_chunks >= 8);
    }

    #[test]
    fn test_gpu_preset_features() {
        let c = M2MConfig::gpu(None);
        // GPU-specific
        assert!(c.enable_cuda, "gpu: cuda should be on");
        assert!(c.enable_gpu_search, "gpu: gpu search should be on");
        assert!(c.enable_hnsw, "gpu: hnsw should be on");
        assert!(c.enable_quantization, "gpu: quantization should be on");
        assert!(c.enable_graph, "gpu: graph should be on");
        assert!(
            c.enable_semantic_memory,
            "gpu: semantic memory should be on"
        );
        // GPU tuning
        assert_eq!(c.gpu_batch_size, 4096);
        assert!(c.gpu_auto_tune);
        // Large capacity
        assert_eq!(c.max_splats, 5_000_000);
        assert_eq!(c.knn_k, 256);
        assert_eq!(c.hrm2_n_coarse, 64);
    }

    // ─── Dataset transformer integration test ───

    #[test]
    fn test_ingest_with_transformer() {
        use crate::splats::SplatStore;
        use ndarray::Array2;

        let mut config = M2MConfig::default();
        config.latent_dim = 4;
        config.max_splats = 1000;
        config.device = "cpu".to_string();
        config.enable_cuda = false;
        config.finalize();

        let mut store = SplatStore::new(config);

        // Create test data: 3 clear clusters
        let data = Array2::from_shape_vec((90, 4), {
            let mut v = Vec::new();
            // Cluster A: around (1,0,0,0)
            for i in 0..30 {
                v.extend_from_slice(&[1.0 + i as f32 * 0.01, 0.0, 0.0, 0.0]);
            }
            // Cluster B: around (0,1,0,0)
            for i in 0..30 {
                v.extend_from_slice(&[0.0, 1.0 + i as f32 * 0.01, 0.0, 0.0]);
            }
            // Cluster C: around (0,0,1,0)
            for i in 0..30 {
                v.extend_from_slice(&[0.0, 0.0, 1.0 + i as f32 * 0.01, 0.0]);
            }
            v
        })
        .unwrap();

        let result = store.ingest_with_transformer(&data, 3, 42);
        assert!(
            result.is_ok(),
            "ingest_with_transformer failed: {:?}",
            result.err()
        );

        let (n_splats, compression, stats) = result.unwrap();
        assert!(n_splats > 0, "Should produce splats");
        assert!(
            n_splats <= 3,
            "Should produce at most 3 clusters, got {}",
            n_splats
        );
        assert!(stats.original_count == 90);
        assert!(compression > 0.0);

        // Store should be searchable
        store.build_index();
        let query = ndarray::Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let results = store.find_neighbors(&query.view(), n_splats);
        assert!(
            !results.is_empty(),
            "Should find neighbors after transformer ingest"
        );
    }

    #[test]
    fn test_ingest_hierarchical() {
        use crate::splats::SplatStore;
        use ndarray::Array2;

        let mut config = M2MConfig::default();
        config.latent_dim = 4;
        config.max_splats = 1000;
        config.device = "cpu".to_string();
        config.enable_cuda = false;
        config.finalize();

        let mut store = SplatStore::new(config);

        // More data for hierarchical to be meaningful
        let data = Array2::from_shape_vec((200, 4), {
            let mut v = Vec::new();
            for i in 0..200 {
                let cluster = i % 5;
                let base = match cluster {
                    0 => [1.0, 0.0, 0.0, 0.0],
                    1 => [0.0, 1.0, 0.0, 0.0],
                    2 => [0.0, 0.0, 1.0, 0.0],
                    3 => [0.0, 0.0, 0.0, 1.0],
                    _ => [0.5, 0.5, 0.0, 0.0],
                };
                for b in base {
                    v.push(b + (i as f32 * 0.001));
                }
            }
            v
        })
        .unwrap();

        let result = store.ingest_hierarchical(&data, 10, 2, 42);
        assert!(
            result.is_ok(),
            "ingest_hierarchical failed: {:?}",
            result.err()
        );

        let (n_splats, _, stats) = result.unwrap();
        assert!(n_splats > 0);
        assert_eq!(stats.original_count, 200);
    }

    #[test]
    fn test_transformer_dimension_mismatch() {
        use crate::splats::SplatStore;
        use ndarray::Array2;

        let mut config = M2MConfig::default();
        config.latent_dim = 8;
        config.max_splats = 100;
        config.finalize();

        let mut store = SplatStore::new(config);

        // Wrong dimension
        let data = Array2::zeros((10, 4));
        let result = store.ingest_with_transformer(&data, 3, 42);
        assert!(result.is_err(), "Should fail with dimension mismatch");
    }

    // ─── Subsystem initialization tests per preset ───

    /// Helper: create a store with CPU-forced config from a preset
    fn store_from_preset(mut config: M2MConfig) -> crate::splats::SplatStore {
        config.device = "cpu".to_string();
        config.enable_cuda = false;
        config.enable_vulkan = false;
        config.enable_gpu_search = false;
        // Estos tests verifican SUBSYSTEM FLAGS por preset, no escala. Los
        // presets distributed/gpu piden max_splats = 10M; a 2560 B/splat
        // (dim 320, mean+covarianza f32) eso es UNA sola preallocación de
        // 25.6 GB que aborta runners de CI con poca RAM.
        config.max_splats = 10_000;
        config.n_splats_init = config.n_splats_init.min(config.max_splats);
        config.finalize();
        crate::splats::SplatStore::new(config)
    }

    #[test]
    fn test_simple_preset_no_subsystems() {
        let store = store_from_preset(M2MConfig::simple(None));
        assert!(
            !store.has_quantization(),
            "simple: no quantization subsystem"
        );
        assert!(!store.has_hnsw(), "simple: no HNSW subsystem");
        assert!(!store.has_lsh(), "simple: no LSH subsystem");
        assert!(!store.has_semantic_memory(), "simple: no semantic memory");
    }

    #[test]
    fn test_advanced_preset_subsystems() {
        let store = store_from_preset(M2MConfig::advanced(None));
        assert!(
            store.has_quantization(),
            "advanced: should have quantization"
        );
        assert!(store.has_hnsw(), "advanced: should have HNSW");
        assert!(!store.has_lsh(), "advanced: LSH not in advanced preset");
        assert!(
            store.has_semantic_memory(),
            "advanced: should have semantic memory"
        );
    }

    #[test]
    fn test_training_preset_subsystems() {
        let store = store_from_preset(M2MConfig::training(None));
        assert!(!store.has_quantization(), "training: no quantization");
        assert!(!store.has_hnsw(), "training: no HNSW");
        assert!(!store.has_lsh(), "training: no LSH");
        assert!(!store.has_semantic_memory(), "training: no semantic memory");
    }

    #[test]
    fn test_distributed_preset_subsystems() {
        let store = store_from_preset(M2MConfig::distributed(None));
        assert!(
            store.has_quantization(),
            "distributed: should have quantization"
        );
        assert!(
            !store.has_hnsw(),
            "distributed: no HNSW (not in distributed config)"
        );
        assert!(
            store.has_semantic_memory(),
            "distributed: should have semantic memory"
        );
    }

    #[test]
    fn test_gpu_preset_subsystems() {
        let store = store_from_preset(M2MConfig::gpu(None));
        assert!(store.has_quantization(), "gpu: should have quantization");
        assert!(store.has_hnsw(), "gpu: should have HNSW");
        assert!(
            store.has_semantic_memory(),
            "gpu: should have semantic memory"
        );
    }

    #[test]
    fn test_fused_search_uses_hnsw() {
        use crate::splats::SplatStore;
        use ndarray::Array2;
        let mut config = M2MConfig::advanced(None);
        config.device = "cpu".to_string();
        config.enable_cuda = false;
        config.enable_vulkan = false;
        config.enable_gpu_search = false;
        config.latent_dim = 16;
        config.max_splats = 500;
        config.finalize();

        let mut store = SplatStore::new(config);
        assert!(store.has_hnsw());
        assert!(store.has_quantization());

        // Add data and build all indexes
        let data = Array2::from_shape_fn((100, 16), |(i, j)| ((i * 16 + j) as f32 * 0.01).sin());
        let mut normalized = data.clone();
        for mut row in normalized.rows_mut() {
            let norm = row.dot(&row).sqrt().max(1e-10);
            row.mapv_inplace(|v| v / norm);
        }
        store.add_splat(&normalized);
        store.build_index();

        // Fused search should work with HNSW + quantization
        let query = normalized.row(0).to_owned();
        let fused = store.find_neighbors_fused(&query.view(), 5);
        assert!(!fused.is_empty(), "Fused search should return results");
        assert_eq!(fused[0].index, 0, "Fused: nearest should be index 0");
    }

    #[test]
    fn test_simple_preset_no_fused_overhead() {
        use crate::splats::SplatStore;
        use ndarray::Array2;

        let mut config = M2MConfig::simple(None);
        config.device = "cpu".to_string();
        config.enable_cuda = false;
        config.latent_dim = 16;
        config.max_splats = 500;
        config.finalize();

        let mut store = SplatStore::new(config);
        // Simple should have NO subsystems — fused falls back to linear scan only
        assert!(!store.has_quantization());
        assert!(!store.has_hnsw());
        assert!(!store.has_lsh());

        let data = Array2::from_shape_fn((50, 16), |(i, j)| ((i * 16 + j) as f32 * 0.01).sin());
        store.add_splat(&data);
        store.build_index();

        let query = ndarray::Array1::from_vec(data.row(0).to_vec());
        let fused = store.find_neighbors_fused(&query.view(), 3);
        assert!(!fused.is_empty());
        assert_eq!(fused[0].index, 0);
    }
}
