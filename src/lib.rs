pub mod config;
pub mod geometry;
pub mod splat_types;
pub mod encoding;
pub mod clustering;
pub mod interfaces;
pub mod engine;
pub mod energy;
pub mod hnsw_index;
pub mod lsh_index;
pub mod hrm2_engine;
pub mod ebm;
pub mod storage;

pub use config::M2MConfig;
pub use geometry::{normalize_sphere, geodesic_distance};
pub use splat_types::{GaussianSplat, SplatEmbedding, SplatCluster};
