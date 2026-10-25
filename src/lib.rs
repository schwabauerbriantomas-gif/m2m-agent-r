pub mod agent_r;
pub mod agent_swarm;
pub mod api_server;
pub mod clustering;
pub mod config;
pub mod dataset_transformer;
pub mod encoding;
pub mod hnsw_index;
pub mod hrm2_engine;
pub mod interfaces;
pub mod lsh_index;
/// RSI Capa 2: crítico LLM (glm-5.3-flash) envuelto en la lib.
/// Feature "critic" = llamada real; sin ella, stub Unavailable.
pub mod meta;
/// RSI curiosidad guiada: preguntas autónomas en puntos críticos.
pub mod meta_curiosity;
/// RSI Capa 3: evolución acotada de parámetros (STOP/AlphaEvolve viable).
pub mod meta_evolve;
pub mod quant;
pub mod quantization;
pub mod receipts;
pub mod splat_types;
pub mod splats;

pub use config::M2MConfig;
pub use splat_types::{GaussianSplat, SplatCluster, SplatEmbedding};
pub use splats::SplatStore;
