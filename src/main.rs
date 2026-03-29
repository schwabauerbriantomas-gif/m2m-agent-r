//! M2M Vector Search CLI — AI-friendly interface.

use clap::{Parser, Subcommand};
use ndarray::{Array1, Array2};
use m2m_vector_search::{
    M2MConfig, SplatStore,
    storage::persistence::{M2MPersistence, StorageBackend},
    quantization::{QuantConfig, QuantAlgorithm, QuantizedStore},
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "m2m", version, about = "M2M Vector Search — Gaussian Splat semantic memory")]
struct Cli {
    /// Storage directory (default: ./m2m_data)
    #[arg(long, global = true, default_value = "./m2m_data")]
    data_dir: String,

    /// Latent dimension (default: 64)
    #[arg(long, global = true, default_value = "64")]
    dim: usize,

    /// Max splats capacity
    #[arg(long, global = true, default_value = "100000")]
    max_splats: usize,

    /// Metadata storage backend: sqlite, json
    #[arg(long, global = true, default_value = "sqlite")]
    backend: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Add vectors from a binary file (rows: u64, cols: u64, f32 data)
    Index {
        /// Path to .bin vector file
        #[arg(short, long)]
        input: PathBuf,

        /// Shard name (default: "default")
        #[arg(short, long, default_value = "default")]
        shard: String,
    },

    /// Search for k nearest neighbors
    Search {
        /// Query vector as comma-separated f32 values
        #[arg(short, long)]
        query: String,

        /// Number of neighbors (default: 10)
        #[arg(short, long, default_value = "10")]
        k: usize,

        /// Output format: json, table, or compact
        #[arg(short, long, default_value = "json")]
        format: String,
    },

    /// Search using a query vector from a binary file
    SearchFile {
        /// Path to .bin query vector file (1 row)
        #[arg(short, long)]
        input: PathBuf,

        /// Number of neighbors
        #[arg(short, long, default_value = "10")]
        k: usize,

        /// Output format
        #[arg(short = 'f', long, default_value = "json")]
        format: String,
    },

    /// Show store statistics
    Status {
        /// Detailed output
        #[arg(short, long)]
        verbose: bool,
    },

    /// SOC: check system criticality
    SocCheck,

    /// SOC: trigger avalanche reorganization
    SocAvalanche {
        /// Seed point index (optional — defaults to highest energy)
        #[arg(short, long)]
        seed: Option<usize>,
    },

    /// SOC: relax the system toward lower energy
    SocRelax {
        /// Number of relaxation iterations
        #[arg(short, long, default_value = "10")]
        iterations: usize,
    },

    /// Save current state to disk
    Save {
        /// Shard name
        #[arg(short, long, default_value = "default")]
        shard: String,
    },

    /// Load state from disk
    Load {
        /// Shard name
        #[arg(short, long, default_value = "default")]
        shard: String,
    },

    /// List all stored shards
    List,

    /// Save document metadata
    DocAdd {
        /// Document ID
        #[arg(short, long)]
        id: String,

        /// Document text content
        #[arg(short, long)]
        text: String,

        /// JSON metadata (optional)
        #[arg(short, long)]
        metadata: Option<String>,
    },

    /// Get document metadata
    DocGet {
        /// Document ID
        #[arg(short, long)]
        id: String,
    },

    /// Delete a document (soft delete)
    DocDel {
        /// Document ID
        #[arg(short, long)]
        id: String,
    },

    /// Backup all data
    Backup {
        /// Backup destination directory
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Quantize vectors with TurboQuant and search compressed codes
    QuantIndex {
        /// Path to .bin vector file
        #[arg(short, long)]
        input: PathBuf,

        /// Bits per value: 3, 4, 6, or 8 (default: 8, near-lossless)
        #[arg(short, long, default_value = "8")]
        bits: u8,

        /// Algorithm: turbo (2-stage) or polar (single-stage)
        #[arg(short, long, default_value = "turbo")]
        algorithm: String,

        /// Random seed
        #[arg(short, long, default_value = "42")]
        seed: u64,
    },

    /// Search using quantized codes
    QuantSearch {
        /// Query vector as comma-separated f32 values
        #[arg(short, long)]
        query: String,

        /// Number of neighbors (default: 10)
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,
    },

    /// Show quantization statistics
    QuantStatus,

    /// Show GPU/CUDA status and info
    GpuInfo,

    /// Ingest dataset via DatasetTransformer (normalize + KMeans → splat centroids)
    Ingest {
        /// Path to .bin vector file
        #[arg(short, long)]
        input: PathBuf,

        /// Number of clusters (default: sqrt of dataset size)
        #[arg(short, long)]
        n_clusters: Option<usize>,

        /// Random seed
        #[arg(short, long, default_value = "42")]
        seed: u64,
    },

    /// Ingest dataset with hierarchical 2-level KMeans
    IngestHierarchical {
        /// Path to .bin vector file
        #[arg(short, long)]
        input: PathBuf,

        /// Number of top-level clusters
        #[arg(short = 'n', long, default_value = "10")]
        n_clusters: usize,

        /// Minimum cluster size before splitting
        #[arg(short, long, default_value = "2")]
        min_cluster_size: usize,

        /// Random seed
        #[arg(short, long, default_value = "42")]
        seed: u64,
    },

    /// Ingest via Leader Clustering — O(n) single-pass, no KMeans iterations
    IngestLeader {
        /// Path to .bin vector file
        #[arg(short, long)]
        input: PathBuf,

        /// Target number of clusters (soft target, actual may differ)
        #[arg(short = 'n', long, default_value = "100")]
        target_clusters: usize,

        /// Cluster radius threshold (auto-computed if omitted)
        #[arg(short = 't', long)]
        threshold: Option<f64>,

        /// Random seed
        #[arg(short, long, default_value = "42")]
        seed: u64,
    },

    /// Search using all enabled backends fused (HRM2 + Quantization + HNSW + LSH)
    FusedSearch {
        /// Query vector as comma-separated f32 values
        #[arg(short, long)]
        query: Option<String>,

        /// Query vector from binary file (u64 rows, u64 cols, f32 data)
        #[arg(long)]
        query_file: Option<PathBuf>,

        /// Number of neighbors
        #[arg(short, long, default_value = "10")]
        k: usize,
    },

    /// Show which subsystems each preset enables
    PresetInfo {
        /// Preset name: simple, advanced, training, distributed, gpu
        #[arg(short, long)]
        preset: Option<String>,
    },

    /// Search using HNSW index only
    HnswSearch {
        /// Query vector as comma-separated f32 values
        #[arg(short, long)]
        query: Option<String>,

        /// Query vector from binary file
        #[arg(long)]
        query_file: Option<PathBuf>,

        /// Number of neighbors
        #[arg(short, long, default_value = "10")]
        k: usize,
    },

    /// Search using LSH index only
    LshSearch {
        /// Query vector as comma-separated f32 values
        #[arg(short, long)]
        query: Option<String>,

        /// Query vector from binary file
        #[arg(long)]
        query_file: Option<PathBuf>,

        /// Number of neighbors
        #[arg(short, long, default_value = "10")]
        k: usize,
    },

    /// Benchmark GPU + DatasetTransformer: ingest + search pipeline
    BenchGpuIngest {
        /// Number of vectors to generate
        #[arg(short = 'n', long, default_value = "100000")]
        n_vectors: usize,

        /// Vector dimension
        #[arg(short = 'd', long, default_value = "640")]
        dim: usize,

        /// Number of clusters for transformer
        #[arg(short = 'k', long, default_value = "100")]
        n_clusters: usize,

        /// Number of search queries
        #[arg(short = 'q', long, default_value = "100")]
        n_queries: usize,
    },

    /// Benchmark GPU vs CPU search performance
    BenchGpu {
        /// Number of vectors to generate
        #[arg(short = 'n', long, default_value = "10000")]
        n_vectors: usize,

        /// Vector dimension
        #[arg(short = 'd', long, default_value = "640")]
        dim: usize,

        /// Number of queries
        #[arg(short = 'q', long, default_value = "100")]
        n_queries: usize,

        /// Top-k
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,

        /// Metric: l2 or cosine
        #[arg(short, long, default_value = "l2")]
        metric: String,
    },

    /// Start HTTP API server for Python bindings
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "8199")]
        port: u16,
    },
}

fn make_config(dim: usize, max_splats: usize) -> M2MConfig {
    let mut c = M2MConfig::default();
    c.latent_dim = dim;
    c.max_splats = max_splats;
    c
}

fn resolve_backend(name: &str) -> StorageBackend {
    match name {
        "json" | "json-file" => StorageBackend::JsonFile,
        "sqlite" => StorageBackend::Sqlite,
        _ => StorageBackend::Sqlite,
    }
}

fn make_persistence(data_dir: &str, backend: &str, rw: bool) -> Result<M2MPersistence, Box<dyn std::error::Error + Send + Sync>> {
    M2MPersistence::with_backend(data_dir, resolve_backend(backend), rw)
}

fn load_or_create_store(data_dir: &str, config: &M2MConfig) -> (SplatStore, Option<M2MPersistence>) {
    let persist = make_persistence(data_dir, "sqlite", true).ok();
    let mut store = SplatStore::new(config.clone());

    // Try loading default shard
    if let Some(ref p) = persist {
        if let Ok(Some(vectors)) = p.load_vectors("default") {
            store.add_splat(&vectors);
            store.build_index();
            eprintln!("[m2m] Loaded {} vectors from default shard", vectors.nrows());
        }
    }

    (store, persist)
}

fn parse_query(s: &str) -> Result<Array1<f32>, String> {
    let vals: Result<Vec<f32>, _> = s.split(',').map(|v| v.trim().parse()).collect();
    match vals {
        Ok(v) => Ok(Array1::from(v)),
        Err(e) => Err(format!("Invalid query vector: {}", e)),
    }
}

fn resolve_query(query: Option<&str>, query_file: Option<&PathBuf>) -> Array1<f32> {
    match (query, query_file) {
        (Some(q), _) => parse_query(q).unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); }),
        (_, Some(f)) => load_query_bin(f).unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); }),
        (None, None) => { eprintln!("Error: provide --query or --query-file"); std::process::exit(1); }
    }
}

fn load_query_bin(path: &PathBuf) -> Result<Array1<f32>, String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8).map_err(|e| format!("Read error: {}", e))?;
    let rows = u64::from_le_bytes(buf8) as usize;
    file.read_exact(&mut buf8).map_err(|e| format!("Read error: {}", e))?;
    let cols = u64::from_le_bytes(buf8) as usize;

    if rows != 1 {
        return Err(format!("Query file must have exactly 1 row, got {}", rows));
    }

    let mut data = vec![0.0f32; cols];
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, cols * 4)
    };
    file.read_exact(bytes).map_err(|e| format!("Read error: {}", e))?;
    Ok(Array1::from(data))
}

fn load_vectors_bin(path: &PathBuf) -> Result<Array2<f32>, String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8).map_err(|e| format!("Read error: {}", e))?;
    let rows = u64::from_le_bytes(buf8) as usize;
    file.read_exact(&mut buf8).map_err(|e| format!("Read error: {}", e))?;
    let cols = u64::from_le_bytes(buf8) as usize;

    let mut data = vec![0.0f32; rows * cols];
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4)
    };
    file.read_exact(bytes).map_err(|e| format!("Read error: {}", e))?;
    Array2::from_shape_vec((rows, cols), data).map_err(|e| format!("Shape error: {}", e))
}

fn format_neighbors(results: &[m2m_vector_search::splats::NeighborResult], format: &str) -> String {
    match format {
        "json" => {
            let entries: Vec<serde_json::Value> = results.iter().map(|r| {
                serde_json::json!({
                    "index": r.index,
                    "distance": (r.distance * 10000.0).round() / 10000.0,
                    "alpha": (r.alpha * 10000.0).round() / 10000.0,
                    "kappa": (r.kappa * 10000.0).round() / 10000.0,
                })
            }).collect();
            serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".into())
        }
        "compact" => {
            results.iter()
                .map(|r| format!("{}:{:.4}", r.index, r.distance))
                .collect::<Vec<_>>()
                .join(" ")
        }
        _ => { // table
            let mut out = String::from("Index\tDistance\tAlpha\t\tKappa\n");
            out.push_str(&"—\t—\t—\t—\n".repeat(1));
            for r in results {
                out.push_str(&format!("{}\t{:.4}\t\t{:.4}\t\t{:.4}\n", r.index, r.distance, r.alpha, r.kappa));
            }
            out
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let config = make_config(cli.dim, cli.max_splats);

    match cli.command {
        Commands::Index { input, shard } => {
            let vectors = match load_vectors_bin(&input) {
                Ok(v) => v,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            eprintln!("[m2m] Loaded {} vectors ({}D) from {}", vectors.nrows(), vectors.ncols(), input.display());

            let persist = make_persistence(&cli.data_dir, &cli.backend, true)
                .unwrap_or_else(|e| { eprintln!("Storage error: {}", e); std::process::exit(1); });

            persist.save_vectors(&vectors, &shard)
                .unwrap_or_else(|e| { eprintln!("Save error: {}", e); std::process::exit(1); });

            // Also index in memory
            let mut store = SplatStore::new(config);
            store.add_splat(&vectors);
            store.build_index();

            println!("{}", serde_json::json!({
                "status": "indexed",
                "shard": shard,
                "n_vectors": vectors.nrows(),
                "dim": vectors.ncols(),
                "n_active": store.n_active(),
                "entropy": (store.entropy() * 10000.0).round() / 10000.0,
            }));
        }

        Commands::Search { query, k, format } => {
            let q = match parse_query(&query) {
                Ok(v) => v,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            let (store, _) = load_or_create_store(&cli.data_dir, &config);
            let results = store.find_neighbors(&q.view(), k);
            println!("{}", format_neighbors(&results, &format));
        }

        Commands::SearchFile { input, k, format } => {
            let q = match load_query_bin(&input) {
                Ok(v) => v,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            let (store, _) = load_or_create_store(&cli.data_dir, &config);
            let results = store.find_neighbors(&q.view(), k);
            println!("{}", format_neighbors(&results, &format));
        }

        Commands::Status { verbose } => {
            let (store, persist) = load_or_create_store(&cli.data_dir, &config);
            let stats = store.get_statistics();

            let mut out = serde_json::json!({
                "n_active": stats.n_active,
                "embedding_dim": stats.embedding_dim,
                "entropy": (store.entropy() * 10000.0).round() / 10000.0,
                "max_splats": cli.max_splats,
            });

            if verbose {
                if let Some(ref p) = persist {
                    out["shards"] = serde_json::json!(p.list_shards());
                }
                let strategy = m2m_vector_search::interfaces::select_index_strategy(
                    stats.n_active, stats.embedding_dim
                );
                out["index_strategy"] = serde_json::json!({
                    "recommended": strategy.recommended,
                    "reason": strategy.reason,
                });
            }

            println!("{}", serde_json::to_string_pretty(&out).unwrap());
        }

        Commands::SocCheck => {
            let (store, _) = load_or_create_store(&cli.data_dir, &config);
            let mu = store.get_mu();
            let alpha = store.get_alpha();
            let kappa = store.get_kappa();

            if mu.is_none() || store.n_active() == 0 {
                println!("{}", serde_json::json!({"error": "No splats loaded"}));
                std::process::exit(1);
            }

            let energy_api = m2m_vector_search::ebm::energy_api::EBMEnergy::new();
            let mut soc = m2m_vector_search::ebm::soc::SOCEngine::new(energy_api, 0.7);
            soc.update_splats(mu.unwrap(), alpha.unwrap(), kappa.unwrap());

            let report = soc.check_criticality();
            println!("{}", serde_json::json!({
                "state": format!("{:?}", report.state),
                "index": (report.index * 10000.0).round() / 10000.0,
                "energy_variance": (report.energy_variance * 10000.0).round() / 10000.0,
                "size_variance": (report.size_variance * 10000.0).round() / 10000.0,
                "needs_relaxation": report.needs_relaxation(),
                "needs_monitoring": report.needs_monitoring(),
            }));
        }

        Commands::SocAvalanche { seed } => {
            let (store, _) = load_or_create_store(&cli.data_dir, &config);
            let mu = store.get_mu();
            let alpha = store.get_alpha();
            let kappa = store.get_kappa();

            if mu.is_none() || store.n_active() == 0 {
                println!("{}", serde_json::json!({"error": "No splats loaded"}));
                std::process::exit(1);
            }

            let energy_api = m2m_vector_search::ebm::energy_api::EBMEnergy::new();
            let mut soc = m2m_vector_search::ebm::soc::SOCEngine::new(energy_api, 0.7);
            soc.update_splats(mu.unwrap(), alpha.unwrap(), kappa.unwrap());

            let result = soc.trigger_avalanche(seed);
            println!("{}", serde_json::json!({
                "affected_clusters": result.affected_clusters,
                "energy_released": (result.energy_released * 10000.0).round() / 10000.0,
                "duration_ms": (result.duration_ms * 1000.0).round() / 1000.0,
                "new_equilibrium": (result.new_equilibrium * 10000.0).round() / 10000.0,
            }));
        }

        Commands::SocRelax { iterations } => {
            let (store, _) = load_or_create_store(&cli.data_dir, &config);
            let mu = store.get_mu();
            let alpha = store.get_alpha();
            let kappa = store.get_kappa();

            if mu.is_none() || store.n_active() == 0 {
                println!("{}", serde_json::json!({"error": "No splats loaded"}));
                std::process::exit(1);
            }

            let energy_api = m2m_vector_search::ebm::energy_api::EBMEnergy::new();
            let mut soc = m2m_vector_search::ebm::soc::SOCEngine::new(energy_api, 0.7);
            soc.update_splats(mu.unwrap(), alpha.unwrap(), kappa.unwrap());

            let result = soc.relax(iterations);
            println!("{}", serde_json::json!({
                "initial_energy": (result.initial_energy * 10000.0).round() / 10000.0,
                "final_energy": (result.final_energy * 10000.0).round() / 10000.0,
                "energy_delta": (result.energy_delta * 10000.0).round() / 10000.0,
                "iterations": result.iterations,
                "improved": result.improved,
            }));
        }

        Commands::Save { shard } => {
            let (store, persist) = load_or_create_store(&cli.data_dir, &config);
            let persist = match persist {
                Some(p) => p,
                None => { eprintln!("Error: no persistence layer"); std::process::exit(1); }
            };

            let mu = store.get_mu();
            let alpha = store.get_alpha();
            let kappa = store.get_kappa();

            if let (Some(mu), Some(alpha), Some(kappa)) = (mu, alpha, kappa) {
                let n = store.n_active();
                let vectors = mu.slice(ndarray::s![..n, ..]).to_owned();
                persist.save_vectors(&vectors, &shard)
                    .unwrap_or_else(|e| { eprintln!("Save error: {}", e); std::process::exit(1); });
                persist.save_energy_state(&vectors, &alpha.to_vec(), &kappa.to_vec())
                    .unwrap_or_else(|e| { eprintln!("Energy save error: {}", e); std::process::exit(1); });
                println!("{}", serde_json::json!({
                    "status": "saved",
                    "shard": shard,
                    "n_vectors": n,
                }));
            } else {
                eprintln!("No data to save");
                std::process::exit(1);
            }
        }

        Commands::Load { shard } => {
            let persist = make_persistence(&cli.data_dir, &cli.backend, true)
                .unwrap_or_else(|e| { eprintln!("Storage error: {}", e); std::process::exit(1); });

            match persist.load_vectors(&shard) {
                Ok(Some(vectors)) => {
                    println!("{}", serde_json::json!({
                        "status": "loaded",
                        "shard": shard,
                        "n_vectors": vectors.nrows(),
                        "dim": vectors.ncols(),
                    }));
                }
                Ok(None) => {
                    println!("{}", serde_json::json!({"status": "not_found", "shard": shard}));
                }
                Err(e) => {
                    eprintln!("Load error: {}", e);
                    std::process::exit(1);
                }
            }
        }

        Commands::List => {
            let persist = make_persistence(&cli.data_dir, &cli.backend, false)
                .unwrap_or_else(|e| { eprintln!("Storage error: {}", e); std::process::exit(1); });
            let shards = persist.list_shards();
            println!("{}", serde_json::json!({"shards": shards}));
        }

        Commands::DocAdd { id, text, metadata } => {
            let persist = make_persistence(&cli.data_dir, &cli.backend, true)
                .unwrap_or_else(|e| { eprintln!("Storage error: {}", e); std::process::exit(1); });
            let meta_json = metadata.as_deref()
                .and_then(|s| serde_json::from_str(s).ok());
            persist.save_metadata(&id, 0, 0, meta_json.as_ref(), Some(&text))
                .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); });
            println!("{}", serde_json::json!({"status": "ok", "id": id}));
        }

        Commands::DocGet { id } => {
            let persist = make_persistence(&cli.data_dir, &cli.backend, false)
                .unwrap_or_else(|e| { eprintln!("Storage error: {}", e); std::process::exit(1); });
            match persist.get_metadata(&id) {
                Ok(Some(meta)) => println!("{}", serde_json::to_string_pretty(&meta).unwrap()),
                Ok(None) => println!("{}", serde_json::json!({"status": "not_found", "id": id})),
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            }
        }

        Commands::DocDel { id } => {
            let persist = make_persistence(&cli.data_dir, &cli.backend, true)
                .unwrap_or_else(|e| { eprintln!("Storage error: {}", e); std::process::exit(1); });
            match persist.soft_delete(&id) {
                Ok(true) => println!("{}", serde_json::json!({"status": "deleted", "id": id})),
                Ok(false) => println!("{}", serde_json::json!({"status": "not_found", "id": id})),
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            }
        }

        Commands::Backup { output } => {
            let persist = make_persistence(&cli.data_dir, &cli.backend, false)
                .unwrap_or_else(|e| { eprintln!("Storage error: {}", e); std::process::exit(1); });
            match persist.backup(output.to_str().unwrap_or("./backup")) {
                Ok(path) => println!("{}", serde_json::json!({"status": "backup_ok", "path": path})),
                Err(e) => { eprintln!("Backup error: {}", e); std::process::exit(1); }
            }
        }

        Commands::QuantIndex { input, bits, algorithm, seed } => {
            let vectors = match load_vectors_bin(&input) {
                Ok(v) => v,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            let dim = vectors.ncols();
            eprintln!("[m2m] Quantizing {} vectors ({}D, {}-bit)", vectors.nrows(), dim, bits);

            let algo = match algorithm.as_str() {
                "polar" => QuantAlgorithm::PolarQuant,
                _ => QuantAlgorithm::TurboQuant,
            };

            let projections = match algo {
                QuantAlgorithm::TurboQuant => (dim / 4).max(4),
                QuantAlgorithm::PolarQuant => 0,
            };

            let qcfg = QuantConfig { bits, projections, seed, algorithm: algo };
            let mut qstore = QuantizedStore::new(dim, qcfg).unwrap();
            let t0 = std::time::Instant::now();
            let indexed = qstore.add_batch(&vectors, 0);
            let elapsed = t0.elapsed();

            let ratio = qstore.compression_ratio();
            let mem = qstore.memory_usage();
            let raw_mem = dim * 4 * indexed;

            println!("{}", serde_json::json!({
                "status": "quantized",
                "algorithm": algorithm,
                "bits": bits,
                "n_vectors": indexed,
                "dim": dim,
                "compression_ratio": (ratio * 100.0).round() / 100.0,
                "memory_quantized_bytes": mem,
                "memory_raw_bytes": raw_mem,
                "encode_time_ms": elapsed.as_millis() as u64,
            }));
        }

        Commands::QuantSearch { query, top_k } => {
            let q_vec = match parse_query(&query) {
                Ok(v) => v,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };

            // Load vectors and build quantized store on-the-fly
            let persist = make_persistence(&cli.data_dir, &cli.backend, false).ok();
            let vectors = if let Some(ref p) = persist {
                p.load_vectors("default").ok().flatten()
            } else { None };

            let vectors = match vectors {
                Some(v) => v,
                None => {
                    eprintln!("Error: no vectors found. Run 'quant-index' first.");
                    std::process::exit(1);
                }
            };

            let dim = vectors.ncols();
            let qcfg = QuantConfig::for_search(dim);
            let mut qstore = QuantizedStore::new(dim, qcfg).unwrap();
            qstore.add_batch(&vectors, 0);

            let t0 = std::time::Instant::now();
            let results = qstore.search(&q_vec.view(), top_k);
            let elapsed = t0.elapsed();

            let entries: Vec<serde_json::Value> = results.iter().map(|(id, score)| {
                serde_json::json!({
                    "id": id,
                    "score": (score * 10000.0).round() / 10000.0,
                })
            }).collect();

            println!("{}", serde_json::json!({
                "query_time_ms": elapsed.as_millis() as u64,
                "n_results": entries.len(),
                "results": entries,
            }));
        }

        Commands::QuantStatus => {
            let qcfg = QuantConfig::for_search(cli.dim);
            let qstore = QuantizedStore::new(cli.dim, qcfg).unwrap();

            let cfg_4bit = QuantConfig::for_compression(cli.dim);
            let qs_4bit = QuantizedStore::new(cli.dim, cfg_4bit).unwrap();

            let cfg_3bit = QuantConfig::for_max_compression(cli.dim);
            let qs_3bit = QuantizedStore::new(cli.dim, cfg_3bit).unwrap();

            println!("{}", serde_json::json!( {
                "dim": cli.dim,
                "presets": {
                    "search_8bit": {
                        "algorithm": "TurboQuant",
                        "bits": 8,
                        "compression_ratio": (qstore.compression_ratio() * 100.0).round() / 100.0,
                        "bytes_per_vector": (qstore.compression_ratio().recip() * cli.dim as f32 * 4.0) as usize,
                    },
                    "balanced_4bit": {
                        "algorithm": "TurboQuant",
                        "bits": 4,
                        "compression_ratio": (qs_4bit.compression_ratio() * 100.0).round() / 100.0,
                        "bytes_per_vector": (qs_4bit.compression_ratio().recip() * cli.dim as f32 * 4.0) as usize,
                    },
                    "max_compression_3bit": {
                        "algorithm": "PolarQuant",
                        "bits": 3,
                        "compression_ratio": (qs_3bit.compression_ratio() * 100.0).round() / 100.0,
                        "bytes_per_vector": (qs_3bit.compression_ratio().recip() * cli.dim as f32 * 4.0) as usize,
                    },
                },
                "features": [
                    "Data-oblivious: no training or calibration needed",
                    "Deterministic: same (dim, bits, seed) always produces same codes",
                    "Instant indexing: no offline training phase",
                    "Unbiased inner product estimation",
                ],
            }));
        }

        Commands::GpuInfo => {
            let cuda_available = m2m_vector_search::gpu::is_cuda_available();
            let gpu_info = m2m_vector_search::gpu::gpu_info();
            let device = m2m_vector_search::config::detect_device();

            println!("{}", serde_json::json!({
                "device": device,
                "cuda_available": cuda_available,
                "gpu_info": gpu_info,
                "cuda_feature": {
                    "compiled": cfg!(feature = "cuda"),
                },
            }));
        }

        Commands::BenchGpu { n_vectors, dim, n_queries, top_k, metric } => {
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("[m2m] bench-gpu requires --features cuda");
                std::process::exit(1);
            }
            #[cfg(feature = "cuda")]
            {
            use std::time::Instant;
            use m2m_vector_search::gpu::cuda_kernel::GpuIndex;

            // Generate random dataset
            eprintln!("[m2m] Generating {} vectors ({}D)...", n_vectors, dim);
            let mut rng = rand::thread_rng();
            let dataset: Vec<f32> = (0..n_vectors * dim)
                .map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0))
                .collect();
            let queries: Vec<f32> = (0..n_queries * dim)
                .map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0))
                .collect();

            // CPU benchmark
            eprintln!("[m2m] CPU benchmark...");
            let t0 = Instant::now();
            let cpu_results = m2m_vector_search::gpu::batch_search(
                &queries, n_queries, &dataset, n_vectors, dim, top_k, &metric,
            );
            let cpu_ms = t0.elapsed().as_millis() as f64;

            // GPU benchmark with persistent dataset
            let gpu_result = if let Some(mut gpu_idx) = GpuIndex::new() {
                // Upload phase (measured separately)
                eprintln!("[m2m] Uploading {} vectors to GPU...", n_vectors);
                let t_upload = Instant::now();
                gpu_idx.upload_dataset(&dataset, n_vectors, dim);
                let upload_ms = t_upload.elapsed().as_millis() as f64;

                // Query phase (dataset already in VRAM)
                eprintln!("[m2m] GPU query benchmark (dataset in VRAM)...");
                let t1 = Instant::now();
                let gpu_results = gpu_idx.topk_search(&queries, n_queries, top_k, &metric);
                let query_ms = t1.elapsed().as_millis() as f64;

                let gpu_results = match gpu_results {
                    Some(r) => r,
                    None => {
                        eprintln!("[m2m] GPU topk_search returned None");
                        // Fallback to full pipeline
                        let t2 = Instant::now();
                        let r = m2m_vector_search::gpu::batch_search(
                            &queries, n_queries, &dataset, n_vectors, dim, top_k, &metric,
                        );
                        let _ = query_ms + t2.elapsed().as_millis() as f64;
                        r
                    }
                };

                // Verify GPU matches CPU
                let matches = cpu_results.len() == gpu_results.len()
                    && cpu_results.iter().zip(gpu_results.iter())
                        .all(|(c, g)| c.0.first() == g.0.first());

                eprintln!("[m2m] GPU/CPU results match: {}", matches);
                Some((upload_ms, query_ms))
            } else {
                None
            };

            let gpu_json = gpu_result.map(|(upload_ms, query_ms)| {
                let total_gpu = upload_ms + query_ms;
                serde_json::json!({
                    "upload_ms": upload_ms,
                    "query_ms": query_ms,
                    "total_ms": total_gpu,
                    "per_query_us": query_ms / n_queries as f64 * 1000.0,
                    "qps_persistent": n_queries as f64 / (query_ms / 1000.0),
                    "qps_with_upload": n_queries as f64 / (total_gpu / 1000.0),
                    "speedup_persistent": cpu_ms / query_ms,
                    "speedup_with_upload": cpu_ms / total_gpu,
                })
            });

            println!("{}", serde_json::json!({
                "config": {
                    "n_vectors": n_vectors,
                    "dim": dim,
                    "n_queries": n_queries,
                    "top_k": top_k,
                    "metric": metric,
                    "dataset_mb": (n_vectors * dim * 4) as f64 / 1_048_576.0,
                },
                "cpu": {
                    "total_ms": cpu_ms,
                    "per_query_us": cpu_ms / n_queries as f64 * 1000.0,
                    "qps": n_queries as f64 / (cpu_ms / 1000.0),
                },
                "gpu": gpu_json,
            }));
            } // cfg cuda
        }

        Commands::Ingest { input, n_clusters, seed } => {
            let vectors = match load_vectors_bin(&input) {
                Ok(v) => v,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            eprintln!("[m2m] Ingesting {} vectors ({}D) via DatasetTransformer", vectors.nrows(), vectors.ncols());

            let mut store = SplatStore::new(config);
            let k = n_clusters.unwrap_or_else(|| (vectors.nrows() as f64).sqrt() as usize);
            let t0 = std::time::Instant::now();

            match store.ingest_with_transformer(&vectors, k, seed) {
                Ok((n_splats, compression, stats)) => {
                    let elapsed = t0.elapsed();
                    store.build_index();
                    println!("{}", serde_json::json!({
                        "status": "ingested",
                        "method": "kmeans",
                        "original_count": stats.original_count,
                        "n_splats": n_splats,
                        "compression_ratio": (compression * 10000.0).round() / 10000.0,
                        "transform_time_s": (stats.transform_time_s * 10000.0).round() / 10000.0,
                        "total_ingest_ms": elapsed.as_millis() as u64,
                        "dim": vectors.ncols(),
                    }));
                }
                Err(e) => { eprintln!("Ingest error: {}", e); std::process::exit(1); }
            }
        }

        Commands::IngestHierarchical { input, n_clusters, min_cluster_size, seed } => {            let vectors = match load_vectors_bin(&input) {
                Ok(v) => v,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            eprintln!("[m2m] Hierarchical ingest {} vectors ({}D)", vectors.nrows(), vectors.ncols());

            let mut store = SplatStore::new(config);
            let t0 = std::time::Instant::now();

            match store.ingest_hierarchical(&vectors, n_clusters, min_cluster_size, seed) {
                Ok((n_splats, compression, stats)) => {
                    let elapsed = t0.elapsed();
                    store.build_index();
                    println!("{}", serde_json::json!({
                        "status": "ingested",
                        "method": "hierarchical_kmeans",
                        "original_count": stats.original_count,
                        "n_splats": n_splats,
                        "compression_ratio": (compression * 10000.0).round() / 10000.0,
                        "total_ingest_ms": elapsed.as_millis() as u64,
                        "dim": vectors.ncols(),
                    }));
                }
                Err(e) => { eprintln!("Ingest error: {}", e); std::process::exit(1); }
            }
        }

        Commands::IngestLeader { input, target_clusters, threshold, seed } => {
            let vectors = match load_vectors_bin(&input) {
                Ok(v) => v,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            eprintln!("[m2m] Leader clustering ingest {} vectors ({}D)", vectors.nrows(), vectors.ncols());

            let mut store = SplatStore::new(config);
            let t0 = std::time::Instant::now();

            match store.ingest_leader(&vectors, target_clusters, seed, threshold) {
                Ok((n_splats, compression, stats)) => {
                    let elapsed = t0.elapsed();
                    store.build_index();
                    // Persist the ingested data
                    let persist = make_persistence(&cli.data_dir, &cli.backend, false).ok();
                    if let Some(p) = persist {
                        if let (Some(mu), Some(alpha), Some(kappa)) = (store.get_mu(), store.get_alpha(), store.get_kappa()) {
                            let n = store.n_active();
                            let vectors = mu.slice(ndarray::s![..n, ..]).to_owned();
                            if let Err(e) = p.save_vectors(&vectors, &"default".to_string()) {
                                eprintln!("Warning: save failed: {}", e);
                            }
                            if let Err(e) = p.save_energy_state(&vectors, &alpha.to_vec(), &kappa.to_vec()) {
                                eprintln!("Warning: energy save failed: {}", e);
                            }
                        }
                    }
                    println!("{}", serde_json::json!({
                        "status": "ingested",
                        "method": "leader_clustering",
                        "original_count": stats.original_count,
                        "n_splats": n_splats,
                        "compression_ratio": (compression * 10000.0).round() / 10000.0,
                        "total_ingest_ms": elapsed.as_millis() as u64,
                        "dim": vectors.ncols(),
                        "threshold": threshold,
                        "saved": true,
                    }));
                }
                Err(e) => { eprintln!("Ingest error: {}", e); std::process::exit(1); }
            }
        }

        Commands::FusedSearch { query, query_file, k } => {
            let q = resolve_query(query.as_deref(), query_file.as_ref());
            eprintln!("[m2m] Query dim: {}", q.len());
            let (store, _) = load_or_create_store(&cli.data_dir, &config);
            eprintln!("[m2m] Store n_active: {}", store.n_active());
            let t0 = std::time::Instant::now();
            eprintln!("[m2m] Starting search...");
            let results = store.find_neighbors(&q.view(), k);
            let elapsed = t0.elapsed();
            eprintln!("[m2m] Search complete");
            let formatted = format_neighbors(&results, "json");
            eprintln!("[m2m] Fused search: {} results in {:?}", results.len(), elapsed);
            println!("{}", formatted);
        }

        Commands::PresetInfo { preset } => {
            let presets = vec![
                ("simple", M2MConfig::simple(None)),
                ("advanced", M2MConfig::advanced(None)),
                ("training", M2MConfig::training(None)),
                ("distributed", M2MConfig::distributed(None)),
                ("gpu", M2MConfig::gpu(None)),
            ];

            let filtered: Vec<_> = if let Some(ref name) = preset {
                let lower = name.to_lowercase();
                presets.into_iter().filter(|(n, _)| *n == lower).collect()
            } else {
                presets
            };

            let info: Vec<serde_json::Value> = filtered.iter().map(|(name, c)| {
                let mut store_config = c.clone();
                store_config.device = "cpu".to_string();
                store_config.enable_cuda = false;
                store_config.finalize();
                let store = SplatStore::new(store_config);
                serde_json::json!({
                    "preset": name,
                    "max_splats": c.max_splats,
                    "latent_dim": c.latent_dim,
                    "subsystems": {
                        "quantization": c.enable_quantization,
                        "hnsw": c.enable_hnsw,
                        "lsh": c.enable_lsh,
                        "semantic_memory": c.enable_semantic_memory,
                        "graph": c.enable_graph,
                        "cuda": c.enable_cuda,
                        "gpu_search": c.enable_gpu_search,
                    },
                    "runtime": {
                        "has_quantization": store.has_quantization(),
                        "has_hnsw": store.has_hnsw(),
                        "has_lsh": store.has_lsh(),
                        "has_semantic_memory": store.has_semantic_memory(),
                    },
                })
            }).collect();

            println!("{}", serde_json::to_string_pretty(&info).unwrap());
        }

        Commands::HnswSearch { query, query_file, k } => {
            let q = resolve_query(query.as_deref(), query_file.as_ref());
            // Load data, build with HNSW-enabled preset
            let mut cfg = M2MConfig::advanced(None);
            cfg.latent_dim = cli.dim;
            cfg.max_splats = cli.max_splats;
            cfg.device = "cpu".to_string();
            cfg.enable_cuda = false;
            cfg.enable_gpu_search = false;
            cfg.finalize();

            let (mut store, _) = load_or_create_store(&cli.data_dir, &cfg);
            eprintln!("[m2m] HNSW enabled in config: {}", cfg.enable_hnsw);
            eprintln!("[m2m] Store has HNSW: {}", store.has_hnsw());
            if !store.has_hnsw() {
                eprintln!("HNSW not enabled. Use 'advanced' or 'gpu' preset.");
                std::process::exit(1);
            }
            store.build_index();
            let t0 = std::time::Instant::now();
            let results = store.find_neighbors_fused(&q.view(), k);
            let elapsed = t0.elapsed();
            eprintln!("[m2m] HNSW search: {} results in {:?}", results.len(), elapsed);
            println!("{}", format_neighbors(&results, "json"));
        }

        Commands::LshSearch { query, query_file, k } => {
            let q = resolve_query(query.as_deref(), query_file.as_ref());
            // Use a config with LSH enabled
            let mut cfg = M2MConfig::default();
            cfg.latent_dim = cli.dim;
            cfg.max_splats = cli.max_splats;
            cfg.enable_lsh = true;
            cfg.device = "cpu".to_string();
            cfg.enable_cuda = false;
            cfg.finalize();

            let (mut store, _) = load_or_create_store(&cli.data_dir, &cfg);
            if !store.has_lsh() {
                eprintln!("LSH not enabled.");
                std::process::exit(1);
            }
            store.build_index();
            let t0 = std::time::Instant::now();
            let results = store.find_neighbors_fused(&q.view(), k);
            let elapsed = t0.elapsed();
            eprintln!("[m2m] LSH search: {} results in {:?}", results.len(), elapsed);
            println!("{}", format_neighbors(&results, "json"));
        }

        Commands::BenchGpuIngest { n_vectors, dim, n_clusters, n_queries } => {
            use std::time::Instant;

            eprintln!("[m2m] Generating {} vectors ({}D)...", n_vectors, dim);
            let mut rng = rand::thread_rng();
            let raw: Vec<f32> = (0..n_vectors * dim)
                .map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0))
                .collect();
            let dataset = ndarray::Array2::from_shape_vec((n_vectors, dim), raw).unwrap();
            let queries: Vec<f32> = (0..n_queries * dim)
                .map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0))
                .collect();

            // Phase 1: Ingest via DatasetTransformer
            let mut ingest_cfg = M2MConfig::gpu(None);
            ingest_cfg.latent_dim = dim;
            ingest_cfg.max_splats = n_clusters * 2;
            ingest_cfg.device = "cpu".to_string();
            ingest_cfg.enable_cuda = false;
            ingest_cfg.enable_gpu_search = false;
            ingest_cfg.finalize();

            let mut store = SplatStore::new(ingest_cfg);
            eprintln!("[m2m] Phase 1: DatasetTransformer ingest...");
            let t_ingest = Instant::now();
            let (n_splats, compression, stats) = store.ingest_with_transformer(&dataset, n_clusters, 42)
                .unwrap_or_else(|e| { eprintln!("Ingest error: {}", e); std::process::exit(1); });
            let ingest_ms = t_ingest.elapsed().as_millis() as f64;

            // Phase 2: Build index
            eprintln!("[m2m] Phase 2: Building index...");
            let t_build = Instant::now();
            store.build_index();
            let build_ms = t_build.elapsed().as_millis() as f64;

            // Phase 3: Linear search on compressed splats
            eprintln!("[m2m] Phase 3: Search {} queries against {} splats...", n_queries, n_splats);
            let query_arr = ndarray::Array2::from_shape_vec((n_queries, dim), queries.clone()).unwrap();
            let t_search = Instant::now();
            for i in 0..n_queries {
                let _ = store.find_neighbors(&query_arr.row(i), 10);
            }
            let search_ms = t_search.elapsed().as_millis() as f64;

            // Phase 4: GPU search on raw data (comparison)
            #[cfg(feature = "cuda")]
            let gpu_search: Option<serde_json::Value> = {
                if let Some(mut gpu_idx) = m2m_vector_search::gpu::cuda_kernel::GpuIndex::new() {
                    let raw_flat: Vec<f32> = dataset.clone().into_raw_vec();
                    eprintln!("[m2m] Phase 4: GPU search on raw {} vectors...", n_vectors);
                    let t_gpu_upload = Instant::now();
                    gpu_idx.upload_dataset(&raw_flat, n_vectors, dim);
                    let gpu_upload_ms = t_gpu_upload.elapsed().as_millis() as f64;

                    let t_gpu_q = Instant::now();
                    let _ = gpu_idx.topk_search(&queries, n_queries, 10, "l2");
                    let gpu_query_ms = t_gpu_q.elapsed().as_millis() as f64;
                    Some(serde_json::json!({
                        "upload_ms": gpu_upload_ms,
                        "query_ms": gpu_query_ms,
                        "total_ms": gpu_upload_ms + gpu_query_ms,
                        "qps": n_queries as f64 / (gpu_query_ms / 1000.0),
                    }))
                } else { None }
            };
            #[cfg(not(feature = "cuda"))]
            let gpu_search: Option<serde_json::Value> = None;

            // Phase 5: Fused search (all backends)
            eprintln!("[m2m] Phase 5: Fused search...");
            let n_fused = n_queries.min(10);
            let t_fused = Instant::now();
            for i in 0..n_fused {
                let _ = store.find_neighbors_fused(&query_arr.row(i), 10);
            }
            let fused_ms = t_fused.elapsed().as_millis() as f64;

            let total_pipeline = ingest_ms + build_ms + search_ms;

            println!("{}", serde_json::json!({
                "pipeline": "gpu_transformer",
                "config": {
                    "n_vectors": n_vectors,
                    "dim": dim,
                    "n_clusters": n_clusters,
                    "n_queries": n_queries,
                    "dataset_mb": (n_vectors * dim * 4) as f64 / 1_048_576.0,
                },
                "transformer": {
                    "original_count": stats.original_count,
                    "n_splats": n_splats,
                    "compression_ratio": (compression * 10000.0).round() / 10000.0,
                    "transform_time_s": stats.transform_time_s,
                    "total_ingest_ms": ingest_ms as u64,
                },
                "index_build_ms": build_ms as u64,
                "linear_search": {
                    "total_ms": search_ms,
                    "per_query_us": search_ms / n_queries as f64 * 1000.0,
                    "qps": n_queries as f64 / (search_ms / 1000.0),
                },
                "fused_search": {
                    "n_queries": n_fused,
                    "total_ms": fused_ms,
                    "per_query_us": fused_ms / n_fused as f64 * 1000.0,
                },
                "gpu_raw_search": gpu_search,
                "total_pipeline_ms": total_pipeline as u64,
            }));
        }

        Commands::Serve { port } => {
            println!("[m2m] Starting API server on port {}...", port);
            let rt = tokio::runtime::Runtime::new()
                .expect("Failed to create tokio runtime");
            rt.block_on(async {
                if let Err(e) = m2m_vector_search::api_server::run_server(port).await {
                    eprintln!("[m2m] Server error: {}", e);
                    std::process::exit(1);
                }
            });
        }
    }
}
