# M2M Vector Search

[![Version](https://img.shields.io/badge/version-2.1.0-blue.svg)](https://github.com/user/m2m-rust)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Rust Edition 2021](https://img.shields.io/badge/rust-2021-orange.svg)](https://blog.rust-lang.org/2021/10/21/Rust-1.56.0.html)

**Native Rust Gaussian Splatting vector search engine** — representing embeddings as probability distributions (Gaussian splats) instead of point vectors, enabling richer similarity semantics through two-level hierarchical retrieval (HRM2), compressed quantization, and knowledge graph overlays.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SplatStore (API)                       │
│         insert · search · upsert · delete · stats            │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   HRM2 Engine (Retrieval)                    │
│   ┌─────────────┐    ┌─────────────┐    ┌───────────────┐  │
│   │ Coarse      │───▶│ Fine        │───▶│ Exact Re-rank │  │
│   │ KMeans++    │    │ KMeans++    │    │ L2 Distance   │  │
│   │ (n_probe)   │    │ (per cluster)│    │ (top-k sort)  │  │
│   └─────────────┘    └─────────────┘    └───────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
┌──────▼──────┐ ┌──────▼──────┐ ┌─────▼──────────┐
│ Quantization│ │ GraphSplat  │ │ SemanticMemory  │
│ TurboQuant  │ │ Knowledge   │ │ BM25 + Vector   │
│ PolarQuant  │ │ Graph       │ │ Hybrid Search   │
│ 3-8 bit     │ │ Overlay     │ │ RRF Fusion      │
└─────────────┘ └─────────────┘ └─────────────────┘
                       │
              ┌────────▼────────┐
              │   Persistence   │
              │   SQLite + JSON │
              └─────────────────┘
```

## Features

- **Gaussian Splat Embeddings** — each vector is a probability distribution (mean + alpha + kappa), not a point. Captures uncertainty and density.
- **HRM2 Two-Level Retrieval** — coarse KMeans++ partitions the space, fine KMeans++ refines within each partition. Configurable probe count controls recall/speed tradeoff.
- **TurboQuant / PolarQuant Compression** — 3-8 bit data-oblivious quantization. No training or codebooks required. 4-10x memory reduction for approximate search (theoretical; not yet benchmarked in Rust).
- **GraphSplat Knowledge Graph** — overlay graph structure on splats with typed nodes (Document/Entity/Concept), directed weighted edges, and graph-augmented retrieval (BFS traversal, relationship queries).
- **Semantic Memory** — hybrid BM25 + vector search with Reciprocal Rank Fusion (RRF). Auto-categorization of memories. Designed for AI agent memory systems.
- **KMeans++ Clustering** — deterministic initialization with ChaCha8 RNG, Lloyd's algorithm, Rayon-parallelized distance computation.
- **CPU-Optimized Engine** — parallel L2 distance with `rayon`, partial sort for top-k (O(n) average), ndarray einsum for batch operations.
- **SQLite Persistence** — full database storage for semantic memory with metadata, categories, and timestamps.
- **Configurable Profiles** — `M2MConfig::simple()` for edge devices, `M2MConfig::advanced()` for agentic workloads. Auto device detection (CUDA > Vulkan > CPU).

## Energy-Based Model (EBM)

The EBM layer treats the vector space as an **energy landscape** where each Gaussian Splat acts as an attractor basin. This enables three capabilities that point-vector systems cannot provide:

### Energy Landscape (`energy.rs`)
Each point in embedding space has an energy value:
```
E(x) = -log(sum_i alpha_i * exp(-kappa_i * ||x - mu_i||^2))
```
- **Low energy** = high confidence (near splat centers, dense regions)
- **High energy** = uncertainty (sparse regions, knowledge gaps)

### Exploration (`ebm/exploration.rs`)
- **Boltzmann sampling**: Sample points from high-energy regions to discover where the model is uncertain
- **Knowledge gap detection**: Identify clusters of unexplored energy space
- **Exploration suggestions**: Generate queries and regions worth investigating
- **Greedy clustering**: Group high-energy points into actionable regions

Use case: An AI agent can ask "what don't I know?" and the EBM returns regions of embedding space with no coverage — guiding active learning.

### Self-Organized Criticality (`ebm/soc.rs`)
- **Criticality monitoring**: Track whether the system is subcritical, critical, or supercritical
- **Avalanche detection**: When energy builds up in regions, trigger cascading reorganization
- **Natural reorganization**: Splats redistribute themselves through relaxation dynamics

Use case: Memory systems that self-reorganize when overloaded — preventing stagnation and maintaining retrieval quality over time.

## Quick Start

### Prerequisites

- Rust 1.56+ (Edition 2021)
- C compiler (for SQLite bundling via `rusqlite`)
- Local `turbo-quant` dependency at `C:/Users/user/projects/turbo-quant`

### Build & Test

```bash
# Clone
git clone <repo-url> m2m-rust && cd m2m-rust

# Build
cargo build --release

# Run tests
cargo test

# Run with stdout
cargo run
```

### Usage

```rust
use m2m_vector_search::{M2MConfig, SplatStore};

fn main() {
    // Use defaults (640D, 100K max splats, CPU)
    let config = M2MConfig::default();

    // Or edge profile: no 3-tier memory, no Langevin dynamics
    // let config = M2MConfig::simple(None);

    let mut store = SplatStore::new(config);

    // Insert vectors
    let embedding = vec![0.1f32; 640];
    let id = store.insert(&embedding);

    // Search for nearest neighbors
    let results = store.search(&embedding, 10);
    for r in &results {
        println!("id={} dist={:.4}", r.index, r.distance);
    }
}
```

## Module Reference

| Module | File | Description |
|--------|------|-------------|
| `config` | `src/config.rs` | `M2MConfig` with presets (simple/advanced/default), device detection |
| `splats` | `src/splats.rs` | `SplatStore` — main public API for insert/search/upsert/delete |
| `hrm2_engine` | `src/hrm2_engine.rs` | `HRM2Engine` — two-level hierarchical retrieval (coarse→fine→exact) |
| `engine` | `src/engine.rs` | `M2MEngine` — CPU-optimized L2 distance computation with Rayon |
| `quantization` | `src/quantization.rs` | TurboQuant/PolarQuant compression (3-8 bit) |
| `quant` | `src/quant/` | Integrated quantization algorithms: TurboQuant, PolarQuant, QJL, rotation (MIT, originally from turbo-quant) |
| `clustering` | `src/clustering.rs` | `KMeans` with KMeans++ init, parallel Lloyd's algorithm |
| `graph_splat` | `src/graph_splat.rs` | `GaussianGraphStore` — knowledge graph overlay on splats |
| `semantic_memory` | `src/semantic_memory.rs` | `SemanticMemoryDB` — hybrid BM25+vector search with RRF fusion |
| `splat_types` | `src/splat_types.rs` | Core types: `GaussianSplat`, `SplatEmbedding`, `SplatCluster` |
| `geometry` | `src/geometry.rs` | Sphere normalization and geodesic distance |
| `encoding` | `src/encoding.rs` | Full embedding builder (mu + alpha + kappa → flat vector) |
| `bm25_index` | `src/bm25_index.rs` | BM25 text search index |
| `storage` | `src/storage.rs` | SQLite persistence layer |
| `energy` | `src/energy.rs` | Energy landscape: `E(x) = -log(sum alpha_i * exp(-kappa_i * ||x - mu_i||^2))`. Lower energy = higher confidence near splat centers. Used by EBM for exploration and self-organization |
| `hnsw_index` | `src/hnsw_index.rs` | HNSW approximate nearest neighbor index |
| `lsh_index` | `src/lsh_index.rs` | Locality-Sensitive Hashing index |

## Performance

No Rust-specific benchmarks have been run yet. The following is from the Python prototype for reference only — **do not assume identical numbers in Rust**:

| Method | Latency | QPS | Speedup |
|--------|---------|-----|---------|
| Linear scan (Python) | 94.79 ms | 10.55 | 1x |
| M2M HRM2 (Python) | 0.99 ms | 1,012.77 | **32.4x** |

The Rust implementation should benefit from:
- Zero-cost abstractions and no GC pauses
- Rayon-parallelized distance computation
- Partial sort top-k (O(n) average vs O(n log n))
- Zero-copy ndarray operations
- Quantized search path for approximate retrieval at 4-10x memory reduction

**To run your own benchmarks:** `cargo bench` (requires `criterion` dev-dependency)

## Configuration

```rust
use m2m_vector_search::config::{M2MConfig, QuantAlgorithm, SearchBackend, FusionMethod};

// Six presets:
let general    = M2MConfig::default();                       // balanced
let edge       = M2MConfig::simple(None);                    // minimal
let agentic    = M2MConfig::advanced(Some("cuda"));          // full power
let research   = M2MConfig::training(None);                  // embedding research
let cluster    = M2MConfig::distributed(None);               // multi-node
let gpu_accel  = M2MConfig::gpu(None);                       // CUDA-accelerated
```

### `M2MConfig::default()` — General Purpose

Balanced config for servers and workstations. TurboQuant compression, GraphSplat knowledge overlay, and hybrid BM25+vector semantic search are all **active by default**.

| Subsystem | Status | Key Parameters |
|-----------|--------|---------------|
| **Search** | HRM2 (coarse→fine→exact) | `knn_k=64`, `hrm2_n_coarse=16`, `hrm2_n_fine=64`, `hrm2_n_probe=4` |
| **Quantization** | ✅ TurboQuant 8-bit | `quant_bits=8`, `quant_projections=160`, `quant_search_fraction=0.1` |
| **Graph Overlay** | ✅ GraphSplat | `graph_max_neighbors=10`, `graph_traverse_depth=3`, `graph_boost_weight=0.05` |
| **Semantic Memory** | ✅ Hybrid BM25+Vector | `semantic_fusion=RRF`, `bm25_k1=1.5`, `bm25_b=0.75`, `decay_halflife=24h` |
| **Indexing** | MapReduce | `mapreduce_n_chunks=8` |
| **Quality** | Reflection ON | `quality_recall_target=0.95` |
| **Energy/EBM** | ✅ Splat + Geometric | `energy_splat_weight=1.0`, `energy_geom_weight=0.1`, `soc_threshold=0.8` |
| **Langevin** | ✅ 200 steps | `langevin_dt=0.001`, `langevin_gamma=0.1` |
| **Memory** | 3-tier (hot/warm/cold) | `max_splats=100,000`, `context_windows=12/64/512` |
| **HNSW / LSH** | ❌ Off | Available but not needed at this scale |
| **Auto-Scaling** | ❌ Off | Single-node deployment |

Use when: general search applications, research, prototyping, production single-node.

---

### `M2MConfig::simple(device)` — Edge / Embedded

Stripped-down for resource-constrained environments. Pure HRM2 search, no compression overhead, no graph, RAM-only.

| Subsystem | Status | Change from Default |
|-----------|--------|-------------------|
| **Search** | HRM2 only | `knn_k=32`, `hrm2_n_coarse=8`, `hrm2_n_fine=32`, `hrm2_n_probe=2` |
| **Quantization** | ❌ Disabled | No compression overhead |
| **Graph Overlay** | ❌ Disabled | No knowledge graph |
| **Semantic Memory** | ❌ Disabled | No BM25, no fusion |
| **MapReduce** | ❌ Disabled | Single-chunk indexing |
| **Quality** | ❌ Disabled | No recall tracking |
| **Langevin** | ❌ 0 steps | No dynamics refinement |
| **Memory** | RAM-only | `max_splats=10,000`, `n_splats_init=1,000` |
| **Energy/EBM** | ❌ Off | No energy computation |
| **Context** | Minimal | `8/32/128` vs `12/64/512` |
| **MoE** | Reduced | `2 experts / 1 active` vs `4/2` |

Use when: Raspberry Pi, IoT, embedded systems, <4GB RAM, no disk I/O budget. Peak throughput with minimal footprint.

---

### `M2MConfig::advanced(device)` — Agentic AI Workloads

Everything enabled and tuned for autonomous agents with self-organizing memory, aggressive compression, and distributed scale.

| Subsystem | Status | Change from Default |
|-----------|--------|-------------------|
| **Search** | HRM2 + HNSW secondary | `knn_k=128`, `hrm2_n_coarse=32`, `hrm2_n_fine=128`, `hrm2_n_probe=8` |
| **Quantization** | ✅ TurboQuant **4-bit** | `quant_bits=4`, `quant_projections=80`, `search_fraction=0.2` |
| **Graph Overlay** | ✅ GraphSplat deep | `max_neighbors=20`, `traverse_depth=5`, `boost_weight=0.1` |
| **Semantic Memory** | ✅ RRF fusion | `vector_weight=0.6`, `bm25_weight=0.4`, `decay_halflife=1h` |
| **HNSW** | ✅ Secondary index | `M=32`, `ef_construction=400`, `ef_search=100` |
| **MapReduce** | ✅ 16 chunks | Parallel bulk indexing |
| **Auto-Scaling** | ✅ 2–20 nodes | `cooldown=30s` |
| **Quality** | ✅ 99% recall target | Near-exact retrieval guarantee |
| **Energy/EBM** | ✅ Full | Knowledge gap detection, self-organized criticality |
| **Langevin** | ✅ 200 steps | Continuous splat refinement |
| **Memory** | 3-tier, 1M splats | `n_splats_init=50,000`, `context=24/128/1024` |

Use when: persistent AI agents, autonomous memory systems, long-running chatbots, distributed deployments. Requires ≥8GB RAM. GPU recommended for 1M+ splats.

---

### `M2MConfig::training(device)` — Embedding Research

Optimized for training and evaluating embedding models. Disables compression (no quantization during training), enables data lake for training data management, noise augmentation, and Matryoshka representation learning.

| Subsystem | Status | Key Parameters |
|-----------|--------|---------------|
| **Training** | ✅ 50 epochs | `batch_size=128`, `seq_length=64`, `lr=3e-4` |
| **Noise Augmentation** | ✅ 7 levels | `[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]` |
| **Matryoshka** | ✅ 5 dimensions | `[32, 64, 128, 256, 640]` |
| **Distillation** | ✅ ON | Knowledge transfer from teacher model |
| **Data Lake** | ✅ 10M entries | Compressed training data storage |
| **Entity Extraction** | ✅ 2-5 grams | `min_confidence=0.3` |
| **Quantization** | ❌ OFF | No compression during training |
| **Graph/Semantic** | ❌ OFF | Not needed for evaluation |
| **Search** | HRM2 moderate | `knn_k=32`, `max_splats=500,000` |

Use when: developing new embedding models, fine-tuning Matryoshka representations, running noise robustness experiments, distillation from larger models.

---

### `M2MConfig::distributed(device)` — Multi-Node Cluster

For production deployments across multiple nodes. Enables cluster coordination, sharding, load balancing, auto-scaling, and compressed inter-node communication.

| Subsystem | Status | Key Parameters |
|-----------|--------|---------------|
| **Auto-Scaling** | ✅ 3–50 nodes | `cooldown=20s` |
| **MapReduce** | ✅ 32 chunks | Distributed bulk indexing |
| **Quantization** | ✅ TurboQuant 4-bit | Efficient inter-node transfer |
| **Quality** | ✅ 97% recall | Cross-node quality monitoring |
| **Semantic Memory** | ✅ RRF | `decay_halflife=2h` |
| **Search** | HRM2 | `knn_k=64` |
| **Capacity** | 10M splats | `n_splats_init=500,000` |
| **Graph/HNSW** | ❌ OFF | Available via advanced() override |

Use when: distributed search clusters, horizontal scaling, production multi-server deployments, sharded indices.

---

### `M2MConfig::gpu(device)` — CUDA-Accelerated Search

Maximum throughput for GPU-equipped machines. Enables GPU batch operations, auto-tuning, aggressive HNSW parameters, and quantized approximate search.

| Subsystem | Status | Key Parameters |
|-----------|--------|---------------|
| **GPU Search** | ✅ CUDA | `gpu_batch_size=4096`, `gpu_auto_tune=true` |
| **HNSW** | ✅ Aggressive | `M=48`, `ef_construction=800`, `ef_search=200` |
| **Quantization** | ✅ TurboQuant 4-bit | `projections=160`, `search_fraction=0.05` |
| **Graph** | ✅ Deep | `max_neighbors=32`, `traverse_depth=5` |
| **Search** | HRM2 high | `knn_k=256`, `hrm2_n_coarse=64`, `hrm2_n_fine=256` |
| **Capacity** | 5M splats | `n_splats_init=100,000` |

Use when: RTX 3090 or better available, batch search workloads, real-time serving with high throughput requirements. Requires NVIDIA GPU with CUDA support.

---

### All Configurable Fields

<details>
<summary>Full field reference (click to expand)</summary>

```toml
[System]
device = "auto"                 # auto | cpu | vulkan | cuda
dtype = "Float32"               # Float32 | Float64 | Int32 | Int64

[Latent Space]
latent_dim = 640
n_splats_init = 10000
max_splats = 100000
knn_k = 64

[Splat Parameters]
init_alpha = 1.0                # Initial opacity
init_kappa = 10.0               # Initial concentration
min_kappa = 1.0
max_kappa = 50.0

[Temperature & Energy]
splat_temperature = 0.1         # Exploration noise
weight_decay_start = 1.0
energy_splat_weight = 1.0       # Splat density energy
energy_geom_weight = 0.1        # Geometric regularization
energy_comp_weight = 0.0        # Composite (placeholder)
global_temperature = 1.0

[SOC - Self-Organized Criticality]
soc_threshold = 0.8
soc_buffer_capacity = 1000
soc_update_interval = 100
phi_convergence_threshold = 0.95

[HRM2 Search]
search_backend = "Hrm2"         # Hrm2 | Hnsw | Lsh | Quantized
hrm2_n_coarse = 16
hrm2_n_fine = 64
hrm2_n_probe = 4

[HNSW]
enable_hnsw = false
hnsw_m = 16
hnsw_ef_construction = 200
hnsw_ef_search = 50

[LSH]
enable_lsh = false
lsh_n_tables = 12
lsh_n_projections = 8

[Quantization]
enable_quantization = true
quant_algorithm = "TurboQuant"   # TurboQuant | PolarQuant | None
quant_bits = 8                   # 4 or 8 bits per dimension
quant_projections = 160
quant_seed = 42
quant_search_fraction = 0.1      # Fraction of candidates to re-rank

[GraphSplat]
enable_graph = true
graph_max_neighbors = 10
graph_traverse_depth = 3
graph_boost_weight = 0.05        # Score boost from graph neighbors

[Semantic Memory]
enable_semantic_memory = true
semantic_fusion = "Rrf"          # Rrf | Weighted | VectorOnly | Bm25Only
semantic_vector_weight = 0.7
semantic_bm25_weight = 0.3
semantic_bm25_k1 = 1.5
semantic_bm25_b = 0.75
semantic_decay_halflife = 86400.0  # seconds (24h)

[Langevin Dynamics]
langevin_steps = 200
langevin_dt = 0.001
langevin_gamma = 0.1
langevin_t = 1.0

[Context Windows]
context_local = 12
context_medium = 64
context_global = 512

[MoE Decoder]
vocab_size = 50257
hidden_dim = 1024
moe_experts = 4
moe_active = 2

[Training]
batch_size = 32
seq_length = 32
noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]

[Optimization]
learning_rate = 1e-4
weight_decay = 0.01
grad_clip = 1.0

[MapReduce Indexing]
enable_mapreduce = true
mapreduce_n_chunks = 8

[Auto-Scaling]
enable_auto_scaling = false
autoscale_min_nodes = 1
autoscale_max_nodes = 10
autoscale_cooldown_secs = 60.0

[Quality Reflection]
enable_quality_reflection = true
quality_recall_target = 0.95

[Training Mode]
enable_training = false
training_epochs = 10
training_eval_interval = 100
training_save_interval = 1000
training_noise_augmentation = true
training_matryoshka_dims = [64, 128, 256, 640]
training_distillation = false

[Data Lake]
enable_data_lake = false
data_lake_max_entries = 1_000_000
data_lake_compress = true

[Entity Extraction]
enable_entity_extraction = true
entity_ngram_sizes = [2, 3, 4]
entity_min_confidence = 0.5

[GPU Acceleration]
enable_gpu_search = false
gpu_batch_size = 1024
gpu_auto_tune = true

[API]
rest_port = 8080
grpc_port = 9090
```

</details>

## License

[AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html)
