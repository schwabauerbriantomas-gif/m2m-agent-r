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

```toml
# Key parameters in M2MConfig
latent_dim = 640          # Embedding dimensionality
max_splats = 100_000      # Maximum number of splats
knn_k = 64                # Neighbors per query
n_splats_init = 10_000    # Initial splat allocation
splat_temperature = 0.1   # Exploration temperature
enable_3_tier_memory = true  # Three-tier memory hierarchy
device = "auto"           # auto | cpu | vulkan | cuda
```

### Presets

#### `M2MConfig::default()` — General Purpose

The balanced configuration. Suitable for servers and workstations with moderate resources.

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `latent_dim` | 640 | Standard embedding size |
| `max_splats` | 100,000 | Up to 100K vectors |
| `knn_k` | 64 | High recall neighbor count |
| `splat_temperature` | 0.1 | Mild exploration |
| `enable_3_tier_memory` | true | Hot/warm/cold memory tiers |
| `memory_tier` | "3-tier" | RAM + disk + compressed |
| `langevin_steps` | 200 | Full Langevin dynamics for splat refinement |
| `langevin_dt` | 0.001 | Fine-grained dynamics step size |
| `soc_threshold` | 0.8 | Self-organized criticality active |
| `context_local/medium/global` | 12/64/512 | Multi-scale context windows |
| `moe_experts` / `moe_active` | 4 / 2 | Mixture of experts routing |

Use when: general search applications, research, prototyping. Good balance of quality and speed.

#### `M2MConfig::simple(device)` — Edge / Embedded

Stripped-down configuration for resource-constrained environments. Removes all heavy features.

| Parameter | Value | Difference from default |
|-----------|-------|----------------------|
| `enable_3_tier_memory` | **false** | No tiered memory — single RAM tier |
| `memory_tier` | "ram-only" | Everything in RAM, no disk offloading |
| `splat_temperature` | **0.0** | No exploration — deterministic behavior |
| `device` | configurable | Pass "cpu", "vulkan", or None for auto |

What's disabled:
- No 3-tier memory hierarchy (no hot/warm/cold tiers)
- No Langevin dynamics for splat refinement
- No exploration sampling
- Reduced memory overhead

Use when: Raspberry Pi, IoT devices, embedded systems, or any environment with <4GB RAM and no disk I/O budget.

#### `M2MConfig::advanced(device)` — Agentic AI Workloads

Full-featured configuration with all intelligence features enabled. Designed for autonomous AI agents that need self-organizing memory and active learning.

| Parameter | Value | Difference from default |
|-----------|-------|----------------------|
| `enable_3_tier_memory` | **true** | Full hot/warm/cold memory |
| `memory_tier` | "3-tier" | RAM + disk + compressed storage |
| `splat_temperature` | **0.1** | Active exploration enabled |
| `device` | configurable | Pass "cuda" for GPU acceleration |

What's additionally active:
- **EBM Exploration**: Agent can ask "what don't I know?" and get directed to knowledge gaps
- **Self-Organized Criticality**: Memory automatically reorganizes when overloaded
- **Langevin Dynamics**: Splats continuously refine their positions via energy-based sampling
- **3-Tier Memory**: Frequently accessed splats stay hot (RAM), rare ones compress to disk
- **Multi-scale Context**: Local (12), medium (64), and global (512) context windows for hierarchical reasoning

Use when: persistent AI agents, autonomous memory systems, long-running chatbots, research on self-organizing vector stores. Requires ≥8GB RAM. GPU (CUDA) recommended for large-scale deployments.

## License

[AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html)
