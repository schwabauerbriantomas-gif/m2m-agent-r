# M2M Rust Benchmarks (2026-03-28)

**Hardware**: AMD Ryzen 5 3400G (4c/8t) | 32GB DDR4 | RTX 3090 (not used, CPU-only)
**Profile**: `cargo bench` (optimized release)

## Results

### KMeans (fit, 10 clusters, 50 iterations, dim=64)
| N      | Time       | Throughput   |
|--------|-----------|--------------|
| 100    | 718 us     | 139 Kelem/s  |
| 1,000  | 24.3 ms    | 41 Kelem/s   |
| 10,000 | 521 ms     | 19 Kelem/s   |

### Encoding (FullEmbeddingBuilder, 640-d output)
| Benchmark              | Time     |
|------------------------|----------|
| Single embedding       | 3.98 us  |
| Batch 1k embeddings    | 5.51 ms  |

### Energy Functions (640-d)
| Benchmark                    | Time     |
|------------------------------|----------|
| e_geom 100 vectors           | 46.2 us  |
| e_splats_vectorized 100x1k   | 95.8 ms  |

### Geometry
| Benchmark               | Time    |
|-------------------------|---------|
| normalize_sphere 10k    | 14.7 ms |

### Clustering Transform
| Benchmark                    | Time     |
|------------------------------|----------|
| kmeans_transform 100 queries | 107 us   |

## Notes
- All benchmarks on CPU (no SIMD explicit, relying on compiler auto-vectorization)
- ndarray 0.16.1 with standard allocator
- f32 precision throughout
- Criterion 0.5 with plotters backend (gnuplot not available)
