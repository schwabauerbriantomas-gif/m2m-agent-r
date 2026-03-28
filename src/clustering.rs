//! KMeans clustering — native Rust implementation (no sklearn).
//! Optimized with KMeans++ init and Lloyd's algorithm.

use ndarray::{Array1, Array2, ArrayView2};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

/// KMeans clustering result.
#[derive(Clone, Debug)]
pub struct KMeans {
    pub n_clusters: usize,
    pub centroids: Option<Array2<f32>>,
    pub labels: Option<Array1<usize>>,
    pub inertia: f32,
    pub n_iter: usize,
    max_iter: usize,
    seed: u64,
}

impl KMeans {
    pub fn new(n_clusters: usize, max_iter: usize, seed: u64) -> Self {
        Self {
            n_clusters,
            centroids: None,
            labels: None,
            inertia: 0.0,
            n_iter: 0,
            max_iter,
            seed,
        }
    }

    /// KMeans++ initialization: pick first centroid randomly, then pick
    /// subsequent centroids proportional to squared distance.
    fn kmeans_plus_plus_init(&self, data: &Array2<f32>, mut rng: ChaCha8Rng) -> Array2<f32> {
        let (n, dim) = (data.nrows(), data.ncols());
        let k = self.n_clusters.min(n);
        let mut centroids = Array2::zeros((k, dim));

        // First centroid: random
        let first = rng.gen_range(0..n);
        centroids.row_mut(0).assign(&data.row(first));

        let mut min_dists: Vec<f32> = vec![f32::MAX; n];

        for c in 1..k {
            // Update min distances to nearest existing centroid
            let cent_row = centroids.row(c - 1);
            for (i, min_d) in min_dists.iter_mut().enumerate().take(n) {
                let row = data.row(i);
                let dist = (&row - &cent_row).mapv(|v| v * v).sum();
                if dist < *min_d {
                    *min_d = dist;
                }
            }

            // Weighted sampling proportional to min_dists
            let total: f32 = min_dists.iter().sum();
            if total < 1e-10 {
                // All points are the same, pick randomly
                let idx = rng.gen_range(0..n);
                centroids.row_mut(c).assign(&data.row(idx));
                continue;
            }

            let mut threshold = rng.gen::<f32>() * total;
            let mut chosen = n - 1;
            for (i, &d) in min_dists.iter().enumerate().take(n) {
                threshold -= d;
                if threshold <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            centroids.row_mut(c).assign(&data.row(chosen));
        }

        centroids
    }

    /// Fit KMeans to data using Lloyd's algorithm.
    pub fn fit(&mut self, data: &Array2<f32>) -> &Self {
        let (n, dim) = (data.nrows(), data.ncols());
        let k = self.n_clusters.min(n);

        if n == 0 || k == 0 {
            return self;
        }

        let rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut centroids = self.kmeans_plus_plus_init(data, rng);

        let mut labels = Array1::zeros(n);
        let mut counts = vec![0usize; k];
        let mut new_centroids = Array2::zeros((k, dim));

        for iteration in 0..self.max_iter {
            // Assignment step: assign each point to nearest centroid
            let mut changed = 0usize;
            new_centroids.fill(0.0);
            counts.iter_mut().for_each(|c| *c = 0);
            let mut total_inertia = 0.0f32;

            for i in 0..n {
                let row = data.row(i);
                let (best_k, best_dist) = (0..k)
                    .map(|c| {
                        let diff = &row - &centroids.row(c);
                        let d = diff.dot(&diff);
                        (c, d)
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();

                if labels[i] != best_k {
                    changed += 1;
                }
                labels[i] = best_k;
                total_inertia += best_dist;
                counts[best_k] += 1;
                let mut cent_row = new_centroids.row_mut(best_k);
                cent_row.scaled_add(1.0, &row);
            }

            self.inertia = total_inertia;

            // Update step: recompute centroids
            for c in 0..k {
                if counts[c] > 0 {
                    let mut row = centroids.row_mut(c);
                    row.assign(&new_centroids.row(c));
                    row.mapv_inplace(|v| v / counts[c] as f32);
                }
                // else: keep old centroid
            }

            self.n_iter = iteration + 1;

            // Convergence check
            if changed == 0 {
                break;
            }
        }

        // Handle clusters with 0 members by re-assigning their centroids
        // to the point furthest from its centroid
        for c in 0..k {
            if counts[c] == 0 {
                // Find point with max distance to its assigned centroid
                let mut best_dist = 0.0f32;
                let mut best_idx = 0;
                for i in 0..n {
                    let assigned = labels[i];
                    let diff = &data.row(i) - &centroids.row(assigned);
                    let d = diff.dot(&diff);
                    if d > best_dist {
                        best_dist = d;
                        best_idx = i;
                    }
                }
                centroids.row_mut(c).assign(&data.row(best_idx));
                labels[best_idx] = c;
            }
        }

        self.centroids = Some(centroids);
        self.labels = Some(labels);
        self
    }

    /// Predict cluster assignments for new data.
    pub fn predict(&self, data: &ArrayView2<f32>) -> Array1<usize> {
        let centroids = self.centroids.as_ref().expect("Must call fit() first");
        let n = data.nrows();
        let mut labels = Array1::zeros(n);

        for i in 0..n {
            let row = data.row(i);
            let best = (0..centroids.nrows())
                .map(|c| {
                    let diff = &row - &centroids.row(c);
                    diff.dot(&diff)
                })
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            labels[i] = best;
        }

        labels
    }

    /// Fit and return labels.
    pub fn fit_predict(&mut self, data: &Array2<f32>) -> Array1<usize> {
        self.fit(data);
        self.labels.clone().unwrap()
    }

    /// Transform: compute distances from each point to each centroid.
    pub fn transform<S>(&self, data: &ndarray::ArrayBase<S, ndarray::Dim<[usize; 2]>>) -> Array2<f32>
    where
        S: ndarray::Data<Elem = f32>,
    {
        let centroids = self.centroids.as_ref().expect("Must call fit() first");
        let n = data.nrows();
        let k = centroids.nrows();
        let mut distances = Array2::zeros((n, k));

        for i in 0..n {
            let row = data.row(i);
            for c in 0..k {
                let diff = &row - &centroids.row(c);
                distances[[i, c]] = diff.dot(&diff).sqrt();
            }
        }

        distances
    }
}
